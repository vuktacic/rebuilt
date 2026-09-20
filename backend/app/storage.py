from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .models import AnalysisEvent, AnalysisInfo, Frame, Guide, JobError, JobMode, JobResponse, JobStatus, ManualPair, ManualReview, PartAnnotation, PartTrack, TrackSummary


ACTIVE_STATUSES = {"queued", "extracting", "diffing", "drafting", "analyzing", "generating"}
ADMISSION_STATUSES = ACTIVE_STATUSES | {"annotating", "pairing"}


class JobRepository:
    """Durable, one-directory-per-job storage with atomic JSON writes."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._mark_interrupted_jobs()

    def _job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def create(self, job_id: str, filename: str, mode: JobMode = "automated") -> JobResponse:
        directory = self._job_dir(job_id)
        (directory / "frames").mkdir(parents=True, exist_ok=False)
        (directory / "input").mkdir()
        response = JobResponse(jobId=job_id, mode=mode, status="queued")
        self._write_json(directory / "metadata.json", {**response.model_dump(), "filename": filename})
        return response

    def input_path(self, job_id: str, filename: str) -> Path:
        suffix = Path(filename).suffix.lower() or ".upload"
        return self._job_dir(job_id) / "input" / f"video{suffix}"

    def frame_path(self, job_id: str, frame_id: str) -> Path:
        return self._job_dir(job_id) / "frames" / f"{frame_id}.jpg"

    def get(self, job_id: str) -> JobResponse | None:
        path = self._job_dir(job_id) / "metadata.json"
        if not path.is_file():
            return None
        with self._lock:
            raw = json.loads(path.read_text(encoding="utf-8"))
        return JobResponse.model_validate(raw)

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        frames: list[Frame] | None = None,
        tracks: list[TrackSummary] | None = None,
        annotations: list[PartAnnotation] | None = None,
        part_tracks: list[PartTrack] | None = None,
        tracking_progress: float | None = None,
        events: list[AnalysisEvent] | None = None,
        analysis: AnalysisInfo | None = None,
        guide: Guide | None = None,
        manual_review: ManualReview | None = None,
        analysis_run_id: str | None = None,
        error: JobError | None = None,
    ) -> JobResponse:
        with self._lock:
            current = self.get(job_id)
            if current is None:
                raise KeyError(job_id)
            updated = current.model_copy(
                update={
                    "status": status if status is not None else current.status,
                    "frames": frames if frames is not None else current.frames,
                    "tracks": tracks if tracks is not None else current.tracks,
                    "annotations": annotations if annotations is not None else current.annotations,
                    "partTracks": part_tracks if part_tracks is not None else current.partTracks,
                    "trackingProgress": tracking_progress if tracking_progress is not None else current.trackingProgress,
                    "events": events if events is not None else current.events,
                    "analysis": analysis if analysis is not None else current.analysis,
                    "guide": guide if guide is not None else current.guide,
                    "manualReview": manual_review if manual_review is not None else current.manualReview,
                    "analysisRunId": analysis_run_id if analysis_run_id is not None else current.analysisRunId,
                    "error": error,
                }
            )
            directory = self._job_dir(job_id)
            self._write_json(directory / "metadata.json", updated.model_dump())
            if guide is not None:
                self._write_json(directory / "guide.json", guide.model_dump())
            return updated

    def save_manual_pairs(self, job_id: str, *, revision: int, pairs: list[ManualPair]) -> JobResponse:
        with self._lock:
            current = self.get(job_id)
            if current is None:
                raise KeyError(job_id)
            if current.revision != revision:
                raise ValueError("PAIR_REVISION_CONFLICT")
            updated = current.model_copy(update={"manualPairs": pairs, "revision": current.revision + 1, "error": None})
            self._write_json(self._job_dir(job_id) / "metadata.json", updated.model_dump())
            return updated

    def claim_manual_run(self, job_id: str, analysis_run_id: str) -> JobResponse:
        with self._lock:
            current = self.get(job_id)
            if current is None:
                raise KeyError(job_id)
            if current.mode != "manual" or current.status != "pairing":
                raise ValueError("MANUAL_RUN_NOT_READY")
            if not current.manualPairs:
                raise ValueError("MANUAL_PAIRS_REQUIRED")
            updated = current.model_copy(update={"status": "diffing", "analysisRunId": analysis_run_id, "manualReview": ManualReview(status="diffing")})
            self._write_json(self._job_dir(job_id) / "metadata.json", updated.model_dump())
            return updated

    def save_guide(self, job_id: str, guide: Guide) -> JobResponse:
        return self.update(job_id, guide=guide, error=None)

    def existing_frame_ids(self, job_id: str) -> set[str]:
        current = self.get(job_id)
        return {frame.frameId for frame in current.frames} if current else set()

    def has_admitted_job(self) -> bool:
        """Return whether persisted interactive or active work occupies the single-job slot."""
        jobs_dir = self.root / "jobs"
        if not jobs_dir.is_dir():
            return False
        with self._lock:
            for metadata in jobs_dir.glob("*/metadata.json"):
                try:
                    raw = json.loads(metadata.read_text(encoding="utf-8"))
                    if JobResponse.model_validate(raw).status in ADMISSION_STATUSES:
                        return True
                except (OSError, ValueError, TypeError):
                    continue
        return False

    def _write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temporary, path)

    def _mark_interrupted_jobs(self) -> None:
        jobs_dir = self.root / "jobs"
        if not jobs_dir.is_dir():
            return
        for metadata in jobs_dir.glob("*/metadata.json"):
            try:
                raw = json.loads(metadata.read_text(encoding="utf-8"))
                if raw.get("status") in {"diffing", "drafting"} and raw.get("mode", "automated") == "manual":
                    raw["status"] = "ready"
                    review: dict[str, Any] = raw.get("manualReview") or {"pairs": []}
                    review["status"] = "degraded"
                    review["guideStatus"] = "degraded"
                    raw["manualReview"] = review
                elif raw.get("status") in ACTIVE_STATUSES:
                    raw["status"] = "failed"
                    raw["error"] = {"code": "INTERRUPTED", "message": "Processing was interrupted by a server restart."}
                else:
                    continue
                raw["error"] = {"code": "INTERRUPTED", "message": "Processing was interrupted by a server restart."}
                self._write_json(metadata, raw)
            except (OSError, ValueError, TypeError):
                continue
