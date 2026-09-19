from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .models import AnalysisEvent, AnalysisInfo, Frame, Guide, JobError, JobResponse, JobStatus, PartAnnotation, PartTrack, TrackSummary


ACTIVE_STATUSES = {"queued", "extracting", "analyzing", "generating"}


class JobRepository:
    """Durable, one-directory-per-job storage with atomic JSON writes."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._mark_interrupted_jobs()

    def _job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def create(self, job_id: str, filename: str) -> JobResponse:
        directory = self._job_dir(job_id)
        (directory / "frames").mkdir(parents=True, exist_ok=False)
        (directory / "input").mkdir()
        response = JobResponse(jobId=job_id, status="queued")
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
        error: JobError | None = None,
    ) -> JobResponse:
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
                "error": error,
            }
        )
        directory = self._job_dir(job_id)
        self._write_json(directory / "metadata.json", updated.model_dump())
        if guide is not None:
            self._write_json(directory / "guide.json", guide.model_dump())
        return updated

    def save_guide(self, job_id: str, guide: Guide) -> JobResponse:
        return self.update(job_id, guide=guide, error=None)

    def existing_frame_ids(self, job_id: str) -> set[str]:
        current = self.get(job_id)
        return {frame.frameId for frame in current.frames} if current else set()

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
                if raw.get("status") in ACTIVE_STATUSES:
                    raw["status"] = "failed"
                    raw["error"] = {"code": "INTERRUPTED", "message": "Processing was interrupted by a server restart."}
                    self._write_json(metadata, raw)
            except (OSError, ValueError, TypeError):
                continue
