from __future__ import annotations

import json
import os
import threading
import re
from pathlib import Path
from typing import Any

from .models import (
    ActionTimeline,
    AnalysisEvent,
    AnalysisInfo,
    AnnotationSuggestionResult,
    Frame,
    Guide,
    GuideVerification,
    JobError,
    JobResponse,
    JobStatus,
    PartAnnotation,
    PartTrack,
    PipelineProgress,
    PipelineRunConfig,
    Storyboard,
    TrackSummary,
)


ACTIVE_STATUSES = {"queued", "extracting", "suggesting", "analyzing", "generating", "verifying", "correcting"}


class RevisionConflictError(RuntimeError):
    """A background review tried to commit against an older guide revision."""


class JobRepository:
    """Durable, one-directory-per-job storage with atomic JSON writes."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._mark_interrupted_jobs()

    def _job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def create(self, job_id: str, filename: str, *, pipeline: str = "plan3", pipeline_config: PipelineRunConfig | None = None) -> JobResponse:
        directory = self._job_dir(job_id)
        (directory / "frames").mkdir(parents=True, exist_ok=False)
        (directory / "input").mkdir()
        response = JobResponse(jobId=job_id, status="queued", pipeline=pipeline, pipelineConfig=pipeline_config)
        self._write_json(directory / "metadata.json", {**response.model_dump(), "filename": filename})
        return response

    def input_path(self, job_id: str, filename: str) -> Path:
        suffix = Path(filename).suffix.lower() or ".upload"
        return self._job_dir(job_id) / "input" / f"video{suffix}"

    def frame_path(self, job_id: str, frame_id: str) -> Path:
        return self._job_dir(job_id) / "frames" / f"{frame_id}.jpg"

    def artifact_path(self, job_id: str, relative_name: str) -> Path:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in relative.parts):
            raise ValueError("artifact path must be a safe relative name")
        return self._job_dir(job_id) / "artifacts" / relative

    def write_artifact(self, job_id: str, relative_name: str, payload: Any) -> Path:
        """Write an append-only diagnostic artifact without exposing secrets or media payloads."""
        target = self.artifact_path(job_id, relative_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        candidate = target
        index = 2
        while candidate.exists():
            candidate = target.with_name(f"{target.stem}-{index:02d}{target.suffix}")
            index += 1
        if isinstance(payload, str):
            content = payload
        else:
            content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        candidate.write_text(content, encoding="utf-8")
        return candidate

    def get(self, job_id: str) -> JobResponse | None:
        path = self._job_dir(job_id) / "metadata.json"
        if not path.is_file():
            return None
        with self._lock:
            raw = json.loads(path.read_text(encoding="utf-8"))
        response = JobResponse.model_validate(raw)
        normalized = self._normalize_legacy(response)
        if normalized.model_dump() != response.model_dump():
            self._write_json(path, {**normalized.model_dump(), **({"filename": raw["filename"]} if "filename" in raw else {})})
        return normalized

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        frames: list[Frame] | None = None,
        tracks: list[TrackSummary] | None = None,
        annotations: list[PartAnnotation] | None = None,
        annotation_suggestions: AnnotationSuggestionResult | None = None,
        part_tracks: list[PartTrack] | None = None,
        tracking_progress: float | None = None,
        events: list[AnalysisEvent] | None = None,
        analysis: AnalysisInfo | None = None,
        guide: Guide | None = None,
        guide_revision: int | None = None,
        verification: GuideVerification | None = None,
        timeline: ActionTimeline | None = None,
        pipeline_progress: PipelineProgress | None = None,
        inference_metrics: dict[str, float] | None = None,
        storyboard: Storyboard | None = None,
        error: JobError | None = None,
        expected_guide_revision: int | None = None,
        expected_storyboard_revision: int | None = None,
    ) -> JobResponse:
        with self._lock:
            current = self.get(job_id)
            if current is None:
                raise KeyError(job_id)
            if expected_guide_revision is not None and current.guideRevision != expected_guide_revision:
                raise RevisionConflictError(
                    f"Guide revision changed from {expected_guide_revision} to {current.guideRevision}."
                )
            if expected_storyboard_revision is not None:
                current_revision = current.storyboard.revision if current.storyboard is not None else 0
                if current_revision != expected_storyboard_revision:
                    raise RevisionConflictError(
                        f"Storyboard revision changed from {expected_storyboard_revision} to {current_revision}."
                    )
            updated = current.model_copy(
                update={
                    "status": status if status is not None else current.status,
                    "frames": frames if frames is not None else current.frames,
                    "tracks": tracks if tracks is not None else current.tracks,
                    "annotations": annotations if annotations is not None else current.annotations,
                    "annotationSuggestions": annotation_suggestions if annotation_suggestions is not None else current.annotationSuggestions,
                    "partTracks": part_tracks if part_tracks is not None else current.partTracks,
                    "trackingProgress": tracking_progress if tracking_progress is not None else current.trackingProgress,
                    "events": events if events is not None else current.events,
                    "analysis": analysis if analysis is not None else current.analysis,
                    "guide": guide if guide is not None else current.guide,
                    "guideRevision": guide_revision if guide_revision is not None else current.guideRevision,
                    "verification": verification if verification is not None else current.verification,
                    "timeline": timeline if timeline is not None else current.timeline,
                    "pipelineProgress": pipeline_progress if pipeline_progress is not None else current.pipelineProgress,
                    "inferenceMetrics": inference_metrics if inference_metrics is not None else current.inferenceMetrics,
                    "storyboard": storyboard if storyboard is not None else current.storyboard,
                    "error": error,
                }
            )
            directory = self._job_dir(job_id)
            self._write_json(directory / "metadata.json", updated.model_dump())
            if guide is not None:
                self._write_json(directory / "guide.json", guide.model_dump())
            return updated

    def save_storyboard(self, job_id: str, storyboard: Storyboard, *, expected_revision: int | None = None) -> JobResponse:
        current = self.get(job_id)
        if current is None:
            raise KeyError(job_id)
        current_revision = current.storyboard.revision if current.storyboard is not None else 0
        if expected_revision is not None and current_revision != expected_revision:
            raise RevisionConflictError(
                f"Storyboard revision changed from {expected_revision} to {current_revision}."
            )
        return self.update(
            job_id,
            storyboard=storyboard,
            status="annotating",
            pipeline_progress=PipelineProgress(stage="snapshots", progress=0, message="Review the selected state snapshots."),
            error=None,
            expected_storyboard_revision=current_revision,
        )

    def save_guide(self, job_id: str, guide: Guide, *, expected_revision: int | None = None) -> JobResponse:
        current = self.get(job_id)
        if current is None:
            raise KeyError(job_id)
        if expected_revision is not None and current.guideRevision != expected_revision:
            raise RevisionConflictError(
                f"Guide revision changed from {expected_revision} to {current.guideRevision}."
            )
        used_ids = {step.stepId for step in guide.steps if step.stepId}
        next_id = 1
        normalized_steps = []
        for step in guide.steps:
            step_id = step.stepId
            if not step_id:
                while f"step-{next_id:04d}" in used_ids:
                    next_id += 1
                step_id = f"step-{next_id:04d}"
                used_ids.add(step_id)
                next_id += 1
            normalized_steps.append(step.model_copy(update={"stepId": step_id}))
        guide = guide.model_copy(update={"steps": normalized_steps})
        next_revision = current.guideRevision + 1
        stale = None
        if current.verification is not None:
            stale = current.verification.model_copy(update={
                "status": "stale",
                "revision": next_revision,
                "message": "The guide changed after this verification; verify the saved revision again.",
            })
        return self.update(
            job_id,
            guide=guide,
            guide_revision=next_revision,
            verification=stale,
            error=None,
            expected_guide_revision=current.guideRevision,
        )

    def existing_frame_ids(self, job_id: str) -> set[str]:
        current = self.get(job_id)
        return {frame.frameId for frame in current.frames} if current else set()

    def _write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temporary, path)

    @staticmethod
    def _normalize_legacy(response: JobResponse) -> JobResponse:
        used: set[int] = set()
        next_part_id = 1
        annotations: list[PartAnnotation] = []
        for annotation in response.annotations:
            part_id = annotation.partId
            if part_id <= 0 or part_id in used:
                while next_part_id in used:
                    next_part_id += 1
                part_id = next_part_id
            used.add(part_id)
            next_part_id = max(next_part_id, part_id + 1)
            annotations.append(annotation.model_copy(update={"partId": part_id}))
        guide = response.guide
        if guide is not None and response.guideRevision == 0:
            guide = guide.model_copy(update={
                "steps": [
                    step.model_copy(update={"stepId": step.stepId or f"step-{index + 1:04d}"})
                    for index, step in enumerate(guide.steps)
                ]
            })
        return response.model_copy(update={"annotations": annotations, "guide": guide})

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
