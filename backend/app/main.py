from __future__ import annotations

import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import Settings, VisionRuntimeError
from .errors import AppError, error_payload
from .models import AnnotationRequest, AnnotationSuggestionRequest, CapabilitiesResponse, CompareRequest, DifferencesRequest, GenerateGuideRequest, Guide, JobCreated, JobResponse, PartAnnotation, PipelineCapability, PipelineRunConfig, StoryboardRequest, VerificationRequest
from .manual_pairs import build_storyboard
from .processing import JobProcessor, ground_annotations, validate_guide
from .storage import ACTIVE_STATUSES, JobRepository, RevisionConflictError
from .vision import NoopVisionAnalyzer, create_vision_analyzer, validate_vision_runtime


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".webm"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class JobCoordinator:
    def __init__(self, repository: JobRepository, processor: JobProcessor):
        self.repository = repository
        self.processor = processor
        self._lock = threading.Lock()
        self._active_job_id: str | None = None

    def start(self, job_id: str) -> None:
        with self._lock:
            self._active_job_id = job_id

        def run() -> None:
            try:
                self.processor.process(job_id)
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

        threading.Thread(target=run, name=f"rebuilt-job-{job_id}", daemon=True).start()

    def track_backward(self, job_id: str) -> None:
        with self._lock:
            self._active_job_id = job_id

        def run() -> None:
            try:
                job = self.repository.get(job_id)
                self.processor.track_annotated(job_id, job.annotations if job is not None else [])
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

        threading.Thread(target=run, name=f"rebuilt-track-{job_id}", daemon=True).start()

    def suggest_annotations(self, job_id: str, frame_index: int | None) -> None:
        with self._lock:
            self._active_job_id = job_id

        def run() -> None:
            try:
                self.processor.suggest_annotations(job_id, frame_index)
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

        threading.Thread(target=run, name=f"rebuilt-suggest-{job_id}", daemon=True).start()

    def verify(self, job_id: str, revision: int | None) -> None:
        with self._lock:
            self._active_job_id = job_id

        def run() -> None:
            try:
                self.processor.verify_job(job_id, revision)
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

        threading.Thread(target=run, name=f"rebuilt-verify-{job_id}", daemon=True).start()

    def analyze(self, job_id: str) -> None:
        with self._lock:
            self._active_job_id = job_id

        def run() -> None:
            try:
                job = self.repository.get(job_id)
                self.processor.analyze_pipeline(job_id, job.annotations if job is not None else [])
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

        threading.Thread(target=run, name=f"rebuilt-analyze-{job_id}", daemon=True).start()

    def compare_manual_pairs(self, job_id: str, pair_id: str | None = None) -> None:
        with self._lock:
            self._active_job_id = job_id

        def run() -> None:
            try:
                self.processor.compare_manual_pairs(job_id, pair_id)
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

        threading.Thread(target=run, name=f"rebuilt-compare-{job_id}", daemon=True).start()

    def generate_manual_guide(self, job_id: str, revision: int | None) -> None:
        with self._lock:
            self._active_job_id = job_id

        def run() -> None:
            try:
                self.processor.generate_manual_guide(job_id, revision)
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

        threading.Thread(target=run, name=f"rebuilt-write-{job_id}", daemon=True).start()

    def busy(self) -> bool:
        with self._lock:
            if self._active_job_id is None:
                return False
            current = self.repository.get(self._active_job_id)
            return current is not None and current.status in ACTIVE_STATUSES


def create_app(settings: Settings | None = None, *, processor: JobProcessor | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    repository = JobRepository(resolved.data_dir)
    vision_error: AppError | None = None
    if processor is None:
        try:
            validate_vision_runtime(resolved)
            analyzer = create_vision_analyzer(resolved)
        except VisionRuntimeError as exc:
            vision_error = AppError(503, exc.code, exc.message)
            analyzer = NoopVisionAnalyzer()
        except AppError as exc:
            vision_error = exc
            analyzer = NoopVisionAnalyzer()
        job_processor = JobProcessor(repository, resolved, analyzer=analyzer)
    else:
        job_processor = processor
    coordinator = JobCoordinator(repository, job_processor)

    app = FastAPI(title="Rebuilt", version="0.1.0")
    app.state.settings = resolved
    app.state.repository = repository
    app.state.coordinator = coordinator
    app.state.vision_error = vision_error

    def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(pipelines=[
            PipelineCapability(pipeline="manual_pairs", label="Manual snapshots + Astra", available=bool(resolved.openai_api_key), missing=[] if resolved.openai_api_key else ["OPENAI_API_KEY"], disclosure="You select settled state snapshots; Astra compares adjacent pairs and Luna writes the reviewed guide."),
        ])

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=error_payload(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content=error_payload("INVALID_INPUT", "The request body or upload is invalid."))

    @app.get("/health")
    async def health() -> dict[str, str]:
        if vision_error is not None:
            return {"status": "degraded", "vision": vision_error.code}
        return {"status": "ok", "vision": "ready"}

    @app.get("/capabilities", response_model=CapabilitiesResponse)
    async def get_capabilities() -> CapabilitiesResponse:
        return capabilities()

    @app.post("/jobs", response_model=JobCreated, status_code=202)
    async def create_job(
        video: Annotated[UploadFile, File(description="A phone recording of the build")],
    ) -> JobCreated:
        pipeline = "manual_pairs"
        selected_capability = capabilities().pipelines[0]
        if not selected_capability.available:
            raise AppError(503, "PIPELINE_UNAVAILABLE", f"The selected pipeline is unavailable: {', '.join(selected_capability.missing)}.")
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Another video is currently being processed.")
        filename = video.filename or "upload"
        suffix = Path(filename).suffix.lower()
        content_type = (video.content_type or "").lower()
        if not content_type.startswith("video/") and suffix not in VIDEO_EXTENSIONS:
            raise AppError(422, "VIDEO_INVALID", "Upload a supported video file.")
        job_id = str(uuid.uuid4())
        pipeline_config = PipelineRunConfig(
            pipeline=pipeline,
            model=resolved.manual_compare_model,
            reasoningConfiguration={"extractionFps": resolved.manual_extraction_fps, "reasoningEffort": resolved.manual_reasoning_effort},
            sam2Enabled=False,
            geminiPreview=False,
            comparisonModel=resolved.manual_compare_model,
            writerModel=resolved.manual_writer_model,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )
        repository.create(job_id, filename, pipeline=pipeline, pipeline_config=pipeline_config)
        destination = repository.input_path(job_id, filename)
        total = 0
        try:
            with destination.open("wb") as output:
                while chunk := await video.read(1024 * 1024):
                    total += len(chunk)
                    if total > resolved.max_upload_bytes:
                        raise AppError(413, "UPLOAD_TOO_LARGE", f"Video uploads are limited to {resolved.max_upload_bytes} bytes.")
                    output.write(chunk)
        except Exception:
            if destination.exists():
                destination.unlink()
            job_dir = repository._job_dir(job_id)
            for path in sorted(job_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            raise
        finally:
            await video.close()
        coordinator.start(job_id)
        return JobCreated(jobId=job_id)

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        return job

    @app.put("/jobs/{job_id}/storyboard", response_model=JobResponse)
    async def save_storyboard(job_id: str, request: StoryboardRequest) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if job.pipeline != "manual_pairs" or job.status != "annotating":
            raise AppError(409, "STORYBOARD_NOT_READY", "Snapshots can only be saved for the manual snapshot pipeline after extraction.")
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Wait for the current snapshot operation to finish.")
        current_revision = job.storyboard.revision if job.storyboard is not None else 0
        if request.revision is not None and request.revision != current_revision:
            raise AppError(409, "STALE_STORYBOARD", "The storyboard changed before these snapshots were saved.")
        storyboard = build_storyboard(job.frames, request.selectedFrameIds, request.context, revision=current_revision)
        if job.guide is not None:
            storyboard = storyboard.model_copy(update={"guideStale": True})
        try:
            return repository.save_storyboard(job_id, storyboard, expected_revision=current_revision)
        except RevisionConflictError as exc:
            raise AppError(409, "STALE_STORYBOARD", "The storyboard changed before these snapshots were saved.") from exc

    @app.post("/jobs/{job_id}/compare", response_model=JobResponse, status_code=202)
    async def compare_storyboard(job_id: str, request: CompareRequest | None = None) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if job.pipeline != "manual_pairs" or job.status != "annotating" or job.storyboard is None:
            raise AppError(409, "STORYBOARD_NOT_READY", "Save at least two snapshots before comparing them.")
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Another snapshot operation is currently running.")
        coordinator.compare_manual_pairs(job_id, request.pairId if request else None)
        return repository.get(job_id) or job

    @app.put("/jobs/{job_id}/differences", response_model=JobResponse)
    async def save_differences(job_id: str, request: DifferencesRequest) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Another snapshot operation is currently running.")
        try:
            job_processor.save_manual_differences(job_id, request.pairs, request.revision)
        except RevisionConflictError as exc:
            raise AppError(409, "STALE_STORYBOARD", "The storyboard changed before these differences were saved.") from exc
        return repository.get(job_id) or JobResponse(jobId=job_id, status="failed")

    @app.post("/jobs/{job_id}/generate-guide", response_model=JobResponse, status_code=202)
    async def generate_manual_guide(job_id: str, request: GenerateGuideRequest | None = None) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if job.pipeline != "manual_pairs" or job.storyboard is None:
            raise AppError(409, "STORYBOARD_NOT_READY", "Save and review snapshot differences first.")
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Another snapshot operation is currently running.")
        coordinator.generate_manual_guide(job_id, request.revision if request else job.storyboard.revision)
        return repository.get(job_id) or job

    @app.put("/jobs/{job_id}/annotations", response_model=JobResponse)
    async def save_annotations(job_id: str, request: AnnotationRequest) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if job.status != "annotating":
            raise AppError(409, "ANNOTATION_NOT_READY", "Annotations can only be edited after frame extraction and before tracking.")
        names = [annotation.name.strip() for annotation in request.annotations]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise AppError(422, "ANNOTATION_INVALID", "Each visible part needs a unique non-empty name.")
        if any(annotation.frameIndex >= len(job.frames) for annotation in request.annotations):
            raise AppError(422, "ANNOTATION_FRAME_INVALID", "An annotation refers to a frame that was not extracted.")
        if any(not annotation.points and annotation.box is None for annotation in request.annotations):
            raise AppError(422, "ANNOTATION_INVALID", "Each part needs a point or bounding box prompt.")
        used: set[int] = set()
        next_id = 1
        annotations: list[PartAnnotation] = []
        for annotation in request.annotations:
            part_id = annotation.partId
            if part_id <= 0 or part_id in used:
                while next_id in used:
                    next_id += 1
                part_id = next_id
            used.add(part_id)
            next_id = max(next_id, part_id + 1)
            annotations.append(annotation.model_copy(update={"partId": part_id, "name": annotation.name.strip()}))
        frame_paths = [repository.frame_path(job_id, frame.frameId) for frame in job.frames]
        annotations = ground_annotations(annotations, job.frames, frame_paths)
        return repository.update(job_id, annotations=annotations, error=None)

    @app.post("/jobs/{job_id}/annotation-suggestions", response_model=JobResponse, status_code=202)
    async def suggest_annotations(job_id: str, request: AnnotationSuggestionRequest | None = None) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if job.status != "annotating":
            raise AppError(409, "ANNOTATION_NOT_READY", "Piece suggestions are available after frame extraction.")
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Another analysis operation is currently running.")
        repository.update(job_id, status="suggesting", error=None)
        coordinator.suggest_annotations(job_id, request.frameIndex if request else None)
        return repository.get(job_id) or job

    @app.post("/jobs/{job_id}/track", response_model=JobResponse, status_code=202)
    async def track_backward(job_id: str) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if job.status != "annotating":
            raise AppError(409, "ANNOTATION_NOT_READY", "This video is not waiting for annotations.")
        if job.pipeline != "plan3":
            raise AppError(409, "PIPELINE_ENDPOINT", "The selected action-first pipeline uses /analyze, not full-video tracking.")
        if not job.annotations:
            raise AppError(422, "ANNOTATIONS_REQUIRED", "Add and save at least one named part before tracking backward.")
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Another video is currently being processed.")
        coordinator.track_backward(job_id)
        return repository.get(job_id) or job

    @app.post("/jobs/{job_id}/analyze", response_model=JobResponse, status_code=202)
    async def analyze_job(job_id: str) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if job.pipeline == "plan3":
            raise AppError(409, "PIPELINE_ENDPOINT", "The baseline pipeline uses the existing tracking endpoint.")
        if job.status != "annotating":
            raise AppError(409, "ANALYSIS_NOT_READY", "Confirm the inventory after frame extraction before analysis.")
        if not job.annotations:
            raise AppError(422, "ANNOTATIONS_REQUIRED", "Confirm at least one named object before analysis.")
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Another video is currently being processed.")
        coordinator.analyze(job_id)
        return repository.get(job_id) or job

    @app.put("/jobs/{job_id}/guide", response_model=Guide)
    async def save_guide(job_id: str, guide: Guide) -> Guide:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if job.status != "ready":
            raise AppError(409, "GUIDE_NOT_READY", "The guide cannot be saved until processing is complete.")
        if job.pipeline == "manual_pairs" and job.storyboard is not None:
            pair_ids = {pair.pairId for pair in job.storyboard.pairs}
            if any(step.sourcePairId is not None and step.sourcePairId not in pair_ids for step in guide.steps):
                raise AppError(422, "INVALID_GUIDE", "A manual guide step references an unknown snapshot pair.")
        validated = validate_guide(guide, job.frames, job.events, job.timeline)
        return repository.save_guide(job_id, validated).guide  # type: ignore[return-value]

    @app.post("/jobs/{job_id}/verify", response_model=JobResponse, status_code=202)
    async def verify_job(job_id: str, request: VerificationRequest | None = None) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if job.status != "ready" or job.guide is None:
            raise AppError(409, "GUIDE_NOT_READY", "The guide cannot be verified until processing is complete.")
        if job.pipeline == "manual_pairs":
            raise AppError(409, "PIPELINE_ENDPOINT", "The manual snapshot workflow uses reviewed differences instead of automated verification.")
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Another analysis operation is currently running.")
        revision = request.revision if request else job.guideRevision
        if revision is not None and revision != job.guideRevision:
            raise AppError(409, "STALE_VERIFICATION", "The guide changed before verification started.")
        if request and request.guide is not None:
            validated = validate_guide(request.guide, job.frames, job.events, job.timeline)
            try:
                saved = repository.save_guide(job_id, validated, expected_revision=job.guideRevision)
            except RevisionConflictError as exc:
                raise AppError(409, "STALE_VERIFICATION", "The guide changed before verification started.") from exc
            revision = saved.guideRevision
        repository.update(job_id, status="verifying", error=None)
        coordinator.verify(job_id, revision)
        return repository.get(job_id) or job

    @app.get("/jobs/{job_id}/frames/{frame_id}")
    async def get_frame(job_id: str, frame_id: str) -> Response:
        if not SAFE_ID.fullmatch(job_id) or not SAFE_ID.fullmatch(frame_id):
            raise AppError(404, "FRAME_NOT_FOUND", "The requested frame does not exist.")
        job = repository.get(job_id)
        if job is None or frame_id not in {frame.frameId for frame in job.frames}:
            raise AppError(404, "FRAME_NOT_FOUND", "The requested frame does not exist.")
        path = repository.frame_path(job_id, frame_id)
        if not path.is_file():
            raise AppError(404, "FRAME_NOT_FOUND", "The requested frame does not exist.")
        return Response(content=path.read_bytes(), media_type="image/jpeg")

    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app


app = create_app()
