from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import Settings, VisionRuntimeError
from .errors import AppError, error_payload
from .models import AnnotationRequest, Guide, JobCreated, JobResponse
from .processing import JobProcessor, validate_guide
from .storage import ACTIVE_STATUSES, JobRepository
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

    @app.post("/jobs", response_model=JobCreated, status_code=202)
    async def create_job(video: Annotated[UploadFile, File(description="A phone recording of the build")]) -> JobCreated:
        if vision_error is not None:
            raise vision_error
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Another video is currently being processed.")
        filename = video.filename or "upload"
        suffix = Path(filename).suffix.lower()
        content_type = (video.content_type or "").lower()
        if not content_type.startswith("video/") and suffix not in VIDEO_EXTENSIONS:
            raise AppError(422, "VIDEO_INVALID", "Upload a supported video file.")
        job_id = str(uuid.uuid4())
        repository.create(job_id, filename)
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
        return repository.update(job_id, annotations=request.annotations, error=None)

    @app.post("/jobs/{job_id}/track", response_model=JobResponse, status_code=202)
    async def track_backward(job_id: str) -> JobResponse:
        if not SAFE_ID.fullmatch(job_id):
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        job = repository.get(job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "The requested job does not exist.")
        if job.status != "annotating":
            raise AppError(409, "ANNOTATION_NOT_READY", "This video is not waiting for annotations.")
        if not job.annotations:
            raise AppError(422, "ANNOTATIONS_REQUIRED", "Add and save at least one named part before tracking backward.")
        if coordinator.busy():
            raise AppError(409, "PROCESSING_BUSY", "Another video is currently being processed.")
        coordinator.track_backward(job_id)
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
        validated = validate_guide(guide, job.frames)
        return repository.save_guide(job_id, validated).guide  # type: ignore[return-value]

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
