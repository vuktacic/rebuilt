from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest

from backend.app.config import Settings
from backend.app.errors import AppError
from backend.app.main import create_app
from backend.app.models import Frame, Guide, GuideStep
from backend.app.processing import ExtractedFrames, JobProcessor, validate_guide
from backend.app.storage import JobRepository


class FakeExtractor:
    def extract(self, input_path: Path, output_dir: Path, job_id: str) -> ExtractedFrames:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_path = output_dir / "frame-0001.jpg"
        frame_path.write_bytes(b"fake-jpeg")
        return ExtractedFrames(
            frames=[Frame(frameId="frame-0001", timestampSeconds=0, imageUrl=f"/jobs/{job_id}/frames/frame-0001")],
            paths=[frame_path],
        )


class FakeGenerator:
    def generate(self, frames: list[Frame], frame_paths: list[Path]) -> Guide:
        return Guide(title="Test build", steps=[GuideStep(text="Place the brick.", frameId=frames[0].frameId)])


class FailingGenerator:
    def generate(self, frames: list[Frame], frame_paths: list[Path]) -> Guide:
        raise AppError(502, "MODEL_FAILED", "model unavailable")


class InvalidGenerator:
    def generate(self, frames: list[Frame], frame_paths: list[Path]) -> Guide:
        return Guide(title="Invalid", steps=[GuideStep(text="Invented frame.", frameId="missing")])


def make_app(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)
    repository = JobRepository(tmp_path)
    processor = JobProcessor(repository, settings, extractor=FakeExtractor(), generator=FakeGenerator())
    return create_app(settings, processor=processor)


async def request(app, method: str, url: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def wait_for_ready(app, job_id: str) -> dict:
    for _ in range(50):
        response = asyncio.run(request(app, "GET", f"/jobs/{job_id}"))
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"ready", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_upload_processing_frame_and_save(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    response = asyncio.run(request(app, "POST", "/jobs", files={"video": ("build.mp4", b"video bytes", "video/mp4")}))
    job_id = response.json()["jobId"]
    job = wait_for_ready(app, job_id)
    assert response.status_code == 202
    assert job["status"] == "ready"
    assert job["guide"]["steps"][0]["frameId"] == "frame-0001"

    frame = asyncio.run(request(app, "GET", f"/jobs/{job_id}/frames/frame-0001"))
    assert frame.status_code == 200
    assert frame.content == b"fake-jpeg"

    saved = asyncio.run(request(
        app,
        "PUT",
        f"/jobs/{job_id}/guide",
        json={"title": "Edited build", "steps": [{"text": "Edited.", "frameId": "frame-0001", "uncertainty": None}]},
    ))
    assert saved.status_code == 200
    assert saved.json()["title"] == "Edited build"
    assert asyncio.run(request(app, "GET", f"/jobs/{job_id}")).json()["guide"]["title"] == "Edited build"


def test_save_rejects_unknown_frame(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    created = asyncio.run(request(app, "POST", "/jobs", files={"video": ("build.mp4", b"video bytes", "video/mp4")}))
    job_id = created.json()["jobId"]
    assert wait_for_ready(app, job_id)["status"] == "ready"
    response = asyncio.run(request(
            app,
            "PUT",
            f"/jobs/{job_id}/guide",
            json={"title": "Edited", "steps": [{"text": "No frame.", "frameId": "missing", "uncertainty": None}]},
        ))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_GUIDE"


def test_upload_size_limit_and_missing_resources(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, max_upload_bytes=3)
    app = create_app(settings, processor=JobProcessor(JobRepository(tmp_path), settings, FakeExtractor(), FakeGenerator()))
    response = asyncio.run(request(app, "POST", "/jobs", files={"video": ("build.mp4", b"1234", "video/mp4")}))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
    assert asyncio.run(request(app, "GET", "/jobs/unknown")).status_code == 404
    assert asyncio.run(request(app, "GET", "/jobs/unknown/frames/frame-0001")).status_code == 404


def test_upload_rejects_non_video_and_model_failures_are_persisted(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    repository = JobRepository(tmp_path)
    app = create_app(settings, processor=JobProcessor(repository, settings, FakeExtractor(), FailingGenerator()))
    invalid_upload = asyncio.run(request(app, "POST", "/jobs", files={"video": ("notes.txt", b"text", "text/plain")}))
    assert invalid_upload.status_code == 422
    created = asyncio.run(request(app, "POST", "/jobs", files={"video": ("build.mp4", b"video", "video/mp4")}))
    failed = wait_for_ready(app, created.json()["jobId"])
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "MODEL_FAILED"


def test_incomplete_model_frame_reference_fails_the_job(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    repository = JobRepository(tmp_path)
    app = create_app(settings, processor=JobProcessor(repository, settings, FakeExtractor(), InvalidGenerator()))
    created = asyncio.run(request(app, "POST", "/jobs", files={"video": ("build.mp4", b"video", "video/mp4")}))
    failed = wait_for_ready(app, created.json()["jobId"])
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "INVALID_GUIDE"


def test_interrupted_jobs_are_marked_failed(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path)
    repository.create("old-job", "old.mp4")
    repository.update("old-job", status="generating")
    restarted = JobRepository(tmp_path)
    job = restarted.get("old-job")
    assert job is not None
    assert job.status == "failed"
    assert job.error is not None and job.error.code == "INTERRUPTED"


def test_validate_guide_requires_text_and_known_frame() -> None:
    frames = [Frame(frameId="frame-0001", timestampSeconds=0, imageUrl="/frame")]
    with pytest.raises(Exception, match="non-empty"):
        validate_guide(Guide(title="x", steps=[GuideStep(text=" ", frameId="frame-0001")]), frames)
