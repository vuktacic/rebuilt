from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from backend.app.config import Settings
from backend.app.errors import AppError
from backend.app.main import create_app
from backend.app.models import AnalysisEvent, AnalysisInfo, Frame, Guide, GuideStep
from backend.app.processing import ExtractedFrames, JobProcessor, OpenAIResponsesGenerator, validate_guide
from backend.app.storage import JobRepository
from backend.app.vision import AnalysisResult


class FakeExtractor:
    def extract(self, input_path: Path, output_dir: Path, job_id: str) -> ExtractedFrames:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / "frame-0001.jpg", output_dir / "frame-0002.jpg"]
        for path in paths:
            path.write_bytes(b"fake-jpeg")
        return ExtractedFrames(
            frames=[
                Frame(frameId="frame-0001", timestampSeconds=0, imageUrl=f"/jobs/{job_id}/frames/frame-0001"),
                Frame(frameId="frame-0002", timestampSeconds=1, imageUrl=f"/jobs/{job_id}/frames/frame-0002"),
            ],
            paths=paths,
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


class NoEventAnalyzer:
    def analyze(self, frame_dir: Path, frame_count: int, job_id: str) -> AnalysisResult:
        return AnalysisResult(
            events=[],
            tracks=[],
            analysis=AnalysisInfo(backend="test", modelVersion="test", configVersion="test"),
        )


class NeverCalledGenerator:
    def generate(self, frames: list[Frame], frame_paths: list[Path]) -> Guide:
        raise AssertionError("a no-event analysis should not generate an unsupported guide")


def make_app(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, openai_api_key="test-key")
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


def wait_for_status(app, job_id: str, expected: set[str]) -> dict:
    for _ in range(50):
        response = asyncio.run(request(app, "GET", f"/jobs/{job_id}"))
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in expected:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {expected}")


def test_final_prompt_asks_the_llm_to_recover_a_missing_piece(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        Frame(frameId="frame-0002", timestampSeconds=1, imageUrl="/before"),
        Frame(frameId="frame-0001", timestampSeconds=0.5, imageUrl="/after"),
    ]
    paths = [tmp_path / "before.jpg", tmp_path / "after.jpg"]
    for path in paths:
        path.write_bytes(b"fake-jpeg")
    event = AnalysisEvent(
        eventId="event-0001",
        kind="uncertain_change",
        startTimestampSeconds=1,
        endTimestampSeconds=1.5,
        affectedTrackIds=["part:1"],
        beforeFrameId="frame-0002",
        afterFrameId="frame-0001",
        evidenceStrength=0.45,
        uncertainty="The local tracker lost red roof at the likely attachment boundary.",
        evidence="red roof is visible while separate, then becomes untracked near blue base.",
    )
    captured: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            guide = {"title": "Roof", "steps": [{"text": "Attach the red roof to the blue base.", "frameId": "frame-0001", "uncertainty": None}]}
            return json.dumps({"output_text": json.dumps(guide)}).encode("utf-8")

    def fake_urlopen(request, timeout: float):
        captured.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr("backend.app.processing.urllib.request.urlopen", fake_urlopen)

    guide = OpenAIResponsesGenerator(Settings(data_dir=tmp_path, openai_api_key="test-key")).generate(
        frames,
        paths,
        [event],
    )

    text_parts = [
        item["text"]
        for item in captured["input"][0]["content"]
        if item["type"] == "input_text"
    ]
    combined_prompt = "\n".join(text_parts)
    assert "local tracker lost a named piece" in combined_prompt
    assert "name the receiving part" in combined_prompt
    assert "affected=part:1" in combined_prompt
    assert guide.steps[0].text == "Attach the red roof to the blue base."


def test_upload_uses_manual_pipeline_and_extracts_frames(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    response = asyncio.run(request(app, "POST", "/jobs", data={"pipeline": "gemini_video"}, files={"video": ("build.mp4", b"video bytes", "video/mp4")}))
    job_id = response.json()["jobId"]
    job = wait_for_status(app, job_id, {"annotating"})
    assert response.status_code == 202
    assert job["pipeline"] == "manual_pairs"
    assert job["status"] == "annotating"
    assert len(job["frames"]) == 2

    frame = asyncio.run(request(app, "GET", f"/jobs/{job_id}/frames/frame-0001"))
    assert frame.status_code == 200
    assert frame.content == b"fake-jpeg"


def test_save_rejects_unknown_frame(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    job_id = "manual-ready"
    repository = app.state.repository
    repository.create(job_id, "build.mp4", pipeline="manual_pairs")
    repository.frame_path(job_id, "frame-0001").write_bytes(b"fake-jpeg")
    repository.update(job_id, status="ready", frames=[Frame(frameId="frame-0001", timestampSeconds=0, imageUrl=f"/jobs/{job_id}/frames/frame-0001")], guide=Guide(title="Ready", steps=[GuideStep(text="Place it.", frameId="frame-0001")]), guide_revision=1)
    response = asyncio.run(request(
            app,
            "PUT",
            f"/jobs/{job_id}/guide",
            json={"title": "Edited", "steps": [{"text": "No frame.", "frameId": "missing", "uncertainty": None}]},
        ))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_GUIDE"


def test_upload_size_limit_and_missing_resources(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, max_upload_bytes=3, openai_api_key="test-key")
    app = create_app(settings, processor=JobProcessor(JobRepository(tmp_path), settings, FakeExtractor(), FakeGenerator()))
    response = asyncio.run(request(app, "POST", "/jobs", files={"video": ("build.mp4", b"1234", "video/mp4")}))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
    assert asyncio.run(request(app, "GET", "/jobs/unknown")).status_code == 404
    assert asyncio.run(request(app, "GET", "/jobs/unknown/frames/frame-0001")).status_code == 404


def test_upload_rejects_non_video_and_model_failures_are_persisted(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, openai_api_key="test-key")
    repository = JobRepository(tmp_path)
    app = create_app(settings, processor=JobProcessor(repository, settings, FakeExtractor(), FailingGenerator()))
    invalid_upload = asyncio.run(request(app, "POST", "/jobs", files={"video": ("notes.txt", b"text", "text/plain")}))
    assert invalid_upload.status_code == 422
    repository.create("model-failure", "build.mp4", pipeline="plan3")
    repository.input_path("model-failure", "build.mp4").write_bytes(b"video")
    app.state.coordinator.processor.process("model-failure")
    failed = repository.get("model-failure").model_dump()
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "MODEL_FAILED"


def test_incomplete_model_frame_reference_fails_the_job(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, openai_api_key="test-key")
    repository = JobRepository(tmp_path)
    app = create_app(settings, processor=JobProcessor(repository, settings, FakeExtractor(), InvalidGenerator()))
    repository.create("invalid-guide", "build.mp4", pipeline="plan3")
    repository.input_path("invalid-guide", "build.mp4").write_bytes(b"video")
    app.state.coordinator.processor.process("invalid-guide")
    failed = repository.get("invalid-guide").model_dump()
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "INVALID_GUIDE"


def test_no_event_analysis_returns_a_reviewable_draft_instead_of_failing(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    repository = JobRepository(tmp_path)
    app = create_app(
        settings,
        processor=JobProcessor(repository, settings, FakeExtractor(), NeverCalledGenerator(), NoEventAnalyzer()),
    )

    repository.create("no-events", "build.mp4", pipeline="plan3")
    repository.input_path("no-events", "build.mp4").write_bytes(b"video")
    app.state.coordinator.processor.process("no-events")
    job = repository.get("no-events").model_dump()

    assert job["status"] == "ready"
    assert job["events"] == []
    assert job["error"] is None
    assert job["guide"] == {
        "title": "Build needs review",
        "steps": [{
            "text": "No reliable physical change was detected automatically. Review the recording and replace this draft with the first verified build step.",
            "frameId": "frame-0001",
            "uncertainty": "The local tracker could not maintain enough evidence for a confident change history.",
        }],
    }


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
