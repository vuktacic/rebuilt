from __future__ import annotations

from pathlib import Path

import asyncio
import time
import httpx
import pytest

from backend.app.config import Settings
from backend.app.errors import AppError
from backend.app.main import create_app
from backend.app.models import AnnotationSuggestion, Frame, PointPrompt
from backend.app.piece_suggestions import PieceSuggestionGenerator, normalized_point_to_pixels
from backend.app.processing import ExtractedFrames, JobProcessor, OpenAIResponsesGenerator
from backend.app.storage import JobRepository


def test_openai_model_setting_is_shared_by_suggestions_and_guides(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, model="gpt-shared-test")

    assert PieceSuggestionGenerator(settings).settings.model == "gpt-shared-test"
    assert OpenAIResponsesGenerator(settings).settings.model == "gpt-shared-test"


def test_piece_suggestions_convert_normalized_points_and_keep_frame_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"not-a-real-jpeg")
    captured: dict[str, object] = {}

    monkeypatch.setattr("backend.app.piece_suggestions._image_dimensions", lambda _: (100, 50))

    def fake_responses_json(settings: Settings, content: list[dict[str, object]]) -> dict[str, object]:
        captured["content"] = content
        return {"suggestions": [{"name": "red brick", "point": {"x": 0.25, "y": 0.5}, "confidence": 0.9}]}

    monkeypatch.setattr("backend.app.piece_suggestions._responses_json", fake_responses_json)
    result = PieceSuggestionGenerator(Settings(data_dir=tmp_path, openai_api_key="test")).suggest(
        Frame(frameId="frame-0001", timestampSeconds=2, imageUrl="/frame"), image,
    )

    assert result == [AnnotationSuggestion(name="red brick", frameIndex=0, point=PointPrompt(x=25, y=25), confidence=0.9)]
    assert captured["content"][1]["type"] == "input_image"  # type: ignore[index]
    assert captured["content"][1]["detail"] == "high"  # type: ignore[index]


def test_piece_suggestion_coordinates_reject_out_of_bounds_values() -> None:
    with pytest.raises(AppError, match="outside normalized image bounds"):
        normalized_point_to_pixels(PointPrompt(x=1.01, y=0.5), 100, 100)


class Extractor:
    def extract(self, input_path: Path, output_dir: Path, job_id: str) -> ExtractedFrames:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / "frame-0000.jpg", output_dir / "frame-0001.jpg"]
        for path in paths:
            path.write_bytes(b"frame")
        return ExtractedFrames(
            frames=[
                Frame(frameId="frame-0000", timestampSeconds=0, imageUrl="/frame-0000"),
                Frame(frameId="frame-0001", timestampSeconds=1, imageUrl="/frame-0001"),
            ],
            paths=paths,
        )


class Analyzer:
    def analyze_annotated(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("suggestion-only extraction should not start tracking")


class Suggestions:
    def __init__(self) -> None:
        self.frames: list[str] = []

    def suggest(self, frame: Frame, path: Path) -> list[AnnotationSuggestion]:
        self.frames.append(frame.frameId)
        return [AnnotationSuggestion(name="blue plate", frameIndex=0, point=PointPrompt(x=12, y=20), confidence=0.8)]


def test_upload_suggestions_run_once_on_the_final_extracted_frame_and_persist(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    repository = JobRepository(tmp_path)
    suggestions = Suggestions()
    processor = JobProcessor(repository, settings, extractor=Extractor(), analyzer=Analyzer(), suggestion_generator=suggestions)  # type: ignore[arg-type]
    repository.create("job-1", "build.mp4")
    repository.input_path("job-1", "build.mp4").write_bytes(b"video")

    processor.process("job-1")

    job = repository.get("job-1")
    assert job is not None
    assert job.status == "annotating"
    assert suggestions.frames == ["frame-0001"]
    assert job.annotationSuggestions is not None
    assert job.annotationSuggestions.frameIndex == 1
    assert job.annotationSuggestions.suggestions[0].name == "blue plate"
    assert JobRepository(tmp_path).get("job-1").annotationSuggestions is not None  # type: ignore[union-attr]


async def _request(app: object, method: str, url: str, **kwargs: object) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def test_suggestion_endpoint_allows_a_bounded_frame_retry(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    repository = JobRepository(tmp_path)
    suggestions = Suggestions()
    app = create_app(settings, processor=JobProcessor(repository, settings, extractor=Extractor(), analyzer=Analyzer(), suggestion_generator=suggestions))  # type: ignore[arg-type]
    created = asyncio.run(_request(app, "POST", "/jobs", files={"video": ("build.mp4", b"video", "video/mp4")}))
    job_id = created.json()["jobId"]
    for _ in range(100):
        job = asyncio.run(_request(app, "GET", f"/jobs/{job_id}")).json()
        if job["status"] == "annotating":
            break
        time.sleep(0.01)
    response = asyncio.run(_request(app, "POST", f"/jobs/{job_id}/annotation-suggestions", json={"frameIndex": 0}))
    assert response.status_code == 202
    for _ in range(100):
        job = asyncio.run(_request(app, "GET", f"/jobs/{job_id}")).json()
        if job["status"] == "annotating" and job["annotationSuggestions"]["frameIndex"] == 0:
            break
        time.sleep(0.01)
    assert suggestions.frames == ["frame-0001", "frame-0000"]
