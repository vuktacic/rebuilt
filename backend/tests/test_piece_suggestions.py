from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.manual_pairs import ManualPairsProvider, approved_context
from backend.app.models import AnnotationSuggestion, AnnotationSuggestionResult, Frame, ManualPair, PairFinding
from backend.app.post_processing import normalized_point_to_pixels
from backend.app.processing import ExtractedFrames, JobProcessor
from backend.app.storage import JobRepository


class Extractor:
    def extract(self, input_path: Path, output_dir: Path, job_id: str) -> ExtractedFrames:
        output_dir.mkdir(parents=True, exist_ok=True)
        frames, paths = [], []
        for index in range(3):
            path = output_dir / f"frame-{index:04d}.jpg"
            path.write_bytes(b"not-an-image")
            frames.append(Frame(frameId=f"frame-{index:04d}", timestampSeconds=float(index), imageUrl=f"/frame-{index:04d}"))
            paths.append(path)
        return ExtractedFrames(frames=frames, paths=paths)


class Suggestions:
    calls = 0

    def suggest(self, frame: Frame, path: Path) -> AnnotationSuggestionResult:
        self.calls += 1
        return AnnotationSuggestionResult(
            status="completed", referenceFrameId=frame.frameId, imageWidth=100, imageHeight=100, model="gpt-5.4",
            requestId="resp-test", rawSuggestions=[AnnotationSuggestion(
                suggestionId="suggestion-0001", name="red brick", description="one-by-two red brick",
                point={"x": 20, "y": 30}, confidence=0.9,
            )],
        )


class EmptySuggestions:
    def suggest(self, frame: Frame, path: Path) -> AnnotationSuggestionResult:
        return AnnotationSuggestionResult(status="completed", referenceFrameId=frame.frameId, imageWidth=100, imageHeight=100, model="gpt-5.4", requestId="resp-empty")


class FailingSuggestions:
    def suggest(self, frame: Frame, path: Path) -> AnnotationSuggestionResult:
        raise RuntimeError("provider unavailable")


class PairProvider:
    def compare(self, pair, before_path, after_path, context):
        assert "red brick" in context
        return PairFinding(status="change", beforeAfterDifference="The brick was added.")

    def write_guide(self, pairs, context):
        included = [pair for pair in pairs if pair.disposition == "include" and pair.reviewedFinding]
        return "Build", {pair.pairId: ("Place the brick.", None) for pair in included}


async def request(app, method: str, url: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def wait_for_status(app, job_id: str, status: str) -> dict:
    for _ in range(100):
        payload = asyncio.run(request(app, "GET", f"/jobs/{job_id}")).json()
        if payload["status"] == status:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {status}")


def test_manual_upload_suggests_once_and_refresh_preserves_raw_result(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, manual_only=True)
    suggestions = Suggestions()
    repository = JobRepository(tmp_path)
    processor = JobProcessor(repository, settings, extractor=Extractor(), suggestion_generator=suggestions)
    app = create_app(settings, processor=processor)
    created = asyncio.run(request(app, "POST", "/jobs", data={"mode": "manual"}, files={"video": ("build.mp4", b"x", "video/mp4")}))
    job = wait_for_status(app, created.json()["jobId"], "pairing")
    assert suggestions.calls == 1
    assert job["annotationSuggestions"]["referenceFrameId"] == "frame-0000"
    refreshed = asyncio.run(request(app, "GET", f"/jobs/{created.json()['jobId']}")).json()
    assert refreshed["annotationSuggestions"]["rawSuggestions"][0]["name"] == "red brick"


def test_approved_piece_revision_and_human_edits_survive_suggestion_retry(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, manual_only=True)
    suggestions = Suggestions()
    repository = JobRepository(tmp_path)
    processor = JobProcessor(repository, settings, extractor=Extractor(), suggestion_generator=suggestions)
    app = create_app(settings, processor=processor)
    created = asyncio.run(request(app, "POST", "/jobs", data={"mode": "manual"}, files={"video": ("build.mp4", b"x", "video/mp4")}))
    job_id = created.json()["jobId"]
    wait_for_status(app, job_id, "pairing")
    saved = asyncio.run(request(app, "PUT", f"/jobs/{job_id}/annotations", json={
        "revision": 0, "annotations": [{"partId": 9, "name": "human name", "description": "edited", "frameIndex": 0, "points": [{"x": 22, "y": 33}] }],
    }))
    assert saved.status_code == 200
    retry = asyncio.run(request(app, "POST", f"/jobs/{job_id}/annotation-suggestions", json={"frameIndex": 1}))
    wait_for_status(app, job_id, "pairing")
    final = asyncio.run(request(app, "GET", f"/jobs/{job_id}")).json()
    assert retry.status_code == 202
    assert final["annotations"][0]["name"] == "human name"
    assert final["annotationSuggestions"]["referenceFrameId"] == "frame-0001"


def test_approved_coordinates_are_only_shared_for_the_pair_reference_frames() -> None:
    annotations = [{
        "partId": 1, "name": "base", "description": "blue base", "frameIndex": 0,
        "points": [{"x": 10, "y": 20}], "referenceFrameId": "frame-0000",
    }]
    from backend.app.models import PartAnnotation

    parsed = [PartAnnotation.model_validate(item) for item in annotations]
    assert '"referencePoint": [{"x": 10.0, "y": 20.0}]' in approved_context(parsed, frame_ids={"frame-0000", "frame-0001"})
    assert '"referencePoint": null' in approved_context(parsed, frame_ids={"frame-0002", "frame-0003"})


def test_manual_review_pairs_and_text_only_guide_survive_refresh(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, manual_only=True)
    repository = JobRepository(tmp_path)
    processor = JobProcessor(repository, settings, extractor=Extractor(), suggestion_generator=Suggestions(), manual_provider=PairProvider())
    app = create_app(settings, processor=processor)
    created = asyncio.run(request(app, "POST", "/jobs", data={"mode": "manual"}, files={"video": ("build.mp4", b"x", "video/mp4")}))
    job_id = created.json()["jobId"]
    pairing = wait_for_status(app, job_id, "pairing")

    approved = asyncio.run(request(app, "PUT", f"/jobs/{job_id}/annotations", json={
        "revision": pairing["revision"], "annotations": [{"partId": 4, "name": "red brick", "description": "edited visual context", "frameIndex": 0, "points": [{"x": 20, "y": 30}]}],
    }))
    assert approved.status_code == 200
    storyboard = asyncio.run(request(app, "PUT", f"/jobs/{job_id}/storyboard", json={
        "revision": approved.json()["revision"], "selectedFrameIds": ["frame-0000", "frame-0001", "frame-0002"], "context": "keep the red brick name",
    }))
    assert storyboard.status_code == 200
    compared = asyncio.run(request(app, "POST", f"/jobs/{job_id}/compare", json={}))
    assert compared.status_code == 202
    reviewed = wait_for_status(app, job_id, "pairing")
    pair = reviewed["manualPairs"][0]
    pair["disposition"] = "include"
    pair["reviewedFinding"] = pair["rawFinding"]
    saved = asyncio.run(request(app, "PUT", f"/jobs/{job_id}/differences", json={
        "revision": reviewed["revision"], "context": "keep the red brick name", "pairs": reviewed["manualPairs"],
    }))
    assert saved.status_code == 200
    started = asyncio.run(request(app, "POST", f"/jobs/{job_id}/generate-guide", json={"revision": saved.json()["revision"]}))
    assert started.status_code == 202
    ready = wait_for_status(app, job_id, "ready")
    assert ready["guide"]["steps"][0]["sourcePairId"] == pair["pairId"]
    refreshed = asyncio.run(request(app, "GET", f"/jobs/{job_id}")).json()
    assert refreshed["manualContext"] == "keep the red brick name"
    assert refreshed["guideStale"] is False


def test_pair_provider_submits_exactly_two_images_and_no_foreign_point(tmp_path: Path) -> None:
    import backend.app.manual_pairs as manual_pairs

    before = tmp_path / "before.jpg"
    after = tmp_path / "after.jpg"
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    captured = []
    original = manual_pairs.responses_json

    def fake_responses_json(settings, *, content, name, schema, model=None):
        captured.append(content)
        return {"status": "change", "changedPieceDescription": "brick", "receivingPieceDescription": "base", "receivingLocation": "top", "beforeAfterDifference": "brick added", "supportedPlacement": "visible", "uncertainty": None, "reason": None, "suggestion": None}

    manual_pairs.responses_json = fake_responses_json
    try:
        provider = ManualPairsProvider(Settings(data_dir=tmp_path, openai_api_key="test"))
        finding = provider.compare(ManualPair(pairId="p1", sequence=1, beforeFrameId="frame-1", afterFrameId="frame-2"), before, after, '{"referencePoint": null}')
    finally:
        manual_pairs.responses_json = original
    assert finding.status == "change"
    assert sum(item.get("type") == "input_image" for item in captured[0]) == 2
    assert "referencePoint" in captured[0][1]["text"]


def test_guide_writer_is_text_only(tmp_path: Path) -> None:
    import backend.app.manual_pairs as manual_pairs

    captured = []
    original = manual_pairs.responses_json

    def fake_responses_json(settings, *, content, name, schema, model=None):
        captured.append(content)
        return {"title": "Build", "steps": [{"pairId": "p1", "text": "Place the brick.", "uncertainty": None}]}

    manual_pairs.responses_json = fake_responses_json
    try:
        provider = ManualPairsProvider(Settings(data_dir=tmp_path, openai_api_key="test"))
        pair = ManualPair(pairId="p1", sequence=1, beforeFrameId="frame-1", afterFrameId="frame-2", disposition="include", reviewedFinding=PairFinding(status="change", beforeAfterDifference="brick added"))
        provider.write_guide([pair], "approved context")
    finally:
        manual_pairs.responses_json = original
    assert all(item.get("type") == "input_text" for item in captured[0])


def test_normalized_suggestion_coordinates_reject_out_of_bounds_points() -> None:
    try:
        normalized_point_to_pixels(type("Point", (), {"x": 1.1, "y": 0.5})(), 100, 100)
    except Exception as exc:
        assert getattr(exc, "code", None) == "MODEL_INVALID_OUTPUT"
    else:
        raise AssertionError("out-of-bounds model point was accepted")


def test_empty_suggestions_are_persisted_as_a_completed_reviewable_result(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, manual_only=True)
    repository = JobRepository(tmp_path)
    app = create_app(settings, processor=JobProcessor(repository, settings, extractor=Extractor(), suggestion_generator=EmptySuggestions()))
    created = asyncio.run(request(app, "POST", "/jobs", data={"mode": "manual"}, files={"video": ("build.mp4", b"x", "video/mp4")}))
    job = wait_for_status(app, created.json()["jobId"], "pairing")
    assert job["annotationSuggestions"]["status"] == "completed"
    assert job["annotationSuggestions"]["rawSuggestions"] == []


def test_provider_failure_is_recoverable_and_duplicate_approved_names_are_rejected(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, manual_only=True)
    repository = JobRepository(tmp_path)
    app = create_app(settings, processor=JobProcessor(repository, settings, extractor=Extractor(), suggestion_generator=FailingSuggestions()))
    created = asyncio.run(request(app, "POST", "/jobs", data={"mode": "manual"}, files={"video": ("build.mp4", b"x", "video/mp4")}))
    job_id = created.json()["jobId"]
    job = wait_for_status(app, job_id, "pairing")
    assert job["annotationSuggestions"]["status"] == "failed"
    duplicate = asyncio.run(request(app, "PUT", f"/jobs/{job_id}/annotations", json={
        "revision": job["revision"], "annotations": [
            {"name": "same", "frameIndex": 0, "points": [{"x": 1, "y": 1}]},
            {"name": " SAME ", "frameIndex": 0, "points": [{"x": 2, "y": 2}]},
        ],
    }))
    assert duplicate.status_code == 422
