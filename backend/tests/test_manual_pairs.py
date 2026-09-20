from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.manual_pairs import ManualPairsProvider, build_storyboard, compare_pairs_bounded
from backend.app.models import Frame, Guide, GuideStep, PairFinding
from backend.app.processing import ExtractedFrames, JobProcessor
from backend.app.storage import JobRepository


def frames() -> list[Frame]:
    return [Frame(frameId=f"frame-{index:04d}", timestampSeconds=float(index), imageUrl=f"/frame-{index:04d}") for index in range(4)]


def finding(status: str = "change") -> PairFinding:
    return PairFinding(status=status, beforeAfterDifference="The after state has one visible added piece.")


def test_storyboard_creates_n_minus_one_ordered_server_pairs() -> None:
    storyboard = build_storyboard(frames(), ["frame-0000", "frame-0002", "frame-0003"], "blue base")
    assert storyboard.selectedFrameIds == ["frame-0000", "frame-0002", "frame-0003"]
    assert [(pair.pairId, pair.beforeFrameId, pair.afterFrameId) for pair in storyboard.pairs] == [
        ("pair-0001", "frame-0000", "frame-0002"),
        ("pair-0002", "frame-0002", "frame-0003"),
    ]
    with pytest.raises(Exception, match="once"):
        build_storyboard(frames(), ["frame-0000", "frame-0000"])
    with pytest.raises(Exception, match="source-time"):
        build_storyboard(frames(), ["frame-0002", "frame-0001"])


def test_astra_receives_exactly_two_labeled_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    before = tmp_path / "before.jpg"; after = tmp_path / "after.jpg"
    before.write_bytes(b"before"); after.write_bytes(b"after")
    captured: dict = {}

    def fake_responses(settings, **kwargs):
        captured.update(kwargs)
        return finding().model_dump()

    monkeypatch.setattr("backend.app.manual_pairs.responses_json", fake_responses)
    storyboard = build_storyboard(frames(), ["frame-0000", "frame-0001"])
    result = ManualPairsProvider(Settings(data_dir=tmp_path, openai_api_key="test")).compare(storyboard.pairs[0], before, after, "red roof")
    assert result.status == "change"
    content = captured["content"]
    assert [item["type"] for item in content if item["type"] == "input_image"] == ["input_image", "input_image"]
    labels = [item["text"] for item in content if item["type"] == "input_text"]
    assert any("BEFORE" in label for label in labels)
    assert any("AFTER" in label for label in labels)
    assert captured["model"] == "gpt-6-astra"
    assert captured["reasoning_effort"] == "low"


def test_pair_batch_preserves_results_when_completion_order_differs(tmp_path: Path) -> None:
    class SlowProvider:
        def compare(self, pair, before_path, after_path, context):
            if pair.pairId == "pair-0001":
                time.sleep(0.03)
            return finding()

    storyboard = build_storyboard(frames(), ["frame-0000", "frame-0001", "frame-0002"])
    results = compare_pairs_bounded(SlowProvider(), storyboard.pairs, lambda frame_id: tmp_path / f"{frame_id}.jpg", "")
    assert list(results) == ["pair-0002", "pair-0001"]
    assert all(isinstance(value, PairFinding) for value in results.values())


class ManualExtractor:
    def extract(self, input_path: Path, output_dir: Path, job_id: str, **kwargs) -> ExtractedFrames:
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        paths = []
        for index in range(3):
            path = output_dir / f"frame-{index:04d}.jpg"
            path.write_bytes(b"fake-jpeg")
            extracted.append(Frame(frameId=f"frame-{index:04d}", timestampSeconds=float(index), imageUrl=f"/jobs/{job_id}/frames/frame-{index:04d}"))
            paths.append(path)
        return ExtractedFrames(frames=extracted, paths=paths)


async def request(app, method: str, url: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def wait_for_status(app, job_id: str, statuses: set[str]) -> dict:
    for _ in range(100):
        payload = asyncio.run(request(app, "GET", f"/jobs/{job_id}")).json()
        if payload["status"] in statuses:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {statuses}")


def test_manual_api_persists_pair_results_and_rejects_stale_reviews(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeManualProvider:
        def __init__(self, settings):
            pass

        def compare(self, pair, before_path, after_path, context):
            return finding()

        def write_guide(self, storyboard):
            included = [pair.pairId for pair in storyboard.pairs if pair.disposition == "include"]
            return "Snapshot guide", {pair_id: (f"Place the change from {pair_id}.", None) for pair_id in included}

    monkeypatch.setattr("backend.app.processing.ManualPairsProvider", FakeManualProvider)
    settings = Settings(data_dir=tmp_path, openai_api_key="test")
    repository = JobRepository(tmp_path)
    processor = JobProcessor(repository, settings, extractor=ManualExtractor())
    app = create_app(settings, processor=processor)
    created = asyncio.run(request(app, "POST", "/jobs", data={"pipeline": "manual_pairs"}, files={"video": ("build.mp4", b"video", "video/mp4")}))
    assert created.status_code == 202, created.text
    job_id = created.json()["jobId"]
    extracted = wait_for_status(app, job_id, {"annotating"})
    saved = asyncio.run(request(app, "PUT", f"/jobs/{job_id}/storyboard", json={"selectedFrameIds": ["frame-0000", "frame-0001", "frame-0002"], "context": "parts"}))
    assert saved.status_code == 200, saved.text
    compared = asyncio.run(request(app, "POST", f"/jobs/{job_id}/compare", json={}))
    assert compared.status_code == 202
    reviewed = wait_for_status(app, job_id, {"annotating"})
    assert all(pair["status"] == "completed" for pair in reviewed["storyboard"]["pairs"])
    revision = reviewed["storyboard"]["revision"]
    stale = asyncio.run(request(app, "PUT", f"/jobs/{job_id}/differences", json={"revision": revision - 1, "pairs": []}))
    assert stale.status_code == 409
    reviews = [{"pairId": pair["pairId"], "reviewedFinding": pair["rawFinding"], "disposition": "include"} for pair in reviewed["storyboard"]["pairs"]]
    saved_reviews = asyncio.run(request(app, "PUT", f"/jobs/{job_id}/differences", json={"revision": revision, "pairs": reviews}))
    assert saved_reviews.status_code == 200
    generated = asyncio.run(request(app, "POST", f"/jobs/{job_id}/generate-guide", json={"revision": saved_reviews.json()["storyboard"]["revision"]}))
    assert generated.status_code == 202
    ready = wait_for_status(app, job_id, {"ready"})
    assert [step["sourcePairId"] for step in ready["guide"]["steps"]] == ["pair-0001", "pair-0002"]
    refreshed = asyncio.run(request(app, "GET", f"/jobs/{job_id}"))
    assert refreshed.json()["storyboard"]["pairs"][0]["rawFinding"]["status"] == "change"
