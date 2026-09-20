from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.models import AnalysisEvent, AnalysisInfo, Frame, Guide, GuideStep
from backend.app.processing import ExtractedFrames, JobProcessor, deduplicate_guide_steps
from backend.app.storage import JobRepository
from backend.app.vision import AnalysisResult


class ManualExtractor:
    def extract(self, input_path: Path, output_dir: Path, job_id: str) -> ExtractedFrames:
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[Frame] = []
        paths: list[Path] = []
        for index in range(3):
            path = output_dir / f"frame-{index:04d}.jpg"
            path.write_bytes(b"fake-jpeg")
            frames.append(Frame(frameId=f"frame-{index:04d}", timestampSeconds=float(index), imageUrl=f"/jobs/{job_id}/frames/frame-{index:04d}"))
            paths.append(path)
        return ExtractedFrames(frames=frames, paths=paths)


class ManualGuideGenerator:
    def generate(self, frames: list[Frame], frame_paths: list[Path]) -> Guide:
        return Guide(title="Unused", steps=[GuideStep(text="Unused.", frameId=frames[0].frameId)])


class SharedImageAnalyzer:
    def analyze(self, frame_dir: Path, frame_count: int, job_id: str) -> AnalysisResult:
        return AnalysisResult(
            events=[
                AnalysisEvent(
                    eventId="event-1",
                    kind="attach",
                    startTimestampSeconds=0,
                    endTimestampSeconds=1,
                    affectedTrackIds=["part-a"],
                    beforeFrameId="frame-0000",
                    afterFrameId="frame-0001",
                    evidenceStrength=1,
                    evidence="visible",
                ),
                AnalysisEvent(
                    eventId="event-2",
                    kind="attach",
                    startTimestampSeconds=0,
                    endTimestampSeconds=1,
                    affectedTrackIds=["part-b"],
                    beforeFrameId="frame-0000",
                    afterFrameId="frame-0001",
                    evidenceStrength=1,
                    evidence="visible",
                ),
            ],
            tracks=[],
            analysis=AnalysisInfo(backend="test", modelVersion="test", configVersion="test"),
        )


class SharedImageGuideGenerator:
    def generate(self, frames: list[Frame], frame_paths: list[Path], events: list[AnalysisEvent] | None = None) -> Guide:
        assert events is not None
        return Guide(title="Build", steps=[GuideStep(text="Attach both visible pieces.", frameId="frame-0001")])


async def request(app, method: str, url: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def wait_for_status(app, job_id: str, status: str) -> dict:
    for _ in range(100):
        response = asyncio.run(request(app, "GET", f"/jobs/{job_id}"))
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] == status:
            return payload
        if payload["status"] == "failed":
            raise AssertionError(payload)
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {status}")


def make_app(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)
    repository = JobRepository(tmp_path)
    processor = JobProcessor(repository, settings, extractor=ManualExtractor(), generator=ManualGuideGenerator())
    return create_app(settings, processor=processor)


def test_manual_job_enters_pairing_with_server_owned_assembly_timestamps(tmp_path: Path) -> None:
    app = make_app(tmp_path)

    created = asyncio.run(request(
        app,
        "POST",
        "/jobs",
        data={"mode": "manual"},
        files={"video": ("build.mp4", b"video", "video/mp4")},
    ))
    assert created.status_code == 202

    job = wait_for_status(app, created.json()["jobId"], "pairing")
    assert job["mode"] == "manual"
    assert [frame["sourceIndex"] for frame in job["frames"]] == [0, 1, 2]
    assert [frame["assemblyTimeSeconds"] for frame in job["frames"]] == [1.0, 0.5, 0.0]
    assert job["manualPairs"] == []
    assert job["revision"] == 0


def test_manual_pair_save_is_revision_guarded_and_validates_assembly_order(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    created = asyncio.run(request(
        app,
        "POST",
        "/jobs",
        data={"mode": "manual"},
        files={"video": ("build.mp4", b"video", "video/mp4")},
    ))
    job_id = created.json()["jobId"]
    wait_for_status(app, job_id, "pairing")
    pairs = [{"pairId": "pair-1", "sequence": 1, "beforeFrameId": "frame-0001", "afterFrameId": "frame-0000"}]

    saved = asyncio.run(request(app, "PUT", f"/jobs/{job_id}/manual-pairs", json={"revision": 0, "pairs": pairs}))
    assert saved.status_code == 200
    assert saved.json()["revision"] == 1
    assert saved.json()["manualPairs"] == pairs

    stale = asyncio.run(request(app, "PUT", f"/jobs/{job_id}/manual-pairs", json={"revision": 0, "pairs": pairs}))
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "PAIR_REVISION_CONFLICT"


def test_pairing_job_blocks_a_second_upload_while_edits_remain_available(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    created = asyncio.run(request(
        app,
        "POST",
        "/jobs",
        data={"mode": "manual"},
        files={"video": ("first.mp4", b"video", "video/mp4")},
    ))
    job_id = created.json()["jobId"]
    wait_for_status(app, job_id, "pairing")

    second_upload = asyncio.run(request(
        app,
        "POST",
        "/jobs",
        files={"video": ("second.mp4", b"video", "video/mp4")},
    ))
    assert second_upload.status_code == 409
    assert second_upload.json()["error"]["code"] == "PROCESSING_BUSY"

    saved = asyncio.run(request(
        app,
        "PUT",
        f"/jobs/{job_id}/manual-pairs",
        json={
            "revision": 0,
            "pairs": [{"pairId": "pair-1", "sequence": 1, "beforeFrameId": "frame-0001", "afterFrameId": "frame-0000"}],
        },
    ))
    assert saved.status_code == 200


def test_restarted_server_keeps_a_pairing_job_admitted(tmp_path: Path) -> None:
    first_app = make_app(tmp_path)
    created = asyncio.run(request(
        first_app,
        "POST",
        "/jobs",
        data={"mode": "manual"},
        files={"video": ("first.mp4", b"video", "video/mp4")},
    ))
    wait_for_status(first_app, created.json()["jobId"], "pairing")

    restarted_app = make_app(tmp_path)
    second_upload = asyncio.run(request(
        restarted_app,
        "POST",
        "/jobs",
        files={"video": ("second.mp4", b"video", "video/mp4")},
    ))

    assert second_upload.status_code == 409
    assert second_upload.json()["error"]["code"] == "PROCESSING_BUSY"


def test_final_guide_merges_steps_that_share_an_evidence_image() -> None:
    guide = Guide(
        title="Build",
        steps=[
            GuideStep(text="Place the red brick.", frameId="frame-0004"),
            GuideStep(text="Add the blue plate.", frameId="frame-0004"),
            GuideStep(text="Press both pieces down.", frameId="frame-0005"),
        ],
    )

    compacted = deduplicate_guide_steps(guide)

    assert [step.frameId for step in compacted.steps] == ["frame-0004", "frame-0005"]
    assert compacted.steps[0].text == "Place the red brick. Add the blue plate."


def test_generation_accepts_one_compact_step_for_multiple_events_on_the_same_image(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    repository = JobRepository(tmp_path)
    processor = JobProcessor(
        repository,
        settings,
        extractor=ManualExtractor(),
        generator=SharedImageGuideGenerator(),
        analyzer=SharedImageAnalyzer(),
    )
    app = create_app(settings, processor=processor)

    created = asyncio.run(request(app, "POST", "/jobs", files={"video": ("build.mp4", b"video", "video/mp4")}))
    job = wait_for_status(app, created.json()["jobId"], "ready")

    assert job["guide"]["steps"] == [{"text": "Attach both visible pieces.", "frameId": "frame-0001", "uncertainty": None}]
