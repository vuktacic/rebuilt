from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.errors import AppError
from backend.app.models import Frame, Guide, GuideStep, GuideVerification, PartAnnotation, PointPrompt, VerificationFinding
from backend.app.models import ActionTimeline, TimelineAction
from backend.app.processing import validate_guide
from backend.app.post_processing import EvidenceBuilder, normalized_point_to_pixels, validate_verification
from backend.app.storage import JobRepository


def test_normalized_suggestion_coordinates_are_converted_to_pixels() -> None:
    assert normalized_point_to_pixels(PointPrompt(x=0.25, y=0.75), 1000, 800) == PointPrompt(x=250, y=600)
    with pytest.raises(AppError, match="outside normalized"):
        normalized_point_to_pixels(PointPrompt(x=2, y=0.5), 1000, 800)


def test_evidence_builder_deduplicates_frames_and_preserves_tracking_provenance(tmp_path: Path) -> None:
    frames = [
        Frame(frameId="frame-0001", timestampSeconds=0, imageUrl="/one"),
        Frame(frameId="frame-0002", timestampSeconds=1, imageUrl="/two"),
        Frame(frameId="frame-0003", timestampSeconds=2, imageUrl="/three"),
    ]
    paths = [tmp_path / f"{index}.jpg" for index in range(3)]
    for path in paths:
        path.write_bytes(b"fake")
    from backend.app.models import AnalysisEvent, PartTrack, TrackObservation

    event = AnalysisEvent(
        eventId="event-1", kind="attach", startTimestampSeconds=0.5, endTimestampSeconds=1.5,
        affectedTrackIds=["part:1"], beforeFrameId="frame-0001", afterFrameId="frame-0003",
        evidenceStrength=0.8, evidence="piece moves onto base",
    )
    bundle = EvidenceBuilder().build(
        frames, paths, [event],
        [PartAnnotation(partId=1, name="red brick", frameIndex=0, points=[PointPrompt(x=2, y=3)], labels=[1])],
        [PartTrack(partId=1, name="red brick", observations=[TrackObservation(frameIndex=0, visible=False)])],
    )
    assert [frame.frameId for frame in bundle.frames] == ["frame-0001", "frame-0002", "frame-0003"]
    assert bundle.context["events"][0]["evidenceFrames"][1]["role"] == "transition"
    assert bundle.context["tracking"][0]["observations"][0]["provenance"] == "missing"


def test_verification_rejects_unknown_frame_and_duplicate_step_coverage(tmp_path: Path) -> None:
    frames = [Frame(frameId="frame-0001", timestampSeconds=0, imageUrl="/one")]
    path = tmp_path / "one.jpg"
    path.write_bytes(b"fake")
    from backend.app.post_processing import EvidenceBundle

    evidence = EvidenceBundle(context={}, frames=frames, paths=[path])
    guide = Guide(title="Build", steps=[GuideStep(stepId="step-1", text="Place it.", frameId="frame-0001")])
    bad = GuideVerification(
        status="completed", revision=1, coverage=1,
        findings=[VerificationFinding(findingId="f-1", stepId="step-1", kind="supported", rationale="Visible", evidenceFrameIds=["missing"])],
    )
    with pytest.raises(AppError, match="unavailable frame"):
        validate_verification(bad, guide, evidence)


def test_legacy_job_get_assigns_missing_annotation_and_step_ids(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path)
    repository.create("legacy", "build.mp4")
    metadata = tmp_path / "jobs" / "legacy" / "metadata.json"
    raw = json.loads(metadata.read_text())
    raw["annotations"] = [{"name": "brick", "frameIndex": 0, "points": [{"x": 1, "y": 1}], "labels": [1], "box": None}]
    raw["guide"] = {"title": "Build", "steps": [{"text": "Place it.", "frameId": "frame-0001", "uncertainty": None}]}
    metadata.write_text(json.dumps(raw))
    job = repository.get("legacy")
    assert job is not None
    assert job.annotations[0].partId == 1
    assert job.guide is not None and job.guide.steps[0].stepId == "step-0001"


def test_action_first_guide_requires_source_provenance_but_allows_empty_review(tmp_path: Path) -> None:
    frames = [Frame(frameId="frame-0001", timestampSeconds=0, imageUrl="/one"), Frame(frameId="frame-0002", timestampSeconds=1, imageUrl="/two")]
    action = TimelineAction(
        actionId="action-1", startTimestampSeconds=0, endTimestampSeconds=1, actionType="unknown",
        beforeFrameId="frame-0001", afterFrameId="frame-0002", evidenceFrameIds=["frame-0001", "frame-0002"],
        evidence="The interval is obscured.",
    )
    with pytest.raises(AppError, match="source action provenance"):
        validate_guide(Guide(title="Build", steps=[GuideStep(stepId="step-1", text="Do something.", frameId="frame-0001")]), frames, action_timeline=ActionTimeline(actions=[action]), require_action_coverage=True)
    reviewed = validate_guide(Guide(title="Build", steps=[GuideStep(stepId="step-1", text="Review this interval.", frameId="frame-0001", kind="review")]), frames, action_timeline=ActionTimeline(), require_action_coverage=True)
    assert reviewed.steps[0].kind == "review"
