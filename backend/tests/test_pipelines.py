from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.action_timeline import EvidenceWindow, build_assembly_mapping, merge_timelines, reverse_reversible_actions, timeline_evidence, timeline_evidence_batches, timeline_windows, validate_timeline
from backend.app.config import Settings
from backend.app.errors import AppError
from backend.app.models import ActionTimeline, Frame, Guide, GuideStep, PartAnnotation, PointPrompt, TimelineAction, VerificationFinding
from backend.app.pipelines import GeminiVideoProvider, GPTActionProvider, _recovery_targets, _window_frames, create_pipeline_provider


def frames_and_paths(tmp_path: Path, count: int = 30) -> tuple[list[Frame], list[Path]]:
    frames = []
    paths = []
    for index in range(count):
        frame = Frame(frameId=f"frame-{index:04d}", timestampSeconds=float(index), imageUrl=f"/frame-{index:04d}")
        path = tmp_path / f"frame-{index:04d}.jpg"
        path.write_bytes(b"fake-jpeg")
        frames.append(frame)
        paths.append(path)
    return frames, paths


def action(action_id: str, start: float, end: float, *, moving: int = 1, receiving: int | None = 2, kind: str = "detach") -> TimelineAction:
    return TimelineAction(
        actionId=action_id,
        startTimestampSeconds=start,
        endTimestampSeconds=end,
        actionType=kind,
        movingPartId=moving,
        receivingPartId=receiving,
        beforeFrameId=f"frame-{int(start):04d}",
        afterFrameId=f"frame-{int(end):04d}",
        evidenceFrameIds=[f"frame-{int(start):04d}", f"frame-{int(end):04d}"],
        relationshipBefore="attached",
        relationshipAfter="separate",
        evidence="The named object separates from the receiving base.",
    )


def test_action_timeline_windows_merge_overlap_and_reverse_only_reversible_actions(tmp_path: Path) -> None:
    frames, paths = frames_and_paths(tmp_path, 45)
    windows = timeline_windows(frames, window_seconds=20, overlap_seconds=2)
    assert windows[0].startTimestampSeconds == 0
    assert windows[0].endTimestampSeconds == 20
    assert windows[1].startTimestampSeconds == 18
    assert windows[-1].endTimestampSeconds == 44

    merged = merge_timelines([
        ActionTimeline(actions=[action("window-a", 4, 6)]),
        ActionTimeline(actions=[action("window-b", 5, 7)]),
    ], duration_seconds=44)
    assert len(merged.actions) == 1
    assert merged.actions[0].startTimestampSeconds == 4
    assert merged.actions[0].endTimestampSeconds == 7
    reversed_actions = reverse_reversible_actions(merged)
    assert reversed_actions[0].actionType == "attach"
    assert reversed_actions[0].beforeFrameId == "frame-0007"

    annotations = [PartAnnotation(partId=1, name="moving", frameIndex=0, points=[PointPrompt(x=1, y=1)]), PartAnnotation(partId=2, name="base", frameIndex=0, points=[PointPrompt(x=2, y=2)])]
    validated = validate_timeline(merged, frames, annotations)
    evidence = timeline_evidence(frames, paths, validated, annotations, max_images=24)
    assert len(evidence.frames) <= 24
    assert evidence.context["actions"][0]["actionId"] == "action-0001"


def test_unresolved_intervals_and_recovery_windows_remain_bounded(tmp_path: Path) -> None:
    frames, paths = frames_and_paths(tmp_path, 30)
    unresolved = action("unknown-window", 4, 5, kind="unknown")
    merged = merge_timelines([ActionTimeline(unresolvedIntervals=[unresolved])], duration_seconds=29)
    assert [item.actionId for item in merged.unresolvedIntervals] == ["unresolved-0001"]
    assert merged.unresolvedIntervals[0].actionType == "unknown"
    targets = _recovery_targets(
        ActionTimeline(actions=[action("a", 4, 5).model_copy(update={"uncertainty": "occluded"}), action("b", 7, 8).model_copy(update={"uncertainty": "occluded"})]),
        max_windows=3,
        window_seconds=8,
        duration=29,
    )
    assert len(targets) == 2
    assert all(target.endTimestampSeconds - target.startTimestampSeconds <= 8 for target in targets)
    selected_frames, _ = _window_frames(frames, paths, EvidenceWindow("window-0001", 0, 20, tuple(range(0, 21))), [
        PartAnnotation(partId=1, name="moving", frameIndex=0, points=[PointPrompt(x=1, y=1)]),
        PartAnnotation(partId=2, name="base", frameIndex=29, points=[PointPrompt(x=2, y=2)]),
    ])
    assert len(selected_frames) <= 24
    assert {selected_frames[0].frameId, selected_frames[-1].frameId} == {"frame-0000", "frame-0029"}


def test_timeline_validation_resolves_requested_source_times_and_rejects_unsubmitted_images(tmp_path: Path) -> None:
    frames, _ = frames_and_paths(tmp_path, 6)
    annotations = [
        PartAnnotation(partId=1, name="moving", frameIndex=0, points=[PointPrompt(x=1, y=1)]),
        PartAnnotation(partId=2, name="base", frameIndex=0, points=[PointPrompt(x=2, y=2)]),
    ]
    candidate = action("a", 1.2, 3.2).model_copy(update={
        "beforeFrameId": None,
        "afterFrameId": None,
        "evidenceFrameIds": [],
        "beforeRequestedTimestampSeconds": 1.2,
        "afterRequestedTimestampSeconds": 3.2,
    })
    validated = validate_timeline(ActionTimeline(actions=[candidate]), frames, annotations, submitted_frame_ids={"frame-0001", "frame-0003"})
    resolved = validated.actions[0]
    assert (resolved.beforeFrameId, resolved.afterFrameId) == ("frame-0001", "frame-0003")
    assert resolved.evidenceRoles["before_state"] == ["frame-0001"]
    assert resolved.resolutionErrorSeconds == pytest.approx(0.2)
    with pytest.raises(AppError, match="not submitted"):
        validate_timeline(ActionTimeline(actions=[candidate.model_copy(update={"evidenceFrameIds": ["frame-0005"]})]), frames, annotations, submitted_frame_ids={"frame-0001", "frame-0003"})


def test_assembly_mapping_reverses_supported_disassembly_and_keeps_unknown_as_review() -> None:
    supported = action("detach", 2, 3)
    unknown = action("unknown", 5, 6, kind="unknown").model_copy(update={"relationshipBefore": "unknown", "relationshipAfter": "unknown"})
    timeline = ActionTimeline(actions=[supported, unknown])
    mapping = build_assembly_mapping(timeline)
    assert mapping[0].disposition == "instruction"
    assert mapping[0].actionType == "attach"
    assert mapping[0].relationshipBefore == "separate"
    assert mapping[0].relationshipAfter == "attached"
    assert mapping[1].disposition == "review"
    assert mapping[1].sourceActionIds == ["unknown"]


def test_merge_does_not_merge_simultaneous_different_moving_parts() -> None:
    first = action("first", 4, 6, moving=1)
    second = action("second", 4.5, 6.5, moving=3)
    merged = merge_timelines([ActionTimeline(actions=[first]), ActionTimeline(actions=[second])], duration_seconds=10)
    assert len(merged.actions) == 2


def test_recovery_prioritizes_missing_actions_and_ignores_supported_findings() -> None:
    supported = action("supported", 1, 2).model_copy(update={"uncertainty": "minor ambiguity"})
    missing = action("missing", 10, 11).model_copy(update={"uncertainty": "occluded"})
    findings = [
        VerificationFinding(findingId="f-supported", stepId=None, kind="supported", rationale="Confirmed", intervalId="supported"),
        VerificationFinding(findingId="f-missing", stepId=None, kind="missing", rationale="No step covers this interval", intervalId="missing"),
    ]
    targets = _recovery_targets(ActionTimeline(actions=[supported, missing]), max_windows=2, window_seconds=8, duration=20, findings=findings)
    assert targets[0].actionId == "missing"


def test_evidence_batches_preserve_late_actions_and_global_context(tmp_path: Path) -> None:
    frames, paths = frames_and_paths(tmp_path, 30)
    annotations = [PartAnnotation(partId=1, name="moving", frameIndex=0, points=[PointPrompt(x=1, y=1)])]
    actions = [action(f"a-{index}", index * 2, index * 2 + 1).model_copy(update={"beforeFrameId": f"frame-{index * 2:04d}", "afterFrameId": f"frame-{index * 2 + 1:04d}", "evidenceFrameIds": [f"frame-{index * 2:04d}", f"frame-{index * 2 + 1:04d}"]}) for index in range(10)]
    timeline = ActionTimeline(actions=actions)
    batches = timeline_evidence_batches(frames, paths, timeline, annotations, max_images=8)
    assert len(batches) > 1
    assert {action_id for batch in batches for action_id in batch.context["batch"]["actionIds"]} == {f"a-{index}" for index in range(10)}
    assert all(batch.context["batch"]["count"] == len(batches) for batch in batches)


def test_pipeline_selection_does_not_fallback_to_another_provider(tmp_path: Path) -> None:
    with pytest.raises(AppError, match="OPENAI_API_KEY"):
        create_pipeline_provider("gpt_targeted_sam2", Settings(data_dir=tmp_path))
    with pytest.raises(AppError, match="GEMINI_API_KEY"):
        create_pipeline_provider("gemini_video", Settings(data_dir=tmp_path))


def test_gemini_file_is_reused_and_deleted_without_logging_uri(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, gemini_api_key="test-key")
    provider = GeminiVideoProvider(settings)
    uploaded = SimpleNamespace(name="files/app-owned", uri="https://example.invalid/private", mime_type="video/mp4", state=SimpleNamespace(name="ACTIVE"))
    calls: list[tuple[str, object]] = []
    interaction_request: dict[str, object] = {}

    class Files:
        def upload(self, *, file: str):
            calls.append(("upload", file))
            return uploaded

        def get(self, *, name: str):
            calls.append(("get", name))
            return uploaded

        def delete(self, *, name: str):
            calls.append(("delete", name))

    class Interactions:
        def create(self, **kwargs: object):
            interaction_request.update(kwargs)
            calls.append(("interaction", kwargs.get("store")))
            return SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps({"actions": [], "unresolvedIntervals": []}))])

    provider._client = SimpleNamespace(files=Files(), interactions=Interactions())
    video = tmp_path / "canonical-silent.mp4"
    video.write_bytes(b"video")
    frames, paths = frames_and_paths(tmp_path, 2)
    annotations = [PartAnnotation(partId=1, name="object", frameIndex=0, points=[PointPrompt(x=1, y=1)])]
    provider.extract_timeline(frames, paths, annotations, video)
    provider.close()
    assert calls[0] == ("upload", str(video))
    assert ("interaction", False) in calls
    response_format = interaction_request["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "text"
    assert response_format["mime_type"] == "application/json"
    assert response_format["schema"]["required"] == ["actions", "unresolvedIntervals"]
    assert ("delete", "files/app-owned") in calls
    assert "https://example.invalid" not in (tmp_path / "gemini-cleanup.json").read_text() if (tmp_path / "gemini-cleanup.json").exists() else True
