from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import backend.app.vision as vision_module
from backend.app.config import Settings
from backend.app.models import AnalysisEvent, AnalysisInfo, Frame, Guide, GuideStep, PartAnnotation, PartTrack, PointPrompt, TrackObservation, TrackSummary
from backend.app.processing import ExtractedFrames, JobProcessor
from backend.app.storage import JobRepository
from backend.app.vision import (
    AnalysisResult,
    Detection,
    EventDetector,
    MlxSam3VisionAnalyzer,
    MlxSam2BackwardVisionAnalyzer,
    PersistentTrackAssociator,
    Sam2BackwardVisionAnalyzer,
    Sam3VisionAnalyzer,
    StableStateChangeDetector,
    WindowTrackStitcher,
    VisionWeightsMissing,
)


def test_sam2_attachment_requires_sustained_overlap_and_reverses_event_order() -> None:
    def observation(frame: int, bbox: tuple[float, float, float, float]) -> TrackObservation:
        return TrackObservation(frameIndex=frame, bbox=bbox, centroid=(bbox[0], bbox[1]), visible=True)

    moving = [observation(frame, (0, 0, 10, 10) if frame < 3 else (30, 0, 10, 10)) for frame in range(7)]
    anchor = [observation(frame, (0, 0, 10, 10)) for frame in range(7)]
    anchor_track = PartTrack(partId=2, name="base plate", observations=anchor)
    separate, attached, target_part_id = Sam2BackwardVisionAnalyzer._attachment_range(moving, [anchor_track])

    assert (separate, attached) == (3, 2)
    assert target_part_id == 2
    analyzer = Sam2BackwardVisionAnalyzer(Settings(data_dir=Path(".data-test"), analysis_fps=2))
    moving_track = PartTrack(
        partId=1,
        name="side panel",
        observations=moving,
        attachmentStartFrame=separate,
        attachmentEndFrame=attached,
    )
    anchor_track.attachmentStartFrame = separate
    anchor_track.attachmentEndFrame = attached
    events = analyzer._events_from_part_tracks([moving_track, anchor_track], {1: 2, 2: 1})
    assert [(event.beforeFrameId, event.afterFrameId) for event in events] == [("frame-0003", "frame-0002")]
    assert events[0].kind == "attach"
    assert events[0].affectedTrackIds == ["part:1", "part:2"]
    assert events[0].evidence == "side panel moves into sustained substantial overlap with base plate."
    assert events[0].startTimestampSeconds < events[0].endTimestampSeconds


def test_sam2_attachment_rejects_transient_overlap() -> None:
    def observation(frame: int, bbox: tuple[float, float, float, float]) -> TrackObservation:
        return TrackObservation(frameIndex=frame, bbox=bbox, centroid=(bbox[0], bbox[1]), visible=True)

    moving = [observation(frame, (0, 0, 10, 10) if frame == 1 else (30, 0, 10, 10)) for frame in range(7)]
    anchor = [observation(frame, (0, 0, 10, 10)) for frame in range(7)]
    anchor_track = PartTrack(partId=2, name="base plate", observations=anchor)
    assert Sam2BackwardVisionAnalyzer._attachment_range(moving, [anchor_track]) == (None, None, None)


def test_sam2_missing_piece_becomes_an_llm_recovery_event() -> None:
    missing = [
        TrackObservation(frameIndex=frame, visible=frame >= 2, bbox=(30, 0, 10, 10) if frame >= 2 else None)
        for frame in range(6)
    ]
    anchor = [
        TrackObservation(frameIndex=frame, visible=True, bbox=(0, 0, 20, 20), centroid=(10, 10))
        for frame in range(6)
    ]
    analyzer = Sam2BackwardVisionAnalyzer(Settings(data_dir=Path(".data-test"), analysis_fps=2))

    events = analyzer._events_from_part_tracks([
        PartTrack(partId=1, name="red roof", observations=missing),
        PartTrack(partId=2, name="blue base", observations=anchor),
    ])

    assert len(events) == 1
    assert events[0].kind == "uncertain_change"
    assert events[0].beforeFrameId == "frame-0002"
    assert events[0].afterFrameId == "frame-0001"
    assert "red roof is visible while separate" in events[0].evidence
    assert "blue base" in events[0].evidence
    assert "tracker lost red roof" in (events[0].uncertainty or "")


def test_sam2_missing_piece_at_recording_boundary_is_preserved_as_uncertain() -> None:
    observations = [
        TrackObservation(frameIndex=0, visible=True, bbox=(0, 0, 10, 10), centroid=(5, 5), provenance="measured", timestampSeconds=0),
        TrackObservation(frameIndex=1, visible=False, provenance="missing", timestampSeconds=0.5),
        TrackObservation(frameIndex=2, visible=False, provenance="missing", timestampSeconds=1),
    ]

    assert Sam2BackwardVisionAnalyzer._missing_transitions(observations) == [(0, 0)]


class SamplingSam2Provider:
    def __init__(self) -> None:
        self.loaded_frame_names: list[str] = []
        self.prompt_frames: list[int] = []
        self.frame_dir: Path | None = None

    def init_state(self, frame_dir: str) -> object:
        self.frame_dir = Path(frame_dir)
        self.loaded_frame_names = sorted(path.name for path in Path(frame_dir).glob("*.jpg"))
        return object()

    def add_new_points_or_box(self, state: object, *, frame_idx: int, **_: object) -> None:
        self.prompt_frames.append(frame_idx)

    def propagate_in_video(self, state: object, *, start_frame_idx: int, reverse: bool):
        assert reverse is True
        assert self.frame_dir is not None and self.frame_dir.is_dir()
        assert len(list(self.frame_dir.glob("*.jpg"))) == start_frame_idx + 1
        for frame_idx in range(start_frame_idx, -1, -1):
            yield frame_idx, [1], [[[[-1.0, -1.0], [-1.0, -1.0]]]]


def test_sam2_fast_profile_tracks_sampled_frames_and_restores_source_frame_indexes(tmp_path: Path) -> None:
    provider = SamplingSam2Provider()
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(5):
        (frame_dir / f"frame-{index:04d}.jpg").write_bytes(b"fake-jpeg")
    analyzer = Sam2BackwardVisionAnalyzer(
        Settings(data_dir=tmp_path, analysis_fps=2, sam2_frame_stride=2),
        provider=provider,
    )

    result = analyzer.analyze_annotated(
        frame_dir,
        frame_count=5,
        job_id="sampled",
        annotations=[PartAnnotation(partId=7, name="panel", frameIndex=4, points=[PointPrompt(x=1, y=1)])],
    )

    assert provider.loaded_frame_names == ["000000.jpg", "000001.jpg", "000002.jpg"]
    assert provider.prompt_frames == [2]
    assert [item.frameIndex for item in result.part_tracks[0].observations] == [0, 1, 2, 3, 4]
    assert result.analysis.metrics["sampled_frames"] == 3.0
    assert result.part_tracks[0].partId == 7


def test_sam2_analyzer_reuses_the_loaded_provider_between_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SamplingSam2Provider()
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "frame-0000.jpg").write_bytes(b"fake-jpeg")
    analyzer = Sam2BackwardVisionAnalyzer(Settings(data_dir=tmp_path, sam2_frame_stride=1))
    loads = 0

    def load_provider() -> SamplingSam2Provider:
        nonlocal loads
        loads += 1
        return provider

    monkeypatch.setattr(analyzer, "_load_provider", load_provider)
    annotation = [PartAnnotation(name="panel", frameIndex=0, points=[PointPrompt(x=1, y=1)])]

    analyzer.analyze_annotated(frame_dir, 1, "first", annotation)
    analyzer.analyze_annotated(frame_dir, 1, "second", annotation)

    assert loads == 1


class MlxSamplingSam2Provider(SamplingSam2Provider):
    def __init__(self) -> None:
        super().__init__()
        self.init_options: dict[str, object] = {}

    def init_state(self, frame_dir: str, **kwargs: object) -> object:
        self.init_options = kwargs
        return super().init_state(frame_dir)


def test_mlx_sam2_uses_batched_feature_precompute_for_reverse_tracking(tmp_path: Path) -> None:
    provider = MlxSamplingSam2Provider()
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "frame-0000.jpg").write_bytes(b"fake-jpeg")
    analyzer = MlxSam2BackwardVisionAnalyzer(
        Settings(data_dir=tmp_path, sam2_mlx_precompute_features=True, sam2_mlx_feature_batch_size=6),
        provider=provider,
    )

    result = analyzer.analyze_annotated(
        frame_dir,
        frame_count=1,
        job_id="mlx",
        annotations=[PartAnnotation(name="panel", frameIndex=0, points=[PointPrompt(x=1, y=1)])],
    )

    assert provider.init_options == {"precompute_image_features": True, "feature_batch_size": 6}
    assert result.analysis.backend == "sam2.1-mlx-backward"


def detection_series(*, hand_occluded: bool = False) -> list[Detection]:
    observations: list[Detection] = []
    for frame_index in range(20):
        attached = 5 <= frame_index < 15
        x_position = 30 if attached else 10
        observations.append(
            Detection(
                frame_index=frame_index,
                predictor_id="piece-1",
                concept="LEGO piece",
                score=0.95,
                bbox=(x_position, 10, 10, 10),
                group_id="assembly-1" if attached else None,
                occluded_by_hand=hand_occluded and 5 <= frame_index <= 9,
            )
        )
    return observations


def test_event_detector_requires_stable_states_and_supports_both_directions() -> None:
    events, tracks = EventDetector(fps=10).detect(detection_series(), frame_count=20)

    assert [event.kind for event in events] == ["attach", "detach"]
    assert events[0].beforeFrameId == "frame-0000"
    assert events[0].afterFrameId == "frame-0009"
    assert events[1].beforeFrameId == "frame-0010"
    assert events[1].afterFrameId == "frame-0019"
    assert tracks[0].trackId == "LEGO piece:piece-1"


def test_event_detector_marks_hand_occlusion_uncertain() -> None:
    events, _ = EventDetector(fps=10).detect(detection_series(hand_occluded=True), frame_count=20)

    assert events[0].kind == "uncertain_change"
    assert events[0].uncertainty == "vision was occluded during the transition"


def test_stable_state_detector_emits_one_change_per_hand_clear_layout() -> None:
    detections: list[Detection] = []
    for frame_index in (0, 1):
        detections.extend([
            Detection(frame_index, f"base-{frame_index}", "LEGO piece", 0.9, (10, 10, 10, 10)),
            Detection(frame_index, f"top-{frame_index}", "LEGO piece", 0.9, (20, 10, 10, 10)),
        ])
    for frame_index in (2, 3, 6, 7):
        detections.append(Detection(frame_index, f"hand-{frame_index}", "hand", 0.9, (0, 0, 100, 100)))
    for frame_index in (4, 5):
        detections.extend([
            Detection(frame_index, f"base-{frame_index}", "LEGO piece", 0.9, (10, 10, 10, 10)),
            Detection(frame_index, f"top-{frame_index}", "LEGO piece", 0.9, (20, 10, 10, 10)),
            Detection(frame_index, f"loose-{frame_index}", "LEGO piece", 0.9, (60, 10, 10, 10)),
        ])
    for frame_index in (8, 9):
        detections.extend([
            Detection(frame_index, f"base-{frame_index}", "LEGO piece", 0.9, (10, 10, 10, 10)),
            Detection(frame_index, f"top-{frame_index}", "LEGO piece", 0.9, (20, 10, 10, 10)),
            Detection(frame_index, f"loose-a-{frame_index}", "LEGO piece", 0.9, (60, 10, 10, 10)),
            Detection(frame_index, f"loose-b-{frame_index}", "LEGO piece", 0.9, (100, 10, 10, 10)),
        ])

    events = StableStateChangeDetector(fps=2).detect(detections, frame_count=10)

    assert [event.kind for event in events] == ["detach", "detach"]
    assert [(event.beforeFrameId, event.afterFrameId) for event in events] == [
        ("frame-0000", "frame-0004"),
        ("frame-0004", "frame-0008"),
    ]


def test_overlapping_reversals_become_one_uncertain_manipulation() -> None:
    observations = []
    for frame_index in range(15):
        attached = 5 <= frame_index < 10
        observations.append(Detection(
            frame_index,
            "piece-1",
            "LEGO piece",
            0.95,
            (30 if attached else 10, 10, 10, 10),
            group_id="assembly-1" if attached else None,
        ))

    events, _ = EventDetector(fps=10).detect(observations, frame_count=15)

    assert len(events) == 1
    assert events[0].kind == "uncertain_change"
    assert "without a separating stable pause" in (events[0].uncertainty or "")


def test_touching_without_motion_is_not_confident_attachment() -> None:
    detections = [
        Detection(index, "piece", "LEGO piece", 0.9, (10, 10, 10, 10), group_id="assembly" if index >= 5 else None)
        for index in range(10)
    ]

    events, _ = EventDetector(fps=10).detect(detections, frame_count=10)

    assert events[0].kind == "uncertain_change"
    assert "relative motion" in (events[0].uncertainty or "")


def test_event_detector_bridges_short_lost_vision_gap() -> None:
    detections = [
        Detection(
            frame_index=index,
            predictor_id="piece-1",
            concept="LEGO piece",
            score=0.95,
            bbox=(30 if index >= 8 else 10, 10, 10, 10),
            group_id="assembly-1" if index >= 8 else None,
        )
        for index in range(15)
        if index not in {5, 6, 7}
    ]
    events, tracks = EventDetector(fps=10, max_gap_seconds=1.5).detect(detections, frame_count=15)

    assert events[0].kind == "uncertain_change"
    assert events[0].beforeFrameId == "frame-0000"
    assert tracks[0].visibility == "occluded"


def test_associator_keeps_id_after_detector_id_churn() -> None:
    detections = [
        Detection(0, "sam-1", "LEGO piece", 0.9, (100, 100, 20, 20)),
        Detection(1, "sam-1", "LEGO piece", 0.9, (102, 100, 20, 20)),
        Detection(4, "sam-new", "LEGO piece", 0.9, (106, 100, 20, 20)),
    ]

    associated = PersistentTrackAssociator(fps=10, max_gap_seconds=1.5).associate(detections)

    assert [item.predictor_id for item in associated] == ["track-0001", "track-0001", "track-0001"]


def test_associator_never_assigns_two_detections_in_one_frame_to_one_track() -> None:
    detections = [
        Detection(0, "piece-a", "LEGO piece", 0.9, (0, 0, 20, 20)),
        Detection(0, "piece-b", "LEGO piece", 0.9, (100, 0, 20, 20)),
        Detection(1, "piece-c", "LEGO piece", 0.9, (0, 0, 20, 20)),
        Detection(2, "piece-a", "LEGO piece", 0.9, (0, 0, 20, 20)),
        Detection(2, "piece-c", "LEGO piece", 0.9, (100, 0, 20, 20)),
    ]

    associated = PersistentTrackAssociator(fps=10, max_gap_seconds=1.5).associate(detections)
    frame_two = [item.predictor_id for item in associated if item.frame_index == 2]

    assert len(frame_two) == len(set(frame_two))


def test_window_stitcher_uses_overlap_when_predictor_ids_swap() -> None:
    stitcher = WindowTrackStitcher()
    first = [
        Detection(frame, predictor_id, "LEGO piece", 0.9, (x, 10, 20, 20))
        for frame in (0, 1)
        for predictor_id, x in (("1", 10), ("2", 100))
    ]
    second = [
        Detection(frame, predictor_id, "LEGO piece", 0.9, (x, 10, 20, 20))
        for frame in (1, 2)
        for predictor_id, x in (("1", 100), ("2", 10))
    ]

    stitcher.add(first)
    stitched = stitcher.add(second)

    ids_by_position = {
        (item.frame_index, item.bbox[0]): item.predictor_id
        for item in stitched
    }
    assert ids_by_position[(0, 10)] == ids_by_position[(2, 10)]
    assert ids_by_position[(0, 100)] == ids_by_position[(2, 100)]
    assert len(stitched) == 6


def test_long_gap_is_reported_as_lost_without_reusing_unknown_membership() -> None:
    detections = [
        Detection(0, "piece", "LEGO piece", 0.9, (10, 10, 10, 10), group_id="assembly"),
        Detection(30, "piece", "LEGO piece", 0.9, (10, 10, 10, 10), group_id=None),
    ]

    events, tracks = EventDetector(fps=10, max_gap_seconds=1.5).detect(detections, frame_count=31)

    assert events == []
    assert tracks[0].visibility == "lost"
    assert tracks[0].membership == "separate"


class ResettingPromptPredictor:
    def __init__(self) -> None:
        self.started = 0
        self.closed: list[str] = []
        self.prompts: list[tuple[str, str]] = []
        self.masks: list[tuple[str, int]] = []

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        if request["type"] == "start_session":
            self.started += 1
            return {"session_id": f"session-{self.started}"}
        if request["type"] == "add_prompt":
            session_id = str(request["session_id"])
            prompt = str(request["text"])
            self.prompts.append((session_id, prompt))
            return {"outputs": {"out_obj_ids": [1], "out_binary_masks": [[[True]]], "out_probs": [0.9]}}
        if request["type"] == "add_mask":
            self.masks.append((str(request["session_id"]), int(request["obj_id"])))
            return {"outputs": {"out_obj_ids": [request["obj_id"]]}}
        if request["type"] == "close_session":
            self.closed.append(str(request["session_id"]))
            return {"is_success": True}
        raise AssertionError(request)

    def handle_stream_request(self, request: dict[str, object]):
        assert request["propagation_direction"] == "forward"
        yield {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": [1, 2],
                "out_boxes_xywh": [[10, 20, 30, 40], [50, 20, 30, 40]],
                "out_probs": [0.9, 0.8],
            },
        }


def test_sam3_runs_each_concept_in_an_independent_session(tmp_path: Path) -> None:
    predictor = ResettingPromptPredictor()
    analyzer = Sam3VisionAnalyzer(Settings(data_dir=tmp_path, vision_worker=False))
    window_dir = tmp_path / "window"
    window_dir.mkdir()
    (window_dir / "000000.jpg").write_bytes(b"fake-jpeg")

    detections = analyzer._run_session(predictor, window_dir, frame_count=1, frame_offset=0)

    assert predictor.started == 3
    assert predictor.closed == ["session-1", "session-2", "session-3"]
    assert predictor.prompts == [
        ("session-1", "LEGO piece"),
        ("session-2", "hand"),
    ]
    assert predictor.masks == [("session-3", 1), ("session-3", 2)]
    assert [detection.concept for detection in detections] == ["LEGO piece", "hand"]


def test_missing_explicit_checkpoint_is_reported_as_weights_error(tmp_path: Path) -> None:
    analyzer = Sam3VisionAnalyzer(Settings(
        data_dir=tmp_path,
        sam3_checkpoint=tmp_path / "missing.pt",
        vision_worker=False,
    ))

    with pytest.raises(VisionWeightsMissing, match="does not exist"):
        analyzer._load_predictor()


def test_worker_retries_transient_exit_within_processing_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[float] = []
    payload = {
        "events": [],
        "tracks": [],
        "analysis": {"backend": "test", "modelVersion": "test", "configVersion": "test"},
    }

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        attempts.append(float(kwargs["timeout"]))
        if len(attempts) == 1:
            return SimpleNamespace(returncode=1, stderr="transient worker failure", stdout="")
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(payload))

    monkeypatch.setattr(vision_module.subprocess, "run", fake_run)
    analyzer = Sam3VisionAnalyzer(Settings(data_dir=tmp_path, processing_timeout_seconds=10))

    result = analyzer._analyze_worker(tmp_path, frame_count=0, job_id="retry")

    assert result.analysis.backend == "test"
    assert len(attempts) == 2
    assert attempts[1] < attempts[0]


def test_analysis_windows_resolve_relative_frame_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    frame_dir = Path("relative-frames")
    frame_dir.mkdir()
    source = frame_dir / "frame-0000.jpg"
    source.write_bytes(b"fake-jpeg")
    analyzer = Sam3VisionAnalyzer(Settings(data_dir=Path("relative-data"), vision_worker=False))
    monkeypatch.setattr(analyzer, "_load_predictor", lambda: object())

    def inspect_window(predictor: object, window_dir: Path, frame_count: int, frame_offset: int) -> list[Detection]:
        linked_frame = window_dir / "000000.jpg"
        assert linked_frame.is_file()
        assert linked_frame.resolve() == source.resolve()
        return []

    monkeypatch.setattr(analyzer, "_run_session", inspect_window)

    result = analyzer._analyze_loaded(frame_dir, frame_count=1, job_id="relative-path")

    assert result.analysis.metrics["frames"] == 1.0


def test_piece_contact_can_form_cluster_without_assembly_prompt(tmp_path: Path) -> None:
    detections = []
    for frame_index in range(12):
        detections.append(Detection(frame_index, "anchor", "LEGO piece", 0.9, (30, 10, 10, 10)))
        detections.append(Detection(
            frame_index,
            "moving",
            "LEGO piece",
            0.9,
            (25 if frame_index >= 6 else 5, 10, 10, 10),
        ))
    analyzer = Sam3VisionAnalyzer(Settings(data_dir=tmp_path, vision_worker=False))

    events, _ = analyzer._events(detections, frame_count=12)

    assert any(event.kind == "attach" for event in events)


class MlxProviderStub:
    def track(self, frame_paths: list[Path], prompts: tuple[str, ...]) -> list[list[dict[str, object]]]:
        assert prompts == ("LEGO piece", "hand")
        return [
            [{"object_id": "piece-7", "label": "LEGO piece", "score": 0.8, "bbox": [10, 20, 30, 40], "mask": [[1]]}],
            [{"object_id": "hand-1", "label": "hand", "score": 0.7, "bbox": [1, 2, 3, 4], "mask": [[1]]}],
        ]


class CapturingMlxProvider:
    def __init__(self) -> None:
        self.frame_names: list[str] = []

    def track(self, frame_paths: list[Path], prompts: tuple[str, ...]) -> list[list[dict[str, object]]]:
        assert prompts == ("LEGO piece", "hand")
        self.frame_names = [path.name for path in frame_paths]
        return [
            [{"object_id": f"piece-{index}", "label": "LEGO piece", "score": 0.8, "bbox": [10, 20, 30, 40], "mask": [[1]]}]
            for index, _ in enumerate(frame_paths)
        ]


def test_mlx_fast_profile_samples_frames_without_losing_source_indexes(tmp_path: Path) -> None:
    provider = CapturingMlxProvider()
    analyzer = MlxSam3VisionAnalyzer(
        Settings(data_dir=tmp_path, mlx_frame_stride=2),
        provider=provider,
    )
    paths = []
    for index in range(4):
        path = tmp_path / f"frame-{index:04d}.jpg"
        path.write_bytes(b"fake-jpeg")
        paths.append(path)

    detections = analyzer._run_mlx_window(provider, paths, frame_offset=10)

    assert provider.frame_names == ["frame-0000.jpg", "frame-0002.jpg"]
    assert [item.frame_index for item in detections] == [10, 12]


def test_mlx_adapter_normalizes_raw_provider_output(tmp_path: Path) -> None:
    analyzer = MlxSam3VisionAnalyzer(Settings(data_dir=tmp_path, mlx_frame_stride=1), provider=MlxProviderStub())
    paths = []
    for index in range(2):
        path = tmp_path / f"frame-{index:04d}.jpg"
        path.write_bytes(b"fake-jpeg")
        paths.append(path)

    detections = analyzer._run_mlx_window(MlxProviderStub(), paths, frame_offset=10)

    assert [(item.frame_index, item.predictor_id, item.concept, item.bbox) for item in detections] == [
        (10, "piece-7", "LEGO piece", (10.0, 20.0, 20.0, 20.0)),
        (11, "hand-1", "hand", (1.0, 2.0, 2.0, 2.0)),
    ]


def test_mlx_xyxy_boxes_are_normalized_to_internal_xywh(tmp_path: Path) -> None:
    analyzer = MlxSam3VisionAnalyzer(Settings(data_dir=tmp_path, mlx_frame_stride=1), provider=MlxProviderStub())

    detection = analyzer._parse_mlx_output(
        {"object_id": "piece-7", "label": "LEGO piece", "score": 0.8, "bbox": [10, 20, 30, 40]},
        frame_index=0,
    )

    assert detection is not None
    assert detection.bbox == (10.0, 20.0, 20.0, 20.0)


class AnalysisExtractor:
    def extract(self, input_path: Path, output_dir: Path, job_id: str) -> ExtractedFrames:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        frames = []
        for index in range(2):
            path = output_dir / f"frame-{index:04d}.jpg"
            path.write_bytes(b"fake-jpeg")
            paths.append(path)
            frames.append(Frame(frameId=f"frame-{index:04d}", timestampSeconds=float(index), imageUrl=f"/jobs/{job_id}/frames/frame-{index:04d}"))
        return ExtractedFrames(frames=frames, paths=paths)


class AnalysisStub:
    def analyze(self, frame_dir: Path, frame_count: int, job_id: str) -> AnalysisResult:
        return AnalysisResult(
            events=[AnalysisEvent(
                eventId="event-0001", kind="attach", startTimestampSeconds=0, endTimestampSeconds=1,
                affectedTrackIds=["piece:1"], beforeFrameId="frame-0000", afterFrameId="frame-0001",
                evidenceStrength=0.9, uncertainty=None, evidence="stable attachment",
            )],
            tracks=[TrackSummary(trackId="piece:1", concept="LEGO piece", firstTimestampSeconds=0, lastTimestampSeconds=1, membership="attached")],
            analysis=AnalysisInfo(backend="stub", modelVersion="test", configVersion="test"),
        )


class PairGenerator:
    def __init__(self) -> None:
        self.frame_ids: list[str] = []

    def generate(self, frames: list[Frame], frame_paths: list[Path]) -> Guide:
        self.frame_ids = [frame.frameId for frame in frames]
        return Guide(title="Build", steps=[GuideStep(text="Attach the piece.", frameId=frames[-1].frameId)])


def test_processor_persists_events_and_sends_only_event_pair(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path)
    generator = PairGenerator()
    processor = JobProcessor(
        repository,
        Settings(data_dir=tmp_path),
        extractor=AnalysisExtractor(),
        analyzer=AnalysisStub(),
        generator=generator,
    )
    repository.create("job", "build.mp4")
    repository.input_path("job", "build.mp4").write_bytes(b"video")

    processor.process("job")

    job = repository.get("job")
    assert job is not None
    assert job.status == "ready"
    assert job.events[0].kind == "attach"
    assert generator.frame_ids == ["frame-0000", "frame-0001"]
