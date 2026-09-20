from __future__ import annotations

from pathlib import Path

from backend.app.config import Settings
from backend.app.guide_frame_presets import GUIDE_FRAME_PRESETS, off_grid_snapshot_timestamps, preset_for, select_preset_frames, snapshot_frame_id
from backend.app.models import AnnotationSuggestion, Frame, Guide, GuideStep, PointPrompt
from backend.app.processing import ExtractedFrames, FFmpegExtractor, JobProcessor
from backend.app.storage import JobRepository


def _frames(count: int = 200) -> tuple[list[Frame], list[Path]]:
    frames = [Frame(frameId=f"frame-{index:04d}", timestampSeconds=index / 5, imageUrl=f"/frames/{index}") for index in range(count)]
    paths = [Path(f"/tmp/frame-{index:04d}.jpg") for index in range(count)]
    return frames, paths


def test_presets_are_limited_to_the_recorded_sources_and_timestamp_ranges() -> None:
    assert set(GUIDE_FRAME_PRESETS) == {"IMG_3320.MOV", "IMG_3325.MOV", "IMG_3327.MOV", "IMG_3330.MOV"}
    assert preset_for("/any/path/img_3325.mov") is GUIDE_FRAME_PRESETS["IMG_3325.MOV"]
    assert preset_for("unrelated.mov") is None
    for preset in GUIDE_FRAME_PRESETS.values():
        assert preset.minimum_timestamp_seconds == 0
        assert preset.timestamps_seconds[-1] <= preset.maximum_timestamp_seconds


def test_preset_selection_requires_exact_timestamps_and_reports_off_grid_positions() -> None:
    frames, paths = _frames()

    assert off_grid_snapshot_timestamps("IMG_3325.MOV", frames) == (5.16, 11.67, 14.5)
    assert select_preset_frames("IMG_3325.MOV", frames, paths) is None

    snapshots = [
        Frame(frameId=snapshot_frame_id(timestamp), timestampSeconds=timestamp, imageUrl=f"/snapshot/{timestamp}")
        for timestamp in (5.16, 11.67, 14.5)
    ]
    combined = sorted(
        zip(frames + snapshots, paths + [Path(f"/tmp/{frame.frameId}.jpg") for frame in snapshots], strict=True),
        key=lambda item: item[0].timestampSeconds,
    )
    selected = select_preset_frames("IMG_3325.MOV", [frame for frame, _ in combined], [path for _, path in combined])

    assert selected is not None
    selected_frames, selected_paths = selected
    assert [frame.timestampSeconds for frame in selected_frames] == [0.0, 5.16, 9.0, 11.67, 14.5, 19.8]
    assert selected_paths[3] == Path("/tmp/snapshot-00011670.jpg")
    assert select_preset_frames("IMG_3320.MOV", frames[:60], paths[:60]) is None


class CapturingGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], object]] = []

    def generate(self, frames: list[Frame], paths: list[Path], events: object = None) -> Guide:
        self.calls.append(([frame.frameId for frame in frames], events))
        return Guide(title="Preset guide", steps=[GuideStep(text="Use the first settled state.", frameId=frames[0].frameId)])


def test_known_source_replaces_event_frames_when_generating_the_guide(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path)
    repository.create("job", "IMG_3320.MOV")
    frames, paths = _frames()
    repository.update("job", frames=frames)
    generator = CapturingGenerator()
    processor = JobProcessor(repository, Settings(data_dir=tmp_path), generator=generator)

    processor._finish_guide("job", frames, paths, [], repository.source_filename("job"))

    job = repository.get("job")
    assert generator.calls == [(["frame-0000", "frame-0024", "frame-0050", "frame-0055", "frame-0070"], None)]
    assert job is not None and job.status == "ready"
    assert repository.source_filename("job") == "IMG_3320.MOV"


class QueueingExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, tuple[float, ...]]] = []

    def extract_snapshots(self, input_path: Path, output_dir: Path, job_id: str, timestamps_seconds: tuple[float, ...]) -> ExtractedFrames:
        self.calls.append((input_path, timestamps_seconds))
        frames: list[Frame] = []
        paths: list[Path] = []
        for timestamp in timestamps_seconds:
            frame_id = snapshot_frame_id(timestamp)
            path = output_dir / f"{frame_id}.jpg"
            path.write_bytes(b"snapshot")
            frames.append(Frame(frameId=frame_id, timestampSeconds=timestamp, imageUrl=f"/jobs/{job_id}/frames/{frame_id}"))
            paths.append(path)
        return ExtractedFrames(
            frames=frames,
            paths=paths,
        )


def test_off_grid_preset_timestamp_is_queued_and_persisted_for_the_guide(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path)
    repository.create("job", "IMG_3325.MOV")
    input_path = repository.input_path("job", "IMG_3325.MOV")
    input_path.write_bytes(b"video")
    frames, paths = _frames()
    repository.update("job", frames=frames)
    generator = CapturingGenerator()
    extractor = QueueingExtractor()
    processor = JobProcessor(repository, Settings(data_dir=tmp_path), extractor=extractor, generator=generator)  # type: ignore[arg-type]

    processor._finish_guide("job", frames, paths, [], repository.source_filename("job"))

    job = repository.get("job")
    assert extractor.calls == [(input_path, (5.16, 11.67, 14.5))]
    assert generator.calls[0][0] == ["frame-0000", "snapshot-00005160", "frame-0045", "snapshot-00011670", "snapshot-00014500", "frame-0099"]
    assert job is not None and "snapshot-00005160" in {frame.frameId for frame in job.frames}


class PresetOnlyExtractor:
    def __init__(self) -> None:
        self.automatic_calls = 0
        self.snapshot_calls: list[tuple[Path, tuple[float, ...]]] = []

    def extract(self, input_path: Path, output_dir: Path, job_id: str) -> ExtractedFrames:
        self.automatic_calls += 1
        raise AssertionError("preset recordings must not use regular FPS extraction")

    def extract_snapshots(self, input_path: Path, output_dir: Path, job_id: str, timestamps_seconds: tuple[float, ...]) -> ExtractedFrames:
        self.snapshot_calls.append((input_path, timestamps_seconds))
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[Frame] = []
        paths: list[Path] = []
        for timestamp in timestamps_seconds:
            frame_id = snapshot_frame_id(timestamp)
            path = output_dir / f"{frame_id}.jpg"
            path.write_bytes(b"snapshot")
            frames.append(Frame(frameId=frame_id, timestampSeconds=timestamp, imageUrl=f"/jobs/{job_id}/frames/{frame_id}"))
            paths.append(path)
        return ExtractedFrames(frames=frames, paths=paths)


class AnnotationAnalyzer:
    def analyze_annotated(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("upload processing should stop for annotation before tracking")


class SuggestionRecorder:
    def __init__(self) -> None:
        self.frame_ids: list[str] = []

    def suggest(self, frame: Frame, path: Path) -> list[AnnotationSuggestion]:
        self.frame_ids.append(frame.frameId)
        return [AnnotationSuggestion(name="blue plate", frameIndex=0, point=PointPrompt(x=1, y=1), confidence=0.9)]


def test_preset_upload_extracts_only_configured_snapshots_before_annotation(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path)
    repository.create("job", "IMG_3320.MOV")
    input_path = repository.input_path("job", "IMG_3320.MOV")
    input_path.write_bytes(b"video")
    extractor = PresetOnlyExtractor()
    suggestions = SuggestionRecorder()
    processor = JobProcessor(
        repository,
        Settings(data_dir=tmp_path),
        extractor=extractor,  # type: ignore[arg-type]
        analyzer=AnnotationAnalyzer(),  # type: ignore[arg-type]
        suggestion_generator=suggestions,  # type: ignore[arg-type]
    )

    processor.process("job")

    job = repository.get("job")
    assert extractor.automatic_calls == 0
    assert extractor.snapshot_calls == [(input_path, GUIDE_FRAME_PRESETS["IMG_3320.MOV"].timestamps_seconds)]
    assert job is not None and [frame.frameId for frame in job.frames] == [
        "snapshot-00000000", "snapshot-00004800", "snapshot-00010000", "snapshot-00011000", "snapshot-00014000",
    ]
    assert suggestions.frame_ids == ["snapshot-00014000"]
    assert job.status == "annotating"


def test_snapshot_extraction_uses_an_exact_output_seek(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "source.mov"
    input_path.write_bytes(b"video")
    extractor = FFmpegExtractor(Settings(data_dir=tmp_path))
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"jpeg")

    monkeypatch.setattr("backend.app.processing.subprocess.run", fake_run)
    extracted = extractor.extract_snapshots(input_path, tmp_path, "job", (12.5,))

    assert commands[0][commands[0].index("-ss") + 1] == "12.500000"
    assert commands[0].index("-ss") > commands[0].index(str(input_path))
    assert extracted.frames[0].frameId == "snapshot-00012500"
