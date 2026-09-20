"""Source-specific settled-state frames for the small recorded demo set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import Frame


@dataclass(frozen=True)
class GuideFramePreset:
    source_fps: str
    minimum_timestamp_seconds: float
    maximum_timestamp_seconds: float
    timestamps_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.minimum_timestamp_seconds != 0 or self.maximum_timestamp_seconds <= 0:
            raise ValueError("A guide-frame preset needs a non-empty source range beginning at zero.")
        if not self.timestamps_seconds or tuple(sorted(set(self.timestamps_seconds))) != self.timestamps_seconds:
            raise ValueError("Guide-frame timestamps must be unique and source ordered.")
        if any(not self.minimum_timestamp_seconds <= timestamp <= self.maximum_timestamp_seconds for timestamp in self.timestamps_seconds):
            raise ValueError("A guide-frame timestamp is outside its recorded source range.")


# Measured with: ffprobe -select_streams v:0 -show_entries
# stream=avg_frame_rate,nb_frames:format=duration. Times are settled source
# states, not frame indexes, so changing REBUILT_ANALYSIS_FPS remains safe.
GUIDE_FRAME_PRESETS: dict[str, GuideFramePreset] = {
    "IMG_3320.MOV": GuideFramePreset(
        source_fps="47600/1587 (29.994 FPS)",
        minimum_timestamp_seconds=0.0,
        maximum_timestamp_seconds=15.868333,
        timestamps_seconds=(0.0, 4.8, 10.0, 11.0, 14.0),
    ),
    "IMG_3325.MOV": GuideFramePreset(
        source_fps="3175/106 (29.953 FPS)",
        minimum_timestamp_seconds=0.0,
        maximum_timestamp_seconds=21.2,
        timestamps_seconds=(0.0, 5.16, 9.0, 11.67, 14.5, 19.8),
    ),
    "IMG_3327.MOV": GuideFramePreset(
        source_fps="32550/1087 (29.945 FPS)",
        minimum_timestamp_seconds=0.0,
        maximum_timestamp_seconds=36.233333,
        timestamps_seconds=(0.0, 7.0, 9.75, 15.0, 22.0, 27.0, 30.0, 35.5),
    ),
    "IMG_3330.MOV": GuideFramePreset(
        source_fps="6320/211 (29.953 FPS)",
        minimum_timestamp_seconds=0.0,
        maximum_timestamp_seconds=21.1,
        timestamps_seconds=(0.0, 4.0, 8.5, 11.2, 15.5, 20.6),
    ),
}


def preset_for(filename: str | None) -> GuideFramePreset | None:
    if not filename:
        return None
    return GUIDE_FRAME_PRESETS.get(Path(filename).name.upper())


def snapshot_frame_id(timestamp_seconds: float) -> str:
    return f"snapshot-{round(timestamp_seconds * 1000):08d}"


def off_grid_snapshot_timestamps(filename: str | None, frames: list[Frame]) -> tuple[float, ...]:
    """Return requested source timestamps not represented by the extraction grid."""
    preset = preset_for(filename)
    if preset is None:
        return ()
    return tuple(
        timestamp
        for timestamp in preset.timestamps_seconds
        if not any(abs(frame.timestampSeconds - timestamp) < 1e-6 for frame in frames)
    )


def select_preset_frames(
    filename: str | None,
    frames: list[Frame],
    paths: list[Path],
) -> tuple[list[Frame], list[Path]] | None:
    """Return exact preset frames after any off-grid snapshots have been queued."""
    preset = preset_for(filename)
    if preset is None or not frames or len(frames) != len(paths):
        return None
    if frames[0].timestampSeconds > preset.minimum_timestamp_seconds:
        return None
    if frames[-1].timestampSeconds < preset.timestamps_seconds[-1]:
        return None
    selected_indexes: list[int] = []
    for timestamp in preset.timestamps_seconds:
        index = next((index for index, frame in enumerate(frames) if abs(frame.timestampSeconds - timestamp) < 1e-6), None)
        if index is None:
            return None
        selected_indexes.append(index)
    if selected_indexes != sorted(set(selected_indexes)):
        return None
    return (
        [frames[index] for index in selected_indexes],
        [paths[index] for index in selected_indexes],
    )
