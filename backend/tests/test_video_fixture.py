from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.config import Settings
from backend.app.processing import FFmpegExtractor


def test_recorded_fixture_extracts_at_analysis_rate(tmp_path: Path) -> None:
    video = Path(__file__).parents[2] / "testdata" / "IMG_3320.MOV"
    if not video.is_file():
        pytest.skip("local recorded fixture is not present")

    extracted = FFmpegExtractor(Settings(data_dir=tmp_path, analysis_fps=10, processing_timeout_seconds=60)).extract(
        video,
        tmp_path / "frames",
        "fixture",
    )

    assert 150 <= len(extracted.frames) <= 170
    assert extracted.frames[0].timestampSeconds == 0
    assert extracted.frames[-1].timestampSeconds > 15
    assert all(left.timestampSeconds < right.timestampSeconds for left, right in zip(extracted.frames, extracted.frames[1:]))
