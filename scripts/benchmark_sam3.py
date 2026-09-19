#!/usr/bin/env python3
"""Run local segmentation/event analysis without calling the guide model."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import Settings
from backend.app.processing import FFmpegExtractor
from backend.app.vision import Sam3VisionAnalyzer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    with tempfile.TemporaryDirectory(prefix="rebuilt-sam3-") as temp_dir:
        root = Path(temp_dir)
        extracted = FFmpegExtractor(settings).extract(args.video, root / "frames", "benchmark")
        result = Sam3VisionAnalyzer(settings).analyze(root / "frames", len(extracted.frames), "benchmark")
    print(json.dumps({
        "analysis": result.analysis.model_dump(),
        "tracks": [track.model_dump() for track in result.tracks],
        "events": [event.model_dump() for event in result.events],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
