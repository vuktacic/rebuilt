#!/usr/bin/env python3
"""Run a local analysis fixture and atomically save a sanitized report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import Settings
from backend.app.processing import FFmpegExtractor
from backend.app.vision import create_vision_analyzer, validate_vision_runtime

REPORT_SCHEMA_VERSION = 1


def _video_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, payload: dict, *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite completed report: {path}; pass --replace to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.is_file():
        raise SystemExit(f"Fixture video does not exist: {video}")

    settings = Settings.from_env()
    backend = validate_vision_runtime(settings)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rebuilt-fixture-") as temporary:
        root = Path(temporary)
        extracted = FFmpegExtractor(settings).extract(video, root / "frames", "fixture")
        result = create_vision_analyzer(settings).analyze(root / "frames", len(extracted.frames), "fixture")
    frame_ids = {frame.frameId for frame in extracted.frames}
    if any(event.beforeFrameId not in frame_ids or event.afterFrameId not in frame_ids for event in result.events):
        raise RuntimeError("Analyzer produced an event that references an absent extracted frame.")
    payload = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "status": "completed",
        "source": {"basename": video.name, "sha256": _video_sha256(video)},
        "runtime": {
            "selectedBackend": backend,
            "modelVersion": result.analysis.modelVersion,
            "configVersion": result.analysis.configVersion,
            "analysisFps": settings.analysis_fps,
            "windowSeconds": settings.analysis_window_seconds,
        },
        "extraction": {
            "frameCount": len(extracted.frames),
            "durationSeconds": extracted.frames[-1].timestampSeconds if extracted.frames else 0.0,
        },
        "elapsedSeconds": time.monotonic() - started,
        "analysis": result.analysis.model_dump(),
        "tracks": [track.model_dump() for track in result.tracks],
        "events": [event.model_dump() for event in result.events],
    }
    _write_report(args.output, payload, replace=args.replace)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
