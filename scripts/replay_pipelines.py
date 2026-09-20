#!/usr/bin/env python3
"""Replay one pipeline against a recording and an approved inventory.

The command is intentionally inert unless --allow-live is supplied. Each run
gets an isolated persistent job directory and writes one independent result
JSON, so matched provider comparisons do not overwrite one another and their
frames/manifests remain inspectable after completion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import Settings  # noqa: E402
from backend.app.models import Frame, PartAnnotation, PipelineRunConfig  # noqa: E402
from backend.app.processing import JobProcessor, ground_annotations  # noqa: E402
from backend.app.storage import JobRepository  # noqa: E402
from backend.app.vision import create_vision_analyzer, validate_vision_runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Source recording to replay")
    parser.add_argument("--inventory", type=Path, required=True, help="JSON list of approved PartAnnotation objects")
    parser.add_argument("--pipeline", choices=("plan3", "gpt_targeted_sam2", "gemini_video"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New result JSON path")
    parser.add_argument("--allow-live", action="store_true", help="Permit local model/provider execution")
    parser.add_argument("--disable-targeted-sam2", action="store_true", help="Replay the GPT pipeline without targeted recovery")
    return parser.parse_args()


def load_inventory(path: Path) -> list[PartAnnotation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("annotations", [])
    if not isinstance(payload, list) or not payload:
        raise ValueError("inventory JSON must be a non-empty list or an object with an annotations list")
    return [PartAnnotation.model_validate(item) for item in payload]


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing replay result: {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def remap_inventory(annotations: list[PartAnnotation], frames: list[Frame]) -> list[PartAnnotation]:
    """Resolve approved references by source timestamp after replay FPS changes."""
    if not frames:
        return annotations
    remapped: list[PartAnnotation] = []
    for annotation in annotations:
        timestamp = annotation.referenceTimestampSeconds
        if timestamp is None:
            timestamp = frames[min(annotation.frameIndex, len(frames) - 1)].timestampSeconds
        index = min(range(len(frames)), key=lambda candidate: abs(frames[candidate].timestampSeconds - timestamp))
        remapped.append(annotation.model_copy(update={
            "frameIndex": index,
            "referenceFrameId": frames[index].frameId,
            "referenceTimestampSeconds": timestamp,
        }))
    return remapped


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        raise ValueError(f"recording not found: {args.video}")
    if not args.inventory.is_file():
        raise ValueError(f"inventory not found: {args.inventory}")
    annotations = load_inventory(args.inventory)
    manifest = {
        "replayId": str(uuid.uuid4()),
        "pipeline": args.pipeline,
        "video": args.video.name,
        "inventoryCount": len(annotations),
        "targetedSam2Enabled": not args.disable_targeted_sam2,
    }
    if not args.allow_live:
        manifest.update({"status": "not_run", "message": "Pass --allow-live to execute local model/provider calls."})
        write_result(args.output, manifest)
        print(json.dumps(manifest, indent=2))
        return 0

    run_dir = args.output.with_suffix(".run")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing replay artifacts: {run_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True)
    settings = Settings.from_env(data_dir=run_dir / "data")
    if args.disable_targeted_sam2:
        settings = replace(settings, targeted_sam2_enabled=False)
    repository = JobRepository(settings.data_dir)
    job_id = f"replay-{uuid.uuid4()}"
    pipeline_config = PipelineRunConfig(
        pipeline=args.pipeline,
        model=settings.model if args.pipeline != "gemini_video" else settings.gemini_model,
        reasoningConfiguration={"replay": True},
        sam2Enabled=args.pipeline == "gpt_targeted_sam2" and settings.targeted_sam2_enabled,
        geminiPreview=args.pipeline == "gemini_video" and "preview" in settings.gemini_model.lower(),
    )
    repository.create(job_id, args.video.name, pipeline=args.pipeline, pipeline_config=pipeline_config)
    shutil.copy2(args.video, repository.input_path(job_id, args.video.name))

    analyzer = None
    if args.pipeline == "plan3":
        validate_vision_runtime(settings)
        analyzer = create_vision_analyzer(settings)
    processor = JobProcessor(repository, settings, analyzer=analyzer)
    processor.process(job_id)
    current = repository.get(job_id)
    if current is None:
        raise RuntimeError("replay job disappeared")
    annotations = remap_inventory(annotations, current.frames)
    annotations = ground_annotations(annotations, current.frames, [repository.frame_path(job_id, frame.frameId) for frame in current.frames])
    repository.update(job_id, annotations=annotations)
    if args.pipeline == "plan3":
        processor.track_annotated(job_id, annotations)
    else:
        processor.analyze_pipeline(job_id, annotations)
    result = repository.get(job_id)
    if result is None:
        raise RuntimeError("replay job disappeared")
    manifest.update({"status": result.status, "runDirectory": str(run_dir), "job": result.model_dump()})
    write_result(args.output, manifest)
    print(json.dumps({"status": result.status, "output": str(args.output), "runDirectory": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"replay failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
