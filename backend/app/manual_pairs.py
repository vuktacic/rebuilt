from __future__ import annotations

import base64
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from .config import Settings
from .errors import AppError
from .models import Frame, Guide, PairFinding, Storyboard, StoryboardPair
from .post_processing import responses_json


PAIR_FINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status", "changedPieceDescription", "receivingPieceDescription", "receivingLocation",
        "beforeAfterDifference", "supportedPlacement", "uncertainty", "reason", "suggestion",
    ],
    "properties": {
        "status": {"type": "string", "enum": ["change", "no_change", "unclear"]},
        "changedPieceDescription": {"type": ["string", "null"]},
        "receivingPieceDescription": {"type": ["string", "null"]},
        "receivingLocation": {"type": ["string", "null"]},
        "beforeAfterDifference": {"type": "string"},
        "supportedPlacement": {"type": ["string", "null"]},
        "uncertainty": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
        "suggestion": {"type": ["string", "null"]},
    },
}

WRITER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "steps"],
    "properties": {
        "title": {"type": "string"},
        "steps": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["pairId", "text", "uncertainty"],
            "properties": {
                "pairId": {"type": "string"},
                "text": {"type": "string"},
                "uncertainty": {"type": ["string", "null"]},
            },
        }},
    },
}


def _image_content(path: Path) -> dict[str, str]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}


def build_storyboard(frames: list[Frame], selected_frame_ids: list[str], context: str = "", *, revision: int = 0) -> Storyboard:
    by_id = {frame.frameId: frame for frame in frames}
    if len(selected_frame_ids) < 2:
        raise AppError(422, "SNAPSHOTS_REQUIRED", "Select at least two settled state snapshots.")
    if len(set(selected_frame_ids)) != len(selected_frame_ids):
        raise AppError(422, "SNAPSHOTS_DUPLICATE", "A snapshot can only be selected once.")
    if any(frame_id not in by_id for frame_id in selected_frame_ids):
        raise AppError(422, "SNAPSHOT_FRAME_INVALID", "A selected snapshot is not an extracted frame.")
    ordered = sorted((by_id[frame_id] for frame_id in selected_frame_ids), key=lambda frame: (frame.timestampSeconds, frame.frameId))
    if [frame.frameId for frame in ordered] != selected_frame_ids:
        raise AppError(422, "SNAPSHOTS_OUT_OF_ORDER", "Snapshots must be saved in source-time order.")
    pairs = [StoryboardPair(
        pairId=f"pair-{index + 1:04d}",
        beforeFrameId=before.frameId,
        afterFrameId=after.frameId,
        beforeTimestampSeconds=before.timestampSeconds,
        afterTimestampSeconds=after.timestampSeconds,
    ) for index, (before, after) in enumerate(zip(ordered, ordered[1:], strict=False))]
    return Storyboard(revision=revision + 1, selectedFrameIds=list(selected_frame_ids), context=context.strip(), pairs=pairs)


class ManualPairsProvider:
    """Astra pair comparison and Luna text-only guide writing for manual_pairs."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def compare(self, pair: StoryboardPair, before_path: Path, after_path: Path, context: str) -> PairFinding:
        content = [
            {"type": "input_text", "text": (
                "Compare exactly these two settled state snapshots from an object-assembly recording. "
                "The first image is BEFORE and the second is AFTER. Distinguish construction changes from hands, "
                "lighting, camera movement, or ordinary repositioning. Do not infer hidden connections. If multiple "
                "actions cannot be separated, return unclear and suggest an intermediate snapshot. Return the visible "
                "changed piece, receiving piece/location, before/after difference, supported placement, and uncertainty."
            )},
            {"type": "input_text", "text": f"Optional user context or piece names: {context or 'none'}"},
            {"type": "input_text", "text": f"BEFORE · {pair.beforeFrameId} · {pair.beforeTimestampSeconds:.3f}s"},
            _image_content(before_path),
            {"type": "input_text", "text": f"AFTER · {pair.afterFrameId} · {pair.afterTimestampSeconds:.3f}s"},
            _image_content(after_path),
        ]
        body = responses_json(
            self.settings,
            content=content,
            name="snapshot_difference",
            schema=PAIR_FINDING_SCHEMA,
            model=self.settings.manual_compare_model,
            reasoning_effort=self.settings.manual_reasoning_effort,
        )
        try:
            return PairFinding.model_validate(body)
        except Exception as exc:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The snapshot comparison returned invalid structured output.") from exc

    def write_guide(self, storyboard: Storyboard) -> tuple[str, dict[str, str | None]]:
        included = [pair for pair in storyboard.pairs if pair.disposition == "include" and pair.reviewedFinding is not None]
        if not included:
            raise AppError(422, "NO_REVIEWED_DIFFERENCES", "Include at least one reviewed difference before generating a guide.")
        differences = [{"pairId": pair.pairId, "finding": pair.reviewedFinding.model_dump()} for pair in included]
        body = responses_json(
            self.settings,
            content=[{"type": "input_text", "text": (
                "Write one concise assembly instruction for every supplied reviewed pair, in supplied order. "
                "Return exactly the supplied pair IDs, once each. Do not invent, merge, reorder, or omit actions. "
                "Preserve uncertainty from each finding. This is text-only: do not choose screenshots or timestamps.\n"
                + json.dumps({"context": storyboard.context, "differences": differences}, ensure_ascii=False)
            )}],
            name="snapshot_guide",
            schema=WRITER_SCHEMA,
            model=self.settings.manual_writer_model,
            reasoning_effort=self.settings.manual_reasoning_effort,
        )
        try:
            title = str(body["title"]).strip()
            steps = body["steps"]
            if not title or not isinstance(steps, list):
                raise ValueError("missing title or steps")
            parsed = {str(item["pairId"]): (str(item["text"]).strip(), item.get("uncertainty")) for item in steps}
        except (KeyError, TypeError, ValueError) as exc:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide writer returned invalid structured output.") from exc
        expected = [pair.pairId for pair in included]
        if list(parsed) != expected or len(parsed) != len(expected) or any(not text for text, _ in parsed.values()):
            raise AppError(502, "MODEL_COVERAGE_INVALID", "The guide writer did not cover the reviewed differences exactly.")
        return title, {pair_id: value for pair_id, value in parsed.items()}


def compare_pairs_bounded(
    provider: ManualPairsProvider,
    pairs: Iterable[StoryboardPair],
    frame_path: Callable[[str], Path],
    context: str,
    on_result: Callable[[str, PairFinding | Exception], None] | None = None,
) -> dict[str, PairFinding | Exception]:
    selected = list(pairs)
    if len(selected) > 3:
        raise AppError(422, "PAIR_BATCH_TOO_LARGE", "At most three snapshot comparisons may run in one request.")
    results: dict[str, PairFinding | Exception] = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="manual-pair") as executor:
        futures: dict[Future[PairFinding], str] = {
            executor.submit(provider.compare, pair, frame_path(pair.beforeFrameId), frame_path(pair.afterFrameId), context): pair.pairId
            for pair in selected
        }
        for future in as_completed(futures):
            pair_id = futures[future]
            try:
                results[pair_id] = future.result()
            except Exception as exc:  # preserve successful pairs when one call fails
                results[pair_id] = exc
            if on_result is not None:
                on_result(pair_id, results[pair_id])
    return results
