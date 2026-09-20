from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .errors import AppError
from .models import Frame, Guide, ManualPair, PairFinding


ASTRA_PROMPT = """Review one labelled before/after pair from a fixed-camera LEGO disassembly.
Describe only the visible physical difference. Use attach when a piece is present in AFTER,
detach when it is absent in AFTER, no_change when no reliable change is visible, and
uncertain_change when occlusion or ambiguity prevents a confident classification.
Do not invent part identities, colors, or frame references. Return one compact finding.
"""


class PairReviewer(Protocol):
    def review(self, pair: ManualPair, frames: list[Frame], frame_paths: list[Path]) -> PairFinding: ...


class GuideDrafter(Protocol):
    def draft(self, findings: list[PairFinding], pairs: list[ManualPair], frames: list[Frame]) -> Guide: ...


def build_pair_review_payload(
    pair: ManualPair,
    frames: list[Frame],
    frame_paths: list[Path],
    *,
    model: str,
    detail: str = "low",
) -> dict[str, Any]:
    """Build the bounded Astra request for exactly one semantic pair."""
    if len(frame_paths) != 2:
        raise AppError(422, "PAIR_EVIDENCE_INVALID", "A manual review requires exactly two evidence images.")
    frame_by_id = {frame.frameId: frame for frame in frames}
    if pair.beforeFrameId not in frame_by_id or pair.afterFrameId not in frame_by_id:
        raise AppError(422, "PAIR_FRAME_INVALID", "A manual pair refers to a frame that was not extracted.")
    content: list[dict[str, Any]] = [{"type": "input_text", "text": ASTRA_PROMPT}]
    ordered = [("BEFORE", pair.beforeFrameId, frame_paths[0]), ("AFTER", pair.afterFrameId, frame_paths[1])]
    for role, frame_id, path in ordered:
        frame = frame_by_id[frame_id]
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({
            "type": "input_text",
            "text": (
                f"pairId={pair.pairId}; {role} {frame.frameId}; "
                f"source={frame.timestampSeconds:.3f}s; "
                f"assembly={frame.assemblyTimeSeconds if frame.assemblyTimeSeconds is not None else 'legacy'}s"
            ),
        })
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}", "detail": detail})
    return {"model": model, "input": [{"role": "user", "content": content}]}


def validate_pair_finding(finding: PairFinding, pair: ManualPair, frames: list[Frame]) -> PairFinding:
    """Reject syntactically valid Astra output that crosses the pair boundary."""
    if finding.pairId != pair.pairId:
        raise AppError(502, "PAIR_FINDING_INVALID", "The visual reviewer returned an unexpected pair ID.")
    allowed = {pair.beforeFrameId, pair.afterFrameId}
    frame_ids = {frame.frameId for frame in frames}
    if not set(finding.evidenceFrameIds).issubset(allowed) or not set(finding.evidenceFrameIds).issubset(frame_ids):
        raise AppError(502, "PAIR_FINDING_INVALID", "The visual reviewer cited evidence outside the submitted pair.")
    if finding.status in {"completed", "needs_review"} and not finding.difference.strip():
        raise AppError(502, "PAIR_FINDING_INVALID", "The visual reviewer returned an empty difference.")
    return finding.model_copy(update={"difference": finding.difference.strip()})


def build_luna_payload(
    findings: list[PairFinding],
    pairs: list[ManualPair],
    frames: list[Frame],
    *,
    model: str,
) -> dict[str, Any]:
    """Build the text-only handoff from validated pair evidence to Luna."""
    frame_ids = {frame.frameId for frame in frames}
    pairs_by_id = {pair.pairId: pair for pair in pairs}
    lines = [
        "Draft a concise editable LEGO assembly guide from validated visual differences.",
        "Use one chronological guide step per physical change; merge findings that share the same AFTER frame into one step.",
        "Use only the supplied frame IDs and do not invent part identities or evidence.",
    ]
    for finding in findings:
        pair = pairs_by_id.get(finding.pairId)
        if pair is None or finding.status == "failed":
            continue
        if not set(finding.evidenceFrameIds).issubset(frame_ids):
            raise AppError(422, "PAIR_FINDING_INVALID", "A guide finding references a frame that was not extracted.")
        lines.append(
            f"pairId={pair.pairId}; sequence={pair.sequence}; BEFORE={pair.beforeFrameId}; "
            f"AFTER={pair.afterFrameId}; action={finding.action}; difference={finding.difference}; "
            f"uncertainty={finding.uncertainty or 'none'}; confidence={finding.confidence:.2f}"
        )
    return {"model": model, "input": [{"role": "user", "content": [{"type": "input_text", "text": "\n".join(lines)}]}]}


class OpenAIResponsesPairReviewer:
    def __init__(self, settings: Settings, *, model: str = "gpt-6-astra", detail: str = "low"):
        self.settings = settings
        self.model = model
        self.detail = detail

    def review(self, pair: ManualPair, frames: list[Frame], frame_paths: list[Path]) -> PairFinding:
        if not self.settings.openai_api_key:
            raise AppError(500, "OPENAI_NOT_CONFIGURED", "OPENAI_API_KEY is not configured.")
        payload = build_pair_review_payload(pair, frames, frame_paths, model=self.model, detail=self.detail)
        payload["text"] = {"format": {"type": "json_schema", "name": "pair_finding", "strict": True, "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["pairId", "status", "action", "difference", "uncertainty", "confidence", "evidenceFrameIds", "attemptCount"],
            "properties": {
                "pairId": {"type": "string"},
                "status": {"enum": ["completed", "needs_review", "failed"]},
                "action": {"enum": ["attach", "detach", "uncertain_change", "no_change"]},
                "difference": {"type": "string"},
                "uncertainty": {"anyOf": [{"enum": ["occluded", "ambiguous", "insufficient_evidence"]}, {"type": "null"}]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidenceFrameIds": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "attemptCount": {"type": "integer", "minimum": 1},
            },
        }}}
        request = urllib.request.Request(
            self.settings.openai_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.api_timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AppError(502, "MODEL_FAILED", "The pair visual review request failed.") from exc
        try:
            output = body.get("output_text") or next(
                part["text"] for item in body["output"] for part in item.get("content", []) if part.get("text")
            )
            return validate_pair_finding(PairFinding.model_validate(json.loads(output)), pair, frames)
        except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The pair visual reviewer returned invalid structured output.") from exc


class OpenAIResponsesGuideDrafter:
    def __init__(self, settings: Settings, *, model: str = "gpt-5.6-luna"):
        self.settings = settings
        self.model = model

    def draft(self, findings: list[PairFinding], pairs: list[ManualPair], frames: list[Frame]) -> Guide:
        if not self.settings.openai_api_key:
            raise AppError(500, "OPENAI_NOT_CONFIGURED", "OPENAI_API_KEY is not configured.")
        payload = build_luna_payload(findings, pairs, frames, model=self.model)
        payload["text"] = {"format": {"type": "json_schema", "name": "lego_guide", "strict": True, "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "steps"],
            "properties": {
                "title": {"type": "string"},
                "steps": {"type": "array", "minItems": 1, "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "frameId", "uncertainty"],
                    "properties": {
                        "text": {"type": "string"},
                        "frameId": {"type": "string"},
                        "uncertainty": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                }},
            },
        }}}
        request = urllib.request.Request(
            self.settings.openai_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.api_timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            output = body.get("output_text") or next(
                part["text"] for item in body["output"] for part in item.get("content", []) if part.get("text")
            )
            guide = Guide.model_validate(json.loads(output))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AppError(502, "MODEL_FAILED", "The guide drafting request failed.") from exc
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide drafter returned invalid structured output.") from exc
        frame_ids = {frame.frameId for frame in frames}
        if any(step.frameId not in frame_ids for step in guide.steps):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide drafter referenced an unavailable frame.")
        return guide
