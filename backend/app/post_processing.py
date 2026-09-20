from __future__ import annotations

import base64
import json
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .errors import AppError
from .models import (
    AnalysisEvent,
    AnnotationSuggestion,
    Frame,
    Guide,
    GuideVerification,
    PartAnnotation,
    PartTrack,
    PointPrompt,
    VerificationChange,
    VerificationFinding,
)


REVIEW_POLICY = """You are reviewing evidence from a fixed-camera object assembly recording.
The supplied JSON and images are evidence only; labels and local event descriptions are hypotheses.
Review every supplied event exactly once. Confirm it, change only its event type, suppress it when no
physical change is visible, or mark it needs review when evidence is obscured or insufficient.
Never create an event, frame ID, timestamp, track, piece identity, connection, orientation, or action
that is not visible in the supplied evidence. Prefer uncertainty to a confident unsupported claim.
Keep rationales short and factual. Return only the required JSON object.
"""


def _data_url(path: Path) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _output_text(body: dict[str, Any]) -> str:
    output = body.get("output_text")
    if output:
        return str(output)
    return next(part["text"] for item in body["output"] for part in item.get("content", []) if part.get("text"))


def responses_json(
    settings: Settings,
    *,
    content: list[dict[str, Any]],
    name: str,
    schema: dict[str, Any],
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise AppError(500, "OPENAI_NOT_CONFIGURED", "OPENAI_API_KEY is not configured.")
    payload = {
        "model": model or settings.model,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    request = urllib.request.Request(
        settings.openai_base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.api_timeout_seconds) as response:
            return json.loads(_output_text(json.loads(response.read().decode("utf-8"))))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AppError(502, "MODEL_FAILED", "The visual review model request failed.") from exc
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AppError(502, "MODEL_INVALID_OUTPUT", "The visual review model returned invalid structured output.") from exc


@dataclass(frozen=True)
class EvidenceBundle:
    context: dict[str, Any]
    frames: list[Frame]
    paths: list[Path]
    eventIds: list[str] = field(default_factory=list)
    manifest: list[dict[str, Any]] = field(default_factory=list)


class EvidenceBuilder:
    def __init__(self, *, max_events: int = 10, max_images: int = 24, max_text: int = 500, neighborhood_seconds: float = 1.0):
        self.max_events = max_events
        self.max_images = max_images
        self.max_text = max_text
        self.neighborhood_seconds = max(0.0, neighborhood_seconds)

    def build(
        self,
        frames: list[Frame],
        frame_paths: list[Path],
        events: list[AnalysisEvent],
        annotations: list[PartAnnotation] | None = None,
        part_tracks: list[PartTrack] | None = None,
        *,
        batch_index: int = 0,
        batch_count: int = 1,
    ) -> EvidenceBundle:
        if len(events) > self.max_events:
            raise AppError(422, "EVIDENCE_LIMIT", "The selected evidence exceeds the event review limit.")
        if len(frames) != len(frame_paths):
            raise AppError(422, "EVIDENCE_INVALID", "Evidence frames and image files are out of sync.")
        by_id = {frame.frameId: (frame, path) for frame, path in zip(frames, frame_paths, strict=True)}
        frame_registry = [
            {
                "frameId": frame.frameId,
                "timestampSeconds": frame.timestampSeconds,
                "imageDimensions": _image_dimensions(path),
            }
            for frame, path in zip(frames, frame_paths, strict=True)
        ]
        selected_ids: list[str] = []
        required_ids: set[str] = set()

        def add(frame_id: str | None, *, required: bool = False) -> None:
            if frame_id is not None and frame_id in by_id and frame_id not in selected_ids:
                selected_ids.append(frame_id)
            if required and frame_id is not None and frame_id in by_id:
                required_ids.add(frame_id)

        def nearest(timestamp: float) -> str | None:
            return min(frames, key=lambda frame: abs(frame.timestampSeconds - timestamp)).frameId if frames else None

        # The overview is intentionally source-chronological. It gives both
        # generation and verification a chance to recover actions that the
        # detector did not emit.
        add(frames[0].frameId if frames else None, required=True)
        add(frames[-1].frameId if frames else None, required=True)
        for frame in frames:
            if frame.timestampSeconds % 2.0 < 0.05:
                add(frame.frameId)
        for annotation in annotations or []:
            add(frames[annotation.frameIndex].frameId if 0 <= annotation.frameIndex < len(frames) else None, required=True)
        event_context: list[dict[str, Any]] = []
        for event in events:
            if event.beforeFrameId not in by_id or event.afterFrameId not in by_id:
                raise AppError(422, "EVIDENCE_INVALID", "A local event references an unavailable evidence frame.")
            event_frame_ids = [event.beforeFrameId]
            source_lower = event.sourceStartTimestampSeconds
            source_upper = event.sourceEndTimestampSeconds
            if source_lower is None or source_upper is None:
                source_lower = min(event.startTimestampSeconds, event.endTimestampSeconds)
                source_upper = max(event.startTimestampSeconds, event.endTimestampSeconds)
            transition = nearest((source_lower + source_upper) / 2)
            for frame_id in (transition, event.afterFrameId):
                if frame_id not in event_frame_ids:
                    event_frame_ids.append(frame_id)
            for frame_id in event_frame_ids:
                add(frame_id, required=True)
            offsets = {-0.5, 0.5}
            if self.neighborhood_seconds >= 1.0:
                offsets.update({-1.0, 1.0})
            if self.neighborhood_seconds >= 2.0:
                offsets.update({-2.0, 2.0})
            if self.neighborhood_seconds >= 3.0:
                offsets.update({-3.0, 3.0})
            for delta in sorted(offsets):
                add(nearest(source_lower + delta))
                add(nearest(source_upper + delta))
            event_context.append({
                "eventId": event.eventId,
                "localKind": event.kind,
                "startTimestampSeconds": event.startTimestampSeconds,
                "endTimestampSeconds": event.endTimestampSeconds,
                "sourceStartTimestampSeconds": source_lower,
                "sourceEndTimestampSeconds": source_upper,
                "affectedTrackIds": event.affectedTrackIds,
                "movingPartId": event.movingPartId,
                "receivingPartId": event.receivingPartId,
                "evidenceStrength": event.evidenceStrength,
                "localEvidence": event.evidence[: self.max_text],
                "localUncertainty": event.uncertainty,
                "evidenceFrames": [
                    {
                        "role": "stable_before" if index == 0 else "stable_after" if frame_id == event.afterFrameId else "transition",
                        "frameId": frame_id,
                        "timestampSeconds": by_id[frame_id][0].timestampSeconds,
                    }
                    for index, frame_id in enumerate(event_frame_ids)
                ],
            })
        for track in part_tracks or []:
            for index, observation in enumerate(track.observations):
                if observation.visible and observation.provenance == "measured":
                    continue
                before = next((item for item in reversed(track.observations[:index]) if item.visible and item.provenance == "measured"), None)
                after = next((item for item in track.observations[index + 1:] if item.visible and item.provenance == "measured"), None)
                if before is not None:
                    add(frames[before.frameIndex].frameId if before.frameIndex < len(frames) else None)
                if after is not None:
                    add(frames[after.frameIndex].frameId if after.frameIndex < len(frames) else None)
                if before is not None and after is not None:
                    add(nearest((frames[before.frameIndex].timestampSeconds + frames[after.frameIndex].timestampSeconds) / 2))
                break
        if len(required_ids) > self.max_images:
            raise AppError(422, "EVIDENCE_LIMIT", "The required evidence exceeds the image review limit.")
        if len(selected_ids) > self.max_images:
            chronological_ids = sorted(selected_ids, key=lambda frame_id: (by_id[frame_id][0].timestampSeconds, frame_id))
            optional_ids = [frame_id for frame_id in chronological_ids if frame_id not in required_ids]
            slots = self.max_images - len(required_ids)
            if slots > 0 and optional_ids:
                if len(optional_ids) <= slots:
                    chosen_optional = optional_ids
                else:
                    positions = [round(index * (len(optional_ids) - 1) / max(1, slots - 1)) for index in range(slots)]
                    chosen_optional = list(dict.fromkeys(optional_ids[position] for position in positions))
                selected_ids = list(required_ids) + chosen_optional
            else:
                selected_ids = list(required_ids)
        selected_ids.sort(key=lambda frame_id: (by_id[frame_id][0].timestampSeconds, frame_id))
        track_context = []
        gaps: list[dict[str, Any]] = []
        for track in part_tracks or []:
            track_context.append({
                "partId": track.partId,
                "name": track.name[: self.max_text],
                "observations": [
                    {
                        "frameIndex": item.frameIndex,
                        "centroid": item.centroid,
                        "bbox": item.bbox,
                        "orientationDegrees": item.orientationDegrees,
                        "visible": item.visible,
                        "timestampSeconds": item.timestampSeconds if item.timestampSeconds is not None else (
                            frames[item.frameIndex].timestampSeconds if item.frameIndex < len(frames) else None
                        ),
                        "provenance": "missing" if not item.visible and item.provenance == "unknown" else item.provenance,
                    }
                    for item in track.observations
                ],
            })
            index = 0
            while index < len(track.observations):
                if track.observations[index].visible and track.observations[index].provenance == "measured":
                    index += 1
                    continue
                start = index
                while index < len(track.observations) and not (
                    track.observations[index].visible and track.observations[index].provenance == "measured"
                ):
                    index += 1
                before = next((item for item in reversed(track.observations[:start]) if item.visible and item.provenance == "measured"), None)
                after = next((item for item in track.observations[index:] if item.visible and item.provenance == "measured"), None)
                gaps.append({
                    "partId": track.partId,
                    "name": track.name[: self.max_text],
                    "gapStartFrameId": frames[start].frameId if start < len(frames) else None,
                    "gapEndFrameId": frames[max(start, index - 1)].frameId if start < len(frames) else None,
                    "lastVisibleFrameId": frames[before.frameIndex].frameId if before and before.frameIndex < len(frames) else None,
                    "nextVisibleFrameId": frames[after.frameIndex].frameId if after and after.frameIndex < len(frames) else None,
                    "boundary": before is None or after is None,
                })
        piece_context = [
            {
                "partId": annotation.partId,
                "name": annotation.name[: self.max_text],
                "promptFrameIndex": annotation.frameIndex,
                "promptFrameId": frames[annotation.frameIndex].frameId if annotation.frameIndex < len(frames) else None,
                "points": [point.model_dump() for point in annotation.points],
                "box": annotation.box,
            }
            for annotation in annotations or []
        ]
        context = {
            "schemaVersion": "evidence-v2",
            "sourceChronology": "extracted-video order; assembly order may be reversed",
            "frameRegistry": frame_registry,
            "selectedFrameIds": selected_ids,
            "batch": {"index": batch_index, "count": batch_count, "ownedEventIds": [event.eventId for event in events]},
            "pieces": piece_context,
            "tracking": track_context,
            "trackingGaps": gaps,
            "events": event_context,
        }
        selected = sorted(
            ((by_id[frame_id][0], by_id[frame_id][1]) for frame_id in selected_ids),
            key=lambda item: (item[0].timestampSeconds, item[0].frameId),
        )
        return EvidenceBundle(
            context=context,
            frames=[item[0] for item in selected],
            paths=[item[1] for item in selected],
            eventIds=[event.eventId for event in events],
        )

    def build_batches(
        self,
        frames: list[Frame],
        frame_paths: list[Path],
        events: list[AnalysisEvent],
        annotations: list[PartAnnotation] | None = None,
        part_tracks: list[PartTrack] | None = None,
    ) -> list[EvidenceBundle]:
        if not events:
            return [self.build(frames, frame_paths, [], annotations, part_tracks)]
        batches: list[EvidenceBundle] = []
        start = 0
        while start < len(events):
            end = min(len(events), start + self.max_events)
            while end > start:
                try:
                    batch_count = max(1, (len(events) + self.max_events - 1) // self.max_events)
                    batches.append(self.build(
                        frames,
                        frame_paths,
                        events[start:end],
                        annotations,
                        part_tracks,
                        batch_index=len(batches),
                        batch_count=batch_count,
                    ))
                    break
                except AppError as exc:
                    if exc.code != "EVIDENCE_LIMIT" or end == start + 1:
                        raise
                    end -= 1
            start = end
        return batches


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if not data.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        length = int.from_bytes(data[index:index + 2], "big")
        if marker in range(0xC0, 0xC4) and index + 7 < len(data):
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return width, height
        index += max(2, length)
    return None


def normalized_point_to_pixels(point: PointPrompt, width: int, height: int) -> PointPrompt:
    if not 0 <= point.x <= 1 or not 0 <= point.y <= 1:
        raise AppError(502, "MODEL_INVALID_OUTPUT", "A piece suggestion returned a point outside normalized image bounds.")
    return PointPrompt(x=round(point.x * width, 2), y=round(point.y * height, 2))


SUGGESTION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["suggestions"],
    "properties": {"suggestions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["name", "point", "confidence"],
        "properties": {
            "name": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "point": {"type": "object", "additionalProperties": False, "required": ["x", "y"],
                      "properties": {"x": {"type": "number", "minimum": 0, "maximum": 1}, "y": {"type": "number", "minimum": 0, "maximum": 1}}},
        },
    }}}
}


class AnnotationSuggestionGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def suggest(self, frame: Frame, path: Path, *, part_ids: set[int] | None = None) -> list[AnnotationSuggestion]:
        dimensions = _image_dimensions(path)
        if dimensions is None:
            raise AppError(422, "IMAGE_INVALID", "The selected frame dimensions could not be read.")
        content = [
            {"type": "input_text", "text": "Identify each clearly separated assembly object in this frame. Give a visual name, one foreground point inside each object, and a confidence from 0 to 1. Use normalized coordinates from 0 to 1. Do not identify an object that is not visually distinct. Do not assume the objects are LEGO."},
            {"type": "input_image", "image_url": _data_url(path)},
        ]
        body = responses_json(self.settings, content=content, name="assembly_object_suggestions", schema=SUGGESTION_SCHEMA)
        suggestions: list[AnnotationSuggestion] = []
        next_id = max(part_ids or {0}) + 1
        name_counts: dict[str, int] = {}
        for item in body.get("suggestions", []):
            try:
                name = str(item["name"]).strip()
                point = normalized_point_to_pixels(PointPrompt.model_validate(item["point"]), *dimensions)
                confidence = float(item["confidence"])
            except (KeyError, TypeError, ValueError):
                raise AppError(502, "MODEL_INVALID_OUTPUT", "The piece suggestion model returned invalid data.") from None
            if not name:
                continue
            key = name.casefold()
            name_counts[key] = name_counts.get(key, 0) + 1
            if name_counts[key] > 1:
                name = f"{name} {name_counts[key]}"
            suggestions.append(AnnotationSuggestion(partId=next_id, name=name, frameIndex=0, point=point, confidence=confidence))
            next_id += 1
        return suggestions


VERIFICATION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["coverage", "findings", "proposedChanges"],
    "properties": {
        "coverage": {"type": "number", "minimum": 0, "maximum": 1},
        "findings": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["findingId", "stepId", "kind", "rationale", "evidenceFrameIds", "suggestedText", "intervalId", "disposition", "movingPartId", "receivingPartId"], "properties": {
            "findingId": {"type": "string"}, "stepId": {"type": ["string", "null"]}, "kind": {"type": "string", "enum": ["supported", "correction", "uncertain", "missing", "rejected"]}, "rationale": {"type": "string"}, "evidenceFrameIds": {"type": "array", "items": {"type": "string"}}, "suggestedText": {"type": ["string", "null"]}, "intervalId": {"type": ["string", "null"]}, "disposition": {"type": "string", "enum": ["supported", "merged", "rejected", "unresolved"]}, "movingPartId": {"type": ["integer", "null"]}, "receivingPartId": {"type": ["integer", "null"]},
        }}},
        "proposedChanges": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["changeId", "kind", "sourceStepIds", "text", "frameId", "evidenceFrameIds", "targetIndex", "sourceEventIds", "movingPartId", "receivingPartId", "uncertainty"], "properties": {
            "changeId": {"type": "string"}, "kind": {"type": "string", "enum": ["rewrite", "merge", "remove", "reorder", "insert"]}, "sourceStepIds": {"type": "array", "items": {"type": "string"}}, "text": {"type": ["string", "null"]}, "frameId": {"type": ["string", "null"]}, "evidenceFrameIds": {"type": "array", "items": {"type": "string"}}, "targetIndex": {"type": ["integer", "null"], "minimum": 0}, "sourceEventIds": {"type": "array", "items": {"type": "string"}}, "movingPartId": {"type": ["integer", "null"]}, "receivingPartId": {"type": ["integer", "null"]}, "uncertainty": {"type": ["string", "null"]},
        }}},
    },
}


class GuideVerifier:
    def __init__(self, settings: Settings):
        self.settings = settings

    def verify(self, guide: Guide, evidence: EvidenceBundle, *, revision: int, review_kind: str = "initial") -> GuideVerification:
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": REVIEW_POLICY + f"This is the {review_kind} review pass. Check this complete editable guide against the supplied piece, tracking, event, and image evidence. Findings must reference only supplied step and frame IDs. Proposed changes are suggestions for one server-controlled correction pass. For recovery, focus on the supplied neighborhood frames and distinguish inadequate evidence from positive rejection."},
            {"type": "input_text", "text": json.dumps({"revision": revision, "guide": guide.model_dump(), "evidence": evidence.context}, ensure_ascii=False)},
        ]
        for frame in evidence.frames:
            path = evidence.paths[evidence.frames.index(frame)]
            content.append({"type": "input_text", "text": f"Evidence frame {frame.frameId} at {frame.timestampSeconds:.2f}s"})
            content.append({"type": "input_image", "image_url": _data_url(path)})
        for attempt in range(2):
            try:
                body = responses_json(self.settings, content=content, name="assembly_guide_verification", schema=VERIFICATION_SCHEMA)
                result = GuideVerification(status="completed", revision=revision, originalGuide=guide, **body)
                return validate_verification(result, guide, evidence)
            except AppError as exc:
                if exc.code != "MODEL_INVALID_OUTPUT" or attempt:
                    raise
                content.append({"type": "input_text", "text": "Correction request: return the same review with only known step IDs and supplied evidence frame IDs; fix the validation errors and return the schema object only."})
            except (TypeError, ValueError) as exc:
                if attempt:
                    raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide verification model returned invalid data.") from exc
                content.append({"type": "input_text", "text": "Correction request: fix the structured output validation errors and return the schema object only."})
        raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide verification model returned invalid structured output.")


def validate_verification(result: GuideVerification, guide: Guide, evidence: EvidenceBundle, *, require_action_provenance: bool = False) -> GuideVerification:
    step_ids = {step.stepId for step in guide.steps}
    frame_ids = {frame.frameId for frame in evidence.frames}
    context_events = {item.get("eventId") for item in evidence.context.get("events", []) if item.get("eventId")}
    context_actions = {item.get("actionId") for item in evidence.context.get("actions", []) if item.get("actionId")}
    context_intervals = context_events | context_actions
    context_parts = {int(item["partId"]) for item in evidence.context.get("pieces", []) if item.get("partId") is not None}
    if result.revision < 0 or not 0 <= result.coverage <= 1:
        raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide verification result has an invalid coverage value.")
    finding_ids: set[str] = set()
    for finding in result.findings:
        if finding.findingId in finding_ids:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide verification contains duplicate finding IDs.")
        finding_ids.add(finding.findingId)
        if finding.stepId is not None and finding.stepId not in step_ids:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide verification referenced an unknown step.")
        if not set(finding.evidenceFrameIds).issubset(frame_ids):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide verification referenced an unavailable frame.")
        if finding.intervalId is not None and context_intervals and finding.intervalId not in context_intervals:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide verification referenced an unknown interval.")
        if context_parts and any(part_id not in context_parts for part_id in (finding.movingPartId, finding.receivingPartId) if part_id is not None):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide verification referenced an unknown part.")
    covered: set[str] = set()
    change_ids: set[str] = set()
    changed_events: set[str] = set()
    changed_actions: set[str] = set()
    for change in result.proposedChanges:
        if change.changeId in change_ids:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide verification contains duplicate change IDs.")
        change_ids.add(change.changeId)
        if change.kind == "insert" and change.sourceStepIds:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "Inserted guide steps must not claim existing source steps.")
        if change.kind != "insert" and not change.sourceStepIds:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "Guide corrections must identify their source steps.")
        if require_action_provenance and change.kind in {"rewrite", "merge", "insert", "reorder"} and not change.sourceActionIds:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "Action-first corrections must preserve source action provenance.")
        if not set(change.sourceStepIds).issubset(step_ids) or covered.intersection(change.sourceStepIds):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide verification contains conflicting step references.")
        covered.update(change.sourceStepIds)
        if change.frameId is not None and change.frameId not in frame_ids:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide correction referenced an unavailable frame.")
        if not set(change.evidenceFrameIds).issubset(frame_ids):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide correction referenced an unavailable frame.")
        if context_events and not set(change.sourceEventIds).issubset(context_events):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide correction referenced an unknown event.")
        if context_actions and not set(change.sourceActionIds).issubset(context_actions):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide correction referenced an unknown action.")
        if changed_events.intersection(change.sourceEventIds):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide corrections contain duplicate event references.")
        changed_events.update(change.sourceEventIds)
        if changed_actions.intersection(change.sourceActionIds):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide corrections contain duplicate action references.")
        changed_actions.update(change.sourceActionIds)
        if context_parts and any(part_id not in context_parts for part_id in (change.movingPartId, change.receivingPartId) if part_id is not None):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide correction referenced an unknown part.")
        if change.kind in {"rewrite", "merge", "insert"} and not (change.text and change.text.strip()):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "Text corrections must contain non-empty instructions.")
    return result


def apply_verification_changes(guide: Guide, result: GuideVerification) -> Guide:
    steps = list(guide.steps)
    by_id = {step.stepId: step for step in steps}
    for change in result.proposedChanges:
        source = [by_id[item] for item in change.sourceStepIds if item in by_id]
        if change.kind == "remove":
            steps = [step for step in steps if step.stepId not in change.sourceStepIds]
        elif change.kind == "rewrite" and source and change.text and change.frameId:
            original = source[0]
            replacement = original.model_copy(update={
                "text": change.text.strip(),
                "frameId": change.frameId,
                "evidenceFrameIds": change.evidenceFrameIds,
                "sourceEventIds": change.sourceEventIds,
                "sourceActionIds": change.sourceActionIds,
                "movingPartId": change.movingPartId,
                "receivingPartId": change.receivingPartId,
                "uncertainty": change.uncertainty,
            })
            steps = [replacement if step.stepId == original.stepId else step for step in steps]
        elif change.kind == "merge" and source and change.text and change.frameId:
            first = source[0].model_copy(update={
                "text": change.text.strip(),
                "frameId": change.frameId,
                "evidenceFrameIds": change.evidenceFrameIds,
                "sourceEventIds": change.sourceEventIds,
                "sourceActionIds": change.sourceActionIds,
                "movingPartId": change.movingPartId,
                "receivingPartId": change.receivingPartId,
                "uncertainty": change.uncertainty,
            })
            steps = [first if step.stepId == source[0].stepId else step for step in steps if step.stepId not in change.sourceStepIds[1:]]
        elif change.kind == "reorder" and source:
            moved = [step for step in steps if step.stepId in change.sourceStepIds]
            steps = [step for step in steps if step.stepId not in change.sourceStepIds]
            target_index = min(change.targetIndex if change.targetIndex is not None else len(steps), len(steps))
            steps[target_index:target_index] = moved
        elif change.kind == "insert" and change.text and change.frameId:
            inserted = Guide.model_validate({"title": guide.title, "steps": [{
                "stepId": change.changeId,
                "text": change.text,
                "frameId": change.frameId,
                "evidenceFrameIds": change.evidenceFrameIds,
                "sourceEventIds": change.sourceEventIds,
                "sourceActionIds": change.sourceActionIds,
                "movingPartId": change.movingPartId,
                "receivingPartId": change.receivingPartId,
                "uncertainty": change.uncertainty,
            }]}).steps[0]
            steps.insert(min(change.targetIndex if change.targetIndex is not None else len(steps), len(steps)), inserted)
    if not steps:
        return guide
    return guide.model_copy(update={"steps": steps})
