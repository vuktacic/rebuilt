from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import tempfile
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable, Protocol

from .action_timeline import (
    EvidenceWindow,
    reverse_reversible_actions,
    timeline_evidence,
    timeline_evidence_batches,
    timeline_windows,
    validate_timeline,
    reference_bundle,
)
from .config import Settings
from .errors import AppError
from .models import (
    ActionTimeline,
    AnnotationSuggestion,
    Frame,
    Guide,
    GuideVerification,
    PartAnnotation,
    PointPrompt,
    TimelineAction,
)
from .post_processing import (
    EvidenceBundle,
    VERIFICATION_SCHEMA,
    _image_dimensions,
    normalized_point_to_pixels,
    responses_json,
    validate_verification,
)


ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["actions", "unresolvedIntervals"],
    "properties": {
        "actions": {"type": "array", "items": {"$ref": "#/$defs/action"}},
        "unresolvedIntervals": {"type": "array", "items": {"$ref": "#/$defs/action"}},
    },
    "$defs": {
        "action": {
            "type": "object", "additionalProperties": False,
            "required": [
                "actionId", "startTimestampSeconds", "endTimestampSeconds", "actionType", "movingPartId",
                "receivingPartId", "beforeRequestedTimestampSeconds", "afterRequestedTimestampSeconds",
                "evidenceRequestedTimestampSeconds", "relationshipBefore", "relationshipAfter", "uncertainty", "evidence",
            ],
            "properties": {
                "actionId": {"type": "string"},
                "startTimestampSeconds": {"type": "number", "minimum": 0},
                "endTimestampSeconds": {"type": "number", "minimum": 0},
                "actionType": {"type": "string", "enum": ["attach", "detach", "move", "separate", "unknown"]},
                "movingPartId": {"type": ["integer", "null"]}, "receivingPartId": {"type": ["integer", "null"]},
                "beforeFrameId": {"type": ["string", "null"]}, "afterFrameId": {"type": ["string", "null"]},
                "beforeRequestedTimestampSeconds": {"type": "number", "minimum": 0}, "afterRequestedTimestampSeconds": {"type": "number", "minimum": 0},
                "evidenceRequestedTimestampSeconds": {"type": "array", "items": {"type": "number", "minimum": 0}},
                "evidenceFrameIds": {"type": "array", "items": {"type": "string"}},
                "relationshipBefore": {"type": "string", "enum": ["attached", "separate", "unknown"]},
                "relationshipAfter": {"type": "string", "enum": ["attached", "separate", "unknown"]},
                "uncertainty": {"type": ["string", "null"]}, "evidence": {"type": "string"},
            },
        },
    },
}

GUIDE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["title", "steps"],
    "properties": {
        "title": {"type": "string"}, "steps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["stepId", "text", "frameId", "uncertainty", "sourceActionIds", "evidenceFrameIds", "kind", "movingPartId", "receivingPartId"],
            "properties": {
                "stepId": {"type": "string"}, "text": {"type": "string"}, "frameId": {"type": "string"},
                "uncertainty": {"type": ["string", "null"]}, "sourceActionIds": {"type": "array", "items": {"type": "string"}},
                "evidenceFrameIds": {"type": "array", "items": {"type": "string"}}, "kind": {"type": "string", "enum": ["instruction", "review"]},
                "movingPartId": {"type": ["integer", "null"]}, "receivingPartId": {"type": ["integer", "null"]},
            },
        }},
    },
}

ACTION_VERIFICATION_SCHEMA = {
    **VERIFICATION_SCHEMA,
    "properties": {
        **VERIFICATION_SCHEMA["properties"],
        "proposedChanges": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": [
            "changeId", "kind", "sourceStepIds", "text", "frameId", "evidenceFrameIds", "targetIndex", "sourceEventIds", "sourceActionIds", "movingPartId", "receivingPartId", "uncertainty",
        ], "properties": {
            "changeId": {"type": "string"}, "kind": {"type": "string", "enum": ["rewrite", "merge", "remove", "reorder", "insert"]},
            "sourceStepIds": {"type": "array", "items": {"type": "string"}}, "text": {"type": ["string", "null"]},
            "frameId": {"type": ["string", "null"]}, "evidenceFrameIds": {"type": "array", "items": {"type": "string"}},
            "targetIndex": {"type": ["integer", "null"]}, "sourceEventIds": {"type": "array", "items": {"type": "string"}},
            "sourceActionIds": {"type": "array", "items": {"type": "string"}}, "movingPartId": {"type": ["integer", "null"]},
            "receivingPartId": {"type": ["integer", "null"]}, "uncertainty": {"type": ["string", "null"]},
        }}},
    },
}


class PipelineProvider(Protocol):
    pipeline_id: str

    def suggest_inventory(self, frame: Frame, path: Path, *, part_ids: set[int]) -> list[AnnotationSuggestion]: ...

    def extract_timeline(self, frames: list[Frame], paths: list[Path], annotations: list[PartAnnotation], video_path: Path | None = None) -> ActionTimeline: ...

    def draft_guide(self, frames: list[Frame], paths: list[Path], annotations: list[PartAnnotation], timeline: ActionTimeline) -> Guide: ...

    def verify_guide(self, guide: Guide, evidence: EvidenceBundle, *, revision: int, review_kind: str) -> GuideVerification: ...

    def recover(self, frames: list[Frame], paths: list[Path], annotations: list[PartAnnotation], timeline: ActionTimeline, findings: list[Any], video_path: Path | None = None, guide: Guide | None = None) -> ActionTimeline | None: ...

    def close(self) -> None: ...


def _image_content(path: Path) -> dict[str, Any]:
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}", "detail": "high"}


def _window_frames(frames: list[Frame], paths: list[Path], window: EvidenceWindow, annotations: list[PartAnnotation]) -> tuple[list[Frame], list[Path]]:
    required = list(dict.fromkeys([
        0,
        len(frames) - 1,
        *(annotation.frameIndex for annotation in annotations),
    ]))
    required = [index for index in required if 0 <= index < len(frames)]
    if len(required) > 24:
        raise AppError(422, "IMAGE_LIMIT", "The approved inventory references more than the provider image request limit.")
    candidates = [index for index in window.frameIndexes if index not in required and 0 <= index < len(frames)]
    slots = 24 - len(required)
    if len(candidates) > slots:
        positions = [round(index * (len(candidates) - 1) / max(1, slots - 1)) for index in range(slots)] if slots else []
        candidates = list(dict.fromkeys(candidates[position] for position in positions))
    indexes = list(dict.fromkeys([*required, *candidates]))
    indexes.sort(key=lambda index: frames[index].timestampSeconds)
    return [frames[index] for index in indexes], [paths[index] for index in indexes]


def _recovery_targets(timeline: ActionTimeline, *, max_windows: int, window_seconds: float, duration: float, findings: list[Any] | None = None, guide: Guide | None = None) -> list[TimelineAction]:
    priority_by_id: dict[str, int] = {}
    for finding in findings or []:
        kind = getattr(finding, "kind", None)
        disposition = getattr(finding, "disposition", None)
        if kind not in {"missing", "uncertain", "rejected", "correction"} and disposition not in {"rejected", "unresolved"}:
            continue
        interval_id = getattr(finding, "intervalId", None)
        if interval_id:
            priority_by_id[interval_id] = min(priority_by_id.get(interval_id, 99), {"missing": 0, "uncertain": 1, "correction": 2, "rejected": 2}.get(kind, 3))
    finding_ids = set(priority_by_id)
    if guide is not None:
        step_ids = {getattr(finding, "stepId", None) for finding in findings or []}
        step_ids.discard(None)
        for step in guide.steps:
            if step.stepId in step_ids:
                for action_id in step.sourceActionIds:
                    finding_ids.add(action_id)
                    priority_by_id[action_id] = min(priority_by_id.get(action_id, 99), 1)
    all_actions = [*timeline.actions, *timeline.unresolvedIntervals]
    prioritized = [action for action in all_actions if action.actionId in finding_ids]
    remaining = [action for action in all_actions if action.actionId not in finding_ids and (action.uncertainty or action in timeline.unresolvedIntervals)]
    candidates = [*prioritized, *remaining]
    candidates.sort(key=lambda action: (priority_by_id.get(action.actionId, 10), action.startTimestampSeconds, action.endTimestampSeconds, action.actionId))
    merged: list[TimelineAction] = []
    for target in candidates:
        center = (target.startTimestampSeconds + target.endTimestampSeconds) / 2
        start = max(0.0, center - window_seconds / 2)
        end = min(duration, start + window_seconds)
        if merged and start <= merged[-1].endTimestampSeconds and max(merged[-1].endTimestampSeconds, end) - min(merged[-1].startTimestampSeconds, start) <= window_seconds:
            merged[-1] = merged[-1].model_copy(update={
                "startTimestampSeconds": min(merged[-1].startTimestampSeconds, start),
                "endTimestampSeconds": max(merged[-1].endTimestampSeconds, end),
                "uncertainty": merged[-1].uncertainty or target.uncertainty,
            })
        else:
            merged.append(target.model_copy(update={"startTimestampSeconds": start, "endTimestampSeconds": end}))
    return merged[:max_windows]


class GPTActionProvider:
    pipeline_id = "gpt_targeted_sam2"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._artifact_writer: Callable[[str, Any], None] | None = None
        self._artifact_counter = 0

    def set_artifact_writer(self, writer: Callable[[str, Any], None] | None) -> None:
        self._artifact_writer = writer

    def _record_artifact(self, name: str, payload: Any) -> None:
        if self._artifact_writer is None:
            return
        self._artifact_counter += 1
        stem, dot, suffix = name.rpartition(".")
        safe_name = f"{stem or name}-{self._artifact_counter:04d}{('.' + suffix) if dot else ''}"
        self._artifact_writer(safe_name, payload)

    @staticmethod
    def _image_manifest(frames: list[Frame], paths: list[Path], roles: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
        roles = roles or {}
        manifest: list[dict[str, Any]] = []
        for frame, path in zip(frames, paths, strict=True):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            dimensions = _image_dimensions(path)
            manifest.append({
                "frameId": frame.frameId,
                "sourceTimestampSeconds": frame.timestampSeconds,
                "dimensions": {"width": dimensions[0], "height": dimensions[1]} if dimensions else None,
                "sha256": digest,
                "roles": [role for role, frame_ids in roles.items() if frame.frameId in frame_ids],
                "submitted": True,
            })
        return manifest

    def _record_request(self, stage: str, frames: list[Frame], paths: list[Path], *, roles: dict[str, list[str]] | None = None, payload: dict[str, Any] | None = None, raw_output: Any = None) -> None:
        manifest = self._image_manifest(frames, paths, roles)
        self._record_artifact(f"submissions/{stage}.json", {"schemaVersion": "submission-manifest-v1", "stage": stage, "images": manifest, **(payload or {})})
        contact_sheet = "<html><body><h1>Submitted evidence</h1>" + "".join(
            f'<figure><figcaption>{item["frameId"]} · source {item["sourceTimestampSeconds"]:.3f}s</figcaption><img src="../../frames/{item["frameId"]}.jpg" /></figure>'
            for item in manifest
        ) + "</body></html>"
        self._record_artifact(f"contact-sheets/{stage}.html", contact_sheet)
        if raw_output is not None:
            self._record_artifact(f"provider-output/{stage}.json", raw_output)

    def suggest_inventory(self, frame: Frame, path: Path, *, part_ids: set[int]) -> list[AnnotationSuggestion]:
        dimensions = _image_dimensions(path)
        if dimensions is None:
            raise AppError(422, "IMAGE_INVALID", "The selected frame dimensions could not be read.")
        schema = {
            "type": "object", "additionalProperties": False, "required": ["suggestions"], "properties": {
                "suggestions": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "point", "confidence"], "properties": {
                    "name": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "point": {"type": "object", "additionalProperties": False, "required": ["x", "y"], "properties": {"x": {"type": "number", "minimum": 0, "maximum": 1}, "y": {"type": "number", "minimum": 0, "maximum": 1}}},
                }}}
            },
        }
        body = responses_json(self.settings, content=[
            {"type": "input_text", "text": "List each clearly separated assembly object visible in this reference image. Return one normalized foreground point per object. Do not assume LEGO."},
            _image_content(path),
        ], name="assembly_inventory", schema=schema)
        width, height = dimensions
        suggestions: list[AnnotationSuggestion] = []
        next_id = max(part_ids or {0}) + 1
        for item in body.get("suggestions", []):
            try:
                name = str(item["name"]).strip()
                point = normalized_point_to_pixels(PointPrompt(x=float(item["point"]["x"]), y=float(item["point"]["y"])), width, height)
                confidence = float(item["confidence"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AppError(502, "MODEL_INVALID_OUTPUT", "The inventory model returned invalid coordinates.") from exc
            if name:
                suggestions.append(AnnotationSuggestion(partId=next_id, name=name, frameIndex=0, point=point, confidence=confidence))
                next_id += 1
        return suggestions

    def extract_timeline(self, frames: list[Frame], paths: list[Path], annotations: list[PartAnnotation], video_path: Path | None = None) -> ActionTimeline:
        parts: list[ActionTimeline] = []
        for window in timeline_windows(frames, window_seconds=self.settings.gpt_timeline_window_seconds, overlap_seconds=self.settings.gpt_timeline_overlap_seconds):
            window_frames, window_paths = _window_frames(frames, paths, window, annotations)
            submitted_ids = {frame.frameId for frame in window_frames}
            identity = reference_bundle(frames, paths, annotations)
            content: list[dict[str, Any]] = [{"type": "input_text", "text": self._timeline_prompt(window, annotations, identity)}]
            for frame, path in zip(window_frames, window_paths, strict=True):
                roles = []
                if frame.frameId in {frames[annotation.frameIndex].frameId for annotation in annotations if 0 <= annotation.frameIndex < len(frames)}:
                    roles.append("identity_reference")
                if frame.frameId in {frames[0].frameId, frames[-1].frameId}:
                    roles.append("global_context")
                content.append({"type": "input_text", "text": f"Submitted image {frame.frameId} at source timestamp {frame.timestampSeconds:.3f}s ({', '.join(roles) or 'window_context'})."})
                content.append(_image_content(path))
            body = responses_json(self.settings, content=content, name="action_timeline", schema=ACTION_SCHEMA)
            parsed = ActionTimeline.model_validate(body)
            self._record_request(
                f"timeline-{window.windowId}",
                window_frames,
                window_paths,
                payload={"window": {"windowId": window.windowId, "startTimestampSeconds": window.startTimestampSeconds, "endTimestampSeconds": window.endTimestampSeconds}, "referenceBundle": identity, "timebase": "absolute-canonical-seconds", "samplingFps": self.settings.gpt_timeline_fps},
                raw_output=body,
            )
            parsed = validate_timeline(parsed, frames, annotations, submitted_frame_ids=submitted_ids, window_bounds=(window.startTimestampSeconds, window.endTimestampSeconds))
            parsed = parsed.model_copy(update={"actions": [action.model_copy(update={"sourceWindowId": window.windowId, "observationIds": [action.actionId]}) for action in parsed.actions], "unresolvedIntervals": [action.model_copy(update={"sourceWindowId": window.windowId, "observationIds": [action.actionId]}) for action in parsed.unresolvedIntervals]})
            parts.append(parsed)
        from .action_timeline import merge_timelines
        return merge_timelines(parts, duration_seconds=frames[-1].timestampSeconds if frames else 0)

    @staticmethod
    def _timeline_prompt(window: EvidenceWindow, annotations: list[PartAnnotation], identity: list[dict[str, Any]]) -> str:
        inventory = ", ".join(f"part {item.partId}={item.name}" for item in annotations) or "no approved inventory yet"
        return (
            "Reconstruct only visible physical actions in source-video order for this fixed-camera assembly recording. "
            "The recording is assumed to be disassembly, so describe what happens in the source chronology first. "
            "Keep ordinary movement of an unchanged assembly separate from attach/detach actions. Receiving bases are explicit participants. "
            "Use unknown action intervals when the manipulation is obscured; do not invent connections, IDs, local frame IDs, or timestamps outside the submitted window. "
            "Return absolute canonical-video timestamps for each interval and for before/after/evidence states; the server resolves those timestamps to submitted images. "
            f"This window is {window.startTimestampSeconds:.2f}-{window.endTimestampSeconds:.2f}s. Approved inventory: {inventory}. "
            f"Identity references, including approved pixel coordinates and reference image hashes: {json.dumps(identity, ensure_ascii=False)}"
        )

    def draft_guide(self, frames: list[Frame], paths: list[Path], annotations: list[PartAnnotation], timeline: ActionTimeline) -> Guide:
        guides: list[Guide] = []
        for evidence in timeline_evidence_batches(frames, paths, timeline, annotations, max_images=self.settings.review_max_images):
            content: list[dict[str, Any]] = [{"type": "input_text", "text": (
                "Draft an editable assembly guide from the server-derived assembly mapping. The mapping fixes source IDs, participants, screenshots, and assembly order. "
                "Reverse only supported reversible actions exactly once. Link every generated step to sourceActionIds. Every review mapping must remain an inline review step. "
                "Do not add, remove, reorder, or regroup source actions beyond the supplied batch. "
                + json.dumps({"timeline": timeline.model_dump(), "evidence": evidence.context, "batchActionIds": evidence.context.get("batch", {}).get("actionIds", [])}, ensure_ascii=False)
            )}]
            content.extend(self._images(evidence))
            body = responses_json(self.settings, content=content, name="action_guide", schema=GUIDE_SCHEMA)
            self._record_request("draft", evidence.frames, evidence.paths, payload={"evidence": evidence.context}, raw_output=body)
            guides.append(Guide.model_validate(body))
        return Guide(title=guides[0].title if guides else "Build needs review", steps=[step for guide in guides for step in guide.steps])

    def verify_guide(self, guide: Guide, evidence: EvidenceBundle, *, revision: int, review_kind: str) -> GuideVerification:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": (
            f"Independently review this {review_kind} guide against the whole action timeline and all supplied images. "
            "Check coverage, receiving parts, order, duplicate actions, and initial/final state consistency. "
            "Use supported, merged, rejected, or unresolved dispositions. Return suggestions only; the server applies at most one correction pass.\n"
            + json.dumps({"revision": revision, "guide": guide.model_dump(), "evidence": evidence.context}, ensure_ascii=False)
        )}]
        content.extend(self._images(evidence))
        body = responses_json(self.settings, content=content, name="action_guide_verification", schema=ACTION_VERIFICATION_SCHEMA)
        self._record_request(f"verification-{review_kind}-{revision}", evidence.frames, evidence.paths, payload={"evidence": evidence.context, "reviewKind": review_kind, "revision": revision}, raw_output=body)
        result = GuideVerification(status="completed", revision=revision, originalGuide=guide, **body)
        return validate_verification(result, guide, evidence, require_action_provenance=True)

    @staticmethod
    def _images(evidence: EvidenceBundle) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for frame, path in zip(evidence.frames, evidence.paths, strict=True):
            content.append({"type": "input_text", "text": f"Evidence frame {frame.frameId} at {frame.timestampSeconds:.2f}s"})
            content.append(_image_content(path))
        return content

    def recover(self, frames: list[Frame], paths: list[Path], annotations: list[PartAnnotation], timeline: ActionTimeline, findings: list[Any], video_path: Path | None = None, guide: Guide | None = None) -> ActionTimeline | None:
        if not self.settings.targeted_sam2_enabled:
            return None
        targets = _recovery_targets(
            timeline,
            max_windows=self.settings.recovery_max_windows,
            window_seconds=self.settings.recovery_window_seconds,
            duration=frames[-1].timestampSeconds if frames else 0.0,
            findings=findings,
            guide=guide,
        )
        if not targets:
            return None
        # The bounded recovery adapter deliberately keeps SAM2 optional. It
        # returns supporting observations to the same GPT action parser; it
        # never makes an attachment decision locally.
        try:
            from .vision import Sam2BackwardVisionAnalyzer
            analyzer = Sam2BackwardVisionAnalyzer(self.settings)
        except Exception:
            return None
        recovered_parts: list[ActionTimeline] = []
        for index, target in enumerate(targets):
            center = (target.startTimestampSeconds + target.endTimestampSeconds) / 2
            start = max(0.0, center - self.settings.recovery_window_seconds / 2)
            end = min(frames[-1].timestampSeconds if frames else 0.0, start + self.settings.recovery_window_seconds)
            indexes = tuple(i for i, frame in enumerate(frames) if start <= frame.timestampSeconds <= end)
            if not indexes:
                continue
            local_frames = [frames[i] for i in indexes]
            local_paths = [paths[i] for i in indexes]
            # No trustworthy approved anchor means review remains unresolved;
            # the implementation intentionally does not expand to full-video tracking.
            selected = [annotation for annotation in annotations if any(abs(annotation.frameIndex - i) <= 1 for i in indexes)][:3]
            if len(selected) < 3:
                available_parts = [annotation for annotation in annotations if annotation.partId not in {item.partId for item in selected}]
                clear_index = indexes[len(indexes) // 2]
                clear_frame = frames[clear_index]
                clear_path = paths[clear_index]
                anchor_schema = {"type": "object", "additionalProperties": False, "required": ["anchors"], "properties": {"anchors": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["partId", "frameId", "point"], "properties": {"partId": {"type": "integer"}, "frameId": {"type": "string"}, "point": {"type": "object", "additionalProperties": False, "required": ["x", "y"], "properties": {"x": {"type": "number", "minimum": 0, "maximum": 1}, "y": {"type": "number", "minimum": 0, "maximum": 1}}}}}}}}
                try:
                    anchor_body = responses_json(self.settings, content=[
                        {"type": "input_text", "text": "Locate up to three approved inventory objects in this clear frame for targeted tracking. Return normalized foreground points and only approved part IDs. These are provisional anchors, not identity proof. Inventory: " + json.dumps([item.model_dump() for item in available_parts])},
                        _image_content(clear_path),
                    ], name="targeted_sam2_anchors", schema=anchor_schema)
                    dimensions = _image_dimensions(clear_path)
                    if dimensions:
                        width, height = dimensions
                        for item in anchor_body.get("anchors", []):
                            annotation = next((candidate for candidate in available_parts if candidate.partId == int(item["partId"])), None)
                            if annotation is None or item["frameId"] != clear_frame.frameId:
                                continue
                            point = normalized_point_to_pixels(PointPrompt(x=float(item["point"]["x"]), y=float(item["point"]["y"])), width, height)
                            selected.append(annotation.model_copy(update={
                                "frameIndex": clear_index,
                                "points": [point],
                                "labels": [1],
                                "provisionalAnchor": True,
                            }))
                            if len(selected) >= 3:
                                break
                except (AppError, KeyError, TypeError, ValueError):
                    pass
            if not selected:
                continue
            local_annotations = [annotation.model_copy(update={"frameIndex": min(range(len(indexes)), key=lambda j: abs(indexes[j] - annotation.frameIndex))}) for annotation in selected]
            try:
                with tempfile.TemporaryDirectory(prefix="rebuilt-targeted-sam2-") as temporary:
                    window_dir = Path(temporary)
                    for local_index, path in enumerate(local_paths):
                        (window_dir / f"frame-{local_index:06d}.jpg").symlink_to(path.resolve())
                    result = analyzer.analyze_annotated(window_dir, len(local_frames), f"recovery-{index + 1}", local_annotations)
            except Exception:
                continue
            local_timeline = ActionTimeline(actions=[target], sourceDirection=timeline.sourceDirection)
            context = timeline_evidence(local_frames, local_paths, local_timeline, local_annotations, max_images=self.settings.review_max_images)
            mapped_tracks = []
            for track in result.part_tracks or []:
                observations = [observation.model_copy(update={
                    "frameIndex": indexes[observation.frameIndex] if 0 <= observation.frameIndex < len(indexes) else observation.frameIndex,
                    "timestampSeconds": frames[indexes[observation.frameIndex]].timestampSeconds if 0 <= observation.frameIndex < len(indexes) else observation.timestampSeconds,
                }) for observation in track.observations]
                mapped_tracks.append(track.model_copy(update={
                    "observations": observations,
                    "attachmentStartFrame": indexes[track.attachmentStartFrame] if track.attachmentStartFrame is not None and track.attachmentStartFrame < len(indexes) else track.attachmentStartFrame,
                    "attachmentEndFrame": indexes[track.attachmentEndFrame] if track.attachmentEndFrame is not None and track.attachmentEndFrame < len(indexes) else track.attachmentEndFrame,
                }))
            context.context["targetedRecovery"] = {"windowStartSeconds": start, "windowEndSeconds": end, "tracks": [track.model_dump() for track in mapped_tracks]}
            body = responses_json(self.settings, content=[
                {"type": "input_text", "text": "Re-evaluate only the targeted ambiguous source interval using the measured SAM2 observations as supporting evidence. Return corrected actions, or preserve unknown if identity or receiving part remains unsupported." + json.dumps(context.context, ensure_ascii=False)},
                *self._images(context),
            ], name="targeted_action_recovery", schema=ACTION_SCHEMA)
            self._record_request(f"recovery-{index + 1:04d}", local_frames, local_paths, payload={"windowStartSeconds": start, "windowEndSeconds": end, "targetActionId": target.actionId, "evidence": context.context}, raw_output=body)
            recovered_parts.append(validate_timeline(ActionTimeline.model_validate(body), local_frames, local_annotations, submitted_frame_ids={frame.frameId for frame in local_frames}, window_bounds=(start, end)))
        if not recovered_parts:
            return None
        from .action_timeline import merge_timelines
        return merge_timelines(recovered_parts, duration_seconds=frames[-1].timestampSeconds if frames else 0)

    def close(self) -> None:
        return None


class GeminiVideoProvider(GPTActionProvider):
    pipeline_id = "gemini_video"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._client: Any | None = None
        self._file: Any | None = None
        self._owned_file_name: str | None = None

    def suggest_inventory(self, frame: Frame, path: Path, *, part_ids: set[int]) -> list[AnnotationSuggestion]:
        """Use Gemini for optional inventory suggestions without uploading the video yet."""
        dimensions = _image_dimensions(path)
        if dimensions is None:
            raise AppError(422, "IMAGE_INVALID", "The selected frame dimensions could not be read.")
        schema = {
            "type": "object", "additionalProperties": False, "required": ["suggestions"], "properties": {
                "suggestions": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "point", "confidence"], "properties": {
                    "name": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "point": {"type": "object", "additionalProperties": False, "required": ["x", "y"], "properties": {"x": {"type": "number", "minimum": 0, "maximum": 1}, "y": {"type": "number", "minimum": 0, "maximum": 1}}},
                }}},
            },
        }
        body = self._interaction([
            {"type": "text", "text": "List each clearly separated assembly object visible in this reference image. Return one normalized foreground point per object. Do not assume LEGO."},
            {"type": "image", "data": base64.b64encode(path.read_bytes()).decode("ascii"), "mime_type": "image/jpeg"},
        ], schema)
        width, height = dimensions
        next_id = max(part_ids or {0}) + 1
        suggestions: list[AnnotationSuggestion] = []
        for item in body.get("suggestions", []):
            try:
                name = str(item["name"]).strip()
                point = normalized_point_to_pixels(PointPrompt(x=float(item["point"]["x"]), y=float(item["point"]["y"])), width, height)
                confidence = float(item["confidence"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AppError(502, "MODEL_INVALID_OUTPUT", "Gemini returned invalid inventory coordinates.") from exc
            if name:
                suggestions.append(AnnotationSuggestion(partId=next_id, name=name, frameIndex=0, point=point, confidence=confidence))
                next_id += 1
        return suggestions

    def _retry_pending_cleanup(self, client: Any) -> None:
        cleanup_path = self.settings.data_dir / "gemini-cleanup.json"
        if not cleanup_path.is_file():
            return
        try:
            names = json.loads(cleanup_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        remaining: list[str] = []
        for name in names if isinstance(names, list) else []:
            try:
                client.files.delete(name=name)
            except Exception:
                remaining.append(str(name))
        if remaining:
            cleanup_path.write_text(json.dumps(remaining) + "\n", encoding="utf-8")
        else:
            cleanup_path.unlink(missing_ok=True)

    def _get_client(self) -> Any:
        if not self.settings.gemini_api_key:
            raise AppError(503, "GEMINI_NOT_CONFIGURED", "GEMINI_API_KEY is not configured.")
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise AppError(503, "GEMINI_SDK_MISSING", "Install the optional google-genai dependency for the Gemini pipeline.") from exc
        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        self._retry_pending_cleanup(self._client)
        return self._client

    def prepare_video(self, video_path: Path) -> None:
        client = self._get_client()
        try:
            uploaded = client.files.upload(file=str(video_path))
            self._owned_file_name = getattr(uploaded, "name", None)
            deadline = time.monotonic() + self.settings.gemini_file_timeout_seconds
            current = uploaded
            while str(getattr(getattr(current, "state", None), "name", getattr(current, "state", ""))).upper() in {"PROCESSING", "STATE_UNSPECIFIED", ""}:
                if time.monotonic() >= deadline:
                    raise AppError(504, "GEMINI_FILE_TIMEOUT", "Gemini did not finish processing the uploaded video before the deadline.")
                time.sleep(self.settings.gemini_poll_seconds)
                current = client.files.get(name=self._owned_file_name)
            state = str(getattr(getattr(current, "state", None), "name", getattr(current, "state", ""))).upper()
            if "FAILED" in state:
                raise AppError(502, "GEMINI_FILE_FAILED", "Gemini failed to process the canonical video derivative.")
            self._file = current
        except AppError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise AppError(502, "GEMINI_UPLOAD_FAILED", "Gemini video upload failed.") from exc

    def _video_content(self, *, start: float | None = None, end: float | None = None, fps: float = 1.0) -> dict[str, Any]:
        if self._file is None:
            raise AppError(503, "GEMINI_FILE_MISSING", "The Gemini video was not prepared for this run.")
        uri = getattr(self._file, "uri", None)
        mime_type = getattr(self._file, "mime_type", None) or "video/mp4"
        processing: dict[str, Any] = {"type": "static", "fps": fps}
        if start is not None:
            processing["start_offset"] = f"{start:.3f}s"
        if end is not None:
            processing["end_offset"] = f"{end:.3f}s"
        return {"type": "video", "uri": uri, "mime_type": mime_type, "processing": processing}

    @staticmethod
    def _gemini_images(evidence: EvidenceBundle) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for frame, path in zip(evidence.frames, evidence.paths, strict=True):
            content.append({"type": "text", "text": f"Resolved local evidence image {frame.frameId} at absolute source timestamp {frame.timestampSeconds:.3f}s."})
            content.append({"type": "image", "data": base64.b64encode(path.read_bytes()).decode("ascii"), "mime_type": "image/jpeg"})
        return content

    def _interaction(self, input_content: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        try:
            interaction = client.interactions.create(
                model=self.settings.gemini_model,
                input=input_content,
                response_format={"type": "text", "mime_type": "application/json", "schema": schema},
                tools=[],
                store=False,
            )
            output = getattr(interaction, "output", None) or getattr(interaction, "outputs", None) or []
            last = output[-1] if output else interaction
            text = getattr(last, "text", None) or getattr(interaction, "text", None)
            if text is None and isinstance(last, dict):
                text = last.get("text") or last.get("json")
            if isinstance(text, dict):
                return text
            return json.loads(str(text))
        except AppError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AppError(502, "GEMINI_INVALID_OUTPUT", "Gemini returned invalid structured output.") from exc
        except Exception as exc:
            raise AppError(502, "GEMINI_REQUEST_FAILED", "The Gemini interaction failed.") from exc

    def extract_timeline(self, frames: list[Frame], paths: list[Path], annotations: list[PartAnnotation], video_path: Path | None = None) -> ActionTimeline:
        if video_path is None:
            raise AppError(422, "VIDEO_MISSING", "Gemini requires a canonical video derivative.")
        self.prepare_video(video_path)
        identity = reference_bundle(frames, paths, annotations)
        reference_evidence = timeline_evidence(frames, paths, ActionTimeline(sourceDirection="disassembly"), annotations, max_images=self.settings.review_max_images)
        body = self._interaction([
            {"type": "text", "text": "Extract the complete source-order action timeline from this fixed-camera disassembly recording at one frame per second. Include unresolved manipulation intervals, explicit moving and receiving parts, and absolute source timestamps. Do not invent connections or local frame IDs. The server resolves timestamps to its canonical local frame registry."},
            {"type": "text", "text": json.dumps({"inventory": identity, "durationSeconds": frames[-1].timestampSeconds if frames else 0, "timebase": "absolute-canonical-seconds", "frameRegistry": [{"frameId": frame.frameId, "timestampSeconds": frame.timestampSeconds} for frame in frames]})},
            self._video_content(fps=1.0),
            *self._gemini_images(reference_evidence),
        ], ACTION_SCHEMA)
        self._record_artifact("video-requests/timeline.json", {"sampling": {"type": "static", "fps": 1.0}, "referenceBundle": identity, "timebase": "absolute-canonical-seconds", "frameRegistryCount": len(frames)})
        self._record_artifact("provider-output/gemini-timeline.json", body)
        return validate_timeline(ActionTimeline.model_validate(body), frames, annotations)

    def draft_guide(self, frames: list[Frame], paths: list[Path], annotations: list[PartAnnotation], timeline: ActionTimeline) -> Guide:
        guides: list[Guide] = []
        for evidence in timeline_evidence_batches(frames, paths, timeline, annotations, max_images=self.settings.review_max_images):
            body = self._interaction([
                {"type": "text", "text": "Draft an editable assembly guide from the server-derived assembly mapping. The mapping fixes source IDs, participants, screenshots, and assembly order. Reverse only supported reversible actions exactly once. Link every generated step to sourceActionIds and preserve review mappings as inline review steps. Only address this batch's action IDs."},
                {"type": "text", "text": json.dumps({"timeline": timeline.model_dump(), "evidence": evidence.context, "batchActionIds": evidence.context.get("batch", {}).get("actionIds", [])})},
                self._video_content(fps=1.0),
                *self._gemini_images(evidence),
            ], GUIDE_SCHEMA)
            self._record_artifact("provider-output/gemini-draft.json", body)
            guides.append(Guide.model_validate(body))
        return Guide(title=guides[0].title if guides else "Build needs review", steps=[step for guide in guides for step in guide.steps])

    def verify_guide(self, guide: Guide, evidence: EvidenceBundle, *, revision: int, review_kind: str) -> GuideVerification:
        body = self._interaction([
            {"type": "text", "text": f"Perform an independent {review_kind} review against the whole video and the resolved local evidence images. Check receiving parts, ordering, coverage, duplicates, and initial/final consistency. Return explicit dispositions and no unsupported confident claims."},
            {"type": "text", "text": json.dumps({"revision": revision, "guide": guide.model_dump(), "evidence": evidence.context})},
            self._video_content(fps=1.0),
            *self._gemini_images(evidence),
        ], ACTION_VERIFICATION_SCHEMA)
        self._record_artifact(f"provider-output/gemini-verification-{review_kind}-{revision}.json", body)
        result = GuideVerification(status="completed", revision=revision, originalGuide=guide, **body)
        return validate_verification(result, guide, evidence, require_action_provenance=True)

    def recover(self, frames: list[Frame], paths: list[Path], annotations: list[PartAnnotation], timeline: ActionTimeline, findings: list[Any], video_path: Path | None = None, guide: Guide | None = None) -> ActionTimeline | None:
        targets = _recovery_targets(
            timeline,
            max_windows=self.settings.recovery_max_windows,
            window_seconds=self.settings.recovery_window_seconds,
            duration=frames[-1].timestampSeconds if frames else 0.0,
            findings=findings,
            guide=guide,
        )
        if not targets:
            return None
        recovered: list[ActionTimeline] = []
        for target in targets:
            center = (target.startTimestampSeconds + target.endTimestampSeconds) / 2
            start = max(0.0, center - self.settings.recovery_window_seconds / 2)
            end = min(frames[-1].timestampSeconds if frames else 0, start + self.settings.recovery_window_seconds)
            local_frames = [frame for frame in frames if start <= frame.timestampSeconds <= end]
            if not local_frames:
                continue
            local_paths = [paths[frames.index(frame)] for frame in local_frames]
            local_annotations = [annotation.model_copy(update={"frameIndex": min(range(len(local_frames)), key=lambda index: abs(local_frames[index].timestampSeconds - (annotation.referenceTimestampSeconds if annotation.referenceTimestampSeconds is not None else annotation.frameIndex)))}) for annotation in annotations] if local_frames else []
            local_evidence = timeline_evidence(local_frames, local_paths, ActionTimeline(actions=[target], sourceDirection=timeline.sourceDirection), local_annotations, max_images=self.settings.review_max_images)
            body = self._interaction([
                {"type": "text", "text": f"Re-evaluate only this ambiguous source interval {start:.2f}-{end:.2f}s at four frames per second. Return corrected actions or preserve unknown."},
                {"type": "text", "text": json.dumps({"inventory": reference_bundle(frames, paths, annotations), "targetAction": target.model_dump(), "timebase": "absolute-canonical-seconds"})},
                self._video_content(start=start, end=end, fps=4.0),
                *self._gemini_images(local_evidence),
            ], ACTION_SCHEMA)
            self._record_artifact(f"provider-output/gemini-recovery-{target.actionId}.json", body)
            recovered.append(validate_timeline(ActionTimeline.model_validate(body), local_frames, annotations, submitted_frame_ids={frame.frameId for frame in local_frames}, window_bounds=(start, end)))
        if not recovered:
            return None
        from .action_timeline import merge_timelines
        return merge_timelines(recovered, duration_seconds=frames[-1].timestampSeconds if frames else 0)

    def close(self) -> None:
        if self._client is None or self._owned_file_name is None:
            return
        try:
            self._client.files.delete(name=self._owned_file_name)
        except Exception:
            cleanup_path = self.settings.data_dir / "gemini-cleanup.json"
            cleanup_path.parent.mkdir(parents=True, exist_ok=True)
            names: list[str] = []
            if cleanup_path.is_file():
                try:
                    names = json.loads(cleanup_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    names = []
            if self._owned_file_name not in names:
                names.append(self._owned_file_name)
            cleanup_path.write_text(json.dumps(names) + "\n", encoding="utf-8")
        finally:
            self._owned_file_name = None
            self._file = None


def create_pipeline_provider(pipeline: str, settings: Settings) -> PipelineProvider:
    if pipeline == "gpt_targeted_sam2":
        if not settings.openai_api_key:
            raise AppError(503, "OPENAI_NOT_CONFIGURED", "OPENAI_API_KEY is required for the GPT pipeline.")
        return GPTActionProvider(settings)
    if pipeline == "gemini_video":
        if not settings.gemini_api_key:
            raise AppError(503, "GEMINI_NOT_CONFIGURED", "GEMINI_API_KEY is required for the Gemini pipeline.")
        return GeminiVideoProvider(settings)
    raise AppError(422, "PIPELINE_INVALID", f"Unsupported pipeline: {pipeline}.")
