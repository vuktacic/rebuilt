from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import AppError
from .models import ActionTimeline, AssemblyAction, Frame, PartAnnotation, TimelineAction
from .post_processing import EvidenceBundle, _image_dimensions


@dataclass(frozen=True)
class EvidenceWindow:
    windowId: str
    startTimestampSeconds: float
    endTimestampSeconds: float
    frameIndexes: tuple[int, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_bundle(frames: list[Frame], paths: list[Path], annotations: list[PartAnnotation]) -> list[dict[str, Any]]:
    """Return server-grounded identity references without embedding media in JSON."""
    bundle: list[dict[str, Any]] = []
    for annotation in annotations:
        if not 0 <= annotation.frameIndex < len(frames) or annotation.frameIndex >= len(paths):
            continue
        frame = frames[annotation.frameIndex]
        dimensions = _image_dimensions(paths[annotation.frameIndex])
        bundle.append({
            "partId": annotation.partId,
            "name": annotation.name,
            "referenceFrameId": annotation.referenceFrameId or frame.frameId,
            "referenceTimestampSeconds": annotation.referenceTimestampSeconds if annotation.referenceTimestampSeconds is not None else frame.timestampSeconds,
            "coordinateSpace": annotation.coordinateSpace,
            "point": [point.model_dump() for point in annotation.points],
            "box": annotation.box,
            "imageDimensions": {"width": dimensions[0], "height": dimensions[1]} if dimensions else None,
            "imageSha256": annotation.referenceImageSha256 or _file_sha256(paths[annotation.frameIndex]),
        })
    return bundle


def timeline_windows(
    frames: list[Frame],
    *,
    window_seconds: float = 20.0,
    overlap_seconds: float = 2.0,
) -> list[EvidenceWindow]:
    if not frames:
        return []
    duration = frames[-1].timestampSeconds
    step = max(0.1, window_seconds - min(overlap_seconds, max(0.0, window_seconds - 0.1)))
    windows: list[EvidenceWindow] = []
    start = 0.0
    index = 0
    while start <= duration or not windows:
        end = min(duration, start + window_seconds)
        indexes = tuple(
            frame_index
            for frame_index, frame in enumerate(frames)
            if start <= frame.timestampSeconds <= end
        )
        if indexes:
            windows.append(EvidenceWindow(f"window-{index + 1:04d}", start, end, indexes))
        if end >= duration:
            break
        start += step
        index += 1
    return windows


def validate_timeline(
    timeline: ActionTimeline,
    frames: list[Frame],
    annotations: list[PartAnnotation],
    *,
    submitted_frame_ids: set[str] | None = None,
    window_bounds: tuple[float, float] | None = None,
    sampling_tolerance_seconds: float = 0.75,
) -> ActionTimeline:
    frame_ids = {frame.frameId for frame in frames}
    part_ids = {annotation.partId for annotation in annotations if annotation.partId > 0}
    duration = frames[-1].timestampSeconds if frames else 0.0
    seen: set[str] = set()
    all_actions = [*timeline.actions, *timeline.unresolvedIntervals]
    normalized_actions: list[TimelineAction] = []
    for action in all_actions:
        if action.actionId in seen:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline contains duplicate action IDs.")
        seen.add(action.actionId)
        if action.startTimestampSeconds > action.endTimestampSeconds or action.endTimestampSeconds > duration + 0.001:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline contains an invalid source interval.")
        if action.movingPartId is not None and action.movingPartId not in part_ids:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline referenced an unknown moving part.")
        if action.receivingPartId is not None and action.receivingPartId not in part_ids:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline referenced an unknown receiving part.")
        if action.movingPartId is not None and action.movingPartId == action.receivingPartId:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "An action cannot move a part onto itself.")
        if action.beforeFrameId is not None and action.beforeFrameId not in frame_ids:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline referenced an unavailable before frame.")
        if action.afterFrameId is not None and action.afterFrameId not in frame_ids:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline referenced an unavailable after frame.")
        if not set(action.evidenceFrameIds).issubset(frame_ids):
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline referenced an unavailable evidence frame.")
        if submitted_frame_ids is not None:
            referenced = {frame_id for frame_id in [action.beforeFrameId, action.afterFrameId, *action.evidenceFrameIds] if frame_id}
            if not referenced.issubset(submitted_frame_ids):
                raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline cited an image that was not submitted for this request.")
        if window_bounds is not None:
            lower, upper = window_bounds
            if action.startTimestampSeconds < lower - sampling_tolerance_seconds or action.endTimestampSeconds > upper + sampling_tolerance_seconds:
                raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline cited an interval outside its submitted window.")
        before_requested = action.beforeRequestedTimestampSeconds if action.beforeRequestedTimestampSeconds is not None else action.startTimestampSeconds
        after_requested = action.afterRequestedTimestampSeconds if action.afterRequestedTimestampSeconds is not None else action.endTimestampSeconds
        candidate_frames = [frame for frame in frames if submitted_frame_ids is None or frame.frameId in submitted_frame_ids]
        if not candidate_frames:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline has no submitted frames for evidence resolution.")
        before_frame = next((frame for frame in reversed(candidate_frames) if frame.timestampSeconds <= before_requested + sampling_tolerance_seconds), None)
        after_frame = next((frame for frame in candidate_frames if frame.timestampSeconds >= after_requested - sampling_tolerance_seconds), None)
        if action.beforeFrameId is not None:
            before_frame = next((frame for frame in candidate_frames if frame.frameId == action.beforeFrameId), None)
        if action.afterFrameId is not None:
            after_frame = next((frame for frame in candidate_frames if frame.frameId == action.afterFrameId), None)
        if before_frame is None:
            before_frame = min(candidate_frames, key=lambda frame: abs(frame.timestampSeconds - before_requested))
        if after_frame is None:
            after_frame = min(candidate_frames, key=lambda frame: abs(frame.timestampSeconds - after_requested))
        if before_frame.frameId == after_frame.frameId:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "A physical action cannot use the same submitted image as both before and after evidence.")
        evidence_ids = list(action.evidenceFrameIds)
        evidence_requested = list(action.evidenceRequestedTimestampSeconds)
        for requested in evidence_requested:
            resolved = min(candidate_frames, key=lambda frame: abs(frame.timestampSeconds - requested))
            if resolved.frameId not in evidence_ids:
                evidence_ids.append(resolved.frameId)
        if before_frame.frameId not in evidence_ids:
            evidence_ids.insert(0, before_frame.frameId)
        if after_frame.frameId not in evidence_ids:
            evidence_ids.append(after_frame.frameId)
        errors = [abs(before_frame.timestampSeconds - before_requested), abs(after_frame.timestampSeconds - after_requested)]
        uncertainty = action.uncertainty
        if before_frame.timestampSeconds > action.startTimestampSeconds + sampling_tolerance_seconds or after_frame.timestampSeconds < action.endTimestampSeconds - sampling_tolerance_seconds:
            uncertainty = uncertainty or "Submitted samples do not bracket the full action interval closely enough."
        if action.actionType in {"attach", "detach", "separate"} and action.relationshipBefore == action.relationshipAfter and action.relationshipBefore != "unknown":
            uncertainty = uncertainty or "The submitted states do not demonstrate a relationship change."
        roles = {key: list(value) for key, value in action.evidenceRoles.items()}
        roles.setdefault("before_state", [before_frame.frameId])
        roles.setdefault("after_state", [after_frame.frameId])
        roles.setdefault("action_evidence", list(evidence_ids))
        normalized_actions.append(action.model_copy(update={
            "beforeFrameId": before_frame.frameId,
            "afterFrameId": after_frame.frameId,
            "beforeRequestedTimestampSeconds": before_requested,
            "afterRequestedTimestampSeconds": after_requested,
            "evidenceRequestedTimestampSeconds": evidence_requested,
            "resolutionErrorSeconds": max(errors),
            "evidenceFrameIds": list(dict.fromkeys(evidence_ids)),
            "evidenceRoles": roles,
            "uncertainty": uncertainty,
        }))
    action_count = len(timeline.actions)
    normalized_timeline_actions = normalized_actions[:action_count]
    normalized_unresolved = normalized_actions[action_count:]
    ordered = sorted(normalized_timeline_actions, key=lambda action: (action.startTimestampSeconds, action.endTimestampSeconds, action.actionId))
    if [action.actionId for action in normalized_timeline_actions] != [action.actionId for action in ordered]:
        raise AppError(502, "MODEL_INVALID_OUTPUT", "The action timeline must remain in source order.")
    unresolved_ordered = sorted(
        normalized_unresolved,
        key=lambda action: (action.startTimestampSeconds, action.endTimestampSeconds, action.actionId),
    )
    if [action.actionId for action in normalized_unresolved] != [action.actionId for action in unresolved_ordered]:
        raise AppError(502, "MODEL_INVALID_OUTPUT", "Unresolved action intervals must remain in source order.")
    normalized = timeline.model_copy(update={
        "durationSeconds": duration,
        "actions": normalized_timeline_actions,
        "unresolvedIntervals": normalized_unresolved,
    })
    return normalized.model_copy(update={"assemblyActions": build_assembly_mapping(normalized)})


def merge_timelines(parts: Iterable[ActionTimeline], *, duration_seconds: float) -> ActionTimeline:
    candidates = [action for part in parts for action in part.actions]
    unresolved_candidates = [action for part in parts for action in part.unresolvedIntervals]
    candidates.sort(key=lambda action: (action.startTimestampSeconds, action.endTimestampSeconds, action.actionId))
    merged: list[TimelineAction] = []
    conflicts: list[TimelineAction] = []
    for action in candidates:
        matching = next(
            (
                existing
                for existing in merged
                if existing.actionType == action.actionType
                and existing.movingPartId == action.movingPartId
                and existing.receivingPartId == action.receivingPartId
                and action.startTimestampSeconds <= existing.endTimestampSeconds
                and existing.startTimestampSeconds <= action.endTimestampSeconds
                and not (existing.sourceWindowId and action.sourceWindowId and existing.sourceWindowId == action.sourceWindowId)
            ),
            None,
        )
        if matching is not None:
            replacement = matching.model_copy(update={
                "startTimestampSeconds": min(matching.startTimestampSeconds, action.startTimestampSeconds),
                "endTimestampSeconds": max(matching.endTimestampSeconds, action.endTimestampSeconds),
                "evidenceFrameIds": list(dict.fromkeys([*matching.evidenceFrameIds, *action.evidenceFrameIds])),
                "beforeFrameId": matching.beforeFrameId or action.beforeFrameId,
                "afterFrameId": action.afterFrameId or matching.afterFrameId,
                "uncertainty": matching.uncertainty or action.uncertainty,
                "evidence": matching.evidence,
            })
            merged[merged.index(matching)] = replacement
            continue
        conflict = next(
            (
                existing
                for existing in merged
                if action.startTimestampSeconds <= existing.endTimestampSeconds
                and existing.startTimestampSeconds <= action.endTimestampSeconds
                and (
                    existing.actionType != action.actionType
                    or existing.movingPartId == action.movingPartId
                    or (existing.movingPartId is None and existing.receivingPartId == action.receivingPartId)
                )
            ),
            None,
        )
        if conflict is not None:
            message = "Overlapping provider interpretations conflict; receiving part or action type is unresolved."
            conflicts.append(action.model_copy(update={"uncertainty": action.uncertainty or message}))
            merged[merged.index(conflict)] = conflict.model_copy(update={"uncertainty": conflict.uncertainty or message})
        else:
            merged.append(action)
    merged.sort(key=lambda action: (action.startTimestampSeconds, action.endTimestampSeconds, action.actionId))
    normalized = [action.model_copy(update={"actionId": f"action-{index + 1:04d}"}) for index, action in enumerate(merged)]
    unresolved_candidates.sort(key=lambda action: (action.startTimestampSeconds, action.endTimestampSeconds, action.actionId))
    unresolved = [*conflicts, *unresolved_candidates]
    unresolved.sort(key=lambda action: (action.startTimestampSeconds, action.endTimestampSeconds, action.actionId))
    unresolved = [action.model_copy(update={"actionId": f"unresolved-{index + 1:04d}"}) for index, action in enumerate(unresolved)]
    return ActionTimeline(durationSeconds=duration_seconds, actions=normalized, unresolvedIntervals=unresolved)


def reverse_reversible_actions(timeline: ActionTimeline) -> list[TimelineAction]:
    reversed_actions: list[TimelineAction] = []
    for action in reversed(timeline.actions):
        if action.actionType not in {"attach", "detach", "move", "separate"}:
            reversed_actions.append(action.model_copy(update={
                "actionType": "unknown",
                "uncertainty": action.uncertainty or "This source interval is not safely reversible.",
            }))
            continue
        action_type = {"attach": "detach", "detach": "attach", "move": "move", "separate": "attach"}[action.actionType]
        reversed_actions.append(action.model_copy(update={
            "actionType": action_type,
            "beforeFrameId": action.afterFrameId,
            "afterFrameId": action.beforeFrameId,
            "beforeRequestedTimestampSeconds": action.afterRequestedTimestampSeconds,
            "afterRequestedTimestampSeconds": action.beforeRequestedTimestampSeconds,
            "relationshipBefore": action.relationshipAfter,
            "relationshipAfter": action.relationshipBefore,
            "evidenceRoles": {
                "before_state": list(action.evidenceRoles.get("after_state", [])),
                "after_state": list(action.evidenceRoles.get("before_state", [])),
                "action_evidence": list(action.evidenceRoles.get("action_evidence", action.evidenceFrameIds)),
            },
        }))
    return reversed_actions


def build_assembly_mapping(timeline: ActionTimeline) -> list[AssemblyAction]:
    """Create the only supported source-to-assembly order before wording."""
    supported: list[AssemblyAction] = []
    review: list[AssemblyAction] = []
    for action in reversed(timeline.actions):
        reversed_action = reverse_reversible_actions(ActionTimeline(actions=[action]))[0]
        valid_connection = (
            action.movingPartId is not None
            and action.receivingPartId is not None
            and action.relationshipBefore == "attached"
            and action.relationshipAfter == "separate"
            and action.actionType in {"detach", "separate"}
            and not action.uncertainty
        )
        if valid_connection:
            supported.append(AssemblyAction(
                mappingId=f"assembly-{len(supported) + 1:04d}",
                sourceActionIds=[action.actionId],
                assemblyOrdinal=len(supported) + 1,
                actionType=reversed_action.actionType,
                movingPartId=reversed_action.movingPartId,
                receivingPartId=reversed_action.receivingPartId,
                beforeFrameId=reversed_action.beforeFrameId,
                afterFrameId=reversed_action.afterFrameId,
                evidenceFrameIds=reversed_action.evidenceFrameIds,
                relationshipBefore=reversed_action.relationshipBefore,
                relationshipAfter=reversed_action.relationshipAfter,
                disposition="instruction",
                sourceDirection=timeline.sourceDirection,
                uncertainty=None,
            ))
        else:
            review.append(AssemblyAction(
                mappingId=f"assembly-review-{len(review) + 1:04d}",
                sourceActionIds=[action.actionId],
                actionType=reversed_action.actionType,
                movingPartId=action.movingPartId,
                receivingPartId=action.receivingPartId,
                beforeFrameId=reversed_action.beforeFrameId,
                afterFrameId=reversed_action.afterFrameId,
                evidenceFrameIds=reversed_action.evidenceFrameIds,
                relationshipBefore=reversed_action.relationshipBefore,
                relationshipAfter=reversed_action.relationshipAfter,
                disposition="review",
                sourceDirection=timeline.sourceDirection,
                uncertainty=action.uncertainty or "The recording does not establish a safely reversible connection for this action.",
            ))
    for action in timeline.unresolvedIntervals:
        review.append(AssemblyAction(
            mappingId=f"assembly-review-{len(review) + 1:04d}",
            sourceActionIds=[action.actionId],
            actionType="unknown",
            movingPartId=action.movingPartId,
            receivingPartId=action.receivingPartId,
            beforeFrameId=action.beforeFrameId,
            afterFrameId=action.afterFrameId,
            evidenceFrameIds=action.evidenceFrameIds,
            disposition="review",
            sourceDirection=timeline.sourceDirection,
            uncertainty=action.uncertainty or "This manipulation interval remains unresolved.",
        ))
    return [*supported, *review]


def timeline_evidence(
    frames: list[Frame],
    paths: list[Path],
    timeline: ActionTimeline,
    annotations: list[PartAnnotation],
    *,
    max_images: int = 24,
    frame_indexes: tuple[int, ...] | None = None,
    action_ids: set[str] | None = None,
) -> EvidenceBundle:
    if len(frames) != len(paths):
        raise AppError(422, "EVIDENCE_INVALID", "Evidence frames and image files are out of sync.")
    by_id = {frame.frameId: (frame, path) for frame, path in zip(frames, paths, strict=True)}
    selected: list[str] = []

    def add(frame_id: str | None) -> None:
        if frame_id is not None and frame_id in by_id and frame_id not in selected:
            selected.append(frame_id)

    for index in frame_indexes or tuple(range(len(frames))):
        if 0 <= index < len(frames):
            add(frames[index].frameId)
    add(frames[0].frameId if frames else None)
    add(frames[-1].frameId if frames else None)
    for annotation in annotations:
        if 0 <= annotation.frameIndex < len(frames):
            add(frames[annotation.frameIndex].frameId)
    actions = [action for action in [*timeline.actions, *timeline.unresolvedIntervals] if action_ids is None or action.actionId in action_ids]
    roles: dict[str, list[str]] = {}
    for action in actions:
        add(action.beforeFrameId)
        add(action.afterFrameId)
        for frame_id in action.evidenceFrameIds:
            add(frame_id)
        for role, frame_ids in action.evidenceRoles.items():
            roles.setdefault(role, []).extend(frame_ids)
    if len(selected) > max_images:
        required = {frames[0].frameId, frames[-1].frameId, *(frames[annotation.frameIndex].frameId for annotation in annotations if 0 <= annotation.frameIndex < len(frames))}
        required.update(frame_id for action in actions for frame_id in action.evidenceFrameIds)
        if len(required) > max_images:
            raise AppError(422, "EVIDENCE_LIMIT", "Required timeline evidence exceeds the image review limit.")
        optional = [frame_id for frame_id in selected if frame_id not in required]
        selected = list(required) + optional[: max_images - len(required)]
    selected.sort(key=lambda frame_id: (by_id[frame_id][0].timestampSeconds, frame_id))
    context = {
        "schemaVersion": "action-evidence-v1",
        "sourceChronology": "extracted-video order; assembly order is a validated reversal of reversible actions",
        "frameRegistry": [
            {"frameId": frame.frameId, "timestampSeconds": frame.timestampSeconds}
            for frame in frames
        ],
        "selectedFrameIds": selected,
        "pieces": reference_bundle(frames, paths, annotations),
        "referenceBundle": reference_bundle(frames, paths, annotations),
        "actions": [action.model_dump() for action in actions],
        "assemblyActions": [mapping.model_dump() for mapping in timeline.assemblyActions],
        "submittedFrameIds": selected,
        "evidenceRoles": {role: list(dict.fromkeys(frame_ids)) for role, frame_ids in roles.items()},
    }
    selected_pairs = [(by_id[frame_id][0], by_id[frame_id][1]) for frame_id in selected]
    return EvidenceBundle(
        context=context,
        frames=[frame for frame, _ in selected_pairs],
        paths=[path for _, path in selected_pairs],
        eventIds=[action.actionId for action in actions],
        manifest=[
            {
                "frameId": frame.frameId,
                "sourceTimestampSeconds": frame.timestampSeconds,
                "dimensions": {"width": dimensions[0], "height": dimensions[1]} if (dimensions := _image_dimensions(path)) else None,
                "sha256": _file_sha256(path),
                "roles": [role for role, frame_ids in roles.items() if frame.frameId in frame_ids],
            }
            for frame, path in selected_pairs
        ],
    )


def timeline_evidence_batches(
    frames: list[Frame],
    paths: list[Path],
    timeline: ActionTimeline,
    annotations: list[PartAnnotation],
    *,
    max_images: int = 24,
) -> list[EvidenceBundle]:
    """Batch action evidence without dropping late or required images."""
    actions = [*timeline.actions, *timeline.unresolvedIntervals]
    if not actions:
        return [timeline_evidence(frames, paths, timeline, annotations, max_images=max_images, frame_indexes=tuple(_global_context_indexes(len(frames), max_images)))]
    # Reserve room for the required before/after evidence of at least one
    # action; the request cap includes both global context and action frames.
    global_indexes = _global_context_indexes(len(frames), max(2, max_images - 2))
    base = {frames[index].frameId for index in global_indexes if 0 <= index < len(frames)}
    base.update(frames[annotation.frameIndex].frameId for annotation in annotations if 0 <= annotation.frameIndex < len(frames))
    if len(base) > max_images:
        raise AppError(422, "EVIDENCE_LIMIT", "Required inventory and global context exceeds the image review limit.")
    batches: list[list[TimelineAction]] = []
    current: list[TimelineAction] = []
    for action in actions:
        required = set(base) | {frame_id for frame_id in action.evidenceFrameIds if frame_id}
        required.update(frame_id for frame_id in [action.beforeFrameId, action.afterFrameId] if frame_id)
        current_required = set(base) | {frame_id for item in current for frame_id in [*item.evidenceFrameIds, item.beforeFrameId, item.afterFrameId] if frame_id}
        if current and len(current_required | required) > max_images:
            batches.append(current)
            current = []
        if len((set(base) | required)) > max_images:
            raise AppError(422, "EVIDENCE_LIMIT", "One action requires more evidence images than the review limit.")
        current.append(action)
    if current:
        batches.append(current)
    count = len(batches)
    result: list[EvidenceBundle] = []
    for index, batch in enumerate(batches):
        ids = {action.actionId for action in batch}
        bundle = timeline_evidence(
            frames,
            paths,
            timeline,
            annotations,
            max_images=max_images,
            frame_indexes=tuple(global_indexes),
            action_ids=ids,
        )
        bundle.context["batch"] = {"index": index, "count": count, "actionIds": sorted(ids)}
        result.append(bundle)
    return result


def _global_context_indexes(frame_count: int, max_images: int) -> list[int]:
    if frame_count <= 0:
        return []
    slots = max(2, min(max_images, 8))
    if frame_count <= slots:
        return list(range(frame_count))
    return list(dict.fromkeys(round(index * (frame_count - 1) / (slots - 1)) for index in range(slots)))
