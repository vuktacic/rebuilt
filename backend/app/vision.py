from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .errors import AppError
from .models import AnalysisEvent, AnalysisInfo, TrackSummary


@dataclass(frozen=True)
class Detection:
    frame_index: int
    predictor_id: str
    concept: str
    score: float
    bbox: tuple[float, float, float, float]
    visible: bool = True
    group_id: str | None = None
    occluded_by_hand: bool = False
    mask_signature: bytes | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.bbox
        return x + width / 2, y + height / 2

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2]) * max(0.0, self.bbox[3])


@dataclass
class AnalysisResult:
    events: list[AnalysisEvent]
    tracks: list[TrackSummary]
    analysis: AnalysisInfo


@dataclass(frozen=True)
class SeedMask:
    concept: str
    mask: Any


class VisionAnalyzer(Protocol):
    def analyze(self, frame_dir: Path, frame_count: int, job_id: str) -> AnalysisResult: ...


class NoopVisionAnalyzer:
    """Explicitly selectable for development and contract tests."""

    def analyze(self, frame_dir: Path, frame_count: int, job_id: str) -> AnalysisResult:
        return AnalysisResult(
            events=[],
            tracks=[],
            analysis=AnalysisInfo(backend="noop", modelVersion="none", configVersion="none"),
        )


class VisionUnavailable(AppError):
    def __init__(self, message: str):
        super().__init__(503, "VISION_UNAVAILABLE", message)


def _iou(left: Detection, right: Detection) -> float:
    lx, ly, lw, lh = left.bbox
    rx, ry, rw, rh = right.bbox
    ix = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    iy = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = ix * iy
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def _touches(piece: Detection, group: Detection) -> bool:
    if max(_iou(piece, group), _mask_iou(piece, group)) >= 0.05:
        return True
    px, py = piece.center
    gx, gy = group.center
    scale = max(group.bbox[2], group.bbox[3], 1.0)
    return ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 <= scale * 0.65


def _pieces_connected(left: Detection, right: Detection) -> bool:
    if max(_iou(left, right), _mask_iou(left, right)) >= 0.01:
        return True
    lx, ly, lw, lh = left.bbox
    rx, ry, rw, rh = right.bbox
    horizontal_gap = max(lx - (rx + rw), rx - (lx + lw), 0.0)
    vertical_gap = max(ly - (ry + rh), ry - (ly + lh), 0.0)
    tolerance = max(3.0, 0.08 * min(max(lw, lh), max(rw, rh)))
    horizontal_overlap = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    vertical_overlap = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    return (
        horizontal_gap <= tolerance and vertical_overlap >= min(lh, rh) * 0.15
    ) or (
        vertical_gap <= tolerance and horizontal_overlap >= min(lw, rw) * 0.15
    )


def _mask_iou(left: Detection, right: Detection) -> float:
    if left.mask_signature is None or right.mask_signature is None:
        return 0.0
    try:
        import numpy as np

        left_mask = np.frombuffer(left.mask_signature, dtype=np.uint8)
        right_mask = np.frombuffer(right.mask_signature, dtype=np.uint8)
        if left_mask.size != right_mask.size:
            return 0.0
        intersection = np.logical_and(left_mask, right_mask).sum()
        union = np.logical_or(left_mask, right_mask).sum()
        return float(intersection / union) if union else 0.0
    except (ImportError, ValueError):
        return 0.0


class PersistentTrackAssociator:
    """Keep application track IDs stable across short detector gaps or ID churn."""

    def __init__(self, fps: float = 10.0, max_gap_seconds: float = 1.5):
        self.max_gap_frames = max(1, int(round(fps * max_gap_seconds)))

    def associate(self, detections: list[Detection]) -> list[Detection]:
        next_id = 1
        active: dict[str, Detection] = {}
        original_ids: dict[tuple[str, str], str] = {}
        associated: list[Detection] = []
        for detection in sorted(detections, key=lambda item: (item.frame_index, item.concept, item.predictor_id)):
            key = (detection.concept, detection.predictor_id)
            chosen = original_ids.get(key)
            if chosen and chosen in active and detection.frame_index - active[chosen].frame_index <= self.max_gap_frames:
                track_id = chosen
            else:
                candidates = [
                    (track_id, previous)
                    for track_id, previous in active.items()
                    if previous.concept == detection.concept
                    and 0 < detection.frame_index - previous.frame_index <= self.max_gap_frames
                ]
                track_id = self._best_candidate(detection, candidates)
                if track_id is None:
                    track_id = f"track-{next_id:04d}"
                    next_id += 1
                original_ids[key] = track_id
            current = Detection(**{**detection.__dict__, "predictor_id": track_id})
            active[track_id] = current
            associated.append(current)
            for stale_id, previous in list(active.items()):
                if detection.frame_index - previous.frame_index > self.max_gap_frames:
                    active.pop(stale_id, None)
        return associated

    @staticmethod
    def _best_candidate(detection: Detection, candidates: list[tuple[str, Detection]]) -> str | None:
        ranked: list[tuple[float, str]] = []
        for track_id, previous in candidates:
            overlap = max(_iou(detection, previous), _mask_iou(detection, previous))
            dx = detection.center[0] - previous.center[0]
            dy = detection.center[1] - previous.center[1]
            distance = (dx * dx + dy * dy) ** 0.5
            allowed = max(40.0, 2.0 * max(detection.bbox[2], detection.bbox[3], previous.bbox[2], previous.bbox[3]))
            area_ratio = min(detection.area, previous.area) / max(detection.area, previous.area, 1.0)
            if (overlap >= 0.05 or distance <= allowed) and area_ratio >= 0.25:
                ranked.append((overlap + area_ratio - distance / (allowed * 4.0), track_id))
        return max(ranked)[1] if ranked else None


class WindowTrackStitcher:
    """Reconcile predictor-local IDs using detections in overlapping windows."""

    def __init__(self) -> None:
        self._detections: list[Detection] = []
        self._next_id = 1

    def add(self, window: list[Detection]) -> list[Detection]:
        current = self._tracklets(window)
        existing = self._tracklets(self._detections)
        assignments: dict[tuple[str, str], str] = {}
        candidates: list[tuple[float, tuple[str, str], tuple[str, str]]] = []
        for current_key, current_items in current.items():
            for existing_key, existing_items in existing.items():
                if current_key[0] != existing_key[0]:
                    continue
                score = self._overlap_score(current_items, existing_items)
                if score is not None:
                    candidates.append((score, current_key, existing_key))

        used_existing: set[tuple[str, str]] = set()
        for _, current_key, existing_key in sorted(candidates, reverse=True):
            if current_key in assignments or existing_key in used_existing:
                continue
            assignments[current_key] = existing_key[1]
            used_existing.add(existing_key)

        for current_key in sorted(current):
            if current_key not in assignments:
                assignments[current_key] = f"track-{self._next_id:04d}"
                self._next_id += 1

        merged = {
            (item.concept, item.predictor_id, item.frame_index): item
            for item in self._detections
        }
        for item in window:
            track_id = assignments[(item.concept, item.predictor_id)]
            stitched = Detection(**{**item.__dict__, "predictor_id": track_id})
            # A new window starts from a fresh detection, so prefer it in the
            # overlap over a mask propagated from the older window.
            merged[(stitched.concept, track_id, stitched.frame_index)] = stitched
        self._detections = sorted(
            merged.values(),
            key=lambda item: (item.frame_index, item.concept, item.predictor_id),
        )
        return self._detections

    @staticmethod
    def _tracklets(detections: list[Detection]) -> dict[tuple[str, str], list[Detection]]:
        grouped: dict[tuple[str, str], list[Detection]] = {}
        for item in detections:
            grouped.setdefault((item.concept, item.predictor_id), []).append(item)
        return grouped

    @staticmethod
    def _overlap_score(current: list[Detection], existing: list[Detection]) -> float | None:
        current_by_frame = {item.frame_index: item for item in current}
        existing_by_frame = {item.frame_index: item for item in existing}
        common_frames = current_by_frame.keys() & existing_by_frame.keys()
        if not common_frames:
            return None
        similarities = [
            max(
                _iou(current_by_frame[frame], existing_by_frame[frame]),
                _mask_iou(current_by_frame[frame], existing_by_frame[frame]),
            )
            for frame in common_frames
        ]
        best = max(similarities)
        if best < 0.05:
            return None
        return sum(similarities) / len(similarities) + best


class EventDetector:
    """Turn per-frame detections into conservative attachment transitions.

    Membership is deliberately separate from visibility. A short mask merge or
    a hand occlusion alone cannot create an event; the relation must remain
    stable on both sides of the transition.
    """

    def __init__(self, fps: float = 10.0, stable_seconds: float = 0.5, max_gap_seconds: float = 1.5):
        self.fps = fps
        self.stable_frames = max(2, int(round(fps * stable_seconds)))
        self.max_gap_frames = max(1, int(round(fps * max_gap_seconds)))

    def detect(self, detections: list[Detection], frame_count: int) -> tuple[list[AnalysisEvent], list[TrackSummary]]:
        track_frames: dict[str, list[Detection]] = {}
        for detection in detections:
            track_id = f"{detection.concept}:{detection.predictor_id}"
            track_frames.setdefault(track_id, []).append(detection)

        events: list[AnalysisEvent] = []
        tracks: list[TrackSummary] = []
        for track_id, observations in sorted(track_frames.items()):
            observations.sort(key=lambda item: item.frame_index)
            by_frame = {observation.frame_index: observation for observation in observations}
            memberships: list[tuple[int, str, bool]] = []
            last_membership: str | None = None
            has_long_gap = frame_count - 1 - observations[-1].frame_index > self.max_gap_frames
            for frame_index in range(observations[0].frame_index, observations[-1].frame_index + 1):
                observation = by_frame.get(frame_index)
                if observation is None:
                    previous_observation = next((item for item in reversed(observations) if item.frame_index < frame_index), None)
                    next_observation = next((item for item in observations if item.frame_index > frame_index), None)
                    gap_is_bridgeable = (
                        previous_observation is not None
                        and next_observation is not None
                        and next_observation.frame_index - previous_observation.frame_index <= self.max_gap_frames
                    )
                    has_long_gap = has_long_gap or not gap_is_bridgeable
                    memberships.append((frame_index, last_membership if gap_is_bridgeable else "unknown", gap_is_bridgeable))
                    continue
                last_membership = "attached" if observation.group_id else "separate"
                memberships.append((frame_index, last_membership, observation.occluded_by_hand or not observation.visible))
            tracks.append(
                TrackSummary(
                    trackId=track_id,
                    concept=observations[0].concept,
                    firstTimestampSeconds=observations[0].frame_index / self.fps,
                    lastTimestampSeconds=observations[-1].frame_index / self.fps,
                    visibility="lost" if has_long_gap else "occluded" if any(flag for _, _, flag in memberships) else "visible",
                    membership="attached" if memberships[-1][1] == "attached" else "separate",
                )
            )
            for index in range(1, len(memberships)):
                before_frame, before_membership, _ = memberships[index - 1]
                after_frame, after_membership, after_occluded = memberships[index]
                if after_frame != before_frame + 1 or before_membership in {"unknown", after_membership} or after_membership == "unknown":
                    continue
                if not self._stable(memberships, index - 1, before_membership, reverse=True):
                    continue
                if not self._stable(memberships, index, after_membership, reverse=False):
                    continue
                before_observation = next(
                    (observation for observation in reversed(observations) if observation.frame_index <= before_frame),
                    None,
                )
                after_observation = next(
                    (observation for observation in observations if observation.frame_index >= after_frame),
                    None,
                )
                if before_observation is None or after_observation is None:
                    continue
                start = max(0, before_observation.frame_index - self.stable_frames + 1)
                end = min(frame_count - 1, after_observation.frame_index + self.stable_frames - 1)
                affected = [track_id]
                movement = ((after_observation.center[0] - before_observation.center[0]) ** 2 + (after_observation.center[1] - before_observation.center[1]) ** 2) ** 0.5
                movement_threshold = max(2.0, 0.05 * max(before_observation.bbox[2], before_observation.bbox[3], after_observation.bbox[2], after_observation.bbox[3]))
                uncertainty_reasons = []
                if after_occluded or any(flag for frame, _, flag in memberships if start <= frame <= end):
                    uncertainty_reasons.append("vision was occluded during the transition")
                if movement < movement_threshold:
                    uncertainty_reasons.append("membership changed without measurable relative motion")
                if before_membership == "separate" and after_membership == "attached":
                    kind = "attach"
                    evidence = "piece remained separate, then stayed attached to an assembly"
                else:
                    kind = "detach"
                    evidence = "piece remained attached, then stayed separate from the assembly"
                if uncertainty_reasons:
                    kind = "uncertain_change"
                    uncertainty = "; ".join(uncertainty_reasons)
                else:
                    uncertainty = None
                events.append(
                    AnalysisEvent(
                        eventId=f"event-{len(events) + 1:04d}",
                        kind=kind,
                        startTimestampSeconds=start / self.fps,
                        endTimestampSeconds=end / self.fps,
                        affectedTrackIds=affected,
                        beforeFrameId=f"frame-{start:04d}",
                        afterFrameId=f"frame-{end:04d}",
                        evidenceStrength=0.55 if uncertainty_reasons else 0.9,
                        uncertainty=uncertainty,
                        evidence=evidence,
                    )
                )
        merged: dict[tuple[str, float, float, str | None], AnalysisEvent] = {}
        for event in events:
            key = (event.kind, event.startTimestampSeconds, event.endTimestampSeconds, event.uncertainty)
            existing = merged.get(key)
            if existing is None:
                merged[key] = event
            else:
                existing.affectedTrackIds = list(dict.fromkeys(existing.affectedTrackIds + event.affectedTrackIds))
                existing.evidenceStrength = min(existing.evidenceStrength, event.evidenceStrength)
        coalesced: list[AnalysisEvent] = []
        for event in sorted(merged.values(), key=lambda item: item.startTimestampSeconds):
            previous = coalesced[-1] if coalesced else None
            shares_track = previous is not None and bool(set(previous.affectedTrackIds) & set(event.affectedTrackIds))
            if previous is not None and shares_track and event.startTimestampSeconds <= previous.endTimestampSeconds:
                reasons = [
                    value
                    for value in (previous.uncertainty, event.uncertainty)
                    if value
                ]
                reasons.append("multiple membership changes occurred without a separating stable pause")
                coalesced[-1] = previous.model_copy(update={
                    "kind": "uncertain_change",
                    "endTimestampSeconds": max(previous.endTimestampSeconds, event.endTimestampSeconds),
                    "affectedTrackIds": list(dict.fromkeys(previous.affectedTrackIds + event.affectedTrackIds)),
                    "afterFrameId": event.afterFrameId,
                    "evidenceStrength": min(previous.evidenceStrength, event.evidenceStrength, 0.55),
                    "uncertainty": "; ".join(dict.fromkeys(reasons)),
                    "evidence": "multiple attachment changes occurred during one manipulation interval",
                })
            else:
                coalesced.append(event)

        normalized = []
        for index, event in enumerate(coalesced, start=1):
            normalized.append(event.model_copy(update={"eventId": f"event-{index:04d}"}))
        return normalized, tracks

    def _stable(self, memberships: list[tuple[int, str, bool]], index: int, value: str, *, reverse: bool) -> bool:
        indices = range(index, index - self.stable_frames, -1) if reverse else range(index, min(len(memberships), index + self.stable_frames))
        selected = list(indices)
        return len(selected) == self.stable_frames and all(memberships[item][1] == value for item in selected)


class Sam3VisionAnalyzer:
    """Lazy adapter around Meta's official SAM3 video predictor.

    SAM3 and PyTorch stay optional application dependencies. This keeps the API
    and offline tests usable on machines that do not have a CUDA environment,
    while production startup reports a structured error when the configured
    local model cannot be loaded.
    """

    PROMPTS = ("LEGO piece", "hand")

    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze(self, frame_dir: Path, frame_count: int, job_id: str) -> AnalysisResult:
        if self.settings.vision_worker:
            return self._analyze_worker(frame_dir, frame_count, job_id)
        return self._analyze_loaded(frame_dir, frame_count, job_id)

    def _analyze_worker(self, frame_dir: Path, frame_count: int, job_id: str) -> AnalysisResult:
        environment = os.environ.copy()
        environment["REBUILT_VISION_WORKER"] = "0"
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        request = json.dumps({"frame_dir": str(frame_dir), "frame_count": frame_count, "job_id": job_id})
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "backend.app.vision_worker"],
                input=request,
                capture_output=True,
                text=True,
                env=environment,
                timeout=self.settings.processing_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AppError(504, "ANALYSIS_TIMEOUT", "Local vision analysis exceeded its time limit.") from exc
        if completed.returncode != 0:
            detail = completed.stderr.lower()
            if "out of memory" in detail or "cuda oom" in detail:
                raise AppError(503, "VISION_OOM", "SAM3 ran out of GPU memory during local analysis.")
            if "no such file" in detail or "checkpoint" in detail and "not found" in detail:
                raise AppError(503, "VISION_WEIGHTS_MISSING", "The configured SAM3 checkpoint is unavailable.")
            raise VisionUnavailable("The local SAM3 worker could not complete analysis.")
        try:
            payload = json.loads(completed.stdout)
            return AnalysisResult(
                events=[AnalysisEvent.model_validate(item) for item in payload["events"]],
                tracks=[TrackSummary.model_validate(item) for item in payload["tracks"]],
                analysis=AnalysisInfo.model_validate(payload["analysis"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VisionUnavailable("The local SAM3 worker returned invalid analysis data.") from exc

    def _analyze_loaded(self, frame_dir: Path, frame_count: int, job_id: str) -> AnalysisResult:
        started = time.monotonic()
        predictor = self._load_predictor()
        frame_paths = sorted(frame_dir.glob("frame-*.jpg"))
        window_size = max(1, int(round(self.settings.analysis_window_seconds * self.settings.analysis_fps)))
        overlap = max(0, min(window_size - 1, int(round(self.settings.analysis_overlap_seconds * self.settings.analysis_fps))))
        step = max(1, window_size - overlap)
        detections: list[Detection] = []
        stitcher = WindowTrackStitcher()
        window_count = 0
        with tempfile.TemporaryDirectory(prefix="rebuilt-sam3-windows-") as temporary_dir:
            window_root = Path(temporary_dir)
            for start in range(0, frame_count, step):
                end = min(frame_count, start + window_size)
                window_dir = window_root / f"window-{start:06d}"
                window_dir.mkdir()
                for local_index, source in enumerate(frame_paths[start:end]):
                    (window_dir / f"{local_index:06d}.jpg").symlink_to(source)
                detections = stitcher.add(self._run_session(predictor, window_dir, end - start, start))
                window_count += 1
                if end == frame_count:
                    break
        events, tracks = self._events(detections, frame_count)
        events = self._select_screenshots(events, frame_dir, detections)
        elapsed = time.monotonic() - started
        return AnalysisResult(
            events=events,
            tracks=tracks,
            analysis=AnalysisInfo(
                backend="sam3",
                modelVersion=f"{self.settings.sam3_model_version}:{self.settings.sam3_precision}",
                configVersion=self.settings.analysis_config_version,
                durationSeconds=elapsed,
                metrics={
                    "frames": float(frame_count),
                    "detections": float(len(detections)),
                    "tracks": float(len(tracks)),
                    "events": float(len(events)),
                    "windows": float(window_count),
                    "lost_tracks": float(sum(track.visibility == "lost" for track in tracks)),
                    "max_gap_frames": float(max(1, round(self.settings.analysis_fps * self.settings.analysis_max_gap_seconds))),
                },
            ),
        )

    def _select_screenshots(self, events: list[AnalysisEvent], frame_dir: Path, detections: list[Detection]) -> list[AnalysisEvent]:
        try:
            import cv2
        except ImportError:
            return events
        sharpness: dict[int, float] = {}
        hand_frames = {detection.frame_index for detection in detections if detection.concept == "hand"}
        stable_frames = max(1, int(round(self.settings.analysis_fps * 0.5)))

        def score(frame_index: int) -> float:
            if frame_index not in sharpness:
                path = frame_dir / f"frame-{frame_index:04d}.jpg"
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
                value = float(cv2.Laplacian(image, cv2.CV_64F).var()) if image is not None else -1.0
                sharpness[frame_index] = value - 1_000_000.0 if frame_index in hand_frames else value
            return sharpness[frame_index]

        selected: list[AnalysisEvent] = []
        for event in events:
            start = max(0, int(round(event.startTimestampSeconds * self.settings.analysis_fps)))
            end = max(start, int(round(event.endTimestampSeconds * self.settings.analysis_fps)))
            before_candidates = list(range(start, min(end + 1, start + stable_frames)))
            after_candidates = list(range(max(start, end - stable_frames + 1), end + 1))
            before = max(before_candidates, key=score, default=start)
            after_options = [candidate for candidate in after_candidates if candidate > before] or after_candidates
            after = max(after_options, key=score, default=end)
            selected.append(event.model_copy(update={
                "beforeFrameId": f"frame-{before:04d}",
                "afterFrameId": f"frame-{after:04d}",
            }))
        return selected

    def _run_session(self, predictor: Any, window_dir: Path, frame_count: int, frame_offset: int) -> list[Detection]:
        seed_dir = window_dir.with_name(f"{window_dir.name}-seed")
        seed_dir.mkdir()
        first_frame = next(iter(sorted(window_dir.glob("*.jpg"))), None)
        if first_frame is None:
            return []
        (seed_dir / "000000.jpg").symlink_to(first_frame)
        seeds: list[SeedMask] = []
        for prompt in self.PROMPTS:
            session_id: str | None = None
            try:
                response = predictor.handle_request({
                    "type": "start_session",
                    "resource_path": str(seed_dir),
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": True,
                })
                session_id = response["session_id"]
                prompt_response = predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt})
                outputs = prompt_response.get("outputs") if isinstance(prompt_response, dict) else None
                if isinstance(outputs, dict):
                    seeds.extend(SeedMask(prompt, mask) for mask in self._batch_items(outputs.get("out_binary_masks", outputs.get("masks"))))
            finally:
                if session_id is not None:
                    try:
                        predictor.handle_request({"type": "close_session", "session_id": session_id})
                    except Exception:
                        pass
        if not seeds:
            return []

        session_id = None
        try:
            response = predictor.handle_request({
                "type": "start_session",
                "resource_path": str(window_dir),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
            })
            session_id = response["session_id"]
            concept_by_id: dict[str, str] = {}
            for object_id, seed in enumerate(seeds, start=1):
                predictor.handle_request({
                    "type": "add_mask",
                    "session_id": session_id,
                    "frame_index": 0,
                    "obj_id": object_id,
                    "mask": seed.mask,
                })
                concept_by_id[str(object_id)] = seed.concept
            return self._collect(predictor, session_id, frame_offset, concept_by_id, "LEGO piece")
        finally:
            if session_id is not None:
                try:
                    predictor.handle_request({"type": "close_session", "session_id": session_id})
                except Exception:
                    pass

    def _load_predictor(self) -> Any:
        token_path = self.settings.sam3_hf_token_path
        token_loaded = bool(os.getenv("HF_TOKEN"))
        if token_path and token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                os.environ.setdefault("HF_TOKEN", token)
                token_loaded = True
        if not token_loaded and not self.settings.sam3_checkpoint:
            raise VisionUnavailable("A Hugging Face token is required to load the SAM3 checkpoint.")
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError as exc:
            raise VisionUnavailable("SAM3 is not installed. Install Meta's SAM3 package in the CUDA analysis environment.") from exc
        predictor: Any | None = None
        try:
            import torch

            precision = self.settings.sam3_precision
            if precision not in {"bf16", "fp32"}:
                raise VisionUnavailable(f"Unsupported SAM3 precision: {precision}")

            kwargs: dict[str, Any] = {}
            if self.settings.sam3_checkpoint:
                kwargs["checkpoint_path"] = str(self.settings.sam3_checkpoint)
            if self.settings.sam3_bpe_path:
                kwargs["bpe_path"] = str(self.settings.sam3_bpe_path)
            if not torch.cuda.is_available():
                raise VisionUnavailable("CUDA is unavailable; SAM3 requires the configured local NVIDIA runtime.")
            previous_dtype = torch.get_default_dtype()
            try:
                if precision == "bf16":
                    torch.set_default_dtype(torch.bfloat16)
                predictor = build_sam3_video_predictor(**kwargs)
            finally:
                torch.set_default_dtype(previous_dtype)
            if precision == "bf16":
                self._restore_required_fp32_modules(predictor)
            free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes < self.settings.sam3_max_gpu_headroom_mib * 1024 * 1024:
                raise VisionUnavailable("Insufficient GPU headroom for SAM3 analysis.")
            return predictor
        except ImportError as exc:
            if predictor is not None:
                predictor.shutdown()
            raise VisionUnavailable("PyTorch is unavailable in the SAM3 environment.") from exc
        except VisionUnavailable:
            if predictor is not None:
                predictor.shutdown()
            raise
        except Exception as exc:
            if predictor is not None:
                try:
                    predictor.shutdown()
                except Exception:
                    pass
            raise VisionUnavailable(f"SAM3 could not be loaded: {type(exc).__name__}") from exc

    @staticmethod
    def _restore_required_fp32_modules(predictor: Any) -> None:
        for name, module in predictor.model.named_modules():
            if Sam3VisionAnalyzer._requires_fp32_module(name):
                module.float()

    @staticmethod
    def _requires_fp32_module(name: str) -> bool:
        return ".transformer.decoder.layers." in name and name.endswith((".linear1", ".linear2"))

    def _collect(
        self,
        predictor: Any,
        session_id: str,
        frame_offset: int,
        concept_by_id: dict[str, str],
        default_concept: str,
    ) -> list[Detection]:
        detections: list[Detection] = []
        request = {"type": "propagate_in_video", "session_id": session_id, "propagation_direction": "forward"}
        for response in predictor.handle_stream_request(request):
            if not isinstance(response, dict):
                continue
            frame_index = frame_offset + int(response.get("frame_index", 0))
            for output in self._output_items(response.get("outputs"), concept_by_id, default_concept):
                detection = self._parse_output(output, frame_index)
                if detection is not None:
                    detections.append(detection)
        return detections

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _batch_items(value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "shape") and len(value.shape) > 0:
            return [value[index] for index in range(len(value))]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _object_ids(self, outputs: Any) -> list[str]:
        if isinstance(outputs, list):
            return [
                str(item.get("object_id") or item.get("track_id") or item.get("id"))
                for item in outputs
                if isinstance(item, dict) and (item.get("object_id") is not None or item.get("track_id") is not None or item.get("id") is not None)
            ]
        if not isinstance(outputs, dict):
            return []
        ids = outputs.get("out_obj_ids", outputs.get("obj_ids", outputs.get("object_ids")))
        return [str(value) for value in self._as_list(ids)]

    def _output_items(self, outputs: Any, concept_by_id: dict[str, str], default_concept: str) -> list[dict[str, Any]]:
        if isinstance(outputs, list):
            return [item for item in outputs if isinstance(item, dict)]
        if not isinstance(outputs, dict):
            return []
        ids = self._object_ids(outputs)
        if ids:
            masks = self._batch_items(outputs.get("out_binary_masks", outputs.get("masks")))
            boxes = self._as_list(outputs.get("out_boxes_xywh", outputs.get("out_boxes", outputs.get("boxes"))))
            scores = self._as_list(outputs.get("out_probs", outputs.get("scores")))
            items: list[dict[str, Any]] = []
            for index, object_id in enumerate(ids):
                item: dict[str, Any] = {"object_id": object_id, "concept": concept_by_id.get(object_id, default_concept)}
                if index < len(masks):
                    item["mask"] = masks[index]
                if index < len(boxes):
                    item["bbox"] = boxes[index]
                if index < len(scores):
                    item["score"] = scores[index]
                items.append(item)
            return items
        return [{"object_id": str(object_id), **value} for object_id, value in outputs.items() if isinstance(value, dict)]

    def _parse_output(self, output: Any, frame_index: int) -> Detection | None:
        if not isinstance(output, dict):
            return None
        mask = output.get("mask")
        if mask is None:
            mask = output.get("segmentation")
        bbox = output.get("bbox")
        if bbox is None:
            bbox = output.get("box")
        if bbox is None:
            bbox = self._mask_bbox(mask)
        if bbox is None or len(bbox) != 4:
            return None
        raw_concept = str(output.get("concept") or output.get("prompt") or "LEGO piece")
        lowered_concept = raw_concept.lower()
        if "hand" in lowered_concept:
            concept = "hand"
        elif "assembly" in lowered_concept:
            concept = "LEGO assembly"
        else:
            concept = "LEGO piece"
        predictor_id = str(output.get("object_id") or output.get("track_id") or output.get("id") or "unknown")
        raw_score = output.get("score", output.get("confidence", 0.0))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        try:
            normalized_bbox = tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        return Detection(
            frame_index,
            predictor_id,
            concept,
            max(0.0, min(1.0, score)),
            normalized_bbox,
            mask_signature=self._mask_signature(mask),
        )

    @staticmethod
    def _mask_bbox(mask: Any) -> tuple[float, float, float, float] | None:
        if mask is None:
            return None
        try:
            array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask
            import numpy as np

            ys, xs = np.where(np.asarray(array).squeeze() > 0)
            if len(xs) == 0:
                return None
            return float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
        except (ImportError, TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _mask_signature(mask: Any) -> bytes | None:
        if mask is None:
            return None
        try:
            import numpy as np

            array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask
            array = np.asarray(array).squeeze() > 0
            if array.ndim != 2 or not array.any():
                return None
            rows = np.linspace(0, array.shape[0] - 1, 32).astype(int)
            columns = np.linspace(0, array.shape[1] - 1, 32).astype(int)
            return np.asarray(array[np.ix_(rows, columns)], dtype=np.uint8).tobytes()
        except (ImportError, TypeError, ValueError, AttributeError):
            return None

    def _events(self, detections: list[Detection], frame_count: int) -> tuple[list[AnalysisEvent], list[TrackSummary]]:
        assemblies = [item for item in detections if "assembly" in item.concept.lower()]
        hands = [item for item in detections if "hand" in item.concept.lower()]
        max_gap_frames = max(1, int(round(self.settings.analysis_fps * self.settings.analysis_max_gap_seconds)))
        pieces = [
            item for item in detections
            if "piece" in item.concept.lower() or "brick" in item.concept.lower()
        ]
        associated = PersistentTrackAssociator(
            self.settings.analysis_fps,
            self.settings.analysis_max_gap_seconds,
        ).associate(pieces)
        enriched: list[Detection] = []
        for item in associated:
            nearby_assemblies = [
                assembly for assembly in assemblies
                if abs(assembly.frame_index - item.frame_index) <= max_gap_frames
            ]
            group = next(
                (
                    assembly for assembly in sorted(
                        nearby_assemblies,
                        key=lambda candidate: abs(candidate.frame_index - item.frame_index),
                    )
                    if _touches(item, assembly)
                ),
                None,
            )
            connected_piece = next(
                (
                    other for other in associated
                    if other.frame_index == item.frame_index
                    and other.predictor_id != item.predictor_id
                    and _pieces_connected(item, other)
                ),
                None,
            )
            hand_occluded = any(hand.frame_index == item.frame_index and _iou(item, hand) > 0.01 for hand in hands)
            group_gap = group is not None and group.frame_index != item.frame_index
            enriched.append(Detection(**{
                **item.__dict__,
                "group_id": (
                    f"{group.concept}:{group.predictor_id}"
                    if group
                    else f"piece-cluster:{connected_piece.predictor_id}"
                    if connected_piece
                    else None
                ),
                "visible": item.visible and not group_gap,
                "occluded_by_hand": hand_occluded,
            }))
        return EventDetector(
            self.settings.analysis_fps,
            max_gap_seconds=self.settings.analysis_max_gap_seconds,
        ).detect(enriched, frame_count)
