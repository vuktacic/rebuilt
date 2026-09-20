from __future__ import annotations

import base64
import json
import mimetypes
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .errors import AppError
from .manual_processing import GuideDrafter, PairReviewer, validate_pair_finding
from .models import AnalysisEvent, Frame, Guide, GuideStep, JobError, ManualReview, PairFinding, PartAnnotation
from .storage import JobRepository
from .vision import VisionAnalyzer


class GuideGenerator(Protocol):
    def generate(self, frames: list[Frame], frame_paths: list[Path], events: list[AnalysisEvent] | None = None) -> Guide: ...


PROMPT = """You turn ordered frames from a fixed-camera LEGO build video into a concise assembly guide.
Describe only visible assembly changes, use a clear completed-state frame for each step when possible,
and mention uncertainty when a hand or occlusion hides placement. Local analysis labels each pair BEFORE
and AFTER; describe additions for attach events and removals for detach events. Return concise, non-duplicative steps in event order. If several supplied events
use the same completed-state frame, combine their visible changes into one step
for that frame rather than repeating the image with near-identical instructions.
Write every title and step in clear, neutral Standard Technical English. Use precise imperative verbs,
consistent part references, and short unambiguous sentences; avoid slang, idioms, conversational filler,
or region-specific phrasing.
Every frameId must be copied exactly from the supplied frame list. Do not invent pieces or frame IDs.
"""


class OpenAIResponsesGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, frames: list[Frame], frame_paths: list[Path], events: list[AnalysisEvent] | None = None) -> Guide:
        if not self.settings.openai_api_key:
            raise AppError(500, "OPENAI_NOT_CONFIGURED", "OPENAI_API_KEY is not configured.")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": PROMPT}]
        if events:
            event_lines = [
                f"Event {event.eventId}: {event.kind} from {event.startTimestampSeconds:.1f}s to {event.endTimestampSeconds:.1f}s; "
                f"BEFORE={event.beforeFrameId}; AFTER={event.afterFrameId}; evidence={event.evidence}; uncertainty={event.uncertainty or 'none'}"
                for event in events[:10]
            ]
            content.append({"type": "input_text", "text": "Analyze these local event pairs in order:\n" + "\n".join(event_lines)})
        for frame, path in zip(frames, frame_paths, strict=True):
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            roles = [
                role
                for event in events or []
                for role, frame_id in (("BEFORE", event.beforeFrameId), ("AFTER", event.afterFrameId))
                if frame_id == frame.frameId
            ]
            label = "/".join(dict.fromkeys(roles)) or "EVENT"
            content.append({"type": "input_text", "text": f"{label} frame {frame.frameId} at {frame.timestampSeconds:.1f}s"})
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}"})
        payload = {
            "model": self.settings.model,
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "lego_guide", "strict": True, "schema": {
                "type": "object", "additionalProperties": False, "required": ["title", "steps"],
                "properties": {"title": {"type": "string"}, "steps": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["text", "frameId", "uncertainty"],
                    "properties": {"text": {"type": "string"}, "frameId": {"type": "string"}, "uncertainty": {"type": ["string", "null"]}}
                }}}
            }}}
        }
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
            raise AppError(502, "MODEL_FAILED", f"The guide model request failed: {exc}") from exc
        try:
            output = body.get("output_text")
            if not output:
                output = next(part["text"] for item in body["output"] for part in item.get("content", []) if part.get("text"))
            return Guide.model_validate(json.loads(output))
        except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide model returned incomplete or invalid structured output.") from exc


@dataclass
class ExtractedFrames:
    frames: list[Frame]
    paths: list[Path]


class FFmpegExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def extract(self, input_path: Path, output_dir: Path, job_id: str) -> ExtractedFrames:
        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(output_dir / "frame-%06d.jpg")
        scale = "scale='if(gt(iw,ih),1280,-2)':'if(gt(iw,ih),-2,1280)'"
        command = [self.settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-y", "-autorotate", "1", "-i", str(input_path), "-vf", f"fps={self.settings.analysis_fps},{scale}", "-q:v", "3", pattern]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=self.settings.processing_timeout_seconds)
        except FileNotFoundError as exc:
            raise AppError(500, "FFMPEG_NOT_FOUND", "FFmpeg is not installed on the server.") from exc
        except subprocess.TimeoutExpired as exc:
            raise AppError(504, "PROCESSING_TIMEOUT", "Video processing exceeded its time limit.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()
            raise AppError(422, "VIDEO_INVALID", detail or "The video could not be decoded.") from exc
        paths = sorted(output_dir.glob("frame-*.jpg"))
        if not paths:
            raise AppError(422, "VIDEO_EMPTY", "The video did not contain any decodable frames.")
        frames = [
            Frame(
                frameId=f"frame-{index:04d}",
                sourceIndex=index,
                timestampSeconds=index / self.settings.analysis_fps,
                assemblyTimeSeconds=(len(paths) - 1 - index) / self.settings.analysis_fps,
                imageUrl=f"/jobs/{job_id}/frames/frame-{index:04d}",
            )
            for index in range(len(paths))
        ]
        for index, path in enumerate(paths):
            target = output_dir / f"frame-{index:04d}.jpg"
            path.rename(target)
        return ExtractedFrames(frames=frames, paths=[output_dir / f"frame-{index:04d}.jpg" for index in range(len(paths))])


def validate_guide(guide: Guide, frames: list[Frame]) -> Guide:
    frame_ids = {frame.frameId for frame in frames}
    title = guide.title.strip()
    if not title:
        raise AppError(422, "INVALID_GUIDE", "The guide title must not be empty.")
    if not guide.steps or any(not step.text.strip() for step in guide.steps):
        raise AppError(422, "INVALID_GUIDE", "Each guide step must contain non-empty instructions.")
    if any(step.frameId not in frame_ids for step in guide.steps):
        raise AppError(422, "INVALID_GUIDE", "Each guide step must reference an extracted frame.")
    return Guide(title=title, steps=[GuideStep(text=step.text.strip(), frameId=step.frameId, uncertainty=step.uncertainty) for step in guide.steps])


def deduplicate_guide_steps(guide: Guide) -> Guide:
    """Keep one final instruction per evidence image without losing visible actions."""
    merged: dict[str, GuideStep] = {}
    order: list[str] = []
    for step in guide.steps:
        existing = merged.get(step.frameId)
        if existing is None:
            merged[step.frameId] = step
            order.append(step.frameId)
            continue
        sentences = list(dict.fromkeys((existing.text.strip(), step.text.strip())))
        uncertainty = existing.uncertainty or step.uncertainty
        merged[step.frameId] = GuideStep(text=" ".join(sentences), frameId=step.frameId, uncertainty=uncertainty)
    return Guide(title=guide.title, steps=[merged[frame_id] for frame_id in order])


class JobProcessor:
    def __init__(
        self,
        repository: JobRepository,
        settings: Settings,
        extractor: FFmpegExtractor | None = None,
        generator: GuideGenerator | None = None,
        analyzer: VisionAnalyzer | None = None,
        manual_reviewer: PairReviewer | None = None,
        manual_drafter: GuideDrafter | None = None,
    ):
        self.repository = repository
        self.settings = settings
        self.extractor = extractor or FFmpegExtractor(settings)
        self.generator = generator or OpenAIResponsesGenerator(settings)
        self.analyzer = analyzer
        self.manual_reviewer = manual_reviewer
        self.manual_drafter = manual_drafter

    def process(self, job_id: str) -> None:
        try:
            job = self.repository.get(job_id)
            if job is None:
                return
            self.repository.update(job_id, status="extracting", error=None)
            input_path = next((path for path in (self.repository._job_dir(job_id) / "input").iterdir() if path.is_file()), None)
            if input_path is None:
                raise AppError(422, "VIDEO_MISSING", "The uploaded video is missing.")
            extracted = self.extractor.extract(input_path, self.repository._job_dir(job_id) / "frames", job_id)
            extracted = ExtractedFrames(
                frames=[
                    frame.model_copy(
                        update={
                            "sourceIndex": frame.sourceIndex if frame.sourceIndex is not None else index,
                            "assemblyTimeSeconds": (
                                frame.assemblyTimeSeconds
                                if frame.assemblyTimeSeconds is not None
                                else (len(extracted.frames) - 1 - index) / self.settings.analysis_fps
                            ),
                        }
                    )
                    for index, frame in enumerate(extracted.frames)
                ],
                paths=extracted.paths,
            )
            if job.mode == "manual":
                self.repository.update(job_id, status="pairing", frames=extracted.frames, error=None)
                return
            if hasattr(self.analyzer, "analyze_annotated"):
                self.repository.update(job_id, status="annotating", frames=extracted.frames, error=None)
                return
            self.repository.update(job_id, status="analyzing" if self.analyzer else "generating", frames=extracted.frames, error=None)
            if self.analyzer is not None:
                result = self.analyzer.analyze(self.repository._job_dir(job_id) / "frames", len(extracted.frames), job_id)
                self.repository.update(job_id, tracks=result.tracks, events=result.events, analysis=result.analysis, error=None)
                if not result.events:
                    guide = Guide(
                        title="Build needs review",
                        steps=[GuideStep(
                            text="No reliable physical change was detected automatically. Review the recording and replace this draft with the first verified build step.",
                            frameId=extracted.frames[0].frameId,
                            uncertainty="The local tracker could not maintain enough evidence for a confident change history.",
                        )],
                    )
                else:
                    frame_ids = {frame.frameId for frame in extracted.frames}
                    referenced_ids = {frame_id for event in result.events for frame_id in (event.beforeFrameId, event.afterFrameId)}
                    if not referenced_ids.issubset(frame_ids):
                        raise AppError(422, "INVALID_ANALYSIS", "Local analysis referenced a frame that was not extracted.")
                    self.repository.update(job_id, status="generating", error=None)
                    generated_guides: list[Guide] = []
                    for offset in range(0, len(result.events), 10):
                        event_batch = result.events[offset:offset + 10]
                        selected_ids = {event.beforeFrameId for event in event_batch} | {event.afterFrameId for event in event_batch}
                        selected = [
                            (frame, path)
                            for frame, path in zip(extracted.frames, extracted.paths, strict=True)
                            if frame.frameId in selected_ids
                        ]
                        frames_for_generation = [frame for frame, _ in selected]
                        paths_for_generation = [path for _, path in selected]
                        generated = validate_guide(
                            self._generate_guide(frames_for_generation, paths_for_generation, event_batch),
                            frames_for_generation,
                        )
                        if len(generated.steps) > len(event_batch):
                            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide model returned more steps than supported event evidence.")
                        generated_guides.append(generated)
                    guide = Guide(
                        title=generated_guides[0].title,
                        steps=[step for generated in generated_guides for step in generated.steps],
                    )
            else:
                guide = validate_guide(
                    self._generate_guide(extracted.frames, extracted.paths, None),
                    extracted.frames,
                )
            self.repository.update(job_id, status="ready", guide=deduplicate_guide_steps(guide), error=None)
        except AppError as exc:
            self.repository.update(job_id, status="failed", error=JobError(code=exc.code, message=exc.message))
        except Exception:
            self.repository.update(job_id, status="failed", error=JobError(code="PROCESSING_FAILED", message="The video could not be processed."))

    def track_annotated(self, job_id: str, annotations: list[PartAnnotation]) -> None:
        """Resume an extracted job after its named final-frame prompts are saved."""
        try:
            job = self.repository.get(job_id)
            if job is None:
                return
            if job.status != "annotating":
                raise AppError(409, "ANNOTATION_NOT_READY", "This video is not waiting for annotations.")
            analyzer = self.analyzer
            if analyzer is None or not hasattr(analyzer, "analyze_annotated"):
                raise AppError(409, "ANNOTATION_UNSUPPORTED", "The configured vision backend does not support manual backward tracking.")
            frame_dir = self.repository._job_dir(job_id) / "frames"
            frame_paths = [frame_dir / f"{frame.frameId}.jpg" for frame in job.frames]
            self.repository.update(job_id, status="analyzing", annotations=annotations, tracking_progress=0, error=None)
            result = analyzer.analyze_annotated(
                frame_dir,
                len(job.frames),
                job_id,
                annotations,
                on_progress=lambda value: self.repository.update(job_id, tracking_progress=value, error=None),
            )
            self.repository.update(
                job_id,
                tracks=result.tracks,
                part_tracks=result.part_tracks or [],
                events=result.events,
                analysis=result.analysis,
                error=None,
            )
            self._finish_guide(job_id, job.frames, frame_paths, result.events)
        except AppError as exc:
            self.repository.update(job_id, status="failed", error=JobError(code=exc.code, message=exc.message))
        except Exception:
            self.repository.update(job_id, status="failed", error=JobError(code="PROCESSING_FAILED", message="Backward tracking could not be completed."))

    def run_manual(self, job_id: str, analysis_run_id: str) -> None:
        """Review each saved pair independently and retain partial results."""
        try:
            job = self.repository.get(job_id)
            if job is None or job.analysisRunId != analysis_run_id or job.status != "diffing":
                return
            if self.manual_reviewer is None:
                raise AppError(503, "MANUAL_REVIEW_UNAVAILABLE", "Manual visual review is not configured.")
            findings: list[PairFinding] = []
            frames_by_id = {frame.frameId: frame for frame in job.frames}
            for pair in job.manualPairs:
                current = self.repository.get(job_id)
                if current is None or current.analysisRunId != analysis_run_id:
                    return
                try:
                    frame_paths = [
                        self.repository.frame_path(job_id, pair.beforeFrameId),
                        self.repository.frame_path(job_id, pair.afterFrameId),
                    ]
                    finding = self.manual_reviewer.review(
                        pair,
                        [frames_by_id[pair.beforeFrameId], frames_by_id[pair.afterFrameId]],
                        frame_paths,
                    )
                    findings.append(validate_pair_finding(finding, pair, [frames_by_id[pair.beforeFrameId], frames_by_id[pair.afterFrameId]]))
                except Exception:
                    findings.append(PairFinding(
                        pairId=pair.pairId,
                        status="failed",
                        action="uncertain_change",
                        difference="Pair review failed.",
                        uncertainty="insufficient_evidence",
                        confidence=0,
                        evidenceFrameIds=[pair.afterFrameId],
                    ))
                self.repository.update(job_id, manual_review=ManualReview(status="diffing", pairs=findings), error=None)
            failed = any(finding.status == "failed" for finding in findings)
            guide = None
            guide_status = "degraded" if failed else "no_eligible_findings"
            review_status = "degraded" if failed else "ready"
            if not failed and self.manual_drafter is not None:
                self.repository.update(job_id, manual_review=ManualReview(status="drafting", pairs=findings), error=None)
                guide = deduplicate_guide_steps(validate_guide(self.manual_drafter.draft(findings, job.manualPairs, job.frames), job.frames))
                guide_status = "ready"
            self.repository.update(
                job_id,
                status="ready",
                guide=guide,
                manual_review=ManualReview(
                    status=review_status,
                    pairs=findings,
                    guideStatus=guide_status,
                ),
                error=None,
            )
        except AppError as exc:
            self.repository.update(job_id, status="ready", manual_review=ManualReview(status="degraded", guideStatus="degraded"), error=JobError(code=exc.code, message=exc.message))
        except Exception:
            self.repository.update(job_id, status="ready", manual_review=ManualReview(status="degraded", guideStatus="degraded"), error=JobError(code="MANUAL_REVIEW_FAILED", message="Manual visual review could not be completed."))

    def _finish_guide(self, job_id: str, frames: list[Frame], paths: list[Path], events: list[AnalysisEvent]) -> None:
        if not events:
            guide = Guide(
                title="Build needs review",
                steps=[GuideStep(
                    text="No reliable attachment interval was inferred. Review the reverse-track overlay and add the first verified build step.",
                    frameId=frames[0].frameId,
                    uncertainty="The local reverse tracker did not find a confident attachment transition.",
                )],
            )
        else:
            frame_ids = {frame.frameId for frame in frames}
            referenced_ids = {frame_id for event in events for frame_id in (event.beforeFrameId, event.afterFrameId)}
            if not referenced_ids.issubset(frame_ids):
                raise AppError(422, "INVALID_ANALYSIS", "Local analysis referenced a frame that was not extracted.")
            self.repository.update(job_id, status="generating", error=None)
            generated_guides: list[Guide] = []
            for offset in range(0, len(events), 10):
                event_batch = events[offset:offset + 10]
                selected_ids = {event.beforeFrameId for event in event_batch} | {event.afterFrameId for event in event_batch}
                selected = [(frame, path) for frame, path in zip(frames, paths, strict=True) if frame.frameId in selected_ids]
                generated = validate_guide(
                    self._generate_guide([frame for frame, _ in selected], [path for _, path in selected], event_batch),
                    [frame for frame, _ in selected],
                )
                if len(generated.steps) > len(event_batch):
                    raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide model returned more steps than supported event evidence.")
                generated_guides.append(generated)
            guide = Guide(title=generated_guides[0].title, steps=[step for generated in generated_guides for step in generated.steps])
        self.repository.update(job_id, status="ready", guide=deduplicate_guide_steps(guide), error=None)

    def _generate_guide(self, frames: list[Frame], paths: list[Path], events: list[AnalysisEvent] | None) -> Guide:
        if events is not None:
            try:
                return self.generator.generate(frames, paths, events=events)
            except TypeError as exc:
                if "events" not in str(exc):
                    raise
        return self.generator.generate(frames, paths)
