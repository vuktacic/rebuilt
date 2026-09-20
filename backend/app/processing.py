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
from .guide_frame_presets import off_grid_snapshot_timestamps, preset_for, select_preset_frames, snapshot_frame_id
from .models import AnalysisEvent, AnnotationSuggestionResult, Frame, Guide, GuideStep, JobError, PartAnnotation
from .piece_suggestions import PieceSuggestionGenerator
from .storage import JobRepository
from .vision import VisionAnalyzer


class GuideGenerator(Protocol):
    def generate(self, frames: list[Frame], frame_paths: list[Path], events: list[AnalysisEvent] | None = None) -> Guide: ...


PROMPT = """You turn ordered frames from a fixed-camera LEGO build video into a concise assembly guide.
Describe only visible assembly changes, use a clear completed-state frame for each step when possible,
and mention uncertainty when a hand or occlusion hides placement. Local analysis labels each pair BEFORE
and AFTER; describe additions for attach events and removals for detach events. Return exactly one
step per supplied event, in event order, without merging separate events.
Write every title and step in clear, neutral Standard Technical English. Use precise imperative verbs,
consistent part references, and short unambiguous sentences; avoid slang, idioms, conversational filler,
or region-specific phrasing.
When local event pairs are supplied, return exactly one step per supplied event in event order. When only
settled snapshots are supplied, return one step per visible change between adjacent snapshots. Every frameId
must be copied exactly from the supplied frame list. Do not invent pieces or frame IDs.
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
            label = "/".join(dict.fromkeys(roles)) or ("EVENT" if events else "SNAPSHOT")
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
        command = [self.settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-y", "-autorotate", "-i", str(input_path), "-vf", f"fps={self.settings.analysis_fps},{scale}", "-q:v", "3", pattern]
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
        frames = [Frame(frameId=f"frame-{index:04d}", timestampSeconds=index / self.settings.analysis_fps, imageUrl=f"/jobs/{job_id}/frames/frame-{index:04d}") for index in range(len(paths))]
        for index, path in enumerate(paths):
            target = output_dir / f"frame-{index:04d}.jpg"
            path.rename(target)
        return ExtractedFrames(frames=frames, paths=[output_dir / f"frame-{index:04d}.jpg" for index in range(len(paths))])

    def extract_snapshots(
        self,
        input_path: Path,
        output_dir: Path,
        job_id: str,
        timestamps_seconds: tuple[float, ...],
    ) -> ExtractedFrames:
        """Extract exact source timestamps that fall between regular FPS samples."""
        output_dir.mkdir(parents=True, exist_ok=True)
        scale = "scale='if(gt(iw,ih),1280,-2)':'if(gt(iw,ih),-2,1280)'"
        frames: list[Frame] = []
        paths: list[Path] = []
        for timestamp in timestamps_seconds:
            frame_id = snapshot_frame_id(timestamp)
            target = output_dir / f"{frame_id}.jpg"
            command = [
                self.settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-y", "-autorotate", "-i", str(input_path),
                "-ss", f"{timestamp:.6f}", "-frames:v", "1", "-vf", scale, "-q:v", "3", str(target),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=self.settings.processing_timeout_seconds)
            except FileNotFoundError as exc:
                raise AppError(500, "FFMPEG_NOT_FOUND", "FFmpeg is not installed on the server.") from exc
            except subprocess.TimeoutExpired as exc:
                raise AppError(504, "PROCESSING_TIMEOUT", "Video processing exceeded its time limit.") from exc
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or "").strip()
                raise AppError(422, "VIDEO_INVALID", detail or "A requested snapshot frame could not be decoded.") from exc
            if not target.is_file():
                raise AppError(422, "VIDEO_EMPTY", "A requested snapshot frame could not be decoded.")
            frames.append(Frame(
                frameId=frame_id,
                timestampSeconds=timestamp,
                imageUrl=f"/jobs/{job_id}/frames/{frame_id}",
            ))
            paths.append(target)
        return ExtractedFrames(frames=frames, paths=paths)


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


class JobProcessor:
    def __init__(
        self,
        repository: JobRepository,
        settings: Settings,
        extractor: FFmpegExtractor | None = None,
        generator: GuideGenerator | None = None,
        analyzer: VisionAnalyzer | None = None,
        suggestion_generator: PieceSuggestionGenerator | None = None,
    ):
        self.repository = repository
        self.extractor = extractor or FFmpegExtractor(settings)
        self.generator = generator or OpenAIResponsesGenerator(settings)
        self.analyzer = analyzer
        self.suggestion_generator = suggestion_generator or PieceSuggestionGenerator(settings)

    def process(self, job_id: str) -> None:
        try:
            job = self.repository.get(job_id)
            if job is None:
                return
            self.repository.update(job_id, status="extracting", error=None)
            input_path = next((path for path in (self.repository._job_dir(job_id) / "input").iterdir() if path.is_file()), None)
            if input_path is None:
                raise AppError(422, "VIDEO_MISSING", "The uploaded video is missing.")
            preset = preset_for(self.repository.source_filename(job_id))
            extracted = (
                self.extractor.extract_snapshots(
                    input_path,
                    self.repository._job_dir(job_id) / "frames",
                    job_id,
                    preset.timestamps_seconds,
                )
                if preset is not None
                else self.extractor.extract(input_path, self.repository._job_dir(job_id) / "frames", job_id)
            )
            if hasattr(self.analyzer, "analyze_annotated"):
                self.repository.update(job_id, status="suggesting", frames=extracted.frames, error=None)
                self.suggest_annotations(job_id, len(extracted.frames) - 1)
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
                        if len(generated.steps) != len(event_batch):
                            raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide model did not return one step for each detected event.")
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
            self.repository.update(job_id, status="ready", guide=guide, error=None)
        except AppError as exc:
            self.repository.update(job_id, status="failed", error=JobError(code=exc.code, message=exc.message))
        except Exception:
            self.repository.update(job_id, status="failed", error=JobError(code="PROCESSING_FAILED", message="The video could not be processed."))

    def suggest_annotations(self, job_id: str, frame_index: int | None = None) -> None:
        job = self.repository.get(job_id)
        if job is None:
            return
        index = len(job.frames) - 1 if frame_index is None else frame_index
        try:
            if job.status not in {"annotating", "suggesting"}:
                raise AppError(409, "ANNOTATION_NOT_READY", "Piece suggestions are available after frame extraction.")
            if index < 0 or index >= len(job.frames):
                raise AppError(422, "ANNOTATION_FRAME_INVALID", "The selected suggestion frame was not extracted.")
            self.repository.update(job_id, status="suggesting", error=None)
            path = self.repository.frame_path(job_id, job.frames[index].frameId)
            suggestions = [item.model_copy(update={"frameIndex": index}) for item in self.suggestion_generator.suggest(job.frames[index], path)]
            result = AnnotationSuggestionResult(status="completed", frameIndex=index, suggestions=suggestions)
        except AppError as exc:
            result = AnnotationSuggestionResult(status="unavailable", frameIndex=frame_index, message=exc.message)
        except Exception:
            result = AnnotationSuggestionResult(status="unavailable", frameIndex=frame_index, message="Piece suggestions are unavailable; add points manually.")
        self.repository.update(job_id, status="annotating", annotation_suggestions=result, error=None)

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
                source_frames=job.frames,
            )
            self.repository.update(
                job_id,
                tracks=result.tracks,
                part_tracks=result.part_tracks or [],
                events=result.events,
                analysis=result.analysis,
                error=None,
            )
            self._finish_guide(job_id, job.frames, frame_paths, result.events, self.repository.source_filename(job_id))
        except AppError as exc:
            self.repository.update(job_id, status="failed", error=JobError(code=exc.code, message=exc.message))
        except Exception:
            self.repository.update(job_id, status="failed", error=JobError(code="PROCESSING_FAILED", message="Backward tracking could not be completed."))

    def _finish_guide(
        self,
        job_id: str,
        frames: list[Frame],
        paths: list[Path],
        events: list[AnalysisEvent],
        source_filename: str | None = None,
    ) -> None:
        queued_frames, queued_paths = self._queue_off_grid_snapshots(job_id, source_filename, frames, paths)
        preset_frames = select_preset_frames(source_filename, queued_frames, queued_paths)
        if preset_frames is not None:
            selected_frames, selected_paths = preset_frames
            self.repository.update(job_id, status="generating", frames=queued_frames, error=None)
            guide = validate_guide(self._generate_guide(selected_frames, selected_paths, None), selected_frames)
            self.repository.update(job_id, status="ready", guide=guide, error=None)
            return
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
                if len(generated.steps) != len(event_batch):
                    raise AppError(502, "MODEL_INVALID_OUTPUT", "The guide model did not return one step for each detected event.")
                generated_guides.append(generated)
            guide = Guide(title=generated_guides[0].title, steps=[step for generated in generated_guides for step in generated.steps])
        self.repository.update(job_id, status="ready", guide=guide, error=None)

    def _queue_off_grid_snapshots(
        self,
        job_id: str,
        source_filename: str | None,
        frames: list[Frame],
        paths: list[Path],
    ) -> tuple[list[Frame], list[Path]]:
        timestamps = off_grid_snapshot_timestamps(source_filename, frames)
        if not timestamps:
            return frames, paths
        input_path = self.repository.uploaded_video_path(job_id)
        if input_path is None:
            raise AppError(422, "VIDEO_MISSING", "The uploaded video is missing.")
        snapshots = self.extractor.extract_snapshots(
            input_path,
            self.repository._job_dir(job_id) / "frames",
            job_id,
            timestamps,
        )
        combined = sorted(
            zip(frames + snapshots.frames, paths + snapshots.paths, strict=True),
            key=lambda item: item[0].timestampSeconds,
        )
        return [frame for frame, _ in combined], [path for _, path in combined]

    def _generate_guide(self, frames: list[Frame], paths: list[Path], events: list[AnalysisEvent] | None) -> Guide:
        if events is not None:
            try:
                return self.generator.generate(frames, paths, events=events)
            except TypeError as exc:
                if "events" not in str(exc):
                    raise
        return self.generator.generate(frames, paths)
