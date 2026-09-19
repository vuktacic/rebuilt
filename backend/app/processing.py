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
from .models import Frame, Guide, GuideStep, JobError
from .storage import JobRepository


class GuideGenerator(Protocol):
    def generate(self, frames: list[Frame], frame_paths: list[Path]) -> Guide: ...


PROMPT = """You turn ordered frames from a fixed-camera LEGO build video into a concise assembly guide.
Describe only visible assembly changes, use a clear completed-state frame for each step when possible,
and mention uncertainty when a hand or occlusion hides placement. Return one step per meaningful change.
Every frameId must be copied exactly from the supplied frame list. Do not invent pieces or frame IDs.
"""


class OpenAIResponsesGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, frames: list[Frame], frame_paths: list[Path]) -> Guide:
        if not self.settings.openai_api_key:
            raise AppError(500, "OPENAI_NOT_CONFIGURED", "OPENAI_API_KEY is not configured.")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": PROMPT}]
        for frame, path in zip(frames, frame_paths, strict=True):
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "input_text", "text": f"Frame {frame.frameId} at {frame.timestampSeconds:.1f}s"})
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
        command = [self.settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-y", "-autorotate", "-i", str(input_path), "-vf", f"fps=1,{scale}", "-q:v", "3", pattern]
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
        frames = [Frame(frameId=f"frame-{index:04d}", timestampSeconds=float(index), imageUrl=f"/jobs/{job_id}/frames/frame-{index:04d}") for index in range(len(paths))]
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


class JobProcessor:
    def __init__(self, repository: JobRepository, settings: Settings, extractor: FFmpegExtractor | None = None, generator: GuideGenerator | None = None):
        self.repository = repository
        self.extractor = extractor or FFmpegExtractor(settings)
        self.generator = generator or OpenAIResponsesGenerator(settings)

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
            self.repository.update(job_id, status="generating", frames=extracted.frames, error=None)
            guide = validate_guide(self.generator.generate(extracted.frames, extracted.paths), extracted.frames)
            self.repository.update(job_id, status="ready", frames=extracted.frames, guide=guide, error=None)
        except AppError as exc:
            self.repository.update(job_id, status="failed", error=JobError(code=exc.code, message=exc.message))
        except Exception:
            self.repository.update(job_id, status="failed", error=JobError(code="PROCESSING_FAILED", message="The video could not be processed."))
