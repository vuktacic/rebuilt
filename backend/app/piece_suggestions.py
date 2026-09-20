from __future__ import annotations

import base64
import json
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import AppError
from .models import AnnotationSuggestion, Frame, PointPrompt


SUGGESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["suggestions"],
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "point", "confidence"],
                "properties": {
                    "name": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "point": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["x", "y"],
                        "properties": {
                            "x": {"type": "number", "minimum": 0, "maximum": 1},
                            "y": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                },
            },
        },
    },
}


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
        if index + 2 > len(data):
            break
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


def _data_url(path: Path) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _output_text(body: dict[str, Any]) -> str:
    output = body.get("output_text")
    if output:
        return str(output)
    return next(part["text"] for item in body["output"] for part in item.get("content", []) if part.get("text"))


def _responses_json(settings: Settings, content: list[dict[str, Any]]) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise AppError(500, "OPENAI_NOT_CONFIGURED", "OPENAI_API_KEY is not configured.")
    payload = {
        "model": settings.model,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": "assembly_object_suggestions", "strict": True, "schema": SUGGESTION_SCHEMA}},
    }
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
        raise AppError(502, "MODEL_FAILED", "The piece suggestion model request failed.") from exc
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AppError(502, "MODEL_INVALID_OUTPUT", "The piece suggestion model returned invalid structured output.") from exc


class PieceSuggestionGenerator:
    """Suggest names and foreground points on one extracted reference frame."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def suggest(self, frame: Frame, path: Path) -> list[AnnotationSuggestion]:
        dimensions = _image_dimensions(path)
        if dimensions is None:
            raise AppError(422, "IMAGE_INVALID", "The selected frame dimensions could not be read.")
        body = _responses_json(self.settings, content=[
            {
                "type": "input_text",
                "text": (
                    "Identify each clearly separated LEGO brick or assembly object visible in this disassembled frame. "
                    "Give each object a concise descriptive visual name, one foreground point inside it, and confidence "
                    "from 0 to 1. Use normalized coordinates from 0 to 1. Do not identify hidden, merged, or ambiguous objects. "
                    "Do not invent catalog numbers or pieces that are not visually distinct."
                ),
            },
            {"type": "input_image", "image_url": _data_url(path), "detail": "high"},
        ])
        suggestions: list[AnnotationSuggestion] = []
        name_counts: dict[str, int] = {}
        width, height = dimensions
        for item in body.get("suggestions", []):
            try:
                name = str(item["name"]).strip()
                point = normalized_point_to_pixels(PointPrompt.model_validate(item["point"]), width, height)
                confidence = float(item["confidence"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AppError(502, "MODEL_INVALID_OUTPUT", "The piece suggestion model returned invalid data.") from exc
            if not name:
                continue
            key = name.casefold()
            name_counts[key] = name_counts.get(key, 0) + 1
            if name_counts[key] > 1:
                name = f"{name} {name_counts[key]}"
            suggestions.append(AnnotationSuggestion(name=name, frameIndex=0, point=point, confidence=confidence))
        return suggestions
