from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_upload_bytes: int = 250 * 1024 * 1024
    model: str = "gpt-5.4"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1/responses"
    processing_timeout_seconds: float = 120.0
    api_timeout_seconds: float = 90.0
    ffmpeg_binary: str = "ffmpeg"

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> "Settings":
        configured_dir = data_dir or Path(os.getenv("REBUILT_DATA_DIR", ".data"))
        return cls(
            data_dir=configured_dir,
            max_upload_bytes=int(os.getenv("REBUILT_MAX_UPLOAD_BYTES", 250 * 1024 * 1024)),
            model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses"),
            processing_timeout_seconds=float(os.getenv("REBUILT_PROCESSING_TIMEOUT", "120")),
            api_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT", "90")),
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg"),
        )
