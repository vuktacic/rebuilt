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
    processing_timeout_seconds: float = 900.0
    api_timeout_seconds: float = 90.0
    ffmpeg_binary: str = "ffmpeg"
    analysis_fps: float = 2.0
    analysis_window_seconds: float = 10.0
    analysis_overlap_seconds: float = 2.0
    analysis_max_gap_seconds: float = 1.5
    vision_backend: str = "sam3"
    sam3_checkpoint: Path | None = None
    sam3_bpe_path: Path | None = None
    sam3_hf_token_path: Path | None = None
    sam3_model_version: str = "facebook/sam3"
    sam3_precision: str = "bf16"
    sam3_max_gpu_headroom_mib: int = 500
    analysis_config_version: str = "v2"
    vision_worker: bool = True

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        precision = os.getenv("SAM3_PRECISION", "bf16").lower()
        checkpoint_name = "sam3-bf16.pt" if precision == "bf16" else "sam3.pt"
        local_checkpoint = project_root / ".models" / "sam3" / checkpoint_name
        configured_dir = data_dir or Path(os.getenv("REBUILT_DATA_DIR", ".data"))
        configured_checkpoint = os.getenv("SAM3_CHECKPOINT")
        return cls(
            data_dir=configured_dir,
            max_upload_bytes=int(os.getenv("REBUILT_MAX_UPLOAD_BYTES", 250 * 1024 * 1024)),
            model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses"),
            processing_timeout_seconds=float(os.getenv("REBUILT_PROCESSING_TIMEOUT", "900")),
            api_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT", "90")),
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg"),
            analysis_fps=float(os.getenv("REBUILT_ANALYSIS_FPS", "2")),
            analysis_window_seconds=float(os.getenv("REBUILT_ANALYSIS_WINDOW_SECONDS", "10")),
            analysis_overlap_seconds=float(os.getenv("REBUILT_ANALYSIS_OVERLAP_SECONDS", "2")),
            analysis_max_gap_seconds=float(os.getenv("REBUILT_ANALYSIS_MAX_GAP_SECONDS", "1.5")),
            vision_backend=os.getenv("REBUILT_VISION_BACKEND", "sam3"),
            sam3_checkpoint=Path(configured_checkpoint) if configured_checkpoint else (local_checkpoint if local_checkpoint.is_file() else None),
            sam3_bpe_path=Path(os.environ["SAM3_BPE_PATH"]) if os.getenv("SAM3_BPE_PATH") else None,
            sam3_hf_token_path=Path(os.getenv("HF_TOKEN_FILE", project_root / "private" / ".hf_token")),
            sam3_model_version=os.getenv("SAM3_MODEL_VERSION", "facebook/sam3"),
            sam3_precision=precision,
            sam3_max_gpu_headroom_mib=int(os.getenv("SAM3_MIN_GPU_HEADROOM_MIB", "500")),
            analysis_config_version=os.getenv("REBUILT_ANALYSIS_CONFIG_VERSION", "v2"),
            vision_worker=os.getenv("REBUILT_VISION_WORKER", "1").lower() not in {"0", "false", "no"},
        )
