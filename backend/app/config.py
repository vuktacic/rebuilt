from __future__ import annotations

import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VALID_VISION_BACKENDS = frozenset({"auto", "noop", "sam2", "sam2-mlx", "sam3", "sam3-cuda", "sam3-mlx"})


class VisionRuntimeError(RuntimeError):
    """A stable, operator-facing diagnosis for unavailable local analysis."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _is_rosetta_translated() -> bool:
    if sys.platform != "darwin":
        return False
    completed = subprocess.run(
        ["sysctl", "-in", "sysctl.proc_translated"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "1"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_vision_backend(
    settings: "Settings",
    *,
    system: str | None = None,
    machine: str | None = None,
    translated: bool | None = None,
    cuda_available: bool | None = None,
) -> str:
    """Resolve the selected provider without importing optional model packages."""
    configured = settings.vision_backend.lower()
    if configured not in VALID_VISION_BACKENDS:
        valid = ", ".join(sorted(VALID_VISION_BACKENDS))
        raise VisionRuntimeError("VISION_BACKEND_INVALID", f"Unknown REBUILT_VISION_BACKEND={configured!r}; use one of: {valid}.")
    if configured == "noop":
        return "noop"

    if configured == "sam2":
        return "sam2"

    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    translated = _is_rosetta_translated() if translated is None else translated
    is_native_macos = system == "Darwin" and machine == "arm64" and not translated

    if configured == "auto":
        return "sam2"
    if configured in {"sam3", "sam3-cuda"}:
        if system == "Darwin":
            raise VisionRuntimeError(
                "VISION_BACKEND_INCOMPATIBLE",
                "sam3-cuda is unavailable on macOS; use REBUILT_VISION_BACKEND=sam3-mlx on native Apple Silicon.",
            )
        return "sam3-cuda"
    if translated:
        raise VisionRuntimeError(
            "VISION_ROSETTA_UNSUPPORTED",
            "sam3-mlx requires a native arm64 Python interpreter; recreate the environment with /opt/homebrew/bin/python3.",
        )
    if system == "Darwin" and machine != "arm64":
        raise VisionRuntimeError(
            "VISION_INTEL_MAC_UNSUPPORTED",
            "sam3-mlx requires Apple Silicon; Intel Macs need a supported CUDA host or noop mode.",
        )
    if not is_native_macos:
        raise VisionRuntimeError(
            "VISION_BACKEND_INCOMPATIBLE",
            "sam3-mlx is supported only on native arm64 macOS with Apple Metal.",
        )
    return "sam2-mlx" if configured == "sam2-mlx" else "sam3-mlx"


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=value entries without evaluating the file as code."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not _ENV_KEY.fullmatch(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _load_dotenv(path: Path, environ: dict[str, str] | None = None) -> None:
    target = os.environ if environ is None else environ
    for key, value in _read_dotenv(path).items():
        target.setdefault(key, value)


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
    sam2_checkpoint: Path | None = None
    sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_tracking_fps: float = 0.5
    sam2_frame_stride: int | None = None
    sam2_apply_postprocessing: bool = False
    sam2_vos_optimized: bool = False
    sam2_mlx_model_id: str = "avbiswas/sam2.1-hiera-small-mlx"
    sam2_mlx_image_size: int = 768
    sam2_mlx_precompute_features: bool = True
    sam2_mlx_feature_batch_size: int = 4
    mlx_model_dir: Path | None = None
    mlx_model_revision: str = "38eced50afd50303f207c0165d0299991373c683"
    mlx_vlm_revision: str = "a5deef1ce4b0b5ef2a01e5a02a36805878a7ee3c"
    mlx_image_size: int = 336
    mlx_frame_stride: int = 2
    fixture_report_dir: Path = Path("reports")

    def resolved_sam2_frame_stride(self) -> int:
        """Translate the requested SAM2 rate into extracted-frame steps."""
        if self.sam2_frame_stride is not None:
            return max(1, self.sam2_frame_stride)
        if self.analysis_fps <= 0 or self.sam2_tracking_fps <= 0:
            raise ValueError("REBUILT_ANALYSIS_FPS and SAM2_TRACKING_FPS must be greater than zero.")
        return max(1, math.ceil(self.analysis_fps / self.sam2_tracking_fps))

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        configured_env_file = os.getenv("REBUILT_ENV_FILE")
        env_file = Path(configured_env_file) if configured_env_file else project_root / ".env"
        if not env_file.is_absolute():
            env_file = project_root / env_file
        _load_dotenv(env_file)
        precision = os.getenv("SAM3_PRECISION", "bf16").lower()
        checkpoint_name = "sam3-bf16.pt" if precision == "bf16" else "sam3.pt"
        local_checkpoint = project_root / ".models" / "sam3" / checkpoint_name
        local_mlx_model = project_root / ".models" / "mlx-sam3"
        local_sam2_checkpoint = project_root / ".models" / "sam2" / "sam2.1_hiera_small.pt"
        configured_dir = data_dir or Path(os.getenv("REBUILT_DATA_DIR", ".data"))
        configured_checkpoint = os.getenv("SAM3_CHECKPOINT")
        configured_sam2_stride = os.getenv("SAM2_FRAME_STRIDE")
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
            vision_backend=os.getenv("REBUILT_VISION_BACKEND", "auto"),
            sam3_checkpoint=Path(configured_checkpoint) if configured_checkpoint else (local_checkpoint if local_checkpoint.is_file() else None),
            sam3_bpe_path=Path(os.environ["SAM3_BPE_PATH"]) if os.getenv("SAM3_BPE_PATH") else None,
            sam3_hf_token_path=Path(os.getenv("HF_TOKEN_FILE", project_root / "private" / ".hf_token")),
            sam3_model_version=os.getenv("SAM3_MODEL_VERSION", "facebook/sam3"),
            sam3_precision=precision,
            sam3_max_gpu_headroom_mib=int(os.getenv("SAM3_MIN_GPU_HEADROOM_MIB", "500")),
            analysis_config_version=os.getenv("REBUILT_ANALYSIS_CONFIG_VERSION", "v2"),
            vision_worker=os.getenv("REBUILT_VISION_WORKER", "1").lower() not in {"0", "false", "no"},
            sam2_checkpoint=Path(os.environ["SAM2_CHECKPOINT"]) if os.getenv("SAM2_CHECKPOINT") else (local_sam2_checkpoint if local_sam2_checkpoint.is_file() else None),
            sam2_model_config=os.getenv("SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml"),
            sam2_tracking_fps=float(os.getenv("SAM2_TRACKING_FPS", "0.5")),
            sam2_frame_stride=max(1, int(configured_sam2_stride)) if configured_sam2_stride else None,
            sam2_apply_postprocessing=os.getenv("SAM2_APPLY_POSTPROCESSING", "0").lower() in {"1", "true", "yes"},
            sam2_vos_optimized=os.getenv("SAM2_VOS_OPTIMIZED", "0").lower() in {"1", "true", "yes"},
            sam2_mlx_model_id=os.getenv("SAM2_MLX_MODEL_ID", "avbiswas/sam2.1-hiera-small-mlx"),
            sam2_mlx_image_size=max(16, int(os.getenv("SAM2_MLX_IMAGE_SIZE", "768"))),
            sam2_mlx_precompute_features=os.getenv("SAM2_MLX_PRECOMPUTE_FEATURES", "1").lower() not in {"0", "false", "no"},
            sam2_mlx_feature_batch_size=max(1, int(os.getenv("SAM2_MLX_FEATURE_BATCH_SIZE", "4"))),
            mlx_model_dir=Path(os.getenv("MLX_SAM3_MODEL_DIR", local_mlx_model)),
            mlx_model_revision=os.getenv("MLX_SAM3_MODEL_REVISION", "38eced50afd50303f207c0165d0299991373c683"),
            mlx_vlm_revision=os.getenv("MLX_VLM_REVISION", "a5deef1ce4b0b5ef2a01e5a02a36805878a7ee3c"),
            mlx_image_size=max(1, int(os.getenv("MLX_SAM3_IMAGE_SIZE", "336"))),
            mlx_frame_stride=max(1, int(os.getenv("MLX_SAM3_FRAME_STRIDE", "2"))),
            fixture_report_dir=Path(os.getenv("REBUILT_FIXTURE_REPORT_DIR", project_root / "reports")),
        )
