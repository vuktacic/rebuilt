from pathlib import Path

import pytest

from backend.app.config import Settings, VisionRuntimeError, _load_dotenv, _read_dotenv, _read_secret_file, resolve_vision_backend


def test_dotenv_parser_reads_values_without_executing_code(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comments are ignored\n"
        "OPENAI_API_KEY=sk-test\n"
        "export OPENAI_MODEL='gpt-test'\n"
        "REBUILT_ANALYSIS_FPS=2\n"
        "BROKEN LINE\n",
        encoding="utf-8",
    )

    values = _read_dotenv(dotenv)
    environment: dict[str, str] = {"OPENAI_MODEL": "from-process"}
    _load_dotenv(dotenv, environment)

    assert values["OPENAI_API_KEY"] == "sk-test"
    assert values["OPENAI_MODEL"] == "gpt-test"
    assert environment["OPENAI_MODEL"] == "from-process"
    assert environment["REBUILT_ANALYSIS_FPS"] == "2"
    assert "BROKEN LINE" not in environment


def test_gemini_key_file_is_a_single_opaque_line(tmp_path: Path) -> None:
    token = tmp_path / ".gemini_token"
    token.write_text("test-gemini-key\n", encoding="utf-8")
    assert _read_secret_file(token) == "test-gemini-key"
    token.write_text("first\nsecond\n", encoding="utf-8")
    assert _read_secret_file(token) is None


def test_settings_load_gemini_key_file_with_environment_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = tmp_path / ".gemini_token"
    token.write_text("file-key\n", encoding="utf-8")
    dotenv = tmp_path / ".env"
    dotenv.write_text("", encoding="utf-8")
    monkeypatch.setenv("REBUILT_ENV_FILE", str(dotenv))
    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(token))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert Settings.from_env(data_dir=tmp_path).gemini_api_key == "file-key"
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    assert Settings.from_env(data_dir=tmp_path).gemini_api_key == "environment-key"


def test_auto_backend_selects_manual_sam2_tracking_on_supported_hosts(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, vision_backend="auto")

    assert resolve_vision_backend(settings, system="Darwin", machine="arm64", translated=False) == "sam2"
    assert resolve_vision_backend(settings, system="Windows", machine="AMD64", cuda_available=True) == "sam2"


def test_explicit_mlx_rejects_rosetta_and_unknown_backends(tmp_path: Path) -> None:
    with pytest.raises(VisionRuntimeError) as rosetta:
        resolve_vision_backend(
            Settings(data_dir=tmp_path, vision_backend="sam3-mlx"),
            system="Darwin",
            machine="x86_64",
            translated=True,
        )
    assert rosetta.value.code == "VISION_ROSETTA_UNSUPPORTED"

    with pytest.raises(VisionRuntimeError) as invalid:
        resolve_vision_backend(Settings(data_dir=tmp_path, vision_backend="banana"))
    assert invalid.value.code == "VISION_BACKEND_INVALID"


def test_explicit_sam2_mlx_selects_the_native_apple_silicon_backend(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, vision_backend="sam2-mlx")

    assert resolve_vision_backend(settings, system="Darwin", machine="arm64", translated=False) == "sam2-mlx"


def test_mlx_fast_profile_defaults_to_a_small_image_size_and_samples_every_other_frame(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.mlx_image_size == 336
    assert settings.mlx_frame_stride == 2


def test_sam2_fast_profile_samples_every_other_extracted_frame(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.sam2_frame_stride == 2
    assert settings.sam2_apply_postprocessing is False
