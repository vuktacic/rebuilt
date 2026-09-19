from pathlib import Path

from backend.app.config import _load_dotenv, _read_dotenv


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
