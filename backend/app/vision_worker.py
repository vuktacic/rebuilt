from __future__ import annotations

import json
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from .config import Settings
from .vision import Sam3VisionAnalyzer


def main() -> int:
    request = json.loads(sys.stdin.read())
    settings = Settings.from_env()
    with redirect_stdout(io.StringIO()):
        result = Sam3VisionAnalyzer(settings)._analyze_loaded(
            Path(request["frame_dir"]),
            int(request["frame_count"]),
            str(request["job_id"]),
        )
    sys.stdout.write(json.dumps({
        "events": [event.model_dump() for event in result.events],
        "tracks": [track.model_dump() for track in result.tracks],
        "analysis": result.analysis.model_dump(),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
