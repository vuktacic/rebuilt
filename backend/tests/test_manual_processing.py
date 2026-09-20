from __future__ import annotations

import base64
from pathlib import Path

import pytest

from backend.app.errors import AppError
from backend.app.models import Frame, ManualPair, PairFinding
from backend.app.manual_processing import build_luna_payload, build_pair_review_payload, validate_pair_finding


@pytest.fixture
def pair_and_frames() -> tuple[ManualPair, list[Frame]]:
    return (
        ManualPair(pairId="pair-1", sequence=1, beforeFrameId="frame-0001", afterFrameId="frame-0000"),
        [
            Frame(
                frameId="frame-0001",
                sourceIndex=1,
                timestampSeconds=2.0,
                assemblyTimeSeconds=1.0,
                imageUrl="/frame-0001",
            ),
            Frame(
                frameId="frame-0000",
                sourceIndex=0,
                timestampSeconds=3.0,
                assemblyTimeSeconds=2.0,
                imageUrl="/frame-0000",
            ),
        ],
    )


def test_astra_payload_contains_only_one_labelled_pair(pair_and_frames, tmp_path: Path) -> None:
    pair, frames = pair_and_frames
    before_path = tmp_path / "before.jpg"
    after_path = tmp_path / "after.jpg"
    before_path.write_bytes(b"before-image")
    after_path.write_bytes(b"after-image")

    payload = build_pair_review_payload(pair, frames, [before_path, after_path], model="gpt-6-astra", detail="low")
    content = payload["input"][0]["content"]

    assert payload["model"] == "gpt-6-astra"
    assert [item["type"] for item in content].count("input_image") == 2
    labels = [item["text"] for item in content if item["type"] == "input_text"]
    assert any("BEFORE frame-0001" in label and "assembly=1.0s" in label for label in labels)
    assert any("AFTER frame-0000" in label and "assembly=2.0s" in label for label in labels)
    assert str(before_path) not in str(payload)
    assert base64.b64encode(b"before-image").decode() in str(payload)


def test_pair_finding_validation_rejects_references_outside_submitted_pair(pair_and_frames) -> None:
    pair, frames = pair_and_frames
    finding = PairFinding(
        pairId=pair.pairId,
        status="completed",
        action="attach",
        difference="A brick is attached.",
        uncertainty=None,
        confidence=0.9,
        evidenceFrameIds=["frame-9999"],
    )

    with pytest.raises(AppError, match="submitted pair"):
        validate_pair_finding(finding, pair, frames)


def test_pair_finding_validation_preserves_a_valid_compact_difference(pair_and_frames) -> None:
    pair, frames = pair_and_frames
    finding = PairFinding(
        pairId=pair.pairId,
        status="completed",
        action="attach",
        difference="Attach the red brick to the top plate.",
        uncertainty=None,
        confidence=0.9,
        evidenceFrameIds=[pair.afterFrameId],
    )

    validated = validate_pair_finding(finding, pair, frames)

    assert validated.pairId == pair.pairId
    assert validated.evidenceFrameIds == [pair.afterFrameId]
    assert validated.difference == "Attach the red brick to the top plate."


def test_luna_payload_contains_findings_and_frame_ids_but_no_images_or_paths(pair_and_frames, tmp_path: Path) -> None:
    pair, frames = pair_and_frames
    finding = PairFinding(
        pairId=pair.pairId,
        status="completed",
        action="attach",
        difference="Attach the red brick to the top plate.",
        confidence=0.9,
        evidenceFrameIds=[pair.afterFrameId],
    )
    payload = build_luna_payload([finding], [pair], frames, model="gpt-5.6-luna")

    serialized = str(payload)
    assert payload["model"] == "gpt-5.6-luna"
    assert "input_image" not in serialized
    assert "Attach the red brick" in serialized
    assert pair.afterFrameId in serialized
    assert str(tmp_path) not in serialized
