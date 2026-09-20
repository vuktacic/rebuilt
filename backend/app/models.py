from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


JobMode = Literal["automated", "manual"]
JobStatus = Literal["queued", "extracting", "annotating", "pairing", "analyzing", "generating", "ready", "failed"]
EventKind = Literal["attach", "detach", "uncertain_change"]
TrackVisibility = Literal["visible", "occluded", "lost"]
TrackMembership = Literal["separate", "attached", "unknown"]


class Frame(BaseModel):
    frameId: str = Field(min_length=1)
    sourceIndex: int | None = Field(default=None, ge=0)
    timestampSeconds: float = Field(ge=0)
    assemblyTimeSeconds: float | None = Field(default=None, ge=0)
    imageUrl: str = Field(min_length=1)


class ManualPair(BaseModel):
    pairId: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    beforeFrameId: str = Field(min_length=1)
    afterFrameId: str = Field(min_length=1)


class ManualPairsRequest(BaseModel):
    revision: int = Field(ge=0)
    pairs: list[ManualPair] = Field(min_length=1)


class TrackSummary(BaseModel):
    trackId: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    firstTimestampSeconds: float = Field(ge=0)
    lastTimestampSeconds: float = Field(ge=0)
    visibility: TrackVisibility = "visible"
    membership: TrackMembership = "unknown"


class PointPrompt(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class PartAnnotation(BaseModel):
    """One named object prompt on an extracted source frame."""

    name: str = Field(min_length=1, max_length=80)
    frameIndex: int = Field(ge=0)
    points: list[PointPrompt] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    box: tuple[float, float, float, float] | None = None


class AnnotationRequest(BaseModel):
    annotations: list[PartAnnotation] = Field(min_length=1)


class TrackObservation(BaseModel):
    frameIndex: int = Field(ge=0)
    centroid: tuple[float, float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    orientationDegrees: float | None = None
    visible: bool


class PartTrack(BaseModel):
    partId: int = Field(ge=1)
    name: str = Field(min_length=1)
    observations: list[TrackObservation] = Field(default_factory=list)
    attachmentStartFrame: int | None = Field(default=None, ge=0)
    attachmentEndFrame: int | None = Field(default=None, ge=0)


class AnalysisEvent(BaseModel):
    eventId: str = Field(min_length=1)
    kind: EventKind
    startTimestampSeconds: float = Field(ge=0)
    endTimestampSeconds: float = Field(ge=0)
    affectedTrackIds: list[str] = Field(min_length=1)
    beforeFrameId: str = Field(min_length=1)
    afterFrameId: str = Field(min_length=1)
    evidenceStrength: float = Field(ge=0, le=1)
    uncertainty: str | None = None
    evidence: str = Field(min_length=1)


class AnalysisInfo(BaseModel):
    backend: str = Field(min_length=1)
    modelVersion: str = Field(min_length=1)
    configVersion: str = Field(min_length=1)
    durationSeconds: float | None = Field(default=None, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)


class GuideStep(BaseModel):
    text: str = Field(min_length=1)
    frameId: str = Field(min_length=1)
    uncertainty: str | None = None


class Guide(BaseModel):
    title: str = Field(min_length=1)
    steps: list[GuideStep] = Field(min_length=1)


class JobError(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class JobResponse(BaseModel):
    jobId: str = Field(min_length=1)
    mode: JobMode = "automated"
    status: JobStatus
    frames: list[Frame] = Field(default_factory=list)
    revision: int = Field(default=0, ge=0)
    manualPairs: list[ManualPair] = Field(default_factory=list)
    tracks: list[TrackSummary] = Field(default_factory=list)
    annotations: list[PartAnnotation] = Field(default_factory=list)
    partTracks: list[PartTrack] = Field(default_factory=list)
    trackingProgress: float | None = Field(default=None, ge=0, le=1)
    events: list[AnalysisEvent] = Field(default_factory=list)
    analysis: AnalysisInfo | None = None
    guide: Guide | None = None
    error: JobError | None = None


class JobCreated(BaseModel):
    jobId: str = Field(min_length=1)
