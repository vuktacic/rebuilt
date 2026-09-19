from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "extracting", "analyzing", "generating", "ready", "failed"]
EventKind = Literal["attach", "detach", "uncertain_change"]
TrackVisibility = Literal["visible", "occluded", "lost"]
TrackMembership = Literal["separate", "attached", "unknown"]


class Frame(BaseModel):
    frameId: str = Field(min_length=1)
    timestampSeconds: float = Field(ge=0)
    imageUrl: str = Field(min_length=1)


class TrackSummary(BaseModel):
    trackId: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    firstTimestampSeconds: float = Field(ge=0)
    lastTimestampSeconds: float = Field(ge=0)
    visibility: TrackVisibility = "visible"
    membership: TrackMembership = "unknown"


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
    status: JobStatus
    frames: list[Frame] = Field(default_factory=list)
    tracks: list[TrackSummary] = Field(default_factory=list)
    events: list[AnalysisEvent] = Field(default_factory=list)
    analysis: AnalysisInfo | None = None
    guide: Guide | None = None
    error: JobError | None = None


class JobCreated(BaseModel):
    jobId: str = Field(min_length=1)
