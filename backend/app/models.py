from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "extracting", "generating", "ready", "failed"]


class Frame(BaseModel):
    frameId: str = Field(min_length=1)
    timestampSeconds: float = Field(ge=0)
    imageUrl: str = Field(min_length=1)


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
    guide: Guide | None = None
    error: JobError | None = None


class JobCreated(BaseModel):
    jobId: str = Field(min_length=1)
