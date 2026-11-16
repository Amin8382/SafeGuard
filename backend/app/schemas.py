from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_loaded: bool
    provider: str | None = None


class ClassesResponse(BaseModel):
    classes: list[str]


class ModelInfoResponse(BaseModel):
    project: str
    task: str
    format: str
    model_path: str
    model_ready: bool
    model_loaded: bool
    provider: str | None = None
    input_size: list[int]
    classes: list[str]
    confidence_threshold: float
    iou_threshold: float


class DetectionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)


class ImageShape(BaseModel):
    width: int
    height: int


class ComplianceSummary(BaseModel):
    counts: dict[str, int]
    violations: dict[str, int]
    status: str
    messages: list[str]


class PredictionResponse(BaseModel):
    image: ImageShape
    latency_ms: float
    inference_fps: float
    detections: list[DetectionResponse]
    summary: ComplianceSummary


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
