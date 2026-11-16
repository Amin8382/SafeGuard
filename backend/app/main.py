from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .compliance import summarize
from .config import (
    CANONICAL_CLASSES,
    CONFIDENCE,
    CORS_ORIGINS,
    IOU,
    MAX_DETECTIONS,
    MAX_UPLOAD_BYTES,
    MODEL_PATH,
)
from .inference import ModelNotReadyError, OnnxYoloDetector
from .schemas import (
    BatchPredictionResponse,
    ClassesResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

app = FastAPI(
    title="SafeGuard API",
    version="1.0.0",
    description="Asynchronous REST API for real-time construction-site PPE detection.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = OnnxYoloDetector(
    MODEL_PATH,
    CANONICAL_CLASSES,
    confidence=CONFIDENCE,
    iou=IOU,
    max_detections=MAX_DETECTIONS,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_ready=detector.ready,
        model_loaded=detector.loaded,
        provider=detector.provider,
    )


@app.get("/classes", response_model=ClassesResponse)
async def classes():
    return ClassesResponse(classes=CANONICAL_CLASSES)


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        project="SafeGuard",
        task="real-time PPE object detection",
        format="ONNX",
        model_path=str(MODEL_PATH),
        model_ready=detector.ready,
        model_loaded=detector.loaded,
        provider=detector.provider,
        input_size=list(detector.input_hw),
        classes=CANONICAL_CLASSES,
        confidence_threshold=CONFIDENCE,
        iou_threshold=IOU,
    )


def _decode_image(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit.")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image.")
    return image


def _run_prediction(image: np.ndarray) -> PredictionResponse:
    if not detector.ready:
        raise HTTPException(status_code=503, detail="Model not ready. Export models/best.onnx first.")
    started = time.perf_counter()
    try:
        detections = detector.predict(image)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000
    inference_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    return PredictionResponse(
        image={"width": int(image.shape[1]), "height": int(image.shape[0])},
        latency_ms=round(latency_ms, 2),
        inference_fps=round(inference_fps, 2),
        detections=[
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 5),
                "bbox": [round(x, 2) for x in d.bbox],
            }
            for d in detections
        ],
        summary=summarize(detections),
    )


async def _predict_upload(file: UploadFile) -> PredictionResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")
    raw = await file.read()
    return _run_prediction(_decode_image(raw))


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    return await _predict_upload(file)


@app.post("/predict/frame", response_model=PredictionResponse)
async def predict_frame(file: UploadFile = File(...)):
    return await _predict_upload(file)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files supplied.")
    if len(files) > 8:
        raise HTTPException(status_code=400, detail="Batch limit is 8 images.")
    results = [await _predict_upload(file) for file in files]
    return BatchPredictionResponse(results=results)
