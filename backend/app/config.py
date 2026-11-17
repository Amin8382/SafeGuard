from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CLASSES = ["person", "helmet", "vest", "no_helmet"]
MODEL_PATH = Path(os.getenv("SAFEGUARD_MODEL_PATH", str(PROJECT_ROOT / "models" / "best.onnx")))
CONFIDENCE = float(os.getenv("SAFEGUARD_CONFIDENCE", "0.25"))
IOU = float(os.getenv("SAFEGUARD_IOU", "0.45"))
MAX_DETECTIONS = int(os.getenv("SAFEGUARD_MAX_DETECTIONS", "100"))
MAX_UPLOAD_BYTES = int(float(os.getenv("SAFEGUARD_MAX_UPLOAD_MB", "12")) * 1024 * 1024)
CORS_ORIGINS = [
    x.strip()
    for x in os.getenv(
        "SAFEGUARD_CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    ).split(",")
    if x.strip()
]
