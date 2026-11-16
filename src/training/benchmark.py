from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from backend.app.config import CANONICAL_CLASSES
from backend.app.inference import OnnxYoloDetector
from src.data.common import iter_images, load_yaml


def resolve_images(data_yaml: Path, split: str, limit: int) -> list[Path]:
    meta = load_yaml(data_yaml)
    root = Path(meta.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    value = meta.get(split)
    if not value:
        raise ValueError(f"Split {split} missing from {data_yaml}")
    folder = Path(value)
    if not folder.is_absolute():
        folder = root / folder
    return list(iter_images(folder))[:limit]


def stats(times: list[float]) -> dict:
    ordered = sorted(times)
    p95_index = max(0, int(round(0.95 * (len(ordered) - 1))))
    avg = statistics.mean(times)
    return {
        "images": len(times),
        "avg_ms": avg,
        "median_ms": statistics.median(times),
        "p95_ms": ordered[p95_index],
        "fps_from_avg": 1000.0 / avg if avg > 0 else 0.0,
    }


def benchmark_pt(model_path: str, images: list[Path], imgsz: int, device: str, warmup: int) -> dict:
    model = YOLO(model_path)
    for image in images[: min(warmup, len(images))]:
        model.predict(str(image), imgsz=imgsz, device=device, verbose=False)
    times = []
    for image in images:
        started = time.perf_counter()
        model.predict(str(image), imgsz=imgsz, device=device, verbose=False)
        times.append((time.perf_counter() - started) * 1000)
    return {"model": model_path, "device": device, **stats(times)}


def benchmark_onnx(model_path: str, images: list[Path], warmup: int) -> dict:
    detector = OnnxYoloDetector(Path(model_path), CANONICAL_CLASSES)
    decoded = []
    for path in images:
        image = cv2.imread(str(path))
        if image is not None:
            decoded.append(image)
    if not decoded:
        raise RuntimeError("No benchmark images could be decoded.")
    for image in decoded[: min(warmup, len(decoded))]:
        detector.predict(image)
    times = []
    for image in decoded:
        started = time.perf_counter()
        detector.predict(image)
        times.append((time.perf_counter() - started) * 1000)
    return {
        "model": model_path,
        "provider": detector.provider,
        **stats(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark selected PT and production ONNX SafeGuard inference.")
    parser.add_argument("--data", default="data/processed/safeguard.yaml")
    parser.add_argument("--pt", default="models/best.pt")
    parser.add_argument("--onnx", default="models/best.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pt-device", default="0")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="reports/benchmark.json")
    args = parser.parse_args()

    images = resolve_images(Path(args.data), "test", args.limit)
    if not images:
        raise RuntimeError("No test images found for benchmarking.")
    payload = {
        "pt": benchmark_pt(args.pt, images, args.imgsz, args.pt_device, args.warmup),
        "onnx": benchmark_onnx(args.onnx, images, args.warmup),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
