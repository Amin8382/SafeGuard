from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from .utils import ultralytics_metrics, write_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate the selected SafeGuard checkpoint on the held-out test split.")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--data", default="data/processed/safeguard.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="reports/final_test_metrics.json")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics_obj = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        project="runs/eval",
        name="final-test",
        exist_ok=True,
        plots=True,
    )
    metrics = ultralytics_metrics(metrics_obj)
    per_class = {}
    maps = getattr(getattr(metrics_obj, "box", None), "maps", None)
    if maps is not None:
        names = getattr(model, "names", {})
        for idx, value in enumerate(maps):
            name = names.get(idx, str(idx)) if isinstance(names, dict) else str(idx)
            per_class[name] = {"mAP50-95": float(value)}
    payload = {"metrics": metrics, "per_class": per_class}
    write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
