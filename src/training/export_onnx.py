from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export the selected SafeGuard checkpoint to ONNX.")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="models/best.onnx")
    parser.add_argument("--simplify", action="store_true", help="Run ONNX graph simplification when dependencies are installed.")
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=True,
            simplify=args.simplify,
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output.resolve():
        shutil.copy2(exported, output)
    print(f"ONNX model ready: {output}")


if __name__ == "__main__":
    main()
