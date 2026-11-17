from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def dataset_readiness(root: Path) -> dict:
    directory_exists = root.exists() and root.is_dir()
    image_count = 0
    label_count = 0
    yaml_present = False

    if directory_exists:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                image_count += 1
            elif suffix == ".txt" and "labels" in path.parts:
                label_count += 1
            elif suffix in {".yaml", ".yml"}:
                yaml_present = True

    dataset_ready = directory_exists and image_count > 0 and label_count > 0 and yaml_present
    return {
        "directory_exists": directory_exists,
        "image_count": image_count,
        "label_count": label_count,
        "yaml_present": yaml_present,
        "dataset_ready": dataset_ready,
    }


def main():
    roboflow = dataset_readiness(Path("data/raw/roboflow"))
    ultralytics = dataset_readiness(Path("data/raw/ultralytics"))

    checks = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": command_exists("git"),
        "docker": command_exists("docker"),
        "node": command_exists("node"),
        "npm": command_exists("npm"),
        "dvc_module": importlib.util.find_spec("dvc") is not None,
        "mlflow_module": importlib.util.find_spec("mlflow") is not None,
        "ultralytics_module": importlib.util.find_spec("ultralytics") is not None,
        "onnxruntime_module": importlib.util.find_spec("onnxruntime") is not None,
        "raw_roboflow": roboflow["dataset_ready"],
        "raw_roboflow_details": roboflow,
        "raw_ultralytics": ultralytics["dataset_ready"],
        "raw_ultralytics_details": ultralytics,
        "processed_dataset": Path("data/processed/safeguard.yaml").exists(),
        "pt_model": Path("models/best.pt").exists(),
        "onnx_model": Path("models/best.onnx").exists(),
    }
    try:
        import torch

        checks["torch"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        checks["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        checks["torch"] = None
        checks["cuda_available"] = False
        checks["cuda_device"] = None

    print(json.dumps(checks, indent=2))
    if sys.version_info[:2] != (3, 11):
        print("WARNING: Python 3.11 is the project baseline.")


if __name__ == "__main__":
    main()
