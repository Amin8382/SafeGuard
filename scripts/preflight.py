from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def main():
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
        "raw_roboflow": Path("data/raw/roboflow").exists(),
        "raw_ultralytics": Path("data/raw/ultralytics").exists(),
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
