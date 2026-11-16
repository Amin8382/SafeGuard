from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def ultralytics_metrics(metrics_obj) -> dict[str, float]:
    out: dict[str, float] = {}
    if metrics_obj is None:
        return out
    results_dict = getattr(metrics_obj, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                pass
    speed = getattr(metrics_obj, "speed", None)
    if isinstance(speed, dict):
        for key, value in speed.items():
            try:
                out[f"speed_{key}_ms"] = float(value)
            except (TypeError, ValueError):
                pass
    return out


def find_metric(metrics: dict[str, float], needle: str) -> float | None:
    needle = needle.lower().replace(" ", "")
    for key, value in metrics.items():
        normalized = key.lower().replace(" ", "")
        if needle in normalized:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
