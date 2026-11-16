from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .utils import find_metric, load_yaml


def dataset_yaml(processed: Path, variant: str) -> Path:
    if variant == "merged":
        return processed / "safeguard.yaml"
    return processed / "variants" / variant / "safeguard.yaml"


def main():
    parser = argparse.ArgumentParser(description="Run the complete SafeGuard experiment matrix.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--matrix", default="configs/experiments.yaml")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    matrix = load_yaml(Path(args.matrix))
    processed = Path(args.processed)
    candidates = Path("models/candidates")
    reports = Path("reports/experiments")
    candidates.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in matrix.get("experiments", []):
        name = exp["name"]
        if args.only and name not in args.only:
            continue
        data = dataset_yaml(processed, exp["dataset"])
        if not data.exists():
            raise FileNotFoundError(f"Dataset for {name} not found: {data}")
        weights = candidates / f"{name}.pt"
        metrics_path = reports / f"{name}.json"
        cmd = [
            sys.executable,
            "-m",
            "src.training.train",
            "--params",
            args.params,
            "--data",
            str(data),
            "--run-name",
            name,
            "--output-weights",
            str(weights),
            "--metrics-output",
            str(metrics_path),
            "--overrides-json",
            json.dumps(exp.get("train", {})),
        ]
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        map5095 = find_metric(metrics, "val/metrics/mAP50-95")
        inference_ms = find_metric(metrics, "val/speed_inference_ms")
        rows.append(
            {
                "name": name,
                "dataset": exp["dataset"],
                "weights": str(weights),
                "val_map50_95": map5095,
                "val_inference_ms": inference_ms,
                "metrics_file": str(metrics_path),
            }
        )

    if not rows:
        raise RuntimeError("No experiments were executed.")

    target_ms = float(matrix.get("target_inference_ms", 50.0))
    realtime_candidates = [
        r for r in rows
        if r["val_inference_ms"] is not None and r["val_inference_ms"] <= target_ms
    ]
    candidate_pool = realtime_candidates or rows
    ranked = sorted(
        candidate_pool,
        key=lambda r: (
            -(r["val_map50_95"] if r["val_map50_95"] is not None else -1.0),
            r["val_inference_ms"] if r["val_inference_ms"] is not None else float("inf"),
        ),
    )
    winner = ranked[0]
    shutil.copy2(winner["weights"], "models/best.pt")

    summary_json = {
        "selection_policy": (
            f"highest validation mAP50-95 among models meeting <= {target_ms} ms inference; "
            "lower inference latency breaks ties. If no model meets the target, rank all models and flag it."
        ),
        "realtime_target_ms": target_ms,
        "realtime_target_met": bool(realtime_candidates),
        "winner": winner,
        "experiments": rows,
    }
    Path("reports/experiment_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )
    with Path("reports/experiment_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
