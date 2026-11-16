from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from ultralytics import YOLO

from .utils import file_sha256, git_commit, load_yaml, ultralytics_metrics, write_json


def run_training(
    params_path: Path,
    data_path: Path,
    run_name: str,
    output_weights: Path,
    metrics_output: Path,
    overrides: dict | None = None,
) -> dict:
    params = load_yaml(params_path)
    train_cfg = dict(params["train"])
    train_cfg.pop("run_name", None)
    if overrides:
        train_cfg.update(overrides)
    augmentation = dict(train_cfg.pop("augmentation", {}) or {})
    project_name = params["project"]["name"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", params.get("mlflow", {}).get("tracking_uri"))
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(project_name)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params({f"train.{k}": v for k, v in train_cfg.items()})
        for key, value in augmentation.items():
            mlflow.log_param(f"augmentation.{key}", value)
        mlflow.log_param("dataset_yaml", str(data_path))
        mlflow.log_param("run_name", run_name)

        manifest = data_path.parent / "dataset_manifest.json"
        manifest_hash = file_sha256(manifest)
        if manifest_hash:
            mlflow.set_tag("dataset_manifest_sha256", manifest_hash)
        commit = git_commit()
        if commit:
            mlflow.set_tag("git_commit", commit)

        for artifact in [
            data_path,
            manifest,
            data_path.parent / "build_report.json",
            data_path.parent / "DATASET_CARD.md",
        ]:
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="dataset")

        model = YOLO(str(train_cfg["model"]))
        results = model.train(
            data=str(data_path),
            epochs=int(train_cfg["epochs"]),
            imgsz=int(train_cfg["imgsz"]),
            batch=int(train_cfg["batch"]),
            device=train_cfg["device"],
            workers=int(train_cfg["workers"]),
            patience=int(train_cfg["patience"]),
            seed=int(train_cfg["seed"]),
            deterministic=bool(train_cfg.get("deterministic", True)),
            project="runs/train",
            name=run_name,
            exist_ok=True,
            **augmentation,
        )

        save_dir = Path(results.save_dir)
        best_src = save_dir / "weights" / "best.pt"
        output_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, output_weights)

        metrics: dict[str, float] = {}
        csv_path = save_dir / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                for key, value in df.iloc[-1].items():
                    try:
                        metrics[f"train_final/{str(key).strip()}"] = float(value)
                    except (TypeError, ValueError):
                        pass

        best_model = YOLO(str(output_weights))
        val_metrics = best_model.val(
            data=str(data_path),
            split="val",
            imgsz=int(train_cfg["imgsz"]),
            device=train_cfg["device"],
        )
        for key, value in ultralytics_metrics(val_metrics).items():
            metrics[f"val/{key}"] = value

        data_meta = load_yaml(data_path)
        if data_meta.get("test"):
            try:
                test_metrics = best_model.val(
                    data=str(data_path),
                    split="test",
                    imgsz=int(train_cfg["imgsz"]),
                    device=train_cfg["device"],
                )
                for key, value in ultralytics_metrics(test_metrics).items():
                    metrics[f"test/{key}"] = value
            except Exception as exc:
                mlflow.set_tag("test_evaluation_error", str(exc)[:500])

        write_json(metrics_output, metrics)
        if metrics:
            mlflow.log_metrics(metrics)

        for artifact_name in [
            "results.csv",
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png",
            "P_curve.png",
            "R_curve.png",
            "F1_curve.png",
        ]:
            artifact = save_dir / artifact_name
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="training")
        mlflow.log_artifact(str(output_weights), artifact_path="model")
        mlflow.set_tag("safeguard_stage", "training")
        mlflow.set_tag("mlflow_run_id", active_run.info.run_id)

        payload = {
            "run_id": active_run.info.run_id,
            "run_name": run_name,
            "weights": str(output_weights),
            "metrics": metrics,
            "train_config": train_cfg,
            "augmentation": augmentation,
        }
        print(json.dumps(payload, indent=2))
        return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--output-weights", default="models/best.pt")
    parser.add_argument("--metrics-output", default="metrics.json")
    parser.add_argument("--overrides-json", default="{}")
    args = parser.parse_args()
    run_training(
        Path(args.params),
        Path(args.data),
        args.run_name,
        Path(args.output_weights),
        Path(args.metrics_output),
        json.loads(args.overrides_json),
    )


if __name__ == "__main__":
    main()
