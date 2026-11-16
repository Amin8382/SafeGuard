# SafeGuard MLOps Design

## DVC

DVC owns data/pipeline reproducibility.

`dvc.yaml` models:

```text
audit_sources
 -> harmonize
 -> merge
 -> near_duplicates
 -> data_report
 -> train_matrix
 -> final_evaluate
 -> export_onnx
 -> benchmark
```

Raw data is tracked with `.dvc` pointer files and stored in a team-selected DVC remote rather than Git.

## MLflow Tracking

Each training run logs:
- YOLO model variant;
- epochs/image size/batch/device/seed;
- augmentation overrides;
- dataset YAML;
- dataset manifest SHA-256;
- Git commit SHA;
- validation/test metrics;
- speed metrics;
- training curves and confusion matrices;
- selected checkpoint.

This connects **code + data + parameters + metrics + model**.

## Experiment question

The first three experiments answer the project question directly:

```text
Roboflow intersection only
vs
Ultralytics intersection only
vs
Merged intersection
```

The same canonical ontology and deterministic split strategy are used to keep the comparison meaningful.

Additional experiments test the deployment trade-off between YOLO11n/YOLO11s, input resolution and augmentation.

## Selection policy

The matrix runner selects the candidate with:

1. meet the <= 50 ms validation-inference real-time target when possible;
2. highest validation mAP50-95 among eligible candidates;
3. lower validation inference latency as the tiebreaker.

If no candidate meets the target, the summary records that explicitly.

The final held-out test split is evaluated only after model selection.

## ONNX

The selected `best.pt` is exported to dynamic ONNX and loaded with ONNX Runtime. The API prefers CUDAExecutionProvider when it is installed; otherwise it uses CPU.

The benchmark compares PT and ONNX on the same held-out images and produces measured latency/FPS instead of assuming optimization always helps.

## Model Registry

`src.training.register_onnx` logs the ONNX graph using MLflow's ONNX flavor and registers it as `SafeGuard-PPE`.

## GitHub Actions

CI validates code/tests/buildability. CD publishes backend/frontend container images to GHCR on the main branch and release tags.

Training is intentionally not run in CI: it requires the DVC data and GPU resources and belongs to the experiment pipeline, not the unit-test pipeline.
