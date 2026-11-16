# SafeGuard Practical Defense Guide

## Recommended live sequence

1. `git log --oneline --decorate --graph` - show structured project evolution.
2. `dvc dag` - explain the data/model dependency graph.
3. Open `reports/DATA_QUALITY_REPORT.md` and `data/processed/DATASET_CARD.md`.
4. Open MLflow - compare Roboflow-only, Ultralytics-only and merged runs.
5. Explain why the best model was selected.
6. Show `models/best.pt -> models/best.onnx` and `reports/benchmark.json`.
7. `docker compose up --build`.
8. Open React and run image detection.
9. Activate webcam and show live boxes/latency/violations.
10. Open `/docs` and call an API endpoint directly.
11. Open source code in the IDE when questioned.

## Questions every team member should be able to answer

### Why not simply concatenate the two datasets?

Because their ontologies are different. A union creates inconsistent supervision for source-specific classes. SafeGuard first maps class names into a semantic intersection and drops source-only boxes.

### Why map by class name rather than class ID?

Class IDs are source-specific and can change between dataset releases. Class-name resolution prevents a numeric reorder from silently mapping `helmet` to another class.

### Why deduplicate before train/validation/test splitting?

If the same image appears in train and test, the test metric measures memorization/leakage rather than generalization. SafeGuard hashes images, removes exact duplicates, then splits by content hash.

### Why DVC if Git already versions code?

Git is optimized for small text/code. DVC tracks large datasets/models and their pipeline dependencies while Git tracks the small `.dvc`, `dvc.yaml`, configuration and source-code files.

### What does `dvc repro` do?

It computes which stages are stale from changed dependencies/parameters and reruns the necessary stages in dependency order.

### What is MLflow Tracking versus Model Registry?

Tracking stores run parameters, metrics and artifacts. Model Registry manages named model versions/candidates and their lifecycle after experiments.

### Why MLflow if Ultralytics already produces `results.csv`?

Ultralytics records one training run locally. MLflow provides a comparable experiment history and ties runs to data/code metadata and production model artifacts.

### What is mAP50 vs mAP50-95?

mAP50 evaluates detections at IoU 0.50. mAP50-95 averages mAP across IoU thresholds 0.50 through 0.95, so it is stricter about localization quality.

### Precision versus recall?

Precision asks how many predicted detections are correct. Recall asks how many real objects were found. In safety monitoring, missing violations can make recall particularly important, but false alarms also matter operationally.

### What is NMS?

Non-Maximum Suppression removes overlapping duplicate predictions by keeping high-confidence boxes and suppressing lower-confidence boxes above an IoU overlap threshold.

### Why YOLO11n and YOLO11s?

They provide different accuracy/speed/model-size trade-offs. The project measures rather than assumes which one is suitable for real-time deployment.

### Why ONNX?

ONNX separates deployment from the original training framework and enables optimized runtime providers. SafeGuard benchmarks PT and ONNX rather than claiming ONNX is automatically faster.

### Why FastAPI and Pydantic?

FastAPI provides asynchronous REST endpoints and generated OpenAPI/Swagger documentation. Pydantic gives explicit validated response contracts.

### Is a worker compliant if `no_helmet` is absent?

Not necessarily. SafeGuard intentionally reports `NO_EXPLICIT_VIOLATION`. Absence of a negative label is not proof that every required item is correctly worn.

### Why doesn't SafeGuard flag missing vest?

The confirmed V1 canonical intersection does not contain a shared `no_vest` class. We do not invent a missing-vest label from absence alone.

### What happens when `models/best.onnx` is missing?

The API stays healthy but reports `model_ready=false`; prediction endpoints return HTTP 503. This is safer than returning fabricated predictions.

### What does Docker Compose orchestrate?

The React frontend, FastAPI inference backend and MLflow service, with shared ports/volumes and health dependencies.

### What does GitHub Actions do?

CI compiles/lints/tests Python, builds the React frontend and verifies Docker builds. CD publishes backend/frontend images to GHCR on main/tags.
