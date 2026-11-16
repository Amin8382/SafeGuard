# SafeGuard - Complete Milestone Plan

This file is the single project roadmap. A milestone being **implemented** means its code/configuration/tests exist in the repository. A milestone being **executed** means the real source datasets and GPU run have produced its generated artifacts. We do not fake training results.

## Milestone 01 - Foundation and architecture

**Goal:** establish the end-to-end project structure before training.

Implemented:
- repository structure;
- canonical class ontology;
- FastAPI/React/Docker skeleton;
- DVC and MLflow integration points;
- testing structure.

Acceptance:
- Python code imports/compiles;
- backend test suite starts cleanly;
- repository has one documented architecture.

## Milestone 02 - Multi-source dataset ingestion and semantic intersection

**Goal:** turn the Roboflow and Ultralytics sources into one consistent problem.

Implemented:
- source YAML discovery;
- class-name-based mapping;
- canonical classes: `person`, `helmet`, `vest`, `gloves`, `no_helmet`;
- source-only annotation filtering;
- source audit against `dataset_lock.yaml`;
- DVC dependency graph.

Acceptance:
- both datasets audit successfully or explicitly report release drift;
- only intersection labels survive harmonization.

## Milestone 03 - Data quality, deduplication and leakage control

**Goal:** create trustworthy train/validation/test data.

Implemented:
- invalid-label and invalid-box audit;
- image/box-size statistics;
- SHA-256 cross-source exact deduplication;
- perceptual near-duplicate candidate report;
- deterministic content-hash split after deduplication;
- per-image provenance CSV;
- merged, Roboflow-only and Ultralytics-only variants;
- dataset card and data-quality report.

Acceptance:
- an exact duplicate cannot exist in two final splits;
- final rebuilds are deterministic;
- every final image is traceable to a source.

## Milestone 04 - Baseline training and MLflow experiment design

**Goal:** answer whether multi-source merging helps.

Implemented experiment matrix:
1. YOLO11n - Roboflow intersection only;
2. YOLO11n - Ultralytics intersection only;
3. YOLO11n - merged intersection;
4. YOLO11s - merged intersection;
5. YOLO11n 768px - merged intersection with controlled augmentation.

Tracked in MLflow:
- model/data/run configuration;
- dataset manifest hash;
- Git commit SHA;
- training metrics;
- validation metrics;
- held-out test metrics when available;
- training plots and best checkpoint.

Acceptance:
- runs are comparable because ontology/split policy are controlled;
- each run is traceable to code and data.

## Milestone 05 - Selection, final evaluation, ONNX and registry

**Goal:** select a deployment candidate using measured evidence.

Implemented:
- real-time candidate gate (<= 50 ms validation inference);
- candidate ranking by validation mAP50-95;
- inference latency as tiebreaker;
- automatic copy to `models/best.pt`;
- separate held-out test evaluation;
- dynamic ONNX export;
- PT-vs-ONNX latency benchmark;
- MLflow ONNX Model Registry registration.

Acceptance:
- selection happens before looking at final test performance;
- deployment model is reproducibly linked to the experiment that produced it.

## Milestone 06 - FastAPI + ONNX Runtime production inference

**Goal:** make the selected model consumable by an application.

Implemented:
- lazy ONNX Runtime session;
- CUDA provider preference when available;
- YOLO letterbox preprocessing;
- YOLO output decoding;
- confidence threshold + NMS;
- coordinate restoration;
- Pydantic API schemas;
- async upload endpoints;
- image/frame/batch inference;
- health, model and class metadata endpoints;
- conservative compliance summary.

Acceptance:
- `/docs` documents the API;
- invalid uploads fail clearly;
- missing model returns 503 rather than fake predictions.

## Milestone 07 - React real-time application

**Goal:** provide the required modern web interface.

Implemented:
- image upload analysis;
- live webcam capture loop;
- bounding boxes/class/confidence display;
- latency and inference-FPS display;
- explicit no-helmet alerts;
- session frames/violation-frame/average-latency analytics;
- API/model readiness state.

Acceptance:
- image and webcam modes use the same production API;
- UI remains explicit about what is detected versus inferred.

## Milestone 08 - Industrialization and defense readiness

**Goal:** make the project reproducible and reviewable in a practical defense.

Implemented:
- Dockerfiles and Docker Compose;
- MLflow persistent SQLite/artifact volume;
- backend/frontend health checks;
- GitHub Actions Python/frontend/Docker CI;
- GHCR container publishing CD;
- Windows bootstrap/run scripts;
- tests;
- preflight checks;
- architecture, MLOps, runbook and defense documentation.

Acceptance:
- after a real `models/best.onnx` is generated, `docker compose up --build` launches the application stack;
- every mandatory technology can be shown and explained individually.
