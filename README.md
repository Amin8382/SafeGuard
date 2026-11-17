# SafeGuard

**Real-time construction-site PPE detection with an end-to-end Vision + MLOps pipeline.**

SafeGuard is the complete course project: it takes two heterogeneous construction-safety datasets, keeps their **semantic intersection**, builds a leakage-safe unified dataset, trains and compares YOLO11 detectors, tracks experiments in MLflow, versions data/pipelines with DVC, exports the selected model to ONNX, serves it through asynchronous FastAPI endpoints, and visualizes predictions from uploaded images or a live webcam in React.

## 1. Confirmed project scope

SafeGuard deliberately uses an **intersection**, not a union, of the two source ontologies.

| Canonical class | Roboflow Construction Site Safety | Ultralytics Construction-PPE |
|---|---|---|
| `person` | `Person` | `Person` |
| `helmet` | `Hardhat` | `helmet` |
| `vest` | `Safety Vest` | `vest` |
| `no_helmet` | `NO-Hardhat` | `no_helmet` |

Roboflow Construction Site Safety v27 does **not** annotate gloves, so `gloves` is intentionally excluded from the strict V1 intersection even though the legacy Construction-PPE source contains that class.

Source-only labels are excluded from V1. This avoids contradictory supervision, for example when one source contains visible boots or machinery that the other source never annotated.

## 2. What is included

- source audit and release-lock checks;
- class-name-based semantic harmonization;
- exact SHA-256 deduplication across sources;
- deterministic post-dedup train/validation/test split;
- provenance tracking for every final image;
- merged, Roboflow-only and Ultralytics-only experimental variants;
- perceptual near-duplicate review report;
- DVC data/pipeline orchestration;
- five-run YOLO11 experiment matrix;
- automatic best-candidate selection by validation mAP50-95, then inference latency;
- held-out test evaluation;
- MLflow parameters, metrics, plots, artifacts and Model Registry integration;
- ONNX export and PT-vs-ONNX benchmark;
- ONNX Runtime inference engine with NMS and coordinate restoration;
- asynchronous FastAPI API with Pydantic response schemas;
- image, frame and small-batch prediction endpoints;
- React image analysis and real-time webcam modes;
- live session analytics and explicit PPE-risk messages;
- Docker Compose for frontend, backend and MLflow;
- GitHub Actions CI plus GHCR image publishing CD workflow;
- unit/integration tests and defense/runbook documentation.

## 3. Repository map

```text
SafeGuard/
├── .github/workflows/       # CI + container publishing CD
├── .dvc/                    # DVC project configuration
├── backend/
│   ├── app/                 # FastAPI + ONNX inference + schemas
│   └── tests/
├── configs/
│   ├── datasets.yaml        # semantic intersection mapping
│   ├── dataset_lock.yaml    # expected source release metadata
│   └── experiments.yaml     # controlled training matrix
├── data/
│   ├── raw/                 # source datasets; not committed to Git
│   ├── interim/             # harmonized source datasets
│   └── processed/           # unified + source-only variants
├── docs/                    # architecture, MLOps, milestones, defense guide
├── frontend/                # React/Vite application
├── models/                  # generated PT/ONNX models; not committed
├── reports/                 # generated audit/evaluation/benchmark reports
├── scripts/                 # Windows/Linux bootstrap and pipeline helpers
├── src/
│   ├── data/                # audit/harmonize/dedup/report pipeline
│   └── training/            # training/eval/benchmark/export/registry
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
└── README.md
```

## 4. Fastest Windows path

Recommended: **Python 3.11**, Git, Node 22+, Docker Desktop, NVIDIA/CUDA-capable PyTorch for GPU training.

```powershell
# from the repository root
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1
```

Check your environment:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\preflight.py
```

### Get the datasets

Install the selected legacy Ultralytics archive (1,416 images / 11 classes):

```powershell
python scripts\bootstrap_data.py --source ultralytics --ultralytics-zip C:\path\to\construction-ppe-legacy.zip
```

The current official archive is intentionally not substituted silently because it is a different 1,000-image / 12-class release.

For Roboflow, either export **Construction Site Safety** in YOLOv11 format into `data/raw/roboflow`, or set your Roboflow key and run:

```powershell
$env:ROBOFLOW_API_KEY="YOUR_KEY"
python scripts\bootstrap_data.py --source roboflow --roboflow-version 27
```

The source audit intentionally fails/flags release differences rather than silently accepting a changed dataset.

### Version the raw data with DVC

```powershell
dvc add data/raw/roboflow data/raw/ultralytics
git add data/raw/*.dvc data/raw/.gitignore
```

Add a DVC remote appropriate for your team before sharing large data files.

## 5. Run the data pipeline

```powershell
.\scripts\run_data_pipeline.ps1
```

Equivalent DVC route:

```powershell
dvc repro data_report
```

Important outputs:

```text
reports/source_audit.json
data/interim/harmonization_report.json
data/processed/safeguard.yaml
data/processed/provenance.csv
data/processed/DATASET_CARD.md
reports/near_duplicates.json
reports/DATA_QUALITY_REPORT.md
```

## 6. Run MLflow and the complete model pipeline

Start MLflow in one terminal:

```powershell
.\.venv\Scripts\Activate.ps1
mlflow server --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000`.

Then, in another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\run_full_pipeline.ps1
```

That sequence performs:

```text
Data audit/harmonization/merge
    -> 5 controlled YOLO11 experiments
    -> automatic candidate ranking
    -> models/best.pt
    -> held-out test evaluation
    -> models/best.onnx
    -> PT-vs-ONNX benchmark
    -> MLflow Model Registry registration
```

No metric is hard-coded in this repository. mAP, precision, recall, latency and FPS are generated from your real data/model runs.

## 7. Run the application

### Development mode

```powershell
.\scripts\run_dev.ps1
```

Or manually:

```powershell
# terminal 1
.\.venv\Scripts\Activate.ps1
mlflow server --host 0.0.0.0 --port 5000

# terminal 2
.\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --port 8000

# terminal 3
cd frontend
npm install
npm run dev
```

URLs:

- React: `http://localhost:5173`
- FastAPI: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`

### Docker Compose - defense/demo path

After `models/best.onnx` exists:

```powershell
docker compose up --build
```

Then:

- SafeGuard UI: `http://localhost:3000`
- Swagger: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`

## 8. API

```text
GET  /health
GET  /classes
GET  /model/info
POST /predict/image
POST /predict/frame
POST /predict/batch
GET  /docs
```

The compliance layer is intentionally conservative. `no_helmet` is an explicit violation label. Absence of that label is reported as **NO_EXPLICIT_VIOLATION**, not as proof of complete PPE compliance.

## 9. Tests and CI/CD

Local:

```powershell
pytest backend/tests tests -q
ruff check src backend tests scripts
```

GitHub Actions performs:

```text
Python compile + Ruff + tests
Frontend dependency install + production build
Docker backend/frontend builds
```

On `main` or a `v*` tag, the CD workflow publishes frontend/backend images to GitHub Container Registry.

## 10. Milestones

The entire project is represented in one repository. The milestone history is documented in [`docs/MILESTONES.md`](docs/MILESTONES.md), and the local Git history is structured to match it.

1. Foundation and architecture
2. Multi-source data engineering + DVC
3. Data QA, deduplication and reproducible splitting
4. YOLO11 experimentation + MLflow
5. Evaluation, model selection, ONNX and registry
6. FastAPI production inference
7. React image/webcam application
8. Docker, CI/CD, testing and defense readiness

## 11. Important generated artifacts

These are intentionally not committed to normal Git history:

```text
data/raw/
data/interim/
data/processed/
models/*.pt
models/*.onnx
mlruns/
mlflow_store/
runs/
```

Raw/processed data belongs in DVC; trained models belong in MLflow/your chosen model artifact store. This keeps the Git repository small and reproducible.

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the exact operational sequence and [`docs/DEFENSE_GUIDE.md`](docs/DEFENSE_GUIDE.md) for the technical defense.

## 12. Push this complete milestone history to GitHub

The repository is designed to carry milestone tags. After an empty remote exists:

```powershell
.\scripts\push_to_github.ps1 -RemoteUrl https://github.com/<owner>/SafeGuard.git
```

This pushes the final `main` branch and every milestone/release tag.
