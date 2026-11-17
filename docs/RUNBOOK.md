# SafeGuard Runbook - Windows / VS Code

## A. Clone/open

```powershell
git clone <YOUR-SAFEGUARD-REPOSITORY-URL>
cd SafeGuard
code .
```

All following commands are run from the repository root unless explicitly stated.

## B. Bootstrap Python

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1
.\.venv\Scripts\Activate.ps1
python scripts\preflight.py
```

## C. Put/fetch source datasets

Expected roots:

```text
data/raw/roboflow/
data/raw/ultralytics/
```

Ultralytics:

```powershell
python scripts\bootstrap_data.py --source ultralytics
```

Roboflow with API key:

```powershell
$env:ROBOFLOW_API_KEY="..."
python scripts\bootstrap_data.py --source roboflow --roboflow-version 27
```

Or export the public Construction Site Safety dataset in YOLOv11 format manually and extract it under `data/raw/roboflow`.

## D. DVC-version raw data

```powershell
dvc add data/raw/roboflow data/raw/ultralytics
git add data/raw/*.dvc data/raw/.gitignore
```

Configure a DVC remote chosen by the team. Do not commit the image folders directly to Git.

## E. Start MLflow

New terminal:

```powershell
.\.venv\Scripts\Activate.ps1
mlflow server --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000`.

## F. Build the unified dataset

```powershell
.\scripts\run_data_pipeline.ps1
```

Inspect:

```text
reports/source_audit.json
reports/DATA_QUALITY_REPORT.md
data/processed/DATASET_CARD.md
data/processed/provenance.csv
```

The source audit is a gate. If source releases differ from the lock file, investigate before training.

## G. Run the complete ML sequence

```powershell
.\scripts\run_full_pipeline.ps1
```

Generated key artifacts:

```text
models/candidates/*.pt
models/best.pt
models/best.onnx
reports/experiment_summary.json
reports/final_test_metrics.json
reports/benchmark.json
```

## H. Run development UI/API

```powershell
.\scripts\run_dev.ps1
```

Or manually use three terminals:

```powershell
# backend
.\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --port 8000
```

```powershell
# frontend
cd frontend
npm install
npm run dev
```

```powershell
# MLflow
.\.venv\Scripts\Activate.ps1
mlflow server --host 0.0.0.0 --port 5000
```

## I. Practical defense launch

After the model is exported:

```powershell
docker compose up --build
```

Open:

- `http://localhost:3000` - SafeGuard application
- `http://localhost:8000/docs` - Swagger
- `http://localhost:5000` - MLflow

Useful defense commands:

```powershell
dvc dag
dvc status
git log --oneline --decorate --graph
pytest backend/tests tests -q
docker compose ps
```

## J. Troubleshooting

### PowerShell refuses activation

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### Backend says model pending

Expected until this exists:

```text
models/best.onnx
```

Run the model pipeline/export first.

### CUDA training not used

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Use a CUDA-enabled PyTorch environment. `params.yaml` uses `device: 0` by default.

### Dataset lock mismatch

Do not suppress it blindly. Compare the actual image/class counts and class names with `configs/dataset_lock.yaml`, then deliberately update the lock only if the team accepts a new source release.
