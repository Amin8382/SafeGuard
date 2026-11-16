$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1
python -m src.data.audit --config configs/datasets.yaml --lock configs/dataset_lock.yaml --output reports/source_audit.json --strict
python -m src.data.harmonize --config configs/datasets.yaml --output data/interim
python -m src.data.merge --input data/interim --output data/processed --config configs/datasets.yaml --lock configs/dataset_lock.yaml
python -m src.data.near_duplicates --root data/processed --threshold 4 --output reports/near_duplicates.json
python -m src.data.report
Write-Host "SafeGuard data pipeline completed." -ForegroundColor Green
