$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$activate = "& '$Root\.venv\Scripts\Activate.ps1'"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$activate; Set-Location '$Root'; mlflow server --host 0.0.0.0 --port 5000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$activate; Set-Location '$Root'; uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Rootrontend'; npm install; npm run dev"
Write-Host "Opened MLflow, FastAPI and React development terminals." -ForegroundColor Green
