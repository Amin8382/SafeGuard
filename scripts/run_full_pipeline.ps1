$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1

python scripts/preflight.py
& $PSScriptRootun_data_pipeline.ps1
python -m src.training.run_matrix --matrix configs/experiments.yaml
python -m src.training.evaluate --weights models/best.pt --data data/processed/safeguard.yaml --output reports/final_test_metrics.json
python -m src.training.export_onnx --weights models/best.pt --imgsz 640 --output models/best.onnx
python -m src.training.benchmark --data data/processed/safeguard.yaml --pt models/best.pt --onnx models/best.onnx --output reports/benchmark.json
python -m src.training.register_onnx --model models/best.onnx --name SafeGuard-PPE
Write-Host "SafeGuard full ML pipeline completed." -ForegroundColor Green
