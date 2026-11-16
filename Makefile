.PHONY: test lint data train evaluate export benchmark api frontend docker preflight

preflight:
	python scripts/preflight.py

test:
	pytest backend/tests tests -q

lint:
	ruff check src backend tests scripts

data:
	python -m src.data.audit --config configs/datasets.yaml --lock configs/dataset_lock.yaml --output reports/source_audit.json
	python -m src.data.harmonize --config configs/datasets.yaml --output data/interim
	python -m src.data.merge --input data/interim --output data/processed --config configs/datasets.yaml --lock configs/dataset_lock.yaml
	python -m src.data.near_duplicates --root data/processed --threshold 4 --output reports/near_duplicates.json
	python -m src.data.report

train:
	python -m src.training.run_matrix --matrix configs/experiments.yaml

evaluate:
	python -m src.training.evaluate --weights models/best.pt --data data/processed/safeguard.yaml

export:
	python -m src.training.export_onnx --weights models/best.pt --imgsz 640 --output models/best.onnx

benchmark:
	python -m src.training.benchmark --data data/processed/safeguard.yaml

api:
	uvicorn backend.app.main:app --reload

frontend:
	cd frontend && npm install && npm run dev

docker:
	docker compose up --build
