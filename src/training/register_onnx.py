from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.onnx
import onnx


def main():
    parser = argparse.ArgumentParser(description='Register the production ONNX model in MLflow Model Registry.')
    parser.add_argument('--model', default='models/best.onnx')
    parser.add_argument('--name', default='SafeGuard-PPE')
    parser.add_argument('--experiment', default='SafeGuard')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    onnx_model = onnx.load(str(model_path))
    with mlflow.start_run(run_name='onnx-production-candidate') as run:
        model_info = mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path='onnx-model',
            registered_model_name=args.name,
        )
        mlflow.set_tag('deployment_format', 'ONNX')
        mlflow.set_tag('safeguard_stage', 'registry_candidate')
        print(f'Registered {args.name} from run {run.info.run_id}: {model_info.model_uri}')


if __name__ == '__main__':
    main()
