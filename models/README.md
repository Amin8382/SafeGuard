# Generated models

This directory is the local deployment handoff.

After the experiment matrix:

```text
models/candidates/*.pt
models/best.pt
```

After ONNX export:

```text
models/best.onnx
```

The binaries are intentionally ignored by Git. Track model artifacts in MLflow/Model Registry. `docker compose` mounts this local directory into the FastAPI service for the practical demo.
