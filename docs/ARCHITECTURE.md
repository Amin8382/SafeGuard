# SafeGuard Architecture

## Runtime

```text
Image upload / webcam
        |
        v
React / Vite
        |
        | multipart image or frame
        v
FastAPI + Pydantic
        |
        v
ONNX Runtime
        |
        v
YOLO11 preprocessing -> graph -> NMS -> restored boxes
        |
        v
Detections + latency + conservative compliance summary
```

## Training and MLOps

```text
Roboflow v30 ---------+
                      |
                      v
                  strict audit
                      ^
                      |
Ultralytics legacy ---+
                      |
                      v
             semantic intersection
                      |
                      v
               exact deduplication
                      |
                      v
             deterministic re-split
                      |
         +------------+------------+
         |            |            |
         v            v            v
    Roboflow-only  Ultralytics-only  merged
         \            |            /
          \-----------+-----------/
                      |
                      v
             YOLO11 experiment matrix
                      |
                      v
                    MLflow
                      |
                      v
        real-time target + mAP selection
                      |
                      v
                  best.pt
                      |
          held-out test evaluation
                      |
                      v
                 ONNX export
                      |
                      v
          PT vs ONNX benchmark
                      |
                      v
              MLflow Registry
```

## Responsibilities

- **Git**: code, configuration, docs, small pointer/metadata files.
- **DVC**: source data, generated dataset pipeline and reproducibility.
- **MLflow Tracking**: experiment history, metrics, plots, run artifacts.
- **MLflow Registry**: named ONNX deployment candidate.
- **YOLO11**: detector training/evaluation.
- **ONNX Runtime**: production inference.
- **FastAPI/Pydantic**: async REST contract and Swagger.
- **React**: upload/webcam visualization.
- **Docker Compose**: local practical-demo orchestration.
- **GitHub Actions**: CI quality/build gates and container publishing CD.
