# SafeGuard Experiment Plan

## Research question

Does training on a harmonized multi-source intersection improve generalization compared with either source alone, while still meeting a real-time deployment target?

## Controlled source comparison

| Run | Dataset | Model | Purpose |
|---|---|---|---|
| A | Roboflow intersection only | YOLO11n | source baseline |
| B | Ultralytics intersection only | YOLO11n | source baseline |
| C | merged SafeGuard intersection | YOLO11n | multi-source effect |

The canonical ontology and deterministic split policy are identical across these variants.

## Deployment-oriented follow-up runs

| Run | Dataset | Model | Main change |
|---|---|---|---|
| D | merged | YOLO11s | capacity vs speed |
| E | merged | YOLO11n | 768px + controlled augmentation |

All exact run definitions live in `configs/experiments.yaml` so the comparison is executable, not just described in documentation.

## Logged metrics

Model quality:
- precision;
- recall;
- mAP50;
- mAP50-95;
- per-class metrics where available.

Deployment:
- validation inference latency;
- final PT and ONNX end-to-end benchmark latency;
- FPS derived from measured latency;
- model format/checkpoint.

## Selection rule

The deployment target is **<= 50 ms model inference** (at least 20 model inferences/second). Among candidates meeting that target, SafeGuard selects the highest validation mAP50-95; lower inference latency breaks ties.

If no model meets the target, the run summary explicitly flags the miss rather than pretending the requirement was satisfied.

Only after candidate selection is the selected checkpoint evaluated on the held-out final test split.
