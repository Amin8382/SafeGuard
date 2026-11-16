# SafeGuard Progress

## Current repository state

All planned implementation milestones are consolidated into this repository.

### Verified in the build workspace
- Python source compiles.
- Data/backend unit tests pass.
- Dataset pipeline tests validate deterministic split/dedup behavior.
- ONNX post-processing is unit tested with synthetic YOLO11 output.
- API correctly reports model-not-ready state when no real export exists.

### Generated only after source data/GPU execution
- real `data/processed/` dataset;
- measured class-distribution/data-quality outputs;
- MLflow experiment runs;
- candidate `.pt` weights;
- final `best.pt` / `best.onnx`;
- real mAP/precision/recall/latency/FPS reports.

Those outputs are deliberately not fabricated or committed as if they had been measured.
