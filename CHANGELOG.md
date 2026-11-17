# Changelog

## Unreleased - Dataset lock correction after source acquisition

- locked Roboflow Construction Site Safety to the actual v27 YOLOv8 export used by the team (2,799 images);
- locked the legacy 1,416-image / 11-class Construction-PPE mirror and its actual class order;
- corrected the strict SafeGuard semantic intersection from five classes to four because Roboflow v27 does not annotate gloves;
- strengthened source-audit release checks and fixed preflight false positives for empty raw-data placeholder directories.

## 1.0.0 - Complete course-project implementation

- consolidated all previous milestone work into one repository;
- finalized semantic-intersection data engineering and strict release audit;
- added data-quality reporting and provenance;
- added complete YOLO11 experiment matrix, real-time selection gate and final evaluation;
- added ONNX export, benchmark and MLflow registry handoff;
- hardened FastAPI with Pydantic schemas and batch/frame endpoints;
- finalized image + live webcam React UI and session analytics;
- added Docker health checks, CI and GHCR CD;
- added Windows run scripts, full runbook and defense guide.
