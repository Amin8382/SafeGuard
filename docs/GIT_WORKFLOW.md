# SafeGuard Git Workflow

## Branch model

- `main`: stable integration branch;
- feature branches: `feat/<short-name>`;
- fixes: `fix/<short-name>`;
- experiments/config: `exp/<short-name>`.

## Commit style

Examples:

```text
feat(data): harmonize Roboflow and Ultralytics intersection labels
feat(train): add MLflow YOLO11 experiment matrix
feat(api): add ONNX Runtime prediction endpoints
feat(ui): add real-time webcam PPE detection
ci: validate Python frontend and Docker builds
docs: add practical defense runbook
```

## What belongs in Git

- source code;
- configs;
- DVC pointer files;
- documentation;
- CI/CD workflows.

## What does not belong in Git

- raw/processed image datasets;
- `.pt` and `.onnx` binary models;
- MLflow local artifacts/databases;
- training runs;
- `node_modules` or virtual environments.

Use DVC/MLflow for those generated or heavy assets.

## First remote synchronization

After creating an empty GitHub repository, from this local repository:

```powershell
.\scripts\push_to_github.ps1 -RemoteUrl https://github.com/<owner>/SafeGuard.git
```

This pushes `main` plus all `milestone-*` and `v1.0.0` tags.
