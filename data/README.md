# SafeGuard data inputs

SafeGuard does **not** commit image datasets to Git. Raw datasets are tracked with DVC.

Expected layout:

```text
data/raw/
├── roboflow/
│   ├── data.yaml (or another YOLO dataset YAML)
│   ├── images/...
│   └── labels/...
└── ultralytics/
    ├── data.yaml or construction-ppe.yaml
    ├── images/...
    └── labels/...
```

## Locked sources

### Roboflow

SafeGuard is configured for the **Construction Site Safety** project, version 30 (`raw-images_latestversion`), 717 images, exported in YOLO format. The harmonizer keeps only the five shared semantic labels.

### Ultralytics

The project specification is locked to the earlier **Construction-PPE 1,416-image / 11-class** release selected by the team.

Ultralytics replaced the public canonical dataset in November 2026 with a different 1,000-image / 12-class release. Therefore SafeGuard does not automatically substitute the current archive. Put your legacy selected archive under `data/raw/ultralytics`, or install it with:

```powershell
python scripts\bootstrap_data.py --source ultralytics --ultralytics-zip C:\path\to\legacy-construction-ppe.zip
```

If the team deliberately migrates to the new release, use the current archive and then explicitly update `configs/dataset_lock.yaml`, document the change, rebuild the dataset and rerun all experiments.

## DVC

After both source datasets are present:

```powershell
dvc add data/raw/roboflow data/raw/ultralytics
git add data/raw/*.dvc data/raw/.gitignore
```

Then configure a shared DVC remote appropriate for the team.
