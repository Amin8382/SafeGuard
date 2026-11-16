from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

CURRENT_ULTRALYTICS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
)


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def flatten_single_folder(root: Path):
    entries = [p for p in root.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        for child in list(nested.iterdir()):
            shutil.move(str(child), root / child.name)
        nested.rmdir()


def extract_zip(archive: Path, dest: Path):
    reset_dir(dest)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    flatten_single_folder(dest)


def install_ultralytics_legacy(dest: Path, archive: Path | None, accept_current: bool):
    if archive:
        if not archive.exists():
            raise FileNotFoundError(archive)
        extract_zip(archive, dest)
        print(f"Ultralytics archive extracted to {dest}")
        return

    if not accept_current:
        raise RuntimeError(
            "SafeGuard is locked to the legacy 1,416-image / 11-class Ultralytics release. "
            "The official public URL now serves a newer release. Supply your selected legacy ZIP with "
            "--ultralytics-zip PATH. Use --accept-current-ultralytics only if the team deliberately "
            "changes the project dataset and then updates configs/dataset_lock.yaml."
        )

    reset_dir(dest)
    archive_path = dest.parent / "construction-ppe-current.zip"
    print(f"Downloading CURRENT Ultralytics release from {CURRENT_ULTRALYTICS_URL}")
    urllib.request.urlretrieve(CURRENT_ULTRALYTICS_URL, archive_path)
    extract_zip(archive_path, dest)
    archive_path.unlink(missing_ok=True)
    print("WARNING: current release downloaded; strict audit is expected to fail until the lock is deliberately updated.")


def download_roboflow(dest: Path, version: int, fmt: str):
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Export Construction Site Safety v30 in YOLOv11 format "
            "into data/raw/roboflow, or set the key and rerun."
        )
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError("Install training requirements first: pip install -r requirements-training.txt") from exc

    reset_dir(dest)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
    dataset = project.version(version).download(fmt, location=str(dest), overwrite=True)
    if getattr(dataset, "location", None):
        print(f"Roboflow dataset downloaded to {dataset.location}")
    flatten_single_folder(dest)


def main():
    parser = argparse.ArgumentParser(description="Install the two locked SafeGuard source datasets.")
    parser.add_argument("--source", choices=["all", "roboflow", "ultralytics"], default="all")
    parser.add_argument("--roboflow-version", type=int, default=30)
    parser.add_argument("--roboflow-format", default="yolov11")
    parser.add_argument("--ultralytics-zip", type=Path)
    parser.add_argument("--accept-current-ultralytics", action="store_true")
    args = parser.parse_args()

    if args.source in {"all", "ultralytics"}:
        install_ultralytics_legacy(
            Path("data/raw/ultralytics"),
            args.ultralytics_zip,
            args.accept_current_ultralytics,
        )
    if args.source in {"all", "roboflow"}:
        download_roboflow(Path("data/raw/roboflow"), args.roboflow_version, args.roboflow_format)


if __name__ == "__main__":
    main()
