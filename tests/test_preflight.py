from pathlib import Path

from scripts.preflight import dataset_readiness


def test_placeholder_raw_directory_is_not_ready(tmp_path: Path):
    root = tmp_path / "roboflow"
    root.mkdir()
    (root / ".gitkeep").write_text("", encoding="utf-8")

    result = dataset_readiness(root)

    assert result["directory_exists"] is True
    assert result["image_count"] == 0
    assert result["label_count"] == 0
    assert result["yaml_present"] is False
    assert result["dataset_ready"] is False


def test_yolo_dataset_with_images_labels_and_yaml_is_ready(tmp_path: Path):
    root = tmp_path / "dataset"
    images = root / "train" / "images"
    labels = root / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "sample.jpg").write_bytes(b"not-a-real-image-needed-for-preflight")
    (labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "data.yaml").write_text("names: [person]\n", encoding="utf-8")

    result = dataset_readiness(root)

    assert result["image_count"] == 1
    assert result["label_count"] == 1
    assert result["yaml_present"] is True
    assert result["dataset_ready"] is True
