import numpy as np

from backend.app.compliance import summarize
from backend.app.inference import OnnxYoloDetector


def test_postprocess_yolo11_shape_and_no_helmet_summary(tmp_path):
    detector = OnnxYoloDetector(
        tmp_path / 'missing.onnx',
        ['person', 'helmet', 'vest', 'no_helmet'],
        confidence=0.25,
        iou=0.45,
    )
    raw = np.zeros((1, 8, 2), dtype=np.float32)
    raw[0, 0:4, 0] = [320, 320, 200, 300]
    raw[0, 4 + 0, 0] = 0.90
    raw[0, 0:4, 1] = [200, 120, 80, 60]
    raw[0, 4 + 3, 1] = 0.85

    detections = detector._postprocess(raw, (640, 640), ratio=1.0, pad=(0, 0))
    names = {d.class_name for d in detections}
    assert names == {'person', 'no_helmet'}
    summary = summarize(detections)
    assert summary['status'] == 'VIOLATION'
    assert summary['violations']['no_helmet'] == 1


def test_no_person_summary_is_not_claimed_compliant():
    summary = summarize([])
    assert summary['status'] == 'NO_PERSON'
