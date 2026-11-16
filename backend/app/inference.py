from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


class ModelNotReadyError(RuntimeError):
    pass


class OnnxYoloDetector:
    def __init__(
        self,
        model_path: Path,
        class_names: list[str],
        confidence: float = 0.25,
        iou: float = 0.45,
        max_detections: int = 100,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections
        self._session = None
        self._input_name = None
        self._input_hw = (640, 640)
        self._provider = None
        self._load_lock = Lock()

    @property
    def ready(self) -> bool:
        return self.model_path.exists()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def provider(self) -> str | None:
        return self._provider

    @property
    def input_hw(self) -> tuple[int, int]:
        return self._input_hw

    def _ensure_session(self):
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            if not self.model_path.exists():
                raise ModelNotReadyError(f"ONNX model not found at {self.model_path}")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ModelNotReadyError("onnxruntime is not installed") from exc

            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
            active = self._session.get_providers()
            self._provider = active[0] if active else None
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            shape = inp.shape
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self._input_hw = (shape[2], shape[3])

    @staticmethod
    def _letterbox(image: np.ndarray, target_hw: tuple[int, int]):
        target_h, target_w = target_hw
        h, w = image.shape[:2]
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        left = pad_w // 2
        top = pad_h // 2
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas, ratio, (left, top)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        boxed, ratio, pad = self._letterbox(image, self._input_hw)
        rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor), ratio, pad

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out

    def _postprocess(
        self,
        raw: np.ndarray,
        original_shape: tuple[int, int],
        ratio: float,
        pad: tuple[int, int],
    ) -> list[Detection]:
        pred = np.squeeze(raw)
        if pred.ndim != 2:
            raise ValueError(f"Unexpected YOLO ONNX output shape: {raw.shape}")

        expected_channels = 4 + len(self.class_names)
        if pred.shape[0] in (expected_channels, expected_channels + 1):
            pred = pred.T

        if pred.shape[1] not in (expected_channels, expected_channels + 1):
            raise ValueError(
                f"Unexpected YOLO output width {pred.shape[1]} for {len(self.class_names)} classes."
            )

        boxes_xywh = pred[:, :4]
        if pred.shape[1] == expected_channels:
            class_scores = pred[:, 4:]
        else:
            objectness = pred[:, 4:5]
            class_scores = pred[:, 5:] * objectness

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = confidences >= self.confidence
        if not np.any(keep):
            return []

        boxes = self._xywh_to_xyxy(boxes_xywh[keep])
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        left, top = pad
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
        orig_h, orig_w = original_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        nms_boxes = [
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            for x1, y1, x2, y2 in boxes
        ]
        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            confidences.astype(float).tolist(),
            self.confidence,
            self.iou,
        )
        if len(indices) == 0:
            return []
        indices = np.array(indices).reshape(-1)[: self.max_detections]

        detections: list[Detection] = []
        for idx in indices:
            x1, y1, x2, y2 = boxes[idx]
            class_id = int(class_ids[idx])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    confidence=float(confidences[idx]),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return detections

    def predict(self, image: np.ndarray) -> list[Detection]:
        self._ensure_session()
        tensor, ratio, pad = self._preprocess(image)
        outputs = self._session.run(None, {self._input_name: tensor})
        return self._postprocess(outputs[0], image.shape[:2], ratio, pad)
