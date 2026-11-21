"""ONNX Runtime detector for a YOLOv8 export (no torch needed at runtime)."""
from __future__ import annotations

import ast
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original pixels

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 1) for v in self.box],
        }


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Plain NumPy non-maximum suppression. boxes are x1,y1,x2,y2."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class Detector:
    def __init__(self, model_path: str | Path, providers: list[str] | None = None):
        # Prefer CoreML (Apple silicon) or CUDA when present; always fall back to CPU.
        wanted = providers or os.getenv("ORT_PROVIDERS", "CoreMLExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider").split(",")
        available = set(ort.get_available_providers())
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), so, providers=[p for p in wanted if p in available] or ["CPUExecutionProvider"]
        )
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.h, self.w = self.session.get_inputs()[0].shape
        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta.get("names", "{}"))
        self.names = [names[i] for i in sorted(names)]

    def _letterbox(self, img: Image.Image) -> tuple[np.ndarray, float, tuple[int, int]]:
        w0, h0 = img.size
        r = min(self.w / w0, self.h / h0)
        nw, nh = round(w0 * r), round(h0 * r)
        canvas = Image.new("RGB", (self.w, self.h), (114, 114, 114))
        px, py = (self.w - nw) // 2, (self.h - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, r, (px, py)

    def predict(self, img: Image.Image, conf: float = 0.25, iou: float = 0.45,
                agnostic: bool = False) -> tuple[list[Detection], float]:
        """agnostic=True: detection only. Every box is labelled "Bunch" and NMS ignores class."""
        img = img.convert("RGB")
        w0, h0 = img.size
        x, r, (px, py) = self._letterbox(img)
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0][0]  # (4 + n_classes, 8400)
        latency_ms = (time.perf_counter() - t0) * 1000
        out = out.T
        cls_scores = out[:, 4:]
        cls_ids = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(out)), cls_ids]
        mask = scores >= conf
        if not mask.any():
            return [], latency_ms
        xywh, scores, cls_ids = out[mask, :4], scores[mask], cls_ids[mask]
        boxes = np.empty_like(xywh)
        boxes[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2 - px) / r
        boxes[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2 - py) / r
        boxes[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2 - px) / r
        boxes[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2 - py) / r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
        if agnostic:
            keep = nms(boxes, scores, iou)
        else:
            # class-aware NMS: offset boxes per class so different classes never suppress each other
            keep = nms(boxes + cls_ids[:, None] * 10_000, scores, iou)
        dets = [
            Detection("Bunch" if agnostic else self.names[int(cls_ids[i])], float(scores[i]),
                      tuple(float(v) for v in boxes[i]))
            for i in keep
        ]
        return dets, latency_ms
