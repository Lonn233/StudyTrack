"""桌面物体检测：YOLO nano。

只关心与"手在操作什么"相关的类别：手机 / 书 / 电脑 / 键盘 / 鼠标 /
杯子 / 人物（人物仅作离席辅助）。低频运行（默认 5 Hz），帧间由
追踪器维持位置。

支持两种后端：
  ultralytics —— 直接用 YOLO API
  onnx        —— onnxruntime，适合轻量部署与树莓派
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

# COCO 类别名 → 本项目语义名
COCO_TO_SEMANTIC = {
    "person": "person",
    "cell phone": "phone",
    "book": "book",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "cup": "cup",
}

# COCO 类别 id（用于 ONNX 输出过滤）
COCO_IDS = {
    "person": 0,
    "cell phone": 67,
    "book": 73,
    "laptop": 63,
    "keyboard": 66,
    "mouse": 64,
    "cup": 41,
}

ID_TO_SEMANTIC = {v: COCO_TO_SEMANTIC[k] for k, v in COCO_IDS.items()}


@dataclass
class DetectedObject:
    """一个检测到的物体（bbox 为归一化坐标）。"""

    label: str                 # 语义名：phone / book / ...
    coco_name: str
    confidence: float
    bbox: tuple                # (x0, y0, x1, y1) 归一化
    track_id: int = -1
    timestamp: float = 0.0

    @property
    def center(self) -> tuple:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class ObjectResult:
    objects: list = field(default_factory=list)
    timestamp: float = 0.0
    inference_ms: float = 0.0
    cached: bool = False

    def by_label(self, label: str) -> list:
        return [o for o in self.objects if o.label == label]

    def has(self, label: str) -> bool:
        return any(o.label == label for o in self.objects)

    def nearest(self, label: str, point: tuple) -> DetectedObject | None:
        """返回离 point 最近的指定类别物体（归一化距离）。"""
        candidates = self.by_label(label)
        if not candidates:
            return None

        px, py = point
        return min(
            candidates,
            key=lambda o: (o.center[0] - px) ** 2 + (o.center[1] - py) ** 2,
        )


class ObjectDetector:
    """YOLO nano 物体检测器（后端可插拔）。"""

    def __init__(
        self,
        backend: str = "ultralytics",
        model_path: str = "yolo11n.pt",
        input_size: int = 320,
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        keep_labels: list | None = None,
    ):
        self.backend_name = backend
        self.model_path = model_path
        self.input_size = int(input_size)
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.keep_labels = set(
            keep_labels or ["person", "phone", "book", "laptop", "keyboard", "mouse", "cup"]
        )

        self.available = False
        self.last_error = ""
        self._model = None
        self._session = None

        if backend == "none":
            return

        try:
            if backend == "ultralytics":
                self._init_ultralytics()
            elif backend == "onnx":
                self._init_onnx()
            else:
                raise ValueError(f"Unknown object backend: {backend}")
            self.available = True

        except Exception as exc:
            self.last_error = str(exc)
            # 不抛出：物体检测不可用时其余功能仍应工作
            print(f"[ObjectDetector] unavailable ({backend}): {exc}")

    # ------------------------------------------------------------------

    def _init_ultralytics(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.model_path)

    def _init_onnx(self) -> None:
        import onnxruntime as ort

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(self.model_path, providers=providers)

    # ------------------------------------------------------------------

    def process(self, frame_bgr) -> ObjectResult:
        now = time.perf_counter()

        if not self.available:
            return ObjectResult(inference_ms=0.0)

        if self.backend_name == "ultralytics":
            objects = self._run_ultralytics(frame_bgr)
        else:
            objects = self._run_onnx(frame_bgr)

        return ObjectResult(
            objects=objects,
            timestamp=time.time(),
            inference_ms=(time.perf_counter() - now) * 1000.0,
        )

    # ------------------------------------------------------------------

    def _run_ultralytics(self, frame_bgr) -> list:
        results = self._model.predict(
            frame_bgr,
            imgsz=self.input_size,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )

        objects: list[DetectedObject] = []
        h, w = frame_bgr.shape[:2]

        for result in results:
            names = result.names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                coco_name = str(names.get(cls_id, "")).lower()
                semantic = COCO_TO_SEMANTIC.get(coco_name)

                if semantic is None or semantic not in self.keep_labels:
                    continue

                x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
                objects.append(
                    DetectedObject(
                        label=semantic,
                        coco_name=coco_name,
                        confidence=float(box.conf[0]),
                        bbox=(
                            max(0.0, x0 / w), max(0.0, y0 / h),
                            min(1.0, x1 / w), min(1.0, y1 / h),
                        ),
                        timestamp=time.time(),
                    )
                )

        return objects

    def _run_onnx(self, frame_bgr) -> list:
        """纯 numpy 预处理 + NMS，适配 YOLO 导出的 ONNX。"""
        import cv2

        h, w = frame_bgr.shape[:2]
        size = self.input_size

        # letterbox
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[None, ...])

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})[0]

        # YOLO 输出: (1, 4+nc, N) 或 (1, N, 4+nc)
        pred = outputs[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.transpose(1, 0)

        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]

        if scores_all.size == 0:
            return []

        class_ids = scores_all.argmax(axis=1)
        confidences = scores_all[np.arange(len(class_ids)), class_ids]

        mask = confidences >= self.confidence
        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # xywh -> xyxy（模型输出在 letterbox 坐标系）
        cx, cy, bw, bh = (
            boxes_xywh[:, 0], boxes_xywh[:, 1],
            boxes_xywh[:, 2], boxes_xywh[:, 3],
        )
        x0 = cx - bw / 2
        y0 = cy - bh / 2

        nms_boxes = np.stack([x0, y0, bw, bh], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, confidences.tolist(),
            self.confidence, self.iou_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()
        objects: list[DetectedObject] = []

        for i in indices:
            sem = ID_TO_SEMANTIC.get(int(class_ids[i]))
            if sem is None or sem not in self.keep_labels:
                continue

            # letterbox 坐标 -> 原图归一化
            bx = (x0[i] - dx) / scale
            by = (y0[i] - dy) / scale
            bw_i = bw[i] / scale
            bh_i = bh[i] / scale

            objects.append(
                DetectedObject(
                    label=sem,
                    coco_name=sem,
                    confidence=float(confidences[i]),
                    bbox=(
                        max(0.0, bx / w), max(0.0, by / h),
                        min(1.0, (bx + bw_i) / w), min(1.0, (by + bh_i) / h),
                    ),
                    timestamp=time.time(),
                )
            )

        return objects

    def close(self) -> None:
        self._model = None
        self._session = None
