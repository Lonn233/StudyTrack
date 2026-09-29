"""手部检测：MediaPipe Hand Landmarker（tasks API）。

输出归一化的手部特征，供后续运动分析与手-物交互使用。
不使用 FaceMesh —— 本项目核心是手与桌面的交互，不是面部注视。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# MediaPipe 手部 21 个关键点索引
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class Hand:
    """单手的一帧观测结果（坐标全部归一化到 0..1）。"""

    label: str                       # "left" / "right"
    handedness_score: float
    landmarks: np.ndarray            # (21, 3) 归一化 x, y, z
    wrist: tuple                     # (x, y) 归一化
    palm_center: tuple
    finger_tips: list                # [(x, y), ...] 五个指尖
    bbox: tuple                      # 归一化 (x0, y0, x1, y1)
    timestamp: float

    @property
    def center(self) -> tuple:
        return self.palm_center


@dataclass
class HandResult:
    hands: list = field(default_factory=list)
    timestamp: float = 0.0
    inference_ms: float = 0.0
    cached: bool = False

    @property
    def count(self) -> int:
        return len(self.hands)

    def get(self, label: str) -> Hand | None:
        for hand in self.hands:
            if hand.label == label:
                return hand
        return None


class HandDetector:
    """MediaPipe Hand Landmarker 封装（VIDEO 模式，跨帧跟踪）。"""

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = model_path
        self.max_hands = int(max_hands)
        self.available = False
        self._landmarker = None
        self._last_ts_ms = 0

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand model not found: {model_path}. "
                "Download hand_landmarker.task into models/."
            )

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=self.max_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self._mp = mp
            self.available = True

        except Exception as exc:
            self.last_error = str(exc)
            raise RuntimeError(f"Failed to init Hand Landmarker: {exc}") from exc

    def process(self, frame_bgr, timestamp: float | None = None) -> HandResult:
        """在 BGR 帧上检测双手。返回归一化坐标。"""
        now = time.perf_counter()
        ts = timestamp if timestamp is not None else time.time()

        # VIDEO 模式要求时间戳严格递增
        ts_ms = int(time.monotonic() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        rgb = frame_bgr[:, :, ::-1]
        rgb = np.ascontiguousarray(rgb)

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )

        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        hands: list[Hand] = []

        if result.hand_landmarks:
            h, w = frame_bgr.shape[:2]

            for idx, landmarks in enumerate(result.hand_landmarks):
                pts = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
                )

                label = "right"
                score = 0.0

                if result.handedness and idx < len(result.handedness):
                    category = result.handedness[idx][0]
                    label = str(category.category_name).lower()
                    score = float(category.score)

                # 镜像画面里 MediaPipe 的左右会反，这里保持模型原始判定，
                # 由配置决定是否翻转；默认摄像头已做水平镜像。
                if label not in ("left", "right"):
                    label = "unknown"

                xs, ys = pts[:, 0], pts[:, 1]

                hands.append(
                    Hand(
                        label=label,
                        handedness_score=score,
                        landmarks=pts,
                        wrist=(float(pts[WRIST, 0]), float(pts[WRIST, 1])),
                        palm_center=(
                            float(pts[[WRIST, 5, 9, 13, 17], 0].mean()),
                            float(pts[[WRIST, 5, 9, 13, 17], 1].mean()),
                        ),
                        finger_tips=[
                            (float(pts[t, 0]), float(pts[t, 1]))
                            for t in FINGER_TIPS
                        ],
                        bbox=(
                            float(xs.min()), float(ys.min()),
                            float(xs.max()), float(ys.max()),
                        ),
                        timestamp=ts,
                    )
                )

        return HandResult(
            hands=hands,
            timestamp=ts,
            inference_ms=(time.perf_counter() - now) * 1000.0,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
