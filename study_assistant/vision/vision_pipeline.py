"""视觉管线：把摄像头 + 手部 + 物体 + 跟踪串成一步。

核心是**分频调度**：
  * 手部模型轻（MediaPipe，约 3-8 ms）→ 跑 15 Hz；
  * 物体模型重（YOLO，桌面约 30-60 ms，Pi 上 100-400 ms）→ 跑 5 Hz，中间
    靠 ObjectTracker 滑行；
  * 不在推理的那一帧返回上一次结果并标记 `fresh=False`，让下游知道
    这是缓存数据（行为融合不关心，但 Debug 视图要标出来）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .board_profile import detect_board, intervals_for_board
from .object_detector import ObjectDetector, ObjectResult
from .object_tracker import ObjectTracker, TrackerSnapshot
from .hand_detector import HandDetector, HandResult


@dataclass
class VisionFrame:
    """一帧完整的视觉观测。"""

    frame: np.ndarray                      # 原始 BGR 帧
    index: int = 0
    now: float = 0.0                       # perf_counter
    timestamp: float = 0.0                 # wall clock

    hands: HandResult = field(default_factory=HandResult)
    objects: ObjectResult = field(default_factory=ObjectResult)
    tracked: TrackerSnapshot = field(default_factory=TrackerSnapshot)

    hands_fresh: bool = False              # 这一帧真的跑了手部推理吗
    objects_fresh: bool = False            # 这一帧真的跑了物体推理吗

    hands_ms: float = 0.0
    objects_ms: float = 0.0
    loop_ms: float = 0.0
    fps: float = 0.0

    # ---- 便捷访问 ----------------------------------------------------

    @property
    def hand_count(self) -> int:
        return self.hands.count

    def hand(self, label: str):
        return self.hands.get(label)

    def hands_visible(self) -> bool:
        return self.hands.count > 0

    def person_visible(self) -> bool:
        """YOLO 看到 person，或检测到手（手在画面内基本说明人也在）。"""
        return self.tracked.has("person") or self.hands.count > 0

    def object_at(self, label: str):
        return self.tracked.nearest_any([label], (0.5, 0.5))

    def hands_age_ms(self, now: float | None = None) -> float:
        """手部结果有多旧（用于 Debug 视图标注缓存）。"""
        now = time.perf_counter() if now is None else now
        return max(0.0, (now - self.now) * 1000.0)


class VisionPipeline:
    """摄像头 + 多个视觉模型的分频调度器。

    用法::

        pipe = VisionPipeline(config)
        pipe.start()
        while running:
            vf = pipe.step()
            ...
        pipe.close()
    """

    def __init__(self, config, camera=None, root: Path | str | None = None):
        self.config = config
        self.root = Path(root) if root else Path(__file__).resolve().parents[2]

        self.board = detect_board()
        intervals = intervals_for_board(self.board)

        vision = config.vision
        self.hands_cfg = dict(vision.get("hands", {}))
        self.objects_cfg = dict(vision.get("objects", {}))

        # 配置里显式写了 interval_ms 就以配置为准，否则用板卡档位
        self.hands_interval_ms = float(
            self.hands_cfg.get("interval_ms") or intervals["hands"]
        )
        self.objects_interval_ms = float(
            self.objects_cfg.get("interval_ms") or intervals["objects"]
        )

        # --- 组件 ---
        self.camera = camera
        self.hand_detector: HandDetector | None = None
        self.object_detector: ObjectDetector | None = None
        self.tracker = ObjectTracker()

        self._init_detectors()

        # --- 调度状态 ---
        self._frame_index = 0
        self._last_hands_t = -1e9
        self._last_objects_t = -1e9
        self._last_hands_result = HandResult()
        self._last_objects_result = ObjectResult()

        self._fps = 0.0
        self._fps_alpha = 0.1
        self._last_step_t = 0.0

        self.warnings: list[str] = []

    # ------------------------------------------------------------------

    def _init_detectors(self) -> None:
        hands_enabled = bool(self.hands_cfg.get("enabled", True))
        objects_enabled = bool(self.objects_cfg.get("enabled", True))

        if hands_enabled:
            model_path = self.root / self.hands_cfg.get(
                "model_path", "models/hand_landmarker.task"
            )
            try:
                self.hand_detector = HandDetector(
                    model_path=str(model_path),
                    max_hands=int(self.hands_cfg.get("max_hands", 2)),
                    min_detection_confidence=float(
                        self.hands_cfg.get("min_detection_confidence", 0.5)
                    ),
                    min_tracking_confidence=float(
                        self.hands_cfg.get("min_tracking_confidence", 0.5)
                    ),
                )
            except Exception as exc:
                self.warnings.append(f"手部检测不可用：{exc}")
                self.hand_detector = None

        if objects_enabled:
            backend = self.objects_cfg.get("backend", "onnx")
            model_path = self.objects_cfg.get("model_path", "models/yolo11n.onnx")
            resolved = self.root / model_path

            if backend == "onnx" and not resolved.exists():
                # 回退到本地 .pt，避免联网下载失败把整条链路拖死
                pt = self.root / "models" / "yolo11n.pt"
                if pt.exists():
                    self.warnings.append(
                        f"{model_path} 不存在，回退 ultralytics 后端（{pt.name}）"
                    )
                    backend, resolved = "ultralytics", pt

            self.object_detector = ObjectDetector(
                backend=backend,
                model_path=str(resolved) if resolved.suffix in (".onnx", ".pt") else model_path,
                input_size=int(self.objects_cfg.get("input_size", 320)),
                confidence=float(self.objects_cfg.get("confidence", 0.35)),
                iou_threshold=float(self.objects_cfg.get("iou_threshold", 0.45)),
                keep_labels=list(self.objects_cfg.get("keep_classes") or []),
            )

            if not self.object_detector.available:
                self.warnings.append(
                    f"物体检测不可用：{self.object_detector.last_error or '未知错误'}"
                )

    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.camera is None:
            from ..camera.camera_backend import create_camera

            self.camera = create_camera(self.config.data)
        self.camera.open()
        self._last_step_t = time.perf_counter()

    def step(self) -> VisionFrame | None:
        """读取一帧并按分频调度决定跑哪些模型。摄像头无帧时返回 None。"""
        if self.camera is None:
            raise RuntimeError("VisionPipeline.start() 必须先调用")

        packet = self.camera.read()

        if packet is None:
            return None

        # 统一时钟：所有时序计算都用 perf_counter，**不要**混用
        # time.monotonic()。Python 3.13 之前这两个在 Windows 上底层不同
        # （GetTickCount64 vs QPC），混用会让"已离开多久"算成随机数。
        now = time.perf_counter()
        frame_bgr = packet.frame

        vf = VisionFrame(
            frame=frame_bgr,
            index=self._frame_index,
            now=now,
            timestamp=packet.wall_time,
            fps=self._fps,
        )

        # --- 手部 ---
        if self.hand_detector is not None:
            elapsed_ms = (now - self._last_hands_t) * 1000.0
            if elapsed_ms >= self.hands_interval_ms:
                vf.hands = self.hand_detector.process(frame_bgr, timestamp=now)
                self._last_hands_t = now
                self._last_hands_result = vf.hands
                vf.hands_fresh = True
            else:
                vf.hands = self._last_hands_result
                vf.hands_fresh = False

            vf.hands_ms = vf.hands.inference_ms if vf.hands_fresh else 0.0

        # --- 物体 ---
        if self.object_detector is not None and self.object_detector.available:
            elapsed_ms = (now - self._last_objects_t) * 1000.0
            if elapsed_ms >= self.objects_interval_ms:
                vf.objects = self.object_detector.process(frame_bgr)
                self._last_objects_t = now
                self._last_objects_result = vf.objects
                vf.objects_fresh = True
            else:
                vf.objects = self._last_objects_result
                vf.objects_fresh = False

            vf.objects_ms = vf.objects.inference_ms if vf.objects_fresh else 0.0
        else:
            vf.objects = ObjectResult()

        # --- 跟踪（每个新检测才更新；缓存帧沿用当前轨迹快照）---
        if vf.objects_fresh:
            tracked = self.tracker.update(vf.objects.objects, now)
        else:
            tracked = self.tracker.snapshot(now)

        vf.tracked = tracked

        # --- 帧率（用摄像头给出的瞬时 fps 做 EMA 平滑，避免数字乱跳）---
        dt = now - self._last_step_t
        self._last_step_t = now

        sample_fps = packet.fps if packet.fps and packet.fps > 0 else (
            1.0 / dt if dt > 1e-6 else 0.0
        )

        if sample_fps > 0:
            self._fps = (
                sample_fps if self._fps <= 0
                else (1.0 - self._fps_alpha) * self._fps + self._fps_alpha * sample_fps
            )

        vf.fps = self._fps
        vf.loop_ms = dt * 1000.0
        self._frame_index += 1

        return vf

    # ------------------------------------------------------------------

    def close(self) -> None:
        for component in (self.hand_detector, self.object_detector):
            if component is not None:
                try:
                    component.close()
                except Exception:
                    pass

        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass

    def reset_tracking(self) -> None:
        self.tracker.reset()
        self._last_hands_t = -1e9
        self._last_objects_t = -1e9

    def switch_camera(self, index: int) -> None:
        """热切换到另一个摄像头索引。

        顺序是「先开新、验证出画面、再放旧」：
          1. 新索引 open() 打不开（不存在 / 被占用）→ 抛 RuntimeError；
          2. 打得开但读不到有效画面（虚拟摄像头没推流、设备刚掉线、
             镜头被物理遮挡）→ 释放新设备并抛 RuntimeError ——
             "能 open" 不等于 "有画面"，这一步不验证用户就会看到黑屏；
          3. 验证通过才替换 self.camera 并释放旧设备。
        换源后旧画面的轨迹与手部时序对新画面没有意义，必须
        reset_tracking()，否则上一次的滑行轨迹会混进新画面参与判定。
        """
        from ..camera.camera_backend import OpenCVCamera

        cam_cfg = self.config.data.get("camera", {})
        new_cam = OpenCVCamera(
            index=int(index),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
            mirror=cam_cfg.get("mirror", True),
        )
        new_cam.open()  # 失败在此抛出，self.camera 未被改动

        # 出画面验证：部分设备 open() 成功但 read() 长阻塞后返回失败，
        # 或只给黑帧。上限 6 次尝试 / 5 秒，超限判定为无有效画面。
        frame_ok = False
        verify_deadline = time.perf_counter() + 5.0
        for _ in range(6):
            packet = new_cam.read()
            if packet is not None and packet.frame is not None:
                if float(np.mean(packet.frame)) >= 4.0:  # 黑帧阈值
                    frame_ok = True
                    break
            if time.perf_counter() >= verify_deadline:
                break

        if not frame_ok:
            try:
                new_cam.release()
            except Exception:
                pass
            raise RuntimeError(
                f"摄像头索引 {index} 能打开但读不到有效画面"
                "（虚拟设备未推流 / 设备已掉线 / 镜头被遮挡）"
            )

        old = self.camera
        self.camera = new_cam
        if old is not None:
            try:
                old.release()
            except Exception:
                pass
        self.reset_tracking()

    # ------------------------------------------------------------------

    @property
    def status_line(self) -> str:
        """一行状态描述（不带前缀，方便调用方自行加 "[视觉]" 等标签）。"""
        return (
            f"{self.board} | 手部 {self.hands_interval_ms:.0f}ms | "
            f"物体 {self.objects_interval_ms:.0f}ms | "
            f"{self._fps:.1f} fps"
        )
