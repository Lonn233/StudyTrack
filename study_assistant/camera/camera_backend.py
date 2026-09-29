"""摄像头后端抽象层。

第一阶段用 OpenCV（Windows/Linux USB 摄像头），未来树莓派切
Picamera2Backend（CSI 排线摄像头）。视觉模块只依赖 CameraBackend
接口，不含任何平台专属代码。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class FramePacket:
    """一帧图像 + 元信息。"""

    frame: Any
    timestamp: float          # time.monotonic()，用于时序计算
    wall_time: float          # time.time()，用于事件时间戳
    fps: float
    index: int


class CameraBackend(ABC):
    """摄像头后端接口。"""

    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def read(self) -> FramePacket | None:
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    @property
    @abstractmethod
    def is_open(self) -> bool:
        ...

    @property
    def width(self) -> int:
        return 0

    @property
    def height(self) -> int:
        return 0


class OpenCVCamera(CameraBackend):
    """USB / 内置摄像头，通过 OpenCV 采集。"""

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mirror: bool = True,
    ):
        self.index = int(index)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self.mirror = bool(mirror)

        self._cap = None
        self._frame_index = 0
        self._last_ts = time.monotonic()

    def open(self) -> None:
        import cv2

        # Windows 上 DSHOW 启动更快；失败则退回默认后端。
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.index)

        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(
                f"Camera index {self.index} could not be opened. "
                "Is another app using it?"
            )

        # 请求 MJPG 压缩流：多数 UVC 相机默认协商 YUY2（未压缩），
        # 外接相机经 USB 2.0 / Hub 时带宽不足会掉到 ~1 fps（画面近似黑屏）。
        # 相机不支持 MJPG 时此调用无效但无害，维持原格式。
        try:
            self._cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
            )
        except Exception:
            pass

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # 丢掉前几帧：很多摄像头自动曝光未稳定时会给黑帧。
        for _ in range(5):
            self._cap.read()

    def read(self) -> FramePacket | None:
        import cv2

        if self._cap is None:
            return None

        ok, frame = self._cap.read()

        if not ok or frame is None:
            return None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        now = time.monotonic()
        fps = 1.0 / max(now - self._last_ts, 1e-6)
        self._last_ts = now
        self._frame_index += 1

        return FramePacket(
            frame=frame,
            timestamp=now,
            wall_time=time.time(),
            fps=fps,
            index=self._frame_index,
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


class PiCamera2Backend(CameraBackend):
    """树莓派 CSI 摄像头（Picamera2 / libcamera）。

    第一阶段不会在 Windows 上实例化，仅预留接口以满足硬件解耦要求。
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mirror: bool = True,
    ):
        self.index = int(index)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self.mirror = bool(mirror)

        self._picam = None
        self._frame_index = 0
        self._last_ts = time.monotonic()

    def open(self) -> None:
        import cv2
        from picamera2 import Picamera2

        self._picam = Picamera2(camera_num=self.index)
        config = self._picam.create_preview_configuration(
            main={"size": (self._width, self._height), "format": "RGB888"},
            buffer_count=4,
        )
        self._picam.configure(config)

        try:
            self._picam.set_controls({"FrameRate": float(self._fps)})
        except Exception:
            pass

        self._picam.start()
        time.sleep(0.35)

    def read(self) -> FramePacket | None:
        import cv2

        if self._picam is None:
            return None

        frame = self._picam.capture_array("main")

        if frame is None:
            return None

        # Picamera2 给 RGB，转成 OpenCV 的 BGR 约定。
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.mirror:
            frame = cv2.flip(frame, 1)

        now = time.monotonic()
        fps = 1.0 / max(now - self._last_ts, 1e-6)
        self._last_ts = now
        self._frame_index += 1

        return FramePacket(
            frame=frame,
            timestamp=now,
            wall_time=time.time(),
            fps=fps,
            index=self._frame_index,
        )

    def release(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None

    @property
    def is_open(self) -> bool:
        return self._picam is not None


def create_camera(config: dict) -> CameraBackend:
    """按配置选择后端。'auto' 在树莓派上选 Picamera2，其余用 OpenCV。"""
    cam_cfg = config.get("camera", {})
    backend = str(cam_cfg.get("backend", "auto")).lower()

    kwargs = dict(
        index=cam_cfg.get("index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
        mirror=cam_cfg.get("mirror", True),
    )

    if backend == "opencv":
        return OpenCVCamera(**kwargs)

    if backend == "picamera2":
        return PiCamera2Backend(**kwargs)

    # auto
    if _is_raspberry_pi():
        try:
            return PiCamera2Backend(**kwargs)
        except Exception:
            pass

    return OpenCVCamera(**kwargs)


def _is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model", "r", encoding="utf-8") as fh:
            return "raspberry pi" in fh.read().lower()
    except Exception:
        return False
