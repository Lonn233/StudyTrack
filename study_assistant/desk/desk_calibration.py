"""桌面 ROI 标定视图（OpenCV 窗口 + 鼠标拖拽）。

第一次运行时引导用户依次框出：桌面 / 学习区 / 左手区 / 右手区 /
纸张区 / 键盘区 / 手机区。结果保存为归一化坐标，与分辨率无关。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from study_assistant.camera.camera_backend import CameraBackend
from study_assistant.desk.roi_manager import ROI_SPECS, Calibration, default_calibration_path

WINDOW = "Study Assistant - Desk Calibration"

# 各 ROI 的显示颜色（BGR）
ROI_COLORS = {
    "desk": (200, 200, 120),
    "study": (120, 230, 120),
    "left_hand": (230, 160, 90),
    "right_hand": (90, 160, 230),
    "paper": (200, 130, 240),
    "keyboard": (150, 200, 250),
    "phone": (120, 120, 250),
}


class _DragState:
    def __init__(self):
        self.dragging = False
        self.start = None
        self.current = None
        self.box = None


class CalibrationView:
    """引导式标定，返回 Calibration 对象。"""

    def __init__(self, camera: CameraBackend, window_scale: float = 1.0):
        self.camera = camera
        self.scale = float(window_scale)

        self.drag = _DragState()
        self.index = 0
        self.calib = Calibration(frame_width=camera.width, frame_height=camera.height)

        self._frame = None
        self._frame_size = (camera.width, camera.height)

    # ------------------------------------------------------------------
    # 鼠标
    # ------------------------------------------------------------------

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if self._frame is None:
            return

        fw, fh = self._frame_size
        sx = fw / max(1, self._frame.shape[1])
        sy = fh / max(1, self._frame.shape[0])

        px, py = x * sx, y * sy

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag.dragging = True
            self.drag.start = (px, py)
            self.drag.current = (px, py)

        elif event == cv2.EVENT_MOUSEMOVE and self.drag.dragging:
            self.drag.current = (px, py)

        elif event == cv2.EVENT_LBUTTONUP and self.drag.dragging:
            self.drag.dragging = False
            self.drag.current = (px, py)
            x0, y0 = self.drag.start
            x1, y1 = self.drag.current

            if abs(x1 - x0) > 8 and abs(y1 - y0) > 8:
                self.drag.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def _draw(self) -> np.ndarray:
        frame = self._frame.copy()
        fw, fh = self._frame_size
        h, w = frame.shape[:2]

        # 已完成的 ROI
        for i, (name, _label, *_rest) in enumerate(ROI_SPECS):
            if i >= self.index:
                continue
            roi = self.calib.get(name)
            if roi is None:
                continue
            x0, y0, x1, y1 = roi.to_pixels(w, h)
            color = ROI_COLORS.get(name, (200, 200, 200))
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, name, (x0 + 4, max(14, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 当前正在拖拽的框
        if self.drag.start and self.drag.current:
            x0, y0 = self.drag.start
            x1, y1 = self.drag.current
            sx = w / max(1, fw)
            sy = h / max(1, fh)
            cv2.rectangle(
                frame,
                (int(x0 * sx), int(y0 * sy)),
                (int(x1 * sx), int(y1 * sy)),
                (0, 255, 255), 2,
            )

        # 顶部提示条
        if self.index < len(ROI_SPECS):
            name, label, *_rest = ROI_SPECS[self.index]
            bar = frame.copy()
            cv2.rectangle(bar, (0, 0), (w, 46), (18, 22, 30), -1)
            frame = cv2.addWeighted(bar, 0.75, frame, 0.25, 0)

            cv2.putText(
                frame,
                f"[{self.index + 1}/{len(ROI_SPECS)}] {label}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 240, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "drag a box  |  SPACE=confirm  R=redo  S=skip  Q=finish",
                (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 200, 210), 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame, "All ROIs set - press Q to save & continue",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (120, 255, 160), 1, cv2.LINE_AA,
            )

        if self.scale != 1.0:
            frame = cv2.resize(
                frame, (int(w * self.scale), int(h * self.scale)),
                interpolation=cv2.INTER_LINEAR,
            )

        return frame

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> Calibration | None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

        print("\n=== 桌面标定 ===")
        print("用鼠标拖拽框出每个区域：")
        for i, (name, label, *_r) in enumerate(ROI_SPECS):
            print(f"  {i + 1}. {name:10s} {label}")
        print("\n操作：拖拽=画框  空格=确认  R=重画  S=跳过  Q=完成并保存\n")

        try:
            while True:
                packet = self.camera.read()
                if packet is None:
                    if cv2.waitKey(30) & 0xFF in (27, ord("q")):
                        break
                    continue

                self._frame = packet.frame
                self._frame_size = (self.camera.width, self.camera.height)

                cv2.imshow(WINDOW, self._draw())
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key == ord("r"):
                    self.drag.box = None
                    self.drag.start = None
                    self.drag.current = None
                    continue

                if key == ord("s"):
                    # 跳过：保留默认值
                    self.drag.box = None
                    self.drag.start = None
                    self.drag.current = None
                    self.index += 1
                    if self.index > len(ROI_SPECS):
                        self.index = len(ROI_SPECS)
                    continue

                if key == 32:  # SPACE
                    if self.index < len(ROI_SPECS) and self.drag.box is not None:
                        name = ROI_SPECS[self.index][0]
                        x0, y0, x1, y1 = self.drag.box

                        fw, fh = self._frame_size
                        roi = self.calib.rois[name]
                        roi.x0 = x0 / fw
                        roi.y0 = y0 / fh
                        roi.x1 = x1 / fw
                        roi.y1 = y1 / fh
                        roi.clamp()

                        print(f"  {name}: ({roi.x0:.2f},{roi.y0:.2f})-({roi.x1:.2f},{roi.y1:.2f})")

                        self.drag.box = None
                        self.drag.start = None
                        self.drag.current = None
                        self.index += 1

        finally:
            cv2.destroyWindow(WINDOW)

        if self.index == 0:
            print("未标定任何区域，已取消。")
            return None

        self.calib.completed = True
        import time as _t

        self.calib.updated_at = _t.time()

        path = default_calibration_path()
        self.calib.save(path)
        print(f"\n标定已保存: {path}")

        return self.calib


def load_or_default(path: Path | None = None) -> Calibration:
    return Calibration.load(path or default_calibration_path())
