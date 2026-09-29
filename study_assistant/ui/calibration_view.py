"""桌面标定的 UI 入口。

真正的拖拽交互在 `desk.desk_calibration`（OpenCV 窗口）里做 —— 那边
用鼠标画框最直观。这个模块只负责：

  1. 判断"要不要提示用户去标定"；
  2. 用当前摄像头跑一次标定并把结果落盘；
  3. 给 GUI 提供标定步骤的说明文案。

为什么标定重要：写字 / 阅读 / 用电脑的区分**主要靠区域**，不是靠物体
检测（书本、纸笔经常检不出来）。没标定的话，W 字形判断只能退化成
"有手 + 有动作"，准确率会明显下降。
"""

from __future__ import annotations

from pathlib import Path

from ..camera.camera_backend import create_camera
from ..desk.roi_manager import ROI_SPECS, Calibration, default_calibration_path

# GUI 里展示的步骤说明（与 ROI_SPECS 顺序一致）
STEP_GUIDE = [
    ("desk", "整个桌面区域", "把摄像头能看到的整张桌子框起来（稍大一点没关系）"),
    ("study", "主要学习区域", "框出你主要写字/看书的那一块，这是最重要的区域"),
    ("left_hand", "左手常用区域", "左手通常会活动到的范围（可跳过）"),
    ("right_hand", "右手常用区域", "右手通常会活动到的范围（可跳过）"),
    ("paper", "书本 / 纸张区域", "本子/纸张通常放的位置 —— 写字判定主要看这里"),
    ("keyboard", "键盘 / 电脑区域", "键盘或笔记本电脑的位置"),
    ("phone", "手机放置区域", "手机平时放的位置（标了能显著提升手机检测）"),
]


def calibration_path() -> Path:
    return default_calibration_path()


def load_calibration() -> Calibration:
    return Calibration.load(calibration_path())


def needs_calibration(calibration: Calibration | None = None) -> bool:
    """判断是否需要引导用户标定。"""
    calibration = calibration or load_calibration()
    return not calibration.completed


def calibration_summary(calibration: Calibration | None = None) -> str:
    """给界面用的一句话摘要。"""
    calibration = calibration or load_calibration()

    if not calibration.completed:
        return "尚未标定 —— 建议先做一次桌面标定，行为判定会准很多"

    valid = [name for name, roi in calibration.rois.items() if roi.is_valid]
    return f"已完成标定（{len(valid)} 个区域：{'、'.join(valid)}）"


def run_calibration(config, camera=None, scale: float = 1.0) -> Calibration | None:
    """跑一次交互式标定并保存。

    camera 为 None 时按配置自建一个（用完即释放）。
    """
    from ..desk.desk_calibration import CalibrationView

    own_camera = camera is None

    if own_camera:
        camera = create_camera(config.data)
        camera.open()

    try:
        view = CalibrationView(camera, window_scale=scale)
        result = view.run()
    finally:
        if own_camera and camera is not None:
            camera.release()

    if result is not None:
        result.save(calibration_path())
        print(f"[标定] 已保存到 {calibration_path()}")

    return result


def auto_calibration(config, camera=None, scale: float = 1.0) -> Calibration:
    """无人值守标定：直接落盘默认 ROI。

    用在验收测试与首次自检里 —— 保证"没有人工干预也能跑通全链路"。
    """
    from ..desk.roi_manager import Calibration as _Calibration

    camera_size = (640, 480)

    if camera is not None:
        camera_size = (camera.width or 640, camera.height or 480)

    calib = _Calibration(
        frame_width=camera_size[0],
        frame_height=camera_size[1],
        completed=True,
        updated_at=__import__("time").time(),
    )
    calib.save(calibration_path())

    return calib
