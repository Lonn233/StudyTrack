"""枚举本机可用的（UVC）摄像头。

设计要点：

- Windows 上用 DirectShow（pygrabber）拿**真实设备名**——它的设备顺序与
  OpenCV ``CAP_DSHOW`` 的索引顺序一致，所以 ``names[i]`` 就是索引 i 的名字。
  pygrabber 缺失或枚举失败时，退化为「索引 0..max_index 逐个探测」。
- 每个候选索引都会短暂打开读一帧（``probe=True`` 时），确认它真的可用，
  并记录实际分辨率。探测是短暂占用，几十到几百毫秒，只在启动和
  「重新扫描」时调用，不进帧循环。
- **双目设备**：多数 UVC 双目相机就是把左右目各注册成一个普通摄像头，
  会以两个条目出现在列表里，分别切换即可看到各自画面。
  深度流（Depth）不在本项目范围内：手/物体检测只需要 RGB，
  深度图需要厂商 SDK（RealSense / Orbbec 等）另行接入。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class CameraInfo:
    """一个可用摄像头条目。"""

    index: int
    name: str            # 设备名；拿不到时用「摄像头 N」
    ok: bool             # OpenCV 能否真的打开并读到帧
    width: int = 0       # 探测时实际读到的分辨率（0 = 未探测）
    height: int = 0


def _dshow_device_names() -> list[str] | None:
    """Windows DirectShow 输入设备名列表；非 Windows / 失败返回 None。"""
    if sys.platform != "win32":
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph

        names = FilterGraph().get_input_devices()
        return list(names) if names else []
    except Exception:
        return None


def _probe(index: int) -> tuple[bool, int, int]:
    """尝试用 CAP_DSHOW 打开索引并读到一帧，返回 (ok, 宽, 高)。"""
    try:
        import cv2

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return False, 0, 0

        ok, w, h = False, 0, 0
        for _ in range(6):  # 头几帧可能是黑帧（自动曝光未稳定），多读几次
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                break
        cap.release()
        return ok, w, h
    except Exception:
        return False, 0, 0


def enumerate_cameras(
    max_index: int = 6, probe: bool = True, skip_index: int | None = None
) -> list[CameraInfo]:
    """列出可用摄像头。

    - 设备名来自 DirectShow；设备名列表之外的索引（最多到 max_index）
      也会探测，名字显示为「摄像头 N」。
    - ``skip_index`` 是当前正被本程序使用的索引——它显然可用，跳过探测
      （否则自己占着自己，会误报「打不开」）。
    """
    names = _dshow_device_names()
    if names is None:
        names = []
        candidates = list(range(max_index + 1))
    else:
        candidates = list(range(max(len(names), 0)))
        for i in range(len(names), max_index + 1):
            candidates.append(i)

    out: list[CameraInfo] = []
    for i in candidates:
        name = names[i] if i < len(names) else f"摄像头 {i}"
        if i == skip_index:
            out.append(CameraInfo(index=i, name=name, ok=True))
            continue
        if probe:
            ok, w, h = _probe(i)
        else:
            ok, w, h = True, 0, 0
        out.append(CameraInfo(index=i, name=name, ok=ok, width=w, height=h))
    return out


def describe(info: CameraInfo, current: bool = False) -> str:
    """菜单 / 日志里的一行描述，如 ``0: Integrated Webcam 640×480``。"""
    parts = [f"{info.index}: {info.name}"]
    if info.width and info.height:
        parts.append(f"{info.width}×{info.height}")
    if not info.ok:
        parts.append("打不开")
    return " · ".join(parts) + ("（当前）" if current else "")
