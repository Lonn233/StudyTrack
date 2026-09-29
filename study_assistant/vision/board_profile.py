"""板卡识别与分频档位。

笔记本上 640px YOLO + MediaPipe 随便跑；树莓派 4B 上必须降频。这里
按 `/proc/device-tree/model` 自动选档，环境变量可逐项覆盖。

设计原则：**只降频，不改逻辑**。所有行为判定基于"持续秒数"，降频对
结论无损（详见 README 的"为什么降频不影响准确率"）。
"""

from __future__ import annotations

import os
from pathlib import Path

# 各板卡推荐的视觉推理间隔（毫秒）
PROFILES: dict[str, dict] = {
    # 笔记本 / 台式机：全速
    "desktop": {
        "hands": 66,        # ~15 Hz
        "objects": 200,     # 5 Hz
    },
    # 树莓派 5 / CM4：四核 A76，能扛住较高频率
    "pi5": {
        "hands": 100,       # 10 Hz
        "objects": 400,     # 2.5 Hz
    },
    # 树莓派 4B（4×A72 @1.5GHz）：本项目的目标板
    "pi4": {
        "hands": 150,       # ~6.7 Hz
        "objects": 600,     # ~1.7 Hz
    },
    # 树莓派 Zero 2 W：救急档，行为融合会明显变钝
    "pi_zero": {
        "hands": 250,
        "objects": 1000,
    },
}

_ENV_KEYS = {
    "hands": "STUDYASSISTANT_HANDS_INTERVAL_MS",
    "objects": "STUDYASSISTANT_OBJECTS_INTERVAL_MS",
}


def detect_board() -> str:
    """返回板卡代号（desktop / pi4 / pi5 / pi_zero / other）。"""
    model_path = Path("/proc/device-tree/model")

    raw = ""
    if model_path.exists():
        try:
            raw = model_path.read_text(errors="ignore").strip().lower()
        except OSError:
            raw = ""

    # 非 Linux（Windows 上 /proc 不存在）——也允许用环境变量强制指定
    if not raw:
        raw = os.environ.get("STUDYASSISTANT_BOARD", "").strip().lower()

    if not raw:
        return "desktop"

    if "zero 2" in raw or "zero2" in raw:
        return "pi_zero"
    if "raspberry pi 5" in raw:
        return "pi5"
    if "raspberry pi 4" in raw or "compute module 4" in raw:
        return "pi4"
    if "raspberry pi" in raw:
        return "pi4"          # 未知型号，按最保守的 Pi 档处理
    if "jetson" in raw:
        return "pi5"

    return "desktop"


def intervals_for_board(board: str | None = None) -> dict:
    """取该板卡的推理间隔，并让环境变量逐项覆盖。

    环境变量写错（非数字 / 非正数）只警告，不抛异常 —— 免得单位写错
    （66 vs 66.0 vs "66ms"）就让整个程序起不来。
    """
    board = board or detect_board()
    intervals = dict(PROFILES.get(board, PROFILES["desktop"]))

    for key, env_name in _ENV_KEYS.items():
        raw = os.environ.get(env_name)

        if raw is None or raw == "":
            continue

        try:
            value = int(float(raw))
        except (TypeError, ValueError):
            print(f"[board] 忽略非法环境变量 {env_name}={raw!r}")
            continue

        if value <= 0:
            print(f"[board] 忽略非正数 {env_name}={value}")
            continue

        intervals[key] = value

    return intervals


def describe(board: str | None = None) -> str:
    board = board or detect_board()
    itv = intervals_for_board(board)
    label = {
        "desktop": "桌面/笔记本",
        "pi4": "树莓派 4B",
        "pi5": "树莓派 5 / CM4",
        "pi_zero": "树莓派 Zero 2 W",
    }.get(board, board)
    return f"{label}：手部 {itv['hands']} ms、物体 {itv['objects']} ms"
