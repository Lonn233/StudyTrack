"""桌面 ROI 定义与管理。

所有行为判断都基于"手相对桌面的位置 + 物体相对桌面的位置 + 手-物空间
关系"，而不是全局图像坐标。标定结果保存在 config/calibration.json。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ROI:
    """归一化矩形区域（0..1），与分辨率无关。"""

    name: str
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 1.0
    y1: float = 1.0
    color: tuple = (120, 220, 255)

    def contains(self, x: float, y: float) -> bool:
        """点是否在区域内（x, y 为归一化坐标）。"""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def clamp(self) -> None:
        self.x0 = min(max(self.x0, 0.0), 1.0)
        self.y0 = min(max(self.y0, 0.0), 1.0)
        self.x1 = min(max(self.x1, 0.0), 1.0)
        self.y1 = min(max(self.y1, 0.0), 1.0)

        if self.x1 < self.x0:
            self.x0, self.x1 = self.x1, self.x0
        if self.y1 < self.y0:
            self.y0, self.y1 = self.y1, self.y0

    @property
    def is_valid(self) -> bool:
        return (self.x1 - self.x0) > 0.01 and (self.y1 - self.y0) > 0.01

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x0 * width),
            int(self.y0 * height),
            int(self.x1 * width),
            int(self.y1 * height),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x0": round(self.x0, 4),
            "y0": round(self.y0, 4),
            "x1": round(self.x1, 4),
            "y1": round(self.y1, 4),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ROI":
        return cls(
            name=data.get("name", "roi"),
            x0=float(data.get("x0", 0.0)),
            y0=float(data.get("y0", 0.0)),
            x1=float(data.get("x1", 1.0)),
            y1=float(data.get("y1", 1.0)),
        )


# 标定需要收集的 ROI 及其默认范围与用途说明。
#
# ⚠️ 默认值刻意做成**互不重叠**：`paper` 在上方中部、`keyboard` 在下方中部。
# 如果两者重叠，"手在键盘上"会同时命中 `paper`，电脑学习判定的 `not paper_contact`
# 前置条件就失效了，COMPUTER_STUDY 永远抢不过 PAPER_STUDY。
# 真实使用时应让用户按自己的桌面重新标定。
ROI_SPECS = [
    ("desk", "整个桌面区域", 0.05, 0.15, 0.95, 0.95),
    ("study", "主要学习区域", 0.20, 0.20, 0.80, 0.92),
    ("left_hand", "左手常用区域", 0.04, 0.35, 0.32, 0.95),
    ("right_hand", "右手常用区域", 0.68, 0.35, 0.96, 0.95),
    ("paper", "书本/纸张常用区域", 0.24, 0.30, 0.72, 0.58),
    ("keyboard", "键盘/电脑区域", 0.22, 0.64, 0.80, 0.94),
    ("phone", "手机通常放置区域（可选）", 0.02, 0.50, 0.20, 0.92),
]


@dataclass
class Calibration:
    """一次完整的桌面标定结果。"""

    frame_width: int = 640
    frame_height: int = 480
    rois: dict = field(default_factory=dict)
    completed: bool = False
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.rois:
            self.rois = {
                name: ROI(name, x0, y0, x1, y1)
                for name, _label, x0, y0, x1, y1 in ROI_SPECS
            }
        else:
            restored = {}
            for name, value in self.rois.items():
                if isinstance(value, ROI):
                    restored[name] = value
                else:
                    restored[name] = ROI.from_dict(value)
            self.rois = restored

    def get(self, name: str) -> ROI | None:
        roi = self.rois.get(name)
        return roi if (roi is not None and roi.is_valid) else None

    def contains(self, roi_name: str, x: float, y: float) -> bool:
        """判断归一化点是否落在某个 ROI 内；ROI 无效时返回 False。"""
        roi = self.get(roi_name)
        return roi.contains(x, y) if roi else False

    def to_dict(self) -> dict:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "completed": self.completed,
            "updated_at": self.updated_at,
            "rois": {name: roi.to_dict() for name, roi in self.rois.items()},
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path | str) -> "Calibration":
        path = Path(path)

        if not path.exists():
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return cls()

        calib = cls(
            frame_width=int(data.get("frame_width", 640)),
            frame_height=int(data.get("frame_height", 480)),
            rois=data.get("rois", {}),
            completed=bool(data.get("completed", False)),
            updated_at=float(data.get("updated_at", 0.0)),
        )
        return calib


def default_calibration_path() -> Path:
    return (
        Path(__file__).resolve().parents[2] / "config" / "calibration.json"
    )
