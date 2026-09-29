"""应用配置加载与默认值合并。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS: dict[str, Any] = {
    "app": {
        "name": "Study Assistant",
        "frame_width": 640,
        "frame_height": 480,
        "target_fps": 30,
    },
    "camera": {
        "backend": "auto",
        "index": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "mirror": True,
    },
    "vision": {
        "hands": {
            "enabled": True,
            "interval_ms": 66,
            "max_hands": 2,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "model_path": "models/hand_landmarker.task",
        },
        "objects": {
            "enabled": True,
            "interval_ms": 200,
            "backend": "onnx",
            "model_path": "models/yolo11n.onnx",
            "input_size": 320,
            "confidence": 0.35,
            "iou_threshold": 0.45,
            "keep_classes": [
                "person",
                "phone",
                "book",
                "laptop",
                "keyboard",
                "mouse",
                "cup",
            ],
        },
    },
    "behavior": {
        "window_seconds": 3.0,
        "fusion_hz": 10,
        "min_confidence": 0.45,
        "to_distracted_seconds": 3.0,
        "to_focused_seconds": 2.0,
        "away_grace_seconds": 3.0,
        "away_confirm_seconds": 8.0,
        # 人最近一次被**真实检出**之后，多久之内还算"确认在场"。
        # 这个信号的作用是**否决**离席：人都还在画面里，怎么可能离席？
        # 没有它的话，摄像头拍不到手（手在桌下）会被判成离席。
        "person_present_max_age": 3.0,
        "interaction_distance_px": 90,
        "interaction_overlap_min": 0.05,
        "writing": {
            "min_motion_energy": 0.015,
            "full_motion_energy": 0.05,
            "max_motion_range": 0.25,
            "min_duration": 2.0,
        },
        "reading": {
            "static_motion_energy": 0.006,
            # 静止度衰减上限。**原来是 0.018，与下限 0.006 挤在一起**：能量一到
            # 0.018 静止度就归零，而写字的能量项要 0.05 才满分 → 0.018~0.05 这
            # 一整段里阅读分已死在硬下限、写字分还在爬，"写字和阅读分不开"。
            # 放宽到 0.06 才能覆盖「认真读书 → 轻微翻动 → 写字」整条轴。
            "max_motion_energy": 0.06,
            # 静止是阅读的**必要条件**（否决权），不是加分项。
            # 手在动就不可能是在看书：能量超过 static_energy_grace 时阅读分→0。
            "static_energy_grace": 0.018,
            "min_duration": 2.5,
            "page_turn_grace": 2.0,
        },
        "phone": {
            "candidate_seconds": 2.0,
            "confirm_seconds": 3.0,
            "event_seconds": 10.0,
            "quick_look_seconds": 5.0,
            # 默认「必须检出手机物体」才定罪。位置（phone ROI）只是先验：
            # 出厂 phone 框覆盖画面左下整条竖带，看书搭着的手、敲键盘下探
            # 的手都会落进去，据此定罪会把看书/打字判成玩手机。假的"玩手机"
            # 会记分心、触发提醒、拉低专注分，代价远高于漏判。
            "require_object_contact": True,
            "position_only_penalty": 0.6,   # 关掉上面那条时，纯位置命中的折价
        },
        "fidgeting": {
            "min_motion_range": 0.30,
            "full_motion_range": 0.55,
            "min_direction_change": 0.5,
            # 小动作必须先**真的有动作**：手几乎不动时关键点的微小抖动
            # 能刷出很高的方向反转率（实测 4/s），没有能量下限就会把
            # "手压着键盘发呆"判成小动作。
            "min_motion_energy": 0.02,
            "min_duration": 4.0,
        },
        "computer": {
            "min_duration": 2.0,
            # --- 打字动作签名（几何平均，缺一环即归零）---
            "typing_dir_min": 2.0,        # 方向反转率 下限（次/秒）
            "typing_dir_full": 6.0,       # 达标（满分）
            "typing_burst_min": 0.45,     # 速度波动比 = std/speed 下限
            "typing_burst_full": 0.75,
            "typing_range_max": 0.12,     # 3s 幅度上限（2 倍处归零）
            "typing_energy_min": 0.10,    # 运动能量下限（排除静止）
            "typing_energy_full": 0.25,
            "typing_min_score": 0.45,     # 签名达标线
            "position_only_penalty": 0.6, # 只用位置、没检出键盘物体时折价
            "require_object_contact": False,  # True = 完全忽略位置通路
            "suppress_fidget": True,      # 打字成立时不再判小动作
        },
    },
    "notifications": {
        "enabled": True,
        "distracted_warn_seconds": 10,
        "distracted_soft_seconds": 20,
        "distracted_alert_seconds": 30,
        "cooldown_seconds": 60,
        "milestones_minutes": [30, 60, 90, 120],
    },
    "session": {
        "break_after_minutes": 45,
        "break_minutes": 5,
        "streak_grace_seconds": 20,
    },
    "storage": {
        "enabled": True,
        "database": "study_assistant.sqlite3",
        "sample_interval_seconds": 5.0,
    },
    "ui": {
        "eyes": {
            "width": 220,
            "height": 120,
            "always_on_top": True,
            "position": "bottom-right",
        },
        "debug": {"enabled": True},
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """配置容器，支持点式访问常用项。"""

    def __init__(self, data: dict, path: Path | None = None):
        self.data = data
        self.path = path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        if path is None:
            path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
        path = Path(path)

        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
        else:
            loaded = {}

        return cls(_deep_merge(_DEFAULTS, loaded), path)

    def get(self, section: str, key: str, default=None):
        return self.data.get(section, {}).get(key, default)

    @property
    def camera(self) -> dict:
        return self.data.get("camera", {})

    @property
    def vision(self) -> dict:
        return self.data.get("vision", {})

    @property
    def behavior(self) -> dict:
        return self.data.get("behavior", {})

    @property
    def notifications(self) -> dict:
        return self.data.get("notifications", {})

    @property
    def storage(self) -> dict:
        return self.data.get("storage", {})

    @property
    def ui(self) -> dict:
        return self.data.get("ui", {})

    @property
    def user_data_dir(self) -> Path:
        configured = self.data.get("app", {}).get("user_data_dir", "")
        if configured:
            return Path(configured)
        return Path.home() / "StudyAssistant"
