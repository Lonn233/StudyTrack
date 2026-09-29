"""合成输入工具：不用摄像头 / 不用模型，直接驱动 BehaviorCore。

为什么需要它：真机上要复现"手握着手机持续 4 秒"这种场景，得真的举着
手机坐 4 秒，还受光照、检测抖动影响。用合成输入可以把**逻辑**与
**感知**彻底分开验证 —— 感知层由 stage1 / pipeline 检查负责，逻辑层
由这里负责。

保真度上的两个刻意设计：
  1. 物体**一定**经过真实的 `ObjectTracker` 再喂给核心（生产路径也是
     这样）。直接把 DetectedObject 塞进去会绕过滑行 / 命中计数，
     测出来的东西和线上不是一回事；
  2. 时间轴是"虚拟秒"，从 100 开始单调递增 —— core 内部只做时间差，
     起点无所谓，但必须单调。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study_assistant.vision.hand_detector import Hand  # noqa: E402
from study_assistant.vision.object_detector import DetectedObject  # noqa: E402
from study_assistant.vision.object_tracker import ObjectTracker  # noqa: E402

# MediaPipe 21 点手型的"标准姿势"（局部坐标，y 向下，原点在掌心附近）
_CANONICAL = np.array([
    [0.00, 0.55, 0.0],      # 0  wrist
    [-0.16, 0.40, 0.0],     # 1
    [-0.29, 0.27, 0.0],     # 2
    [-0.38, 0.16, 0.0],     # 3
    [-0.44, 0.07, 0.0],     # 4  thumb tip
    [-0.11, 0.10, 0.0],     # 5
    [-0.13, -0.08, 0.0],    # 6
    [-0.14, -0.19, 0.0],    # 7
    [-0.15, -0.28, 0.0],    # 8  index tip
    [0.00, 0.06, 0.0],      # 9
    [0.00, -0.12, 0.0],     # 10
    [0.00, -0.24, 0.0],     # 11
    [0.00, -0.33, 0.0],     # 12 middle tip
    [0.11, 0.10, 0.0],      # 13
    [0.13, -0.07, 0.0],     # 14
    [0.15, -0.18, 0.0],     # 15
    [0.16, -0.27, 0.0],     # 16 ring tip
    [0.22, 0.16, 0.0],      # 17
    [0.26, 0.03, 0.0],      # 18
    [0.29, -0.07, 0.0],     # 19
    [0.32, -0.15, 0.0],     # 20 pinky tip
], dtype=np.float32)

FINGER_TIPS = (4, 8, 12, 16, 20)
PALM_INDICES = (0, 5, 9, 13, 17)


def make_hand(
    label: str = "right",
    center: tuple = (0.5, 0.5),
    size: float = 0.22,
    rotation: float = 0.0,
    timestamp: float = 0.0,
) -> Hand:
    """构造一只处于给定位置的合成手。

    size 是手的"高度"占画面的比例（真实握笔的手大约 0.15~0.30）。
    """
    pts = _CANONICAL.copy()

    if rotation:
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
        pts[:, :2] = pts[:, :2] @ rot.T

    pts[:, :2] = pts[:, :2] * size + np.array(center, dtype=np.float32)

    xs, ys = pts[:, 0], pts[:, 1]

    return Hand(
        label=label,
        handedness_score=0.95,
        landmarks=pts,
        wrist=(float(pts[0, 0]), float(pts[0, 1])),
        palm_center=(
            float(pts[list(PALM_INDICES), 0].mean()),
            float(pts[list(PALM_INDICES), 1].mean()),
        ),
        finger_tips=[(float(pts[i, 0]), float(pts[i, 1])) for i in FINGER_TIPS],
        bbox=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        timestamp=timestamp,
    )


def make_object(
    label: str = "phone",
    cx: float = 0.5,
    cy: float = 0.5,
    w: float = 0.08,
    h: float = 0.14,
    confidence: float = 0.8,
    timestamp: float = 0.0,
) -> DetectedObject:
    """构造一个合成物体（归一化 bbox）。"""
    return DetectedObject(
        label=label,
        coco_name=label,
        confidence=confidence,
        bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        timestamp=timestamp,
    )


# ----------------------------------------------------------------------
# 坐标：必须与 roi_manager.ROI_SPECS 的默认范围一致
# ----------------------------------------------------------------------

#: 各区域的中心点。每个点只能落在自己那个区域里 —— 否则会同时命中
#: paper 与 keyboard，场景就失去意义了。
ROI_CENTER = {
    "desk": (0.50, 0.55),
    "study": (0.50, 0.55),
    "left_hand": (0.28, 0.72),
    "right_hand": (0.84, 0.70),
    "paper": (0.48, 0.44),
    "keyboard": (0.51, 0.79),
    "phone": (0.11, 0.70),
}

#: 参与行为判定的区域（其余 ROI 只是背景 / 描述性用途）
BEHAVIOR_ROIS = ("paper", "keyboard", "phone")

#: 明确落在**所有行为区域之外**（只属于 desk）的位置，用来做"手在桌上
#: 但不做任何有意义的事"这类场景（小动作 / 发呆）。
NEUTRAL_SPOT = (0.50, 0.17)


# ----------------------------------------------------------------------
# 场景驱动器
# ----------------------------------------------------------------------


class Scenario:
    """按固定步长推进 BehaviorCore，并保持时间单调递增。"""

    def __init__(
        self,
        core,
        dt: float = 1.0 / 30.0,
        start: float = 100.0,
        frame_width: int = 640,
        frame_height: int = 480,
    ):
        self.core = core
        self.dt = float(dt)
        self.now = float(start)
        self.frame_width = frame_width
        self.frame_height = frame_height

        # 物体一律走真实追踪器
        self.tracker = ObjectTracker()

        self.events: list = []
        self.states: list = []        # [(t, FocusState)]
        self.behaviors: list = []     # [(t, Behavior)]
        self.trajectory: list = []    # [(x, y)] 主手路径

        self.first_t: float | None = None

    # ------------------------------------------------------------------

    def step(
        self,
        count: int = 1,
        hands=None,
        objects=None,
        hands_fresh: bool = True,
        detect_objects: bool = True,
        camera_ok: bool = True,
    ) -> None:
        """推进 count 帧。

        hands   —— list[Hand]（None / [] 表示这一帧没检测到手）
        objects —— list[DetectedObject]；内部会经 ObjectTracker 变成轨迹
        detect_objects —— False 表示这一帧**没跑**物体推理（测试滑行）
        """
        hands = list(hands or [])

        for _ in range(count):
            self.now += self.dt

            if self.first_t is None:
                self.first_t = self.now

            if detect_objects:
                tracked = self.tracker.update(list(objects or []), self.now).objects
                objects_fresh = True
            else:
                tracked = self.tracker.snapshot(self.now).objects
                objects_fresh = False

            update = self.core.update(
                hands=hands,
                tracked_objects=tracked,
                now=self.now,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                camera_ok=camera_ok,
                hands_fresh=hands_fresh,
                objects_fresh=objects_fresh,
            )

            if update.events:
                self.events.extend(update.events)

            snapshot = update.snapshot

            if snapshot is not None:
                self.states.append((self.now, snapshot.state))
                self.behaviors.append((self.now, snapshot.behavior))

            if hands:
                self.trajectory.append(tuple(hands[0].palm_center))

    # ------------------------------------------------------------------
    # 语义化动作
    # ------------------------------------------------------------------

    def wiggle(
        self,
        seconds: float,
        roi: str = "paper",
        label: str = "right",
        amplitude: float = 0.012,
        frequency: float = 2.5,
        size: float = 0.22,
        objects=None,
    ) -> None:
        """在某个区域内做小幅来回运动（模拟写字 / 敲键盘）。"""
        base = ROI_CENTER[roi]
        steps = max(1, int(round(seconds / self.dt)))

        for i in range(steps):
            phase = i * self.dt * frequency * 2 * math.pi
            x = base[0] + math.sin(phase) * amplitude
            y = base[1] + math.cos(phase * 1.7) * amplitude * 0.5
            self.step(1, hands=[make_hand(label, (x, y), size=size)], objects=objects)

    def hold_still(
        self,
        seconds: float,
        roi: str = "paper",
        label: str = "right",
        size: float = 0.22,
        jitter: float = 0.0002,
        objects=None,
    ) -> None:
        """手停在某个区域里基本不动（模拟看书写字时手压着书）。

        jitter 默认 0.0002 —— 这个量级对应手掌速度约 0.004/s，落在
        `reading.static_motion_energy`（0.006）以下。如果给大了（例如
        0.0015 → 速度 0.03/s），就会被判成"在写字"，测试也就不再是
        在测阅读了。
        """
        base = ROI_CENTER[roi]
        steps = max(1, int(round(seconds / self.dt)))

        for i in range(steps):
            x = base[0] + math.sin(i * 0.7) * jitter
            y = base[1] + math.cos(i * 0.9) * jitter
            self.step(1, hands=[make_hand(label, (x, y), size=size)], objects=objects)

    def type_burst(
        self,
        seconds: float,
        roi: str = "keyboard",
        label: str = "right",
        burst: float = 0.14,
        pause: float = 0.09,
        amplitude: float = 0.014,
        size: float = 0.22,
        objects=None,
    ) -> None:
        """键击式脉冲：快速小幅击打 + 短暂停顿交替。

        为什么不能用 `wiggle` 冒充打字：`wiggle` 是**连续正弦**，方向反转率
        只有 2/s 左右、速度波动比约 0.26，而真实打字是"击打—停顿"交替的
        脉冲，速度波动比 ≈0.8。用平滑正弦测电脑学习，实际测到的是"手在
        键盘区来回划"—— 那正是应当被打字动作签名挡掉的假阳性。

        静止压着键盘则连 `typing_energy_min`（0.10）都过不了，
        所以"手压着键盘发呆"不会被判成电脑学习。
        """
        base = ROI_CENTER[roi]
        elapsed = 0.0

        while elapsed < seconds:
            steps = max(1, int(round(burst / self.dt)))
            for i in range(steps):
                phase = i * self.dt * 11.0 * 2 * math.pi
                x = base[0] + math.sin(phase) * amplitude
                y = base[1] + math.cos(phase * 1.3) * amplitude * 0.6
                self.step(1, hands=[make_hand(label, (x, y), size=size)], objects=objects)

            steps = max(1, int(round(pause / self.dt)))
            for i in range(steps):
                x = base[0] + math.sin(i * 0.7) * 0.0002
                y = base[1] + math.cos(i * 0.9) * 0.0002
                self.step(1, hands=[make_hand(label, (x, y), size=size)], objects=objects)

            elapsed += burst + pause

    def large_wave(
        self,
        seconds: float,
        center: tuple = NEUTRAL_SPOT,
        label: str = "right",
        amplitude_x: float = 0.20,
        amplitude_y: float = 0.06,
        speed: float = 6.0,
        objects=None,
    ) -> None:
        """大幅无序挥动（模拟摆弄东西 / 摸头发 / 转笔）。"""
        steps = max(1, int(round(seconds / self.dt)))

        for i in range(steps):
            phase = i * self.dt * speed
            x = center[0] + math.sin(phase) * amplitude_x
            y = center[1] + math.sin(phase * 2.3) * amplitude_y
            self.step(1, hands=[make_hand(label, (x, y), size=0.24)], objects=objects)

    def no_hands(self, seconds: float, objects=None) -> None:
        """手完全不在画面里（人可能还在，也可能走了）。"""
        steps = max(1, int(round(seconds / self.dt)))

        for _ in range(steps):
            self.step(1, hands=[], objects=objects)

    def no_objects(self, seconds: float, hands=None) -> None:
        """物体检测这一帧没跑（测试滑行与容忍）。"""
        steps = max(1, int(round(seconds / self.dt)))

        for _ in range(steps):
            self.step(1, hands=hands, objects=[], detect_objects=False)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def history(self, seq) -> list:
        """把序列压成 [(value, duration), ...]。"""
        out = []

        for _t, value in seq:
            if out and out[-1][0] is value:
                out[-1][1] += self.dt
            else:
                out.append([value, self.dt])

        return [(v, d) for v, d in out]

    def state_history(self) -> list:
        return self.history(self.states)

    def behavior_history(self) -> list:
        return self.history(self.behaviors)

    def elapsed(self) -> float:
        return (self.now - self.first_t) if self.first_t else 0.0

    def first_time_in_state(self, state, after: float = 0.0) -> float | None:
        """相对场景开始的第几秒**首次**进入该状态；从未进入返回 None。"""
        if self.first_t is None:
            return None

        for t, value in self.states:
            if value is state and (t - self.first_t) >= after:
                return t - self.first_t

        return None

    def first_time_in_behavior(self, behavior, after: float = 0.0) -> float | None:
        if self.first_t is None:
            return None

        for t, value in self.behaviors:
            if value is behavior and (t - self.first_t) >= after:
                return t - self.first_t

        return None

    def state_at(self, elapsed_seconds: float):
        """取场景开始后第 N 秒时的状态。"""
        if self.first_t is None:
            return None

        target = self.first_t + elapsed_seconds
        result = None

        for t, value in self.states:
            if t <= target:
                result = value
            else:
                break

        return result

    def behavior_at(self, elapsed_seconds: float):
        if self.first_t is None:
            return None

        target = self.first_t + elapsed_seconds
        result = None

        for t, value in self.behaviors:
            if t <= target:
                result = value
            else:
                break

        return result

    def ever_in_state(self, state) -> bool:
        return any(v is state for _t, v in self.states)

    def ever_in_behavior(self, behavior) -> bool:
        return any(v is behavior for _t, v in self.behaviors)

    def events_of(self, kind: str) -> list:
        return [e for e in self.events if e.kind == kind]

    def event_levels(self, kind: str) -> list:
        return [e.level for e in self.events if e.kind == kind]


# ----------------------------------------------------------------------


def load_test_config(overrides: dict | None = None):
    """加载工程配置（可覆盖），用于测试。"""
    from study_assistant.config import Config, _deep_merge

    config = Config.load(ROOT / "config" / "config.yaml")

    if overrides:
        config.data = _deep_merge(config.data, overrides)

    return config


def make_core(config=None, calibration=None, database=None, warmup_seconds=0.0,
              silent: bool = True):
    """建一个 BehaviorCore（默认关闭预热，测试里不需要）。

    silent=True 时不发系统提示音、不弹系统通知 —— 否则几十个场景跑下来
    会一直叮叮响，而且拖慢测试。
    """
    from study_assistant.core import BehaviorCore
    from study_assistant.desk.roi_manager import Calibration
    from study_assistant.notifications.notification_manager import NotificationManager

    config = config or load_test_config()

    notifications = NotificationManager(config, enable_system_sound=not silent)

    core = BehaviorCore(
        config,
        calibration or Calibration(),
        database=database,
        warmup_seconds=warmup_seconds,
        notifications=notifications,
    )
    core.hand_detector_available = True

    return core


def new_scenario(config=None, calibration=None, database=None, dt=1.0 / 30.0, **kwargs):
    """建一个已 start 的场景（时间轴从 0 之后开始）。"""
    core = make_core(config, calibration, database, warmup_seconds=0.0)
    core.start(now=-1e9)          # 把会话起点推到很早，预热直接失效

    return Scenario(core, dt=dt, **kwargs)
