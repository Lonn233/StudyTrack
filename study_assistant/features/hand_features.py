"""手部运动特征提取。

对每只手维护一段轨迹历史，并在 1s / 3s / 5s 三个时间窗口上计算：
速度、加速度、轨迹长度、运动范围、方向变化率。

设计要点：**不能**用"手在动 = 分心"这种逻辑。写字本身就是持续运动。
真正有意义的是运动模式（幅度 / 频率 / 方向稳定性）与目标物的结合。
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

WINDOWS = (1.0, 3.0, 5.0)


@dataclass
class HandMotion:
    """一段窗口内的运动统计（长度单位均为归一化屏幕比例）。"""

    label: str
    window: float
    samples: int = 0
    speed: float = 0.0            # 平均速度 /秒
    speed_std: float = 0.0        # 速度波动（写字是持续小波动）
    accel: float = 0.0            # 平均加速度
    trajectory_length: float = 0.0
    motion_range: float = 0.0     # 轨迹包围盒对角线长度
    direction_change_rate: float = 0.0   # 方向反转次数 /秒
    net_displacement: float = 0.0 # 首尾位移
    is_moving: bool = False

    @property
    def is_local(self) -> bool:
        """小幅局部运动：轨迹长度相对运动范围明显偏长（来回划）。"""
        if self.motion_range < 1e-6:
            return self.trajectory_length < 0.5
        return self.trajectory_length / max(self.motion_range, 1e-6) > 1.4


class _Track:
    """单只手的轨迹缓冲。"""

    def __init__(self, max_seconds: float = 5.5):
        self.max_seconds = max_seconds
        self.samples: deque = deque()   # (t, x, y)

    def add(self, t: float, x: float, y: float) -> None:
        self.samples.append((t, x, y))
        cutoff = t - self.max_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def slice(self, now: float, window: float) -> list:
        cutoff = now - window
        return [s for s in self.samples if s[0] >= cutoff]

    def clear(self) -> None:
        self.samples.clear()


class HandTracker:
    """维护左右手轨迹并输出多窗口运动特征。"""

    def __init__(self, windows: tuple = WINDOWS):
        self.windows = windows
        self._tracks: dict[str, _Track] = {}
        self._last_present: dict[str, float] = {}
        # 第一次真正跑手部推理的时刻。用来回答"从没见手时，手离开多久了"
        self._started_t: float = 0.0

    def update(self, hands: list, now: float) -> None:
        """用当前帧的手部检测结果更新轨迹。"""
        if not self._started_t:
            self._started_t = now

        seen = set()

        for hand in hands:
            label = hand.label
            seen.add(label)

            if label not in self._tracks:
                self._tracks[label] = _Track()

            track = self._tracks[label]

            # 长时间消失后重新出现：清空历史，避免跨断层的假速度
            last = self._last_present.get(label)
            if last is not None and now - last > 1.5:
                track.clear()

            track.add(now, hand.palm_center[0], hand.palm_center[1])
            self._last_present[label] = now

        # 消失超过窗口的手直接清掉，避免陈旧数据被当作"静止"
        for label in list(self._tracks.keys()):
            last = self._last_present.get(label)
            if last is not None and now - last > 2.0 and label not in seen:
                self._tracks[label].clear()

    def motion(self, label: str, window: float, now: float) -> HandMotion:
        track = self._tracks.get(label)
        result = HandMotion(label=label, window=window)

        if track is None:
            return result

        pts = track.slice(now, window)
        result.samples = len(pts)

        if len(pts) < 2:
            return result

        # 速度序列
        speeds = []
        direction_changes = 0
        prev_dir = None

        prev_t, prev_x, prev_y = pts[0]

        for t, x, y in pts[1:]:
            dt = t - prev_t

            if dt <= 1e-6:
                continue

            dx, dy = x - prev_x, y - prev_y
            dist = math.hypot(dx, dy)
            speeds.append(dist / dt)
            result.trajectory_length += dist

            if dist > 1e-4:
                direction = math.atan2(dy, dx)
                if prev_dir is not None:
                    delta = abs(math.atan2(
                        math.sin(direction - prev_dir),
                        math.cos(direction - prev_dir),
                    ))
                    # 接近反向（>120°）记一次方向反转
                    if delta > math.radians(120):
                        direction_changes += 1
                prev_dir = direction

            prev_t, prev_x, prev_y = t, x, y

        if speeds:
            result.speed = sum(speeds) / len(speeds)
            mean = result.speed
            result.speed_std = math.sqrt(
                sum((s - mean) ** 2 for s in speeds) / len(speeds)
            )

            if len(speeds) > 1:
                accels = [
                    (speeds[i] - speeds[i - 1]) / max(pts[i + 1][0] - pts[i][0], 1e-6)
                    for i in range(1, len(speeds))
                ]
                result.accel = sum(abs(a) for a in accels) / len(accels)

        duration = max(pts[-1][0] - pts[0][0], 1e-6)
        result.direction_change_rate = direction_changes / duration

        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        result.motion_range = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        result.net_displacement = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])

        # 有意义的运动：轨迹长度超过阈值
        result.is_moving = result.trajectory_length > 0.02

        return result

    def all_motions(self, now: float) -> dict:
        """返回 {window: {label: HandMotion}}。"""
        out = {}
        for window in self.windows:
            out[window] = {
                label: self.motion(label, window, now)
                for label in self._tracks
            }
        return out

    def present_labels(self, now: float, max_age: float = 0.4) -> list:
        """最近仍然可见的手（避免用陈旧数据判断"手在"）。"""
        return [
            label for label, last in self._last_present.items()
            if now - last <= max_age
        ]

    def last_seen(self, label: str) -> float | None:
        return self._last_present.get(label)

    def recent_points(self, label: str, window: float = 3.0, now: float | None = None) -> list:
        """最近 window 秒内该手的轨迹点 [(x, y), ...]，用于画轨迹线。"""
        track = self._tracks.get(label)

        if track is None:
            return []

        now = time.perf_counter() if now is None else now
        return [(x, y) for _t, x, y in track.slice(now, window)]

    def trajectories(self, window: float = 3.0, now: float | None = None) -> dict:
        """所有手的轨迹：{label: [(x, y), ...]}。"""
        now = time.perf_counter() if now is None else now
        return {
            label: self.recent_points(label, window, now)
            for label in list(self._tracks.keys())
        }

    def absent_for(self, now: float | None = None) -> float:
        """所有手都不见了多久（秒）。

        从未见过手时返回"从开始跟踪到现在"的时长，而不是一个哨兵大数。
        原来这里返回 `1e9`，本意是"肯定算长时间缺席"，但这个值会一路
        传到界面上，变成「手离开画面 1000000000.0s」这种显然不对的文案。
        语义上「从未见手」=「整个会话期间手都不在」，所以按会话时长算
        才是诚实的答案（也顺带让"开局就没人"从"立刻判离席"变成正常的
        8 秒确认）。
        """
        now = time.perf_counter() if now is None else now

        if not self._last_present:
            return max(0.0, now - self._started_t) if self._started_t else 0.0

        return now - max(self._last_present.values())

    def clear(self) -> None:
        for track in self._tracks.values():
            track.clear()
        self._last_present.clear()
        # 重置"开始跟踪"的时刻，否则重开会话后 absent_for 会带着上一段
        # 会话的时长继续累加
        self._started_t = 0.0
