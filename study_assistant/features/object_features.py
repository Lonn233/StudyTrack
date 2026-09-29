"""物体侧的时序特征。

YOLO 是 5 Hz 的，而且会漏检。所以"手机在桌上"这件事不能看单帧，必须
维护"持续存在时长"，并且容忍短暂的漏检（gap tolerance）。

这里回答三个问题：
  1. 这个物体现在在不在？（含漏检容忍）
  2. 它已经连续存在多久了？（PHONE_USE 需要"持续"才算）
  3. 它自己在动吗？（手机被拿起来 vs 一直在桌上）
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# 每种物体允许的"漏检容忍"时长（秒）——秒数内再次出现视为连续
DEFAULT_GAP_TOLERANCE = {
    "phone": 1.0,      # 手机小、易漏，给多一点耐心
    "book": 1.0,
    "laptop": 1.5,     # 大物体，漏检通常是遮挡
    "keyboard": 1.5,
    "mouse": 0.8,
    "cup": 0.8,
    "person": 2.0,
}


@dataclass
class ObjectPresence:
    """某一个类别当前的在场情况。"""

    label: str
    present: bool = False
    since: float = 0.0            # 本次连续在场的起点
    duration: float = 0.0         # 已连续在场秒数
    last_real_seen: float = 0.0   # 最后一次真实（非滑行）看到
    last_visible: float = 0.0     # 最后一次"认为在场"（含容忍窗口）
    gone_for: float = 0.0         # 已消失多久（present=False 时有意义）
    count: int = 0
    stale: bool = False           # 当前依据的是滑行位置

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "present": self.present,
            "duration": round(self.duration, 2),
            "count": self.count,
            "stale": self.stale,
        }


class ObjectPresenceTracker:
    """把逐帧的检测结果转成"持续在场时长"。"""

    def __init__(self, gap_tolerance: dict | None = None):
        self.gap_tolerance = dict(DEFAULT_GAP_TOLERANCE)
        if gap_tolerance:
            self.gap_tolerance.update(gap_tolerance)

        self._state: dict[str, ObjectPresence] = {}

    # ------------------------------------------------------------------

    def _state_for(self, label: str) -> ObjectPresence:
        if label not in self._state:
            self._state[label] = ObjectPresence(label=label)
        return self._state[label]

    def tolerance(self, label: str) -> float:
        return float(self.gap_tolerance.get(label, 1.0))

    # ------------------------------------------------------------------

    def update(self, tracked_objects: list, now: float | None = None) -> dict:
        """用当前跟踪结果更新所有类别的在场状态。

        tracked_objects: list[TrackedObject]（来自 ObjectTracker 快照）
        返回 {label: ObjectPresence}
        """
        now = time.perf_counter() if now is None else now

        seen: dict[str, list] = {}
        for obj in tracked_objects:
            seen.setdefault(obj.label, []).append(obj)

        # --- 本次见到的类别 ---
        for label, objs in seen.items():
            state = self._state_for(label)
            real = [o for o in objs if not o.stale]

            state.count = len(objs)
            state.last_visible = now
            state.stale = len(real) == 0

            if real:
                state.last_real_seen = now

            if not state.present:
                # 从"不在场"转为"在场"——但如果在容忍窗口内，就不算中断
                if state.since and (now - state.last_visible) <= self.tolerance(label):
                    # 短漏检，延续原来的起点（不改 since）
                    pass
                else:
                    state.since = now

                state.present = True

        # --- 本次没见到的类别：看是否超过容忍窗口 ---
        for label, state in self._state.items():
            if label in seen and state.present:
                continue

            gap = now - state.last_visible if state.last_visible else 1e9
            state.count = 0
            state.stale = False

            if state.present and gap <= self.tolerance(label):
                pass                     # 容忍期内仍视为在场
            else:
                state.present = False
                state.since = 0.0

            state.gone_for = gap

        # --- 刷新时长 ---
        for state in self._state.values():
            state.duration = (now - state.since) if (state.present and state.since) else 0.0

        return self._state

    # ------------------------------------------------------------------

    def get(self, label: str) -> ObjectPresence:
        return self._state_for(label)

    def is_present(self, label: str, min_duration: float = 0.0) -> bool:
        state = self.get(label)
        return state.present and state.duration >= min_duration

    def duration(self, label: str) -> float:
        return self.get(label).duration

    def gone_for(self, label: str) -> float:
        return self.get(label).gone_for

    def any_visible(self) -> bool:
        return any(s.present for s in self._state.values())

    def clear(self) -> None:
        self._state.clear()

    def snapshot(self) -> dict:
        return {label: s.as_dict() for label, s in self._state.items()}
