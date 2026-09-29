"""专注评分与统计。

两个不同的概念，别混：
  * **状态**（FocusState）—— 现在是专注 / 分心 / 离席，离散；
  * **专注分**（focus score）—— 0..100 的连续指标，用来画曲线、算
    "这一小时学得怎么样"。

评分公式（可解释、可调）：
    基础分 = 150 × 有效专注时长 / 总时长          （封顶 100）
    扣分   = 分心惩罚 + 离席惩罚 + 切换惩罚
    - 分心惩罚：每秒 -1.2，最多扣 40
    - 离席惩罚：每秒 -0.5，最多扣 20
    - 切换惩罚：状态切换过频（> 12 次/10分钟）额外扣，最多 15

为什么这么设计：单纯看"专注占比"会把"频繁切来切去但每次都很短"的
情况算得很好，而实际上这种学习质量很差 —— 所以加了切换惩罚。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .states import Behavior, FocusState


@dataclass
class FocusStats:
    """一段时间内的统计结果。"""

    total_seconds: float = 0.0
    focused_seconds: float = 0.0
    distracted_seconds: float = 0.0
    away_seconds: float = 0.0
    uncertain_seconds: float = 0.0
    paused_seconds: float = 0.0

    transitions: int = 0
    behavior_seconds: dict = field(default_factory=dict)   # {Behavior.value: 秒}
    longest_focus_streak: float = 0.0
    current_focus_streak: float = 0.0

    focus_score: float = 0.0
    focus_ratio: float = 0.0

    def as_dict(self) -> dict:
        return {
            "total_seconds": round(self.total_seconds, 1),
            "focused_seconds": round(self.focused_seconds, 1),
            "distracted_seconds": round(self.distracted_seconds, 1),
            "away_seconds": round(self.away_seconds, 1),
            "uncertain_seconds": round(self.uncertain_seconds, 1),
            "paused_seconds": round(self.paused_seconds, 1),
            "transitions": self.transitions,
            "focus_score": round(self.focus_score, 1),
            "focus_ratio": round(self.focus_ratio, 3),
            "longest_focus_streak": round(self.longest_focus_streak, 1),
            "behavior_seconds": {
                k: round(v, 1) for k, v in self.behavior_seconds.items()
            },
        }


def compute_focus_score(
    counts: dict,
    transitions: int = 0,
    behavior_seconds: dict | None = None,
    longest_streak: float = 0.0,
) -> float:
    """按状态累计时长算专注分（0..100）。

    counts: {FocusState: 秒}
    """
    total = sum(
        counts.get(s, 0.0)
        for s in (
            FocusState.FOCUSED, FocusState.DISTRACTED,
            FocusState.AWAY, FocusState.UNCERTAIN,
        )
    )

    if total < 5.0:
        return 0.0

    focused = counts.get(FocusState.FOCUSED, 0.0)
    distracted = counts.get(FocusState.DISTRACTED, 0.0)
    away = counts.get(FocusState.AWAY, 0.0)

    # --- 基础分 ---
    base = 100.0 * focused / total

    # --- 分心惩罚 ---
    distracted_ratio = distracted / total
    distracted_penalty = min(40.0, 40.0 * distracted_ratio)

    # --- 离席惩罚 ---
    away_ratio = away / total
    away_penalty = min(20.0, 30.0 * away_ratio)

    # --- 切换惩罚：每 10 分钟超过 12 次开始扣 ---
    hours = total / 3600.0
    expected = max(1.0, 12.0 * (hours * 6.0))
    excess = max(0.0, transitions - expected) / max(expected, 1.0)
    switch_penalty = min(15.0, 15.0 * excess)

    # --- 长专注奖励：单段超过 15 分钟加一点分 ---
    streak_bonus = min(8.0, max(0.0, (longest_streak - 900.0) / 900.0) * 8.0)

    score = base - distracted_penalty - away_penalty - switch_penalty + streak_bonus

    return max(0.0, min(100.0, score))


def build_stats(
    counts: dict,
    transitions: list,
    behavior_seconds: dict,
    now: float,
    session_start: float,
    longest_streak: float = 0.0,
    current_streak: float = 0.0,
) -> FocusStats:
    """把状态机的累计数据整理成统计对象。"""
    total = sum(
        counts.get(s, 0.0)
        for s in (
            FocusState.FOCUSED, FocusState.DISTRACTED,
            FocusState.AWAY, FocusState.UNCERTAIN,
        )
    )

    stats = FocusStats(
        total_seconds=total,
        focused_seconds=counts.get(FocusState.FOCUSED, 0.0),
        distracted_seconds=counts.get(FocusState.DISTRACTED, 0.0),
        away_seconds=counts.get(FocusState.AWAY, 0.0),
        uncertain_seconds=counts.get(FocusState.UNCERTAIN, 0.0),
        paused_seconds=counts.get(FocusState.PAUSED, 0.0),
        transitions=len(transitions),
        behavior_seconds=dict(behavior_seconds),
        longest_focus_streak=longest_streak,
        current_focus_streak=current_streak,
    )

    stats.focus_ratio = (stats.focused_seconds / total) if total > 0 else 0.0
    stats.focus_score = compute_focus_score(
        counts,
        transitions=len(transitions),
        behavior_seconds=behavior_seconds,
        longest_streak=longest_streak,
    )

    return stats


def grade(score: float) -> tuple[str, str]:
    """把分数翻译成等级与一句人话。"""
    if score >= 90:
        return "A", "状态极佳，保持"
    if score >= 75:
        return "B", "整体不错，偶有走神"
    if score >= 60:
        return "C", "还行，但中断偏多"
    if score >= 40:
        return "D", "分心较多，建议收一收"
    return "E", "几乎没学进去，先解决干扰源"


def format_duration(seconds: float) -> str:
    """秒 → 1h 23m 45s / 5m 12s / 42s"""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
