"""行为与专注状态的统一定义。

这是整个项目的"词汇表"：视觉层产出原始观测，特征层产出运动/交互指标，
行为层把指标映射到这里的枚举，状态机再决定是否对外宣布。
"""

from __future__ import annotations

from enum import Enum


class Behavior(str, Enum):
    """用户正在做什么。"""

    WRITING = "WRITING"                # 写字 / 做题
    READING = "READING"                # 阅读（书或纸）
    COMPUTER_STUDY = "COMPUTER_STUDY"  # 用电脑学习 / 敲代码
    PHONE_USE = "PHONE_USE"            # 玩手机（分心）
    FIDGETING = "FIDGETING"            # 无目的摆弄（分心）
    IDLE = "IDLE"                      # 手在但没动作，也没明确目标
    HAND_AWAY = "HAND_AWAY"            # 手离开画面
    AWAY = "AWAY"                      # 人离开
    UNKNOWN = "UNKNOWN"                # 证据不足

    @property
    def is_productive(self) -> bool:
        return self in (Behavior.WRITING, Behavior.READING, Behavior.COMPUTER_STUDY)

    @property
    def is_distracting(self) -> bool:
        return self in (Behavior.PHONE_USE, Behavior.FIDGETING)


class FocusState(str, Enum):
    """对外宣布的专注状态（带时序滞后，不会瞬变）。"""

    FOCUSED = "FOCUSED"
    DISTRACTED = "DISTRACTED"
    AWAY = "AWAY"
    UNCERTAIN = "UNCERTAIN"
    PAUSED = "PAUSED"       # 用户主动暂停

    @property
    def is_active(self) -> bool:
        """是否算作"正在学"（用于有效时长统计）。"""
        return self in (FocusState.FOCUSED,)


# 行为中文名（UI / 日志用）
BEHAVIOR_LABELS = {
    Behavior.WRITING: "写字",
    Behavior.READING: "阅读",
    Behavior.COMPUTER_STUDY: "电脑学习",
    Behavior.PHONE_USE: "玩手机",
    Behavior.FIDGETING: "小动作",
    Behavior.IDLE: "静止",
    Behavior.HAND_AWAY: "手离开",
    Behavior.AWAY: "离席",
    Behavior.UNKNOWN: "识别中",
}

FOCUS_LABELS = {
    FocusState.FOCUSED: "专注",
    FocusState.DISTRACTED: "分心",
    FocusState.AWAY: "离席",
    FocusState.UNCERTAIN: "识别中",
    FocusState.PAUSED: "已暂停",
}

# 行为 → 状态的基础映射（状态机在此基础上叠加滞后与离席逻辑）
BEHAVIOR_TO_FOCUS = {
    Behavior.WRITING: FocusState.FOCUSED,
    Behavior.READING: FocusState.FOCUSED,
    Behavior.COMPUTER_STUDY: FocusState.FOCUSED,
    Behavior.PHONE_USE: FocusState.DISTRACTED,
    Behavior.FIDGETING: FocusState.DISTRACTED,
    Behavior.IDLE: FocusState.UNCERTAIN,
    Behavior.HAND_AWAY: FocusState.UNCERTAIN,
    Behavior.AWAY: FocusState.AWAY,
    Behavior.UNKNOWN: FocusState.UNCERTAIN,
}

# 行为 → 悬浮窗表情（键必须是 ui.eye_widget.EyeMood 里的常量值）
BEHAVIOR_TO_MOOD = {
    Behavior.WRITING: "WRITING",
    Behavior.READING: "READING",
    Behavior.COMPUTER_STUDY: "FOCUSED",
    Behavior.PHONE_USE: "PHONE_USE",
    Behavior.FIDGETING: "FIDGETING",
    Behavior.IDLE: "SLEEPY",
    Behavior.HAND_AWAY: "CURIOUS",
    Behavior.AWAY: "AWAY",
    Behavior.UNKNOWN: "NEUTRAL",
}

# 专注状态 → 表情（在行为未知、只有状态可用时兜底）
STATE_TO_MOOD = {
    FocusState.FOCUSED: "FOCUSED",
    FocusState.DISTRACTED: "WORRIED",
    FocusState.AWAY: "AWAY",
    FocusState.UNCERTAIN: "NEUTRAL",
    FocusState.PAUSED: "PAUSED",
}


def behavior_label(behavior: Behavior | str) -> str:
    if isinstance(behavior, str):
        try:
            behavior = Behavior(behavior)
        except ValueError:
            return str(behavior)
    return BEHAVIOR_LABELS.get(behavior, behavior.value)


def focus_label(state: FocusState | str) -> str:
    if isinstance(state, str):
        try:
            state = FocusState(state)
        except ValueError:
            return str(state)
    return FOCUS_LABELS.get(state, state.value)
