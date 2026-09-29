"""专注状态机（带滞后 / 迟滞）。

输入：时序融合后的稳定行为
输出：对外宣布的 FocusState

迟滞规则（防止状态疯狂跳变）：
  * 进入 DISTRACTED 要连续 `to_distracted_seconds`（默认 3s）；
  * 回到 FOCUSED 要连续 `to_focused_seconds`（默认 2s）—— 恢复比沦陷
    更容易，因为学习过程中本来就有小停顿；
  * 手短暂离开（< away_grace_seconds）**保持上一状态**，不算离席；
  * 超过 `away_confirm_seconds` 才确认 AWAY；
  * 相机丢失画面 → UNCERTAIN，不产生任何分心事件。

这个模块是"责任判定"的唯一出口：只有它说 DISTRACTED，通知管理器才
允许提醒。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .states import BEHAVIOR_TO_FOCUS, Behavior, FocusState


@dataclass
class StateTransition:
    """一次状态切换。"""

    from_state: FocusState
    to_state: FocusState
    timestamp: float
    reason: str = ""
    behavior: Behavior = Behavior.UNKNOWN


@dataclass
class StateSnapshot:
    """状态机的当前全貌。"""

    state: FocusState = FocusState.UNCERTAIN
    behavior: Behavior = Behavior.UNKNOWN
    confidence: float = 0.0
    since: float = 0.0
    duration: float = 0.0            # 当前状态已持续多久
    candidate: FocusState | None = None   # 正在逼近的状态
    candidate_progress: float = 0.0       # 0..1
    reason: str = ""
    counts: dict = field(default_factory=dict)   # 各状态累计秒数


class FocusStateMachine:
    """带滞后的状态机。"""

    def __init__(self, config):
        b = config.behavior

        self.to_distracted = float(b.get("to_distracted_seconds", 3.0))
        self.to_focused = float(b.get("to_focused_seconds", 2.0))
        self.away_grace = float(b.get("away_grace_seconds", 3.0))
        self.away_confirm = float(b.get("away_confirm_seconds", 8.0))

        self.state = FocusState.UNCERTAIN
        self._state_start = 0.0
        self._last_t = 0.0

        # 候选状态追踪（连续多久了）
        self._candidate: FocusState | None = None
        self._candidate_since = 0.0
        self._candidate_last_seen = 0.0

        self._accumulated: dict[FocusState, float] = {
            s: 0.0 for s in FocusState
        }

        self.transitions: list[StateTransition] = []
        self._max_transitions = 500

        # 外部强制暂停
        self.paused = False

        self._behavior = Behavior.UNKNOWN
        self._confidence = 0.0
        self._reason = "初始化"

        # 手/人 离席时间由外部注入
        self._hands_absent_for = 0.0
        self._person_absent_for = 0.0
        self._person_present = False
        self._camera_ok = True

    # ------------------------------------------------------------------

    def set_paused(self, paused: bool, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        if paused and not self.paused:
            self._transition(FocusState.PAUSED, now, "用户暂停", self._behavior)
        self.paused = bool(paused)

    def update(
        self,
        estimate,
        now: float | None = None,
        hands_absent_for: float = 0.0,
        person_absent_for: float = 0.0,
        person_present: bool = False,
        camera_ok: bool = True,
    ) -> StateSnapshot:
        now = time.perf_counter() if now is None else now
        dt = max(0.0, now - self._last_t) if self._last_t else 0.0
        self._last_t = now

        self._hands_absent_for = hands_absent_for
        self._person_absent_for = person_absent_for
        self._person_present = bool(person_present)
        self._camera_ok = camera_ok

        behavior = getattr(estimate, "behavior", Behavior.UNKNOWN)
        self._behavior = behavior
        self._confidence = float(getattr(estimate, "confidence", 0.0))

        # 累计各状态时间（用于统计）
        if dt > 0:
            self._accumulated[self.state] = self._accumulated.get(self.state, 0.0) + dt

        if self.paused:
            return self.snapshot(now)

        # ---------- 1) 相机 / 离席 优先判定 ----------
        if not camera_ok:
            self._seek(FocusState.UNCERTAIN, now, "摄像头无画面")
            return self.snapshot(now)

        away_signal = self._away_signal()

        if away_signal is not None:
            # 归到 UNCERTAIN 说明不是离席，而是"看不到手"——原因要如实写，
            # 这一行会直接显示在调试面板的"切换原因"里
            reason = (
                "看不到手（人在画面内）"
                if away_signal is FocusState.UNCERTAIN
                else "离席判定"
            )
            self._seek(away_signal, now, reason)
            return self.snapshot(now)

        # ---------- 2) 手离开 / 离席 类行为不驱动状态迁移 ----------
        # 「手短暂离开要保持上一状态」这条规则必须由 _away_signal 独家
        # 负责。否则 HAND_AWAY 会经由行为映射把状态推成 UNCERTAIN，
        # 让这个 grace 窗口形同虚设。
        if behavior in (Behavior.HAND_AWAY, Behavior.AWAY):
            return self.snapshot(now)

        # ---------- 3) 由行为映射目标状态 ----------
        target = BEHAVIOR_TO_FOCUS.get(behavior, FocusState.UNCERTAIN)

        # 行为识别不出来时不要贸然切换，保持现状（避免"识别中"乱跳）
        if behavior is Behavior.UNKNOWN:
            return self.snapshot(now)

        self._seek(target, now, f"行为={behavior.value}")
        return self.snapshot(now)

    # ------------------------------------------------------------------

    def _away_signal(self) -> FocusState | None:
        """判定是否处于离席路径。

        关键：手短暂离开（< away_grace）什么都不做 —— 保持原状态。

        第二关键：**手不在画面 ≠ 人离席**。摄像头架得高、只拍到上半身
        时，手放在腿上或桌子下面就会长时间检不到。这种情况若判成离席，
        会凭空累计离席时长（直接拉低专注分）并在事件流里写下"离开座位"
        这种与事实相反的记录。所以只要人还被真实检出，长时间看不到手
        只说明"证据不足"，不是离席。
        """
        # 人明确离开画面很久 → AWAY
        if self._person_absent_for >= self.away_confirm:
            return FocusState.AWAY

        # 手长时间不在画面里
        if self._hands_absent_for >= self.away_confirm:
            # 人还在画面里 → 不是离席，而是"看不到手，无法判断"
            if self._person_present:
                return FocusState.UNCERTAIN
            return FocusState.AWAY

        # 宽容期：什么都不做（保持上一状态）
        if self._hands_absent_for > self.away_grace:
            # 在 grace 和 confirm 之间，保持但不切换
            return self.state if self.state is not FocusState.UNCERTAIN else None

        return None

    # ------------------------------------------------------------------

    def _seek(self, target: FocusState, now: float, reason: str) -> None:
        """试图向 target 迁移，满足滞后条件才真正切换。"""
        if target is self.state:
            # 已经在该状态：清掉候选
            self._candidate = None
            self._candidate_since = 0.0
            self._reason = reason
            return

        # --- 候选计时 ---
        if target is self._candidate:
            # 候选延续（允许小空档）
            if now - self._candidate_last_seen > 1.0:
                # 中间断太久，重新计
                self._candidate_since = now
        else:
            self._candidate = target
            self._candidate_since = now

        self._candidate_last_seen = now
        candidate_duration = now - self._candidate_since

        # --- 需要的持续时间 ---
        required = self._required_duration(self.state, target)

        if candidate_duration >= required:
            self._transition(target, now, reason, self._behavior)
            self._candidate = None
            self._candidate_since = 0.0

    def _required_duration(self, current: FocusState, target: FocusState) -> float:
        """不同迁移方向需要不同的确认时间。"""
        # 走向分心最难，走向专注最容易
        if target is FocusState.DISTRACTED:
            return self.to_distracted

        if target is FocusState.FOCUSED:
            # 从分心/离席回到专注要稍微确认一下，但比"沦陷"快
            return self.to_focused

        if target is FocusState.AWAY:
            # ⚠️ 不要在这里再等 away_confirm —— `_away_signal()` 已经是
            # "缺失时间 >= away_confirm"之后才发出的信号，再计一次 8 秒
            # 等于要求离开 16 秒才认，实测会导致永远切不到 AWAY。
            return 0.5

        # UNCERTAIN 之类的中间态快速通过
        return 1.0

    # ------------------------------------------------------------------

    def _transition(
        self,
        target: FocusState,
        now: float,
        reason: str,
        behavior: Behavior,
    ) -> None:
        if target is self.state:
            return

        transition = StateTransition(
            from_state=self.state,
            to_state=target,
            timestamp=now,
            reason=reason,
            behavior=behavior,
        )

        self.transitions.append(transition)

        if len(self.transitions) > self._max_transitions:
            self.transitions = self.transitions[-self._max_transitions:]

        self.state = target
        self._state_start = now
        self._reason = reason

    # ------------------------------------------------------------------

    def snapshot(self, now: float | None = None) -> StateSnapshot:
        now = time.perf_counter() if now is None else now

        progress = 0.0
        if self._candidate is not None:
            required = self._required_duration(self.state, self._candidate)
            progress = min(1.0, (now - self._candidate_since) / max(required, 1e-6))

        return StateSnapshot(
            state=self.state,
            behavior=self._behavior,
            confidence=self._confidence,
            since=self._state_start,
            duration=max(0.0, now - self._state_start) if self._state_start else 0.0,
            candidate=self._candidate,
            candidate_progress=progress,
            reason=self._reason,
            counts=dict(self._accumulated),
        )

    # ------------------------------------------------------------------

    @property
    def accumulated(self) -> dict:
        return dict(self._accumulated)

    def total_tracked(self) -> float:
        return sum(self._accumulated.values())

    def reset(self) -> None:
        self.state = FocusState.UNCERTAIN
        self._state_start = 0.0
        self._last_t = 0.0
        self._candidate = None
        self._candidate_since = 0.0
        self._candidate_last_seen = 0.0
        self._accumulated = {s: 0.0 for s in FocusState}
        self.transitions.clear()
        self.paused = False
        self._behavior = Behavior.UNKNOWN
        self._reason = "已重置"
