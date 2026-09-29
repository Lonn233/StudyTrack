"""会话管理：把状态机的输出变成"可记录的统计 + 可提醒的事件"。

职责边界：
  * 状态机说"现在分心" —— 会话管理器决定"这个分心要不要记一笔、要
    不要提醒、提醒到什么程度"；
  * 所有写数据库的动作都从这里发出，UI 不直接碰 DB；
  * 提醒是**升级式**的：先只改表情（10s），再到轻提示（20s），最后
    才出声（30s）。并且有冷却，避免每 30 秒骂一次。

分心事件的分级（来自配置）：
  distracted_warn_seconds   10  → 表情变化（不需要通知）
  distracted_soft_seconds   20  → 桌面通知 / 气泡
  distracted_alert_seconds  30  → 声音 + 通知
  cooldown_seconds          60  → 两次提醒之间的最短间隔
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..behavior.focus_score import FocusStats, build_stats, grade
from ..behavior.states import Behavior, FocusState, behavior_label, focus_label
from .timer_manager import TimerManager


@dataclass
class SessionEvent:
    """会话层产生的、需要被 UI / 通知消费的事件。"""

    kind: str                 # session_start | session_end | distraction |
                              # milestone | break_due | break_over | recovery |
                              # away_start | away_end | behavior_shift
    message: str = ""
    level: int = 0            # 0 静默 / 1 表情 / 2 轻提示 / 3 强提醒
    behavior: str = ""
    state: str = ""
    duration: float = 0.0
    minutes: int = 0
    ts: float = 0.0

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "message": self.message,
            "level": self.level,
            "behavior": self.behavior,
            "state": self.state,
            "duration": round(self.duration, 1),
            "minutes": self.minutes,
            "ts": self.ts,
        }


class SessionManager:
    """一次学习会话的完整生命周期。"""

    def __init__(self, config, database=None):
        self.config = config
        self.db = database

        n = config.notifications
        self.notify_enabled = bool(n.get("enabled", True))
        self.warn_seconds = float(n.get("distracted_warn_seconds", 10))
        self.soft_seconds = float(n.get("distracted_soft_seconds", 20))
        self.alert_seconds = float(n.get("distracted_alert_seconds", 30))
        self.cooldown_seconds = float(n.get("cooldown_seconds", 60))

        self.sample_interval = float(
            config.storage.get("sample_interval_seconds", 5.0)
        )
        self.storage_enabled = bool(config.storage.get("enabled", True))

        self.timer = TimerManager(config)

        self.session_id: int | None = None
        self.started_wall: float = 0.0
        self.active = False

        # --- 分心追踪 ---
        self._distracted_since: float | None = None
        self._distracted_streak: float = 0.0
        self._last_notify_at: float = -1e9
        self._notify_level: int = 0

        # --- 行为切换 ---
        self._last_behavior: Behavior = Behavior.UNKNOWN
        self._behavior_since: float = 0.0

        # --- 离席 ---
        self._away_since: float | None = None

        # --- 采样节流 ---
        self._last_sample_t: float = -1e9

        # --- 事件缓冲（UI 最近事件流）---
        self.recent_events: list[SessionEvent] = []
        self.max_recent = 100

        # --- 统计 ---
        self._last_state: FocusState = FocusState.UNCERTAIN
        self.distraction_count = 0

    # ------------------------------------------------------------------

    def start(self, now: float | None = None) -> list[SessionEvent]:
        now = time.perf_counter() if now is None else now

        self.timer.reset(now)
        self.active = True
        self.started_wall = time.time()

        if self.db is not None and self.storage_enabled:
            self.session_id = self.db.start_session(self.started_wall)

        event = SessionEvent(
            kind="session_start",
            message="开始学习",
            ts=self.started_wall,
        )
        self._push(event)

        return [event]

    def end(self, now: float | None = None) -> tuple[FocusStats, list[SessionEvent]]:
        now = time.perf_counter() if now is None else now
        stats = self.snapshot_stats(now)

        self.active = False

        if self.db is not None and self.storage_enabled:
            self.db.finish_session(stats)

        letter, comment = grade(stats.focus_score)
        event = SessionEvent(
            kind="session_end",
            message=f"本次学习结束 · 专注分 {stats.focus_score:.0f}（{letter}）· {comment}",
            ts=time.time(),
        )
        self._push(event)

        return stats, [event]

    # ------------------------------------------------------------------

    def update(
        self,
        state_snapshot,
        estimate=None,
        now: float | None = None,
    ) -> list[SessionEvent]:
        """每个循环调用一次。返回本次产生的事件列表。"""
        now = time.perf_counter() if now is None else now
        events: list[SessionEvent] = []

        state = getattr(state_snapshot, "state", FocusState.UNCERTAIN)
        behavior = getattr(state_snapshot, "behavior", Behavior.UNKNOWN)
        confidence = float(getattr(state_snapshot, "confidence", 0.0))

        # ---------- 1) 计时 ----------
        timer_events = self.timer.tick(state, behavior, now)

        for te in timer_events:
            event = SessionEvent(
                kind=te.kind,
                message=te.message,
                level=3 if te.kind == "milestone" else 2,
                minutes=te.minutes,
                ts=time.time(),
            )
            events.append(event)
            self._push(event)

            if self.db is not None and self.storage_enabled:
                if te.kind == "milestone":
                    self.db.add_milestone(te.minutes)
                self.db.add_event(
                    kind=te.kind,
                    state=state.value,
                    detail=te.message,
                    duration=te.value,
                )

        # ---------- 2) 分心升级 ----------
        if state is FocusState.DISTRACTED:
            if self._distracted_since is None:
                self._distracted_since = now
                self.distraction_count += 1
                self._notify_level = 0

                event = SessionEvent(
                    kind="distraction",
                    message=f"开始分心：{behavior_label(behavior)}",
                    level=1,
                    behavior=behavior.value,
                    state=state.value,
                    ts=time.time(),
                )
                events.append(event)
                self._push(event)

                if self.db is not None and self.storage_enabled:
                    self.db.add_event(
                        kind="distraction",
                        behavior=behavior.value,
                        state=state.value,
                        detail="进入分心状态",
                    )

            self._distracted_streak = now - self._distracted_since

            # 分级提醒
            if self.notify_enabled:
                level = self._escalation_level(self._distracted_streak)

                if (
                    level > self._notify_level
                    and (now - self._last_notify_at) >= self.cooldown_seconds
                ):
                    self._notify_level = level
                    self._last_notify_at = now

                    # ⚠️ 这里必须用 `distraction_reminder`，不能复用
                    # `distraction` —— 否则 UI 事件流无法区分"刚进入分心"
                    # 与"分心 30 秒后的强提醒"，图标与颜色都会错。
                    event = SessionEvent(
                        kind="distraction_reminder",
                        message=self._reminder_text(behavior, self._distracted_streak),
                        level=level,
                        behavior=behavior.value,
                        state=state.value,
                        duration=self._distracted_streak,
                        ts=time.time(),
                    )
                    events.append(event)
                    self._push(event)

                    if self.db is not None and self.storage_enabled:
                        self.db.add_event(
                            kind="distraction_reminder",
                            behavior=behavior.value,
                            state=state.value,
                            duration=self._distracted_streak,
                            detail=f"level={level}",
                        )
        else:
            # 恢复
            if self._distracted_since is not None:
                streak = now - self._distracted_since

                # 只记"真正分心"的（超过 5 秒），避免眨眼级别的噪声
                if streak >= 5.0:
                    event = SessionEvent(
                        kind="recovery",
                        message=f"已回到学习（分心 {streak:.0f} 秒）",
                        level=1,
                        duration=streak,
                        ts=time.time(),
                    )
                    events.append(event)
                    self._push(event)

                self._distracted_since = None
                self._distracted_streak = 0.0
                self._notify_level = 0

        # ---------- 3) 离席 ----------
        if state is FocusState.AWAY:
            if self._away_since is None:
                self._away_since = now
                event = SessionEvent(
                    kind="away_start",
                    message="离开座位",
                    level=1,
                    ts=time.time(),
                )
                events.append(event)
                self._push(event)

                if self.db is not None and self.storage_enabled:
                    self.db.add_event(kind="away_start", state=state.value)
        else:
            if self._away_since is not None:
                duration = now - self._away_since

                if duration >= 5.0:
                    event = SessionEvent(
                        kind="away_end",
                        message=f"回到座位（离开 {duration:.0f} 秒）",
                        level=1,
                        duration=duration,
                        ts=time.time(),
                    )
                    events.append(event)
                    self._push(event)

                    if self.db is not None and self.storage_enabled:
                        self.db.add_event(
                            kind="away_end",
                            state=state.value,
                            duration=duration,
                        )

                self._away_since = None

        # ---------- 4) 行为切换记录 ----------
        if behavior is not Behavior.UNKNOWN and behavior is not self._last_behavior:
            previous = self._last_behavior

            if previous is not Behavior.UNKNOWN and self._behavior_since:
                previous_duration = now - self._behavior_since

                if previous_duration >= 3.0:
                    event = SessionEvent(
                        kind="behavior_shift",
                        message=f"{behavior_label(previous)} → {behavior_label(behavior)}",
                        level=0,
                        behavior=behavior.value,
                        state=state.value,
                        duration=previous_duration,
                        ts=time.time(),
                    )
                    events.append(event)
                    self._push(event)

                    # 行为切换是分析页最有价值的数据，一定落库
                    if self.db is not None and self.storage_enabled:
                        self.db.add_event(
                            kind="behavior_shift",
                            behavior=behavior.value,
                            state=state.value,
                            duration=previous_duration,
                            detail=event.message,
                        )

            self._last_behavior = behavior
            self._behavior_since = now

        # ---------- 5) 行为采样 ----------
        if (now - self._last_sample_t) >= self.sample_interval:
            self._last_sample_t = now

            if self.db is not None and self.storage_enabled:
                scores = {}
                if estimate is not None:
                    scores = {
                        getattr(b, "value", str(b)): s
                        for b, s in (getattr(estimate, "scores", {}) or {}).items()
                    }

                self.db.add_sample(
                    behavior=behavior.value if isinstance(behavior, Behavior) else str(behavior),
                    state=state.value,
                    confidence=confidence,
                    scores=scores,
                )

        self._last_state = state
        return events

    # ------------------------------------------------------------------

    def _escalation_level(self, streak: float) -> int:
        if streak >= self.alert_seconds:
            return 3
        if streak >= self.soft_seconds:
            return 2
        if streak >= self.warn_seconds:
            return 1
        return 0

    @staticmethod
    def _reminder_text(behavior: Behavior, streak: float) -> str:
        base = {
            Behavior.PHONE_USE: "手机放下吧",
            Behavior.FIDGETING: "手上有小动作，收一收",
            Behavior.IDLE: "发呆了？回到书上",
        }.get(behavior, "注意力跑掉了")

        return f"{base}（已持续 {streak:.0f} 秒）"

    def _push(self, event: SessionEvent) -> None:
        self.recent_events.append(event)

        if len(self.recent_events) > self.max_recent:
            self.recent_events = self.recent_events[-self.max_recent:]

    # ------------------------------------------------------------------

    def snapshot_stats(self, now: float | None = None) -> FocusStats:
        now = time.perf_counter() if now is None else now

        # 把当前状态的这一段也算进去，报表才不会"少一段"
        counts = self.timer.counts()

        return build_stats(
            counts=counts,
            transitions=self._transitions(),
            behavior_seconds=self.timer.behavior_seconds,
            now=now,
            session_start=self.timer.started_at,
            longest_streak=self.timer.longest_streak,
            current_streak=self.timer.current_streak,
        )

    def _transitions(self) -> list:
        machine = getattr(self, "_machine_ref", None)
        return getattr(machine, "transitions", []) if machine is not None else []

    def bind_state_machine(self, machine) -> None:
        """让统计能拿到状态切换次数（切换惩罚要用）。"""
        self._machine_ref = machine

    # ------------------------------------------------------------------

    @property
    def focused_seconds(self) -> float:
        return self.timer.focused_seconds

    @property
    def total_seconds(self) -> float:
        return self.timer.total_seconds

    @property
    def focus_ratio(self) -> float:
        return self.timer.focus_ratio

    @property
    def current_distraction(self) -> float:
        """当前已持续分心多久（未分心为 0）。"""
        return self._distracted_streak if self._distracted_since is not None else 0.0

    def live_score(self) -> float:
        """即时专注分（用于 HUD 实时显示）。"""
        return self.snapshot_stats().focus_score
