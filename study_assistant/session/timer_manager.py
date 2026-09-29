"""计时与里程碑。

负责三件事：
  1. 累计各状态的秒数（有效专注时长、分心时长、离席时长）；
  2. 里程碑（每 30/60/90/120 分钟有效专注触发一次庆祝）；
  3. 连续用眼提醒（连续专注 45 分钟 → 建议休息，并进入休息计时）。

注意"有效专注时间"是**累计**的，而不是墙上时间 —— 离席半小时不该算
成学习了半小时。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..behavior.states import FocusState


@dataclass
class TimerEvent:
    """计时器产生的提醒事件。"""

    kind: str            # milestone | break_due | break_over | focus_streak
    message: str = ""
    minutes: int = 0
    value: float = 0.0


class TimerManager:
    """会话计时器。"""

    def __init__(self, config):
        session_cfg = config.data.get("session", {})

        self.milestones = sorted(
            int(m) for m in config.notifications.get("milestones_minutes", [30, 60, 90, 120])
        )

        self.break_after_minutes = float(session_cfg.get("break_after_minutes", 45))
        self.break_minutes = float(session_cfg.get("break_minutes", 5))

        # 连续专注后，允许多长的"小停顿"不打断连续段
        self.streak_grace = float(session_cfg.get("streak_grace_seconds", 20.0))

        self.reset()

    # ------------------------------------------------------------------

    def reset(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now

        self.started_at = now
        self._last_t = now

        self.total_seconds = 0.0
        self.focused_seconds = 0.0
        self.distracted_seconds = 0.0
        self.away_seconds = 0.0
        self.uncertain_seconds = 0.0
        self.paused_seconds = 0.0

        self.behavior_seconds: dict[str, float] = {}

        # 连续专注
        self.current_streak = 0.0
        self.longest_streak = 0.0
        self._streak_gap_start: float | None = None

        # 里程碑 / 休息
        self._fired_milestones: set[int] = set()
        self._last_break_at_focus = 0.0
        self.in_break = False
        self.break_started = 0.0
        self.breaks_taken = 0

    # ------------------------------------------------------------------

    def tick(
        self,
        state: FocusState,
        behavior=None,
        now: float | None = None,
    ) -> list[TimerEvent]:
        """推进计时器，返回本次产生的事件。"""
        now = time.perf_counter() if now is None else now

        dt = max(0.0, now - self._last_t)
        self._last_t = now

        # 单次 tick 超过 5 秒视为程序被挂起，不把这段时间计入
        if dt > 5.0:
            dt = 0.0

        events: list[TimerEvent] = []

        # ---------- 1) 累计 ----------
        if state is FocusState.PAUSED:
            self.paused_seconds += dt
            return events          # 暂停期间不推进任何计时

        self.total_seconds += dt

        if state is FocusState.FOCUSED:
            self.focused_seconds += dt
        elif state is FocusState.DISTRACTED:
            self.distracted_seconds += dt
        elif state is FocusState.AWAY:
            self.away_seconds += dt
        else:
            self.uncertain_seconds += dt

        if behavior is not None:
            key = getattr(behavior, "value", str(behavior))
            self.behavior_seconds[key] = self.behavior_seconds.get(key, 0.0) + dt

        # ---------- 2) 连续专注段 ----------
        if state is FocusState.FOCUSED:
            if self._streak_gap_start is not None:
                # 从停顿中恢复
                self._streak_gap_start = None

            self.current_streak += dt
            self.longest_streak = max(self.longest_streak, self.current_streak)

        else:
            # 非专注：给一个 grace，短停顿不断段
            if self._streak_gap_start is None:
                self._streak_gap_start = now
            elif (now - self._streak_gap_start) > self.streak_grace:
                if self.current_streak > 0:
                    self.longest_streak = max(self.longest_streak, self.current_streak)
                self.current_streak = 0.0

        # ---------- 3) 里程碑（按有效专注累计时长）----------
        focused_minutes = self.focused_seconds / 60.0

        for milestone in self.milestones:
            if milestone not in self._fired_milestones and focused_minutes >= milestone:
                self._fired_milestones.add(milestone)

                events.append(
                    TimerEvent(
                        kind="milestone",
                        message=f"已专注学习 {milestone} 分钟",
                        minutes=milestone,
                        value=self.focused_seconds,
                    )
                )

        # ---------- 4) 休息提醒 ----------
        since_break = self.focused_seconds - self._last_break_at_focus

        if not self.in_break and since_break >= self.break_after_minutes * 60.0:
            self.in_break = True
            self.break_started = now
            self.breaks_taken += 1

            events.append(
                TimerEvent(
                    kind="break_due",
                    message=f"连续专注 {self.break_after_minutes:.0f} 分钟，休息一下吧",
                    minutes=int(self.break_after_minutes),
                    value=since_break,
                )
            )

        elif self.in_break:
            if (now - self.break_started) >= self.break_minutes * 60.0:
                self.in_break = False
                self._last_break_at_focus = self.focused_seconds

                events.append(
                    TimerEvent(
                        kind="break_over",
                        message="休息结束，继续加油",
                        minutes=int(self.break_minutes),
                    )
                )

        return events

    # ------------------------------------------------------------------

    def take_break_now(self, now: float | None = None) -> None:
        """用户主动休息：重置连续专注计时。"""
        now = time.perf_counter() if now is None else now
        self._last_break_at_focus = self.focused_seconds
        self.current_streak = 0.0
        self.in_break = True
        self.break_started = now
        self.breaks_taken += 1

    def end_break(self) -> None:
        self.in_break = False
        self._last_break_at_focus = self.focused_seconds

    # ------------------------------------------------------------------

    @property
    def wall_seconds(self) -> float:
        """墙上时间（含暂停）。"""
        return sum((
            self.total_seconds, self.paused_seconds,
        ))

    @property
    def focus_ratio(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return self.focused_seconds / self.total_seconds

    def counts(self) -> dict:
        return {
            FocusState.FOCUSED: self.focused_seconds,
            FocusState.DISTRACTED: self.distracted_seconds,
            FocusState.AWAY: self.away_seconds,
            FocusState.UNCERTAIN: self.uncertain_seconds,
            FocusState.PAUSED: self.paused_seconds,
        }

    def next_milestone(self) -> int | None:
        for milestone in self.milestones:
            if milestone not in self._fired_milestones:
                return milestone
        return None

    def milestone_progress(self) -> float:
        """距离下一个里程碑的进度 0..1。"""
        nxt = self.next_milestone()
        if nxt is None:
            return 1.0

        focused_minutes = self.focused_seconds / 60.0
        previous = 0.0

        for milestone in self.milestones:
            if milestone >= nxt:
                break
            previous = float(milestone)

        span = max(1.0, nxt - previous)
        return max(0.0, min(1.0, (focused_minutes - previous) / span))
