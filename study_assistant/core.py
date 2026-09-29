"""行为核心：从"视觉观测"到"专注状态 + 事件"的完整逻辑链。

这个类存在的唯一理由是**让逻辑只有一份实现**：`app.py` 与验收测试都
通过它推进状态，所以测试验证的就是线上跑的那段代码，而不是一份长得
像的复制品。

一个 `update()` 的内部顺序（顺序本身就是设计）：

    手部轨迹采样 → 物体在场时长 → 手-物交互
        → 逐行为打分（BehaviorFusionEngine）
        → 滑窗 + 连续时长（TemporalEngine）
        → 滞后状态机（FocusStateMachine）
        → 会话统计 / 分级提醒 / 落库（SessionManager）

注意 `hands_fresh` 这个参数：它决定"这一帧要不要往轨迹里采一个点"。
缓存帧（分频调度跳过推理的帧）如果也采样，同一个位置会被重复写入，
平均速度会被稀释到接近 0，写字判定直接失效。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .behavior.behavior_fusion import BehaviorFusionEngine
from .behavior.hand_object_engine import HandObjectEngine
from .behavior.state_machine import FocusStateMachine
from .behavior.states import Behavior, FocusState
from .behavior.temporal_engine import TemporalEngine
from .features.hand_features import HandTracker
from .features.object_features import ObjectPresenceTracker
from .notifications.notification_manager import NotificationManager
from .session.session_manager import SessionManager
from .storage.database import Database


@dataclass
class CoreUpdate:
    """一次 update() 的全部产出。"""

    now: float = 0.0
    interactions: object = None
    verdict: object = None
    estimate: object = None
    snapshot: object = None
    events: list = field(default_factory=list)

    hands_absent_for: float = 0.0
    person_absent_for: float = 0.0
    person_present: bool = False  # 人是否被**真实检出**且在新鲜窗口内
    fused: bool = False          # 本次是否真的跑了融合（分频可能跳过）

    @property
    def state(self) -> FocusState:
        return self.snapshot.state if self.snapshot else FocusState.UNCERTAIN

    @property
    def behavior(self) -> Behavior:
        return self.snapshot.behavior if self.snapshot else Behavior.UNKNOWN


class BehaviorCore:
    """状态容器的聚合根。"""

    def __init__(
        self,
        config,
        calibration,
        database: Database | None = None,
        warmup_seconds: float = 5.0,
        notifications: NotificationManager | None = None,
    ):
        self.config = config
        self.calibration = calibration
        self.db = database
        self.warmup_seconds = float(warmup_seconds)

        # 人"确认在场"的新鲜窗口 —— 用于否决离席判定
        self.person_present_max_age = float(
            config.behavior.get("person_present_max_age", 3.0)
        )

        # --- 特征 ---
        self.hand_tracker = HandTracker()
        self.presence = ObjectPresenceTracker()

        # --- 行为 ---
        self.engine = HandObjectEngine(config, calibration)
        self.fusion = BehaviorFusionEngine(config)
        self.temporal = TemporalEngine(config)
        self.machine = FocusStateMachine(config)

        # --- 会话 ---
        self.session = SessionManager(config, database=database)
        self.session.bind_state_machine(self.machine)

        # --- 通知 ---
        self.notifications = notifications or NotificationManager(config)

        self.hand_detector_available = True
        self.start_t = 0.0
        self.last = CoreUpdate()

    # ------------------------------------------------------------------

    def start(self, now: float | None = None) -> list:
        now = time.perf_counter() if now is None else now
        self.start_t = now

        events = self.session.start(now)
        self._dispatch(events)
        return events

    # ------------------------------------------------------------------

    def update(
        self,
        hands: list,
        tracked_objects: list,
        now: float,
        frame_width: int = 640,
        frame_height: int = 480,
        camera_ok: bool = True,
        hands_fresh: bool = True,
        objects_fresh: bool = True,
    ) -> CoreUpdate:
        """推进一次。hands / tracked_objects 可以为空列表。"""
        result = CoreUpdate(now=now)

        hands = hands or []
        tracked_objects = tracked_objects or []

        # ---------- 1) 手部轨迹采样 ----------
        # 只有真正跑了推理的帧才采样，否则会把速度稀释成 0
        if hands_fresh:
            self.hand_tracker.update(hands, now)

        # ---------- 2) 物体在场时长 ----------
        self.presence.update(tracked_objects, now)

        # ---------- 3) 手-物交互 ----------
        interactions = self.engine.update(
            hands, tracked_objects, now,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        result.interactions = interactions

        # ---------- 4) 预热期的离席屏蔽 ----------
        elapsed = now - self.start_t if self.start_t else 0.0

        if elapsed < self.warmup_seconds:
            hands_absent = 0.0
            person_absent = 0.0
            person_present = False
        else:
            hands_absent = self.hand_tracker.absent_for(now)

            person_state = self.presence.get("person")
            person_absent = (
                (now - person_state.last_visible)
                if person_state.last_visible else 0.0
            )

            # 人到底在不在画面里？
            # `last_real_seen == 0` 表示"这个场景里根本没见过人"，这与
            # "人此刻就在这里"是两回事：前者不能否决离席，否则一旦某个
            # 场景从没检出过人，就再也退不出离席路径了。所以这里必须用
            # 真实检出时刻，不能用 present（present 含滑行/容忍窗口）。
            person_present = bool(
                person_state.last_real_seen
                and (now - person_state.last_real_seen) <= self.person_present_max_age
            )

            # 手部检测器不可用时，不能拿"没检测到手"当离席依据
            if not self.hand_detector_available:
                hands_absent = 0.0

        result.hands_absent_for = hands_absent
        result.person_absent_for = person_absent
        result.person_present = person_present

        # ---------- 5) 融合分频 ----------
        if not self.temporal.should_fuse(now):
            result.snapshot = self.machine.snapshot(now)
            result.estimate = self.temporal.last_estimate
            self.last = result
            return result

        result.fused = True

        # ---------- 6) 逐行为打分 ----------
        verdict = self.fusion.update(
            hands=hands,
            motions=self.hand_tracker.all_motions(now),
            interactions=interactions,
            presence=self.presence,
            now=now,
            hands_absent_for=hands_absent,
            person_absent_for=person_absent,
            person_present=person_present,
            camera_ok=camera_ok,
        )
        result.verdict = verdict

        # ---------- 7) 滑窗 + 连续时长 ----------
        estimate = self.temporal.update(verdict, now)
        result.estimate = estimate

        # ---------- 8) 滞后状态机 ----------
        snapshot = self.machine.update(
            estimate,
            now=now,
            hands_absent_for=hands_absent,
            person_absent_for=person_absent,
            person_present=person_present,
            camera_ok=camera_ok,
        )
        result.snapshot = snapshot

        # ---------- 9) 会话统计 / 提醒 ----------
        events = self.session.update(snapshot, estimate, now)
        result.events = events
        self._dispatch(events)

        self.last = result
        return result

    # ------------------------------------------------------------------

    def _dispatch(self, events) -> None:
        """把事件交给通知管理器（UI 由调用方再消费一次）。"""
        for event in events:
            kind = event.kind

            if kind in ("distraction", "distraction_reminder") and event.level >= 2:
                self.notifications.distract(event.message, event.level, kind=kind)

            elif kind == "milestone":
                self.notifications.celebrate(event.message, event.minutes)

            elif kind == "break_due":
                self.notifications.suggest_break(event.message)

    # ------------------------------------------------------------------

    def end(self, now: float | None = None):
        now = time.perf_counter() if now is None else now

        stats, events = self.session.end(now)
        self._dispatch(events)

        return stats, events

    def snapshot_stats(self, now: float | None = None):
        return self.session.snapshot_stats(now)

    def set_paused(self, paused: bool, now: float | None = None) -> None:
        self.machine.set_paused(paused, now)

    def reset(self, now: float | None = None) -> list:
        now = time.perf_counter() if now is None else now

        self.machine.reset()
        self.engine.reset()
        self.presence.clear()
        self.hand_tracker.clear()
        self.temporal.reset()

        events = self.session.start(now)
        self._dispatch(events)
        return events

    @property
    def current_distraction(self) -> float:
        return self.session.current_distraction

    def live_score(self) -> float:
        return self.session.live_score()
