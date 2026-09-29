"""提醒分发：通知 / 声音 / 表情。

刻意做成**回调驱动**，不直接依赖 PySide6 —— 这样：
  * 无头环境下能跑（只打印），验收测试不需要图形界面；
  * 单元测试里能直接断言"某个等级该不该提醒"。

调用方（app.py）把 Qt 的托盘气泡、提示音、眼睛表情接进来即可。

分级语义（与 session_manager 一致）：
  level 1 → 只改表情（不出声、不弹窗）
  level 2 → 气泡通知
  level 3 → 气泡 + 声音
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field


@dataclass
class NotificationRecord:
    """一次提醒的历史记录。"""

    ts: float
    level: int
    title: str
    message: str
    kind: str = ""
    sound: bool = False

    @property
    def local_time(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.ts))

    def as_dict(self) -> dict:
        return {
            "ts": self.ts,
            "time": self.local_time,
            "level": self.level,
            "title": self.title,
            "message": self.message,
            "kind": self.kind,
        }


class NotificationManager:
    """带冷却与分级的提醒分发器。"""

    def __init__(
        self,
        config,
        on_notify=None,
        on_sound=None,
        on_mood=None,
        clock=None,
        enable_system_sound: bool = True,
    ):
        n = config.notifications

        self.enabled = bool(n.get("enabled", True))
        self.cooldown = float(n.get("cooldown_seconds", 60))

        # 回调：on_notify(title, message, level) / on_sound(level) /
        #       on_mood(mood_name)
        self.on_notify = on_notify
        self.on_sound = on_sound
        self.on_mood = on_mood

        self._clock = clock or time.perf_counter

        self._last_at: dict[str, float] = {}
        self.history: list[NotificationRecord] = []
        self.max_history = 200

        # 是否允许直接调用系统蜂鸣。测试 / headless 场景设为 False。
        self.enable_system_sound = bool(enable_system_sound)

        self._native_sound = (
            self.enable_system_sound
            and platform.system() in ("Windows", "Linux", "Darwin")
        )

    # ------------------------------------------------------------------

    def notify(
        self,
        message: str,
        level: int = 1,
        title: str = "学习助手",
        kind: str = "",
        mood: str | None = None,
        force: bool = False,
    ) -> NotificationRecord | None:
        """发一条提醒。返回记录；被冷却拦下或静默等级会返回 None。"""
        if not self.enabled and not force:
            return None

        now = self._clock()

        # --- 冷却（按 kind+level 分组，避免不同类型互相压制）---
        bucket = f"{kind}:{level}"
        last = self._last_at.get(bucket, -1e9)

        if not force and (now - last) < self.cooldown:
            return None

        self._last_at[bucket] = now

        record = NotificationRecord(
            ts=time.time(),
            level=int(level),
            title=title,
            message=message,
            kind=kind,
            sound=(level >= 3),
        )

        self.history.append(record)

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # --- 表情（任何等级都可以改）---
        if mood and self.on_mood is not None:
            self._safe(self.on_mood, mood)

        # --- 气泡（level >= 2）---
        if level >= 2 and self.on_notify is not None:
            self._safe(self.on_notify, title, message, level)

        # --- 声音（level >= 3）---
        if record.sound:
            self.sound(level)

        return record

    # ------------------------------------------------------------------

    def sound(self, level: int = 3) -> None:
        """播提示音：优先用注入的回调，否则尝试系统蜂鸣。"""
        if self.on_sound is not None:
            self._safe(self.on_sound, level)
            return

        if not self._native_sound:
            return

        try:
            if platform.system() == "Windows":
                import winsound

                # 轻微的三声短提示，不要太刺耳
                for freq in (880, 880, 1175):
                    winsound.Beep(freq, 110)
            else:
                # Linux/macOS：终端响铃，实在没有再退化成静默
                print("\a", end="", flush=True)
        except Exception:
            pass

    def set_mood(self, mood: str) -> None:
        if self.on_mood is not None:
            self._safe(self.on_mood, mood)

    # ------------------------------------------------------------------

    def celebrate(self, message: str, minutes: int = 0) -> NotificationRecord | None:
        """里程碑庆祝：绕过冷却（里程碑本身不会频繁触发）。"""
        return self.notify(
            message,
            level=3,
            title="🎉 里程碑",
            kind="milestone",
            mood="celebrate",
            force=True,
        )

    def suggest_break(self, message: str) -> NotificationRecord | None:
        return self.notify(
            message,
            level=3,
            title="☕ 该休息了",
            kind="break",
            mood="sleepy",
            force=True,
        )

    def distract(self, message: str, level: int, kind: str = "distraction"):
        """分心提醒：由 session_manager 给出 level，这里负责表现。"""
        mood = {
            1: "worried",
            2: "worried",
            3: "annoyed",
        }.get(level, "worried")

        return self.notify(message, level=level, kind=kind, mood=mood)

    # ------------------------------------------------------------------

    def _safe(self, callback, *args) -> None:
        """回调异常不能让主循环挂掉（UI 回调尤其容易抛）。"""
        try:
            callback(*args)
        except Exception as exc:
            print(f"[Notification] 回调异常（已忽略）：{exc}")

    def recent(self, limit: int = 20) -> list[NotificationRecord]:
        return self.history[-limit:]

    def reset(self) -> None:
        self._last_at.clear()
        self.history.clear()
