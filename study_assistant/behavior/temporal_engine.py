"""时序融合：把逐帧打分变成"稳定的行为判断 + 持续时长"。

为什么必须有这一层：
  * 单帧判断必然抖动。MediaPipe 每 3 帧丢一次手、YOLO 每 10 帧丢一次
    书，如果直接拿瞬时 top 当结论，UI 会疯闪、事件记录会碎成渣；
  * 行为判定本身大多写着"持续 N 秒才算"。这个"N 秒"必须由本模块
    统一提供，否则每个行为各写一套计数器，语义会漂移。

实现方式：
  * 每个行为的原始分进入一条时间序列（默认 3 秒窗口）；
  * `sustained` = 该行为**连续**作为 top1 的秒数，允许 `tolerance`
    秒的短暂中断（默认 0.8s，用来吸收漏检）；
  * 稳定性：窗口均值与均值方差共同决定 `stability`，供状态机判断
    "这次切换是不是可信"。
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from .states import Behavior


@dataclass
class BehaviorEstimate:
    """时序融合后的行为结论。"""

    behavior: Behavior = Behavior.UNKNOWN
    confidence: float = 0.0
    raw_score: float = 0.0
    sustained: float = 0.0                 # 连续保持这个行为的秒数
    scores: dict = field(default_factory=dict)      # 窗口均值
    stability: float = 0.0                 # 0..1，越高越稳定
    runner_up: Behavior = Behavior.UNKNOWN
    margin: float = 0.0                    # top1 与 top2 的分差

    # 尚未攒够持续时间、正在逼近的行为（供 UI 展示"候选"）
    pending: Behavior = Behavior.UNKNOWN
    pending_progress: float = 0.0          # 0..1

    # 逐行为的证据文字（{Behavior: [str, ...]}），直接透传融合层的输出。
    # 必须在这里透出来：仪表盘的"实时证据"面板读的是 estimate.evidence，
    # 而 BehaviorEstimate 原本没有这个字段 —— getattr 拿到 None 直接返回，
    # 面板永远显示"—"，等于这块 UI 从来没接过线。
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "behavior": self.behavior.value,
            "confidence": round(self.confidence, 3),
            "sustained": round(self.sustained, 2),
            "stability": round(self.stability, 3),
            "runner_up": self.runner_up.value,
            "margin": round(self.margin, 3),
        }


class TemporalEngine:
    """滑动窗口 + 连续时长统计。"""

    def __init__(self, config):
        b = config.behavior

        self.window = float(b.get("window_seconds", 3.0))
        self.fusion_hz = float(b.get("fusion_hz", 10))
        self.fusion_interval = 1.0 / max(self.fusion_hz, 1e-6)
        self.min_raw = float(b.get("min_confidence", 0.45))

        # 短暂中断容忍（秒）——漏检兜底
        self.gap_tolerance = float(b.get("evidence_gap_tolerance", 0.8))

        # 每个行为的原始分历史：(t, score)
        self._history: dict[Behavior, deque] = {
            b: deque() for b in Behavior
        }

        # 连续 top1 追踪
        self._run_behavior: Behavior = Behavior.UNKNOWN
        self._run_start: float = 0.0
        self._run_last_seen: float = 0.0

        self._last_fusion_t = -1e9
        self._last_estimate = BehaviorEstimate()

        # 行为 → 最短持续时间（秒），由配置给出
        self.min_durations: dict[Behavior, float] = {
            Behavior.PAPER_STUDY: float(b.get("paper_study", {}).get("min_duration", 2.0)),
            Behavior.COMPUTER_STUDY: float(b.get("computer", {}).get("min_duration", 2.0)),
            Behavior.FIDGETING: float(b.get("fidgeting", {}).get("min_duration", 4.0)),
            Behavior.PHONE_USE: float(b.get("phone", {}).get("confirm_seconds", 3.0)),
        }

    # ------------------------------------------------------------------

    def should_fuse(self, now: float) -> bool:
        """是否需要跑一次融合（按 fusion_hz 降频，节省 UI 线程算力）。"""
        return (now - self._last_fusion_t) >= self.fusion_interval

    def update(self, verdict, now: float | None = None) -> BehaviorEstimate:
        """把一帧打分并入时序，输出稳定结论。

        调用前应先用 `should_fuse()` 控制频率，但不强制 —— 每帧都调
        也是安全的，只是没必要。
        """
        now = time.perf_counter() if now is None else now
        self._last_fusion_t = now

        # ---------- 1) 原始分入队 ----------
        raw = getattr(verdict, "raw_scores", None) or verdict.scores

        for behavior in Behavior:
            score = float(raw.get(behavior, 0.0))
            series = self._history[behavior]
            series.append((now, score))

            cutoff = now - self.window
            while series and series[0][0] < cutoff:
                series.popleft()

        # ---------- 2) 窗口均值 ----------
        means = {}
        for behavior in Behavior:
            series = self._history[behavior]
            if not series:
                means[behavior] = 0.0
                continue
            means[behavior] = sum(s for _, s in series) / len(series)

        # ---------- 3) 本轮即时 top（用当前帧的原始分判定）----------
        # 用「窗口均值」定 top 会让切换变得迟钝，用「当前帧」定 top 又会抖。
        # 折中：以窗口均值为准，但要求当前帧分数不为零。
        candidates = [
            (b, means[b])
            for b in Behavior
            if means[b] > 0.0 and float(raw.get(b, 0.0)) > 0.0
        ]

        if not candidates:
            top_behavior, top_score = Behavior.UNKNOWN, 0.0
            runner_up, second = Behavior.UNKNOWN, 0.0
        else:
            candidates.sort(key=lambda kv: kv[1], reverse=True)
            top_behavior, top_score = candidates[0]
            runner_up, second = candidates[1] if len(candidates) > 1 else (Behavior.UNKNOWN, 0.0)

        # ---------- 4) 连续时长 ----------
        if top_behavior is self._run_behavior:
            self._run_last_seen = now
        else:
            gap = now - self._run_last_seen

            # 换 top：若距离上次见到旧 top 没超过容忍值，视为同一段连续的
            # "主导行为"（防止 A/B/A 抖动把时长清零）
            if (self._run_behavior is not Behavior.UNKNOWN
                    and top_behavior is not Behavior.UNKNOWN
                    and gap <= self.gap_tolerance
                    and means.get(self._run_behavior, 0.0) > means.get(top_behavior, 0.0) * 0.75):
                # 保持原来的连续段，只更新 last_seen
                self._run_last_seen = now
            else:
                self._run_behavior = top_behavior
                self._run_start = now
                self._run_last_seen = now

        # 若中间有较长的空档，连续段从"重建"算起
        if now - self._run_last_seen > self.gap_tolerance:
            self._run_start = now

        sustained = max(0.0, now - self._run_start)

        # ---------- 5) 稳定性 ----------
        stability = self._stability(top_behavior, means, raw)

        # ---------- 6) 是否够格对外宣布 ----------
        behavior = top_behavior
        min_dur = self.min_durations.get(behavior, 0.0)

        if top_score < 0.15:
            behavior = Behavior.UNKNOWN
        elif sustained < min_dur:
            # 还没攒够持续时间 —— 不宣布，但把 sustained 透出去让上层做
            # "候选"展示（例如手机候选状态）
            behavior = Behavior.UNKNOWN

        # 手机是特例：即使没到 confirm_seconds 也要让上层知道"有嫌疑"
        estimate = BehaviorEstimate(
            behavior=behavior,
            confidence=float(top_score),
            raw_score=float(raw.get(top_behavior, 0.0)),
            sustained=sustained,
            scores=means,
            stability=stability,
            runner_up=runner_up,
            margin=max(0.0, top_score - second),
            evidence=getattr(verdict, "evidence", None) or {},
        )

        # 记录"正在逼近"的行为（未达 min_duration 的那个）
        estimate.pending = (
            top_behavior
            if (behavior is Behavior.UNKNOWN and top_behavior is not Behavior.UNKNOWN)
            else Behavior.UNKNOWN
        )
        estimate.pending_progress = (
            min(1.0, sustained / min_dur) if min_dur > 0 else 0.0
        )

        self._last_estimate = estimate
        return estimate

    # ------------------------------------------------------------------

    def _stability(self, behavior: Behavior, means: dict, raw: dict) -> float:
        """稳定性 = 1 - 归一化标准差（窗口内该行为的分数波动）。"""
        series = self._history.get(behavior)
        if not series or len(series) < 3:
            return 0.0

        values = [s for _, s in series]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)

        # 分数在 0..1 之间，std 达到 0.35 就算极不稳定
        return max(0.0, min(1.0, 1.0 - std / 0.35))

    # ------------------------------------------------------------------

    def sustained(self, behavior: Behavior) -> float:
        return self._last_estimate.sustained if behavior is self._last_estimate.behavior else 0.0

    def time_since(self, behavior: Behavior, now: float | None = None) -> float:
        """该行为有多久没作为 top1 出现了（用于"离开分心状态多久了"）。"""
        now = time.perf_counter() if now is None else now
        series = self._history.get(behavior)

        if not series:
            return 1e9

        # 找最近一次分数超过阈值的时刻
        last_strong = None
        for t, s in reversed(series):
            if s >= 0.25:
                last_strong = t
                break

        return (now - last_strong) if last_strong else 1e9

    @property
    def last_estimate(self) -> BehaviorEstimate:
        return self._last_estimate

    def reset(self) -> None:
        for series in self._history.values():
            series.clear()
        self._run_behavior = Behavior.UNKNOWN
        self._run_start = 0.0
        self._run_last_seen = 0.0
        self._last_estimate = BehaviorEstimate()
