"""手-物交互的几何计算与持续时长统计。

这是整套判断的地基：**"手在做什么" = "手在哪 + 手离什么近 + 近了多久"**。

不要用"手动了就分心"。要回答的是：
  * 手是不是长时间贴在书写区域（写字）
  * 手是不是握着手机（分心）
  * 手是不是在键盘上（电脑学习）

几何量：
  * distance      —— 手部锚点到物体框的**外部**最短距离（像素）
  * overlap_ratio —— 落在物体框内的手部关键点比例
  * containment   —— 手部 bbox 被物体框覆盖的面积比例
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# 用于测距的手部锚点：手腕 + 掌心 + 五个指尖
ANCHOR_INDICES = (0, 4, 8, 12, 16, 20)


@dataclass
class InteractionMeasure:
    """一次"手 vs 物体"的瞬时几何测量。"""

    hand_label: str
    object_label: str
    object_track_id: int = -1

    distance_px: float = 1e9
    distance_norm: float = 1e9
    overlap_ratio: float = 0.0       # 关键点落入物体框的比例
    containment: float = 0.0         # 手部 bbox 被物体框覆盖比例
    crossed: bool = False            # 手中心是否已在物体框内
    contact: bool = False            # 综合判定：算不算"接触/操作"

    # 画线用（手部锚点 → 物体中心，归一化坐标）
    hand_point: tuple = (0.5, 0.5)
    object_point: tuple = (0.5, 0.5)

    def as_dict(self) -> dict:
        return {
            "hand": self.hand_label,
            "object": self.object_label,
            "distance_px": round(self.distance_px, 1),
            "overlap": round(self.overlap_ratio, 3),
            "contact": self.contact,
        }


def _anchors(hand) -> list:
    """取手部锚点（归一化）。缺少 landmarks 时退回 wrist/palm/tips。"""
    pts = []

    landmarks = getattr(hand, "landmarks", None)

    if landmarks is not None and len(landmarks) >= 21:
        for idx in ANCHOR_INDICES:
            pts.append((float(landmarks[idx][0]), float(landmarks[idx][1])))

    if not pts:
        pts.append(tuple(hand.wrist))
        pts.append(tuple(hand.palm_center))
        pts.extend([tuple(t) for t in hand.finger_tips])

    return pts


def _all_points(hand) -> list:
    """全部 21 个关键点（用于 overlap 统计）。"""
    landmarks = getattr(hand, "landmarks", None)

    if landmarks is not None and len(landmarks) >= 21:
        return [(float(p[0]), float(p[1])) for p in landmarks]

    return _anchors(hand)


def measure(
    hand,
    obj,
    frame_width: int = 640,
    frame_height: int = 480,
    distance_px_threshold: float = 90.0,
    overlap_min: float = 0.05,
) -> InteractionMeasure:
    """计算一只手与一个物体之间的交互几何。

    距离用"框外距离"：手在框内时距离为 0，这样"握着手机"不会被
    判成距离很远。
    """
    result = InteractionMeasure(
        hand_label=getattr(hand, "label", "unknown"),
        object_label=getattr(obj, "label", "unknown"),
        object_track_id=getattr(obj, "track_id", -1),
        hand_point=tuple(hand.palm_center),
        object_point=tuple(obj.center),
    )

    x0, y0, x1, y1 = obj.bbox

    # --- 锚点最短外部距离（像素）---
    best_px = 1e9

    for px, py in _anchors(hand):
        # 归一化 -> 像素
        ax, ay = px * frame_width, py * frame_height

        # 到框的横向 / 纵向外部距离
        dx = max(x0 * frame_width - ax, 0.0, ax - x1 * frame_width)
        dy = max(y0 * frame_height - ay, 0.0, ay - y1 * frame_height)

        d = math.hypot(dx, dy)
        if d < best_px:
            best_px = d

    result.distance_px = best_px
    result.distance_norm = best_px / max(frame_width, 1)

    # --- 关键点落入比例 ---
    points = _all_points(hand)
    inside = sum(
        1 for px, py in points
        if (x0 <= px <= x1) and (y0 <= py <= y1)
    )
    result.overlap_ratio = inside / len(points) if points else 0.0

    # --- 手心是否已在框内 ---
    cx, cy = hand.palm_center
    result.crossed = (x0 <= cx <= x1) and (y0 <= cy <= y1)

    # --- 手部 bbox 被覆盖比例 ---
    hx0, hy0, hx1, hy1 = hand.bbox
    ix0, iy0 = max(hx0, x0), max(hy0, y0)
    ix1, iy1 = min(hx1, x1), min(hy1, y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    hand_area = max(1e-9, (hx1 - hx0) * (hy1 - hy0))
    result.containment = min(1.0, inter / hand_area)

    # --- 综合接触判定 ---
    result.contact = (
        best_px <= distance_px_threshold
        or result.overlap_ratio >= overlap_min
        or result.crossed
    )

    return result


@dataclass
class ContactWindow:
    """一段连续接触的统计。"""

    key: str                       # "left:phone"
    start: float = 0.0
    last: float = 0.0
    active: bool = False
    duration: float = 0.0
    peak_overlap: float = 0.0
    min_distance_px: float = 1e9
    samples: int = 0

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "duration": round(self.duration, 2),
            "active": self.active,
            "peak_overlap": round(self.peak_overlap, 3),
            "min_distance_px": round(self.min_distance_px, 1),
        }


class InteractionTracker:
    """跟踪每一组 (手, 物体类别) 的接触持续时长。

    关键点：接触要"断得干脆"。中间丢掉一两帧不该把 3 秒的持续接触
    清零，否则 PHONE_USE 永远攒不够 confirm_seconds。
    """

    def __init__(self, break_grace: float = 0.6):
        self.break_grace = float(break_grace)
        self._windows: dict[str, ContactWindow] = {}

    @staticmethod
    def key(hand_label: str, object_label: str) -> str:
        return f"{hand_label}:{object_label}"

    def update(
        self,
        measures: list,
        now: float,
    ) -> dict:
        """用本帧所有 (手, 物体) 测量更新接触窗口。

        返回 {key: ContactWindow}（只包含活跃或刚结束的窗口）。
        """
        touched: set[str] = set()

        for m in measures:
            if not m.contact:
                continue

            k = self.key(m.hand_label, m.object_label)
            window = self._windows.setdefault(k, ContactWindow(key=k))
            touched.add(k)

            if not window.active:
                window.start = now
                window.peak_overlap = 0.0
                window.min_distance_px = 1e9
                window.samples = 0

            window.active = True
            window.last = now
            window.duration = now - window.start
            window.peak_overlap = max(window.peak_overlap, m.overlap_ratio)
            window.min_distance_px = min(window.min_distance_px, m.distance_px)
            window.samples += 1

        # --- 没接触的：给一点宽限再判定断开 ---
        for k, window in list(self._windows.items()):
            if k in touched or not window.active:
                continue

            if (now - window.last) > self.break_grace:
                window.active = False
                window.duration = window.last - window.start

        return {k: w for k, w in self._windows.items() if w.active or w.duration > 0.0}

    # ------------------------------------------------------------------

    def duration(self, hand_label: str, object_label: str) -> float:
        window = self._windows.get(self.key(hand_label, object_label))
        return window.duration if window else 0.0

    def is_active(self, hand_label: str, object_label: str) -> bool:
        window = self._windows.get(self.key(hand_label, object_label))
        return bool(window and window.active)

    def active_object(self, object_label: str) -> bool:
        """任意一只手正在接触该物体。"""
        return any(
            w.active and w.key.endswith(f":{object_label}")
            for w in self._windows.values()
        )

    def best_window(self, object_label: str) -> ContactWindow | None:
        """接触该物体最久的那只手。"""
        candidates = [
            w for w in self._windows.values()
            if w.key.endswith(f":{object_label}") and w.active
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda w: w.duration)

    def clear(self) -> None:
        self._windows.clear()

    def snapshot(self) -> dict:
        return {k: w.as_dict() for k, w in self._windows.items() if w.active}
