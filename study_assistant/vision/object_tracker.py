"""跨帧物体跟踪与位置维持。

YOLO 只跑 5 Hz，但行为融合跑 10 Hz、特征窗口按秒计。中间这段必须靠
追踪器把物体"钉"在画面上，否则每次推理空档物体就消失，交互判断会疯狂
闪烁。

策略：
  * IoU 贪心匹配（同类优先，其次任意类）——够用且没有额外依赖；
  * EMA 平滑 bbox，抑制检测抖动；
  * **coast（滑行）**：物体丢失后仍保留 position 一段时间（默认 1.2 s，
    约 6 个推理周期），期间标记 stale=True，让下游知道这是推测位置；
  * 超过 coast 时间才真正移除。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def iou(box_a: tuple, box_b: tuple) -> float:
    """两个归一化 xyxy 框的 IoU。"""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)

    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih

    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter

    return inter / union if union > 1e-9 else 0.0


def bbox_distance(box_a: tuple, box_b: tuple) -> float:
    """两框中心点距离（归一化）。"""
    acx, acy = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
    bcx, bcy = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


@dataclass
class TrackedObject:
    """一个被持续跟踪的物体。"""

    track_id: int
    label: str
    coco_name: str
    bbox: tuple                     # 平滑后的归一化 xyxy
    confidence: float
    first_seen: float
    last_seen: float                # 最后一次**真实检测**时间
    hits: int = 1
    missed: int = 0
    stale: bool = False             # True = 当前是滑行位置，非真实检测

    @property
    def center(self) -> tuple:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    @property
    def age(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.bbox
        return (x0 - margin) <= x <= (x1 + margin) and (y0 - margin) <= y <= (y1 + margin)

    def distance_to(self, x: float, y: float) -> float:
        """点到框的**外部**距离；点在框内返回 0。"""
        x0, y0, x1, y1 = self.bbox

        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)

        return (dx * dx + dy * dy) ** 0.5


@dataclass
class TrackerSnapshot:
    """某一时刻的完整跟踪结果。"""

    objects: list = field(default_factory=list)
    timestamp: float = 0.0

    def by_label(self, label: str) -> list:
        return [o for o in self.objects if o.label == label]

    def has(self, label: str) -> bool:
        return any(o.label == label for o in self.objects)

    def get(self, track_id: int) -> TrackedObject | None:
        for obj in self.objects:
            if obj.track_id == track_id:
                return obj
        return None

    def nearest(self, label: str, point: tuple) -> TrackedObject | None:
        candidates = self.by_label(label)
        if not candidates:
            return None

        px, py = point
        return min(candidates, key=lambda o: ((o.center[0] - px) ** 2 + (o.center[1] - py) ** 2))

    def nearest_any(self, labels: list, point: tuple) -> TrackedObject | None:
        best, best_d = None, 1e9
        for obj in self.objects:
            if obj.label not in labels:
                continue
            dx = obj.center[0] - point[0]
            dy = obj.center[1] - point[1]
            d = dx * dx + dy * dy
            if d < best_d:
                best, best_d = obj, d
        return best

    def labels(self) -> set:
        return {o.label for o in self.objects}


class ObjectTracker:
    """把低频检测结果变成连续、稳定的物体轨迹。"""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_distance: float = 0.22,
        coast_seconds: float = 1.2,
        smoothing: float = 0.45,
        min_hits: int = 1,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_distance = float(max_distance)
        self.coast_seconds = float(coast_seconds)
        self.smoothing = float(smoothing)     # EMA 系数（越大越跟手）
        self.min_hits = int(min_hits)

        self._tracks: dict[int, TrackedObject] = {}
        self._next_id = 1

    # ------------------------------------------------------------------

    def update(self, detections: list, now: float | None = None) -> TrackerSnapshot:
        """用一批新检测更新轨迹。

        detections: list[DetectedObject]（归一化 bbox），通常来自 ObjectDetector。
        """
        now = time.perf_counter() if now is None else now

        # 1) 贪心匹配：先按 IoU 从高到低配对
        pairs = []

        for det_idx, det in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.label != det.label:
                    continue

                score = iou(track.bbox, det.bbox)

                if score < self.iou_threshold:
                    # IoU 不够就看中心距离（快速移动/小物体时 IoU 会掉到 0）
                    if bbox_distance(track.bbox, det.bbox) > self.max_distance:
                        continue

                pairs.append((score, det_idx, track_id))

        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()

        for score, det_idx, track_id in pairs:
            if det_idx in matched_dets or track_id in matched_tracks:
                continue

            matched_dets.add(det_idx)
            matched_tracks.add(track_id)

            det = detections[det_idx]
            track = self._tracks[track_id]

            alpha = self.smoothing
            track.bbox = tuple(
                alpha * new + (1.0 - alpha) * old
                for old, new in zip(track.bbox, det.bbox)
            )
            track.confidence = det.confidence
            track.last_seen = now
            track.hits += 1
            track.missed = 0
            track.stale = False

        # 2) 未匹配的检测 → 新建轨迹
        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                continue

            self._tracks[self._next_id] = TrackedObject(
                track_id=self._next_id,
                label=det.label,
                coco_name=det.coco_name,
                bbox=det.bbox,
                confidence=det.confidence,
                first_seen=now,
                last_seen=now,
            )
            self._next_id += 1

        # 3) 未匹配的轨迹 → 滑行或淘汰
        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue

            track.missed += 1
            gap = now - track.last_seen

            if gap > self.coast_seconds:
                del self._tracks[track_id]
            else:
                track.stale = True

        return self.snapshot(now)

    def snapshot(self, now: float | None = None) -> TrackerSnapshot:
        now = time.perf_counter() if now is None else now
        visible = [
            t for t in self._tracks.values()
            if t.hits >= self.min_hits
        ]
        # 稳定的物体优先（hits 多、非滑行）
        visible.sort(key=lambda t: (t.stale, -t.hits))
        return TrackerSnapshot(objects=visible, timestamp=now)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @property
    def count(self) -> int:
        return len(self._tracks)
