"""画面标注（ROI / 手部骨架 / 物体框 / 交互连线 / 轨迹 / HUD）。

被 Dashboard 与 Debug 视图共用，避免两套标注逻辑漂移。

⚠️ 这里全部用 OpenCV 绘制，**文字只能是 ASCII**（cv2.putText 不支持
中文，会渲染成方块）。中文文案请交给 Qt 侧的 HUD 面板去画。
"""

from __future__ import annotations

import cv2
import numpy as np

# ROI 显示颜色（BGR）
ROI_COLORS = {
    "desk": (120, 200, 200),
    "study": (120, 230, 120),
    "left_hand": (90, 160, 230),
    "right_hand": (230, 160, 90),
    "paper": (240, 130, 200),
    "keyboard": (250, 200, 150),
    "phone": (250, 120, 120),
}

# 物体类别显示颜色（BGR）
OBJECT_COLORS = {
    "phone": (80, 80, 255),
    "book": (200, 130, 240),
    "laptop": (230, 180, 90),
    "keyboard": (240, 200, 120),
    "mouse": (200, 200, 160),
    "cup": (180, 220, 180),
    "person": (170, 170, 170),
}

HAND_COLOR = {
    "left": (255, 200, 90),
    "right": (90, 200, 255),
    "unknown": (200, 200, 200),
}

# MediaPipe 手部骨架连线（与 vision.hand_detector 保持一致，这里复制一份
# 是为了让 overlay 不依赖 mediapipe 是否安装）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------


def to_pixels(bbox_norm: tuple, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_norm
    return (
        int(x0 * width), int(y0 * height),
        int(x1 * width), int(y1 * height),
    )


def draw_label(
    frame,
    text: str,
    x: int,
    y: int,
    color,
    scale: float = 0.45,
    filled: bool = True,
) -> None:
    """在指定位置画带底色的文字标签（自动防止越界）。"""
    h, w = frame.shape[:2]

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
    )

    y = max(th + 4, min(y, h - 4))
    x = max(0, min(x, w - tw - 6))

    if filled:
        cv2.rectangle(
            frame,
            (x - 3, y - th - 4),
            (x + tw + 4, y + baseline),
            (16, 20, 28),
            -1,
        )

    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA,
    )


# ----------------------------------------------------------------------
# 各图层
# ----------------------------------------------------------------------


def draw_rois(frame, calibration, show_labels: bool = True) -> None:
    """画所有有效 ROI。"""
    if calibration is None:
        return

    h, w = frame.shape[:2]

    for name, roi in calibration.rois.items():
        if not roi.is_valid:
            continue

        x0, y0, x1, y1 = roi.to_pixels(w, h)
        color = ROI_COLORS.get(name, (180, 180, 180))

        # 虚线感：画四角 + 细边，避免糊住画面
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

        corner = max(8, min(18, (x1 - x0) // 4))
        for (cx, cy, dx, dy) in (
            (x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1),
        ):
            cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, 2, cv2.LINE_AA)

        if show_labels:
            draw_label(frame, name, x0 + 3, y0 - 2, color, 0.40, filled=True)


def draw_hands(frame, hands, show_landmarks: bool = True) -> None:
    """画手部骨架、bbox、掌心。"""
    if not hands:
        return

    h, w = frame.shape[:2]

    for hand in hands:
        color = HAND_COLOR.get(hand.label, HAND_COLOR["unknown"])
        landmarks = getattr(hand, "landmarks", None)

        # --- 骨架 ---
        if show_landmarks and landmarks is not None and len(landmarks) >= 21:
            pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]

            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)

            for i, (px, py) in enumerate(pts):
                # 指尖画大一点
                radius = 3 if i in (4, 8, 12, 16, 20) else 2
                cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)

        # --- bbox ---
        x0, y0, x1, y1 = to_pixels(hand.bbox, w, h)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

        # --- 掌心 ---
        px, py = int(hand.palm_center[0] * w), int(hand.palm_center[1] * h)
        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 6, color, 1, cv2.LINE_AA)

        label = f"{hand.label} {hand.handedness_score:.2f}"
        draw_label(frame, label, x0, y0 - 2, color, 0.40)


def draw_objects(frame, tracked_objects, show_confidence: bool = True) -> None:
    """画物体框（滑行位置用虚线表示）。"""
    if not tracked_objects:
        return

    h, w = frame.shape[:2]

    for obj in tracked_objects:
        color = OBJECT_COLORS.get(obj.label, (200, 200, 200))

        if obj.stale:
            # 滑行位置：颜色减半 + 只用细框，提醒"这是推测"
            color = tuple(int(c * 0.55) for c in color)

        x0, y0, x1, y1 = to_pixels(obj.bbox, w, h)

        thickness = 1 if obj.stale else 2
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

        if show_confidence:
            text = f"{obj.label} {obj.confidence:.2f}"
            if obj.stale:
                text += " ~"
            draw_label(frame, text, x0, y0 - 2, color, 0.42)

        # 中心点
        cx, cy = int(obj.center[0] * w), int(obj.center[1] * h)
        cv2.drawMarker(
            frame, (cx, cy), color, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA
        )


def draw_interactions(frame, measures, only_contact: bool = True) -> None:
    """画手-物交互连线。接触越紧线越粗、越亮。"""
    if not measures:
        return

    h, w = frame.shape[:2]

    for m in measures:
        if only_contact and not m.contact:
            continue

        # 需要端点坐标，这里用 measure 里存的锚点信息不可得，
        # 改由调用方在 measure 上附加（见 hand_object_engine 的 debug 字段）
        start = getattr(m, "hand_point", None)
        end = getattr(m, "object_point", None)

        if start is None or end is None:
            continue

        color = OBJECT_COLORS.get(m.object_label, (200, 200, 200))

        # 距离越近越亮
        intensity = max(0.35, min(1.0, 1.0 - m.distance_px / 200.0))
        color = tuple(int(c * intensity) for c in color)

        thickness = 2 if m.contact else 1
        cv2.line(
            frame,
            (int(start[0] * w), int(start[1] * h)),
            (int(end[0] * w), int(end[1] * h)),
            color, thickness, cv2.LINE_AA,
        )


def draw_trajectories(frame, trajectories: dict, max_points: int = 60) -> None:
    """画手部运动轨迹（越新越亮）。"""
    if not trajectories:
        return

    h, w = frame.shape[:2]

    for label, points in trajectories.items():
        if not points:
            continue

        color = HAND_COLOR.get(label, HAND_COLOR["unknown"])
        recent = points[-max_points:]

        for i in range(1, len(recent)):
            alpha = i / len(recent)

            seg_color = tuple(int(c * (0.35 + 0.65 * alpha)) for c in color)
            thickness = 1 + int(2 * alpha)

            cv2.line(
                frame,
                (int(recent[i - 1][0] * w), int(recent[i - 1][1] * h)),
                (int(recent[i][0] * w), int(recent[i][1] * h)),
                seg_color, thickness, cv2.LINE_AA,
            )


def draw_hud(
    frame,
    lines: list,
    origin: tuple = (8, 18),
    line_height: int = 16,
    accent=(120, 240, 255),
) -> None:
    """左上角信息板（ASCII 多行文本）。"""
    if not lines:
        return

    x, y = origin
    h, w = frame.shape[:2]

    # 半透明底
    panel_h = line_height * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - 14), (x + 300, y + panel_h - 14), (14, 18, 26), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        color = accent if i == 0 else (205, 215, 225)
        cv2.putText(
            frame, line, (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA,
        )


def draw_center_banner(frame, text: str, color=(120, 240, 255), scale: float = 0.7) -> None:
    """底部居中横幅（状态提示）。"""
    h, w = frame.shape[:2]

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)

    x = max(6, (w - tw) // 2)
    y = h - 14

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - th - 8), (x + tw + 10, y + 8), (14, 18, 26), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA,
    )


# ----------------------------------------------------------------------
# 组合渲染
# ----------------------------------------------------------------------


def render_debug_frame(
    vision_frame,
    calibration=None,
    interactions=None,
    trajectories=None,
    options: dict | None = None,
    hud_lines: list | None = None,
    banner: str | None = None,
) -> np.ndarray:
    """把一帧 VisionFrame 渲染成带全部标注的 BGR 图。"""
    options = options or {}

    show_roi = options.get("show_roi", True)
    show_hands = options.get("show_hands", True)
    show_objects = options.get("show_objects", True)
    show_interaction = options.get("show_interaction_lines", True)
    show_trajectory = options.get("show_trajectory", True)

    frame = vision_frame.frame.copy()

    if show_roi:
        draw_rois(frame, calibration)

    if show_trajectory and trajectories:
        draw_trajectories(frame, trajectories)

    if show_objects:
        draw_objects(frame, vision_frame.tracked.objects)

    if show_interaction and interactions is not None:
        draw_interactions(frame, interactions.measures)

    if show_hands:
        draw_hands(frame, vision_frame.hands.hands)

    if hud_lines:
        draw_hud(frame, hud_lines)

    if banner:
        draw_center_banner(frame, banner)

    return frame


def resize_to_width(frame, target_width: int):
    """等比缩放到指定宽度（画布太大时用）。"""
    h, w = frame.shape[:2]

    if w <= target_width:
        return frame

    scale = target_width / w
    return cv2.resize(
        frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA
    )
