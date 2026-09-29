"""Debug 视图：把管线内部的一切都摊开看。

验收 CV 移植时最有用的一个窗口 —— 它回答的问题是：
  * 手部推理真的在跑吗（还是异常被吞了、只是"返回值合法"）？
  * 每个检测器的耗时是多少（Pi 上够不够用）？
  * 交互距离的实际数值是多少（阈值调得对不对）？
  * 物体是真实检测还是滑行位置？

所以这里刻意不做美化，只做"信息密度"。

两个文本构建函数（`build_perf_text` / `build_feature_text`）放在模块级，
这样 headless 模式也能把同样的内容 dump 到文件，不必起 Qt。
"""

from __future__ import annotations

import time

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..behavior.states import Behavior, behavior_label, focus_label
from .overlay import render_debug_frame

DEBUG_QSS = """
QWidget      { background: #0b0f14; color: #cbd5e1;
               font-family: Consolas, "Microsoft YaHei UI", monospace; font-size: 11px; }
QLabel#title { color: #64748b; font-weight: 600; padding: 2px 0; }
QCheckBox    { color: #94a3b8; spacing: 5px; }
QCheckBox::indicator { width: 12px; height: 12px; }
QPlainTextEdit { background: #0e141b; border: 1px solid #1f2937;
                 border-radius: 6px; color: #9fb3c8; padding: 6px; }
"""

DEFAULT_OPTIONS = {
    "show_roi": True,
    "show_hands": True,
    "show_objects": True,
    "show_interaction_lines": True,
    "show_trajectory": True,
}


def _fmt_motion(motion) -> str:
    """把一条 HandMotion 压成一行。"""
    if motion is None or motion.samples < 2:
        return "样本不足"

    return (
        f"n={motion.samples:<3d} "
        f"v={motion.speed:.3f} sd={motion.speed_std:.3f} "
        f"L={motion.trajectory_length:.3f} R={motion.motion_range:.3f} "
        f"dir={motion.direction_change_rate:.1f}/s "
        f"{'局部' if motion.is_local else '位移'}"
    )


# ----------------------------------------------------------------------
# 文本构建（无 Qt 依赖）
# ----------------------------------------------------------------------


def build_perf_text(vf, board_info: str = "") -> str:
    """性能 / 分频 / 跟踪器面板。"""
    lines = [
        f"板卡       : {board_info or '未知'}",
        f"帧序号     : {vf.index}",
        f"采集帧率   : {vf.fps:.1f} fps",
        f"循环耗时   : {vf.loop_ms:.1f} ms",
        "",
        "—— 推理分频 ——",
        f"手部       : {'本次已跑' if vf.hands_fresh else '使用缓存'}  "
        f"{vf.hands_ms:.1f} ms  手数={vf.hands.count}",
        f"物体       : {'本次已跑' if vf.objects_fresh else '使用缓存'}  "
        f"{vf.objects_ms:.1f} ms  数量={len(vf.objects.objects)}",
        "",
        "—— 跟踪器 ——",
        f"轨迹数     : {len(vf.tracked.objects)}",
    ]

    for obj in vf.tracked.objects:
        flag = "滑行" if obj.stale else "实时"
        lines.append(
            f"  #{obj.track_id:<3d} {obj.label:<9s} "
            f"conf={obj.confidence:.2f} hits={obj.hits:<3d} [{flag}] "
            f"area={obj.area:.3f}"
        )

    return "\n".join(lines)


def build_feature_text(
    vf, interactions, estimate=None, snapshot=None,
    session=None, hands_tracker=None,
) -> str:
    """特征 / 交互 / 状态面板。"""
    lines: list[str] = []

    # ---- 状态与行为 ----
    if snapshot is not None:
        lines.append("—— 状态机 ——")
        lines.append(
            f"状态       : {focus_label(snapshot.state)}  "
            f"持续 {snapshot.duration:.1f}s"
        )
        lines.append(f"行为       : {behavior_label(snapshot.behavior)}")
        lines.append(f"切换原因   : {snapshot.reason}")

        if snapshot.candidate is not None:
            lines.append(
                f"候选       : {focus_label(snapshot.candidate)} "
                f"{snapshot.candidate_progress:.0%}"
            )
        lines.append("")

    # ---- 行为打分 ----
    if estimate is not None:
        lines.append("—— 行为打分（窗口均值）——")

        scores = getattr(estimate, "scores", {}) or {}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        for behavior, score in ranked[:5]:
            if score <= 0.01:
                continue
            bar = "#" * int(score * 20)
            lines.append(f"{behavior_label(behavior):<8s} {score:5.3f} {bar}")

        lines.append(
            f"稳定性 {estimate.stability:.2f}  "
            f"领先 {estimate.margin:.3f}  "
            f"sustained {estimate.sustained:.1f}s"
        )

        pending = getattr(estimate, "pending", None)

        if pending is not None and getattr(pending, "value", "UNKNOWN") != "UNKNOWN":
            lines.append(
                f"逼近中     : {behavior_label(pending)} "
                f"{estimate.pending_progress:.0%}"
            )

        # 打字动作签名 —— 拿真机画面标定 typing_* 阈值时就看这里。
        # 它放在证据里，而证据只有当前/候选行为才会显示，所以这里主动捞出来。
        for line in (
            line
            for b in (Behavior.COMPUTER_STUDY, Behavior.IDLE)
            for line in ((getattr(estimate, "evidence", None) or {}).get(b) or [])
            if "打字" in line
        ):
            lines.append(f"打字签名   : {line}")
        lines.append("")

    # ---- 手部运动 ----
    if hands_tracker is not None:
        lines.append("—— 手部运动特征 ——")

        now = vf.now

        for label in hands_tracker.present_labels(now, max_age=0.6):
            lines.append(f"[{label}]")
            for window in hands_tracker.windows:
                motion = hands_tracker.motion(label, window, now)
                lines.append(f"  {window:>3.0f}s {_fmt_motion(motion)}")

        absent = hands_tracker.absent_for(now)
        if absent < 1e6:
            lines.append(f"手部缺失   : {absent:.2f}s")

        lines.append("")

    # ---- 交互 ----
    if interactions is not None:
        lines.append("—— 手-物交互 ——")
        lines.append(
            f"手机   : {'接触' if interactions.phone_contact else '—'}  "
            f"{interactions.phone_duration:.1f}s  {interactions.phone_hands}"
        )
        # 手机和键盘一样，两条通路必须分开看：**没检出手机 = 不许定罪**。
        # 位置命中只说明"手落在那个框里"，看书、敲键盘都会落进去。
        lines.append(
            f"  ├ 位置命中: {'是' if interactions.phone_roi_contact else '否'}"
            f"  {interactions.phone_roi_duration:.1f}s"
        )
        lines.append(
            f"  └ 物体检测: {'是' if interactions.phone_object_contact else '否'}"
            f"  {interactions.phone_object_duration:.1f}s"
        )
        lines.append(
            f"书写区 : {'接触' if interactions.paper_contact else '—'}  "
            f"{interactions.paper_duration:.1f}s  {interactions.paper_hands}"
        )
        lines.append(
            f"键盘   : {'接触' if interactions.keyboard_contact else '—'}  "
            f"{interactions.keyboard_duration:.1f}s"
        )
        # 两条通路必须分开看：位置命中 ≠ 检出键盘。只有「物体检测」这一路
        # 才是真证据；「位置命中」是先验，还要过打字动作签名才算数。
        lines.append(
            f"  ├ 位置命中: {'是' if interactions.keyboard_roi_contact else '否'}"
            f"  {interactions.keyboard_roi_duration:.1f}s"
        )
        lines.append(
            f"  └ 物体检测: {'是' if interactions.keyboard_object_contact else '否'}"
            f"  {interactions.keyboard_object_duration:.1f}s"
            f"  laptop={'是' if interactions.laptop_contact else '否'}"
        )

        objects = " ".join(
            name for name, flag in (
                ("laptop", interactions.laptop_contact),
                ("book", interactions.book_contact),
            ) if flag
        )
        lines.append(f"电脑/书: {objects or '—'}")

        if interactions.hands_off_desk:
            lines.append(f"手离桌面: {interactions.hands_off_desk}")

        near = sorted(interactions.measures, key=lambda m: m.distance_px)[:6]

        if near:
            lines.append("")
            lines.append("最近的 手→物 距离：")
            for m in near:
                mark = "*" if m.contact else " "
                lines.append(
                    f" {mark} {m.hand_label:<5s}→{m.object_label:<9s} "
                    f"d={m.distance_px:6.1f}px ov={m.overlap_ratio:.2f} "
                    f"ct={m.containment:.2f}"
                )

        # ---- ROI 命中 ----
        lines.append("")
        lines.append("—— 手在哪些 ROI ——")
        for label in interactions.hand_on_desk:
            hits = [
                name for name, flag in (
                    ("study", interactions.hand_in_study.get(label)),
                    ("paper", interactions.hand_in_paper_roi.get(label)),
                    ("keyboard", interactions.hand_in_keyboard_roi.get(label)),
                    ("phone", interactions.hand_in_phone_roi.get(label)),
                ) if flag
            ]
            lines.append(f"  {label:<6s} {'/'.join(hits) if hits else '仅桌面'}")

    # ---- 会话累计 ----
    if session is not None:
        lines.append("")
        lines.append("—— 会话累计 ——")
        lines.append(f"总时长     : {session.total_seconds:.0f}s")
        lines.append(f"有效专注   : {session.focused_seconds:.0f}s")
        lines.append(f"当前分心   : {session.current_distraction:.0f}s")
        lines.append(f"即时专注分 : {session.live_score():.0f}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Qt 窗口
# ----------------------------------------------------------------------


class DebugView(QWidget):
    """调试窗口（独立于主界面，可随时开关）。"""

    options_changed = Signal(dict)

    def __init__(self, options: dict | None = None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Debug Vision — 管线内部")
        self.resize(980, 800)
        self.setStyleSheet(DEBUG_QSS)

        self.options = dict(DEFAULT_OPTIONS)
        self.options.update(options or {})

        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ---- 图层开关 ----
        toggles = QHBoxLayout()
        toggles.setSpacing(14)

        for key, label in (
            ("show_roi", "ROI 区域"),
            ("show_hands", "手部骨架"),
            ("show_objects", "物体框"),
            ("show_interaction_lines", "交互连线"),
            ("show_trajectory", "运动轨迹"),
        ):
            box = QCheckBox(label)
            box.setChecked(bool(self.options.get(key, True)))
            box.toggled.connect(lambda checked, k=key: self._on_toggle(k, checked))
            toggles.addWidget(box)

        toggles.addStretch(1)
        root.addLayout(toggles)

        # ---- 画面 ----
        self.video = QLabel("等待画面…")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumHeight(420)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setStyleSheet(
            "background: #06090d; border: 1px solid #1f2937; border-radius: 8px;"
        )
        root.addWidget(self.video, stretch=3)

        # ---- 指标文本 ----
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)

        left_col = QVBoxLayout()
        left_col.setSpacing(2)
        left_col.addWidget(self._title("管线与性能"))
        self.perf_text = QPlainTextEdit()
        self.perf_text.setReadOnly(True)
        self.perf_text.setMaximumBlockCount(400)
        left_col.addWidget(self.perf_text)

        right_col = QVBoxLayout()
        right_col.setSpacing(2)
        right_col.addWidget(self._title("特征与交互"))
        self.feature_text = QPlainTextEdit()
        self.feature_text.setReadOnly(True)
        self.feature_text.setMaximumBlockCount(400)
        right_col.addWidget(self.feature_text)

        metrics_row.addLayout(left_col, stretch=1)
        metrics_row.addLayout(right_col, stretch=1)
        root.addLayout(metrics_row, stretch=2)

    def _title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("title")
        return label

    def _on_toggle(self, key: str, checked: bool) -> None:
        self.options[key] = bool(checked)
        self.options_changed.emit(dict(self.options))

    # ------------------------------------------------------------------

    def update_debug(
        self,
        vision_frame,
        calibration=None,
        interactions=None,
        trajectories=None,
        estimate=None,
        snapshot=None,
        session=None,
        hands_tracker=None,
        board_info: str = "",
    ) -> None:
        """刷新调试窗口。"""
        if vision_frame is None:
            return

        rendered = render_debug_frame(
            vision_frame,
            calibration=calibration,
            interactions=interactions,
            trajectories=trajectories,
            options=self.options,
        )

        self._set_frame(rendered)

        self.perf_text.setPlainText(build_perf_text(vision_frame, board_info))

        self.feature_text.setPlainText(
            build_feature_text(
                vision_frame, interactions, estimate, snapshot, session, hands_tracker
            )
        )

    # ------------------------------------------------------------------

    def _set_frame(self, frame_bgr: np.ndarray) -> None:
        try:
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            h, w, _ = rgb.shape

            image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

            self.video.setPixmap(
                QPixmap.fromImage(image).scaled(
                    self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        except Exception as exc:
            self.video.setText(f"渲染失败：{exc}")

    def dump(self, path: str) -> None:
        """把当前两个面板的内容写到文件（用于远程验收）。"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# Debug dump @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            fh.write("## 性能\n")
            fh.write(self.perf_text.toPlainText())
            fh.write("\n\n## 特征\n")
            fh.write(self.feature_text.toPlainText())
