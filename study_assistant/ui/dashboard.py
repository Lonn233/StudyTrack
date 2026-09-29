"""主界面 Dashboard（PySide6）—— 浅色现代版。

布局（自上而下）：

    ┌──────────────────────────────────────────────────────────┐
    │  Study Assistant   视图  叠加图层  摄像头  标定  帮助   已标定 │ ← 第一层：菜单
    ├──────────────────────────────────────────────────────────┤
    │  [开始] [暂停] [重启] [结束]   00:42:17    专注 · 写字    │ ← 第二层：监视控制
    ├───────────────────────────────┬──────────────────────────┤
    │                               │  当前状态 / 行为 / 原因   │
    │        摄像头画面              │  时间记录 2×2 + 里程碑    │
    │   （含 ROI / 骨架 / 物体框）    │  行为概率                 │
    │                               │  实时证据                 │
    │  fps · 手部耗时 · 物体耗时     │  事件流                   │
    ├───────────────────────────────┤  智能体（8bit 像素眼）    │
    └───────────────────────────────┴──────────────────────────┘

设计约束：
  * 颜色一律来自 `ui.theme` 的令牌，本文件**不得出现裸的十六进制颜色**
    （状态/行为语义色除外，它们也走 theme 的映射函数）；
  * 界面**不阻塞**视觉循环 —— 由 app.py 以固定频率调用 `update_view()`，
    所有重活都在视觉管线里做完，这里只做显示与格式转换；
  * 画面转 QImage 时**必须**拷贝（`np.ascontiguousarray` + `copy()`），
    否则 numpy 缓冲被下一帧覆写会出现撕裂；
  * 智能体眼睛内嵌在右下角（`eyes_widget`），`app.py` 把它当作
    `app.eyes` 使用 —— 外部接口与旧的悬浮窗完全一致。
"""

from __future__ import annotations

import time

import numpy as np

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..behavior.focus_score import format_duration, grade
from ..behavior.states import (
    BEHAVIOR_LABELS,
    Behavior,
    FocusState,
    behavior_label,
    focus_label,
)
from . import theme as T
from .eye_widget import EyesWidget, mood_label

# 事件类型 → 图标（纯文本符号，像素风友好）
EVENT_ICONS = {
    "distraction": "!",
    "distraction_reminder": "!",
    "recovery": "+",
    "milestone": "*",
    "break_due": "~",
    "break_over": ">",
    "away_start": "->",
    "away_end": "<-",
    "session_start": ">>",
    "session_end": "[]",
    "behavior_shift": "·",
}


class StatTile(QFrame):
    """时间记录的一小格（inset 表面：浅灰底、无边框）。"""

    def __init__(self, title: str, value: str = "0", sub: str = ""):
        super().__init__()
        self.setObjectName("inset")
        self.setMinimumHeight(64)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(0)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricLabel")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")

        self.sub_label = QLabel(sub)
        self.sub_label.setObjectName("metricSub")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.sub_label)

    def set(self, value: str, sub: str = "", color: str | None = None) -> None:
        self.value_label.setText(value)
        self.sub_label.setText(sub)

        # 颜色只作用于数值，标签保持中性灰
        self.value_label.setStyleSheet(
            f"color: {color}; font-size: 18px; font-weight: 600;"
            if color
            else f"color: {T.TEXT_STRONG}; font-size: 18px; font-weight: 600;"
        )


class Dashboard(QMainWindow):
    """主窗口。

    继承 QMainWindow 是为了直接用 menuBar() 承载第一层导航；
    中心 widget 是整体布局容器。
    """

    # 用户操作信号（由 app.py 连接）
    signal_pause = Signal()                 # 请求暂停
    signal_start = Signal()                 # 开始 / 继续
    signal_reset = Signal()                 # 重启会话（收尾旧段 + 开新段）
    signal_end_session = Signal()           # 结束并保存
    signal_calibrate = Signal()             # 重新标定
    signal_toggle_debug = Signal()          # 开关调试窗口
    signal_options_changed = Signal(dict)   # 叠加图层开关变化（整份 options）
    signal_break = Signal()                 # 休息一下
    signal_quit = Signal()                  # 退出
    signal_camera_selected = Signal(int)    # 切换到指定摄像头索引
    signal_camera_rescan = Signal()         # 重新扫描摄像头
    signal_position_prior_changed = Signal(bool)  # 位置先验开/关（标定框是否参与判定）

    # 叠加图层 key → 菜单文案（与 DebugView 的开关同一份 options）
    LAYER_LABELS = (
        ("show_roi", "ROI 区域"),
        ("show_hands", "手部骨架"),
        ("show_objects", "物体框"),
        ("show_interaction_lines", "交互连线"),
        ("show_trajectory", "运动轨迹"),
    )

    def __init__(self, config, calibration=None, options: dict | None = None):
        super().__init__()

        self.setWindowTitle("桌面学习行为检测与专注陪伴助手")
        self.resize(1200, 860)
        self.setMinimumSize(1000, 700)

        self.calibration = calibration
        self.config = config          # 供菜单读初始开关状态（位置先验等）
        self._last_events_seen = 0
        self.paused = False

        # 叠加图层状态（app.py 会用同一份 dict 渲染画面）
        self.layer_options = dict(options or {})

        self._build_menu()
        self._build_ui()

        self.set_paused_state(False)

    # ------------------------------------------------------------------
    # 第一层导航：菜单栏
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        bar = self.menuBar()
        bar.setStyleSheet(
            f"QMenuBar{{background:{T.WHITE};border-bottom:1px solid {T.BORDER};}}"
            f"QMenuBar::item{{padding:6px 12px;color:{T.TEXT_MUTED};border-radius:{T.RADIUS_SM}px;}}"
            f"QMenuBar::item:selected{{background:{T.SURFACE};color:{T.TEXT};}}"
        )

        # --- 视图 ---
        m_view = bar.addMenu("视图")
        act_debug = m_view.addAction("调试窗口")
        act_debug.triggered.connect(self.signal_toggle_debug.emit)
        m_view.addSeparator()
        act_quit = m_view.addAction("退出")
        act_quit.triggered.connect(self.signal_quit.emit)

        # --- 叠加图层 ---
        m_layer = bar.addMenu("叠加图层")
        self._layer_actions = {}
        for key, label in self.LAYER_LABELS:
            act = m_layer.addAction(label)
            act.setCheckable(True)
            act.setChecked(bool(self.layer_options.get(key, True)))
            act.toggled.connect(lambda checked, k=key: self._on_layer(k, checked))
            self._layer_actions[key] = act

        # --- 标定 ---
        m_calib = bar.addMenu("标定")
        self._calib_status_action = m_calib.addAction("标定状态：未知")
        self._calib_status_action.setEnabled(False)
        m_calib.addSeparator()
        # 位置先验开关：关闭时判定只依赖物体检测与手部动作，
        # 标定框不参与任何结论（config 默认 false，见 behavior.use_position_prior）
        pos_default = False
        try:
            pos_default = bool((self.config.behavior or {}).get(
                "use_position_prior", False
            ))
        except Exception:
            pass
        act_pos = m_calib.addAction("启用位置先验（按标定框位置判定）")
        act_pos.setCheckable(True)
        act_pos.setChecked(pos_default)
        act_pos.toggled.connect(self.signal_position_prior_changed.emit)
        self._pos_prior_action = act_pos
        act_calib = m_calib.addAction("重新标定桌面…")
        act_calib.triggered.connect(self.signal_calibrate.emit)

        # --- 摄像头 ---
        m_cam = bar.addMenu("摄像头")
        self._cam_menu = m_cam
        # 状态行（禁用，只读）：显示当前正在用的设备
        self._cam_status_action = m_cam.addAction("当前：（扫描中）")
        self._cam_status_action.setEnabled(False)
        self._cam_sep = m_cam.addSeparator()   # 设备列表插在这条分隔线之前
        act_rescan = m_cam.addAction("重新扫描摄像头")
        act_rescan.triggered.connect(self.signal_camera_rescan.emit)
        self._cam_device_actions: list = []

        # --- 帮助 ---
        m_help = bar.addMenu("帮助")
        act_doc = m_help.addAction("使用说明（README）")
        act_doc.triggered.connect(self._open_readme)
        act_dir = m_help.addAction("打开数据文件夹")
        act_dir.triggered.connect(self._open_data_dir)
        m_help.addSeparator()
        act_about = m_help.addAction("关于")
        act_about.triggered.connect(self._show_about)

        self._sync_calib_status()

        # 品牌名放在菜单栏最左（角标控件）。间距用 contentsMargins，
        # 不用 QSS padding（QLabel sizeHint 不含 QSS padding，会裁字）。
        brand = QLabel("Study Assistant")
        brand.setStyleSheet(
            f"font-weight: 600; font-size: 14px; color: {T.TEXT};"
        )
        brand.setContentsMargins(12, 0, 16, 0)
        bar.setCornerWidget(brand, Qt.TopLeftCorner)

    def _on_layer(self, key: str, checked: bool) -> None:
        self.layer_options[key] = bool(checked)
        # 发整份 dict，app.py 直接 update 进自己的 options
        self.signal_options_changed.emit(dict(self.layer_options))

    def _sync_calib_status(self) -> None:
        done = bool(getattr(self.calibration, "completed", False))
        self._calib_status_action.setText(
            "标定状态：已标定" if done else "标定状态：未标定"
        )
        # 顶部控制条右侧的轻量提示。_build_menu 先于 _build_ui 执行，
        # 那时控件还没建出来，用 getattr 兜住初始化顺序。
        hint = getattr(self, "calib_hint", None)
        if hint is not None:
            hint.setText("已标定" if done else "未标定")
            hint.setStyleSheet(
                f"color: {T.TEXT_FAINT}; font-size: {T.FONT_SIZE_TINY}px;"
                f"padding-left: 12px;"
            )

    def set_calibration(self, calibration) -> None:
        """标定完成后由 app.py 调用。"""
        self.calibration = calibration
        self._sync_calib_status()

    # --- 摄像头菜单 ---------------------------------------------------

    def refresh_camera_menu(self, cameras, current_index: int | None) -> None:
        """重填摄像头设备列表（app.py 在启动与重新扫描后调用）。

        ``cameras`` 是 camera_enum.enumerate_cameras() 的结果；
        设备项插在「状态行」与「重新扫描」之间，当前项打勾。
        """
        from ..camera.camera_enum import describe

        for act in getattr(self, "_cam_device_actions", []):
            self._cam_menu.removeAction(act)
            act.deleteLater()
        self._cam_device_actions = []

        cur = next((c for c in cameras if c.index == current_index), None)
        if current_index is None:
            self._cam_status_action.setText("当前：（未选择）")
        elif cur is not None:
            self._cam_status_action.setText(f"当前：{describe(cur, current=True)}")
        else:
            self._cam_status_action.setText(f"当前：索引 {current_index}")

        for info in cameras:
            is_cur = info.index == current_index
            act = QAction(describe(info, current=is_cur), self._cam_menu)
            act.setCheckable(True)
            act.setChecked(is_cur)
            if not info.ok and not is_cur:
                act.setEnabled(False)
                act.setToolTip("该索引打不开（可能不存在或被其他程序占用）")
            # 同一索引重复选择由 app 层忽略；这里统一发信号
            act.triggered.connect(
                lambda checked=False, idx=info.index: self.signal_camera_selected.emit(idx)
            )
            self._cam_menu.insertAction(self._cam_sep, act)
            self._cam_device_actions.append(act)

    def post_system_event(self, message: str, kind: str = "camera", level: int = 0) -> None:
        """往事件流里插一条系统事件（摄像头切换等，不经行为状态机）。"""
        self._append_events([
            {"kind": kind, "message": message, "level": level, "ts": time.time()}
        ])

    def _open_readme(self) -> None:
        from pathlib import Path

        readme = Path(__file__).resolve().parents[2] / "README.md"
        if readme.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(readme)))

    def _open_data_dir(self) -> None:
        from pathlib import Path

        data_dir = Path.home() / "StudyAssistant"
        data_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(data_dir)))

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于",
            "桌面学习行为检测与专注陪伴助手\n\n"
            "不靠人脸/视线，只看「手在哪、在动什么、和什么物体在一起」。\n"
            "手部 15Hz · 物体 5Hz · 行为判定全部可解释、可调。\n\n"
            "提示：同一时刻只能有一个实例占用摄像头。",
        )

    # ------------------------------------------------------------------
    # 第二层导航 + 主体布局
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------------- 第二层：监视控制条 ----------------
        action_bar = QFrame()
        action_bar.setObjectName("navAction")
        action_row = QHBoxLayout(action_bar)
        action_row.setContentsMargins(16, 10, 16, 10)
        action_row.setSpacing(8)

        self.btn_start = QPushButton("开始")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.signal_start.emit)

        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setObjectName("ghost")
        self.btn_pause.clicked.connect(self.signal_pause.emit)

        self.btn_restart = QPushButton("重启")
        self.btn_restart.setObjectName("ghost")
        self.btn_restart.clicked.connect(self.signal_reset.emit)

        self.btn_end = QPushButton("结束")
        self.btn_end.setObjectName("ghost")
        self.btn_end.clicked.connect(self.signal_end_session.emit)

        self.btn_break = QPushButton("休息一下")
        self.btn_break.setObjectName("ghost")
        self.btn_break.clicked.connect(self.signal_break.emit)

        for b in (self.btn_start, self.btn_pause, self.btn_restart,
                  self.btn_end, self.btn_break):
            action_row.addWidget(b)

        # 会话时钟（间距用 addSpacing；字体用 QFont 直接设 —— QSS 的
        # font-weight 不参与 QLabel.sizeHint 计算，粗体渲染会比预估值宽，
        # 结果就是最后几个字符被裁掉）
        action_row.addSpacing(12)
        self.clock_label = QLabel("00:00")
        clock_font = self.clock_label.font()
        clock_font.setPointSizeF(10.5)
        clock_font.setBold(True)
        self.clock_label.setFont(clock_font)
        self.clock_label.setStyleSheet(f"color: {T.TEXT_MUTED};")
        action_row.addWidget(self.clock_label)

        action_row.addStretch(1)

        # 当前状态 + 行为（实时）
        self.state_badge = QLabel("识别中")
        self.state_badge.setStyleSheet(
            f"background: {T.AMBER_BG}; color: {T.AMBER};"
            f"border-radius: 999px; padding: 4px 14px; font-weight: 600;"
        )
        action_row.addWidget(self.state_badge)

        action_row.addSpacing(6)
        self.behavior_badge = QLabel("行为：—")
        behavior_font = self.behavior_badge.font()
        behavior_font.setPointSizeF(9.5)
        self.behavior_badge.setFont(behavior_font)
        self.behavior_badge.setStyleSheet(f"color: {T.TEXT_MUTED};")
        action_row.addWidget(self.behavior_badge)

        # 标定状态轻提示（最右）
        action_row.addSpacing(12)
        self.calib_hint = QLabel("—")
        hint_font = self.calib_hint.font()
        hint_font.setPointSizeF(8.5)
        self.calib_hint.setFont(hint_font)
        self.calib_hint.setStyleSheet(f"color: {T.TEXT_FAINT};")
        action_row.addWidget(self.calib_hint)

        root.addWidget(action_bar)

        # ---------------- 主体：左画面 + 右控制板 ----------------
        body = QHBoxLayout()
        body.setContentsMargins(16, 12, 16, 16)
        body.setSpacing(T.GAP)
        root.addLayout(body, stretch=1)

        # ===== 左：摄像头画面 =====
        left = QVBoxLayout()
        left.setSpacing(T.GAP)

        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(6, 6, 6, 6)

        self.video_label = QLabel("等待摄像头…")
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(560, 400)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.video_label)

        left.addWidget(video_frame, stretch=1)

        # 性能小条
        self.fps_label = QLabel("— fps")
        self.fps_label.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: {T.FONT_SIZE_TINY}px;"
        )
        perf_row = QHBoxLayout()
        perf_row.setContentsMargins(4, 0, 4, 0)
        perf_row.addWidget(self.fps_label)
        perf_row.addStretch(1)
        left.addLayout(perf_row)

        body.addLayout(left, stretch=3)

        # ===== 右：控制板 =====
        right = QVBoxLayout()
        right.setSpacing(T.GAP)

        # --- 当前状态卡 ---
        status_card = QFrame()
        status_card.setObjectName("card")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(12, 10, 12, 10)
        status_layout.setSpacing(4)

        status_title = QLabel("当前状态")
        status_title.setObjectName("sectionLabel")
        status_layout.addWidget(status_title)

        self.status_state_label = QLabel("识别中")
        self.status_state_label.setStyleSheet(
            f"color: {T.AMBER}; font-size: 20px; font-weight: 600;"
        )
        status_layout.addWidget(self.status_state_label)

        self.status_behavior_label = QLabel("行为：—（—）")
        self.status_behavior_label.setStyleSheet(
            f"color: {T.TEXT}; font-size: 14px;"
        )
        status_layout.addWidget(self.status_behavior_label)

        self.status_reason_label = QLabel(" ")
        self.status_reason_label.setWordWrap(True)
        # wordWrap 标签的 sizeHint 宽度 = 整行不换行的宽度，会把卡片最小
        # 宽度撑到几百 px，右边整列被顶出视口。横向 Ignored 让布局随便
        # 压缩它，实际换行由 heightForWidth 处理。
        self.status_reason_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self.status_reason_label.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: {T.FONT_SIZE_TINY}px;"
        )
        status_layout.addWidget(self.status_reason_label)

        right.addWidget(status_card)

        # --- 时间记录 2×2 ---
        self.tile_focus = StatTile("学习时间", "0m 00s", "本次会话累计")
        self.tile_distract = StatTile("分心时间", "0m 00s", "含手机 / 小动作")
        self.tile_away = StatTile("离席时间", "0m 00s", "离开座位")
        self.tile_score = StatTile("专注分", "0", "0-100")

        grid = QGridLayout()
        grid.setSpacing(8)
        for i, tile in enumerate(
            (self.tile_focus, self.tile_distract, self.tile_away, self.tile_score)
        ):
            grid.addWidget(tile, i // 2, i % 2)

        right.addLayout(grid)

        # --- 里程碑 ---
        milestone_row = QHBoxLayout()
        milestone_row.setSpacing(8)
        self.milestone_label = QLabel("下个里程碑：—")
        self.milestone_label.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: {T.FONT_SIZE_SMALL}px;"
        )
        self.milestone_bar = QProgressBar()
        self.milestone_bar.setRange(0, 100)
        self.milestone_bar.setValue(0)
        self.milestone_bar.setFixedHeight(6)
        self.milestone_bar.setMinimumWidth(24)
        self.milestone_bar.setTextVisible(False)
        self.milestone_bar.setStyleSheet(
            f"QProgressBar{{background:{T.SURFACE_ALT};border:none;border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{T.GREEN};border-radius:3px;}}"
        )
        milestone_row.addWidget(self.milestone_label)
        milestone_row.addWidget(self.milestone_bar, stretch=1)
        right.addLayout(milestone_row)

        # --- 行为概率（两列 5 行，压缩纵向空间）---
        behavior_card = QFrame()
        behavior_card.setObjectName("card")
        behavior_layout = QVBoxLayout(behavior_card)
        behavior_layout.setContentsMargins(12, 10, 12, 10)
        behavior_layout.setSpacing(6)

        btitle = QLabel("行为概率")
        btitle.setObjectName("sectionLabel")
        behavior_layout.addWidget(btitle)

        self.behavior_rows = {}
        bgrid = QGridLayout()
        bgrid.setSpacing(4)
        bgrid.setHorizontalSpacing(16)

        for idx, behavior in enumerate(Behavior):
            name = QLabel(BEHAVIOR_LABELS.get(behavior, behavior.value))
            name.setStyleSheet(
                f"font-size: {T.FONT_SIZE_TINY}px; color: {T.TEXT_MUTED};"
            )
            name.setFixedWidth(52)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(5)
            # QProgressBar 的默认最小宽度很大（~100px），两列拼起来会把
            # 卡片最小宽度顶到超出滚动视口，右半列被整块推出屏幕外。
            bar.setMinimumWidth(24)
            color = T.behavior_color(behavior)
            bar.setStyleSheet(
                f"QProgressBar{{background:{T.SURFACE_ALT};border:none;"
                f"border-radius:2px;}}"
                f"QProgressBar::chunk{{background:{color};border-radius:2px;}}"
            )

            value = QLabel("0%")
            value.setStyleSheet(
                f"font-size: {T.FONT_SIZE_TINY}px; color: {T.TEXT_FAINT};"
            )
            value.setFixedWidth(30)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(name)
            row.addWidget(bar, stretch=1)
            row.addWidget(value)

            cell = QWidget()
            cell.setLayout(row)
            bgrid.addWidget(cell, idx // 2, idx % 2)

            self.behavior_rows[behavior] = (bar, value)

        behavior_layout.addLayout(bgrid)

        right.addWidget(behavior_card)

        # --- 实时证据 ---
        evidence_card = QFrame()
        evidence_card.setObjectName("card")
        evidence_layout = QVBoxLayout(evidence_card)
        evidence_layout.setContentsMargins(12, 10, 12, 10)
        evidence_layout.setSpacing(4)

        etitle = QLabel("实时证据")
        etitle.setObjectName("sectionLabel")
        evidence_layout.addWidget(etitle)

        self.evidence_label = QLabel("—")
        self.evidence_label.setWordWrap(True)
        self.evidence_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self.evidence_label.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: {T.FONT_SIZE_SMALL}px;"
        )
        self.evidence_label.setMinimumHeight(44)
        evidence_layout.addWidget(self.evidence_label)

        right.addWidget(evidence_card)

        # --- 事件流 ---
        event_card = QFrame()
        event_card.setObjectName("card")
        event_layout = QVBoxLayout(event_card)
        event_layout.setContentsMargins(12, 10, 12, 10)
        event_layout.setSpacing(6)

        evtitle = QLabel("事件流")
        evtitle.setObjectName("sectionLabel")
        event_layout.addWidget(evtitle)

        self.event_list = QListWidget()
        self.event_list.setMinimumHeight(70)
        self.event_list.setStyleSheet(
            f"QListWidget{{border:1px solid {T.BORDER};border-radius:{T.RADIUS_MD}px;}}"
            f"QListWidget::item{{border-bottom:1px solid {T.SURFACE};}}"
        )
        event_layout.addWidget(self.event_list, stretch=1)

        right.addWidget(event_card, stretch=1)

        # --- 智能体（8bit 眼睛） ---
        agent_card = QFrame()
        agent_card.setObjectName("card")
        agent_layout = QVBoxLayout(agent_card)
        agent_layout.setContentsMargins(12, 10, 12, 10)
        agent_layout.setSpacing(6)

        atitle_row = QHBoxLayout()
        atitle = QLabel("智能体")
        atitle.setObjectName("sectionLabel")
        self.agent_mood_label = QLabel("—")
        self.agent_mood_label.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: {T.FONT_SIZE_TINY}px;"
        )
        atitle_row.addWidget(atitle)
        atitle_row.addStretch(1)
        atitle_row.addWidget(self.agent_mood_label)
        agent_layout.addLayout(atitle_row)

        # 眼睛内嵌（不再是独立悬浮窗）。parent 传 agent_card → widget 走
        # "嵌进主界面"的绘制分支（白屏 + 浅边框）。
        self.eyes_widget = EyesWidget(width=236, height=128, parent=agent_card)
        self.eyes_widget.setStyleSheet(
            f"background: {T.WHITE}; border: 1px solid {T.BORDER};"
            f"border-radius: {T.RADIUS_MD}px;"
        )
        agent_layout.addWidget(
            self.eyes_widget, stretch=0, alignment=Qt.AlignHCenter
        )

        acaption = QLabel("会眨眼 · 会看你 · 会做表情")
        acaption.setAlignment(Qt.AlignCenter)
        acaption.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: {T.FONT_SIZE_TINY}px;"
        )
        agent_layout.addWidget(acaption)

        right.addWidget(agent_card)

        # 右侧控制板放进滚动容器：内容高度超过窗口时不出血、不裁字，
        # 而是出现一条细滚动条（样式见 theme.QSS）。
        # 内层再垫 12px 右边距：否则竖滚动条会压在内容上。
        right_widget = QWidget()
        outer = QVBoxLayout(right_widget)
        outer.setContentsMargins(0, 0, 12, 0)
        outer.addLayout(right)

        from PySide6.QtWidgets import QScrollArea

        scroll = QScrollArea()
        scroll.setWidget(right_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(372)
        scroll.setMaximumWidth(430)
        scroll.setStyleSheet(
            "QScrollArea{background: transparent;}"
            "QScrollArea > QWidget > QWidget{background: transparent;}"
        )
        body.addWidget(scroll, stretch=2)

    # ------------------------------------------------------------------
    # 暂停态
    # ------------------------------------------------------------------

    def set_paused_state(self, paused: bool) -> None:
        """由 app.py 在暂停/恢复后调用，刷新按钮可用性与文案。"""
        self.paused = bool(paused)

        self.btn_start.setText("继续" if self.paused else "开始")
        self.btn_pause.setEnabled(not self.paused)
        self.btn_end.setEnabled(True)

        if self.paused:
            self.state_badge.setText("已暂停")
            self.state_badge.setStyleSheet(
                f"background: {T.BLUE_BG}; color: {T.BLUE};"
                f"border-radius: 999px; padding: 4px 14px; font-weight: 600;"
            )

    # ------------------------------------------------------------------
    # 更新接口
    # ------------------------------------------------------------------

    def update_view(
        self,
        annotated_frame=None,
        state_snapshot=None,
        estimate=None,
        stats=None,
        session=None,
        vision_frame=None,
        events=None,
        fps: float = 0.0,
    ) -> None:
        """由主循环调用，刷新整个界面。"""
        if annotated_frame is not None:
            self._set_frame(annotated_frame)

        if vision_frame is not None:
            self.fps_label.setText(
                f"{vision_frame.fps:.1f} fps · 手 {vision_frame.hands_ms:.0f}ms"
                f" · 物 {vision_frame.objects_ms:.0f}ms"
                f" · {len(vision_frame.tracked.objects)} 物体"
            )
        elif fps:
            self.fps_label.setText(f"{fps:.1f} fps")

        if state_snapshot is not None:
            self._update_state(state_snapshot)

        if estimate is not None:
            self._update_behaviors(estimate)

        if stats is not None:
            self._update_stats(stats)
            self._update_clock(stats.total_seconds)

        if session is not None:
            self._update_milestone(session)

        if events:
            self._append_events(events)

        # 眼睛状态文字
        if self.eyes_widget is not None:
            self.agent_mood_label.setText(mood_label(self.eyes_widget.mood))

    # ------------------------------------------------------------------

    def _set_frame(self, frame_bgr: np.ndarray) -> None:
        """BGR numpy → QPixmap。必须拷贝，否则会被下一帧覆写。"""
        try:
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            h, w, _ = rgb.shape

            image = QImage(
                rgb.data, w, h, 3 * w, QImage.Format_RGB888
            ).copy()

            pixmap = QPixmap.fromImage(image).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.video_label.setPixmap(pixmap)
        except Exception as exc:
            self.video_label.setText(f"画面渲染失败：{exc}")

    def _update_state(self, snapshot) -> None:
        state = snapshot.state
        color = T.state_color(state)
        bg = T.state_bg(state)

        text = focus_label(state)

        # 候选状态：显示进度，让用户知道"正在被判定"
        if snapshot.candidate is not None and snapshot.candidate_progress > 0.15:
            text += f" → {focus_label(snapshot.candidate)} {snapshot.candidate_progress:.0%}"

        if snapshot.state is FocusState.PAUSED:
            # 暂停态由 set_paused_state 主导，这里不覆盖徽章文案
            self.status_state_label.setText("已暂停")
            self.status_state_label.setStyleSheet(
                f"color: {T.BLUE}; font-size: 20px; font-weight: 600;"
            )
            self.status_behavior_label.setText(
                f"行为：{behavior_label(snapshot.behavior)}"
                f"（{snapshot.confidence:.0%}）"
            )
            self.status_reason_label.setText(snapshot.reason or " ")
            return

        self.state_badge.setText(text)
        self.state_badge.setStyleSheet(
            f"background: {bg}; color: {color};"
            f"border-radius: 999px; padding: 4px 14px; font-weight: 600;"
        )

        self.status_state_label.setText(focus_label(state))
        self.status_state_label.setStyleSheet(
            f"color: {color}; font-size: 20px; font-weight: 600;"
        )

        behavior = snapshot.behavior
        self.behavior_badge.setText(
            f"行为：{behavior_label(behavior)}（{snapshot.confidence:.0%}）"
        )
        self.status_behavior_label.setText(
            f"行为：{behavior_label(behavior)}（{snapshot.confidence:.0%}）"
        )
        self.status_reason_label.setText(snapshot.reason or " ")

    def _update_behaviors(self, estimate) -> None:
        scores = getattr(estimate, "scores", {}) or {}

        for behavior, (bar, label) in self.behavior_rows.items():
            value = float(scores.get(behavior, 0.0))
            bar.setValue(int(round(value * 100)))
            label.setText(f"{value * 100:.0f}%")

        # 证据
        evidence = getattr(estimate, "evidence", None)

        if not evidence:
            return

        current = getattr(estimate, "behavior", None)
        if current is None or current is Behavior.UNKNOWN:
            current = getattr(estimate, "pending", None)

        lines = evidence.get(current) or []

        if lines:
            body = "\n".join(f"· {line}" for line in lines[:5])
            self.evidence_label.setText(
                f"<b style='color:{T.TEXT}'>{behavior_label(current)}</b><br>{body}"
            )
        else:
            self.evidence_label.setText("—")

    def _update_stats(self, stats) -> None:
        self.tile_focus.set(
            format_duration(stats.focused_seconds),
            f"累计 {format_duration(stats.total_seconds)}",
            T.GREEN,
        )

        letter, comment = grade(stats.focus_score)
        score_color = (
            T.GREEN if stats.focus_score >= 75
            else T.AMBER if stats.focus_score >= 50
            else T.RED
        )
        self.tile_score.set(f"{stats.focus_score:.0f}", f"{letter} · {comment}", score_color)

        self.tile_distract.set(
            format_duration(stats.distracted_seconds),
            f"最长专注 {format_duration(stats.longest_focus_streak)}",
            T.RED if stats.distracted_seconds > 60 else T.TEXT_STRONG,
        )

        self.tile_away.set(
            format_duration(stats.away_seconds),
            f"专注占比 {stats.focus_ratio * 100:.0f}%",
            T.TEXT_STRONG,
        )

    def _update_clock(self, total_seconds: float) -> None:
        seconds = int(max(0, total_seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        self.clock_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def _update_milestone(self, session) -> None:
        timer = session.timer

        nxt = timer.next_milestone()
        progress = timer.milestone_progress()

        if nxt is None:
            self.milestone_label.setText("里程碑：全部达成")
            self.milestone_bar.setValue(100)
        else:
            self.milestone_label.setText(
                f"下个里程碑 {nxt} 分钟 · {progress * 100:.0f}%"
            )
            self.milestone_bar.setValue(int(progress * 100))

    def _append_events(self, events) -> None:
        for event in events:
            if isinstance(event, dict):
                kind = event.get("kind", "")
                message = event.get("message", "")
                local_time = time.strftime(
                    "%H:%M:%S", time.localtime(event.get("ts", time.time()))
                )
                level = int(event.get("level", 0))
            else:
                kind = getattr(event, "kind", "")
                message = getattr(event, "message", "")
                local_time = getattr(event, "local_time", "")
                level = int(getattr(event, "level", 0))

            icon = EVENT_ICONS.get(kind, "·")

            color = (
                T.RED if level >= 2 and "distract" in kind
                else T.AMBER if kind == "milestone"
                else T.TEXT_MUTED
            )

            item = QListWidgetItem(f"{icon} {local_time}  {message}")
            item.setForeground(QColor(color))
            item.setToolTip(kind)

            self.event_list.insertItem(0, item)

        # 只保留最近 120 条，避免无限增长
        while self.event_list.count() > 120:
            self.event_list.takeItem(self.event_list.count() - 1)
