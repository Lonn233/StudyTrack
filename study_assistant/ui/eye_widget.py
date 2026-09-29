"""8bit 像素风智能体眼睛（PySide6 + QPainter）。

设计取向
--------
**白色屏幕 + 深灰黑像素眼**，像一块小小的电子宠物屏幕：
  * 眼睛不是椭圆，而是**方形像素块**拼出来的 —— 这是 8bit 质感的关键。
    抗锯齿关掉（`Antialiasing=False`），每个"像素"就是一个实心方块，
    边界必须是硬的；开抗锯齿立刻就变成"卡通圆"，不是像素风；
  * 颜色只有三档：深灰黑（眼球）、更深的灰（瞳孔/描边）、白色（屏幕底）。
    情绪**不靠换色表达**（像素画里没有渐变），只靠**形状**：
    眼睑高度、瞳孔大小、瞳孔在眼内的高度、眉毛角度；
  * 眨眼、注视偏移、说话（庆祝）全部按像素网格量化 —— 移动步长是
    整数个像素，不是连续插值，否则会有"半像素模糊"。

对外接口与旧版**完全一致**（`set_mood` / `look_at` / `mood` / `EyeMood`
/ `mood_label` / `MOOD_STYLE`），因此 `app.py` 与 `states.py` 里已有的
`BEHAVIOR_TO_MOOD` / `STATE_TO_MOOD` 映射一个字都不用改。
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

# ----------------------------------------------------------------------
# 像素风配色（只有这三档，不随表情变化）
# ----------------------------------------------------------------------

SCREEN = QColor("#FFFFFF")      # 屏幕底：纯白
INK = QColor("#3A3A3A")         # 眼球本体：深灰黑
INK_DARK = QColor("#1F1F1F")    # 瞳孔 / 描边：更深
SCREEN_EDGE = QColor("#E3E3E6")  # 屏幕边框：极浅，和整机浅色主题一致


class EyeMood:
    """眼睛表情状态。

    ⚠️ 这些常量值被 `study_assistant/behavior/states.py` 的
    `BEHAVIOR_TO_MOOD` / `STATE_TO_MOOD` 以**字符串字面量**引用，
    改动会静默失配（表情退化成默认），因此不要重命名。
    """

    NEUTRAL = "NEUTRAL"
    IDLE = "IDLE"
    FOCUSED = "FOCUSED"
    WRITING = "WRITING"
    READING = "READING"
    PHONE_USE = "PHONE_USE"
    FIDGETING = "FIDGETING"
    SUSPICIOUS = "SUSPICIOUS"
    WORRIED = "WORRIED"
    ANNOYED = "ANNOYED"
    SLEEPY = "SLEEPY"
    CURIOUS = "CURIOUS"
    AWAY = "AWAY"
    MILESTONE = "MILESTONE"
    CELEBRATE = "CELEBRATE"
    BREAK = "BREAK"
    PAUSED = "PAUSED"


# ----------------------------------------------------------------------
# 表情参数（像素语义）
# ----------------------------------------------------------------------
#
#   openness : 眼睛竖向张开度，1.0 = 满高，0 = 闭成一条线
#   pupil    : 瞳孔相对眼球的缩放，1.0 = 常规
#   brow     : 眉毛整体上抬（正）/ 下压（负），单位是"像素行"
#   brow_tilt: 眉毛倾斜。负 = 内侧下压（皱眉、不满），正 = 内侧上抬（好奇）
#   squash   : 眼球下缘被压掉的像素行数，做出"眯成一条"的老式像素表情
#   sleepy   : 上眼睑多盖几行（困倦），与 openness 叠加
#
MOOD_STYLE = {
    EyeMood.NEUTRAL: dict(
        openness=1.0, pupil=1.0, brow=0.0, brow_tilt=0.0, squash=0.0, sleepy=0.0,
        label="正在识别",
    ),
    EyeMood.IDLE: dict(
        openness=1.0, pupil=1.0, brow=0.0, brow_tilt=0.0, squash=0.0, sleepy=0.0,
        label="发呆了？",
    ),
    EyeMood.FOCUSED: dict(
        openness=0.92, pupil=0.95, brow=0.0, brow_tilt=0.25, squash=0.0, sleepy=0.0,
        label="专注中",
    ),
    EyeMood.WRITING: dict(
        openness=0.72, pupil=0.85, brow=0.0, brow_tilt=-0.35, squash=0.0, sleepy=0.2,
        label="在写字",
    ),
    EyeMood.READING: dict(
        openness=0.80, pupil=0.80, brow=0.0, brow_tilt=0.15, squash=0.15, sleepy=0.1,
        label="在阅读",
    ),
    EyeMood.PHONE_USE: dict(
        openness=0.62, pupil=1.00, brow=0.0, brow_tilt=-0.75, squash=0.0, sleepy=0.0,
        label="手机放下了吗",
    ),
    EyeMood.FIDGETING: dict(
        openness=1.00, pupil=1.20, brow=1.0, brow_tilt=0.55, squash=0.0, sleepy=0.0,
        label="手上有小动作",
    ),
    EyeMood.SUSPICIOUS: dict(
        openness=0.55, pupil=0.80, brow=0.0, brow_tilt=-0.60, squash=0.30, sleepy=0.0,
        label="看什么呢",
    ),
    EyeMood.WORRIED: dict(
        openness=0.70, pupil=0.85, brow=0.0, brow_tilt=-0.45, squash=0.10, sleepy=0.0,
        label="有点分心",
    ),
    EyeMood.ANNOYED: dict(
        openness=0.45, pupil=0.75, brow=-1.0, brow_tilt=-1.00, squash=0.35, sleepy=0.0,
        label="该收心了",
    ),
    EyeMood.SLEEPY: dict(
        openness=0.55, pupil=0.80, brow=0.0, brow_tilt=0.0, squash=0.0, sleepy=0.55,
        label="休息一下",
    ),
    EyeMood.CURIOUS: dict(
        openness=1.00, pupil=1.15, brow=1.0, brow_tilt=0.45, squash=0.0, sleepy=0.0,
        label="在找你",
    ),
    EyeMood.AWAY: dict(
        openness=0.85, pupil=1.00, brow=0.0, brow_tilt=0.0, squash=0.0, sleepy=0.0,
        label="人不在座位",
    ),
    EyeMood.MILESTONE: dict(
        openness=0.40, pupil=1.25, brow=1.0, brow_tilt=0.40, squash=0.45, sleepy=0.0,
        label="里程碑！",
    ),
    EyeMood.CELEBRATE: dict(
        openness=0.40, pupil=1.30, brow=1.0, brow_tilt=0.45, squash=0.45, sleepy=0.0,
        label="太棒了",
    ),
    EyeMood.BREAK: dict(
        openness=0.28, pupil=0.70, brow=0.0, brow_tilt=0.0, squash=0.0, sleepy=0.0,
        label="休息中",
    ),
    EyeMood.PAUSED: dict(
        openness=0.28, pupil=0.70, brow=0.0, brow_tilt=0.0, squash=0.0, sleepy=0.0,
        label="已暂停",
    ),
}

DEFAULT_STYLE = MOOD_STYLE[EyeMood.IDLE]


def mood_label(mood: str) -> str:
    """取表情对应的中文标签（未知表情回落到自身名字）。"""
    style = MOOD_STYLE.get(mood)
    return style.get("label", mood) if style else str(mood)


# 像素网格：整块屏幕按这个格数切分，所有绘制都落在格线上。
# 28×16 格是"能画出五官"与"格子足够大、看得出像素感"的平衡点。
GRID_COLS = 28
GRID_ROWS = 16


class EyesWidget(QWidget):
    """一块像素屏幕上的两只眼睛。

    可以作为独立悬浮窗（旧用法），也可以嵌进主窗口的智能体面板
    （`parent` 不传时是顶层窗口；传了 parent 就不是）。
    """

    def __init__(self, width: int = 220, height: int = 120, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Study Assistant")
        self.setFixedSize(width, height)

        # 只有作为顶层窗口时才加悬浮属性；嵌进主界面时加这些会出怪
        if parent is None:
            self.setWindowFlags(
                Qt.FramelessWindowHint
                | Qt.WindowStaysOnTopHint
                | Qt.Tool
            )
            self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.mood = EyeMood.IDLE
        self._target_mood = EyeMood.IDLE

        # 眨眼
        self._blink = 0.0
        self._blink_timer = 0.0
        self._next_blink = random.uniform(1.5, 4.0)
        self._blinking = False
        self._blink_phase = 0.0

        # 注视偏移（-1..1）
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._target_gaze_x = 0.0
        self._target_gaze_y = 0.0

        # 平滑逼近当前值
        self._openness = 1.0
        self._brow = 0.0
        self._pupil_scale = 1.0
        self._squash = 0.0
        self._sleepy = 0.0
        self._celebrate = 0.0

        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)      # 20 fps 足够，像素动画不需要 30

        self._drag_origin = None

    # ------------------------------------------------------------------
    # 外部接口（与旧版保持一致）
    # ------------------------------------------------------------------

    def set_mood(self, mood: str) -> None:
        if mood != self._target_mood:
            self._target_mood = mood
            if mood in (EyeMood.MILESTONE, EyeMood.CELEBRATE):
                self._celebrate = 1.0

    def look_at(self, gx: float, gy: float) -> None:
        """让瞳孔朝某个方向看（gx, gy 为 -1..1）。"""
        self._target_gaze_x = max(-1.0, min(1.0, float(gx)))
        self._target_gaze_y = max(-1.0, min(1.0, float(gy)))

    # ------------------------------------------------------------------
    # 动画
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        dt = 1.0 / 20.0
        self._t += dt

        style = MOOD_STYLE.get(self._target_mood, DEFAULT_STYLE)

        # 平滑逼近目标外观。**在绘制阶段才量化到像素格**，
        # 这里保持连续，否则眨眼会一格一格地跳。
        self._openness += (style["openness"] - self._openness) * 0.18
        self._brow += (style["brow"] - self._brow) * 0.18
        self._pupil_scale += (style["pupil"] - self._pupil_scale) * 0.18
        self._squash += (style["squash"] - self._squash) * 0.18
        self._sleepy += (style["sleepy"] - self._sleepy) * 0.18
        self.mood = self._target_mood

        # 眨眼计时（庆祝时更密）
        self._blink_timer += dt
        interval = self._next_blink * (0.4 if self._celebrate > 0 else 1.0)

        if self._blink_timer >= interval:
            self._blink_timer = 0.0
            self._next_blink = random.uniform(2.0, 5.5)
            self._blink_phase = 0.0
            self._blinking = True

        if self._blinking:
            self._blink_phase += dt
            p = self._blink_phase / 0.18

            if p >= 1.0:
                self._blinking = False
                self._blink = 0.0
            else:
                self._blink = math.sin(p * math.pi)

        # 离席时四处张望
        if self._target_mood == EyeMood.AWAY:
            self._target_gaze_x = math.sin(self._t * 0.9) * 0.9
            self._target_gaze_y = math.sin(self._t * 1.37) * 0.45

        self._gaze_x += (self._target_gaze_x - self._gaze_x) * 0.20
        self._gaze_y += (self._target_gaze_y - self._gaze_y) * 0.20

        if self._celebrate > 0:
            self._celebrate = max(0.0, self._celebrate - dt / 2.0)

        self.update()

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        # **像素风的命门**：关抗锯齿。开了就变成卡通圆，不是 8bit。
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)

        w, h = self.width(), self.height()

        # --- 屏幕底 ---
        if self.parent() is None:
            # 独立悬浮窗：自己画圆角白屏（窗口本身透明）
            painter.setPen(QPen(SCREEN_EDGE, 1))
            painter.setBrush(SCREEN)
            painter.drawRoundedRect(1, 1, w - 2, h - 2, 12, 12)
        else:
            # 嵌进主界面：面板已经给了白底，这里只画一小块屏幕
            painter.setPen(QPen(SCREEN_EDGE, 1))
            painter.setBrush(SCREEN)
            painter.drawRect(0, 0, w - 1, h - 1)

        # --- 像素网格 ---
        #
        # **单位约定（很重要，别再搞混）**：
        #   * 所有几何量（眼球位置/宽高/瞳孔边长）都用**格**为单位，
        #     全部是整数；格号 (gx, gy) 表示"第 gx 列 / 第 gy 行"；
        #   * `X(gx)` / `Y(gy)` 把格号换算成设备像素；
        #   * 绝对不要再对格值调用像素换算 —— 那是把"6 格"当成"6 像素"，
        #     除以格边长就会得到 0，眼睛会塌成一条线（我第一版就是这么错的）。
        #
        cell = min(w / GRID_COLS, h / GRID_ROWS)
        ox = (w - cell * GRID_COLS) / 2.0      # 居中留白
        oy = (h - cell * GRID_ROWS) / 2.0

        def X(gx: float) -> int:
            """格号 → 设备像素 x。"""
            return int(round(ox + gx * cell))

        def Y(gy: float) -> int:
            """格号 → 设备像素 y。"""
            return int(round(oy + gy * cell))

        def SZ(n: float) -> int:
            """n 格 → n 格的像素边长（至少 1px，免得画不出来）。"""
            return max(1, int(round(n * cell)))

        # --- 庆祝时整体上跳（整格跳，不做半格插值）---
        jump = 0.0
        if self._celebrate > 0:
            jump = -1.6 * abs(math.sin(self._t * 9.0)) * self._celebrate

        # --- 眼睛几何（单位：格）---
        eye_w = 8.0        # 眼球宽
        eye_h = 7.0        # 眼球满高
        eye_gap = 2.0      # 两眼之间的空隙
        base_row = 5.0     # 眼球顶部所在行

        left_x = (GRID_COLS - (2 * eye_w + eye_gap)) / 2.0
        right_x = left_x + eye_w + eye_gap

        # 竖向张开度 → 行数（连续值，下面的绘制按需取整）
        open_ratio = max(0.0, self._openness * (1.0 - self._blink))
        open_ratio = max(0.0, open_ratio - self._sleepy * 0.55)
        rows = eye_h * open_ratio

        top = base_row + jump

        self._draw_eye(painter, X(left_x), Y(top), eye_w, rows, X, Y, SZ)
        self._draw_eye(painter, X(right_x), Y(top), eye_w, rows, X, Y, SZ)

        # --- 眉毛 ---
        self._draw_brow(painter, X(left_x), Y(top), eye_w, X, Y)
        self._draw_brow(painter, X(right_x), Y(top), eye_w, X, Y)

        # --- 状态文字 ---
        #
        # ⚠️ 字体必须显式给一个系统里真实存在的中文字体名。只设 pointSize
        # 不设 family 时，Qt 会去它自带的 lib/fonts 目录找字体 —— 那个目录
        # 在 PySide6 中**是空的**（Qt 从 5.x 起不再随包发字体），结果是所有
        # 中文都渲染成空心方块。这里直接点名系统字体，绕开它。
        painter.setPen(QPen(INK, 1))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        font.setFamily("Microsoft YaHei UI")
        painter.setFont(font)
        painter.drawText(
            QRect(0, h - 22, w, 18),
            Qt.AlignCenter,
            mood_label(self._target_mood),
        )

        painter.end()

    # ------------------------------------------------------------------

    def _draw_eye(self, painter, x0, y0, eye_w, rows, X, Y, SZ) -> None:
        """画一只像素眼。

        结构（三层，8bit 眼睛的标准画法）：
          ① 眼球外框：深灰黑实心块 —— 这是"眼眶"；
          ② 眼白：内缩 1 格的白块；
          ③ 瞳孔：**方形**深色块，按注视方向在白区里位移，左上角带 1 格白高光。

        参数里的 `eye_w` / `rows` 都是**格**；`x0` / `y0` 是**设备像素**。
        """
        if rows < 0.6:
            # 闭眼：一条横线（比眼睛窄一点，像眼皮压下来）
            painter.fillRect(x0, y0 + SZ(2.5), SZ(eye_w * 0.92), SZ(1.0), INK)
            return

        # ① 眼球外框 —— **只做 0.5 格的极细描边**。
        # 8bit 眼睛的关键是"眼白大、瞳仁大、框极细"；框一厚就变成方盒子了
        # （我第一版用 1 格描边，8×7 的眼睛看起来就是两个带洞的方块）。
        frame = SZ(0.55)
        painter.fillRect(x0, y0, SZ(eye_w), SZ(rows), INK)

        # ② 眼白：内缩半个格
        inset = 0.55
        inner_x = x0 + frame
        inner_y = y0 + frame
        inner_w = max(0.0, eye_w - inset * 2)
        inner_h = max(0.0, rows - inset * 2)

        if inner_w < 0.6 or inner_h < 0.6:
            return

        painter.fillRect(inner_x, inner_y, SZ(inner_w), SZ(inner_h), SCREEN)

        # ③ 瞳孔：方形，占眼白的高度大头
        p_side = 3.4 * self._pupil_scale
        p_side = max(1.0, min(p_side, inner_w, inner_h))

        # 可位移范围（格）
        room_x = max(0.0, inner_w - p_side)
        room_y = max(0.0, inner_h - p_side)

        # **瞳孔位置必须落在整格上**。用连续值算完再交给 `SZ()` 取整，
        # 会让 x / y 各自四舍五入到不同的格，方块边缘就出现台阶 ——
        # 看起来像"楼梯"，不像像素画。先把位置量化到整格，再换算像素。
        gx = round(self._gaze_x * (room_x / 2.0))
        gy = round(self._gaze_y * (room_y / 2.0))
        gx = max(-room_x / 2.0, min(room_x / 2.0, gx))
        gy = max(-room_y / 2.0, min(room_y / 2.0, gy))

        p_col = round(room_x / 2.0 + gx)       # 相对眼白左边缘的整格偏移
        p_row = round(room_y / 2.0 + gy)

        p_gx = inset + p_col
        p_gy = inset + p_row

        p_pix = SZ(p_side)
        px_pix = x0 + SZ(p_gx)
        py_pix = y0 + SZ(p_gy)

        painter.fillRect(px_pix, py_pix, p_pix, p_pix, INK_DARK)

        # 高光：瞳孔左上角 1 格白点 —— 8bit 眼睛的灵魂
        if p_side >= 2.0:
            painter.fillRect(px_pix, py_pix, SZ(1.0), SZ(1.0), SCREEN)

        # squash：把眼球下缘切掉几行（老式像素表情的"眯眼"）
        cut = self._squash * rows
        if cut >= 1.0:
            cut = min(cut, rows)
            painter.fillRect(
                x0, y0 + SZ(rows - cut), SZ(eye_w), SZ(cut), SCREEN
            )

    def _draw_brow(self, painter, x0, y0, eye_w, X, Y) -> None:
        """眉毛：1 格高的像素条，两端按 brow_tilt 抬升 / 下压。

        `brow`（整体上抬为正）用 _brow 的平滑值；`brow_tilt` 决定倾斜：
        内侧端抬、外侧端压 = 好奇 / 期待；反过来 = 皱眉 / 不满。
        """
        style = MOOD_STYLE.get(self._target_mood, DEFAULT_STYLE)
        tilt = style.get("brow_tilt", 0.0)
        lift = self._brow

        # 眉毛基线：眼球上方 2 格，再按 lift 上下移 1 格
        base = -2.0 - lift

        steps = int(round(eye_w))
        for i in range(steps):
            frac = i / max(1, steps - 1)          # 0 = 内侧, 1 = 外侧
            row_off = tilt * 1.8 * (0.5 - frac)
            painter.fillRect(
                X(i),
                Y(base + row_off),
                max(1, int(round(X(i + 1) - X(i)))),
                max(1, int(round(Y(1) - Y(0)))),
                INK_DARK,
            )

    # ------------------------------------------------------------------
    # 拖动（仅独立悬浮窗时）
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.parent() is None and event.button() == Qt.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if (self._drag_origin is not None
                and event.buttons() & Qt.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_origin)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_origin = None
        event.accept()
