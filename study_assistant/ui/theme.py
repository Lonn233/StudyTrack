"""浅色主题与设计令牌（对齐 ChatGPT 的视觉语言）。

**为什么单独一个模块**：颜色/字号/圆角散落在各个 widget 的
`setStyleSheet(f"...{color}...")` 里，改一次主题要翻五个文件，而且很容易
漏掉某处硬编码 —— 界面就会出现「一处深色、一处浅色」的斑块。所有控件
一律从本模块取令牌，**不允许在别的文件里写死十六进制颜色**。

设计取向（照 ChatGPT 的观感）：
  * 纯白主表面 + 极浅灰页面底 + 中性灰文字，**不用彩色大面积填充**；
  * 边框极细（1px）且对比度很低，靠留白而不是靠框线分区；
  * 强调色只在「当前状态」「选中项」「进度条」出现，其余一律灰阶；
  * 圆角统一 8 / 12，字体统一（中文优先 Microsoft YaHei UI）。

**语义色不是装饰**：`STATE` / `BEHAVIOR` 两套映射把「专注/分心」翻译成
颜色，全项目只有这一份，dashboard 与证据条共用，避免同一个状态在两个
地方显示成两种颜色。
"""

from __future__ import annotations

from ..behavior.states import Behavior, FocusState

# ----------------------------------------------------------------------
# 调色板
# ----------------------------------------------------------------------

# 中性阶（照 ChatGPT 的灰阶取值）
WHITE = "#FFFFFF"           # 主表面：卡片、输入框
CANVAS = "#F7F7F8"          # 页面底：卡片之间的缝隙
SURFACE = "#F1F1F3"         # 次级表面：内嵌小格、hover
SURFACE_ALT = "#EAEAEC"     # 更深的次级：进度条槽
BORDER = "#E3E3E6"          # 极浅边框（默认）
BORDER_STRONG = "#D0D0D5"   # 稍强边框（hover / 分隔）

TEXT = "#0D0D0D"            # 主文字（近黑，不是纯黑）
TEXT_STRONG = "#0D0D0D"     # 强调文字（同主文字，语义区分）
TEXT_MUTED = "#6E6E80"      # 次文字：标签、单位、说明
TEXT_FAINT = "#9B9BA5"      # 更弱：占位、禁用、辅助

ACCENT = "#0D0D0D"          # 主强调 = 近黑（ChatGPT 的按钮就是黑的）
ACCENT_TEXT = "#FFFFFF"     # 强调按钮上的文字

# 语义色（浅底 + 深字，都不用大面积高饱和填充）
GREEN = "#10A37F"           # 绿：专注 / 电脑学习
GREEN_BG = "#E8F6F1"
RED = "#D93025"             # 红：分心 / 玩手机
RED_BG = "#FCEDEC"
AMBER = "#B26A00"           # 琥珀：证据不足 / 待确认
AMBER_BG = "#FAF1E3"
BLUE = "#1F6FEB"            # 蓝：写字 / 暂停
BLUE_BG = "#EAF1FD"
GRAY = "#8E8E99"            # 灰：离席 / 手离开
GRAY_BG = "#F1F1F3"
PURPLE = "#7C5CD6"          # 紫：阅读（与写字区分）
PURPLE_BG = "#F1EDFB"

# ----------------------------------------------------------------------
# 字体
# ----------------------------------------------------------------------

# 中文优先：雅黑 UI 在 Windows 上字形最稳；后面是 mac / linux 的回退
FONT_STACK = '"Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "Segoe UI", sans-serif'
FONT_MONO = '"Cascadia Mono", Consolas, "Courier New", monospace'

FONT_SIZE_BASE = 13
FONT_SIZE_SMALL = 12
FONT_SIZE_TINY = 11

# ----------------------------------------------------------------------
# 尺寸
# ----------------------------------------------------------------------

RADIUS_SM = 6
RADIUS_MD = 8
RADIUS_LG = 12

PAD = 12
GAP = 12

# ----------------------------------------------------------------------
# 状态 / 行为 → 颜色
# ----------------------------------------------------------------------

# 专注状态：唯一一份定义。dashboard 的徽章、证据条、眼睛的高光都从这里取。
STATE_COLORS = {
    FocusState.FOCUSED: (GREEN, GREEN_BG),
    FocusState.DISTRACTED: (RED, RED_BG),
    FocusState.AWAY: (GRAY, GRAY_BG),
    FocusState.UNCERTAIN: (AMBER, AMBER_BG),
    FocusState.PAUSED: (BLUE, BLUE_BG),
}

# 行为：学习类用冷色、分心类用暖色、其余灰阶。仅用于行为条，不用作填充。
BEHAVIOR_COLORS = {
    Behavior.PAPER_STUDY: PURPLE,
    Behavior.COMPUTER_STUDY: GREEN,
    Behavior.PHONE_USE: RED,
    Behavior.FIDGETING: AMBER,
    Behavior.IDLE: GRAY,
    Behavior.HAND_AWAY: TEXT_FAINT,
    Behavior.AWAY: TEXT_FAINT,
    Behavior.UNKNOWN: TEXT_FAINT,
}


def state_color(state) -> str:
    return STATE_COLORS.get(state, (AMBER, AMBER_BG))[0]


def state_bg(state) -> str:
    return STATE_COLORS.get(state, (AMBER, AMBER_BG))[1]


def behavior_color(behavior) -> str:
    return BEHAVIOR_COLORS.get(behavior, TEXT_FAINT)


# ----------------------------------------------------------------------
# 全局样式表
# ----------------------------------------------------------------------

# 说明：**不用** `QWidget { background: ... }` —— 那会把所有子控件都刷成
# 同一个底色，卡片就分不出来了。底色只设在顶层窗口与显式命名的容器上。
QSS = f"""
* {{
    font-family: {FONT_STACK};
    color: {TEXT};
}}

/* 说明：**不要**在 `*` 里设 font-size。QSS 的 font-size 会让
   QLabel.sizeHint 按 QSS 字体测量，而 setFont() 的 QFont 负责实际绘制 ——
   两套字体系统打架，文字就会被裁掉最后一截（时钟/徽章都踩过）。
   要改字号要么走下面按 objectName 的 QSS 规则，要么用 setFont()，
   同一个控件上只能选一种。 */

QWidget#root {{
    background: {CANVAS};
}}

/* ---------- 导航栏 ---------- */
QFrame#navTop {{
    background: {WHITE};
    border-bottom: 1px solid {BORDER};
}}
QFrame#navAction {{
    background: {WHITE};
    border-bottom: 1px solid {BORDER};
}}
QLabel#brand {{
    font-size: 14px;
    font-weight: 600;
    color: {TEXT};
    padding-right: 6px;
}}
QLabel#navSection {{
    color: {TEXT_FAINT};
    font-size: {FONT_SIZE_TINY}px;
    padding-left: 4px;
}}

QToolButton#nav {{
    background: transparent;
    border: none;
    border-radius: {RADIUS_SM}px;
    padding: 5px 10px;
    color: {TEXT_MUTED};
}}
QToolButton#nav:hover {{
    background: {SURFACE};
    color: {TEXT};
}}
QToolButton#nav:pressed {{
    background: {SURFACE_ALT};
}}
QToolButton#nav:checked {{
    background: {SURFACE};
    color: {TEXT};
    font-weight: 600;
}}

/* 导航栏上的菜单（QMenu） */
QMenu {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_MD}px;
    padding: 5px;
}}
QMenu::item {{
    padding: 6px 22px 6px 12px;
    border-radius: {RADIUS_SM}px;
    color: {TEXT};
}}
QMenu::item:selected {{
    background: {SURFACE};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

/* ---------- 控制按钮（开始/暂停/重启/结束） ---------- */
QPushButton#primary {{
    background: {ACCENT};
    color: {ACCENT_TEXT};
    border: 1px solid {ACCENT};
    border-radius: {RADIUS_MD}px;
    padding: 7px 18px;
    font-weight: 600;
}}
QPushButton#primary:hover  {{ background: #2A2A2A; border-color: #2A2A2A; }}
QPushButton#primary:pressed{{ background: #000000; border-color: #000000; }}
QPushButton#primary:disabled {{
    background: {SURFACE_ALT}; color: {TEXT_FAINT}; border-color: {SURFACE_ALT};
}}

QPushButton#ghost {{
    background: {WHITE};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_MD}px;
    padding: 7px 16px;
}}
QPushButton#ghost:hover  {{ background: {SURFACE}; }}
QPushButton#ghost:pressed{{ background: {SURFACE_ALT}; }}
QPushButton#ghost:checked{{
    background: {SURFACE};
    border-color: {TEXT};
    font-weight: 600;
}}
QPushButton#ghost:disabled {{
    color: {TEXT_FAINT}; border-color: {BORDER};
}}

/* ---------- 卡片 ---------- */
QFrame#card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LG}px;
}}
QFrame#inset {{
    background: {SURFACE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}
QLabel#cardTitle {{
    color: {TEXT_MUTED};
    font-size: {FONT_SIZE_SMALL}px;
}}
QLabel#sectionLabel {{
    color: {TEXT_MUTED};
    font-size: {FONT_SIZE_TINY}px;
    font-weight: 600;
    letter-spacing: 0.6px;
}}
QLabel#metricLabel {{
    color: {TEXT_MUTED};
    font-size: {FONT_SIZE_SMALL}px;
}}
QLabel#metricValue {{
    color: {TEXT_STRONG};
    font-size: 20px;
    font-weight: 600;
}}
QLabel#metricSub {{
    color: {TEXT_FAINT};
    font-size: {FONT_SIZE_TINY}px;
}}
QLabel#muted {{
    color: {TEXT_MUTED};
    font-size: {FONT_SIZE_SMALL}px;
}}

/* ---------- 画面 ---------- */
QLabel#video {{
    background: #EDEDEF;
    border: none;
    border-radius: {RADIUS_MD}px;
    color: {TEXT_FAINT};
}}
QFrame#videoFrame {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LG}px;
}}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background: {SURFACE_ALT};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {GRAY};
    border-radius: 4px;
}}

/* ---------- 事件流 ---------- */
QListWidget {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_MD}px;
    color: {TEXT};
    outline: none;
}}
QListWidget::item {{
    padding: 5px 8px;
    border-bottom: 1px solid {SURFACE};
    color: {TEXT};
}}
QListWidget::item:selected {{
    background: {SURFACE};
    color: {TEXT};
}}

/* ---------- 滚动条：细、无按钮 ---------- */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG}; border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG}; border-radius: 5px; min-width: 28px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QToolTip {{
    background: {TEXT};
    color: {ACCENT_TEXT};
    border: none;
    border-radius: {RADIUS_SM}px;
    padding: 5px 8px;
}}
"""


def apply(app) -> None:
    """给 QApplication 套全局样式表。"""
    app.setStyleSheet(QSS)
