"""验收测试：行为判定 / 状态机 / 会话统计 / 提醒分级。

    python tests/acceptance_test.py            # 跑全部
    python tests/acceptance_test.py A C E      # 只跑指定场景

不需要摄像头、不需要模型、不需要图形界面 —— 全部通过
`tests/synthetic.py` 构造的合成手 / 合成物体驱动真实的 `BehaviorCore`。
报告写到 out/acceptance_report.txt。

场景：
  A 写字 → 专注          B 阅读 → 专注        C 玩手机 → 分心
  D 中短期手机不打扰      E 离席宽容期与确认    F 电脑学习（物体+打字动作）→ 专注
  G 小动作 → 分心        H 状态滞后            I 统计 / 里程碑 / 数据库
  J 提醒分级与冷却        K 物体漏检容忍        L 契约与不变量
  M 人在画面内但手拍不到 → 不得判为离席
  N 手机位置先验不得单独定罪（看书/打字不得被判玩手机）
  O 写字 / 阅读 的区分度（沿运动能量轴扫描）
"""

from __future__ import annotations

import sys
import os
import math
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import synthetic as S  # noqa: E402

from study_assistant.behavior.focus_score import (  # noqa: E402
    compute_focus_score,
    format_duration,
)
from study_assistant.behavior.states import (  # noqa: E402
    BEHAVIOR_TO_FOCUS,
    Behavior,
    FocusState,
)

LOG: list[str] = []
RESULTS: list[tuple[str, bool, str]] = []


def say(msg: str = "") -> None:
    LOG.append(str(msg))
    print(msg)


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def flush() -> None:
    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "acceptance_report.txt").write_text("\n".join(LOG), encoding="utf-8")


def section(title: str) -> None:
    say()
    say("=" * 74)
    say(title)
    say("=" * 74)


def hist(pairs) -> str:
    return str([(v.value, round(d, 1)) for v, d in pairs])


# --- 常用物体组合 ------------------------------------------------------


def phone_object():
    cx, cy = S.ROI_CENTER["phone"]
    return [S.make_object("phone", cx=cx, cy=cy, w=0.10, h=0.16)]


def book_object():
    cx, cy = S.ROI_CENTER["paper"]
    return [S.make_object("book", cx=cx, cy=cy, w=0.28, h=0.18)]


def desk_objects():
    cx, cy = S.ROI_CENTER["keyboard"]
    return [
        S.make_object("keyboard", cx=cx, cy=cy, w=0.34, h=0.14),
        S.make_object("laptop", cx=cx, cy=cy - 0.06, w=0.30, h=0.12),
    ]


# ======================================================================
# A. 纸质学习（书写动作）→ 专注
# ======================================================================


def scenario_a() -> None:
    section("A. 纸质学习-书写（手在书写区 + 持续小幅运动）→ PAPER_STUDY / FOCUSED")

    scn = S.new_scenario()
    scn.step(3, hands=[S.make_hand("right", S.ROI_CENTER["paper"])])
    scn.wiggle(12.0, "paper", amplitude=0.012, frequency=2.5)

    behavior = scn.behaviors[-1][1]
    first_paper = scn.first_time_in_behavior(Behavior.PAPER_STUDY)
    first_focus = scn.first_time_in_state(FocusState.FOCUSED)

    check("最终行为为 PAPER_STUDY", behavior is Behavior.PAPER_STUDY, f"实际 {behavior.value}")
    check(
        "判定为 FOCUSED",
        scn.ever_in_state(FocusState.FOCUSED),
        f"首次 {first_focus:.1f}s" if first_focus is not None else "从未进入",
    )
    check(
        "在 8 秒内收敛到 PAPER_STUDY",
        first_paper is not None and first_paper <= 8.0,
        f"{first_paper:.1f}s" if first_paper is not None else "未进入",
    )
    check(
        "从未误判为分心",
        not scn.ever_in_state(FocusState.DISTRACTED),
        hist(scn.state_history()),
    )

    stats = scn.core.snapshot_stats()
    check(
        "有效专注时长被正确累计",
        stats.focused_seconds >= 5.0,
        f"focused={stats.focused_seconds:.1f}s / total={stats.total_seconds:.1f}s",
    )


# ======================================================================
# B. 纸质学习（静止阅读）→ 专注
# ======================================================================


def scenario_b() -> None:
    section("B. 纸质学习-阅读（手压着书、基本不动）→ PAPER_STUDY / FOCUSED")

    scn = S.new_scenario()
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["paper"])], objects=book_object())
    scn.hold_still(14.0, "paper", objects=book_object())

    behavior = scn.behaviors[-1][1]
    first_paper = scn.first_time_in_behavior(Behavior.PAPER_STUDY)

    check("最终行为为 PAPER_STUDY", behavior is Behavior.PAPER_STUDY, f"实际 {behavior.value}")
    check(
        "判定为 FOCUSED",
        scn.ever_in_state(FocusState.FOCUSED),
        f"首次 {scn.first_time_in_state(FocusState.FOCUSED)}",
    )
    check(
        "在 6 秒内判定为 PAPER_STUDY（静止阅读支路生效）",
        first_paper is not None and first_paper <= 6.0,
        f"首次 {first_paper:.1f}s" if first_paper is not None else "未进入",
    )


# ======================================================================
# C. 玩手机 → 分心
# ======================================================================


def scenario_c() -> None:
    section("C. 玩手机（手持续接触手机）→ PHONE_USE / DISTRACTED")

    scn = S.new_scenario()
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])], objects=phone_object())
    scn.hold_still(15.0, "phone", objects=phone_object())

    behavior = scn.behaviors[-1][1]
    first_phone = scn.first_time_in_behavior(Behavior.PHONE_USE)
    first_distract = scn.first_time_in_state(FocusState.DISTRACTED)

    check("最终行为为 PHONE_USE", behavior is Behavior.PHONE_USE, f"实际 {behavior.value}")
    check(
        "判定为 DISTRACTED",
        scn.ever_in_state(FocusState.DISTRACTED),
        f"首次 {first_distract:.1f}s" if first_distract is not None else "从未进入",
    )
    check(
        "PHONE_USE 需持续确认（>=2.5s 才确认）",
        first_phone is not None and first_phone >= 2.5,
        f"{first_phone:.1f}s" if first_phone is not None else "未进入",
    )
    check("记录了分心事件", len(scn.events_of("distraction")) >= 1,
          f"{len(scn.events_of('distraction'))} 条")
    check(
        "分心期间未被累计为有效专注",
        scn.core.snapshot_stats().focused_seconds < 3.0,
        f"focused={scn.core.snapshot_stats().focused_seconds:.1f}s",
    )


# ======================================================================
# D. 中短期看手机不打扰
# ======================================================================


def scenario_d() -> None:
    section("D. 中短期手机使用（12 秒后放下）→ 分心按等级处理，不产生打扰")

    scn = S.new_scenario()

    scn.wiggle(10.0, "paper", amplitude=0.012)
    check("先进入 FOCUSED", scn.ever_in_state(FocusState.FOCUSED))

    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])], objects=phone_object())
    scn.hold_still(12.0, "phone", objects=phone_object())

    scn.wiggle(10.0, "paper", amplitude=0.012)

    reminders = scn.events_of("distraction_reminder")
    soft_or_worse = [e for e in reminders if e.level >= 2]

    check(
        "未产生轻提醒及以上（level>=2）",
        len(soft_or_worse) == 0,
        f"{[(e.level, round(e.duration, 1)) for e in reminders]}",
    )
    check(
        "分心过程被记录（level 1 允许）",
        len(scn.events_of("distraction")) >= 1,
        f"{len(scn.events_of('distraction'))} 条",
    )
    check(
        "记录了一次「回到学习」",
        len(scn.events_of("recovery")) >= 1,
        f"{len(scn.events_of('recovery'))} 条",
    )
    check(
        "重新进入专注状态",
        scn.ever_in_state(FocusState.FOCUSED) and scn.behaviors[-1][1] is Behavior.PAPER_STUDY,
        f"最终行为 {scn.behaviors[-1][1].value}",
    )


# ======================================================================
# E. 离席：宽容期与确认
# ======================================================================


def scenario_e() -> None:
    section("E. 离席：手消失 3 秒内保持原状态，超过 8 秒才确认 AWAY")

    scn = S.new_scenario()
    scn.wiggle(12.0, "paper", amplitude=0.012)

    base = scn.elapsed()
    state_before = scn.state_at(base - 0.5)

    check(
        "离席前已处于 FOCUSED",
        state_before is FocusState.FOCUSED,
        f"实际 {state_before.value if state_before else None}",
    )

    scn.no_hands(2.0)
    after_2 = scn.state_at(base + 2.0)
    check(
        "手消失 2s：保持 FOCUSED（宽容期生效）",
        after_2 is FocusState.FOCUSED,
        f"实际 {after_2.value if after_2 else None}",
    )

    scn.no_hands(4.0)
    after_6 = scn.state_at(base + 6.0)
    check(
        "手消失 6s：仍未宣布 AWAY",
        after_6 is not FocusState.AWAY,
        f"实际 {after_6.value if after_6 else None}",
    )

    scn.no_hands(6.0)
    after_12 = scn.state_at(base + 12.0)
    check(
        "手消失 12s：确认 AWAY",
        after_12 is FocusState.AWAY,
        f"实际 {after_12.value if after_12 else None}",
    )

    first_away = scn.first_time_in_state(FocusState.AWAY, after=base)
    check(
        "AWAY 出现在离开后 8~11 秒之间（不重复计时）",
        first_away is not None and (base + 7.5) <= first_away <= (base + 11.5),
        f"首次 {first_away:.1f}s（离开起点 {base:.1f}s）" if first_away else "未进入",
    )

    check("记录了离席事件", len(scn.events_of("away_start")) >= 1,
          f"{len(scn.events_of('away_start'))} 条")

    stats = scn.core.snapshot_stats()
    check(
        "离席时长被单独统计（不混入分心）",
        stats.away_seconds >= 2.0 and stats.distracted_seconds < 1.0,
        f"away={stats.away_seconds:.1f}s distracted={stats.distracted_seconds:.1f}s",
    )

    # 回到座位
    scn.wiggle(10.0, "paper", amplitude=0.012)
    check(
        "回到座位后恢复 FOCUSED",
        scn.state_at(scn.elapsed()) is FocusState.FOCUSED,
        f"实际 {scn.state_at(scn.elapsed()).value}",
    )
    check("记录了回座事件", len(scn.events_of("away_end")) >= 1)


# ======================================================================
# F. 电脑学习 → 专注
# ======================================================================


def comp_raw(scn) -> float:
    """取窗口内 COMPUTER_STUDY 的原始分（用于比较"有/无键盘物体"）。"""
    est = getattr(scn.core.last, "estimate", None)
    scores = getattr(est, "scores", {}) or {}
    return float(scores.get(Behavior.COMPUTER_STUDY, 0.0))


def scenario_f() -> None:
    section("F. 电脑学习（检出键盘物体 + 打字动作）→ COMPUTER_STUDY / FOCUSED")

    # ---- 正向：检出 keyboard 物体 + 手与它交互 + 真正的键击脉冲 ----
    scn = S.new_scenario()
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["keyboard"])],
             objects=desk_objects())
    scn.type_burst(14.0, "keyboard", objects=desk_objects())

    behavior = scn.behaviors[-1][1]

    check("最终行为为 COMPUTER_STUDY", behavior is Behavior.COMPUTER_STUDY,
          f"实际 {behavior.value}")
    check(
        "判定为 FOCUSED",
        scn.ever_in_state(FocusState.FOCUSED),
        f"首次 {scn.first_time_in_state(FocusState.FOCUSED)}",
    )
    check(
        "未误判为纸质学习（键盘区与书写区默认不重叠）",
        not scn.ever_in_behavior(Behavior.PAPER_STUDY),
        hist(scn.behavior_history()),
    )
    check(
        "从未被判为分心",
        not scn.ever_in_state(FocusState.DISTRACTED),
        hist(scn.state_history()),
    )
    check(
        "打字签名成立时不再同时点亮小动作",
        not scn.ever_in_behavior(Behavior.FIDGETING),
        hist(scn.behavior_history()),
    )

    # 证据文字是**真机标定 typing_* 阈值的唯一读数来源**。曾经「实时证据」
    # 面板读一个不存在的字段、永远是「—」（死面板），所以这里守住它真的
    # 有内容，而不是只守住"行为判对了"。
    comp_ev = (getattr(scn.core.last.estimate, "evidence", None) or {}).get(
        Behavior.COMPUTER_STUDY, []
    )
    joined = " | ".join(comp_ev)
    check(
        "实时证据里能看到打字签名四项读数（标定阈值靠它）",
        "打字签名" in joined and "方向反转" in joined and "脉冲波动" in joined,
        joined[:110],
    )

    # ==================================================================
    # 对照组：位置命中**不能**单独判定电脑学习
    # 这是用户报告的原始问题 —— 手在键盘区里晃，界面同时点亮
    # 「电脑学习」和「小动作」。下面每条都必须在修复前会失败。
    # ==================================================================

    # ---- 对照 1：键盘区内大幅乱挥（一个键盘物体都没检出）----
    c1 = S.new_scenario()
    c1.large_wave(10.0, center=S.ROI_CENTER["keyboard"])
    check(
        "对照：键盘区内大幅乱挥 → 不判 COMPUTER_STUDY",
        not c1.ever_in_behavior(Behavior.COMPUTER_STUDY),
        hist(c1.behavior_history()),
    )
    check(
        "对照：键盘区内大幅乱挥 → 判 FIDGETING（位置不能让小动作消失）",
        c1.ever_in_behavior(Behavior.FIDGETING),
        hist(c1.behavior_history()),
    )

    # ---- 对照 2：手压着键盘完全不动（连键盘物体都检出了）----
    keys = [S.make_object("keyboard", cx=S.ROI_CENTER["keyboard"][0],
                          cy=S.ROI_CENTER["keyboard"][1], w=0.34, h=0.14)]
    c2 = S.new_scenario()
    c2.hold_still(12.0, "keyboard", objects=keys)
    check(
        "对照：手压着键盘静止（已检出键盘物体）→ 仍不判 COMPUTER_STUDY"
        "（动作签名是必要条件）",
        not c2.ever_in_behavior(Behavior.COMPUTER_STUDY),
        hist(c2.behavior_history()),
    )
    check(
        "对照：手压着键盘静止 → 也不该被判成 FIDGETING（要真有动作才算小动作）",
        not c2.ever_in_behavior(Behavior.FIDGETING),
        hist(c2.behavior_history()),
    )
    idle_ev = (getattr(c2.core.last.estimate, "evidence", None) or {}).get(
        Behavior.IDLE, []
    )
    check(
        "对照：静止压键盘时证据里说明「打字签名不足」，不是只显示静止",
        any("打字签名" in line for line in idle_ev),
        " | ".join(idle_ev)[:110],
    )

    # ---- 对照 3：连续小幅划动（写字式平滑正弦，不是键击脉冲）----
    c3 = S.new_scenario()
    c3.wiggle(10.0, "keyboard", amplitude=0.012, frequency=3.5,
              objects=desk_objects())
    check(
        "对照：平滑连续划动（非脉冲）→ 不判 COMPUTER_STUDY",
        not c3.ever_in_behavior(Behavior.COMPUTER_STUDY),
        hist(c3.behavior_history()),
    )

    # ---- 对照 4：位置框**故意标错**到画面另一角 ----
    # 4a 有键盘物体 → 检测通路必须能独立把事情办成；
    # 4b 没键盘物体 → 位置通路已失效，就不该再判电脑学习。
    # 两条合起来证明「位置」与「检测」是两条**独立可分离**的通路，
    # 而不是一条把另一条吞掉。
    from study_assistant.desk.roi_manager import ROI, Calibration

    bad_calib = Calibration()
    bad_calib.rois["keyboard"] = ROI("keyboard", 0.02, 0.08, 0.18, 0.22)
    c4 = S.new_scenario(calibration=bad_calib)
    c4.type_burst(12.0, "keyboard", objects=desk_objects())
    check(
        "对照：位置框标错到画面角落，但检出键盘物体 + 打字 → 仍判 COMPUTER_STUDY",
        c4.ever_in_behavior(Behavior.COMPUTER_STUDY),
        hist(c4.behavior_history()),
    )

    c4b = S.new_scenario(calibration=bad_calib)
    c4b.type_burst(12.0, "keyboard", objects=[])
    check(
        "对照：位置框标错 + 没检出键盘物体 → 不判 COMPUTER_STUDY"
        "（两条通路确实独立）",
        not c4b.ever_in_behavior(Behavior.COMPUTER_STUDY),
        hist(c4b.behavior_history()),
    )

    # 这一条是给用户看的：动作明明像打字，却因为位置/物体都确认不了而没判
    # 电脑学习时，证据里必须说清楚原因 —— 否则面板一片空白，用户只会得到
    # 「我明明在敲键盘，它检测不出来」却无从下手。
    comp_ev_c4b = (getattr(c4b.core.last.estimate, "evidence", None) or {}).get(
        Behavior.COMPUTER_STUDY, []
    )
    check(
        "对照：动作像打字但位置无法确认 → 证据如实说明原因（不再是空白）",
        any("位置无法确认" in s for s in comp_ev_c4b),
        f"{comp_ev_c4b}",
    )

    # ---- 对照 5：同样的打字动作但一个键盘物体都没检出 ----
    # 位置兜底仍然生效（否则摄像头拍不到键盘的场景会彻底失效），
    # 但必须被折价 —— 这条证明「物体检测」真的能改变输出，不是装饰。
    c5 = S.new_scenario()
    c5.type_burst(12.0, "keyboard", objects=[])
    check(
        "对照：未检出键盘物体时位置兜底仍生效（不把功能做没了）",
        c5.ever_in_behavior(Behavior.COMPUTER_STUDY),
        hist(c5.behavior_history()),
    )
    check(
        "对照：无键盘物体的电脑学习分数被折价（位置 ≠ 等价证据）",
        comp_raw(c5) < comp_raw(scn),
        f"仅位置 {comp_raw(c5):.3f} < 有物体 {comp_raw(scn):.3f}",
    )


# ======================================================================
# G. 小动作 → 分心
# ======================================================================


def scenario_g() -> None:
    section("G. 无目的大幅动作（手在行为区域之外乱晃）→ FIDGETING / DISTRACTED")

    scn = S.new_scenario()
    scn.large_wave(18.0)

    behavior = scn.behaviors[-1][1]
    first_fid = scn.first_time_in_behavior(Behavior.FIDGETING)
    first_distract = scn.first_time_in_state(FocusState.DISTRACTED)

    check("最终行为为 FIDGETING", behavior is Behavior.FIDGETING, f"实际 {behavior.value}")
    check(
        "判定为 DISTRACTED",
        scn.ever_in_state(FocusState.DISTRACTED),
        f"首次 {first_distract:.1f}s" if first_distract is not None else "从未进入",
    )
    check(
        "FIDGETING 需 >=3s 持续才确认",
        first_fid is not None and first_fid >= 3.0,
        f"{first_fid:.1f}s" if first_fid is not None else "未进入",
    )
    check(
        "未误判为纸质学习（行为区域判定生效）",
        not scn.ever_in_behavior(Behavior.PAPER_STUDY),
        hist(scn.behavior_history()),
    )


# ======================================================================
# H. 状态滞后
# ======================================================================


def scenario_h() -> None:
    section("H. 滞后：分心证据只持续 2 秒时不应切换状态")

    scn = S.new_scenario()
    scn.wiggle(12.0, "paper", amplitude=0.012)
    check("先进入 FOCUSED", scn.ever_in_state(FocusState.FOCUSED))

    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])], objects=phone_object())
    scn.hold_still(2.0, "phone", objects=phone_object())
    scn.wiggle(10.0, "paper", amplitude=0.012)

    check(
        "全程从未进入 DISTRACTED",
        not scn.ever_in_state(FocusState.DISTRACTED),
        hist(scn.state_history()),
    )
    check(
        "短暂干扰后仍为 PAPER_STUDY",
        scn.behaviors[-1][1] is Behavior.PAPER_STUDY,
        f"实际 {scn.behaviors[-1][1].value}",
    )
    check(
        "未产生任何打扰",
        len(scn.events_of("distraction_reminder")) == 0,
        f"{len(scn.events_of('distraction_reminder'))} 条",
    )


# ======================================================================
# I. 统计 / 里程碑 / 数据库
# ======================================================================


def scenario_i() -> None:
    section("I. 统计、专注分、里程碑与 SQLite 持久化")

    # ---- I-1 评分公式 ----
    score = compute_focus_score(
        {
            FocusState.FOCUSED: 2700.0,
            FocusState.DISTRACTED: 300.0,
            FocusState.AWAY: 0.0,
            FocusState.UNCERTAIN: 0.0,
        },
        transitions=5,
    )
    check("专注分：90% 专注、无离席、切换少 → 86 分左右", 80 <= score <= 92, f"{score:.1f}")

    heavy = compute_focus_score(
        {
            FocusState.FOCUSED: 900.0,
            FocusState.DISTRACTED: 2400.0,
            FocusState.AWAY: 600.0,
            FocusState.UNCERTAIN: 100.0,
        },
        transitions=45,
    )
    check("专注分：大量分心 → 低分", heavy < 40, f"{heavy:.1f}")

    check(
        "样本过短（<5s）时不做判断",
        compute_focus_score({FocusState.UNCERTAIN: 2.0}) == 0.0,
    )

    check(
        "时长格式化可读",
        format_duration(3661) == "1h 01m 01s" and format_duration(65) == "1m 05s",
        f"{format_duration(3661)} / {format_duration(65)}",
    )

    # ---- I-2 里程碑 ----
    scn = S.new_scenario()
    scn.wiggle(6.0, "paper", amplitude=0.012)

    scn.core.session.timer.focused_seconds = 30 * 60 - 1.0
    scn.wiggle(3.0, "paper", amplitude=0.012)

    milestones = scn.events_of("milestone")
    check(
        "达到 30 分钟有效专注时触发里程碑",
        any(e.minutes == 30 for e in milestones),
        f"{[(e.minutes, e.message) for e in milestones]}",
    )

    # ---- I-3 数据库 ----
    # 注意：不要用 tempfile.TemporaryDirectory —— SQLite 的 WAL 文件在
    # Windows 上释放得比 Python 对象销毁慢一拍，退出时的递归删除会抛
    # PermissionError（WinError 32）。
    from study_assistant.storage.database import Database

    db_dir = ROOT / "out" / "test_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "test.sqlite3"

    for suffix in ("", "-wal", "-shm"):
        stale = Path(str(db_path) + suffix)
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass

    db = Database(db_path)
    check("数据库可创建并初始化", db.enabled and db_path.exists(), str(db_path))

    scn_db = S.new_scenario(database=db)
    scn_db.wiggle(12.0, "paper", amplitude=0.012)

    # 制造一次分心，确保 events 表有内容
    scn_db.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])],
                objects=phone_object())
    scn_db.hold_still(14.0, "phone", objects=phone_object())
    scn_db.wiggle(10.0, "paper", amplitude=0.012)

    scn_db.core.session.timer.focused_seconds = 30 * 60
    scn_db.wiggle(2.0, "paper", amplitude=0.012)

    scn_db.core.end()
    db.close()
    del db, scn_db

    db2 = Database(db_path)
    sessions = db2.recent_sessions(10)
    events = db2.recent_events(200)
    samples = db2.samples_for_session(sessions[0][0]) if sessions else []
    totals = db2.totals()
    daily = db2.daily_focus(7)
    db2.close()
    del db2

    check("会话写入数据库", len(sessions) >= 1, f"{len(sessions)} 行")
    check(
        "会话的统计字段非空",
        bool(sessions) and sessions[0][3] > 0,
        f"total_seconds={sessions[0][3] if sessions else None}",
    )
    check("行为采样写入数据库", len(samples) >= 2, f"{len(samples)} 条")
    check("事件写入数据库", len(events) >= 3, f"{len(events)} 条")
    check("汇总查询可用", totals["sessions"] >= 1, str(totals))
    check("按日聚合查询可用", len(daily) >= 1, str(daily))
    check(
        "专注分写入数据库",
        bool(sessions) and sessions[0][7] > 0,
        f"focus_score={sessions[0][7] if sessions else None}",
    )


# ======================================================================
# J. 提醒分级与冷却
# ======================================================================


def scenario_j() -> None:
    section("J. 提醒分级：10s 表情 / 20s 轻提示 / 30s 强提醒，且受冷却约束")

    config = S.load_test_config({"notifications": {"cooldown_seconds": 5}})

    scn = S.new_scenario(config=config)
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])], objects=phone_object())
    scn.hold_still(50.0, "phone", objects=phone_object())

    reminders = scn.events_of("distraction_reminder")
    levels = [e.level for e in reminders]
    durations = [round(e.duration, 1) for e in reminders]

    check("出现 level 1（10s 表情级）", 1 in levels, f"levels={levels} durations={durations}")
    check("出现 level 2（20s 轻提示）", 2 in levels, f"levels={levels}")
    check("出现 level 3（30s 强提醒）", 3 in levels, f"levels={levels}")
    check("提醒等级单调不降", levels == sorted(levels), f"levels={levels}")

    first = durations[0] if durations else 0.0
    check("首次提醒发生在 10 秒左右", 9.0 <= first <= 14.0, f"{first}s")

    # --- 冷却抑制 ---
    config_cool = S.load_test_config({"notifications": {"cooldown_seconds": 600}})
    scn2 = S.new_scenario(config=config_cool)
    scn2.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])], objects=phone_object())
    scn2.hold_still(50.0, "phone", objects=phone_object())

    cool_levels = scn2.event_levels("distraction_reminder")
    check("长冷却时提醒被抑制", len(cool_levels) <= 1, f"{len(cool_levels)} 条 {cool_levels}")

    # --- 关闭通知 ---
    config_off = S.load_test_config({"notifications": {"enabled": False}})
    scn3 = S.new_scenario(config=config_off)
    scn3.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])], objects=phone_object())
    scn3.hold_still(40.0, "phone", objects=phone_object())

    check(
        "关闭通知后不再提醒（但状态判定照旧）",
        len(scn3.events_of("distraction_reminder")) == 0
        and scn3.ever_in_state(FocusState.DISTRACTED),
        f"reminders={len(scn3.events_of('distraction_reminder'))} "
        f"distracted={scn3.ever_in_state(FocusState.DISTRACTED)}",
    )


# ======================================================================
# K. 物体滑行 / 漏检容忍
# ======================================================================


def scenario_k() -> None:
    section("K. 物体漏检容忍：短暂丢失不会让「手机在场」瞬间归零")

    from study_assistant.features.object_features import ObjectPresenceTracker
    from study_assistant.vision.object_tracker import ObjectTracker

    tracker = ObjectTracker()
    presence = ObjectPresenceTracker()
    det = phone_object()

    t = 100.0
    for _ in range(90):                      # 连续 3 秒
        t += 1.0 / 30.0
        presence.update(tracker.update(det, t).objects, t)

    check("持续检测时在场时长为正", presence.duration("phone") > 2.5,
          f"{presence.duration('phone'):.2f}s")

    for _ in range(12):                      # 漏检 0.4 秒
        t += 1.0 / 30.0
        presence.update(tracker.update([], t).objects, t)

    check(
        "漏检 0.4s 内仍认为在场（容忍窗口）",
        presence.is_present("phone"),
        f"present={presence.is_present('phone')} gone={presence.gone_for('phone'):.2f}s",
    )
    check(
        "漏检期间轨迹仍在（滑行位置）",
        tracker.count >= 1 and any(o.stale for o in tracker.snapshot(t).objects),
        f"轨迹数 {tracker.count} stale="
        f"{[o.stale for o in tracker.snapshot(t).objects]}",
    )

    for _ in range(90):                      # 再丢 3 秒
        t += 1.0 / 30.0
        presence.update(tracker.update([], t).objects, t)

    check(
        "漏检 3s 后判定为不在场",
        not presence.is_present("phone"),
        f"gone={presence.gone_for('phone'):.2f}s 轨迹数={tracker.count}",
    )

    # --- 手机离场后不应再判 PHONE_USE ---
    scn = S.new_scenario()
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])], objects=phone_object())
    scn.hold_still(10.0, "phone", objects=phone_object())

    check("手机在手时判为 PHONE_USE", scn.ever_in_behavior(Behavior.PHONE_USE))

    # 手移到书写区，手机从画面消失
    scn.wiggle(16.0, "paper", amplitude=0.012, objects=[])

    check(
        "手机离开后不再判 PHONE_USE",
        scn.behaviors[-1][1] is Behavior.PAPER_STUDY,
        f"实际 {scn.behaviors[-1][1].value}",
    )
    check(
        "手机离开后回到专注",
        scn.state_at(scn.elapsed()) is FocusState.FOCUSED,
        f"实际 {scn.state_at(scn.elapsed()).value}",
    )

    # --- 手指尖判定：手不碰手机但就在旁边 ---
    from study_assistant.behavior.hand_object_engine import HandObjectEngine
    from study_assistant.features.interaction_features import measure

    far_hand = S.make_hand("right", (0.62, 0.30), size=0.20)
    phone = S.make_object("phone", cx=0.11, cy=0.70, w=0.10, h=0.16)

    m = measure(far_hand, phone, 640, 480, 90.0, 0.05)
    check(
        "远处的手不判为与手机接触",
        not m.contact,
        f"distance={m.distance_px:.0f}px overlap={m.overlap_ratio:.2f}",
    )

    near_hand = S.make_hand("right", (0.13, 0.70), size=0.20)
    m2 = measure(near_hand, phone, 640, 480, 90.0, 0.05)
    check(
        "压在手机上的手判为接触",
        m2.contact and m2.distance_px == 0.0,
        f"distance={m2.distance_px:.0f}px overlap={m2.overlap_ratio:.2f}",
    )
    del HandObjectEngine


# ======================================================================
# L. 契约检查
# ======================================================================


def scenario_l() -> None:
    section("L. 契约与不变量")

    expected_behaviors = {
        "PAPER_STUDY", "COMPUTER_STUDY", "PHONE_USE",
        "FIDGETING", "IDLE", "HAND_AWAY", "AWAY", "UNKNOWN",
    }
    actual = {b.value for b in Behavior}
    check("行为枚举完整（8 种）", actual == expected_behaviors,
          f"缺 {expected_behaviors - actual} 多 {actual - expected_behaviors}")

    expected_states = {"FOCUSED", "DISTRACTED", "AWAY", "UNCERTAIN", "PAUSED"}
    actual_states = {s.value for s in FocusState}
    check("状态枚举完整（5 种）", actual_states == expected_states,
          f"缺 {expected_states - actual_states}")

    check("每个行为都有状态映射", all(b in BEHAVIOR_TO_FOCUS for b in Behavior),
          f"缺 {[b.value for b in Behavior if b not in BEHAVIOR_TO_FOCUS]}")
    check(
        "分心行为 → DISTRACTED",
        BEHAVIOR_TO_FOCUS[Behavior.PHONE_USE] is FocusState.DISTRACTED
        and BEHAVIOR_TO_FOCUS[Behavior.FIDGETING] is FocusState.DISTRACTED,
    )
    check(
        "两种学习行为 → FOCUSED",
        all(BEHAVIOR_TO_FOCUS[b] is FocusState.FOCUSED
            for b in (Behavior.PAPER_STUDY, Behavior.COMPUTER_STUDY)),
    )

    # ---- ROI 不重叠 ----
    from study_assistant.desk.roi_manager import Calibration

    calib = Calibration()

    def area_overlap(a, b) -> float:
        x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
        x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    paper, keyboard, phone = calib.get("paper"), calib.get("keyboard"), calib.get("phone")

    check("paper 与 keyboard 默认不重叠", area_overlap(paper, keyboard) == 0.0,
          f"{area_overlap(paper, keyboard):.4f}")
    check("paper 与 phone 默认不重叠", area_overlap(paper, phone) == 0.0)
    check("keyboard 与 phone 默认不重叠", area_overlap(keyboard, phone) == 0.0)
    check("共 7 个 ROI", len(calib.rois) == 7, f"{len(calib.rois)}")

    # 只有参与行为判定的区域之间才必须互不交集：一个点同时落进 paper 和
    # keyboard，就等于"手既在写字又在打字"，两个行为都会拿满分。
    stray = []
    for name in S.BEHAVIOR_ROIS:
        cx, cy = S.ROI_CENTER[name]
        for other in S.BEHAVIOR_ROIS:
            if other == name:
                continue
            if calib.contains(other, cx, cy):
                stray.append(f"{name}@{other}")
    check("行为区域的测试坐标互不越界", not stray, f"越界 {stray}")

    # ---- 每帧输出结构 ----
    scn = S.new_scenario()
    scn.wiggle(4.0, "paper", amplitude=0.012)

    update = scn.core.last
    check("每帧都返回 snapshot", update.snapshot is not None)
    check("snapshot 状态是合法枚举", isinstance(update.snapshot.state, FocusState),
          str(update.snapshot.state))
    check(
        "interaction report 字段齐全",
        all(hasattr(update.interactions, name) for name in (
            "hand_in_study", "hand_in_paper_roi", "hand_in_keyboard_roi",
            "hand_in_phone_roi", "hand_on_desk", "phone_contact",
            "paper_contact", "keyboard_contact", "measures",
            # 键盘两条通路必须分开报，否则位置会把检测吞掉
            "keyboard_object_contact", "keyboard_object_duration",
            "keyboard_roi_contact", "keyboard_roi_duration",
            # 手机同理：位置命中与物体检测必须分开报，否则位置单独就能定罪
            "phone_object_contact", "phone_object_duration",
            "phone_roi_contact", "phone_roi_duration",
        )),
    )

    # ---- 手部轨迹窗口 ----
    from study_assistant.features.hand_features import HandTracker

    check("手部运动窗口为 1/3/5 秒", HandTracker().windows == (1.0, 3.0, 5.0),
          str(HandTracker().windows))

    # ---- 板卡分频 ----
    import os

    from study_assistant.vision.board_profile import PROFILES, intervals_for_board

    check("板卡档位齐全", {"desktop", "pi4", "pi5", "pi_zero"} <= set(PROFILES))

    pi4 = intervals_for_board("pi4")
    desktop = intervals_for_board("desktop")
    check("Pi 4 档位明显比桌面慢", pi4["hands"] >= 150 and pi4["objects"] >= desktop["objects"],
          f"pi4={pi4} desktop={desktop}")

    os.environ["STUDYASSISTANT_HANDS_INTERVAL_MS"] = "999"
    try:
        overridden = intervals_for_board("pi4")
        check("环境变量可覆盖档位", overridden["hands"] == 999, str(overridden))
    finally:
        del os.environ["STUDYASSISTANT_HANDS_INTERVAL_MS"]

    # ---- 从未见手时的缺席时长：不能是哨兵大数 ----
    # 这个值会透到界面（实时证据 / HUD）。曾经返回 1e9，界面上就显示
    # 「手离开画面 1000000000.0s」。
    from study_assistant.features.hand_features import HandTracker

    tracker = HandTracker()
    tracker.update([], 1000.0)          # 第一帧就没手
    never = tracker.absent_for(1006.0)
    check(
        "从未见手时 absent_for 是会话时长（不是 1e9 之类的哨兵值）",
        abs(never - 6.0) < 0.01,
        f"absent_for={never}",
    )

    tracker.clear()
    check(
        "clear() 之后重新计时（不带着上一段会话累加）",
        tracker.absent_for(1010.0) == 0.0,
        f"absent_for={tracker.absent_for(1010.0)}",
    )

    # ---- 计数一致性 ----
    stats = scn.core.snapshot_stats()
    counted = (stats.focused_seconds + stats.distracted_seconds
               + stats.away_seconds + stats.uncertain_seconds)
    check(
        "各状态时长之和 == 总时长",
        abs(counted - stats.total_seconds) < 0.05,
        f"sum={counted:.2f} total={stats.total_seconds:.2f}",
    )

    # ---- 暂停 ----
    scn.core.set_paused(True)
    before = scn.core.session.total_seconds
    scn.wiggle(3.0, "paper", amplitude=0.012)
    after = scn.core.session.total_seconds
    check("暂停期间不计入任何时长", abs(after - before) < 0.05,
          f"{before:.2f} -> {after:.2f}")
    check("暂停后状态为 PAUSED", scn.core.last.snapshot.state is FocusState.PAUSED,
          str(scn.core.last.snapshot.state))

    scn.core.set_paused(False)
    scn.wiggle(4.0, "paper", amplitude=0.012)
    check("恢复后回到 FOCUSED", scn.core.last.snapshot.state is FocusState.FOCUSED,
          str(scn.core.last.snapshot.state))


# ======================================================================
# M. 人在画面内但手拍不到 → 不得判为离席
# ======================================================================


def scenario_m() -> None:
    section("M. 人在画面里、手拍不到：不得判为离席（应为证据不足）")

    # 真实场景：摄像头架得高只拍到上半身，手放在桌子下面 / 腿上。
    # 实测（调试面板）：`手数=0` 而 `person 0.91`，界面却显示「离席」，
    # 事件流里写下「离开座位」，还平白累计了 11s 离席时长 —— 直接拉低
    # 专注分。这里把整条链路钉住。
    person = [S.make_object("person", cx=0.5, cy=0.5, w=0.5, h=0.6, confidence=0.91)]

    scn = S.new_scenario()
    scn.wiggle(10.0, "paper", amplitude=0.012, objects=person)

    check(
        "先进入 FOCUSED",
        scn.state_at(scn.elapsed()) is FocusState.FOCUSED,
        f"实际 {scn.state_at(scn.elapsed()).value}",
    )

    base = scn.elapsed()
    focused_before = scn.core.snapshot_stats().focused_seconds

    # 手出画面 20 秒，人一直在画面里
    scn.no_hands(20.0, objects=person)

    check(
        "人的在场被识别（person_present）",
        scn.core.last.person_present is True,
        f"person_present={scn.core.last.person_present}",
    )

    check(
        "从未进入 AWAY（人还在画面里不可能是离席）",
        not scn.ever_in_state(FocusState.AWAY),
        f"历史 {[(v.value, round(d, 1)) for v, d in scn.state_history()]}",
    )

    check(
        "没有记录离席事件",
        len(scn.events_of("away_start")) == 0,
        f"{len(scn.events_of('away_start'))} 条",
    )

    stats = scn.core.snapshot_stats()
    check(
        "离席时长为 0（不产生虚假的离席惩罚）",
        stats.away_seconds < 0.05,
        f"away={stats.away_seconds:.2f}s",
    )

    check(
        "行为为 HAND_AWAY（如实说明「看不到手」）",
        scn.behavior_at(scn.elapsed()) is Behavior.HAND_AWAY,
        f"实际 {scn.behavior_at(scn.elapsed())}",
    )

    check(
        "最终状态为 UNCERTAIN（证据不足，不是专注也不是离席）",
        scn.state_at(scn.elapsed()) is FocusState.UNCERTAIN,
        f"实际 {scn.state_at(scn.elapsed()).value}",
    )

    # 证据文字必须真的流到 estimate —— 仪表盘的「实时证据」面板读的是
    # estimate.evidence。BehaviorEstimate 一度没有这个字段，导致
    # getattr 拿到 None 后直接返回，那块面板永远是「—」。
    est_evidence = getattr(scn.core.last.estimate, "evidence", None) or {}
    check(
        "证据文字透传到 estimate（实时证据面板不再是死面板）",
        bool(est_evidence),
        f"evidence 覆盖的行为={sorted(b.value for b in est_evidence)}",
    )
    hand_away_ev = est_evidence.get(Behavior.HAND_AWAY, [])
    check(
        "HAND_AWAY 的证据说明了「人在画面内」",
        any("人在画面内" in s for s in hand_away_ev),
        f"{hand_away_ev}",
    )

    # 关键不变量：看不到手就不能一路算专注。
    # 边界 = 离席确认窗口（8s，其间保持上一状态）+ 候选状态确认（1s）+
    # 抖动余量。刻意不写成"必须为 0"：宽容期本身就是防抖设计。
    limit = scn.core.machine.away_confirm + 2.0
    grow = stats.focused_seconds - focused_before
    check(
        f"看不到手的 20s 内最多记 ~{limit:.0f}s 专注（不会一路算专注）",
        grow <= limit,
        f"专注时长增长 {grow:.1f}s / 20.0s（上限 {limit:.0f}s）",
    )

    # 手回到桌面 → 恢复
    scn.wiggle(10.0, "paper", amplitude=0.012, objects=person)
    check(
        "手回到画面后恢复 FOCUSED",
        scn.state_at(scn.elapsed()) is FocusState.FOCUSED,
        f"实际 {scn.state_at(scn.elapsed()).value}",
    )

    # ---- 对照组：人真的走了，必须仍然判为离席 ----
    # 没有这一组，上面的「修好了」可能只是把离席判定整个废掉了。
    scn2 = S.new_scenario()
    scn2.wiggle(10.0, "paper", amplitude=0.012, objects=person)
    scn2.no_hands(20.0, objects=[])          # 人不在了

    check(
        "对照组：人离开画面后仍然会判为 AWAY",
        scn2.ever_in_state(FocusState.AWAY),
        f"历史 {[(v.value, round(d, 1)) for v, d in scn2.state_history()]}",
    )
    check(
        "对照组：离席时长被正常累计",
        scn2.core.snapshot_stats().away_seconds >= 2.0,
        f"away={scn2.core.snapshot_stats().away_seconds:.1f}s",
    )


# ======================================================================
# N. 手机的「位置先验」不得单独定罪（看书 / 打字曾被判成玩手机）
# ======================================================================


def scenario_n() -> None:
    section("N. 手机位置先验不得单独定罪（看书/打字不得被判玩手机）")

    # 背景：出厂 phone ROI = x∈[0.02,0.20], y∈[0.50,0.92] —— 画面左下**整条
    # 竖带**。看书时搭在桌边的手、打字时下探的手都会落进去。旧实现是
    # `phone_contact = 物体接触 OR 位置命中`，位置单独就能定罪 —— 于是
    # 「明明在看书」被判玩手机、「明明在敲键盘」也被判玩手机；证据文字还
    # 写着「手-手机接触 N s」，其实画面里根本没有手机。
    #
    # 修法：默认必须**检出 phone 物体**才定罪，位置只是先验。
    # 每条都给对照组 —— 否则「修好了」可能只是把玩手机判定整个废掉了。

    # --- 1) 手静止在 phone 框里，画面里没有任何物体 ---
    scn = S.new_scenario()
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])])
    scn.hold_still(8.0, "phone")

    report = scn.core.engine.last_report
    check(
        "位置命中被如实报告（位置=真 / 物体=假）",
        report.phone_roi_contact is True and report.phone_object_contact is False,
        f"位置={report.phone_roi_contact} 物体={report.phone_object_contact}",
    )
    check(
        "位置单独不得定罪：从未判为 PHONE_USE",
        not scn.ever_in_behavior(Behavior.PHONE_USE),
        f"历史 {hist(scn.behavior_history())}",
    )
    phone_ev = (getattr(scn.core.last.estimate, "evidence", None) or {}).get(
        Behavior.PHONE_USE, []
    )
    check(
        "证据文字不再谎称「手-手机接触」（画面里没有手机）",
        all("检出手机且手与它接触" not in s for s in phone_ev)
        and any("未检出手机" in s for s in phone_ev),
        f"{phone_ev}",
    )

    # --- 2) 看书：手静止在 phone 框里，手下压着一本**检出的书** ---
    scn = S.new_scenario()
    book = [S.make_object("book", cx=S.ROI_CENTER["phone"][0],
                          cy=S.ROI_CENTER["phone"][1], w=0.16, h=0.22)]
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])], objects=book)
    scn.hold_still(8.0, "phone", objects=book)

    check(
        "明明在看书 → 判 PAPER_STUDY，且从未判玩手机",
        scn.behavior_at(scn.elapsed()) is Behavior.PAPER_STUDY
        and not scn.ever_in_behavior(Behavior.PHONE_USE),
        f"历史 {hist(scn.behavior_history())}",
    )

    # --- 3) 敲键盘的手落在 phone 框里（无物体）---
    scn = S.new_scenario()
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])])
    scn.type_burst(8.0, "phone")

    check(
        "敲键盘的手落在手机框内 → 不判玩手机",
        not scn.ever_in_behavior(Behavior.PHONE_USE),
        f"历史 {hist(scn.behavior_history())}",
    )

    # --- 4) 对照（真阳性必须还在）：手压在**检出的手机**上 ---
    scn = S.new_scenario()
    obj = phone_object()
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])], objects=obj)
    scn.hold_still(8.0, "phone", objects=obj)

    check(
        "对照组：手压在真实检出的手机上**必须**判 PHONE_USE",
        scn.ever_in_behavior(Behavior.PHONE_USE),
        f"历史 {hist(scn.behavior_history())}",
    )
    check(
        "对照组：此时确实是物体通路为真（不是靠位置蹭到的）",
        scn.core.engine.last_report.phone_object_contact is True,
        f"物体={scn.core.engine.last_report.phone_object_contact}",
    )

    # --- 5) 差分对照：把「必须检出物体」关掉，位置才重新能定罪 ---
    # 证明被默认关掉的那条通路**还在**（不是把它删了），只是默认不启用。
    cfg = S.load_test_config({"behavior": {"phone": {"require_object_contact": False}}})
    scn = S.new_scenario(config=cfg)
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])])
    scn.hold_still(8.0, "phone")

    check(
        "差分对照：require_object_contact=False 时位置重新可定罪（通路仍在）",
        scn.ever_in_behavior(Behavior.PHONE_USE),
        f"历史 {hist(scn.behavior_history())}",
    )


# ======================================================================
# O. 纸质学习：全能量轴覆盖（静止阅读 → 书写动作，合并后不得有死区）
# ======================================================================


def _paper_motion(seconds: float, jitter: float, freq: float = 2.5):
    """手在书写区做给定幅度的小幅运动，返回 (纸质学习分, 能量, 场景)。

    读书与写字共用同一根「运动能量」轴。合并成一个行为之后，验收目标从
    "两条分分得开"变成"**整根轴上分数都不能塌**"——静止端靠静止阅读支路、
    活动端靠书写动作支路，中间不允许出现两边都接不住的死区。
    """
    scn = S.new_scenario()
    base = S.ROI_CENTER["paper"]

    captured = []
    original = scn.core.fusion.update

    def spy(*a, **k):
        v = original(*a, **k)
        captured.append(v)
        return v

    scn.core.fusion.update = spy

    scn.step(2, hands=[S.make_hand("right", base)])

    steps = max(1, int(round(seconds / scn.dt)))
    for i in range(steps):
        phase = i * scn.dt * freq * 2 * math.pi
        x = base[0] + math.sin(phase) * jitter
        y = base[1] + math.cos(phase * 1.7) * jitter * 0.5
        scn.step(1, hands=[S.make_hand("right", (x, y))])

    verdict = captured[-1]

    energy = None
    for s in verdict.evidence.get(Behavior.PAPER_STUDY, []):
        if "运动能量" in s:
            try:
                energy = float(s.split("运动能量")[1].strip().split()[0])
            except Exception:
                energy = None

    return (
        verdict.raw_scores.get(Behavior.PAPER_STUDY, 0.0),
        energy,
        scn,
    )


def scenario_o() -> None:
    section("O. 纸质学习：全能量轴覆盖（静止 → 写字，分数不得塌陷）")

    # 背景：2026-09-29 起 READING 与 WRITING 合并为 PAPER_STUDY。合并前用户
    # 反馈"写字和读书分不开"（旧实现阅读有 0.550 硬下限）；合并后这个问题
    # 自然消失，但**新的风险**是：两条支路如果哪条在中间能量带接不住，会出现
    # "手明明在纸面上、分数却掉下去"的死区。本组沿能量轴密集采样，
    # 确认整根轴上纸质学习分都保持在高位。

    # 铺轴：从"完全不动"到"明显在写"（抖动 → 能量，实测映射）
    jitters = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.012]
    rows = []
    for j in jitters:
        p, e, _scn = _paper_motion(10.0, j)
        rows.append({"j": j, "p": p, "e": e})

    say()
    say(f"  {'抖动':>8} {'能量':>9} {'纸质学习分':>10}")
    for row in rows:
        e = "n/a" if row["e"] is None else f"{row['e']:.4f}"
        say(f"  {row['j']:>8.4f} {e:>9} {row['p']:>10.3f}")

    # ---- 1) 静止端：静止阅读支路必须把分撑起来 ----
    still = rows[0]
    check(
        "完全静止（读书）→ 纸质学习分 ≥0.80（静止阅读支路生效）",
        still["p"] >= 0.80,
        f"分数 {still['p']:.3f}",
    )

    # ---- 2) 活动端：书写动作支路必须把分撑起来 ----
    active = [row for row in rows
              if row["e"] is not None and row["e"] >= 0.03]
    check(
        "存在足够多的「明显在写」采点用于判定",
        len(active) >= 3,
        f"{len(active)} 个采点能量 ≥0.03",
    )
    min_active = min(row["p"] for row in active) if active else 0.0
    check(
        "明显在写 → 纸质学习分 ≥0.70（书写动作支路生效）",
        min_active >= 0.70,
        f"最小分数 {min_active:.3f}（采点 {len(active)} 个）",
    )

    # ---- 3) 整根轴不得有死区 ----
    # 合并前"写字 vs 阅读"互相抢分的过渡带，合并后必须被两条支路之一接住。
    min_all = min(row["p"] for row in rows)
    worst = min(rows, key=lambda r: r["p"])
    check(
        "全能量轴（含中间过渡带）纸质学习分 ≥0.60（无死区）",
        min_all >= 0.60,
        f"最低点 抖动{worst['j']:.4f} → {worst['p']:.3f}",
    )

    # ---- 4) 对照组：真阅读场景（书 + 手压着不动）仍判 PAPER_STUDY ----
    scn = S.new_scenario()
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["paper"])],
             objects=book_object())
    scn.hold_still(14.0, "paper", objects=book_object())
    check(
        "对照组：手压着书不动 14s → 判 PAPER_STUDY",
        scn.behavior_at(scn.elapsed()) is Behavior.PAPER_STUDY,
        f"历史 {hist(scn.behavior_history())}",
    )

    # ---- 5) 对照组：翻页（手短暂离开但书还在）不得掉出纸质学习 ----
    scn = S.new_scenario()
    scn.step(2, hands=[S.make_hand("right", S.ROI_CENTER["paper"])],
             objects=book_object())
    scn.hold_still(8.0, "paper", objects=book_object())
    scn.no_hands(2.0, objects=book_object())     # 翻页：手短暂离开
    scn.hold_still(6.0, "paper", objects=book_object())
    check(
        "对照组：阅读中途手短暂离开（翻页）→ 仍判 PAPER_STUDY",
        scn.behavior_at(scn.elapsed()) is Behavior.PAPER_STUDY,
        f"历史 {hist(scn.behavior_history())}",
    )


# ======================================================================
# P. 位置先验关闭：标定框不得影响任何结论（判定只看物体检测 + 动作）
# ======================================================================


def scenario_p() -> None:
    section("P. 位置先验关闭：paper/keyboard/phone/desk 框不参与判定")

    cfg_on = S.load_test_config()  # 测试默认：先验开启（对照组用）
    cfg_off = S.load_test_config({"behavior": {"use_position_prior": False}})

    # ---- P1 对照组（先验开启）：手在 paper 框 + 书写动作 → PAPER_STUDY ----
    scn_on = S.new_scenario(config=cfg_on)
    scn_on.step(3, hands=[S.make_hand("right", S.ROI_CENTER["paper"])])
    scn_on.wiggle(12.0, "paper", amplitude=0.012, frequency=2.5)
    check(
        "P1 对照（先验开启）：paper 框 + 书写动作 → PAPER_STUDY",
        scn_on.ever_in_behavior(Behavior.PAPER_STUDY),
        hist(scn_on.behavior_history()),
    )

    # ---- P2 先验关闭：完全相同的输入、无 book 物体 → 不得判 PAPER_STUDY ----
    scn_off = S.new_scenario(config=cfg_off)
    scn_off.step(3, hands=[S.make_hand("right", S.ROI_CENTER["paper"])])
    scn_off.wiggle(12.0, "paper", amplitude=0.012, frequency=2.5)
    check(
        "P2 先验关闭：同输入（未检出 book）→ 从不判 PAPER_STUDY",
        not scn_off.ever_in_behavior(Behavior.PAPER_STUDY),
        hist(scn_off.behavior_history()),
    )

    # ---- P3 先验关闭 + 检出 book：物体 + 书写动作必须仍判 PAPER_STUDY ----
    # 对照意义：证明 P2 是"位置被关掉"，不是"纸质学习判定整个被废掉"。
    scn_book = S.new_scenario(config=cfg_off)
    scn_book.step(3, hands=[S.make_hand("right", S.ROI_CENTER["paper"])],
                  objects=book_object())
    scn_book.wiggle(12.0, "paper", amplitude=0.012, frequency=2.5,
                    objects=book_object())
    check(
        "P3 先验关闭：检出 book + 书写动作 → 仍判 PAPER_STUDY",
        scn_book.ever_in_behavior(Behavior.PAPER_STUDY),
        hist(scn_book.behavior_history()),
    )

    # ---- P4 先验关闭：手在 keyboard 框 + 键击动作、无键盘物体 → 不判电脑学习 ----
    scn_kb = S.new_scenario(config=cfg_off)
    scn_kb.step(2, hands=[S.make_hand("right", S.ROI_CENTER["keyboard"])])
    scn_kb.type_burst(14.0, "keyboard")
    check(
        "P4 先验关闭：keyboard 框 + 键击（未检出键盘物体）→ 从不判 COMPUTER_STUDY",
        not scn_kb.ever_in_behavior(Behavior.COMPUTER_STUDY),
        hist(scn_kb.behavior_history()),
    )

    # ---- P5 对照组（先验开启）：同输入 → 位置兜底仍可判电脑学习 ----
    scn_kb_on = S.new_scenario(config=cfg_on)
    scn_kb_on.step(2, hands=[S.make_hand("right", S.ROI_CENTER["keyboard"])])
    scn_kb_on.type_burst(14.0, "keyboard")
    check(
        "P5 对照（先验开启）：同输入 → 位置兜底判 COMPUTER_STUDY",
        scn_kb_on.ever_in_behavior(Behavior.COMPUTER_STUDY),
        hist(scn_kb_on.behavior_history()),
    )

    # ---- P6 先验关闭：手在 phone 框 + 无手机物体 → 不得判玩手机 ----
    scn_ph = S.new_scenario(config=cfg_off)
    scn_ph.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])])
    scn_ph.hold_still(12.0, "phone")
    check(
        "P6 先验关闭：phone 框 + 手停留（未检出手机）→ 从不判 PHONE_USE",
        not scn_ph.ever_in_behavior(Behavior.PHONE_USE),
        hist(scn_ph.behavior_history()),
    )

    # ---- P7 先验关闭 + 检出手机物体：检测通路完好，必须仍判玩手机 ----
    scn_ph2 = S.new_scenario(config=cfg_off)
    scn_ph2.step(2, hands=[S.make_hand("right", S.ROI_CENTER["phone"])],
                 objects=phone_object())
    scn_ph2.hold_still(12.0, "phone", objects=phone_object())
    check(
        "P7 先验关闭：检出手机 + 手与它接触 → 仍判 PHONE_USE",
        scn_ph2.ever_in_behavior(Behavior.PHONE_USE),
        hist(scn_ph2.behavior_history()),
    )


# ======================================================================
# 运行器
# ======================================================================

SCENARIOS = {
    "A": scenario_a, "B": scenario_b, "C": scenario_c, "D": scenario_d,
    "E": scenario_e, "F": scenario_f, "G": scenario_g, "H": scenario_h,
    "I": scenario_i, "J": scenario_j, "K": scenario_k, "L": scenario_l,
    "M": scenario_m, "N": scenario_n, "O": scenario_o, "P": scenario_p,
}


def main(argv) -> int:
    wanted = [a.upper() for a in argv[1:]] or list(SCENARIOS)

    say("=" * 74)
    say("桌面学习行为检测与专注陪伴助手 —— 验收测试")
    say(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"场景：{', '.join(wanted)}")
    say("=" * 74)

    for key in wanted:
        func = SCENARIOS.get(key)

        if func is None:
            say(f"\n[跳过] 未知场景 {key}")
            continue

        try:
            func()
        except Exception as exc:
            say(f"\n[异常] 场景 {key} 抛出异常：{exc}")
            traceback.print_exc()
            check(f"场景 {key} 无异常", False, repr(exc))

    say()
    say("=" * 74)
    failed = [n for n, ok, _ in RESULTS if not ok]
    say(f"共 {len(RESULTS)} 项断言：{len(RESULTS) - len(failed)} 通过 / {len(failed)} 失败")

    if failed:
        say("")
        say("失败项：")
        for name, ok, detail in RESULTS:
            if not ok:
                say(f"  x {name}  -- {detail}")
    else:
        say("全部通过 OK")

    say("=" * 74)

    flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
