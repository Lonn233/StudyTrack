"""真机标定：用你自己的手和键盘，量出「打字动作签名」的真实读数。

为什么必须真机量一次
--------------------
`computer.typing_*` 那组阈值最初是在**合成输入**下定的 —— 合成打字的
`direction_change_rate` 能到 24/s。真机上 MediaPipe 的跟踪抖动会把方向
反转率明显拉低，直接照搬合成阈值的结果就是「真打字过不了签名 →
电脑学习永不触发」。阈值必须用你自己的手、你自己的摄像头量一次。

工具做三件事（不是只打印一堆数字就完事）
----------------------------------------
1. 录两段：**打字**（正样本）与**手放键盘上不动**（负对照）—— 没有负对照
   就无法知道阈值是"刚好卡住打字"还是"什么都能过"；
2. 用与线上**完全同一份公式**（`typing_signature_from_raw`）算每帧签名分；
3. **闭环验算**：拿建议的阈值重新代回同一份公式，正样本必须过、负对照必须
   不过。不成立就明确报「本次标定无效」，而不是闷头写进配置。

用法（摄像头同时只能被一个程序占用 → 跑之前先关掉 GUI / 标定窗口）::

    python tools/calibrate_typing.py                # 只量、只打印建议
    python tools/calibrate_typing.py --seconds 25
    python tools/calibrate_typing.py --apply        # 量完写回 config/config.yaml
    python tools/calibrate_typing.py --self-test    # 无摄像头自检（合成输入）

标定要点：只让**正在打字的那只手**出现在画面里；另一只手入画会把各项的
最大值抬高（签名取的是所有手的最大值），量出来的读数偏乐观。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from study_assistant.behavior.behavior_fusion import (  # noqa: E402
    TYPING_RAW_NAMES,
    typing_signature_from_raw,
)

#: 会被写回 config.yaml 的键（只允许这 8 个，防止误伤别的配置）
APPLY_KEYS = (
    "typing_dir_min",
    "typing_dir_full",
    "typing_burst_min",
    "typing_burst_full",
    "typing_range_max",
    "typing_energy_min",
    "typing_energy_full",
    "typing_min_score",
)


# ======================================================================
# 一、分析（纯函数：不碰摄像头、不碰文件，可单测）
# ======================================================================


def percentile(values: list, q: float) -> float:
    """线性插值分位数。空列表返回 0.0。"""
    if not values:
        return 0.0

    xs = sorted(float(v) for v in values)

    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return xs[int(pos)]

    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize(samples: list) -> dict:
    """把逐帧签名读数汇总成分位数。"""
    out = {"n": len(samples)}

    for key in TYPING_RAW_NAMES:
        col = [s["raw"][key] for s in samples]
        out[key] = {
            "p25": percentile(col, 0.25),
            "p50": percentile(col, 0.50),
            "p75": percentile(col, 0.75),
            "max": max(col) if col else 0.0,
        }

    scores = [s["score"] for s in samples]
    out["score"] = {
        "p25": percentile(scores, 0.25),
        "p50": percentile(scores, 0.50),
        "p75": percentile(scores, 0.75),
    }
    return out


def suggest(typing: dict, current: dict) -> dict:
    """由**打字**样本的分位数推出建议阈值。

    取 p25（而不是 p50）做锚点是刻意的：阈值要对"打得慢一点的那几秒"也
    成立，不能只对中位数成立 —— 否则打一段就要掉一次判定，用户看到的是
    "时好时坏"。

    每个 ramp 都用「下限 ≈ 0.5×p25、满值 ≈ p75」的构图，让真实打字落在
    ramp 的中上段（有裕量），而不是压在阈值边缘。
    """
    dir_lo = 0.5 * typing["dir_rate"]["p25"]
    burst_lo = 0.6 * typing["std_ratio"]["p25"]
    energy_lo = 0.5 * typing["energy"]["p25"]

    return {
        "typing_dir_min": _round(dir_lo, 3),
        "typing_dir_full": _round(max(dir_lo * 1.6, typing["dir_rate"]["p75"]), 3),
        "typing_burst_min": _round(burst_lo, 3),
        "typing_burst_full": _round(
            max(burst_lo * 1.4, typing["std_ratio"]["p75"]), 3
        ),
        # 幅度是「上限」：p75 再留 30% 裕量，且不低于出厂值的一半 ——
        # 键盘就在手底下，3s 幅度不该超过画面高度的百分之几十。
        "typing_range_max": _round(max(0.06, typing["range_3s"]["p75"] * 1.3), 3),
        "typing_energy_min": _round(energy_lo, 4),
        "typing_energy_full": _round(max(energy_lo * 1.6, typing["energy"]["p75"]), 4),
        "typing_min_score": float(current.get("typing_min_score", 0.45)),
    }


def _round(value: float, digits: int) -> float:
    return float(f"{value:.{digits}f}")


def _threshold_kwargs(thresholds: dict) -> dict:
    return dict(
        dir_lo=thresholds["typing_dir_min"],
        dir_hi=thresholds["typing_dir_full"],
        burst_lo=thresholds["typing_burst_min"],
        burst_hi=thresholds["typing_burst_full"],
        range_max=thresholds["typing_range_max"],
        energy_lo=thresholds["typing_energy_min"],
        energy_hi=thresholds["typing_energy_full"],
    )


def rescore(samples: list, thresholds: dict) -> list:
    """用给定阈值 + **同一份公式**把原始读数重算成签名分。

    这里刻意不自己写一遍 ramp：`typing_signature_from_raw` 就是线上判定
    用的那个函数，闭环验算才有意义。
    """
    kwargs = _threshold_kwargs(thresholds)
    return [
        typing_signature_from_raw(s["raw"], **kwargs)["score"] for s in samples
    ]


def verify_closed_loop(typing_samples: list, still_samples: list, thresholds: dict) -> dict:
    """闭环验算：正样本必须过线、负对照必须不过线。

    返回 {"ok": bool, "typing_p25": float, "still_p75": float,
          "typing_p50": float, "notes": [str, ...]}
    """
    min_score = float(thresholds["typing_min_score"])

    typing_scores = rescore(typing_samples, thresholds)
    still_scores = rescore(still_samples, thresholds) if still_samples else []

    typing_p25 = percentile(typing_scores, 0.25)
    typing_p50 = percentile(typing_scores, 0.50)
    still_p75 = percentile(still_scores, 0.75)

    notes = []
    ok = True

    if not typing_samples:
        notes.append("正样本为空（标定期间没检测到手？）")
        ok = False
    elif typing_p25 < min_score:
        notes.append(
            f"打字 p25 签名分 {typing_p25:.3f} < 达标线 {min_score:.3f}"
            "：真打字也过不了签名，阈值过严"
        )
        ok = False

    if not still_samples:
        notes.append("负对照为空：没有对照就无法确认阈值有没有区分力")
        ok = False
    elif still_p75 >= min_score:
        notes.append(
            f"手不动 p75 签名分 {still_p75:.3f} ≥ 达标线 {min_score:.3f}"
            "：负对照也能过线，说明这组阈值没区分力"
        )
        ok = False

    return {
        "ok": ok,
        "typing_p25": typing_p25,
        "typing_p50": typing_p50,
        "still_p75": still_p75,
        "notes": notes,
    }


# ======================================================================
# 二、写回 config.yaml（按行改值，注释与缩进原样保留）
# ======================================================================

_KEY_RE = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
    r"(?P<value>[^#\n]*?)\s*(?P<comment>#.*)?$"
)


def patch_yaml_text(text: str, updates: dict) -> tuple:
    """按行把 `key: value` 的值换掉，其余行（含全部注释）逐字符不动。

    为什么不用 `yaml.safe_dump` 重写整个文件：那会把 config.yaml 里成片的
    中文注释全部冲掉 —— 这个工程的注释本身就是文档。
    """
    lines_out = []
    changed = []

    for line in text.splitlines():
        m = _KEY_RE.match(line)

        if m and m.group("key") in updates:
            key = m.group("key")
            indent = m.group("indent")
            comment = m.group("comment")
            tail = f"  {comment}" if comment else ""
            lines_out.append(f"{indent}{key}: {updates[key]}{tail}")
            changed.append(key)
        else:
            lines_out.append(line)

    result = "\n".join(lines_out)
    if text.endswith("\n"):
        result += "\n"

    return result, changed


def apply_thresholds(thresholds: dict, config_path: Path) -> list:
    """备份 + 按行写回。返回实际改动的键。"""
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    text = config_path.read_text(encoding="utf-8")
    updates = {k: thresholds[k] for k in APPLY_KEYS if k in thresholds}

    new_text, changed = patch_yaml_text(text, updates)

    missing = sorted(set(updates) - set(changed))
    if missing:
        raise RuntimeError(
            "配置文件里找不到这些键，拒绝写入（避免把配置改成半截）："
            + ", ".join(missing)
        )

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = config_path.with_name(f"{config_path.name}.bak-{stamp}")
    shutil.copy2(config_path, backup)
    config_path.write_text(new_text, encoding="utf-8")

    print(f"[已写回] {config_path}")
    print(f"[已备份] {backup}")
    return changed


# ======================================================================
# 三、采样
# ======================================================================


def _signature_of(core, now: float) -> dict:
    """当前帧的打字签名（走线上同一个方法）。"""
    return core.fusion.typing_signature(core.hand_tracker.all_motions(now))


def collect_synthetic(kind: str, total: float = 8.0, warmup: float = 3.0,
                      chunk: float = 0.5) -> list:
    """无摄像头自检用：用合成手驱动真实 BehaviorCore。

    `type_burst`（键击脉冲）作正样本、`hold_still`（手压着键盘不动）作负
    对照。分块调用是为了能在每块之后取一次签名 —— 一次跑完只能得到一个
    样本，分位数就没有意义了。
    """
    from synthetic import new_scenario  # tests/synthetic.py

    if kind == "typing":
        def drive(sc, seconds):
            sc.type_burst(seconds)
    elif kind == "still":
        def drive(sc, seconds):
            sc.hold_still(seconds, roi="keyboard")
    else:
        raise ValueError(f"未知的合成样本类型：{kind}")

    sc = new_scenario()
    drive(sc, warmup)

    samples = []
    blocks = max(1, int(round((total - warmup) / chunk)))

    for _ in range(blocks):
        drive(sc, chunk)
        samples.append(_signature_of(sc.core, sc.now))

    return samples


def collect_live(args, config) -> tuple:
    """真机采样：打字 + 手不动两段。返回 (typing_samples, still_samples)。"""
    import cv2

    from study_assistant.core import BehaviorCore
    from study_assistant.desk.roi_manager import Calibration
    from study_assistant.notifications.notification_manager import NotificationManager
    from study_assistant.vision.vision_pipeline import VisionPipeline

    notifications = NotificationManager(config, enable_system_sound=False)
    core = BehaviorCore(
        config,
        Calibration(),
        database=None,
        warmup_seconds=0.0,
        notifications=notifications,
    )
    core.hand_detector_available = True
    core.start()

    pipe = VisionPipeline(config, root=ROOT)
    for warning in pipe.warnings:
        print(f"[警告] {warning}")

    if pipe.hand_detector is None:
        raise RuntimeError("手部检测器不可用，无法标定（检查 models/hand_landmarker.task）")

    pipe.start()

    window = "typing-calibration"
    if args.preview:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 720, 540)

    phases = [
        ("typing", "正在打字", args.seconds, "TYPE NOW  (type on your keyboard)"),
        ("still", "手不动", args.still_seconds,
         "HANDS STILL on the keyboard (negative control)"),
    ]

    samples = {"typing": [], "still": []}
    overlay = {}

    try:
        for idx, (kind, cn_name, duration, hint) in enumerate(phases):
            print(f"\n=== 第 {idx + 1}/{len(phases)} 段：{cn_name} "
                  f"（{duration:.0f}s）===")
            print(f"    {hint}")

            start = time.perf_counter()
            overlay = {}

            while True:
                elapsed = time.perf_counter() - start
                if elapsed >= duration:
                    break

                vf = pipe.step()

                if vf is None:
                    continue

                height, width = vf.frame.shape[:2]

                core.update(
                    hands=vf.hands.hands,
                    tracked_objects=vf.tracked.objects,
                    now=vf.now,
                    frame_width=width,
                    frame_height=height,
                    camera_ok=True,
                    hands_fresh=vf.hands_fresh,
                    objects_fresh=vf.objects_fresh,
                )

                # 窗口要填满（1s/3s 窗口），手也要在画面里，样本才有意义
                usable = elapsed > 3.0 and vf.hands.count > 0
                sample = None

                if usable:
                    sample = _signature_of(core, vf.now)
                    samples[kind].append(sample)

                overlay = {"elapsed": elapsed, "duration": duration,
                           "hint": hint, "hand": vf.hands.count,
                           "sample": sample}

                if args.preview:
                    _draw_overlay(cv2, vf.frame.copy(), window, overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt
    finally:
        pipe.close()
        if args.preview:
            cv2.destroyAllWindows()

    return samples["typing"], samples["still"]


def _draw_overlay(cv2, frame, window: str, info: dict) -> None:
    """画提示（只用 ASCII —— cv2.putText 不带中文字库）。"""
    height, width = frame.shape[:2]
    remain = max(0.0, info["duration"] - info["elapsed"])

    cv2.rectangle(frame, (0, 0), (width, 96), (0, 0, 0), -1)
    cv2.putText(frame, info["hint"], (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"{remain:4.1f}s   hands={info['hand']}", (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    sample = info.get("sample")
    if sample:
        raw = sample["raw"]
        cv2.putText(
            frame,
            f"dir {raw['dir_rate']:.2f}/s  burst {raw['std_ratio']:.2f}  "
            f"rng {raw['range_3s']:.3f}  E {raw['energy']:.3f}  "
            f"sig {sample['score']:.2f}",
            (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
        )
    else:
        cv2.putText(frame, "waiting for a hand in frame...", (12, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 1)

    cv2.imshow(window, frame)


# ======================================================================
# 四、报告
# ======================================================================


def _fmt_cell(stat: dict, key: str, digits: int = 3) -> str:
    return f"{stat[key]['p25']:.{digits}f}/{stat[key]['p50']:.{digits}f}/{stat[key]['p75']:.{digits}f}"


def print_report(typing: dict, still: dict, current: dict, suggested: dict,
                 check: dict) -> None:
    print("\n" + "=" * 78)
    print("真机标定结果")
    print("=" * 78)

    head = (f"{'阶段':<16}{'样本':>6}  {'dir/s p25/50/75':>22}  "
            f"{'burst p25/50/75':>22}  {'签名分 p25/50/75':>22}")
    print(head)
    print("-" * len(head))

    for label, stat in (("打字(正样本)", typing), ("手不动(负对照)", still)):
        print(
            f"{label:<16}{stat['n']:>6}  {_fmt_cell(stat, 'dir_rate', 2):>22}  "
            f"{_fmt_cell(stat, 'std_ratio', 2):>22}  "
            f"{_fmt_cell(stat, 'score', 3):>22}"
        )

    print(f"\n{'3s 幅度 p25/50/75':<24}"
          f"打字 {_fmt_cell(typing, 'range_3s', 4)}   "
          f"手不动 {_fmt_cell(still, 'range_3s', 4)}")
    print(f"{'能量 p25/50/75':<24}"
          f"打字 {_fmt_cell(typing, 'energy', 4)}   "
          f"手不动 {_fmt_cell(still, 'energy', 4)}")

    print("\n现行阈值（config.yaml）→ 建议阈值")
    print("-" * 66)
    for key in APPLY_KEYS:
        now_value = current.get(key, "—")
        new_value = suggested[key]
        flag = "" if str(now_value) == str(new_value) else "   <-- 改"
        print(f"  {key:<22} {str(now_value):>10}  ->  {new_value:>10}{flag}")

    print("\n闭环验算（建议阈值代回**同一份公式**重算上面那些原始读数）")
    print("-" * 66)
    min_score = float(suggested["typing_min_score"])
    print(f"  打字 p25 签名分 = {check['typing_p25']:.3f}  "
          f"（p50 {check['typing_p50']:.3f}）需 ≥ {min_score:.3f}")
    print(f"  手不动 p75 签名分 = {check['still_p75']:.3f} 需 < {min_score:.3f}")

    if check["ok"]:
        print("  => 本次标定有效：正样本过线、负对照不过线")
    else:
        print("  => 本次标定**无效**：")
        for note in check["notes"]:
            print(f"     · {note}")


def load_current_thresholds(config) -> dict:
    computer = config.behavior.get("computer", {}) or {}
    mapping = {
        "typing_dir_min": "typing_dir_min",
        "typing_dir_full": "typing_dir_full",
        "typing_burst_min": "typing_burst_min",
        "typing_burst_full": "typing_burst_full",
        "typing_range_max": "typing_range_max",
        "typing_energy_min": "typing_energy_min",
        "typing_energy_full": "typing_energy_full",
        "typing_min_score": "typing_min_score",
    }
    return {
        out_key: computer.get(src_key)
        for out_key, src_key in mapping.items()
    }


# ======================================================================
# 五、入口
# ======================================================================


def self_test_patcher(config_path: Path, updates: dict) -> tuple:
    """自检 `--apply` 的写回路径：**不碰磁盘**，只在内存里改文本再回解析。

    为什么必须单独验：`--apply` 是把用户真在用的 config.yaml 按行改写，
    改坏了就是配置损坏 —— 而这条路径在开发机上跑不到（没摄像头、不该真写）。
    所以拿真实配置文本在内存里过一遍，逐条确认：
      1. 目标键确实被替换；
      2. 除目标键所在行，**其余每一行逐字符不变**（注释是这个工程的一部分）；
      3. 改完还是合法 YAML，且**回解析**得到的就是要写的值；
      4. 除目标键外，整份配置的解析结果与改前完全相同。
    """
    import yaml

    original = config_path.read_text(encoding="utf-8")
    patched, changed = patch_yaml_text(original, updates)

    notes = []
    ok = True

    missing = sorted(set(updates) - set(changed))
    if missing:
        ok = False
        notes.append(f"这些键没被替换到：{missing}")

    old_lines = original.splitlines()
    new_lines = patched.splitlines()

    if len(old_lines) != len(new_lines):
        ok = False
        notes.append(f"行数变了：{len(old_lines)} -> {len(new_lines)}")

    for old, new in zip(old_lines, new_lines):
        if old == new:
            continue
        if "#" in old and "#" not in new:
            ok = False
            notes.append(f"注释被吃掉了：{old.strip()!r}")
            break

    try:
        before = yaml.safe_load(original)
        after = yaml.safe_load(patched)
    except Exception as exc:
        return False, [f"改完的文本不再是合法 YAML：{exc}"]

    computer = (after.get("behavior") or {}).get("computer") or {}
    for key, value in updates.items():
        if computer.get(key) != value:
            ok = False
            notes.append(f"{key} 回解析得到 {computer.get(key)!r}，期望 {value!r}")

    for key in updates:
        (before.get("behavior") or {}).get("computer", {}).pop(key, None)
        computer.pop(key, None)

    if before != after:
        ok = False
        notes.append("除目标键以外还有别的内容被改了")

    return ok, notes


def run_self_test() -> int:
    """无摄像头自检：合成正/负样本走完整分析链，验证工具本身能用。"""
    from study_assistant.config import Config

    config_path = ROOT / "config" / "config.yaml"
    config = Config.load(config_path)

    print("=== 自检：合成输入（无需摄像头）===")
    typing_samples = collect_synthetic("typing")
    still_samples = collect_synthetic("still")

    current = load_current_thresholds(config)
    typing_stats = summarize(typing_samples)
    still_stats = summarize(still_samples)
    suggested = suggest(typing_stats, current)
    check = verify_closed_loop(typing_samples, still_samples, suggested)

    print_report(typing_stats, still_stats, current, suggested, check)

    # 自检的硬门槛：合成打字必须明显分得开静止的手
    separation = typing_stats["dir_rate"]["p25"] - still_stats["dir_rate"]["p75"]

    print("\n=== 自检判定 ===")
    print(f"合成打字 dir/s p25 = {typing_stats['dir_rate']['p25']:.2f}")
    print(f"手不动  dir/s p75 = {still_stats['dir_rate']['p75']:.2f}")
    print(f"分离度 = {separation:.2f}/s")

    analysis_ok = check["ok"] and separation > 1.0
    print(f"[{'OK' if analysis_ok else 'FAIL'}] 分析链（签名 + 建议 + 闭环验算）")

    patcher_ok, notes = self_test_patcher(config_path, suggested)
    print(f"[{'OK' if patcher_ok else 'FAIL'}] config.yaml 写回路径"
          "（内存内改写 + 回解析，未落盘）")
    for note in notes:
        print(f"     · {note}")

    good = analysis_ok and patcher_ok
    print("自检通过：分析链与写回路径都可用" if good
          else "自检失败：见上面 FAIL 项")
    return 0 if good else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="真机标定打字动作签名（阈值靠你自己的手量出来）"
    )
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="打字阶段时长（秒），默认 20")
    parser.add_argument("--still-seconds", type=float, default=8.0,
                        help="手不动负对照时长（秒），默认 8")
    parser.add_argument("--apply", action="store_true",
                        help="标定有效时写回 config/config.yaml（会先备份）")
    parser.add_argument("--force", action="store_true",
                        help="闭环验算没通过也照写（默认拒绝）")
    parser.add_argument("--no-preview", dest="preview", action="store_false",
                        help="不显示预览窗口")
    parser.add_argument("--self-test", action="store_true",
                        help="用合成输入自检工具本身，不开摄像头")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    from study_assistant.config import Config

    config_path = ROOT / "config" / "config.yaml"
    config = Config.load(config_path)

    print("标定期间请只让**正在打字的那只手**出现在画面里（另一只手入画会")
    print("把读数抬高，标出来的阈值偏乐观）。按 q 可随时中止。\n")

    try:
        typing_samples, still_samples = collect_live(args, config)
    except KeyboardInterrupt:
        print("\n[中止] 未写入任何配置。")
        return 130
    except Exception as exc:  # 摄像头 / 模型问题都从这里出去
        print(f"\n[失败] {exc}")
        return 2

    if not typing_samples:
        print("\n[失败] 打字阶段一个样本都没采到 —— 手没进画面，"
              "或摄像头没出帧。标定未完成，未写入任何配置。")
        return 2

    current = load_current_thresholds(config)
    typing_stats = summarize(typing_samples)
    still_stats = summarize(still_samples)
    suggested = suggest(typing_stats, current)
    check = verify_closed_loop(typing_samples, still_samples, suggested)

    # 闭环不过线时，先试一次"把达标线降到实测 p25 的 85%" —— 这是唯一一个
    # 允许自动回调的键，其余四个量都由实测分位数直接定，不该再人为放宽。
    if not check["ok"] and typing_samples:
        relaxed = dict(suggested)
        relaxed["typing_min_score"] = _round(
            max(0.20, check["typing_p25"] * 0.85), 3
        )
        retry = verify_closed_loop(typing_samples, still_samples, relaxed)
        if retry["ok"]:
            print(f"\n[提示] 达标线由 {suggested['typing_min_score']} 降到 "
                  f"{relaxed['typing_min_score']} 后才闭环通过。")
            suggested, check = relaxed, retry

    print_report(typing_stats, still_stats, current, suggested, check)

    if not args.apply:
        print("\n（未加 --apply，以上仅为建议。加上 --apply 可写回 config.yaml）")
        return 0

    if not check["ok"] and not args.force:
        print("\n[拒绝写入] 闭环验算没通过。请重标一次（打字时手别离开画面、"
              "另一只手别入画），或加 --force 强行写入。")
        return 1

    apply_thresholds(suggested, config_path)
    print("\n[完成] 重跑 tools/calibrate_typing.py 应能看到「本次标定有效」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
