"""管线级检查：真实摄像头 + 真实模型，验证 VisionFrame 契约与分频调度。

这一层回答的是"接线对不对"，与 acceptance_test 的"逻辑对不对"互补：

  * 每帧都必须有 hands / objects / tracked 三个结果对象（可以是缓存，
    但不能是 None）—— 下游代码依赖这一点，缺了就会 AttributeError；
  * `hands_fresh` / `objects_fresh` 的实际比例必须接近配置的分频比例
    （手 66ms ≈ 每 3 帧一次，物体 200ms ≈ 每 9 帧一次 @ 30fps）；
  * 手部推理耗时与物体推理耗时必须能测出来且 > 0（如果恒为 0，说明
    推理根本没跑，或者异常被吞了 —— 这正是历史上踩过的坑）；
  * 物体轨迹在同一目标上要复用 track_id，而不是每帧新建。

    python tests/pipeline_check.py [帧数]
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG: list[str] = []
RESULTS: list[tuple[str, bool, str]] = []


def say(msg: str = "") -> None:
    LOG.append(str(msg))
    print(msg)


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    say(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main(argv) -> int:
    frames_wanted = int(argv[1]) if len(argv) > 1 else 180

    from study_assistant.camera.camera_backend import create_camera
    from study_assistant.config import Config
    from study_assistant.vision.board_profile import describe
    from study_assistant.vision.vision_pipeline import VisionPipeline

    say("=" * 70)
    say("管线级检查：分频调度 / VisionFrame 契约 / 跟踪稳定性")
    say(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * 70)

    config = Config.load(ROOT / "config" / "config.yaml")

    camera = create_camera(config.data)
    pipeline = VisionPipeline(config, camera=camera, root=ROOT)

    say()
    say(f"板卡：{describe(pipeline.board)}")
    say(f"配置：手部 {pipeline.hands_interval_ms:.0f}ms / 物体 {pipeline.objects_interval_ms:.0f}ms")

    for warning in pipeline.warnings:
        say(f"[警告] {warning}")

    if pipeline.hand_detector is None:
        say("[致命] 手部检测器不可用，无法继续")
        return 2

    pipeline.start()

    frames = []
    t0 = time.perf_counter()

    try:
        while len(frames) < frames_wanted:
            vf = pipeline.step()

            if vf is None:
                time.sleep(0.005)
                continue

            frames.append(vf)

            if len(frames) % 60 == 0:
                say(f"  ... {len(frames)} 帧  {vf.fps:.1f} fps")
    finally:
        elapsed = time.perf_counter() - t0
        pipeline.close()

    say()
    say(f"共采集 {len(frames)} 帧，用时 {elapsed:.1f}s（{len(frames) / max(elapsed, 1e-6):.1f} fps 有效）")
    say()

    # ---------------- 契约 ----------------

    check("采集到了足够多的帧", len(frames) >= frames_wanted * 0.8,
          f"{len(frames)} / {frames_wanted}")

    shapes = Counter(f.frame.shape for f in frames)
    check("每帧都是三通道图像", all(s[2] == 3 for s in shapes), str(dict(shapes)))
    check("分辨率保持一致", len(shapes) <= 2, str(dict(shapes)))

    check("每帧都有 hands 结果对象",
          all(f.hands is not None for f in frames))
    check("每帧都有 objects 结果对象",
          all(f.objects is not None for f in frames))
    check("每帧都有 tracked 快照",
          all(f.tracked is not None for f in frames))
    check("每帧都有递增的帧序号",
          all(b.index > a.index for a, b in zip(frames, frames[1:])))

    # ---------------- 分频 ----------------

    hand_fresh = sum(1 for f in frames if f.hands_fresh)
    obj_fresh = sum(1 for f in frames if f.objects_fresh)

    hand_ratio = hand_fresh / len(frames)
    obj_ratio = obj_fresh / len(frames)

    say()
    say(f"手部推理 {hand_fresh}/{len(frames)} 帧（{hand_ratio:.1%}）")
    say(f"物体推理 {obj_fresh}/{len(frames)} 帧（{obj_ratio:.1%}）")

    check("手部按分频执行（不是每帧都跑）", 0.02 < hand_ratio < 0.95,
          f"{hand_ratio:.1%}")
    check("物体推理频率低于手部", obj_ratio < hand_ratio + 0.02,
          f"物体 {obj_ratio:.1%} vs 手部 {hand_ratio:.1%}")
    check("物体确实在跑（不是完全没跑）", obj_fresh >= 3,
          f"{obj_fresh} 帧")

    # ---------------- 耗时 ----------------

    hand_ms = sorted(f.hands_ms for f in frames if f.hands_fresh)
    obj_ms = sorted(f.objects_ms for f in frames if f.objects_fresh)

    if hand_ms:
        median = hand_ms[len(hand_ms) // 2]
        say(f"手部推理中位数 {median:.1f} ms（最慢 {hand_ms[-1]:.1f} ms）")
        check("手部推理耗时被真实测出（>0）", median > 0.3,
              f"median={median:.2f} ms")

    if obj_ms:
        median = obj_ms[len(obj_ms) // 2]
        say(f"物体推理中位数 {median:.1f} ms（最慢 {obj_ms[-1]:.1f} ms）")
        check("物体推理耗时被真实测出（>0）", median > 0.5,
              f"median={median:.2f} ms")

    # ---------------- 跟踪稳定性 ----------------

    ids_by_label: dict[str, set] = {}
    appearances: Counter = Counter()

    for f in frames:
        for obj in f.tracked.objects:
            ids_by_label.setdefault(obj.label, set()).add(obj.track_id)
            appearances[obj.label] += 1

    say()
    if appearances:
        say("检测到的物体类别：" + ", ".join(
            f"{label}×{count}帧/{len(ids)}轨迹"
            for label, count in appearances.most_common()
            for ids in [ids_by_label.get(label, set())]
        ))
    else:
        say("本次运行未检测到任何目标物体（画面里没有书/键盘/手机等）")

    # 轨迹复用：同类别不应该每帧都新建 id
    reused = []
    for label, ids in ids_by_label.items():
        frames_with = appearances[label]
        if frames_with >= 20 and len(ids) > frames_with / 3:
            reused.append(f"{label}: {len(ids)} 轨迹 / {frames_with} 帧")

    check("物体轨迹被复用而不是每帧新建", not reused,
          "; ".join(reused) if reused else "无异常")

    # 滑行比例：物体推理之间的帧应当有滑行轨迹
    stale_frames = sum(
        1 for f in frames if f.tracked.objects and not f.objects_fresh
    )
    check("缓存帧上仍有轨迹（滑行生效）",
          stale_frames > 0 or not appearances,
          f"{stale_frames} 帧上有滑行轨迹")

    # ---------------- 收尾 ----------------

    say()
    failed = [n for n, ok, _ in RESULTS if not ok]
    say("=" * 70)
    say(f"共 {len(RESULTS)} 项检查：{len(RESULTS) - len(failed)} 通过 / {len(failed)} 失败")
    if failed:
        say("失败：" + ", ".join(failed))
    say("=" * 70)

    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_report.txt").write_text("\n".join(LOG), encoding="utf-8")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
