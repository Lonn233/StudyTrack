"""阶段自检 1：摄像头 + 配置 + 手部检测 + 物体检测。

不弹窗口（除必要外），把结果写到 out/stage1_report.txt，方便在
PowerShell 里查看。

    python tests/stage1_check.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG: list[str] = []


def say(msg: str = "") -> None:
    LOG.append(str(msg))
    print(msg)


def flush() -> None:
    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage1_report.txt").write_text("\n".join(LOG), encoding="utf-8")


results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    say(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


say("=" * 64)
say("Stage 1 check: config / camera / hands / objects")
say("=" * 64)

# --- config ----------------------------------------------------------
config = None
try:
    from study_assistant.config import Config

    config = Config.load(ROOT / "config" / "config.yaml")
    check(
        "config loads",
        True,
        f"camera={config.camera.get('width')}x{config.camera.get('height')}, "
        f"hands_interval={config.vision['hands']['interval_ms']}ms, "
        f"objects_backend={config.vision['objects']['backend']}",
    )
except Exception as exc:
    traceback.print_exc()
    check("config loads", False, repr(exc))

# --- calibration ------------------------------------------------------
try:
    from study_assistant.desk.roi_manager import Calibration, default_calibration_path

    calib_path = default_calibration_path()
    calib = Calibration.load(calib_path)
    check(
        "calibration loads",
        True,
        f"completed={calib.completed}, rois={list(calib.rois.keys())}",
    )
except Exception as exc:
    traceback.print_exc()
    check("calibration loads", False, repr(exc))

# --- camera -----------------------------------------------------------
packets = []
camera = None
try:
    from study_assistant.camera.camera_backend import create_camera

    camera = create_camera(config.data)
    camera.open()

    for _ in range(10):
        p = camera.read()
        if p is not None:
            packets.append(p)
        time.sleep(0.02)

    ok = len(packets) >= 5
    shape = packets[-1].frame.shape if packets else None
    fps = sum(p.fps for p in packets) / len(packets) if packets else 0
    check("camera produces frames", ok, f"{len(packets)} frames, shape={shape}, ~{fps:.1f} fps")
except Exception as exc:
    traceback.print_exc()
    check("camera produces frames", False, repr(exc))

# --- hands ------------------------------------------------------------
hand_det = None
try:
    from study_assistant.vision.hand_detector import HandDetector

    hand_det = HandDetector(
        model_path=str(ROOT / config.vision["hands"]["model_path"]),
        max_hands=config.vision["hands"]["max_hands"],
    )
    check("HandDetector initializes", hand_det.available)

    if packets:
        frame = packets[-1].frame
        t0 = time.perf_counter()
        hr = hand_det.process(frame)
        dt = (time.perf_counter() - t0) * 1000
        check(
            "hand inference runs",
            True,
            f"{hr.count} hand(s), {hr.inference_ms:.1f} ms (wall {dt:.0f} ms)",
        )

        # 连续多帧确认不崩
        for p in packets[-5:]:
            hand_det.process(p.frame)
        check("hand inference stable over frames", True, "5 more frames OK")
    else:
        check("hand inference runs", False, "no camera frames to test")
except Exception as exc:
    traceback.print_exc()
    check("HandDetector", False, repr(exc))

# --- objects ----------------------------------------------------------
obj_det = None
try:
    from study_assistant.vision.object_detector import ObjectDetector

    obj_cfg = config.vision["objects"]
    backend = obj_cfg.get("backend", "ultralytics")

    # ultralytics 需要 .pt；若配置写的是 onnx 但文件不存在，回退 ultralytics
    model_path = obj_cfg.get("model_path", "models/yolo11n.onnx")
    if backend == "onnx" and not (ROOT / model_path).exists():
        say(f"  note: {model_path} missing -> falling back to ultralytics backend")
        backend = "ultralytics"
        model_path = "models/yolo11n.pt"   # 本地已下载的权重

    if not Path(model_path).is_absolute():
        model_path = str(ROOT / model_path)

    obj_det = ObjectDetector(
        backend=backend,
        model_path=model_path,
        input_size=obj_cfg.get("input_size", 320),
        confidence=obj_cfg.get("confidence", 0.35),
    )
    check(
        "ObjectDetector initializes",
        obj_det.available,
        f"backend={backend} error={obj_det.last_error or 'none'}",
    )

    if obj_det.available and packets:
        t0 = time.perf_counter()
        res = obj_det.process(packets[-1].frame)
        dt = (time.perf_counter() - t0) * 1000

        found = ", ".join(
            f"{o.label}:{o.confidence:.2f}" for o in res.objects
        ) or "(none visible)"

        check(
            "object inference runs",
            True,
            f"{len(res.objects)} object(s) [{found}], {res.inference_ms:.0f} ms (wall {dt:.0f} ms)",
        )
    elif not obj_det.available:
        check("object inference runs", False, "detector unavailable")
    else:
        check("object inference runs", False, "no camera frames")
except Exception as exc:
    traceback.print_exc()
    check("ObjectDetector", False, repr(exc))

# --- cleanup ----------------------------------------------------------
for closer in (hand_det, obj_det):
    try:
        if closer is not None:
            closer.close()
    except Exception:
        pass

try:
    if camera is not None:
        camera.release()
except Exception:
    pass

say()
failed = [n for n, ok, _ in results if not ok]
say(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    say("FAILED: " + ", ".join(failed))
flush()
sys.exit(1 if failed else 0)
