"""把 models/yolo11n.pt 导出为 models/yolo11n.onnx（静态输入尺寸）。

要点：
  * **不要** 带 NMS 导出（nms=True 会改变输出格式为 (1,300,6)），
    本项目的 ObjectDetector._run_onnx 期望原始输出 (1, 4+nc, N)。
  * imgsz 默认 320 —— 树莓派 4 上 640→320 约 4 倍加速，是单项最大收益。
  * opset 12 —— onnxruntime 1.20 / 1.30 都吃得下，兼容性最好。

    python tools/export_onnx.py [imgsz]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

IMGSZ = int(sys.argv[1]) if len(sys.argv) > 1 else 320
SRC = ROOT / "models" / "yolo11n.pt"
DST = ROOT / "models" / f"yolo11n.onnx"

print(f"[export] source = {SRC}  exists={SRC.exists()}")
if not SRC.exists():
    raise SystemExit("yolo11n.pt not found -- run tools/download_models.py first")

from ultralytics import YOLO  # noqa: E402

t0 = time.perf_counter()
model = YOLO(str(SRC))
out = model.export(
    format="onnx",
    imgsz=IMGSZ,
    opset=12,
    simplify=False,
    dynamic=False,
    nms=False,          # 关键：保留原始输出
    half=False,
)
print(f"[export] done in {time.perf_counter() - t0:.1f}s -> {out}")

# ultralytics 通常写在 pt 同目录，确保落在 models/
out_path = Path(out)
if not out_path.exists():
    raise SystemExit(f"export reported {out} but file missing")
if out_path.resolve() != DST.resolve():
    import shutil

    shutil.move(str(out_path), str(DST))

print(f"[export] final = {DST}  {DST.stat().st_size / 1e6:.2f} MB")

# --- 校验：用 onnxruntime 跑一张假图，打印输出形状 ---------------------
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(str(DST), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
dummy = np.zeros((1, 3, IMGSZ, IMGSZ), dtype=np.float32)
res = sess.run(None, {inp.name: dummy})[0]
print(f"[export] input='{inp.name}' shape={inp.shape}")
print(f"[export] output shape = {res.shape}  (expect (1, 84, N))")
