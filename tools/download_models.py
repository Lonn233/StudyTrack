"""下载 YOLO 权重到 models/。

本机 HTTPS 有中间人代理，CA 不在 certifi 里，因此 urllib 默认会报
CERTIFICATE_VERIFY_FAILED、curl 报 35。这里显式使用不校验的 SSL
上下文并带超时，避免挂死。

    python tools/download_models.py
"""

from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

TARGETS = [
    (
        "yolo11n.pt",
        [
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
        ],
    ),
]

CTX = ssl._create_unverified_context()


def download(name: str, urls: list[str]) -> bool:
    dest = MODELS / name

    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"[skip] {name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    MODELS.mkdir(parents=True, exist_ok=True)

    for url in urls:
        print(f"[try ] {name} <- {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, context=CTX, timeout=90) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                chunks = []
                read = 0
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    read += len(chunk)
                    if total:
                        pct = read * 100 // total
                        print(f"\r       {pct:3d}%  {read / 1e6:5.1f}/{total / 1e6:.1f} MB",
                              end="", flush=True)
                print()

            data = b"".join(chunks)
            if len(data) < 100_000:
                print(f"[warn] suspiciously small ({len(data)} bytes), skip")
                continue

            dest.write_bytes(data)
            print(f"[ ok ] {name} saved ({len(data) / 1e6:.1f} MB)")
            return True

        except Exception as exc:
            print(f"[fail] {type(exc).__name__}: {exc}")

    return False


def main() -> int:
    ok = True
    for name, urls in TARGETS:
        if not download(name, urls):
            ok = False

    if not ok:
        print("\nSome models could not be downloaded.")
        print("Manual option: download yolo11n.pt in a browser and put it in models/")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
