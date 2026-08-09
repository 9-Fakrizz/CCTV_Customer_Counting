#!/usr/bin/env python3
"""
รันเทส Customer-Tracker แบบระบุชื่อคลิปง่ายๆ

ตัวอย่าง:
  python test_run.py input2.mp4
  python test_run.py input1.mp4 -c venues/example_salon/config.test10s.json
  python test_run.py input2.mp4 --preview
  python test_run.py input2.mp4 --no-preview --max-frames 500

อาร์กิวเมนต์หลังชื่อคลิปจะส่งต่อให้ `python -m customer_tracker` ทั้งหมด
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _resolve_video(name: str) -> Path:
    p = Path(name.strip())
    if p.is_absolute():
        return p.resolve()
    cand = (ROOT / p).resolve()
    if cand.is_file():
        return cand
    # ชื่อล้วน เช่น input2.mp4 -> videos/input2.mp4
    if p.parent == Path("."):
        cand = (ROOT / "videos" / p.name).resolve()
    return cand


def main() -> int:
    p = argparse.ArgumentParser(
        description="รัน customer_tracker กับคลิปที่ระบุ (ดีฟอลต์ config.fast.json)",
    )
    p.add_argument(
        "video",
        help="ชื่อไฟล์ เช่น input2.mp4 หรือ videos/input2.mp4",
    )
    p.add_argument(
        "-c",
        "--config",
        default="venues/example_salon/config.fast.json",
        help="path ไปยัง config.json (จากรากโปรเจกต์)",
    )
    args, passthrough = p.parse_known_args()

    video = _resolve_video(args.video)
    if not video.is_file():
        print(f"ไม่พบไฟล์คลิป: {video}", file=sys.stderr)
        print("  ใส่ชื่อในโฟลเดอร์ videos/ เช่น input2.mp4", file=sys.stderr)
        return 2

    cfg = (ROOT / args.config).resolve()
    if not cfg.is_file():
        print(f"ไม่พบ config: {cfg}", file=sys.stderr)
        return 2

    os.chdir(ROOT)
    cmd = [
        sys.executable,
        "-m",
        "customer_tracker",
        "--config",
        str(cfg),
        "--video",
        str(video),
    ]
    cmd.extend(passthrough)
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
