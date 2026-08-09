from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_venue_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Customer-Tracker: seat-based customer counting from CCTV video."
    )
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to venue config.json",
    )
    p.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="Override video file path or stream URL (optional)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (debug / quick test)",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Open a live window with overlay (dev machine). "
            "Forces stride=1. On Pi / headless, omit this flag. "
            "May need: pip install opencv-python"
        ),
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable overlay window even if config sets output.preview_on_run",
    )
    args = p.parse_args(argv)

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = load_venue_config(args.config.resolve())
    out = cfg.output or {}
    preview = (
        (args.preview or bool(out.get("preview_on_run", False)))
        and not args.no_preview
    )

    # Lazy import so `--help` works even without OpenCV installed.
    from .pipeline import run_from_config_path

    result = run_from_config_path(
        args.config.resolve(),
        override_video=str(args.video).strip() if args.video else None,
        max_frames=args.max_frames,
        preview=preview,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
