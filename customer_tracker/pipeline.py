from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .config import VenueConfig, load_venue_config, project_root_from_config
from .detector import PersonDetector, resize_for_inference
from .geometry import (
    Point,
    axis_aligned_intersection_area,
    axis_aligned_iou,
    distance_point_to_polygon,
    norm_polygon_to_pixels,
    point_in_polygon,
    polygon_axis_aligned_bbox,
    seat_anchor_from_xyxy,
)
from .output import EventSink, OutputSettings
from .seat_engine import SeatEngine


def _anchor_for_detection(
    det: Any,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    mode: str,
) -> Tuple[float, float]:
    """
    Anchor point used for zone tests. If using a pose model and mode is 'head',
    prefer the pose head keypoint (det.head_xy) over bbox heuristics.
    """
    m = (mode or "center").strip().lower()
    if m == "head":
        hx = getattr(det, "head_xy", None)
        if hx is not None:
            return hx
    return seat_anchor_from_xyxy(x1, y1, x2, y2, m)


def _apply_detection_offset(dets: Any, ox: int, oy: int) -> Any:
    """Shift detections (xyxy/head/kpts) by crop offsets."""
    if not dets or (ox == 0 and oy == 0):
        return dets
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        d.xyxy = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
        hx = getattr(d, "head_xy", None)
        if hx is not None:
            d.head_xy = (hx[0] + ox, hx[1] + oy)
        kpts = getattr(d, "kpts", None)
        if kpts:
            d.kpts = [(kx + ox, ky + oy, kc) for (kx, ky, kc) in kpts]
    return dets


def _bbox_edge_distance(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
) -> float:
    """Euclidean distance between two axis-aligned bboxes (0 if they overlap)."""
    dx = 0.0
    if ax2 < bx1:
        dx = bx1 - ax2
    elif bx2 < ax1:
        dx = ax1 - bx2
    dy = 0.0
    if ay2 < by1:
        dy = by1 - ay2
    elif by2 < ay1:
        dy = ay1 - by2
    return float(math.hypot(dx, dy))


def _preview_available() -> bool:
    try:
        return callable(getattr(cv2, "imshow", None))
    except Exception:  # pragma: no cover
        return False


def _resize_for_preview(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    nh = int(round(h * (max_width / float(w))))
    return cv2.resize(frame, (max_width, nh))


def _preview_show_frame(
    frame: np.ndarray,
    delay_ms: int,
    window: str = "Customer-Tracker",
) -> Tuple[bool, float]:
    """
    Show one frame in an OpenCV window.
    Returns (continue_running, seek_seconds): seek_seconds != 0 means jump in the clip.
    Requires full `opencv-python` (not headless). On failure, logs once and returns (True, 0.0).
    """
    if not _preview_available():
        if not getattr(_preview_show_frame, "_bad_backend_warned", False):
            print(
                "Preview: OpenCV has no GUI (use: pip install opencv-python).",
                file=sys.stderr,
            )
            _preview_show_frame._bad_backend_warned = True  # type: ignore[attr-defined]
        return True, 0.0
    try:
        disp = _resize_for_preview(frame)
        cv2.imshow(window, disp)
        key = cv2.waitKey(max(1, delay_ms)) & 0xFF
        if key == ord("q") or key == 27:
            return False, 0.0
        if key == ord("]"):
            return True, 10.0
        if key == ord("["):
            return True, -10.0
        if key == ord(".") or key == ord(">"):
            return True, 2.0
        if key == ord(",") or key == ord("<"):
            return True, -2.0
        if key == ord("f"):
            return True, 5.0
        if key == ord("b"):
            return True, -5.0
        return True, 0.0
    except Exception as e:  # pragma: no cover - headless / no display
        if not getattr(_preview_show_frame, "_fail_warned", False):
            print(
                "Preview failed (install opencv-python or connect a display). "
                "Continuing without window.",
                file=sys.stderr,
            )
            print(f"  ({e})", file=sys.stderr)
            _preview_show_frame._fail_warned = True  # type: ignore[attr-defined]
        return True, 0.0


def _foot_points_one_per_seat(
    seat_candidates: Dict[str, List[Tuple[float, float, float, float]]],
) -> List[Tuple[float, float]]:
    """
    ต่อเก้าอี้หนึ่งที่ ถ้ามีหลายคนในโซน (เช่น ช่าง + ลูกค้า) เลือกจุดเดียว:
    กล่องที่พื้นที่เล็กสุด (มักเป็นลูกค้านั่ง) แล้วตามด้วย aspect ต่ำสุด (นั่งมากกว่า)
    """
    foot_points: List[Tuple[float, float]] = []
    for _sid, cands in seat_candidates.items():
        if not cands:
            continue
        if len(cands) == 1:
            fx, fy, _, _ = cands[0]
            foot_points.append((fx, fy))
            continue
        fx, fy, _, _ = min(cands, key=lambda t: (t[2], t[3]))
        foot_points.append((fx, fy))
    return foot_points


def _parse_standing_aspect_min(rules: Dict[str, Any]) -> Optional[float]:
    """
    bbox สูง/กว้าง (h/w) สูง = มักเป็นคนยืน (ช่าง) — ไม่นับเป็นลูกค้าในเก้าอี้
    None = ปิด heuristics นี้; ค่าเริ่ม 2.0; ใส่ 0 ใน config = ปิด
    """
    raw = rules.get("standing_aspect_min")
    if raw is None:
        return 2.0
    try:
        v = float(raw)
        if v <= 0:
            return None
        return v
    except (TypeError, ValueError):
        return 2.0


def _event_time_from_video_sec(
    video_sec: float,
    tz: ZoneInfo,
    anchor: Optional[datetime] = None,
    *,
    realtime: bool = False,
) -> datetime:
    if realtime:
        return datetime.now(tz)
    if anchor is not None:
        return anchor + timedelta(seconds=float(video_sec))
    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start + timedelta(seconds=float(video_sec))


def _is_stream_source(raw: str) -> bool:
    s = str(raw or "").strip().lower()
    return (
        s.startswith("rtsp://")
        or s.startswith("rtsps://")
        or s.startswith("http://")
        or s.startswith("https://")
    )


def _source_display(src: Union[str, Path]) -> str:
    return str(src) if isinstance(src, str) else str(src.resolve())


def _open_capture(src: Union[str, Path]) -> cv2.VideoCapture:
    return cv2.VideoCapture(str(src))


def run_venue(
    cfg: VenueConfig,
    project_root: Path,
    override_video: Optional[Union[str, Path]] = None,
    max_frames: Optional[int] = None,
    preview: bool = False,
) -> Dict[str, Any]:
    root = project_root.resolve()
    tz = ZoneInfo(cfg.timezone)
    anchor_iso = (cfg.output or {}).get("session_anchor_iso")
    anchor: Optional[datetime] = None
    if anchor_iso:
        anchor = datetime.fromisoformat(str(anchor_iso))
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=tz)

    src_path = override_video or cfg.resolved_source_path(root)
    src: Union[str, Path] = src_path if not isinstance(src_path, str) else src_path.strip()

    declared_type = str((cfg.source_type or "")).strip().lower()
    raw_src = str((cfg.source_path or "")).strip()
    is_stream = bool(
        declared_type in ("rtsp", "stream", "camera", "url") or _is_stream_source(raw_src)
    )
    if override_video is not None:
        # Explicit override wins (can be a file path or a URL like rtsp://...)
        is_stream = bool(isinstance(src, str) and _is_stream_source(src))
    elif is_stream and raw_src:
        src = raw_src
    else:
        if not isinstance(src, Path) or not src.is_file():
            raise FileNotFoundError(f"Video/source not found: {src}")

    cap = _open_capture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {_source_display(src)}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 0:
        total_frames = 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1e-3:
        fps = 25.0

    inf = cfg.inference or {}
    max_w = int(inf.get("max_width", 960))
    stride = max(1, int(inf.get("process_stride", 3)))
    model = str(inf.get("model", "yolo11n.pt"))
    conf = float(inf.get("conf", 0.35))
    imgsz = int(inf.get("imgsz", 640))
    iou = float(inf.get("iou", 0.5))
    max_det = int(inf.get("max_det", 50))

    detector = PersonDetector(
        model_name=model,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        max_det=max_det,
    )

    if preview:
        print(
            "Preview: window 'Customer-Tracker' — q=quit  [ ] -10s/+10s  , . -2s/+2s  b/f -5s/+5s",
            flush=True,
        )

    rules = cfg.rules or {}
    min_occ = float(rules.get("min_occupancy_sec", 300))
    empty_db = float(rules.get("empty_debounce_sec", 90))
    grace = float(rules.get("grace_absent_sec", 10))
    seat_anchor_mode = str(rules.get("seat_anchor", "center")).strip().lower()
    min_h_frac = rules.get("min_box_height_frac")
    min_box_h_frac = float(min_h_frac) if min_h_frac is not None else None
    max_ar = rules.get("max_box_aspect_height_width")
    max_box_aspect = float(max_ar) if max_ar is not None else None
    standing_aspect_min = _parse_standing_aspect_min(rules)
    require_standing_nearby = bool(rules.get("require_standing_nearby", False))
    standing_nearby_dist_norm = float(rules.get("standing_nearby_dist_norm", 0.12))
    staff_aspect_min = float(rules.get("staff_aspect_min", 0.8))
    require_blue_nearby = bool(rules.get("require_blue_nearby", False))
    blue_nearby_dist_norm = float(
        rules.get("blue_nearby_dist_norm", rules.get("near_customer_pair_dist_norm", 0.22))
    )
    # Gate mode: either "standing nearby" or "blue nearby"
    stand_gate = bool(require_standing_nearby and (not require_blue_nearby))

    mscaf = rules.get("min_seat_customer_box_area_frac")
    min_seat_customer_area_frac = (
        float(mscaf) if mscaf is not None else None
    )
    mscconf = rules.get("min_seat_customer_conf")
    min_seat_customer_conf = float(mscconf) if mscconf is not None else 0.0
    msna = rules.get("min_standing_nearby_box_area_frac")
    min_standing_nearby_area_frac = (
        float(msna) if msna is not None else None
    )
    synthetic_seat_box = bool(rules.get("synthetic_seat_box_for_pairing", True))
    proxy_occ = bool(rules.get("proxy_seat_occupancy_from_barber", True))
    proxy_barber_iou = float(rules.get("proxy_barber_seat_iou_min", 0.04))
    require_real_customer = bool(rules.get("require_real_customer_for_counting", True))
    customer_persist_sec = float(rules.get("customer_persist_sec", 1.5))
    blue_persist_sec = float(rules.get("blue_persist_sec", 0.9))
    if blue_persist_sec < 0:
        blue_persist_sec = 0.0
    preview_speed = float(rules.get("preview_speed", 1.0))
    if preview_speed <= 0:
        preview_speed = 1.0

    out_cfg = cfg.output or {}
    data_dir = cfg.resolved_data_dir(root)
    sink = EventSink(
        OutputSettings(
            data_dir=data_dir,
            venue_id=cfg.venue_id,
            timezone=cfg.timezone,
            write_events_csv=bool(out_cfg.get("write_events_csv", True)),
            write_daily_csv=bool(out_cfg.get("write_daily_csv", True)),
            write_daily_xlsx=bool(out_cfg.get("write_daily_xlsx", True)),
        )
    )

    debug_path = cfg.resolved_debug_video_path(root)
    writer: Optional[cv2.VideoWriter] = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    engine: Optional[SeatEngine] = None

    frame_idx = 0
    processed = 0
    total_counts = 0
    t0_wall = time.perf_counter()
    last_standing_nearby: Dict[str, bool] = {}
    last_waiting_barber: Dict[str, bool] = {}
    last_blue_nearby: Dict[str, bool] = {}
    last_blue_seen_sec: Dict[str, float] = {}
    # Short persistence so customer doesn't flicker when detector misses a few frames.
    last_sitter_seen_sec: Dict[str, float] = {}
    last_sitter_anchor: Dict[str, Tuple[float, float]] = {}
    last_dets: Any = []
    last_foot_points: List[Tuple[float, float]] = []

    # Reconnect settings for realtime streams (best-effort).
    reconnect_cfg = (cfg.inference or {}).get("reconnect") or {}
    reconnect_enabled = bool(reconnect_cfg.get("enabled", True)) if is_stream else False
    reconnect_backoff_sec = float(reconnect_cfg.get("backoff_sec", 1.0))
    reconnect_max_tries = int(reconnect_cfg.get("max_tries", 0))  # 0 = infinite
    reconnect_tries = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            if not reconnect_enabled:
                break
            reconnect_tries += 1
            if reconnect_max_tries > 0 and reconnect_tries > reconnect_max_tries:
                break
            cap.release()
            time.sleep(max(0.1, reconnect_backoff_sec))
            cap = _open_capture(src)
            if not cap.isOpened():
                continue
            continue
        if max_frames is not None and frame_idx >= max_frames:
            break

        loop_start = time.perf_counter()
        h, w = frame.shape[:2]
        video_sec = frame_idx / fps

        if frame_idx % stride != 0:
            frame_idx += 1
            if writer is not None or preview:
                _draw_overlay(
                    frame,
                    cfg,
                    w,
                    h,
                    last_dets,
                    engine,
                    last_foot_points,
                    video_sec,
                    min_occ,
                    stride,
                    fps,
                    preview_hint=preview,
                    seat_anchor_mode=seat_anchor_mode,
                    standing_aspect_min=standing_aspect_min,
                    require_standing_nearby=require_standing_nearby,
                    standing_nearby_map=last_standing_nearby,
                    waiting_barber_map=last_waiting_barber,
                    blue_nearby_map=last_blue_nearby,
                )
                if writer is not None:
                    writer.write(frame)
            if preview:
                # Show video at normal speed even when skipping inference frames.
                elapsed = time.perf_counter() - loop_start
                slot_ms = (1000.0 / fps) / preview_speed
                wait_ms = max(1, int(round(slot_ms - elapsed * 1000.0)))
                run, seek_sec = _preview_show_frame(frame, wait_ms)
                if not run:
                    break
                if seek_sec != 0:
                    nf = frame_idx + int(round(seek_sec * fps))
                    if total_frames > 0:
                        nf = max(0, min(nf, total_frames - 1))
                    else:
                        nf = max(0, nf)
                    if max_frames is not None:
                        nf = min(nf, max_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, nf)
                    frame_idx = nf
                    continue
            continue

        # Optional ROI crop before inference to save compute/memory (useful on Raspberry Pi).
        # inference.roi_norm = [x1_norm, y1_norm, x2_norm, y2_norm] in 0..1
        roi = inf.get("roi_norm")
        ox = 0
        oy = 0
        infer_frame = frame
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            try:
                rx1, ry1, rx2, ry2 = [float(v) for v in roi]
                rx1 = max(0.0, min(1.0, rx1))
                ry1 = max(0.0, min(1.0, ry1))
                rx2 = max(0.0, min(1.0, rx2))
                ry2 = max(0.0, min(1.0, ry2))
                if rx2 > rx1 and ry2 > ry1:
                    px1 = int(round(rx1 * w))
                    py1 = int(round(ry1 * h))
                    px2 = int(round(rx2 * w))
                    py2 = int(round(ry2 * h))
                    px1 = max(0, min(px1, w - 1))
                    py1 = max(0, min(py1, h - 1))
                    px2 = max(px1 + 1, min(px2, w))
                    py2 = max(py1 + 1, min(py2, h))
                    ox, oy = px1, py1
                    infer_frame = frame[py1:py2, px1:px2]
            except Exception:
                infer_frame = frame
                ox = 0
                oy = 0

        resized, scale = resize_for_inference(infer_frame, max_w)
        sx, sy = scale
        dets = detector.detect_persons(resized, (sx, sy))
        dets = _apply_detection_offset(dets, ox, oy)
        last_dets = dets

        seat_polys = {
            s["id"]: norm_polygon_to_pixels(s["polygon_norm"], w, h)
            for s in cfg.seats
        }
        if not seat_polys:
            raise ValueError("config must define at least one seat")

        staff_polys = [
            norm_polygon_to_pixels(z, w, h) for z in cfg.staff_zones_norm
        ]
        exclude_polys = [
            norm_polygon_to_pixels(z, w, h) for z in cfg.exclude_zones_norm
        ]
        fh = float(h)
        standing_nearby: Dict[str, bool] = {sid: False for sid in seat_polys}
        if stand_gate:
            nearby_px = standing_nearby_dist_norm * min(float(w), float(h))
            for d in dets:
                x1, y1, x2, y2 = d.xyxy
                bh = max(1e-6, y2 - y1)
                bw = max(1e-6, x2 - x1)
                aspect = bh / bw
                if min_box_h_frac is not None and bh / fh < min_box_h_frac:
                    continue
                if max_box_aspect is not None and aspect > max_box_aspect:
                    continue
                # use softer threshold for staff-nearby (staff can be wide when bending/arms)
                if aspect < staff_aspect_min:
                    continue
                if min_standing_nearby_area_frac is not None:
                    afrac = (bh * bw) / (float(w) * float(h))
                    if afrac < min_standing_nearby_area_frac:
                        continue
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if any(point_in_polygon(cx, cy, ep) for ep in exclude_polys):
                    continue
                if staff_polys and any(
                    point_in_polygon(cx, cy, sp) for sp in staff_polys
                ):
                    continue
                # If this detection is the seated customer inside a seat zone, it must NOT satisfy
                # the "standing nearby" gate. But a barber can overlap the seat zone while working,
                # so we should not require the point to be outside the seat polygon.
                sit_thr = (
                    standing_aspect_min if standing_aspect_min is not None else 2.0
                )
                ax, ay = _anchor_for_detection(d, x1, y1, x2, y2, seat_anchor_mode)
                is_sitter_in_seat = False
                if aspect < sit_thr:
                    for _sid, poly0 in seat_polys.items():
                        if point_in_polygon(ax, ay, poly0):
                            is_sitter_in_seat = True
                            break
                if is_sitter_in_seat:
                    continue
                for sid, poly in seat_polys.items():
                    if distance_point_to_polygon(cx, cy, poly) <= nearby_px:
                        standing_nearby[sid] = True

        seat_candidates: Dict[str, List[Tuple[float, float, float, float]]] = (
            defaultdict(list)
        )
        # Tracks seats with a detected seated customer (before any proxy occupancy injection).
        seat_has_sitter: Dict[str, bool] = {sid: False for sid in seat_polys}
        # Keep best seated-customer bbox per seat for "blue nearby" gate.
        best_customer_bbox: Dict[str, Tuple[float, float, float, float]] = {}
        best_customer_key: Dict[str, Tuple[float, float]] = {}
        best_customer_det_idx: Dict[str, int] = {}
        for det_idx, d in enumerate(dets):
            x1, y1, x2, y2 = d.xyxy
            bh = max(1e-6, y2 - y1)
            bw = max(1e-6, x2 - x1)
            aspect = bh / bw
            area = bh * bw
            if min_seat_customer_conf > 0.0 and float(getattr(d, "conf", 0.0)) < min_seat_customer_conf:
                continue
            if min_box_h_frac is not None and bh / fh < min_box_h_frac:
                continue
            if max_box_aspect is not None and aspect > max_box_aspect:
                continue
            fx, fy = _anchor_for_detection(d, x1, y1, x2, y2, seat_anchor_mode)
            if any(point_in_polygon(fx, fy, ep) for ep in exclude_polys):
                continue
            if staff_polys and any(
                point_in_polygon(fx, fy, sp) for sp in staff_polys
            ):
                continue
            matched_sid: Optional[str] = None
            standing_in_seat = False
            for sid, poly in seat_polys.items():
                if point_in_polygon(fx, fy, poly):
                    matched_sid = sid
                    if (
                        standing_aspect_min is not None
                        and aspect >= standing_aspect_min
                    ):
                        standing_in_seat = True
                    break
            if matched_sid is None:
                continue
            if standing_in_seat:
                continue
            if min_seat_customer_area_frac is not None:
                afrac = (bh * bw) / (float(w) * float(h))
                if afrac < min_seat_customer_area_frac:
                    continue
            seat_candidates[matched_sid].append((fx, fy, area, aspect))
            seat_has_sitter[matched_sid] = True
            last_sitter_seen_sec[matched_sid] = float(video_sec)
            last_sitter_anchor[matched_sid] = (float(fx), float(fy))
            cur = best_customer_key.get(matched_sid)
            k = (area, aspect)
            if cur is None or k < cur:
                best_customer_key[matched_sid] = k
                best_customer_bbox[matched_sid] = (float(x1), float(y1), float(x2), float(y2))
                # remember which detection index is the seated customer for this seat
                best_customer_det_idx[matched_sid] = int(det_idx)

        # Persistence: if we saw a sitter very recently, keep the seat as "has sitter"
        # and (optionally) reuse last anchor so timing doesn't flicker.
        seat_has_sitter_persist: Dict[str, bool] = {}
        for sid in seat_polys:
            last_t = last_sitter_seen_sec.get(sid)
            seat_has_sitter_persist[sid] = bool(
                seat_has_sitter.get(sid, False)
                or (
                    last_t is not None
                    and (float(video_sec) - float(last_t)) <= customer_persist_sec
                )
            )
        for sid in seat_polys:
            if seat_candidates.get(sid):
                continue
            if not seat_has_sitter_persist.get(sid, False):
                continue
            if sid in last_sitter_anchor:
                ax, ay = last_sitter_anchor[sid]
                seat_candidates[sid].append((ax, ay, float(w * h) * 0.0005, 1.5))

        # Overlay-only: sitter present (or persisted) but no nearby barber yet.
        waiting_barber: Dict[str, bool] = {
            sid: bool(seat_has_sitter_persist.get(sid, False))
            and (not standing_nearby.get(sid, False))
            for sid in seat_polys
        }

        # Blue-nearby gate: compute per-seat whether the max-intersection "blue" person is near the customer.
        blue_nearby: Dict[str, bool] = {sid: False for sid in seat_polys}
        if require_blue_nearby:
            for sid in seat_polys:
                if not seat_has_sitter_persist.get(sid, False):
                    continue
                cb = best_customer_bbox.get(sid)
                if cb is None:
                    continue
                sx1, sy1, sx2, sy2 = cb
                best_inter = 0.0
                best_box: Optional[Tuple[float, float, float, float]] = None
                best_dist = 1e18
                # Skip the seated customer's own bbox; otherwise it "wins" by intersecting itself.
                cust_idx = best_customer_det_idx.get(sid)
                for idx, d in enumerate(dets):
                    x1, y1, x2, y2 = d.xyxy
                    if cust_idx is not None and idx == int(cust_idx):
                        continue
                    if (
                        (x1, y1, x2, y2) == (sx1, sy1, sx2, sy2)
                        or axis_aligned_iou(x1, y1, x2, y2, sx1, sy1, sx2, sy2) > 0.999
                    ):
                        continue
                    inter = axis_aligned_intersection_area(sx1, sy1, sx2, sy2, x1, y1, x2, y2)
                    dist = _bbox_edge_distance(sx1, sy1, sx2, sy2, x1, y1, x2, y2)
                    if inter > best_inter or (inter == best_inter and dist < best_dist):
                        best_inter = inter
                        best_box = (float(x1), float(y1), float(x2), float(y2))
                        best_dist = dist
                if best_box is None:
                    continue
                # Per requirement: overlapping is enough; treat "touching" as overlap too.
                if best_inter > 0.0 or best_dist <= 1e-6:
                    blue_nearby[sid] = True
                    last_blue_seen_sec[sid] = float(video_sec)
                else:
                    blue_nearby[sid] = False

            # Persistence: avoid flicker when detections drop briefly.
            if blue_persist_sec > 0.0:
                for sid in seat_polys:
                    if blue_nearby.get(sid, False):
                        continue
                    tlast = last_blue_seen_sec.get(sid)
                    if tlast is None:
                        continue
                    if (float(video_sec) - float(tlast)) <= blue_persist_sec:
                        blue_nearby[sid] = True

            # waiting = sitter but blue not nearby
            waiting_barber = {
                sid: bool(seat_has_sitter_persist.get(sid, False))
                and (not blue_nearby.get(sid, False))
                for sid in seat_polys
            }

        if proxy_occ and standing_aspect_min is not None:
            for sid, poly in seat_polys.items():
                if seat_candidates.get(sid):
                    continue
                vx1, vy1, vx2, vy2 = polygon_axis_aligned_bbox(poly)
                for d in dets:
                    x1, y1, x2, y2 = d.xyxy
                    bh = max(1e-6, y2 - y1)
                    bw = max(1e-6, x2 - x1)
                    asp = bh / bw
                    if min_box_h_frac is not None and bh / fh < min_box_h_frac:
                        continue
                    if max_box_aspect is not None and asp > max_box_aspect:
                        continue
                    if asp < staff_aspect_min:
                        continue
                    iou = axis_aligned_iou(
                        x1, y1, x2, y2, vx1, vy1, vx2, vy2
                    )
                    if iou < proxy_barber_iou:
                        continue
                    px_c = sum(p[0] for p in poly) / len(poly)
                    py_c = sum(p[1] for p in poly) / len(poly)
                    seat_candidates[sid].append(
                        (px_c, py_c, float(w * h) * 0.001, 1.0)
                    )
                    break

        if stand_gate:
            for sid in list(seat_candidates.keys()):
                if not standing_nearby.get(sid, False):
                    del seat_candidates[sid]
                    continue
                if require_real_customer and (not seat_has_sitter_persist.get(sid, False)):
                    del seat_candidates[sid]
            last_standing_nearby = standing_nearby.copy()
            last_waiting_barber = waiting_barber.copy()
            last_blue_nearby = blue_nearby.copy()
        elif require_blue_nearby:
            for sid in list(seat_candidates.keys()):
                if require_real_customer and (not seat_has_sitter_persist.get(sid, False)):
                    del seat_candidates[sid]
                    continue
                if not blue_nearby.get(sid, False):
                    del seat_candidates[sid]
            last_blue_nearby = blue_nearby.copy()
            last_waiting_barber = waiting_barber.copy()
        elif require_real_customer:
            # Even without the standing gate, never count proxy-only seats.
            for sid in list(seat_candidates.keys()):
                if not seat_has_sitter_persist.get(sid, False):
                    del seat_candidates[sid]

        foot_points = _foot_points_one_per_seat(seat_candidates)
        last_foot_points = foot_points

        if engine is None:
            engine = SeatEngine.from_polygons(
                seat_polys,
                min_occ,
                empty_db,
                grace,
            )

        dt = stride / fps
        occupied = engine.assign_foot_points(foot_points)
        counted = engine.step(dt, occupied)
        for sid in counted:
            total_counts += 1
            ev_time = _event_time_from_video_sec(
                video_sec, tz, anchor, realtime=is_stream
            )
            sink.record_customer_count(sid, when=ev_time)

        should_draw = (debug_path is not None) or preview
        if should_draw:
            if debug_path is not None:
                if writer is None:
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(debug_path),
                        fourcc,
                        fps,
                        (w, h),
                    )
            _draw_overlay(
                frame,
                cfg,
                w,
                h,
                dets,
                engine,
                foot_points,
                video_sec,
                min_occ,
                stride,
                fps,
                preview_hint=preview,
                seat_anchor_mode=seat_anchor_mode,
                standing_aspect_min=standing_aspect_min,
                require_standing_nearby=require_standing_nearby,
                standing_nearby_map=standing_nearby if stand_gate else last_standing_nearby,
                waiting_barber_map=waiting_barber if stand_gate else last_waiting_barber,
                blue_nearby_map=blue_nearby if require_blue_nearby else last_blue_nearby,
            )
            if writer is not None:
                writer.write(frame)
            if preview:
                elapsed = time.perf_counter() - loop_start
                slot_ms = (1000.0 / fps) / preview_speed
                wait_ms = max(1, int(round(slot_ms - elapsed * 1000.0)))
                run, seek_sec = _preview_show_frame(frame, wait_ms)
                if not run:
                    break
                if seek_sec != 0:
                    nf = frame_idx + int(round(seek_sec * fps))
                    if total_frames > 0:
                        nf = max(0, min(nf, total_frames - 1))
                    else:
                        nf = max(0, nf)
                    if max_frames is not None:
                        nf = min(nf, max_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, nf)
                    frame_idx = nf
                    continue

        processed += 1
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if preview:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    elapsed = time.perf_counter() - t0_wall
    return {
        "frames": frame_idx,
        "processed_frames": processed,
        "customer_events": total_counts,
        "elapsed_sec": elapsed,
        "source": _source_display(src),
        "debug_video": str(debug_path) if debug_path else None,
        "data_dir": str(data_dir),
        "preview": preview,
    }


def _put_text_large(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    fg: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """White/light text with dark outline for readability on busy scenes."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    outline = max(3, thickness + 2)
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0:
                continue
            cv2.putText(
                frame,
                text,
                (x + dx, y + dy),
                font,
                font_scale,
                (0, 0, 0),
                outline,
                cv2.LINE_AA,
            )
    cv2.putText(frame, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


def _collect_sitting_customer_boxes(
    dets: Any,
    seat_id_polys: List[Tuple[str, List[Point]]],
    seat_anchor_mode: str,
    standing_aspect_min: Optional[float],
    exclude_polys: List[List[Point]],
    staff_polys: List[List[Point]],
    fh: float,
    w: int,
    h: int,
    min_box_h_frac: Optional[float],
    max_box_aspect: Optional[float],
    min_box_area_frac: Optional[float],
    min_conf: float = 0.0,
    synthetic_seat_box: bool = True,
) -> List[Tuple[str, float, float, float, float, bool]]:
    """
    กล่องคนที่นั่งในโซนเก้าอี้ (aspect ต่ำ) — ใช้จับคู่กับคนยืน (ระยะ / IoU)
    bool สุดท้าย: True = กล่อง proxy จาก polygon เมื่อไม่มี detection (เช่น ก้มหัว/ผ้าคลุม)
    """
    sit_thr = (
        standing_aspect_min if standing_aspect_min is not None else 2.0
    )
    fram = float(w) * float(h)
    out: List[Tuple[str, float, float, float, float, bool]] = []
    for d in dets:
        if min_conf > 0.0 and float(getattr(d, "conf", 0.0)) < min_conf:
            continue
        x1, y1, x2, y2 = d.xyxy
        bh = max(1e-6, float(y2 - y1))
        bw = max(1e-6, float(x2 - x1))
        aspect = bh / bw
        if min_box_h_frac is not None and bh / fh < min_box_h_frac:
            continue
        if max_box_aspect is not None and aspect > max_box_aspect:
            continue
        ax, ay = _anchor_for_detection(d, x1, y1, x2, y2, seat_anchor_mode)
        if any(point_in_polygon(ax, ay, ep) for ep in exclude_polys):
            continue
        if staff_polys and any(point_in_polygon(ax, ay, sp) for sp in staff_polys):
            continue
        for sid, poly in seat_id_polys:
            if point_in_polygon(ax, ay, poly):
                if aspect < sit_thr:
                    if min_box_area_frac is not None:
                        afrac = (bh * bw) / fram
                        if afrac < min_box_area_frac:
                            break
                    out.append((sid, x1, y1, x2, y2, False))
                break
    if synthetic_seat_box:
        found_sid = {t[0] for t in out}
        for sid, poly in seat_id_polys:
            if sid in found_sid or len(poly) < 3:
                continue
            vx1, vy1, vx2, vy2 = polygon_axis_aligned_bbox(poly)
            out.append((sid, vx1, vy1, vx2, vy2, True))
    return out


def _near_customer_barber_labels_by_max_iou(
    dets: Any,
    sitting_boxes: List[Tuple[str, float, float, float, float, bool]],
    standing_aspect_min: Optional[float],
    pair_px: float,
    near_customer_pair_min_iou: Optional[float],
    proxy_barber_overlap_min: float,
    fh: float,
    min_box_h_frac: Optional[float],
    max_box_aspect: Optional[float],
    min_standing_area_frac: Optional[float],
    w: int,
    h: int,
) -> Dict[int, Tuple[str, float, float, float]]:
    """
    ต่อเก้าอี้ละคน: คนยืนที่ทับกับกล่องนั่งมากที่สุด
    กล่องจริง: ใช้ IoU + จำกัดระยะกลางกล่อง
    กล่อง proxy โซน: ใช้ (พื้นที่ทับ / พื้นที่กล่องช่าง) — ไม่ยึดระยะ
    """
    out: Dict[int, Tuple[str, float, float, float]] = {}
    if not sitting_boxes or standing_aspect_min is None:
        return out
    fram = float(w) * float(h)
    min_iou = (
        float(near_customer_pair_min_iou)
        if near_customer_pair_min_iou is not None
        else 0.01
    )
    winners: Dict[int, Tuple[str, float, float, float]] = {}

    for sid, sx1, sy1, sx2, sy2, is_virtual in sitting_boxes:
        scx = (sx1 + sx2) / 2.0
        scy = (sy1 + sy2) / 2.0
        best_score = -1.0
        best_idx: Optional[int] = None
        for idx, d in enumerate(dets):
            x1, y1, x2, y2 = d.xyxy
            bh = max(1e-6, float(y2 - y1))
            bw = max(1e-6, float(x2 - x1))
            aspect = bh / bw
            if min_box_h_frac is not None and bh / fh < min_box_h_frac:
                continue
            if max_box_aspect is not None and aspect > max_box_aspect:
                continue
            if aspect < standing_aspect_min:
                continue
            if min_standing_area_frac is not None:
                if (bh * bw) / fram < min_standing_area_frac:
                    continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if not is_virtual and math.hypot(cx - scx, cy - scy) > pair_px:
                continue
            inter = axis_aligned_intersection_area(
                sx1, sy1, sx2, sy2, x1, y1, x2, y2
            )
            barber_a = (x2 - x1) * (y2 - y1)
            if is_virtual:
                score = inter / (barber_a + 1e-9)
            else:
                score = axis_aligned_iou(x1, y1, x2, y2, sx1, sy1, sx2, sy2)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            continue
        thr = proxy_barber_overlap_min if is_virtual else min_iou
        if best_score < thr:
            continue
        prev = winners.get(best_idx)
        if prev is None or best_score > prev[1]:
            winners[best_idx] = (sid, best_score, scx, scy)

    for idx, (sid, score, scx, scy) in winners.items():
        out[idx] = (f"near customer {sid}", score, scx, scy)
    return out


def _role_label_and_color(
    ax: float,
    ay: float,
    aspect_ratio: float,
    seat_id_polys: List[Tuple[str, List[Point]]],
    staff_polys: List[List[Point]],
    exclude_polys: List[List[Point]],
    standing_aspect_min: Optional[float],
) -> Tuple[str, Tuple[int, int, int]]:
    """
    exclude > staff zone (ถ้ามีใน config) > โซนเก้าอี้:
    ในโซนเก้าอี้ — กล่องสูงแคบ (aspect สูง) มักเป็นช่างยืน; ต่ำกว่าเกณฑ์ = ลูกค้านั่ง
    """
    for ep in exclude_polys:
        if point_in_polygon(ax, ay, ep):
            return ("excluded", (200, 150, 255))
    for sp in staff_polys:
        if point_in_polygon(ax, ay, sp):
            return ("barber", (0, 200, 255))
    for sid, poly in seat_id_polys:
        if point_in_polygon(ax, ay, poly):
            if (
                standing_aspect_min is not None
                and aspect_ratio >= standing_aspect_min
            ):
                return ("barber", (0, 200, 255))
            return (f"customer {sid}", (0, 255, 120))
    return ("other", (200, 200, 200))


def _best_customer_idx_by_seat(
    dets: Any,
    seat_id_polys: List[Tuple[str, List[Point]]],
    seat_anchor_mode: str,
    standing_aspect_min: Optional[float],
    staff_polys: List[List[Point]],
    exclude_polys: List[List[Point]],
    fh: float,
    min_box_h_frac: Optional[float],
    max_box_aspect: Optional[float],
    min_box_area_frac: Optional[float],
    min_conf: float,
    w: int,
    h: int,
) -> Dict[str, int]:
    """
    Overlay-only: choose at most 1 customer bbox per seat.
    Criteria: inside seat polygon, not excluded/staff, not standing (aspect < threshold),
    then pick the smallest area (and then smallest aspect) to bias towards seated customer.
    """
    sit_thr = standing_aspect_min if standing_aspect_min is not None else 2.0
    fram = float(w) * float(h)
    best: Dict[str, Tuple[int, float, float]] = {}
    for idx, d in enumerate(dets):
        if min_conf > 0.0 and float(getattr(d, "conf", 0.0)) < min_conf:
            continue
        x1, y1, x2, y2 = d.xyxy
        bh = max(1e-6, float(y2 - y1))
        bw = max(1e-6, float(x2 - x1))
        aspect = bh / bw
        area = bh * bw
        if min_box_h_frac is not None and bh / fh < min_box_h_frac:
            continue
        if max_box_aspect is not None and aspect > max_box_aspect:
            continue
        if aspect >= sit_thr:
            continue
        if min_box_area_frac is not None and (area / fram) < min_box_area_frac:
            continue
        ax, ay = _anchor_for_detection(d, x1, y1, x2, y2, seat_anchor_mode)
        if any(point_in_polygon(ax, ay, ep) for ep in exclude_polys):
            continue
        if staff_polys and any(point_in_polygon(ax, ay, sp) for sp in staff_polys):
            continue
        matched_sid: Optional[str] = None
        for sid, poly in seat_id_polys:
            if point_in_polygon(ax, ay, poly):
                matched_sid = sid
                break
        if matched_sid is None:
            continue
        cur = best.get(matched_sid)
        key = (idx, area, aspect)
        if cur is None or (area, aspect) < (cur[1], cur[2]):
            best[matched_sid] = (idx, area, aspect)
    return {sid: idx for sid, (idx, _, _) in best.items()}


def _fill_poly_light(
    frame: np.ndarray, pts: np.ndarray, bgr: Tuple[int, int, int], alpha: float
) -> None:
    if alpha <= 0:
        return
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], bgr)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)


def _draw_overlay(
    frame: np.ndarray,
    cfg: VenueConfig,
    w: int,
    h: int,
    dets: Any,
    engine: Optional[SeatEngine],
    foot_points: List[Tuple[float, float]],
    video_sec: float,
    min_occ: float,
    stride: int,
    fps: float,
    preview_hint: bool = False,
    seat_anchor_mode: str = "center",
    standing_aspect_min: Optional[float] = None,
    require_standing_nearby: bool = False,
    standing_nearby_map: Optional[Dict[str, bool]] = None,
    waiting_barber_map: Optional[Dict[str, bool]] = None,
    blue_nearby_map: Optional[Dict[str, bool]] = None,
) -> None:
    out = cfg.output or {}
    title = str(out.get("overlay_title") or "").strip()

    # Persist lightweight staff pairing state across frames (preview/debug overlay only).
    # Maps seat_id -> last chosen staff bbox (x1,y1,x2,y2).
    if not hasattr(_draw_overlay, "_staff_pair_state"):
        _draw_overlay._staff_pair_state = {}  # type: ignore[attr-defined]
    staff_state: Dict[str, Tuple[float, float, float, float]] = getattr(  # type: ignore[assignment]
        _draw_overlay, "_staff_pair_state", {}
    )

    for s in cfg.seats:
        poly = norm_polygon_to_pixels(s["polygon_norm"], w, h)
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        _fill_poly_light(frame, pts, (0, 180, 0), 0.12)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
        cx = int(sum(p[0] for p in poly) / len(poly))
        cy = int(sum(p[1] for p in poly) / len(poly))
        sid = str(s.get("id", ""))
        _put_text_large(frame, sid, (cx - 40, cy + 8), 0.9, (0, 255, 100), 2)

    seat_id_polys: List[Tuple[str, List[Point]]] = [
        (str(s["id"]), norm_polygon_to_pixels(s["polygon_norm"], w, h))
        for s in cfg.seats
    ]
    seat_id_bboxes: Dict[str, Tuple[float, float, float, float]] = {
        sid: polygon_axis_aligned_bbox(poly) for sid, poly in seat_id_polys
    }
    seat_centers: Dict[str, Tuple[float, float]] = {
        sid: (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
        for sid, poly in seat_id_polys
        if poly
    }
    staff_polys_px = [
        norm_polygon_to_pixels(z, w, h) for z in cfg.staff_zones_norm
    ]
    exclude_polys_px = [
        norm_polygon_to_pixels(z, w, h) for z in cfg.exclude_zones_norm
    ]

    for z in cfg.staff_zones_norm:
        poly = norm_polygon_to_pixels(z, w, h)
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        _fill_poly_light(frame, pts, (0, 140, 255), 0.12)
        cv2.polylines(frame, [pts], True, (0, 165, 255), 3)
        cx = int(sum(p[0] for p in poly) / len(poly))
        cy = int(sum(p[1] for p in poly) / len(poly))
        _put_text_large(frame, "staff", (cx - 28, cy), 0.55, (0, 200, 255), 2)

    for z in cfg.exclude_zones_norm:
        poly = norm_polygon_to_pixels(z, w, h)
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        _fill_poly_light(frame, pts, (200, 0, 200), 0.15)
        cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
        cx = int(sum(p[0] for p in poly) / len(poly))
        cy = int(sum(p[1] for p in poly) / len(poly))
        _put_text_large(frame, "exclude", (cx - 50, cy), 0.7, (255, 180, 255), 2)

    rules_ov = cfg.rules or {}
    min_h_ov = rules_ov.get("min_box_height_frac")
    min_box_h_frac_ov = float(min_h_ov) if min_h_ov is not None else None
    max_ar_ov = rules_ov.get("max_box_aspect_height_width")
    max_box_aspect_ov = float(max_ar_ov) if max_ar_ov is not None else None
    near_pair_norm = float(rules_ov.get("near_customer_pair_dist_norm", 0.22))
    pair_px = near_pair_norm * min(float(w), float(h))
    fh_ov = float(h)
    min_area_ov = rules_ov.get("min_seat_customer_box_area_frac")
    min_box_area_frac_ov = (
        float(min_area_ov) if min_area_ov is not None else None
    )
    near_iou_ov = rules_ov.get("near_customer_pair_min_iou")
    near_customer_pair_min_iou = (
        float(near_iou_ov) if near_iou_ov is not None else None
    )
    msna_ov = rules_ov.get("min_standing_nearby_box_area_frac")
    min_standing_nearby_area_frac_ov = (
        float(msna_ov) if msna_ov is not None else None
    )
    synthetic_seat_box_ov = bool(
        rules_ov.get("synthetic_seat_box_for_pairing", True)
    )
    proxy_barber_overlap_ov = float(
        rules_ov.get("proxy_barber_overlap_min", 0.12)
    )
    pose_kpt_conf_min = float(rules_ov.get("pose_kpt_conf_min", 0.20))
    staff_overlap_min = float(rules_ov.get("staff_seat_overlap_min", proxy_barber_overlap_ov))
    staff_lock_iou = float(rules_ov.get("staff_lock_iou", 0.35))
    staff_aspect_min = float(rules_ov.get("staff_aspect_min", 0.8))
    # Staff scoring: 'ratio' = inter/barber_area, 'area' = raw intersection area (px^2).
    # For robustness (avoid tiny leg boxes winning), default to 'area'.
    staff_overlap_mode = str(rules_ov.get("staff_overlap_mode", "area")).strip().lower()
    min_seat_customer_conf_ov = float(rules_ov.get("min_seat_customer_conf", 0.0))
    sitting_boxes = _collect_sitting_customer_boxes(
        dets,
        seat_id_polys,
        seat_anchor_mode,
        standing_aspect_min,
        exclude_polys_px,
        staff_polys_px,
        fh_ov,
        w,
        h,
        min_box_h_frac_ov,
        max_box_aspect_ov,
        min_box_area_frac_ov,
        min_seat_customer_conf_ov,
        synthetic_seat_box_ov,
    )
    barber_near_labels = _near_customer_barber_labels_by_max_iou(
        dets,
        sitting_boxes,
        standing_aspect_min,
        pair_px,
        near_customer_pair_min_iou,
        proxy_barber_overlap_ov,
        fh_ov,
        min_box_h_frac_ov,
        max_box_aspect_ov,
        min_standing_nearby_area_frac_ov,
        w,
        h,
    )

    # Decide exactly one "customer" bbox per seat for overlay.
    customer_idx_by_seat = _best_customer_idx_by_seat(
        dets,
        seat_id_polys,
        seat_anchor_mode,
        standing_aspect_min,
        staff_polys_px,
        exclude_polys_px,
        fh_ov,
        min_box_h_frac_ov,
        max_box_aspect_ov,
        min_box_area_frac_ov,
        min_seat_customer_conf_ov,
        w,
        h,
    )
    customer_idx_set = set(customer_idx_by_seat.values())
    customer_box_by_seat: Dict[str, Tuple[float, float, float, float]] = {}
    for sid, cidx in customer_idx_by_seat.items():
        if 0 <= int(cidx) < len(dets):
            cx1, cy1, cx2, cy2 = dets[int(cidx)].xyxy
            customer_box_by_seat[str(sid)] = (
                float(cx1),
                float(cy1),
                float(cx2),
                float(cy2),
            )

    # Blue highlight: for each seated customer, pick the ONE detection whose bbox
    # has the largest intersection area with that customer's bbox.
    # No other heuristics/thresholds.
    blue_idx_set: set[int] = set()
    for _sid, (sx1, sy1, sx2, sy2) in customer_box_by_seat.items():
        best_inter = 0.0
        best_idx: Optional[int] = None
        best_dist = 1e18
        for idx, d in enumerate(dets):
            if idx in customer_idx_set:
                continue
            x1, y1, x2, y2 = d.xyxy
            inter = axis_aligned_intersection_area(sx1, sy1, sx2, sy2, x1, y1, x2, y2)
            dist = _bbox_edge_distance(sx1, sy1, sx2, sy2, x1, y1, x2, y2)
            if inter > best_inter or (inter == best_inter and dist < best_dist):
                best_inter = inter
                best_idx = idx
                best_dist = dist
        if best_idx is not None and (best_inter > 0.0 or best_dist <= 1e-6):
            blue_idx_set.add(int(best_idx))

    # Staff labels by overlap with customer bbox when available (more precise),
    # otherwise fall back to overlap with seat zone bbox.
    staff_labels: Dict[int, Tuple[str, float, float, float]] = {}
    if seat_id_bboxes:
        fram = float(w) * float(h)
        best_by_seat: Dict[str, Tuple[int, float, float, float]] = {}
        staff_customer_overlap_min = float(
            rules_ov.get("staff_customer_overlap_min", staff_overlap_min)
        )
        staff_customer_inter_min_frac = float(
            rules_ov.get("staff_customer_inter_min_area_frac", 0.0)
        )
        staff_seat_inter_min_frac = float(
            rules_ov.get("staff_seat_inter_min_area_frac", 0.0)
        )
        for idx, d in enumerate(dets):
            x1, y1, x2, y2 = d.xyxy
            bh = max(1e-6, float(y2 - y1))
            bw = max(1e-6, float(x2 - x1))
            aspect = bh / bw
            if min_box_h_frac_ov is not None and bh / fh_ov < min_box_h_frac_ov:
                continue
            if max_box_aspect_ov is not None and aspect > max_box_aspect_ov:
                continue
            # Staff can be wider than "standing_aspect_min" due to pose (arms / bending).
            # Use a softer threshold here; counting logic still uses standing_aspect_min.
            if aspect < staff_aspect_min:
                continue
            if min_standing_nearby_area_frac_ov is not None:
                if (bh * bw) / fram < min_standing_nearby_area_frac_ov:
                    continue
            # ignore excluded/staff zones if they exist
            ax, ay = _anchor_for_detection(d, x1, y1, x2, y2, seat_anchor_mode)
            if any(point_in_polygon(ax, ay, ep) for ep in exclude_polys_px):
                continue
            if staff_polys_px and any(point_in_polygon(ax, ay, sp) for sp in staff_polys_px):
                continue
            # If this detection is inside a seat zone, treat it as the seated customer,
            # not staff (customer label must win).
            if idx in customer_idx_set:
                continue

            # Prefer overlap with the chosen customer bbox for that seat (if available).
            seat_items = (
                list(customer_box_by_seat.items())
                if customer_box_by_seat
                else list(seat_id_bboxes.items())
            )
            for sid, box in seat_items:
                if customer_box_by_seat:
                    sx1, sy1, sx2, sy2 = box
                else:
                    sx1, sy1, sx2, sy2 = box  # type: ignore[misc]
                # Require the staff candidate to be near the seated customer/seat.
                ccx = (sx1 + sx2) / 2.0
                ccy = (sy1 + sy2) / 2.0
                dc = math.hypot(((x1 + x2) / 2.0) - ccx, ((y1 + y2) / 2.0) - ccy)
                if dc > pair_px:
                    continue
                inter = axis_aligned_intersection_area(
                    sx1, sy1, sx2, sy2, x1, y1, x2, y2
                )
                barber_a = (x2 - x1) * (y2 - y1)
                if staff_overlap_mode in ("area", "intersect", "intersection"):
                    score = float(inter)
                    min_frac = (
                        staff_customer_inter_min_frac
                        if customer_box_by_seat
                        else staff_seat_inter_min_frac
                    )
                    if min_frac > 0.0 and (score / (fram + 1e-9)) < min_frac:
                        continue
                else:
                    score = float(inter) / (barber_a + 1e-9)
                    thr = (
                        staff_customer_overlap_min
                        if customer_box_by_seat
                        else staff_overlap_min
                    )
                    if score < thr:
                        continue
                # prefer same person as last frame for this seat (lightweight lock)
                prev = staff_state.get(sid)
                if prev is not None:
                    lock = axis_aligned_iou(x1, y1, x2, y2, prev[0], prev[1], prev[2], prev[3])
                    if lock >= staff_lock_iou:
                        if staff_overlap_mode not in ("area", "intersect", "intersection"):
                            score += 0.05
                cur = best_by_seat.get(sid)
                if cur is None or score > cur[1]:
                    cx, cy = seat_centers.get(
                        sid, ((sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0)
                    )
                    best_by_seat[sid] = (idx, score, cx, cy)

        for sid, (idx, score, cx, cy) in best_by_seat.items():
            # update state
            x1, y1, x2, y2 = dets[idx].xyxy
            staff_state[sid] = (float(x1), float(y1), float(x2), float(y2))
            staff_labels[idx] = (f"staff {sid}", score, cx, cy)

    for idx, d in enumerate(dets):
        x1, y1, x2, y2 = d.xyxy
        bh = max(1e-6, float(y2 - y1))
        bw = max(1e-6, float(x2 - x1))
        aspect_ratio = bh / bw
        if preview_hint:
            # Pose landmarks (if available): lets you verify we truly have keypoints.
            kpts = getattr(d, "kpts", None)
            if kpts:
                for (kx, ky, kc) in kpts:
                    if float(kc) < pose_kpt_conf_min:
                        continue
                    ikx, iky = int(round(kx)), int(round(ky))
                    cv2.circle(frame, (ikx, iky), 2, (0, 255, 255), -1)
                _put_text_large(frame, "POSE", (int(round(x1)), max(24, int(round(y1)) - 6)), 0.45, (0, 255, 255), 2)

            # Head marker: green when it's from pose keypoint; cyan when it's bbox heuristic.
            has_pose_head = getattr(d, "head_xy", None) is not None
            hx, hy = _anchor_for_detection(d, x1, y1, x2, y2, "head")
            ihx, ihy = int(round(hx)), int(round(hy))
            head_col = (0, 255, 0) if has_pose_head else (255, 255, 0)
            cv2.circle(frame, (ihx, ihy), 5, head_col, -1)
            cv2.circle(frame, (ihx, ihy), 8, (0, 0, 0), 2)
        ax, ay = _anchor_for_detection(d, x1, y1, x2, y2, seat_anchor_mode)
        role, col = _role_label_and_color(
            ax,
            ay,
            aspect_ratio,
            seat_id_polys,
            staff_polys_px,
            exclude_polys_px,
            standing_aspect_min,
        )
        link_target: Optional[Tuple[float, float, float]] = None  # (scx, scy, score)
        if idx in barber_near_labels:
            role, score, scx, scy = barber_near_labels[idx]
            col = (255, 200, 0)
            link_target = (float(scx), float(scy), float(score))
        if idx in staff_labels:
            # staff label wins over near-customer label
            if not role.startswith("customer "):
                role, score, scx, scy = staff_labels[idx]
                col = (255, 200, 0)
                link_target = (float(scx), float(scy), float(score))

        # Force exactly one customer bbox per seat in overlay.
        if idx in customer_idx_set:
            for sid, cidx in customer_idx_by_seat.items():
                if cidx == idx:
                    role = f"customer {sid}"
                    col = (0, 255, 120)
                    break
        else:
            # If inside seat zone but not the chosen customer bbox, don't show as customer.
            in_any_seat = False
            for sid, poly in seat_id_polys:
                if point_in_polygon(ax, ay, poly):
                    in_any_seat = True
                    # if already staff label, keep it; else keep whatever role is
                    # (typically 'other' or 'barber' based on aspect/staff zones)
                    break
            if in_any_seat and role.startswith("customer "):
                role = "other"
                col = (200, 200, 200)

        # Final override: highlight the max-intersection-with-customer detection in BLUE.
        if idx in blue_idx_set and idx not in customer_idx_set:
            col = (255, 0, 0)
        ix1, iy1, ix2, iy2 = [int(round(v)) for v in (x1, y1, x2, y2)]
        thickness = 4 if (idx in barber_near_labels or idx in staff_labels or idx in blue_idx_set) else 2
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), col, thickness)
        line1 = role
        line2 = f"{d.conf:.2f}"
        ly1 = max(28, iy1 - 8)
        ly2 = max(48, iy1 - 28)
        _put_text_large(frame, line1, (ix1, ly1), 0.55, col, 2)
        _put_text_large(frame, line2, (ix1, ly2), 0.5, (255, 255, 255), 2)
        if link_target is not None:
            scx, scy, score = link_target
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cv2.line(
                frame,
                (int(round(cx)), int(round(cy))),
                (int(round(scx)), int(round(scy))),
                (255, 200, 0),
                2,
            )
            _put_text_large(
                frame,
                f"overlap={score:.3f}",
                (ix1, iy2 + 18 if (iy2 + 18) < h else max(18, iy1 - 48)),
                0.48,
                (255, 200, 0),
                2,
            )

    for fx, fy in foot_points:
        ix, iy = int(round(fx)), int(round(fy))
        cv2.circle(frame, (ix, iy), 5, (255, 0, 255), -1)
        cv2.circle(frame, (ix, iy), 8, (255, 255, 255), 2)

    if engine is not None:
        occupied_now = engine.assign_foot_points(foot_points)
        y0 = 36
        line_h = 38
        for sid, seat in engine.seats.items():
            acc = seat.accumulated_sec
            if occupied_now.get(sid, False):
                st = (
                    "COUNTED"
                    if seat.counted_this_session
                    else f"{acc:.0f}s / {min_occ:.0f}s"
                )
            else:
                if waiting_barber_map is not None and waiting_barber_map.get(sid, False):
                    st = "WAIT BARBER"
                else:
                    st = f"EMPTY {seat.empty_debounce_sec:.0f}s"
            _put_text_large(
                frame,
                f"{sid}: {st}",
                (12, y0),
                0.95,
                (255, 255, 255),
                2,
            )
            y0 += line_h
        sa = str((cfg.rules or {}).get("seat_anchor", "")).strip()
        if sa:
            _put_text_large(
                frame,
                f"anchor={sa}",
                (12, y0 + 4),
                0.55,
                (180, 220, 255),
                2,
            )
            y0 += line_h
        if (
            require_standing_nearby
            and standing_aspect_min is not None
            and standing_nearby_map is not None
        ):
            sm = standing_nearby_map
            ok = ",".join(sorted(s for s, v in sm.items() if v)) or "—"
            _put_text_large(
                frame,
                f"standing_nearby={ok}",
                (12, y0 + 4),
                0.55,
                (180, 240, 180),
                2,
            )
            y0 += line_h
        if blue_nearby_map is not None:
            sm = blue_nearby_map
            ok = ",".join(sorted(s for s, v in sm.items() if v)) or "—"
            _put_text_large(
                frame,
                f"blue_nearby={ok}",
                (12, y0 + 4),
                0.55,
                (255, 220, 180),
                2,
            )

    if title:
        tw = int(cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0])
        _put_text_large(frame, title, (w - tw - 16, 44), 1.0, (255, 255, 255), 2)

    _put_text_large(
        frame,
        f"t={video_sec:.1f}s  stride={stride}  ~{fps:.0f} fps",
        (10, h - 28),
        0.7,
        (220, 220, 220),
        2,
    )
    if preview_hint:
        _put_text_large(
            frame,
            "PREVIEW  q quit  [ ] -10s/+10s  , . -2s/+2s  b/f -5s/+5s",
            (10, h - 62),
            0.55,
            (0, 255, 255),
            2,
        )


def run_from_config_path(
    config_path: Path,
    override_video: Optional[Union[str, Path]] = None,
    max_frames: Optional[int] = None,
    preview: bool = False,
) -> Dict[str, Any]:
    cfg = load_venue_config(config_path)
    root = project_root_from_config(config_path)
    return run_venue(
        cfg,
        root,
        override_video=override_video,
        max_frames=max_frames,
        preview=preview,
    )
