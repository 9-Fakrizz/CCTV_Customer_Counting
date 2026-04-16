from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray casting; vertices in order along the boundary."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            denom = (yj - yi) + 1e-12
            xinters = (xj - xi) * (y - yi) / denom + xi
            if x < xinters:
                inside = not inside
        j = i
    return inside


def distance_point_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Shortest distance from P to segment AB."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def polygon_axis_aligned_bbox(
    polygon: Sequence[Point],
) -> Tuple[float, float, float, float]:
    """Bounding box แกนเดียวกับ polygon (pixel)."""
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def axis_aligned_intersection_area(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def axis_aligned_iou(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
) -> float:
    """IoU ของสองกล่อง axis-aligned (สำหรับกล่องคน YOLO)."""
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    a2 = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a1 + a2 - inter + 1e-9
    return inter / union


def distance_point_to_polygon(px: float, py: float, polygon: Sequence[Point]) -> float:
    """0 if inside polygon; else shortest distance to boundary."""
    if point_in_polygon(px, py, polygon):
        return 0.0
    n = len(polygon)
    if n < 2:
        return float("inf")
    dmin = float("inf")
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        d = distance_point_to_segment(px, py, ax, ay, bx, by)
        if d < dmin:
            dmin = d
    return dmin


def norm_polygon_to_pixels(
    polygon_norm: Iterable[Sequence[float]], width: int, height: int
) -> List[Point]:
    out: List[Point] = []
    for xy in polygon_norm:
        out.append((float(xy[0]) * width, float(xy[1]) * height))
    return out


def foot_point_from_xyxy(x1: float, y1: float, x2: float, y2: float) -> Point:
    """Bottom-center of bbox (typical contact point for seated person)."""
    return ((x1 + x2) / 2.0, y2)


def seat_anchor_from_xyxy(
    x1: float, y1: float, x2: float, y2: float, mode: str = "center"
) -> Point:
    """
    Point used to test inside seat / staff / exclude polygons.

    - center: กลางกล่องคน (เทียบโซนแบบ “ทั้งคน” ไม่ยึดที่เท้า) — ค่าเริ่ม
    - head: จุดประมาณ “หัว” จาก bbox (กึ่งกลางด้านบนเลื่อนลงเล็กน้อย)
    - foot: กลางขอบล่างของกล่อง (เท้า)
    - lower_fifth / lower_quarter: จุดกลางแถบล่างของกล่อง
    - lower_third_center: กลาง 1/3 ล่างของกล่อง
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h = max(1e-6, y2 - y1)
    m = (mode or "center").strip().lower()
    if m == "center":
        return (cx, cy)
    if m == "head":
        # heuristic: head is near top-center, but avoid exact y1 (bbox jitter)
        return (cx, y1 + 0.12 * h)
    if m == "foot":
        return (cx, y2)
    if m == "lower_fifth":
        return (cx, y2 - 0.1 * h)
    if m == "lower_quarter":
        return (cx, y2 - 0.125 * h)
    if m == "lower_third_center":
        return (cx, y1 + (5.0 / 6.0) * h)
    return (cx, cy)
