from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .geometry import Point, point_in_polygon


@dataclass
class SeatRuntime:
    seat_id: str
    polygon: List[Point]
    accumulated_sec: float = 0.0
    absent_streak_sec: float = 0.0
    empty_debounce_sec: float = 0.0
    counted_this_session: bool = False

    def reset_session(self) -> None:
        self.accumulated_sec = 0.0
        self.absent_streak_sec = 0.0
        self.empty_debounce_sec = 0.0
        self.counted_this_session = False


@dataclass
class SeatEngine:
    min_occupancy_sec: float
    empty_debounce_sec: float
    grace_absent_sec: float
    seats: Dict[str, SeatRuntime] = field(default_factory=dict)

    @classmethod
    def from_polygons(
        cls,
        seat_polygons: Dict[str, List[Point]],
        min_occupancy_sec: float,
        empty_debounce_sec: float,
        grace_absent_sec: float,
    ) -> "SeatEngine":
        eng = cls(
            min_occupancy_sec=min_occupancy_sec,
            empty_debounce_sec=empty_debounce_sec,
            grace_absent_sec=grace_absent_sec,
        )
        for sid, poly in seat_polygons.items():
            eng.seats[sid] = SeatRuntime(seat_id=sid, polygon=poly)
        return eng

    def assign_foot_points(
        self, foot_points: List[Point]
    ) -> Dict[str, bool]:
        """Return which seats have at least one person's foot inside."""
        occupied: Dict[str, bool] = {s: False for s in self.seats}
        for fx, fy in foot_points:
            for sid, seat in self.seats.items():
                if point_in_polygon(fx, fy, seat.polygon):
                    occupied[sid] = True
                    break
        return occupied

    def step(
        self, dt: float, occupied_by_seat: Dict[str, bool]
    ) -> List[str]:
        """
        Advance timers; return seat_ids that produced a new customer count this step.
        """
        counted: List[str] = []
        for sid, seat in self.seats.items():
            present = occupied_by_seat.get(sid, False)
            if present:
                # If the seat was truly empty (absent beyond grace), start a new session
                # on re-entry so we never continue timing from an old occupant.
                if seat.absent_streak_sec > self.grace_absent_sec:
                    seat.reset_session()
                seat.absent_streak_sec = 0.0
                seat.empty_debounce_sec = 0.0
                seat.accumulated_sec += dt
                if (
                    not seat.counted_this_session
                    and seat.accumulated_sec >= self.min_occupancy_sec
                ):
                    seat.counted_this_session = True
                    counted.append(sid)
            else:
                seat.absent_streak_sec += dt
                if seat.absent_streak_sec <= self.grace_absent_sec:
                    continue
                seat.empty_debounce_sec += dt
                if seat.empty_debounce_sec >= self.empty_debounce_sec:
                    seat.reset_session()
        return counted
