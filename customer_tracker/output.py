from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover
    Workbook = None  # type: ignore


@dataclass
class OutputSettings:
    data_dir: Path
    venue_id: str
    timezone: str
    write_events_csv: bool = True
    write_daily_csv: bool = True
    write_daily_xlsx: bool = True


class EventSink:
    """Append events + maintain daily totals (CSV + optional XLSX)."""

    def __init__(self, settings: OutputSettings) -> None:
        self.settings = settings
        self.tz = ZoneInfo(settings.timezone)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._events_csv = settings.data_dir / "events.csv"
        self._daily_csv = settings.data_dir / "daily_totals.csv"
        self._daily_xlsx = settings.data_dir / "daily_totals.xlsx"
        self._ensure_headers()

    def _ensure_headers(self) -> None:
        if self.settings.write_events_csv and not self._events_csv.exists():
            with open(self._events_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "timestamp_iso",
                        "venue_id",
                        "seat_id",
                        "event",
                        "notes",
                    ]
                )
        if self.settings.write_daily_csv and not self._daily_csv.exists():
            with open(self._daily_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["date", "total_customers"])

    def record_customer_count(self, seat_id: str, when: Optional[datetime] = None) -> None:
        dt = when or datetime.now(self.tz)
        ts = dt.isoformat(timespec="seconds")
        if self.settings.write_events_csv:
            with open(self._events_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        ts,
                        self.settings.venue_id,
                        seat_id,
                        "customer_counted",
                        "",
                    ]
                )
        d = dt.date()
        self._bump_daily(d)

    def _bump_daily(self, d: date) -> None:
        key = d.isoformat()
        totals: dict[str, int] = {}
        if self.settings.write_daily_csv and self._daily_csv.exists():
            with open(self._daily_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                for row in rows:
                    if row.get("date") and row.get("total_customers") is not None:
                        try:
                            totals[row["date"]] = int(row["total_customers"])
                        except ValueError:
                            continue
        totals[key] = totals.get(key, 0) + 1

        if self.settings.write_daily_csv:
            with open(self._daily_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["date", "total_customers"])
                for dk in sorted(totals.keys()):
                    w.writerow([dk, totals[dk]])

        if self.settings.write_daily_xlsx and Workbook is not None:
            wb = Workbook()
            ws = wb.active
            assert ws is not None
            ws.title = "daily"
            ws.append(["date", "total_customers"])
            for dk in sorted(totals.keys()):
                ws.append([dk, totals[dk]])
            self._daily_xlsx.parent.mkdir(parents=True, exist_ok=True)
            wb.save(self._daily_xlsx)
