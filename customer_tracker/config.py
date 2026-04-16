from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VenueConfig:
    venue_id: str
    display_name: str
    timezone: str
    source_type: str
    source_path: str
    inference: Dict[str, Any]
    rules: Dict[str, Any]
    seats: List[Dict[str, Any]]
    staff_zones_norm: List[List[List[float]]]
    exclude_zones_norm: List[List[List[float]]]
    output: Dict[str, Any]
    config_path: Path

    def resolved_source_path(self, project_root: Path) -> Path:
        p = Path(self.source_path)
        if p.is_absolute():
            return p
        return (project_root / p).resolve()

    def resolved_data_dir(self, project_root: Path) -> Path:
        d = self.output.get("data_dir", "data")
        p = Path(str(d))
        if p.is_absolute():
            return p
        return (project_root / p).resolve()

    def resolved_debug_video_path(self, project_root: Path) -> Optional[Path]:
        dv = self.output.get("debug_video") or {}
        if not dv.get("enabled"):
            return None
        raw = dv.get("path")
        if not raw:
            return None
        p = Path(str(raw))
        if p.is_absolute():
            return p
        return (project_root / p).resolve()


def load_venue_config(path: Path) -> VenueConfig:
    path = path.resolve()
    with open(path, encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    src = raw.get("source") or {}
    out = raw.get("output") or {}
    return VenueConfig(
        venue_id=str(raw["venue_id"]),
        display_name=str(raw.get("display_name", raw["venue_id"])),
        timezone=str(raw.get("timezone", "Asia/Bangkok")),
        source_type=str(src.get("type", "file")),
        source_path=str(src.get("path", "")),
        inference=dict(raw.get("inference") or {}),
        rules=dict(raw.get("rules") or {}),
        seats=list(raw.get("seats") or []),
        staff_zones_norm=list(raw.get("staff_zones_norm") or []),
        exclude_zones_norm=list(raw.get("exclude_zones_norm") or []),
        output=dict(out),
        config_path=path,
    )


def project_root_from_config(config_path: Path) -> Path:
    """Resolve repo root from a config path like `venues/<id>/config.json`."""
    p = config_path.resolve()
    if p.parent.parent.name == "venues":
        return p.parent.parent.parent
    return p.parent
