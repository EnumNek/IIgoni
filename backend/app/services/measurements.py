"""Измерения трафика: импорт из SUMO (edgeData) и CSV, синтетические данные для демонстрации.

Формат SUMO edgeData (выход `<edgeData>` в .sumocfg / additional):
    <meandata>
      <interval begin="0.00" end="900.00" id="ed">
        <edge id="123#0" occupancy="12.5" speed="9.8" traveltime="14.2" density="20.1" entered="130"/>
    occupancy в SUMO — в процентах, в БД хранится доля 0..1.
"""
from __future__ import annotations

import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from ..models import RoadSegment, TrafficMeasurement


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def import_sumo_edgedata(session: Session, path: Path, source: str = "sumo", run_label: str | None = None) -> int:
    known = set(session.scalars(select(RoadSegment.id)))
    rows = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "interval":
            continue
        begin, end = _f(el.get("begin")), _f(el.get("end"))
        span = (end - begin) if begin is not None and end is not None and end > begin else None
        for edge in el.iter("edge"):
            eid = edge.get("id")
            if eid not in known:
                continue
            occ = _f(edge.get("occupancy"))
            entered = _f(edge.get("entered"))
            rows.append({
                "edge_id": eid, "source": source, "run_label": run_label or path.stem,
                "interval_begin": begin, "interval_end": end,
                "occupancy": min(occ / 100.0, 1.0) if occ is not None else None,
                "mean_speed": _f(edge.get("speed")), "travel_time": _f(edge.get("traveltime")),
                "density": _f(edge.get("density")),
                "flow": entered / span * 3600 if entered is not None and span else None,
            })
        el.clear()
    for i in range(0, len(rows), 2000):
        session.execute(insert(TrafficMeasurement), rows[i:i + 2000])
    session.commit()
    return len(rows)


def import_csv(session: Session, path: Path, source: str = "sensor") -> int:
    """CSV с колонками: edge_id, occupancy, mean_speed, travel_time[, flow, measured_at, interval_begin, interval_end]."""
    known = set(session.scalars(select(RoadSegment.id)))
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("edge_id") not in known:
                continue
            rows.append({
                "edge_id": r["edge_id"], "source": r.get("source") or source, "run_label": r.get("run_label"),
                "interval_begin": _f(r.get("interval_begin")), "interval_end": _f(r.get("interval_end")),
                "occupancy": _f(r.get("occupancy")), "mean_speed": _f(r.get("mean_speed")),
                "travel_time": _f(r.get("travel_time")), "flow": _f(r.get("flow")),
            })
    for i in range(0, len(rows), 2000):
        session.execute(insert(TrafficMeasurement), rows[i:i + 2000])
    session.commit()
    return len(rows)


def seed_synthetic(session: Session, results: dict[str, dict], intervals: int = 4, seed: int = 1) -> int:
    """Синтетические «замеры» = прогноз базового сценария + шум ±15 %; source='synthetic'.

    Нужны только для демонстрации сущности «Измерение трафика» в интерфейсе,
    пока нет реальных данных; в отчётах их использовать нельзя.
    """
    rng = random.Random(seed)
    rows = []
    for eid, r in results.items():
        if r.get("closed"):
            continue
        for k in range(intervals):
            noise = 1 + rng.uniform(-0.15, 0.15)
            rows.append({
                "edge_id": eid, "source": "synthetic", "run_label": "demo",
                "interval_begin": k * 900.0, "interval_end": (k + 1) * 900.0,
                "occupancy": min(1.0, r["occupancy"] * noise), "mean_speed": r["mean_speed"] / noise,
                "travel_time": (r["travel_time"] or 0) * noise, "flow": None,
            })
    for i in range(0, len(rows), 2000):
        session.execute(insert(TrafficMeasurement), rows[i:i + 2000])
    session.commit()
    return len(rows)
