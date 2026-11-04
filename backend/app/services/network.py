"""Загрузка графа сети и его синхронизация с БД.

Источник графа (по убыванию приоритета):
  1. sim/data/network.json — штатный выход части sim/ (контракт);
  2. backend/data/network_stub.json — ранее построенная заглушка;
  3. map.osm (sim/data/map.osm или в корне репозитория) -> заглушка из OSM;
  4. синтетическая решётка 4×4.
Как только в sim/data появится network.json, при следующем старте бэкенд
загрузит его (сравнение по SHA-256) и обновит таблицы узлов и участков.

Граф хранится в двух видах:
  * в памяти (dict в формате контракта) — его получает ml/predict.py;
  * в PostGIS (network_nodes, road_segments) — для API, пространственных
    запросов и ссылочной целостности результатов прогноза.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.orm import Session

from ..config import Settings, settings
from ..geo import linestring_wkt, parse_wkt_coords, point_wkt
from ..models import NetworkNode, NetworkVersion, RoadSegment
from ..stubs.network_builder import build_from_osm, build_grid, polyline_length

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ нормализация
def _normalize_geometry(raw) -> list[list[float]]:
    """Приводит геометрию к списку [lon, lat]. Понимает несколько распространённых форм."""
    if raw is None:
        return []
    if isinstance(raw, dict):  # GeoJSON LineString
        raw = raw.get("coordinates") or []
    if isinstance(raw, str):  # формат shape SUMO: "x1,y1 x2,y2"
        pts = []
        for pair in raw.split():
            parts = pair.split(",")
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
        raw = pts
    out = []
    for p in raw:
        if isinstance(p, dict):
            lon, lat = p.get("lon", p.get("x")), p.get("lat", p.get("y"))
        else:
            lon, lat = p[0], p[1]
        out.append([float(lon), float(lat)])
    if out and not all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in out):
        return []  # координаты в метрах (проекция SUMO) — без проекции не перевести, берём узлы
    return out


def normalize_network(raw: dict) -> dict:
    """Проверяет граф по контракту и приводит типы. Лишние поля рёбер сохраняются."""
    nodes = []
    for n in raw.get("nodes", []):
        nodes.append({**n, "id": str(n["id"]), "lon": float(n["lon"]), "lat": float(n["lat"]),
                      "type": str(n.get("type") or "priority")})
    pos = {n["id"]: [n["lon"], n["lat"]] for n in nodes}
    edges, skipped = [], 0
    for e in raw.get("edges", []):
        f, t = str(e["from"]), str(e["to"])
        if f not in pos or t not in pos:
            skipped += 1
            continue
        geom = _normalize_geometry(e.get("geometry")) or [pos[f], pos[t]]
        if len(geom) == 1:
            geom = [geom[0], pos[t]]
        length = e.get("length")
        edges.append({
            **e, "id": str(e["id"]), "from": f, "to": t,
            "length": float(length) if length is not None else round(polyline_length(geom), 2),
            "lanes": max(1, int(e.get("lanes") or 1)),
            "speed": float(e.get("speed") or 13.89),
            "priority": int(e.get("priority") or 0),
            "geometry": geom,
        })
    if skipped:
        log.warning("Пропущено %d рёбер со ссылками на несуществующие узлы", skipped)
    used = {e["from"] for e in edges} | {e["to"] for e in edges}
    nodes = [n for n in nodes if n["id"] in used]
    return {"meta": dict(raw.get("meta") or {}), "nodes": nodes, "edges": edges}


def bounds_of(net: dict) -> list[float]:
    lons = [n["lon"] for n in net["nodes"]]
    lats = [n["lat"] for n in net["nodes"]]
    return [min(lons), min(lats), max(lons), max(lats)] if lons else [0, 0, 0, 0]


# ------------------------------------------------------------------ хранилище
@dataclass
class TripSample:
    """Фиксированный набор пар «откуда–куда» для метрики «среднее время в пути»."""

    origins: np.ndarray
    destinations: list[np.ndarray]


@dataclass
class NetworkStore:
    cfg: Settings = field(default_factory=lambda: settings)
    network: dict | None = None
    source: str | None = None
    path: Path | None = None
    hash: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _edges: dict[str, dict] = field(default_factory=dict)
    _nodes: dict[str, dict] = field(default_factory=dict)
    _trips: TripSample | None = None

    # -------------------------------------------------- загрузка
    def load(self, force: bool = False) -> dict:
        with self._lock:
            if self.network is not None and not force:
                return self.network
            raw, source, path = self._resolve()
            text = json.dumps(raw, sort_keys=True, ensure_ascii=False)
            net = normalize_network(raw)
            net["meta"].setdefault("source", source)
            net["meta"]["bounds"] = net["meta"].get("bounds") or bounds_of(net)
            self.network, self.source, self.path = net, source, path
            self.hash = hashlib.sha256(text.encode()).hexdigest()
            self._edges = {e["id"]: e for e in net["edges"]}
            self._nodes = {n["id"]: n for n in net["nodes"]}
            self._trips = None
            log.info("Сеть загружена: %s (%s), узлов %d, участков %d",
                     path, source, len(net["nodes"]), len(net["edges"]))
            return net

    def _resolve(self) -> tuple[dict, str, Path]:
        cfg = self.cfg
        if cfg.network_path.exists():
            return json.loads(cfg.network_path.read_text(encoding="utf-8")), "sim", cfg.network_path
        osm = next((p for p in cfg.osm_candidates if p.exists()), None)
        stub = cfg.stub_network_path
        if stub.exists():
            raw = json.loads(stub.read_text(encoding="utf-8"))
            src = (raw.get("meta") or {}).get("source", "stub")
            # Если появился OSM, а заглушка была решёткой, — перестраиваем
            if not (src == "stub-grid" and osm is not None):
                return raw, src, stub
        if osm is not None:
            log.warning("sim/data/network.json не найден — строю граф-заглушку из %s", osm)
            raw = build_from_osm(osm)
        else:
            log.warning("Нет ни network.json, ни map.osm — строю синтетическую решётку 4×4")
            raw = build_grid()
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return raw, raw["meta"]["source"], stub

    # -------------------------------------------------- доступ
    @property
    def edges(self) -> dict[str, dict]:
        self.load()
        return self._edges

    @property
    def nodes(self) -> dict[str, dict]:
        self.load()
        return self._nodes

    # -------------------------------------------------- синхронизация с БД
    def sync_to_db(self, session: Session) -> NetworkVersion:
        net = self.load()
        latest = session.scalars(select(NetworkVersion).order_by(NetworkVersion.id.desc()).limit(1)).first()
        db_edges = session.scalar(select(func.count()).select_from(RoadSegment)) or 0
        if latest and latest.hash == self.hash and db_edges == len(net["edges"]):
            return latest

        node_rows = [{"id": n["id"], "type": n["type"], "lon": n["lon"], "lat": n["lat"],
                      "geom": point_wkt(n["lon"], n["lat"])} for n in net["nodes"]]
        edge_rows = [{"id": e["id"], "from_node_id": e["from"], "to_node_id": e["to"], "length": e["length"],
                      "lanes": e["lanes"], "speed": e["speed"], "priority": e["priority"],
                      "name": e.get("name"), "road_type": e.get("road_type") or e.get("type"),
                      "geom": linestring_wkt(e["geometry"])} for e in net["edges"]]
        _upsert(session, NetworkNode, node_rows)
        _upsert(session, RoadSegment, edge_rows)
        keep_edges = {r["id"] for r in edge_rows}
        keep_nodes = {r["id"] for r in node_rows}
        stale_edges = [i for i in session.scalars(select(RoadSegment.id)) if i not in keep_edges]
        for chunk in _chunks(stale_edges, 500):
            session.execute(delete(RoadSegment).where(RoadSegment.id.in_(chunk)))
        stale_nodes = [i for i in session.scalars(select(NetworkNode.id)) if i not in keep_nodes]
        for chunk in _chunks(stale_nodes, 500):
            session.execute(delete(NetworkNode).where(NetworkNode.id.in_(chunk)))
        version = NetworkVersion(hash=self.hash, source=self.source, path=str(self.path),
                                 nodes_count=len(node_rows), edges_count=len(edge_rows))
        session.add(version)
        session.commit()
        log.info("Граф записан в БД: %d узлов, %d участков (удалено устаревших: %d)",
                 len(node_rows), len(edge_rows), len(stale_edges))
        return version

    # -------------------------------------------------- выборка для API
    def query_db(self, session: Session, bbox: list[float] | None = None) -> dict:
        """Граф из БД; при bbox — только участки, пересекающие прямоугольник (ST_Intersects в PostGIS)."""
        stmt = select(RoadSegment.id, RoadSegment.from_node_id, RoadSegment.to_node_id, RoadSegment.length,
                      RoadSegment.lanes, RoadSegment.speed, RoadSegment.priority, RoadSegment.name,
                      RoadSegment.road_type, RoadSegment.geom)
        is_pg = session.get_bind().dialect.name == "postgresql"
        if bbox and is_pg:
            stmt = stmt.where(func.ST_Intersects(RoadSegment.geom, func.ST_MakeEnvelope(*bbox, 4326)))
        edges = []
        for row in session.execute(stmt):
            geom = parse_wkt_coords(row.geom)
            if bbox and not is_pg and not _bbox_hit(geom, bbox):
                continue
            edges.append({"id": row.id, "from": row.from_node_id, "to": row.to_node_id, "length": row.length,
                          "lanes": row.lanes, "speed": row.speed, "priority": row.priority,
                          "geometry": geom, "name": row.name, "type": row.road_type})
        used = {e["from"] for e in edges} | {e["to"] for e in edges}
        nodes = [{"id": r.id, "lon": r.lon, "lat": r.lat, "type": r.type}
                 for r in session.execute(select(NetworkNode.id, NetworkNode.lon, NetworkNode.lat, NetworkNode.type))
                 if r.id in used]
        net = self.load()
        meta = {**net["meta"], "source": self.source, "hash": self.hash,
                "nodes_count": len(nodes), "edges_count": len(edges)}
        return {"meta": meta, "nodes": nodes, "edges": edges}

    # -------------------------------------------------- выборка пар для метрики «время в пути»
    def trip_sample(self) -> TripSample:
        if self._trips is not None:
            return self._trips
        net = self.load()
        idx = {n["id"]: i for i, n in enumerate(net["nodes"])}
        n = len(idx)
        deg = np.zeros(n, dtype=int)
        for e in net["edges"]:
            deg[idx[e["from"]]] += 1
            deg[idx[e["to"]]] += 1
        # Кандидаты: тупики на границе (въезды в район) и самые «нагруженные» перекрёстки
        order = sorted(range(n), key=lambda i: (-deg[i], hashlib.md5(net["nodes"][i]["id"].encode()).hexdigest()))
        dead_ends = [i for i in range(n) if net["nodes"][i]["type"] == "dead_end"]
        cand = sorted(set(dead_ends[:16] + order[:16]))
        lon = np.array([net["nodes"][i]["lon"] for i in cand])
        lat = np.array([net["nodes"][i]["lat"] for i in cand])
        x = lon * 111_320 * math.cos(math.radians(float(lat.mean()) if len(lat) else 0))
        y = lat * 110_540
        far = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) >= 500
        free = self.edge_costs({e["id"]: e["length"] / max(e["speed"], 0.5) for e in net["edges"]})
        dist = dijkstra(free, directed=True, indices=cand) if cand else np.zeros((0, n))
        dests = []
        for k in range(len(cand)):
            ok = [cand[j] for j in range(len(cand)) if far[k, j] and np.isfinite(dist[k, cand[j]])]
            dests.append(np.array(ok, dtype=int))
        self._trips = TripSample(origins=np.array(cand, dtype=int), destinations=dests)
        return self._trips

    def edge_costs(self, costs: dict[str, float]) -> csr_matrix:
        """Разреженная матрица весов графа (для параллельных рёбер — минимальный вес)."""
        net = self.load()
        idx = {n["id"]: i for i, n in enumerate(net["nodes"])}
        best: dict[tuple[int, int], float] = {}
        for e in net["edges"]:
            c = costs.get(e["id"])
            if c is None or not math.isfinite(c):
                continue
            key = (idx[e["from"]], idx[e["to"]])
            if key not in best or c < best[key]:
                best[key] = c
        n = len(idx)
        if not best:
            return csr_matrix((n, n))
        rows, cols = zip(*best.keys())
        return csr_matrix((np.array(list(best.values())) + 1e-6, (rows, cols)), shape=(n, n))


def _upsert(session: Session, model, rows: list[dict]) -> None:
    existing = set(session.scalars(select(model.id)))
    new = [r for r in rows if r["id"] not in existing]
    old = [r for r in rows if r["id"] in existing]
    for chunk in _chunks(new, 1000):
        session.execute(insert(model), chunk)
    for chunk in _chunks(old, 1000):
        session.execute(update(model), chunk)
    session.flush()


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _bbox_hit(geom: list[list[float]], bbox: list[float]) -> bool:
    xs = [p[0] for p in geom]
    ys = [p[1] for p in geom]
    return not (max(xs) < bbox[0] or min(xs) > bbox[2] or max(ys) < bbox[1] or min(ys) > bbox[3])


store = NetworkStore()
