"""Заглушка части sim/: построение графа сети в формате контракта.

В штатной схеме граф `sim/data/network.json` формирует часть sim/ через SUMO
(netconvert). Пока его нет, бэкенд строит граф сам:

1. если есть OSM-выгрузка района (sim/data/map.osm или map.osm в корне
   репозитория) — разбирает её чистым Python (без SUMO): берёт проезжие
   дороги, режет их на участки по перекрёсткам, учитывает односторонность,
   число полос (lanes, lanes:forward/backward) и ограничение скорости
   (maxspeed). Идентификаторы сделаны в стиле netconvert: узлы = OSM id,
   рёбра = "<way_id>#<номер>" и "-<way_id>#<номер>" для обратного направления;
2. иначе — синтетическую решётку 4×4 (как `netgenerate --grid`), узлы A0..D3.

Формат результата (контракт):
    {nodes:[{id,lon,lat,type}], edges:[{id,from,to,length,lanes,speed,priority,geometry}]}
Дополнительно пишутся необязательные поля edges[].name, edges[].type и раздел meta.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Классы дорог OSM, по которым ездит автотранспорт. service (проезды во дворах,
# парковки) сознательно исключён: в районе это ~750 коротких путей, которые
# загромождают граф и почти не влияют на транзитные потоки.
DRIVABLE = {
    "motorway": (13, 2, 90), "trunk": (12, 2, 80), "primary": (11, 2, 60),
    "secondary": (10, 2, 60), "tertiary": (9, 1, 60), "unclassified": (7, 1, 60),
    "residential": (6, 1, 40), "living_street": (3, 1, 20),
    "motorway_link": (12, 1, 60), "trunk_link": (11, 1, 60), "primary_link": (10, 1, 50),
    "secondary_link": (9, 1, 50), "tertiary_link": (8, 1, 40),
}  # класс -> (priority, полос по умолчанию в одном направлении, км/ч по умолчанию)

EARTH_R = 6_371_000.0
TLS_SNAP_DISTANCE = 40.0  # м: светофор в OSM часто стоит на подходе, а не в самом перекрёстке


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def polyline_length(coords: list[list[float]]) -> float:
    return sum(haversine(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1))


def _parse_speed(raw: str | None, default_kmh: float) -> float:
    """maxspeed OSM -> м/с. Поддерживает '60', '40 mph', 'RU:urban'/'KZ:urban' (=60)."""
    if raw:
        raw = raw.strip().lower()
        if raw.endswith("urban"):
            return 60 / 3.6
        num = "".join(ch for ch in raw.split(";")[0] if ch.isdigit() or ch == ".")
        try:
            val = float(num)
            if val > 0:
                return val * (1.609 if "mph" in raw else 1.0) / 3.6
        except ValueError:
            pass
    return default_kmh / 3.6


def _parse_int(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        return max(1, int(float(raw.split(";")[0])))
    except ValueError:
        return None


def build_from_osm(osm_path: Path) -> dict:
    nodes: dict[str, tuple[float, float]] = {}
    signal_nodes: set[str] = set()
    ways: list[tuple[str, list[str], dict[str, str]]] = []
    bounds: list[float] | None = None

    for _, el in ET.iterparse(osm_path, events=("end",)):
        tag = el.tag
        if tag == "node":
            nid = el.get("id")
            nodes[nid] = (float(el.get("lon")), float(el.get("lat")))
            for t in el.iter("tag"):
                if t.get("k") == "highway" and t.get("v") == "traffic_signals":
                    signal_nodes.add(nid)
            el.clear()
        elif tag == "way":
            tags = {t.get("k"): t.get("v") for t in el.iter("tag")}
            hw = tags.get("highway")
            if hw in DRIVABLE and tags.get("area") != "yes" and tags.get("access") not in {"no", "private"}:
                ways.append((el.get("id"), [nd.get("ref") for nd in el.iter("nd")], tags))
            el.clear()
        elif tag == "bounds":
            bounds = [float(el.get("minlon")), float(el.get("minlat")), float(el.get("maxlon")), float(el.get("maxlat"))]
        elif tag == "relation":
            el.clear()

    if bounds is None:
        lons = [c[0] for c in nodes.values()]
        lats = [c[1] for c in nodes.values()]
        bounds = [min(lons), min(lats), max(lons), max(lats)]

    def inside(nid: str) -> bool:
        if nid not in nodes:
            return False
        lon, lat = nodes[nid]
        return bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]

    # 1) Обрезка по границе района: путь делится на непрерывные «внутренние» куски
    runs: list[tuple[str, list[str], dict[str, str]]] = []
    for wid, refs, tags in ways:
        cur: list[str] = []
        for ref in refs:
            if inside(ref):
                cur.append(ref)
            else:
                if len(cur) >= 2:
                    runs.append((wid, cur, tags))
                cur = []
        if len(cur) >= 2:
            runs.append((wid, cur, tags))

    # 2) Узлы графа: концы кусков и точки, где сходятся несколько путей
    usage: dict[str, int] = defaultdict(int)
    for _, refs, _ in runs:
        for ref in refs:
            usage[ref] += 1
        usage[refs[0]] += 1
        usage[refs[-1]] += 1
    is_vertex = {nid for nid, cnt in usage.items() if cnt >= 2}

    edges: list[dict] = []
    seg_counter: dict[str, int] = defaultdict(int)
    for wid, refs, tags in runs:
        hw = tags["highway"]
        priority, default_lanes, default_kmh = DRIVABLE[hw]
        oneway_tag = (tags.get("oneway") or "").lower()
        roundabout = tags.get("junction") in {"roundabout", "circular"}
        forward_only = oneway_tag in {"yes", "true", "1"} or roundabout or hw == "motorway"
        backward_only = oneway_tag == "-1"
        total_lanes = _parse_int(tags.get("lanes"))
        lanes_fw = _parse_int(tags.get("lanes:forward"))
        lanes_bw = _parse_int(tags.get("lanes:backward"))
        if forward_only or backward_only:
            lanes_fw = lanes_bw = total_lanes or default_lanes
        else:
            if total_lanes and not lanes_fw:
                lanes_fw = max(1, math.ceil(total_lanes / 2))
            if total_lanes and not lanes_bw:
                lanes_bw = max(1, total_lanes - (lanes_fw or 1))
            lanes_fw = lanes_fw or default_lanes
            lanes_bw = lanes_bw or default_lanes
        speed = _parse_speed(tags.get("maxspeed"), default_kmh)
        name = tags.get("name")

        # 3) Разрезание куска на участки между узлами графа
        segments: list[list[str]] = []
        cur = [refs[0]]
        for ref in refs[1:]:
            cur.append(ref)
            if ref in is_vertex:
                segments.append(cur)
                cur = [ref]
        if len(cur) >= 2:
            segments.append(cur)

        for seg in segments:
            if seg[0] == seg[-1]:  # петля: делим пополам, чтобы не было ребра узел->тот же узел
                if len(seg) < 4:
                    continue
                mid = len(seg) // 2
                is_vertex.add(seg[mid])
                parts = [seg[: mid + 1], seg[mid:]]
            else:
                parts = [seg]
            for part in parts:
                coords = [list(nodes[r]) for r in part]
                length = round(polyline_length(coords), 2)
                if length < 0.5:
                    continue
                k = seg_counter[wid]
                seg_counter[wid] += 1
                base = {"length": length, "speed": round(speed, 2), "priority": priority,
                        "name": name, "type": f"highway.{hw}"}
                if not backward_only:
                    edges.append({"id": f"{wid}#{k}", "from": part[0], "to": part[-1], "lanes": lanes_fw,
                                  "geometry": coords, **base})
                if not forward_only:
                    edges.append({"id": f"-{wid}#{k}", "from": part[-1], "to": part[0], "lanes": lanes_bw,
                                  "geometry": coords[::-1], **base})

    edges = _largest_component(edges)
    used = {e["from"] for e in edges} | {e["to"] for e in edges}

    neighbours: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        neighbours[e["from"]].add(e["to"])
        neighbours[e["to"]].add(e["from"])

    node_types = {nid: ("dead_end" if len(neighbours[nid]) <= 1 else "priority") for nid in used}
    # 4) Светофоры: узел-перекрёсток со светофором или ближайший перекрёсток к сигналу на подходе
    junctions = [nid for nid in used if len(neighbours[nid]) >= 3]
    for sid in signal_nodes:
        if sid not in nodes:
            continue
        if sid in used and len(neighbours[sid]) >= 3:
            node_types[sid] = "traffic_light"
            continue
        slon, slat = nodes[sid]
        best, best_d = None, TLS_SNAP_DISTANCE
        for jid in junctions:
            d = haversine(slon, slat, *nodes[jid])
            if d < best_d:
                best, best_d = jid, d
        if best:
            node_types[best] = "traffic_light"

    out_nodes = [{"id": nid, "lon": nodes[nid][0], "lat": nodes[nid][1], "type": node_types[nid]}
                 for nid in sorted(used)]
    return {
        "meta": {
            "source": "stub-osm",
            "generator": "backend/app/stubs/network_builder.py",
            "osm_file": osm_path.name,
            "bounds": bounds,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "nodes": out_nodes,
        "edges": edges,
    }


def _largest_component(edges: list[dict]) -> list[dict]:
    """Оставляет крупнейшую слабосвязную компоненту (обрывки у границы района отбрасываются)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        ra, rb = find(e["from"]), find(e["to"])
        if ra != rb:
            parent[ra] = rb
    sizes: dict[str, int] = defaultdict(int)
    for e in edges:
        sizes[find(e["from"])] += 1
    if not sizes:
        return edges
    root = max(sizes, key=sizes.get)
    return [e for e in edges if find(e["from"]) == root]


def build_grid(n: int = 4, spacing: float = 200.0, center: tuple[float, float] = (69.1476, 54.8655)) -> dict:
    """Синтетическая решётка n×n (аналог `netgenerate --grid --grid.number=4`).

    Узлы называются как в netgenerate: столбцы A, B, C, ... и строки 0..n-1;
    рёбра — "A0A1" (из A0 в A1). Средние линии решётки сделаны магистралями
    (2 полосы, 60 км/ч), внутренние перекрёстки — со светофорами.
    """
    lon0, lat0 = center
    dlat = spacing / 111_320.0
    dlon = spacing / (111_320.0 * math.cos(math.radians(lat0)))
    off = (n - 1) / 2
    letters = [chr(ord("A") + i) for i in range(n)]
    nodes = []
    pos: dict[str, tuple[float, float]] = {}
    for i, col in enumerate(letters):
        for j in range(n):
            nid = f"{col}{j}"
            lon, lat = lon0 + (i - off) * dlon, lat0 + (j - off) * dlat
            pos[nid] = (lon, lat)
            inner = 0 < i < n - 1 and 0 < j < n - 1
            corner = i in (0, n - 1) and j in (0, n - 1)
            nodes.append({"id": nid, "lon": lon, "lat": lat,
                          "type": "traffic_light" if inner else ("dead_end" if corner else "priority")})
    arterial_cols = {letters[n // 2]} if n > 2 else set()
    arterial_rows = {n // 2} if n > 2 else set()
    edges = []
    for i, col in enumerate(letters):
        for j in range(n):
            a = f"{col}{j}"
            for b, arterial in (
                (f"{letters[i + 1]}{j}" if i + 1 < n else None, j in arterial_rows),
                (f"{col}{j + 1}" if j + 1 < n else None, col in arterial_cols),
            ):
                if not b:
                    continue
                for u, v in ((a, b), (b, a)):
                    coords = [list(pos[u]), list(pos[v])]
                    edges.append({
                        "id": f"{u}{v}", "from": u, "to": v, "length": round(polyline_length(coords), 2),
                        "lanes": 2 if arterial else 1, "speed": round((60 if arterial else 50) / 3.6, 2),
                        "priority": 10 if arterial else 6, "geometry": coords,
                        "name": None, "type": "highway.secondary" if arterial else "highway.residential",
                    })
    lons = [p[0] for p in pos.values()]
    lats = [p[1] for p in pos.values()]
    return {
        "meta": {
            "source": "stub-grid",
            "generator": "backend/app/stubs/network_builder.py",
            "bounds": [min(lons), min(lats), max(lons), max(lats)],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "nodes": nodes,
        "edges": edges,
    }
