"""Шаг 1. Построение дорожной сети района и экспорт графа в network.json.

    python -m sim.build_network            # map.osm, если есть, иначе решётка 4×4
    python -m sim.build_network --grid     # принудительно синтетическая решётка
    python -m sim.build_network --osm path/to/map.osm

Результат:
    sim/data/base.net.xml  — сеть SUMO (её же используют agents/ для RL);
    sim/data/network.json  — граф по контракту проекта.

Формат network.json (контракт + необязательные поля, помечены «доп.»):
    {
      "meta":  {...}                                    # доп.: источник, bbox, версия SUMO
      "nodes": [{id, lon, lat, type,
                 tl_id, tl_type}],                      # доп.: светофор узла и его тип
      "edges": [{id, from, to, length, lanes, speed, priority, geometry,
                 name, road_type, closed}]              # доп.: название улицы, класс, закрыто ли
    }
    length — м, speed — м/с (разрешённая скорость), geometry — [[lon, lat], ...].
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from sim import config
from sim.sumo_tools import binary, run, sumo_version

# Параметры netconvert для OSM. Каждая опция обоснована в README.
OSM_NETCONVERT_OPTIONS = [
    "--geometry.remove",          # склеить рёбра, разделённые «геометрическими» точками OSM
    "--roundabouts.guess",        # распознать кольцевые развязки
    "--ramps.guess",              # распознать съезды/въезды
    "--junctions.join",           # объединить сложные перекрёстки (двойные проезжие части) в один узел
    "--tls.guess-signals",        # светофоры OSM, стоящие вокруг перекрёстка, → один светофор узла
    "--tls.discard-simple",       # убрать «светофоры» на простых пешеходных переходах
    "--tls.join",                 # объединить близкие светофоры одного перекрёстка
    "--tls.default-type", "static",  # базовые светофоры — с фиксированным циклом
    "--keep-edges.by-vclass", "passenger",  # только дороги для легковых
    "--keep-edges.components", "1",         # только крупнейшая связная компонента
    "--remove-edges.isolated",
    "--output.street-names",
    "--output.original-names",    # id исходных OSM way — для связи с картой
    "--no-warnings",
]


def find_osm() -> Path | None:
    for path in config.OSM_CANDIDATES:
        if path.exists():
            return path
    return None


def read_osm_bounds(osm_path: Path) -> tuple[float, float, float, float] | None:
    """Читает <bounds> из заголовка OSM-файла: (minlon, minlat, maxlon, maxlat)."""
    with open(osm_path, encoding="utf-8") as fh:
        head = fh.read(4096)
    m = re.search(r"<bounds\s+([^>]*)/?>", head)
    if not m:
        return None
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    try:
        return (float(attrs["minlon"]), float(attrs["minlat"]),
                float(attrs["maxlon"]), float(attrs["maxlat"]))
    except KeyError:
        return None


def build_from_osm(osm_path: Path, out_net: Path, crop: bool = True) -> dict:
    """map.osm → netconvert → .net.xml. Возвращает метаданные источника."""
    typemaps = config.TYPEMAP_URBAN
    cmd = [binary("netconvert"), "--osm-files", osm_path, "--type-files", typemaps,
           "-o", out_net, *OSM_NETCONVERT_OPTIONS]
    bounds = read_osm_bounds(osm_path)
    if crop and bounds:
        # OSM-выгрузка содержит улицы целиком, даже если они уходят далеко за
        # выбранную область. Оставляем только рёбра, пересекающие область.
        cmd += ["--keep-edges.in-geo-boundary", ",".join(f"{v:.7f}" for v in bounds)]
    run(cmd)
    return {"source": "osm", "osm_file": _rel(osm_path), "osm_bounds": bounds, "cropped": bool(crop and bounds)}


def build_grid(out_net: Path) -> dict:
    """Запасной вариант: синтетическая решётка 4×4 через netgenerate."""
    run([
        binary("netgenerate"), "--grid",
        "--grid.number", str(config.GRID_NUMBER),
        "--grid.length", str(config.GRID_LENGTH),
        "--grid.attach-length", str(config.GRID_ATTACH_LENGTH),
        "--default.lanenumber", str(config.GRID_LANES),
        "--default.speed", str(config.GRID_SPEED),
        "--default-junction-type", "traffic_light",
        "--tls.default-type", "static",
        "--no-turnarounds.tls",
        "-o", out_net,
    ])
    return {"source": "netgenerate", "grid": f"{config.GRID_NUMBER}x{config.GRID_NUMBER}",
            "grid_length_m": config.GRID_LENGTH,
            "anchor_lonlat": [config.GRID_ANCHOR_LON, config.GRID_ANCHOR_LAT]}


def normalize_net(net_path: Path) -> None:
    """Однократная перевыгрузка сети через netconvert (-s net -o net).

    Сценарии строятся командой `netconvert -s base.net.xml -e патч ...`, т.е.
    сеть каждый раз перезаписывается netconvert'ом. Если базовая сеть сама
    уже прошла такую перевыгрузку, она является «неподвижной точкой»
    (проверено: повторная перевыгрузка не меняет файл), и все отличия
    сценария от базы вызваны только патчем, а не служебными пересчётами.
    """
    tmp = net_path.with_suffix(".tmp.xml")
    run([binary("netconvert"), "-s", net_path, "-o", tmp, "--no-warnings"])
    tmp.replace(net_path)


# --------------------------------------------------------------------------
# Экспорт в network.json
# --------------------------------------------------------------------------

def _make_projector(net):
    """Функция (x, y) → (lon, lat).

    Для сети из OSM используется её собственная проекция UTM (через pyproj).
    У синтетической решётки проекции нет — привязываем её к условной точке
    локальным плоским приближением (точности хватает для отображения на карте).
    """
    if net.hasGeoProj():
        proj = net.getGeoProj()
        ox, oy = net.getLocationOffset()

        def to_lonlat(x, y):
            lon, lat = proj(x - ox, y - oy, inverse=True)
            return lon, lat
        return to_lonlat

    lon0, lat0 = config.GRID_ANCHOR_LON, config.GRID_ANCHOR_LAT
    xmin, ymin, xmax, ymax = net.getBoundary()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    def to_lonlat(x, y):
        return lon0 + (x - cx) / m_per_deg_lon, lat0 + (y - cy) / m_per_deg_lat
    return to_lonlat


def _tl_types(net_path: Path) -> dict[str, str]:
    """id светофора → тип программы (static/actuated/...) из .net.xml."""
    import xml.etree.ElementTree as ET
    result: dict[str, str] = {}
    for _, el in ET.iterparse(net_path, events=("end",)):
        if el.tag == "tlLogic":
            result.setdefault(el.get("id"), el.get("type", "static"))
        if el.tag in ("edge", "tlLogic", "junction", "connection"):
            el.clear()
    return result


def export_network_json(net_path: Path, out_json: Path, source_meta: dict) -> dict:
    import sumolib

    net = sumolib.net.readNet(str(net_path), withInternal=False, withPrograms=False)
    to_lonlat = _make_projector(net)
    tl_types = _tl_types(net_path)

    # Какой светофор управляет узлом: берём из связей светофоров.
    node_tl: dict[str, str] = {}
    for tls in net.getTrafficLights():
        for conn in tls.getConnections():
            node_tl.setdefault(conn[0].getEdge().getToNode().getID(), tls.getID())

    nodes = []
    for node in net.getNodes():
        lon, lat = to_lonlat(*node.getCoord()[:2])
        ntype = node.getType()
        tl_id = node_tl.get(node.getID())
        nodes.append({
            "id": node.getID(),
            "lon": round(lon, 7),
            "lat": round(lat, 7),
            "type": ntype,
            "tl_id": tl_id,
            "tl_type": tl_types.get(tl_id) if tl_id else None,
        })

    edges = []
    for edge in net.getEdges():
        if edge.getFunction() == "internal":
            continue
        # Форма ребра — по центру его полос, поэтому встречные направления
        # одной улицы на карте не накладываются друг на друга.
        geometry = [[round(c, 7) for c in to_lonlat(x, y)] for x, y, *_ in edge.getShape()]
        edges.append({
            "id": edge.getID(),
            "from": edge.getFromNode().getID(),
            "to": edge.getToNode().getID(),
            "length": round(edge.getLength(), 2),
            "lanes": edge.getLaneNumber(),
            "speed": round(edge.getSpeed(), 2),
            "priority": edge.getPriority(),
            "geometry": geometry,
            "name": edge.getName() or None,
            "road_type": edge.getType() or None,
            "closed": False,
        })

    lons = [n["lon"] for n in nodes]
    lats = [n["lat"] for n in nodes]
    meta = {
        **source_meta,
        "net_file": _rel(net_path),
        "crs": "EPSG:4326",
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_traffic_lights": len({n["tl_id"] for n in nodes if n["tl_id"]}),
        "total_length_km": round(sum(e["length"] for e in edges) / 1000, 2),
        "units": {"length": "m", "speed": "m/s"},
        "sumo_version": sumo_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    network = {"meta": meta, "nodes": nodes, "edges": edges}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(network, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return network


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


def build(osm: Path | None = None, force_grid: bool = False, crop: bool = True,
          out_net: Path = config.BASE_NET, out_json: Path = config.NETWORK_JSON) -> dict:
    out_net.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    osm = None if force_grid else (osm or find_osm())
    if osm is not None:
        print(f"[build_network] OSM: {osm}")
        meta = build_from_osm(osm, out_net, crop=crop)
    else:
        reason = "задан --grid" if force_grid else "map.osm не найден"
        print(f"[build_network] {reason} — строю синтетическую решётку 4×4 (netgenerate)")
        meta = build_grid(out_net)
    normalize_net(out_net)
    network = export_network_json(out_net, out_json, meta)
    m = network["meta"]
    print(f"[build_network] {out_net.name}: узлов {m['n_nodes']}, рёбер {m['n_edges']}, "
          f"светофоров {m['n_traffic_lights']}, {m['total_length_km']} км полос-направлений "
          f"за {time.perf_counter() - t0:.1f} с → {out_json}")
    return network


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--osm", type=Path, help="путь к .osm (по умолчанию sim/data/map.osm или ./map.osm)")
    parser.add_argument("--grid", action="store_true", help="синтетическая решётка 4×4 вместо OSM")
    parser.add_argument("--no-crop", action="store_true", help="не обрезать сеть по <bounds> OSM-файла")
    parser.add_argument("--out-net", type=Path, default=config.BASE_NET)
    parser.add_argument("--out-json", type=Path, default=config.NETWORK_JSON)
    args = parser.parse_args(argv)
    build(args.osm, args.grid, not args.no_crop, args.out_net, args.out_json)


if __name__ == "__main__":
    main()
