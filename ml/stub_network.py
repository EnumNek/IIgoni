"""Заглушка части sim/: дорожная сеть в формате контракта network.json.

Пока sim/ не выдала sim/data/network.json, ml/ строит сеть сама — тем же
инструментом (SUMO), что и sim/, и с тем же форматом выхода. Результат
кладётся в ml/data/stub/ (в чужие папки ml/ не пишет).

Порядок выбора источника (режим ``auto``):
  1. sim/data/network.json — настоящий результат части sim/;
  2. sim/data/map.osm — OSM-карта района -> netconvert;
  3. map.osm в корне репозитория — та же карта района (так её загрузил
     автор проекта; когда sim/ перенесёт её в sim/data/, сработает п. 2);
  4. синтетическая решётка 4×4 через netgenerate (если SUMO не установлен —
     такая же решётка строится на чистом Python).

Запуск:  python -m ml.stub_network [--source auto|osm|grid] [--osm PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from ml.config import REPO_ROOT, SIM_DATA_DIR, STUB_DIR

log = logging.getLogger(__name__)

# Центр района из map.osm (Петропавловск) — используется для привязки
# решётки netgenerate к координатам, чтобы она отображалась на карте.
DEFAULT_CENTER = (69.1476, 54.8655)  # lon, lat

# Решётка 4×4: «магистрали» — столбец B и строка 2 (2 полосы, 60 км/ч,
# светофоры на пересечениях), остальные улицы — 1 полоса, 50 км/ч.
GRID_N = 4
GRID_LEN = 200.0
GRID_ATTACH = 150.0
ARTERIAL_COL = "B"
ARTERIAL_ROW = 2


def sumo_home() -> Path | None:
    try:
        import sumo  # pip install eclipse-sumo

        return Path(sumo.SUMO_HOME)
    except ImportError:
        exe = shutil.which("netconvert")
        return Path(exe).resolve().parent.parent if exe else None


def sumo_tool(name: str) -> str | None:
    home = sumo_home()
    if home is not None:
        p = home / "bin" / name
        if p.exists():
            return str(p)
    return shutil.which(name)


def _import_sumolib():
    try:
        import sumolib  # noqa: F401
    except ImportError:
        home = sumo_home()
        if home is None:
            raise
        sys.path.append(str(home / "tools"))
    import sumolib

    return sumolib


def _run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Команда завершилась с ошибкой: {' '.join(cmd)}\n{res.stderr[-2000:]}")


def net_xml_to_network(net_path: str | Path, center: tuple[float, float] | None = None) -> dict:
    """Конвертирует SUMO .net.xml в network.json по контракту.

    Если у сети нет геопривязки (netgenerate), координаты x/y в метрах
    переносятся на местность вокруг ``center`` (lon, lat).
    """
    sumolib = _import_sumolib()
    net = sumolib.net.readNet(str(net_path), withInternal=False)
    geo = net.hasGeoProj()
    lon0, lat0 = center or DEFAULT_CENTER
    xmin, ymin, xmax, ymax = net.getBoundary()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    def to_lonlat(x: float, y: float) -> tuple[float, float]:
        if geo:
            lon, lat = net.convertXY2LonLat(x, y)
        else:
            lat = lat0 + (y - cy) / 110540.0
            lon = lon0 + (x - cx) / (111320.0 * math.cos(math.radians(lat0)))
        return round(lon, 7), round(lat, 7)

    nodes = []
    for n in net.getNodes():
        lon, lat = to_lonlat(*n.getCoord()[:2])
        nodes.append({"id": n.getID(), "lon": lon, "lat": lat, "type": n.getType()})
    edges = []
    for e in net.getEdges():
        if e.getFunction() == "internal" or e.getID().startswith(":"):
            continue
        if not e.allows("passenger"):
            continue
        shape = e.getShape()
        edges.append({
            "id": e.getID(),
            "from": e.getFromNode().getID(),
            "to": e.getToNode().getID(),
            "length": round(e.getLength(), 2),
            "lanes": sum(1 for ln in e.getLanes() if ln.allows("passenger")) or e.getLaneNumber(),
            "speed": round(e.getSpeed(), 2),
            "priority": e.getPriority(),
            "geometry": [list(to_lonlat(x, y)) for x, y in [p[:2] for p in shape]],
        })
    used = {e["from"] for e in edges} | {e["to"] for e in edges}
    nodes = [n for n in nodes if n["id"] in used]
    return {"nodes": nodes, "edges": edges,
            "meta": {"source": str(net_path), "generator": "ml/stub_network.py"}}


def osm_center(osm_path: Path) -> tuple[float, float] | None:
    """Центр области из тега <bounds> OSM-файла (читаем только начало файла)."""
    try:
        for _, el in ET.iterparse(osm_path, events=("start",)):
            if el.tag == "bounds":
                a = {k: float(v) for k, v in el.attrib.items()}
                return (a["minlon"] + a["maxlon"]) / 2, (a["minlat"] + a["maxlat"]) / 2
            if el.tag in ("node", "way"):
                return None
    except (ET.ParseError, KeyError, OSError):
        return None
    return None


def build_osm_network(osm_path: Path, out_dir: Path) -> dict:
    """map.osm -> network.net.xml (netconvert) -> network.json."""
    netconvert = sumo_tool("netconvert")
    if netconvert is None:
        raise RuntimeError("netconvert не найден: pip install eclipse-sumo")
    out_dir.mkdir(parents=True, exist_ok=True)
    net_xml = out_dir / "network.net.xml"
    _run([netconvert, "--osm-files", str(osm_path), "-o", str(net_xml),
          "--keep-edges.by-vclass", "passenger", "--remove-edges.isolated",
          "--geometry.remove", "--roundabouts.guess", "--ramps.guess",
          "--junctions.join", "--tls.guess-signals", "--tls.discard-simple", "--tls.join",
          "--no-turnarounds.except-deadend", "--output.street-names", "--no-warnings"])
    network = net_xml_to_network(net_xml)
    network["meta"]["source"] = f"OSM: {osm_path.name}"
    return network


def _grid_names(n: int) -> list[str]:
    """Имена столбцов как в netgenerate: A, B, …, Z, AA, AB, …"""
    out = []
    for i in range(n):
        s, k = "", i
        while True:
            s = chr(ord("A") + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        out.append(s)
    return out


def _grid_patch(edge_ids_with_nodes: list[tuple[str, str, str]], n: int = GRID_N) -> tuple[list[str], set[str]]:
    """Какие рёбра решётки — магистрали, и в каких узлах светофоры.
    Для 4×4 — столбец B и строка 2; для больших решёток — каждый 3-й столбец/строка."""
    letters = _grid_names(n)
    cols = [ord(ARTERIAL_COL) - 65] if n == GRID_N else list(range(1, n, 3))
    rows = [ARTERIAL_ROW] if n == GRID_N else list(range(2, n, 3))
    art_nodes_c = [{f"{letters[c]}{j}" for j in range(n)} | {f"bottom{c}", f"top{c}"} for c in cols]
    art_nodes_r = [{f"{x}{r}" for x in letters} | {f"left{r}", f"right{r}"} for r in rows]
    groups = art_nodes_c + art_nodes_r
    arterial = [eid for eid, u, v in edge_ids_with_nodes if any(u in g and v in g for g in groups)]
    tls = {f"{letters[c]}{j}" for c in cols for j in range(n)} | {f"{x}{r}" for r in rows for x in letters}
    return arterial, tls


def build_grid_network(out_dir: Path) -> dict:
    """Решётка 4×4 (netgenerate) с подъездными рёбрами по периметру."""
    netgenerate, netconvert = sumo_tool("netgenerate"), sumo_tool("netconvert")
    if netgenerate is None or netconvert is None:
        log.warning("SUMO не найден — решётка строится на Python (без network.net.xml)")
        return build_grid_python()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "grid_raw.net.xml"
        _run([netgenerate, "--grid", f"--grid.number={GRID_N}", f"--grid.length={GRID_LEN}",
              f"--grid.attach-length={GRID_ATTACH}", "--default.lanenumber=1",
              "--default.speed=13.89", "--default.priority=1", "--no-turnarounds.except-deadend",
              "-o", str(raw)])
        base = net_xml_to_network(raw)
        arterial, tls = _grid_patch([(e["id"], e["from"], e["to"]) for e in base["edges"]])
        patch = Path(tmp) / "patch.edg.xml"
        root = ET.Element("edges")
        for eid in arterial:
            ET.SubElement(root, "edge", id=eid, numLanes="2", speed="16.67", priority="3")
        ET.ElementTree(root).write(patch)
        net_xml = out_dir / "network.net.xml"
        _run([netconvert, "-s", str(raw), "-e", str(patch), "--tls.set", ",".join(sorted(tls)),
              "--no-turnarounds.except-deadend", "-o", str(net_xml), "--no-warnings"])
    network = net_xml_to_network(net_xml)
    network["meta"]["source"] = "netgenerate grid 4x4"
    return network


def build_grid_python(n: int = GRID_N) -> dict:
    """Та же решётка n×n, но без SUMO (идентификаторы как у netgenerate).
    Используется, если SUMO не установлен, и для замера масштабируемости инференса."""
    GRID_N_ = n
    letters = _grid_names(GRID_N_)
    lon0, lat0 = DEFAULT_CENTER
    pos: dict[str, tuple[float, float]] = {}
    off = (GRID_N_ - 1) * GRID_LEN / 2
    for i, c in enumerate(letters):
        for j in range(GRID_N_):
            pos[f"{c}{j}"] = (i * GRID_LEN - off, j * GRID_LEN - off)
        pos[f"bottom{i}"] = (i * GRID_LEN - off, -GRID_ATTACH - off)
        pos[f"top{i}"] = (i * GRID_LEN - off, (GRID_N_ - 1) * GRID_LEN + GRID_ATTACH - off)
    for j in range(GRID_N_):
        pos[f"left{j}"] = (-GRID_ATTACH - off, j * GRID_LEN - off)
        pos[f"right{j}"] = ((GRID_N_ - 1) * GRID_LEN + GRID_ATTACH - off, j * GRID_LEN - off)

    def ll(p):
        return (round(lon0 + p[0] / (111320.0 * math.cos(math.radians(lat0))), 7),
                round(lat0 + p[1] / 110540.0, 7))

    pairs = []
    for i, c in enumerate(letters):
        for j in range(GRID_N_):
            if j + 1 < GRID_N_:
                pairs.append((f"{c}{j}", f"{c}{j + 1}"))
            if i + 1 < GRID_N_:
                pairs.append((f"{c}{j}", f"{letters[i + 1]}{j}"))
        pairs.append((f"{c}0", f"bottom{i}"))
        pairs.append((f"{c}{GRID_N_ - 1}", f"top{i}"))
    for j in range(GRID_N_):
        pairs.append((f"A{j}", f"left{j}"))
        pairs.append((f"{letters[-1]}{j}", f"right{j}"))
    triples = []
    for a, b in pairs:
        triples += [(f"{a}{b}", a, b), (f"{b}{a}", b, a)]
    arterial, tls = _grid_patch(triples, GRID_N_)
    art = set(arterial)
    edges = []
    for eid, u, v in triples:
        L = math.dist(pos[u], pos[v])
        a = eid in art
        edges.append({"id": eid, "from": u, "to": v, "length": round(L, 2), "lanes": 2 if a else 1,
                      "speed": 16.67 if a else 13.89, "priority": 3 if a else 1,
                      "geometry": [list(ll(pos[u])), list(ll(pos[v]))]})
    nodes = []
    for nid, p in pos.items():
        lon, lat = ll(p)
        ntype = "traffic_light" if nid in tls else ("dead_end" if not nid[0].isupper() else "priority")
        nodes.append({"id": nid, "lon": lon, "lat": lat, "type": ntype})
    return {"nodes": nodes, "edges": edges,
            "meta": {"source": f"python grid {n}x{n}", "generator": "ml/stub_network.py"}}


def find_osm() -> Path | None:
    for p in (SIM_DATA_DIR / "map.osm", REPO_ROOT / "map.osm"):
        if p.exists():
            return p
    return None


def resolve_network(source: str = "auto", osm: str | Path | None = None,
                    out_dir: Path = STUB_DIR, rebuild: bool = False) -> tuple[dict, Path]:
    """Возвращает (network, путь к network.json) согласно порядку выбора источника."""
    if source == "auto":
        real = SIM_DATA_DIR / "network.json"
        if real.exists():
            with open(real, encoding="utf-8") as f:
                return json.load(f), real
        source = "osm" if (osm or find_osm()) else "grid"
    if source not in ("osm", "grid"):
        raise ValueError(f"Неизвестный источник сети: {source}")
    target_dir = out_dir / source
    path = target_dir / "network.json"
    if path.exists() and not rebuild:
        with open(path, encoding="utf-8") as f:
            return json.load(f), path
    if source == "osm":
        osm_path = Path(osm) if osm else find_osm()
        if osm_path is None or not osm_path.exists():
            raise FileNotFoundError("OSM-карта не найдена (ожидалась sim/data/map.osm или map.osm)")
        network = build_osm_network(osm_path, target_dir)
    else:
        network = build_grid_network(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(network, f, ensure_ascii=False)
    log.info("Заглушка сети: %s (%d узлов, %d рёбер)", path, len(network["nodes"]),
             len(network["edges"]))
    return network, path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto", choices=["auto", "osm", "grid"])
    ap.add_argument("--osm", default=None, help="путь к .osm (по умолчанию ищется автоматически)")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    network, path = resolve_network(args.source, args.osm, rebuild=args.rebuild)
    tl = sum(1 for n in network["nodes"] if str(n.get("type", "")).startswith("traffic_light"))
    print(f"{path}: {len(network['nodes'])} узлов, {len(network['edges'])} рёбер, {tl} светофоров")


if __name__ == "__main__":
    main()
