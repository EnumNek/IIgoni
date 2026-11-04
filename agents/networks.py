"""Дорожные сети и транспортный спрос для обучения и оценки агентов.

Откуда берётся сеть (по приоритету):
1. Сеть из соседней части ``sim/`` — файл ``sim/data/*.net.xml``
   (сценарий ``sim``). Когда sim/ будет готов, агенты автоматически
   переключатся на неё.
2. Сеть sim/ с изменениями пользователя: ``network.json``, в ``meta``
   которого записаны ``base_network`` (= sim/data/network.json) и
   ``changes`` (так её передаёт backend/). Сеть сценария собирается так же,
   как в датасете sim/: base.net.xml + патч netconvert (sim.changes), поэтому
   id светофоров и фазы незатронутых перекрёстков сохраняются и подходят
   модели, обученные на сети sim/.
3. Пока сети sim/ нет — собственные синтетические решётки 2×2 и 4×4
   (``grid2x2``, ``grid4x4``), построенные утилитой SUMO ``netgenerate``.
4. Любая другая сеть: путь к ``.net.xml`` или к ``network.json``
   (формат контракта; если рядом нет .net.xml, он конвертируется
   в сеть SUMO через ``netconvert``).

Спрос генерируется утилитой SUMO ``randomTrips.py``:

* решётки и прочие сети — поездки между граничными («въездными» и
  «выездными») улицами (+ ``duarouter``), интенсивность меняется во времени
  по профилю часа пик (config.DEMAND_PROFILE), главное направление
  запад↔восток нагружено сильнее (config.MAIN_AXIS_FACTOR);
* сеть sim/ и её варианты — та же модель спроса, что в датасете sim/
  (sim/config.py: 6000 авт/ч, поездки между перекрёстками, транзит через
  границу района), с другим семенем и масштабом для каждого варианта спроса.

Запуск из корня репозитория::

    python -m agents.networks                 # подготовить сети по умолчанию
    python -m agents.networks --scenario grid4x4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # запуск как `python agents/networks.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import config  # noqa: E402  (config выставляет SUMO_HOME)

import sumolib  # noqa: E402


# ---------------------------------------------------------------------------
# Описание сценария
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    """Сеть + всё, что для неё сгенерировано (маршруты, программы actuated)."""

    key: str            # имя сценария: grid2x2, grid4x4, sim, sim-<хэш>, <имя>-<хэш>
    dir: Path           # agents/networks/<key>/ — сгенерированные файлы
    net_file: Path      # сеть SUMO (.net.xml)
    synthetic: bool     # True для наших решёток (известна геометрия «осей»)
    # Чьи модели использовать: у варианта сети sim/ с изменениями — модели "sim"
    models_key: str | None = None
    # Модель спроса: "boundary" — поездки между граничными улицами (решётки,
    # внешние сети); "sim" — модель спроса датасета sim/ (см. generate_sim_trips)
    demand: str = "boundary"

    @property
    def actuated_file(self) -> Path:
        return self.dir / "tls_actuated.add.xml"

    @property
    def network_json(self) -> Path:
        return self.dir / "network.json"

    @property
    def models_dir(self) -> Path:
        return config.MODELS_DIR / (self.models_key or self.key)

    @property
    def routes_dir(self) -> Path:
        return self.dir / ("routes" if self.demand == "boundary" else f"routes-{self.demand}")

    @property
    def sumo_args(self) -> list[str]:
        """Опции SUMO, которых требует модель спроса сценария."""
        if self.demand != "sim":
            return []
        from sim import config as sim_config  # только константы, без зависимостей

        # Как в датасете sim/ (sim/simulate.py): поездки между перекрёстками и
        # динамическая маршрутизация по текущим временам проезда.
        return ["--junction-taz", "--ignore-route-errors",
                "--device.rerouting.probability", "1",
                "--device.rerouting.period", str(sim_config.REROUTING_PERIOD),
                "--device.rerouting.adaptation-interval", str(sim_config.REROUTING_ADAPTATION_INTERVAL)]

    def train_route(self, seed: int) -> Path:
        """Маршруты для обучения: случайный масштаб спроса (обобщение)."""
        path = self.routes_dir / f"train_s{seed}.rou.xml"
        if not path.exists():
            rng = random.Random(seed)
            scale = rng.uniform(*config.TRAIN_DEMAND_SCALE)
            generate_routes(self, path, seed=seed, scale=scale)
        return path

    def eval_route(self, level: str = "nominal", seed: int = config.EVAL_SEEDS[0]) -> Path:
        """Маршруты для оценки: фиксированный уровень спроса, свои семена."""
        path = self.routes_dir / f"eval_{level}_s{seed}.rou.xml"
        if not path.exists():
            generate_routes(self, path, seed=seed, scale=config.EVAL_DEMAND_LEVELS[level])
        return path


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """Запуск утилиты SUMO с понятной ошибкой при неудаче."""
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Команда завершилась с ошибкой:\n{' '.join(cmd)}\n{res.stdout}\n{res.stderr}")


def _tool(name: str) -> str:
    return sumolib.checkBinary(name)


# ---------------------------------------------------------------------------
# 1. Синтетические решётки (netgenerate)
# ---------------------------------------------------------------------------
def grid_tls_ids(n: int) -> list[str]:
    """Имена внутренних узлов решётки netgenerate: столбцы A, B, …; строки 0, 1, …"""
    return [f"{chr(ord('A') + i)}{j}" for i in range(n) for j in range(n)]


def build_grid(key: str, out_dir: Path) -> Path:
    """Строит решётку N×N со светофорами на всех перекрёстках.

    Граничные узлы (концы въездных улиц) — без светофора: светофор там
    ничего бы не регулировал. Программа светофоров — стандартная
    статическая программа SUMO (цикл 90 с, равные фазы): она же служит
    базовым вариантом «фиксированные светофоры».
    """
    p = config.GRID_SCENARIOS[key]
    out_dir.mkdir(parents=True, exist_ok=True)
    net_file = out_dir / f"{key}.net.xml"
    cmd = [
        _tool("netgenerate"), "--grid",
        "--grid.number", str(p["n"]),
        "--grid.length", str(p["block_length"]),
        "--grid.attach-length", str(p["attach_length"]),
        "--default.lanenumber", str(p["lanes"]),
        "--default.speed", str(p["speed"]),
        "--turn-lanes", str(p["turn_lanes"]),
        "--turn-lanes.length", str(p["turn_lane_length"]),
        "--default-junction-type", "priority",
        "--tls.set", ",".join(grid_tls_ids(p["n"])),
        "--tls.default-type", "static",
        "--no-turnarounds", "true",
        "--no-warnings", "true",
        "--output-file", str(net_file),
    ]
    _run(cmd)
    return net_file


# ---------------------------------------------------------------------------
# 2. Сеть из sim/
# ---------------------------------------------------------------------------
_SIM_NET_PREFERRED = ("network", "base", "map", "net")


def find_sim_network() -> Path | None:
    """Ищет базовую сеть SUMO, подготовленную частью sim/.

    Проверяются sim/data/*.net.xml, а также подпапки net/ и sumo/.
    Предпочтение — файлам с именами network/base/map/net.
    """
    candidates: list[Path] = []
    for sub in ("", "net", "sumo"):
        d = config.SIM_DATA_DIR / sub
        if d.is_dir():
            candidates += sorted(d.glob("*.net.xml"))
    if not candidates:
        return None

    def rank(p: Path) -> tuple[int, str]:
        stem = p.name[: -len(".net.xml")].lower()
        for i, name in enumerate(_SIM_NET_PREFERRED):
            if stem == name or stem.startswith(name):
                return (i, p.name)
        return (len(_SIM_NET_PREFERRED), p.name)

    return sorted(candidates, key=rank)[0]


def default_scenarios() -> list[str]:
    """Сценарии по умолчанию: сеть sim/, если она есть, иначе решётки."""
    return ["sim"] if find_sim_network() else list(config.GRID_SCENARIOS)


# ---------------------------------------------------------------------------
# 3. Программы actuated (для сравнения)
# ---------------------------------------------------------------------------
def _is_green(state: str) -> bool:
    return "y" not in state.lower() and any(c in "Gg" for c in state)


def make_actuated_additional(net_file: Path, out_file: Path,
                             min_dur: int = 5, max_dur: int = 50) -> Path:
    """Создаёт дополнительный файл SUMO с actuated-версиями программ.

    Берём те же фазы, что в сети (та же фазовая структура, что у
    фиксированного режима и у RL), меняем тип на ``actuated`` и задаём
    зелёным фазам minDur/maxDur. Значения 5/50 c — те же, что ставит
    сам netconvert (``--tls.min-dur``/``--tls.max-dur`` по умолчанию).
    SUMO сам размещает индукционные петли на подходах. Программа
    загружается через ``-a`` и, будучи загруженной последней,
    становится активной.
    """
    root = ET.parse(net_file).getroot()
    add = ET.Element("additional")
    for tl in root.iter("tlLogic"):
        new = ET.SubElement(add, "tlLogic", id=tl.get("id"), type="actuated",
                            programID="actuated", offset=tl.get("offset", "0"))
        for ph in tl.iter("phase"):
            attrs = {"duration": ph.get("duration"), "state": ph.get("state")}
            if _is_green(ph.get("state")):
                dur = float(ph.get("duration"))
                attrs["minDur"] = str(min(min_dur, dur))
                attrs["maxDur"] = str(max(max_dur, dur))
            ET.SubElement(new, "phase", attrs)
    ET.indent(add)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(add).write(out_file, encoding="utf-8", xml_declaration=True)
    return out_file


# ---------------------------------------------------------------------------
# 4. Экспорт / импорт network.json (формат контракта)
# ---------------------------------------------------------------------------
def _xy_to_lonlat(net, x: float, y: float) -> tuple[float, float]:
    """Координаты SUMO -> (lon, lat).

    Если у сети есть геопривязка (сеть из OSM) — точная проекция SUMO.
    Иначе (синтетическая решётка) — локальная равнопромежуточная
    аппроксимация вокруг config.GEO_ORIGIN (погрешность < 1 м на 1 км).
    """
    if net.hasGeoProj():
        lon, lat = net.convertXY2LonLat(x, y)
        return round(lon, 7), round(lat, 7)
    lon0, lat0 = config.GEO_ORIGIN
    lat = lat0 + y / 110_574.0
    lon = lon0 + x / (111_320.0 * math.cos(math.radians(lat0)))
    return round(lon, 7), round(lat, 7)


def export_network_json(net_file: Path, out_file: Path) -> Path:
    """Сохраняет граф сети в формате контракта sim/data/network.json."""
    net = sumolib.net.readNet(str(net_file), withInternal=False)
    nodes = []
    for node in net.getNodes():
        lon, lat = _xy_to_lonlat(net, *node.getCoord())
        nodes.append({"id": node.getID(), "lon": lon, "lat": lat, "type": node.getType()})
    edges = []
    for e in net.getEdges(withInternal=False):
        edges.append({
            "id": e.getID(),
            "from": e.getFromNode().getID(),
            "to": e.getToNode().getID(),
            "length": round(e.getLength(), 2),
            "lanes": e.getLaneNumber(),
            "speed": round(e.getSpeed(), 2),
            "priority": e.getPriority(),
            "geometry": [list(_xy_to_lonlat(net, x, y)) for x, y in e.getShape()],
        })
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return out_file


def json_to_net(json_file: Path, out_net: Path) -> Path:
    """Конвертирует network.json (контракт) в сеть SUMO через netconvert.

    Узлы и рёбра записываются в «plain XML» SUMO с геокоординатами
    (x=lon, y=lat), netconvert проецирует их в UTM, строит перекрёстки,
    связи полос и программы светофоров для узлов типа traffic_light.
    """
    data = json.loads(Path(json_file).read_text(encoding="utf-8"))
    out_net.parent.mkdir(parents=True, exist_ok=True)
    nod = ET.Element("nodes")
    for n in data["nodes"]:
        ntype = str(n.get("type") or "priority")
        ntype = "traffic_light" if "traffic_light" in ntype else ntype
        if ntype not in ("traffic_light", "priority", "right_before_left", "dead_end",
                         "allway_stop", "unregulated", "priority_stop", "zipper"):
            ntype = "priority"
        ET.SubElement(nod, "node", id=str(n["id"]), x=str(n["lon"]), y=str(n["lat"]), type=ntype)
    edg = ET.Element("edges")
    for e in data["edges"]:
        attrs = {
            "id": str(e["id"]), "from": str(e["from"]), "to": str(e["to"]),
            "numLanes": str(int(e.get("lanes") or 1)), "speed": str(float(e.get("speed") or 13.89)),
            "priority": str(int(e.get("priority") or 1)),
        }
        geom = e.get("geometry") or []
        if len(geom) > 2:
            attrs["shape"] = " ".join(f"{lon},{lat}" for lon, lat in geom)
        ET.SubElement(edg, "edge", attrs)
    nod_file = out_net.with_suffix("").with_suffix(".nod.xml")
    edg_file = out_net.with_suffix("").with_suffix(".edg.xml")
    ET.ElementTree(nod).write(nod_file, encoding="utf-8", xml_declaration=True)
    ET.ElementTree(edg).write(edg_file, encoding="utf-8", xml_declaration=True)
    _run([_tool("netconvert"), "--node-files", str(nod_file), "--edge-files", str(edg_file),
          "--proj.utm", "true", "--no-turnarounds", "true", "--no-warnings", "true",
          "--output-file", str(out_net)])
    return out_net


# ---------------------------------------------------------------------------
# 5. Спрос
# ---------------------------------------------------------------------------
def boundary_edges(net) -> tuple[list, list]:
    """Въездные (нет входящих связей) и выездные (нет исходящих) рёбра."""
    edges = [e for e in net.getEdges(withInternal=False) if e.allows("passenger")]
    entries = [e for e in edges if not e.getIncoming()]
    exits = [e for e in edges if not e.getOutgoing()]
    return entries, exits


def _is_main_axis(edge) -> bool:
    """Главная ось решётки — горизонтальная (запад↔восток)."""
    (x1, y1), (x2, y2) = edge.getFromNode().getCoord(), edge.getToNode().getCoord()
    return abs(x2 - x1) >= abs(y2 - y1)


def generate_routes(sc: Scenario, out_file: Path, seed: int, scale: float = 1.0) -> Path:
    """Генерирует файл маршрутов SUMO (randomTrips.py + duarouter).

    * Источники и стоки — граничные рёбра сети, вес главной оси больше.
    * Интенсивность (авт/ч) по 10-минутным интервалам:
      PEAK_RATE · Σ(весов въездов) · DEMAND_PROFILE[i] · scale
      (для решёток PEAK_RATE_PER_ENTRY, для реальных сетей
      EXTERNAL_PEAK_RATE_PER_LANE — см. config.py).
    * ``--binomial`` даёт случайные (почти пуассоновские) моменты въезда.

    Для сети sim/ и её вариантов — модель спроса датасета sim/
    (см. generate_sim_trips).
    """
    if sc.demand == "sim":
        return generate_sim_trips(sc, out_file, seed=seed, scale=scale)
    net = sumolib.net.readNet(str(sc.net_file))
    entries, exits = boundary_edges(net)
    if not entries or not exits:
        raise RuntimeError(f"В сети {sc.net_file} не найдено граничных рёбер для генерации спроса")

    def weight(e) -> float:
        if sc.synthetic:  # решётка: главная ось тяжелее
            return config.MAIN_AXIS_FACTOR if _is_main_axis(e) else 1.0
        # реальная сеть: вес въезда ~ числу полос, магистрали (приоритет
        # SUMO >= 7: primary/secondary) тяжелее жилых улиц
        main = e.getPriority() >= config.EXTERNAL_MAIN_PRIORITY
        return e.getLaneNumber() * (1.0 if main else config.EXTERNAL_MINOR_FACTOR)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.parent / f".tmp_{out_file.stem}"
    tmp.mkdir(exist_ok=True)
    try:
        for suffix, group in (("src", entries), ("dst", exits)):
            root = ET.Element("edgedata")
            interval = ET.SubElement(root, "interval", begin="0", end=str(config.EPISODE_SECONDS))
            for e in group:
                ET.SubElement(interval, "edge", id=e.getID(), value=str(weight(e)))
            # randomTrips читает веса построчным парсером (sumolib parse_fast),
            # поэтому каждый элемент обязан быть на своей строке.
            ET.indent(root)
            ET.ElementTree(root).write(tmp / f"w.{suffix}.xml", encoding="utf-8")

        rate = config.PEAK_RATE_PER_ENTRY if sc.synthetic else config.EXTERNAL_PEAK_RATE_PER_LANE
        total_peak = rate * sum(weight(e) for e in entries) * scale
        rates = [f"{total_peak * p:.1f}" for p in config.DEMAND_PROFILE]
        # Минимальное расстояние между началом и концом поездки: без него
        # появляются поездки «въехал и выехал на той же стороне», которые
        # при запрете разворотов превращаются в петли вокруг квартала и
        # искусственно забивают сеть.
        if sc.synthetic:
            min_dist = 1.25 * config.GRID_SCENARIOS[sc.key]["block_length"]
        else:
            min_dist = config.MIN_TRIP_DISTANCE_M
        cmd = [
            sys.executable, str(config.SUMO_TOOLS / "randomTrips.py"),
            "-n", str(sc.net_file),
            "-o", str(tmp / "trips.xml"),
            "-r", str(out_file),
            "--seed", str(seed),
            "-b", "0", "-e", str(config.DEMAND_SECONDS),
            "--insertion-rate", *rates,
            "--binomial", "4",
            "--min-distance", str(min_dist),
            "--weights-prefix", str(tmp / "w"),
            "--prefix", f"s{seed}_",
            "--trip-attributes", 'departLane="best" departSpeed="max"',
            "--remove-loops",
        ]
        _run(cmd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        # duarouter оставляет рядом файл альтернативных маршрутов — не нужен
        out_file.with_name(out_file.name.replace(".rou.xml", ".rou.alt.xml")).unlink(missing_ok=True)
    return out_file


def generate_sim_trips(sc: Scenario, out_file: Path, seed: int, scale: float = 1.0) -> Path:
    """Спрос для сети sim/: те же параметры randomTrips, что у датасета sim/.

    sim/generate_traffic.py калибровкой подобрал 6000 авт/ч (сеть близка к
    пропускной способности, см. sim/README.md). Здесь та же модель — поездки
    между перекрёстками (``--junction-taz``), чаще с границы района
    (``--fringe-factor``) и с многополосных улиц (``--lanes``), — но своё
    семя и масштаб на каждый вариант спроса: обучение (семена 0–9),
    валидация и тест не пересекаются. Маршрут выбирает SUMO при въезде
    (опции — Scenario.sumo_args), поэтому файл подходит и для сети с
    закрытыми участками.
    """
    from sim import config as sim_config

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.parent / f".tmp_{out_file.stem}"
    tmp.mkdir(exist_ok=True)
    try:
        _run([
            sys.executable, str(config.SUMO_TOOLS / "randomTrips.py"),
            "-n", str(sc.net_file), "-o", str(out_file),
            "-b", "0", "-e", str(config.DEMAND_SECONDS),
            "--insertion-rate", f"{sim_config.INSERTION_RATE * scale:.1f}", "--random-depart",
            "--fringe-factor", str(sim_config.FRINGE_FACTOR), "--lanes",
            "--min-distance", f"{sim_config.MIN_TRIP_DISTANCE:g}",
            "--junction-taz", "--validate", "--remove-loops",
            "--seed", str(seed), "--prefix", f"s{seed}_",
            "--trip-attributes", 'departLane="best" departSpeed="max"',
        ], cwd=tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_file


# ---------------------------------------------------------------------------
# 6. Сеть sim/ с изменениями (сценарий из backend/)
# ---------------------------------------------------------------------------
SIM_NETWORK_JSON = config.SIM_DATA_DIR / "network.json"


def sim_changes_of(json_file: Path) -> list[dict] | None:
    """Изменения относительно сети sim/, если network.json — её вариант.

    backend/ передаёт сеть сценария JSON-файлом, в meta которого
    ``base_network`` — путь к sim/data/network.json, ``changes`` — список
    изменений по контракту. Для любого другого файла — None.
    """
    try:
        meta = json.loads(Path(json_file).read_text(encoding="utf-8")).get("meta") or {}
    except (OSError, ValueError, AttributeError):
        return None
    base, changes = meta.get("base_network"), meta.get("changes")
    if not base or not isinstance(changes, list) or not SIM_NETWORK_JSON.exists():
        return None
    if Path(base).expanduser().resolve() != SIM_NETWORK_JSON.resolve():
        return None
    return changes


def sim_variant(changes: list[dict]) -> Scenario:
    """Сеть sim/ с изменениями — так же, как сеть сценария в датасете sim/.

    Изменения применяются к графу sim/data/network.json (sim.changes —
    общий контракт), разница с базой компилируется в патч netconvert и
    накладывается на sim/data/base.net.xml. Закрытый участок не удаляется, а
    запрещается для движения, поэтому id светофоров и фазы незатронутых
    перекрёстков те же, что в базовой сети: используются модели "sim".
    Одинаковые итоговые состояния сети дают один и тот же сценарий (кэш).
    Неверное изменение — sim.changes.ChangeError (это ValueError).
    """
    base_net = find_sim_network()
    if base_net is None or not SIM_NETWORK_JSON.exists():
        raise FileNotFoundError("Сеть sim/ ещё не готова (нет sim/data/*.net.xml и network.json)")
    from sim.changes import apply_changes, diff_states, to_netconvert_patches

    base = json.loads(SIM_NETWORK_JSON.read_text(encoding="utf-8"))
    diff = diff_states(base, apply_changes(base, changes))
    edg, nod = to_netconvert_patches(diff)
    if edg is None and nod is None:
        return get_scenario("sim")
    key = "sim-" + hashlib.sha1(json.dumps(diff, sort_keys=True).encode()).hexdigest()[:10]
    d = config.NETWORKS_DIR / key
    net = d / f"{key}.net.xml"
    if not net.exists():
        d.mkdir(parents=True, exist_ok=True)
        cmd = [_tool("netconvert"), "-s", str(base_net), "-o", str(net), "--no-warnings", "true"]
        for text, name, flag in ((edg, "patch.edg.xml", "-e"), (nod, "patch.nod.xml", "-n")):
            if text:
                (d / name).write_text(text, encoding="utf-8")
                cmd += [flag, str(d / name)]
        _run(cmd)
    return _ensure_artifacts(Scenario(key, d, net, synthetic=False, models_key="sim", demand="sim"))


# ---------------------------------------------------------------------------
# 7. Разрешение сценария по имени или пути
# ---------------------------------------------------------------------------
def _ensure_artifacts(sc: Scenario) -> Scenario:
    if not sc.actuated_file.exists():
        make_actuated_additional(sc.net_file, sc.actuated_file)
    if not sc.network_json.exists():
        export_network_json(sc.net_file, sc.network_json)
    return sc


def _path_key(path: Path) -> str:
    stem = path.name.split(".")[0]
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    return f"{stem}-{digest}"


def get_scenario(spec: str | Path) -> Scenario:
    """Возвращает сценарий (и при необходимости строит сеть).

    ``spec``: ``grid2x2`` | ``grid4x4`` | ``sim`` | путь к .net.xml |
    путь к network.json | путь к папке сценария agents/networks/<key>.
    """
    spec_str = str(spec)
    if spec_str in config.GRID_SCENARIOS:
        d = config.NETWORKS_DIR / spec_str
        net = d / f"{spec_str}.net.xml"
        if not net.exists():
            build_grid(spec_str, d)
        return _ensure_artifacts(Scenario(spec_str, d, net, synthetic=True))

    if spec_str == "sim":
        net = find_sim_network()
        if net is None:
            raise FileNotFoundError(
                "Сеть sim/ ещё не готова (нет sim/data/*.net.xml) — используйте grid2x2/grid4x4")
        return _ensure_artifacts(Scenario("sim", config.NETWORKS_DIR / "sim", net, synthetic=False, demand="sim"))

    path = Path(spec).expanduser()
    if not path.is_absolute() and not path.exists():
        path = config.REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Сеть не найдена: {spec}")
    path = path.resolve()

    if path.is_dir():  # папка сценария
        nets = sorted(path.glob("*.net.xml"))
        if not nets:
            raise FileNotFoundError(f"В папке {path} нет файла .net.xml")
        return get_scenario(nets[0])

    if path.name.endswith(".net.xml"):
        if path.parent.parent == config.NETWORKS_DIR:  # одна из наших сетей
            key = path.parent.name
            variant = key.startswith("sim-")  # вариант сети sim/ (см. sim_variant)
            return _ensure_artifacts(Scenario(key, path.parent, path, synthetic=key in config.GRID_SCENARIOS,
                                              models_key="sim" if variant else None,
                                              demand="sim" if variant else "boundary"))
        sim_net = find_sim_network()
        if sim_net is not None and sim_net.resolve() == path:
            return get_scenario("sim")
        key = _path_key(path)
        return _ensure_artifacts(Scenario(key, config.NETWORKS_DIR / key, path, synthetic=False))

    if path.suffix == ".json":
        if path.parent.parent == config.NETWORKS_DIR:  # network.json нашей сети
            return get_scenario(path.parent)
        if path.parent.resolve() == config.SIM_DATA_DIR.resolve() and find_sim_network():
            return get_scenario("sim")
        changes = sim_changes_of(path)
        if changes is not None and find_sim_network():
            return sim_variant(changes)
        siblings = sorted(path.parent.glob("*.net.xml"))
        if siblings:
            return get_scenario(siblings[0])
        key = _path_key(path)
        d = config.NETWORKS_DIR / key
        net = d / f"{key}.net.xml"
        if not net.exists():
            json_to_net(path, net)
        return _ensure_artifacts(Scenario(key, d, net, synthetic=False))

    raise ValueError(f"Неизвестный формат сети: {spec}")


def prepare(spec: str, all_routes: bool = True) -> Scenario:
    """Строит сеть и (опционально) все файлы маршрутов сценария."""
    sc = get_scenario(spec)
    if all_routes:
        for s in config.TRAIN_SEEDS:
            sc.train_route(s)
        for level in config.EVAL_DEMAND_LEVELS:
            for s in config.EVAL_SEEDS:
                sc.eval_route(level, s)
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description="Подготовка сетей и спроса для RL-агентов")
    ap.add_argument("--scenario", nargs="+", default=None,
                    help="grid2x2 grid4x4 sim или путь к сети (по умолчанию — sim/ или решётки)")
    ap.add_argument("--no-routes", action="store_true", help="только сеть, без маршрутов")
    args = ap.parse_args()
    for spec in args.scenario or default_scenarios():
        sc = prepare(spec, all_routes=not args.no_routes)
        net = sumolib.net.readNet(str(sc.net_file))
        entries, exits = boundary_edges(net)
        print(f"[{sc.key}] сеть: {sc.net_file.relative_to(config.REPO_ROOT)}; "
              f"светофоров: {len(net.getTrafficLights())}; въездов: {len(entries)}; выездов: {len(exits)}")


if __name__ == "__main__":
    main()
