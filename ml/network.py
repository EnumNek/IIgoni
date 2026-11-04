"""Представление дорожной сети и применение изменений пользователя.

Сеть приходит в формате контракта (sim/data/network.json):

    {"nodes": [{"id", "lon", "lat", "type"}],
     "edges": [{"id", "from", "to", "length", "lanes", "speed", "priority", "geometry"}]}

Здесь она раскладывается на две части:

* ``Topology`` — неизменяемая структура: идентификаторы, связность, длины,
  приоритеты, координаты. Не меняется ни одним типом изменений.
* ``State`` — изменяемые атрибуты: число полос, разрешённая скорость,
  признак закрытия ребра и наличие светофора в узле. Именно их меняют
  изменения ``{type, target_id, value}``.

Изменения применяются последовательно к копии базового состояния, поэтому
модель видит не «список действий», а итоговое состояние сети и его разницу
с базой. Благодаря этому, например, пара «close X» + «open X» корректно
сводится к отсутствию изменений, а порядок действий учитывается.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CHANGE_TYPES = ("lanes", "speed", "close", "open", "traffic_light")
EDGE_CHANGE_TYPES = ("lanes", "speed", "close", "open")

DEFAULT_SPEED = 13.89  # м/с (50 км/ч) — если в network.json скорость не указана
DEFAULT_LANES = 1
MAX_LANES = 8
# Скорость больше этого порога считаем заданной в км/ч (типичная ошибка интерфейса):
# 50 м/с = 180 км/ч — нереалистично для городской улицы.
KMH_THRESHOLD = 50.0

_TRUE_WORDS = {"1", "true", "yes", "on", "add", "install", "enable", "traffic_light",
               "fixed", "static", "actuated", "rl"}
_FALSE_WORDS = {"0", "false", "no", "off", "remove", "delete", "disable", "none",
                "priority", "uncontrolled"}


def is_tl_type(node_type: str | None) -> bool:
    """Тип узла SUMO со светофором: traffic_light, traffic_light_right_on_red, ..."""
    return bool(node_type) and str(node_type).startswith("traffic_light")


@dataclass(frozen=True)
class Topology:
    """Неизменяемая часть сети (рёбра — участки дорог, узлы — перекрёстки)."""

    node_ids: list[str]
    node_pos: dict[str, int]
    node_type: list[str]
    node_lon: np.ndarray
    node_lat: np.ndarray
    edge_ids: list[str]
    edge_pos: dict[str, int]
    src: np.ndarray  # индекс узла-начала ребра
    dst: np.ndarray  # индекс узла-конца ребра
    length: np.ndarray  # м
    priority: np.ndarray
    geometry: list[list[list[float]] | None] = field(repr=False)
    fringe: np.ndarray = field(repr=False)  # узел — граница района (один сосед)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edge_ids)

    def signature(self) -> str:
        """Хэш структуры сети — для кэширования производных структур (line graph)."""
        h = hashlib.sha1()
        h.update("|".join(self.edge_ids).encode())
        h.update(self.src.tobytes())
        h.update(self.dst.tobytes())
        return h.hexdigest()[:16]


@dataclass
class State:
    """Изменяемые атрибуты сети."""

    lanes: np.ndarray  # (E,) float
    speed: np.ndarray  # (E,) float, м/с
    closed: np.ndarray  # (E,) bool
    tl: np.ndarray  # (V,) bool — есть ли светофор в узле

    def copy(self) -> "State":
        return State(self.lanes.copy(), self.speed.copy(), self.closed.copy(), self.tl.copy())


def load_network(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _haversine(lon1, lat1, lon2, lat2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def parse_network(network: dict) -> tuple[Topology, State]:
    """Разбирает network.json в (Topology, базовое State).

    Разбор намеренно терпим к неполным данным (заглушки соседних частей):
    отсутствующие поля заполняются значениями по умолчанию, узлы, на которые
    ссылаются рёбра, но которых нет в списке nodes, создаются автоматически.
    """
    nodes = list(network.get("nodes", []))
    edges = list(network.get("edges", []))
    if not edges:
        raise ValueError("В сети нет рёбер (network['edges'] пуст)")

    node_ids: list[str] = []
    node_pos: dict[str, int] = {}
    node_type: list[str] = []
    lon: list[float] = []
    lat: list[float] = []

    def add_node(nid: str, n: dict | None) -> int:
        node_pos[nid] = len(node_ids)
        node_ids.append(nid)
        n = n or {}
        node_type.append(str(n.get("type") or "unknown"))
        lon.append(float(n["lon"]) if n.get("lon") is not None else float("nan"))
        lat.append(float(n["lat"]) if n.get("lat") is not None else float("nan"))
        return node_pos[nid]

    for n in nodes:
        nid = str(n["id"])
        if nid not in node_pos:
            add_node(nid, n)

    edge_ids, src, dst, length, priority, lanes, speed, closed, geometry = ([] for _ in range(9))
    edge_pos: dict[str, int] = {}
    missing_nodes = 0
    for e in edges:
        eid = str(e["id"])
        if eid in edge_pos:
            raise ValueError(f"Дублирующийся id ребра: {eid}")
        u, v = str(e["from"]), str(e["to"])
        for nid in (u, v):
            if nid not in node_pos:
                add_node(nid, None)
                missing_nodes += 1
        edge_pos[eid] = len(edge_ids)
        edge_ids.append(eid)
        src.append(node_pos[u])
        dst.append(node_pos[v])
        geom = e.get("geometry")
        geometry.append(geom if geom else None)
        L = e.get("length")
        if L is None or not np.isfinite(float(L)) or float(L) <= 0:
            if geom and len(geom) >= 2:
                L = sum(_haversine(*geom[i][:2], *geom[i + 1][:2]) for i in range(len(geom) - 1))
            else:
                L = _haversine(lon[node_pos[u]], lat[node_pos[u]], lon[node_pos[v]], lat[node_pos[v]])
            L = L if np.isfinite(L) and L > 0 else 100.0
        length.append(max(float(L), 1.0))
        lanes.append(float(e.get("lanes") or DEFAULT_LANES))
        speed.append(float(e.get("speed") or DEFAULT_SPEED))
        priority.append(float(e.get("priority") or 0))
        closed.append(bool(e.get("closed", False)))
    if missing_nodes:
        log.warning("В network.json не хватало %d узлов — созданы автоматически", missing_nodes)

    src_a = np.asarray(src, dtype=np.int64)
    dst_a = np.asarray(dst, dtype=np.int64)
    V = len(node_ids)

    # Граничный узел района: связан ровно с одним соседним узлом
    # (тупик или место, где дорога уходит за пределы карты).
    neigh: list[set[int]] = [set() for _ in range(V)]
    for a, b in zip(src_a, dst_a):
        if a != b:
            neigh[a].add(b)
            neigh[b].add(a)
    fringe = np.array([len(s) <= 1 or node_type[i] == "dead_end" for i, s in enumerate(neigh)])

    topo = Topology(
        node_ids=node_ids, node_pos=node_pos, node_type=node_type,
        node_lon=np.asarray(lon), node_lat=np.asarray(lat),
        edge_ids=edge_ids, edge_pos=edge_pos, src=src_a, dst=dst_a,
        length=np.asarray(length), priority=np.asarray(priority),
        geometry=geometry, fringe=fringe,
    )
    state = State(
        lanes=np.asarray(lanes, dtype=np.float64),
        speed=np.asarray(speed, dtype=np.float64),
        closed=np.asarray(closed, dtype=bool),
        tl=np.array([is_tl_type(t) for t in node_type], dtype=bool),
    )
    return topo, state


def _parse_bool(value, current: bool) -> bool:
    """Значение для изменения traffic_light: есть ли светофор после изменения.

    Как в общем контракте (sim/changes.py): null, как и false/"off"/"none",
    означает «убрать светофор»; true/"static"/"actuated" — светофор есть.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in _TRUE_WORDS:
        return True
    if s in _FALSE_WORDS:
        return False
    raise ValueError(f"Непонятное значение для traffic_light: {value!r}")


def normalize_change(change: dict) -> dict:
    """Проверяет формат одного изменения и приводит его к каноническому виду."""
    if not isinstance(change, dict):
        raise ValueError(f"Изменение должно быть объектом, получено: {change!r}")
    ctype = change.get("type")
    if ctype not in CHANGE_TYPES:
        raise ValueError(f"Неизвестный тип изменения {ctype!r}; допустимы {CHANGE_TYPES}")
    if change.get("target_id") is None:
        raise ValueError(f"У изменения нет target_id: {change!r}")
    return {"type": ctype, "target_id": str(change["target_id"]), "value": change.get("value")}


def apply_changes(topo: Topology, base: State, changes: list[dict] | None) -> State:
    """Применяет список изменений к копии базового состояния.

    Семантика (принятая в ml/, совпадает с генератором синтетических данных):
      * lanes  — value: новое число полос (целое >= 1);
      * speed  — value: новая разрешённая скорость, м/с (значения > 50
                 считаются км/ч и переводятся, с предупреждением);
      * close  — ребро target_id закрывается для движения (value игнорируется);
      * open   — ребро снова открывается (в т.ч. помеченное ``closed`` в network.json);
      * traffic_light — target_id: id узла; value: true/1/"add" — поставить
                 светофор, false/0/"remove"/null — убрать (как в sim/changes.py).
    """
    state = base.copy()
    for raw in changes or []:
        ch = normalize_change(raw)
        ctype, tid, value = ch["type"], ch["target_id"], ch["value"]
        if ctype in EDGE_CHANGE_TYPES:
            if tid not in topo.edge_pos:
                raise ValueError(f"Изменение {ctype}: ребро {tid!r} не найдено в сети")
            i = topo.edge_pos[tid]
            if ctype == "close":
                state.closed[i] = True
            elif ctype == "open":
                state.closed[i] = False
            elif ctype == "lanes":
                try:
                    n = int(round(float(value)))
                except (TypeError, ValueError):
                    raise ValueError(f"lanes: value должно быть числом, получено {value!r}") from None
                if n < 1:
                    raise ValueError("lanes: число полос должно быть >= 1 (для закрытия используйте 'close')")
                state.lanes[i] = min(n, MAX_LANES)
            elif ctype == "speed":
                try:
                    s = float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"speed: value должно быть числом, получено {value!r}") from None
                if not np.isfinite(s) or s <= 0:
                    raise ValueError(f"speed: скорость должна быть > 0, получено {value!r}")
                if s > KMH_THRESHOLD:
                    log.warning("speed=%s для ребра %s похоже на км/ч — переведено в м/с", s, tid)
                    s = s / 3.6
                state.speed[i] = s
        else:  # traffic_light
            if tid not in topo.node_pos:
                raise ValueError(f"Изменение traffic_light: узел {tid!r} не найден в сети")
            j = topo.node_pos[tid]
            state.tl[j] = _parse_bool(value, bool(state.tl[j]))
    return state
