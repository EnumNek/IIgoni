"""Изменения дорожной сети по контракту проекта.

Изменение — словарь {type, target_id, value}:

    type            target_id   value
    -------------   ---------   --------------------------------------------------
    "lanes"         id ребра    новое число полос, целое 1..6
    "speed"         id ребра    новая разрешённая скорость, м/с (1..40)
    "close"         id ребра    не используется (null) — движение по ребру запрещено
    "open"          id ребра    не используется (null) — снять запрет движения
    "traffic_light" id узла     "static" | "actuated" | "off"
                                (синонимы: true/"on" → "static", false/null/"none" → "off")

Ключевое решение: список изменений сначала применяется к графу network.json
(чистый Python, без SUMO) и даёт ИТОГОВОЕ состояние сети. Изменения
применяются по порядку, поэтому [close A, open A] ничего не меняет, а
open снимает закрытие и с ребра, закрытого ещё в исходной сети (closed=true).
Затем отличия итогового состояния от базового компилируются в патчи
netconvert. Так нейросеть (ml/) и SUMO гарантированно понимают изменения
одинаково — ml/ может импортировать apply_changes() из этого модуля.

Модуль не зависит от SUMO и pandas.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Iterable
from xml.sax.saxutils import quoteattr

CHANGE_TYPES = ("lanes", "speed", "close", "open", "traffic_light")
TL_VALUES = ("static", "actuated", "off")
LANES_RANGE = (1, 6)
SPEED_RANGE = (1.0, 40.0)   # м/с: 3.6 … 144 км/ч


class ChangeError(ValueError):
    """Изменение не соответствует контракту или сети."""


def _index(network: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    return ({e["id"]: e for e in network["edges"]}, {n["id"]: n for n in network["nodes"]})


def normalize_tl_value(value: Any) -> str:
    if value is True or (isinstance(value, str) and value.lower() == "on"):
        return "static"
    if value is None or value is False or (isinstance(value, str) and value.lower() in ("off", "none")):
        return "off"
    if isinstance(value, str) and value.lower() in TL_VALUES:
        return value.lower()
    raise ChangeError(f"traffic_light: недопустимое value={value!r}, ожидается одно из {TL_VALUES}")


def validate_change(change: dict, edges: dict[str, dict], nodes: dict[str, dict]) -> dict:
    """Проверяет одно изменение и возвращает его нормализованную копию."""
    if not isinstance(change, dict):
        raise ChangeError(f"изменение должно быть объектом, получено {type(change).__name__}")
    ctype = change.get("type")
    target = change.get("target_id")
    value = change.get("value")
    if ctype not in CHANGE_TYPES:
        raise ChangeError(f"неизвестный type={ctype!r}, ожидается одно из {CHANGE_TYPES}")
    target = None if target is None else str(target)

    if ctype == "traffic_light":
        if target not in nodes:
            raise ChangeError(f"traffic_light: узел {target!r} не найден в сети")
        tl = normalize_tl_value(value)
        if tl != "off" and nodes[target]["type"] == "dead_end":
            raise ChangeError(f"traffic_light: узел {target!r} — тупик, светофор невозможен")
        return {"type": ctype, "target_id": target, "value": tl}

    if target not in edges:
        raise ChangeError(f"{ctype}: ребро {target!r} не найдено в сети")
    if ctype == "lanes":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ChangeError(f"lanes: value должно быть целым числом, получено {value!r}")
        lo, hi = LANES_RANGE
        if not lo <= int(value) <= hi:
            raise ChangeError(f"lanes: value={value} вне диапазона {lo}..{hi}")
        return {"type": ctype, "target_id": target, "value": int(value)}
    if ctype == "speed":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ChangeError(f"speed: value должно быть числом (м/с), получено {value!r}")
        lo, hi = SPEED_RANGE
        if not lo <= float(value) <= hi:
            raise ChangeError(f"speed: value={value} м/с вне диапазона {lo}..{hi}")
        return {"type": ctype, "target_id": target, "value": round(float(value), 2)}
    # close / open
    return {"type": ctype, "target_id": target, "value": None}


def validate_changes(network: dict, changes: Iterable[dict]) -> list[dict]:
    edges, nodes = _index(network)
    return [validate_change(c, edges, nodes) for c in changes]


def apply_changes(network: dict, changes: Iterable[dict]) -> dict:
    """Применяет изменения к графу network.json и возвращает НОВЫЙ граф.

    Исходный словарь не изменяется. Элементы, итоговое состояние которых
    отличается от исходного, получают поле "changed": true (подсветка на карте,
    признак для ML). Пара [close A, open A] ребро A не помечает.
    """
    result = copy.deepcopy(network)
    edges, nodes = _index(result)
    for element in (*result["edges"], *result["nodes"]):
        element.pop("changed", None)
    for change in validate_changes(network, changes):
        ctype, target, value = change["type"], change["target_id"], change["value"]
        if ctype == "traffic_light":
            node = nodes[target]
            if value == "off":
                if node["type"].startswith("traffic_light"):
                    node["type"] = "priority"
                node["tl_id"] = None
                node["tl_type"] = None
            else:
                node["type"] = "traffic_light"
                node["tl_id"] = node.get("tl_id") or target
                node["tl_type"] = value
            continue
        edge = edges[target]
        if ctype == "lanes":
            edge["lanes"] = value
        elif ctype == "speed":
            edge["speed"] = value
        elif ctype == "close":
            edge["closed"] = True
        elif ctype == "open":
            edge["closed"] = False
    src_edges, src_nodes = _index(network)
    for eid, edge in edges.items():
        src = src_edges[eid]
        if _edge_state(edge) != _edge_state(src):
            edge["changed"] = True
    for nid, node in nodes.items():
        if _node_state(node) != _node_state(src_nodes[nid]):
            node["changed"] = True
    return result


def _edge_state(edge: dict) -> tuple:
    return (edge["lanes"], round(float(edge["speed"]), 2), bool(edge.get("closed")))


def _node_state(node: dict) -> str:
    """Режим управления узлом: "off" (нерегулируемый) или тип программы светофора."""
    if node["type"].startswith("traffic_light"):
        return node.get("tl_type") or "static"
    return "off"


# --------------------------------------------------------------------------
# Компиляция итогового состояния в патчи netconvert
# --------------------------------------------------------------------------

def diff_states(base: dict, final: dict) -> dict:
    """Отличия итогового графа от базового: что именно нужно поменять в SUMO."""
    b_edges, b_nodes = _index(base)
    f_edges, f_nodes = _index(final)
    edges: dict[str, dict] = {}
    for eid, fe in f_edges.items():
        be = b_edges[eid]
        d = {}
        if fe["lanes"] != be["lanes"]:
            d["lanes"] = fe["lanes"]
        if abs(fe["speed"] - be["speed"]) > 1e-6:
            d["speed"] = fe["speed"]
        if bool(fe.get("closed")) != bool(be.get("closed")):
            d["closed"] = bool(fe.get("closed"))
        if d:
            edges[eid] = d
    nodes: dict[str, str] = {}
    for nid, fn in f_nodes.items():
        if _node_state(fn) != _node_state(b_nodes[nid]):
            nodes[nid] = _node_state(fn)
    return {"edges": edges, "nodes": nodes}


# Классы транспорта, которым открыта дорога. Совпадает с typemaps/urban.typ.xml.
OPEN_PERMISSIONS = ("passenger bus truck trailer delivery evehicle taxi motorcycle emergency "
                    "authority army vip coach private hov custom1 custom2")


def to_netconvert_patches(diff: dict) -> tuple[str | None, str | None]:
    """Возвращает тексты файлов патча (*.edg.xml, *.nod.xml) или None, если менять нечего.

    Решения:
    * close — ребро не удаляется, а запрещается для всех классов транспорта
      (disallow="all"). Геометрия, нумерация рёбер и программы светофоров
      сохраняются — как при реальном перекрытии улицы ограждением.
    * open — восстанавливаются разрешения по умолчанию.
    * lanes/speed — netconvert сам пересчитает связи полос на перекрёстках
      и программы затронутых светофоров.
    * traffic_light off — узел становится нерегулируемым (priority).
    """
    edge_lines = []
    for eid, d in diff["edges"].items():
        attrs = {}
        if "lanes" in d:
            attrs["numLanes"] = str(d["lanes"])
        if "speed" in d:
            attrs["speed"] = f"{d['speed']:.2f}"
        if d.get("closed") is True:
            attrs["disallow"] = "all"
        elif d.get("closed") is False:
            attrs["allow"] = OPEN_PERMISSIONS
        attr_s = " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items())
        edge_lines.append(f"    <edge id={quoteattr(eid)} {attr_s}/>")
    node_lines = []
    for nid, tl in diff["nodes"].items():
        if tl == "off":
            node_lines.append(f'    <node id={quoteattr(nid)} type="priority"/>')
        else:
            node_lines.append(f'    <node id={quoteattr(nid)} type="traffic_light" tlType={quoteattr(tl)}/>')
    edg = "<edges>\n" + "\n".join(edge_lines) + "\n</edges>\n" if edge_lines else None
    nod = "<nodes>\n" + "\n".join(node_lines) + "\n</nodes>\n" if node_lines else None
    return edg, nod
