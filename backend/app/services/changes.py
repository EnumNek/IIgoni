"""Проверка и применение изменений сети.

Формат изменения (контракт): {type: "lanes"|"speed"|"close"|"open"|"traffic_light", target_id, value}

    lanes          target_id = id участка, value = число полос (1..8)
    speed          target_id = id участка, value = разрешённая скорость, м/с (1..40)
    close          target_id = id участка, value не используется (null)
    open           target_id = id участка, value не используется — отменяет close
    traffic_light  target_id = id узла,    value = true (поставить светофор) / false (убрать)

Изменения применяются строго по порядку (поле seq в БД), поэтому
«закрыть -> открыть» или «2 полосы -> 3 полосы» дают ожидаемый итог.
"""
from __future__ import annotations

from typing import Any

from ..models import CHANGE_TYPES

EDGE_CHANGES = {"lanes", "speed", "close", "open"}
LANES_RANGE = (1, 8)
SPEED_RANGE = (1.0, 40.0)  # м/с: 3,6 … 144 км/ч


class ChangeError(ValueError):
    """Некорректное изменение (неизвестный тип, нет такого участка/узла, значение вне диапазона)."""


def _as_bool(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "add", "on"}:
            return True
        if v in {"false", "0", "no", "remove", "off"}:
            return False
    raise ChangeError(f"Для traffic_light ожидается true/false, получено {value!r}")


def normalize_change(change: dict, edge_ids, node_ids) -> dict:
    """Проверяет изменение и приводит value к каноническому типу."""
    ctype = str(change.get("type", "")).strip()
    target = change.get("target_id")
    value = change.get("value")
    if ctype not in CHANGE_TYPES:
        raise ChangeError(f"Неизвестный тип изменения {ctype!r}; допустимо: {', '.join(CHANGE_TYPES)}")
    if target is None or str(target) == "":
        raise ChangeError("Не указан target_id")
    target = str(target)

    if ctype in EDGE_CHANGES:
        if target not in edge_ids:
            raise ChangeError(f"Участок {target!r} не найден в сети")
    elif target not in node_ids:
        raise ChangeError(f"Узел {target!r} не найден в сети")

    if ctype == "lanes":
        try:
            lanes = int(value)
        except (TypeError, ValueError):
            raise ChangeError("Для lanes нужно целое число полос") from None
        if not LANES_RANGE[0] <= lanes <= LANES_RANGE[1]:
            raise ChangeError(f"Число полос должно быть от {LANES_RANGE[0]} до {LANES_RANGE[1]}")
        value = lanes
    elif ctype == "speed":
        try:
            speed = float(value)
        except (TypeError, ValueError):
            raise ChangeError("Для speed нужно число (м/с)") from None
        if not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
            raise ChangeError(f"Скорость должна быть от {SPEED_RANGE[0]} до {SPEED_RANGE[1]} м/с")
        value = round(speed, 3)
    elif ctype in {"close", "open"}:
        value = None
    else:  # traffic_light
        value = _as_bool(value)
    return {"type": ctype, "target_id": target, "value": value}


def apply_changes(network: dict, changes: list[dict]) -> dict:
    """Возвращает копию сети с применёнными изменениями (исходный dict не меняется).

    Закрытый участок остаётся в списке с признаком "closed": true — так его
    можно показать на карте; для передачи в симулятор см. export_for_simulation().
    """
    net = {
        **{k: v for k, v in network.items() if k not in {"nodes", "edges"}},
        "nodes": [dict(n) for n in network.get("nodes", [])],
        "edges": [dict(e) for e in network.get("edges", [])],
    }
    edges = {e["id"]: e for e in net["edges"]}
    nodes = {n["id"]: n for n in net["nodes"]}
    original_type = {nid: n.get("type", "priority") for nid, n in nodes.items()}

    for ch in changes:
        ctype, target, value = ch["type"], str(ch["target_id"]), ch.get("value")
        if ctype in EDGE_CHANGES:
            edge = edges.get(target)
            if edge is None:
                continue
            if ctype == "lanes":
                edge["lanes"] = int(value)
            elif ctype == "speed":
                edge["speed"] = float(value)
            elif ctype == "close":
                edge["closed"] = True
            elif ctype == "open":
                edge["closed"] = False
        elif ctype == "traffic_light":
            node = nodes.get(target)
            if node is None:
                continue
            if _as_bool(value):
                node["type"] = "traffic_light"
            else:
                was = original_type.get(target, "priority")
                node["type"] = "priority" if was == "traffic_light" else was
    return net


def export_for_simulation(net: dict) -> dict:
    """Сеть для передачи во внешние модули (agents/evaluate.py): закрытые участки удалены."""
    closed = [e["id"] for e in net["edges"] if e.get("closed")]
    edges = [{k: v for k, v in e.items() if k != "closed"} for e in net["edges"] if not e.get("closed")]
    meta = dict(net.get("meta") or {})
    meta["closed_edges"] = closed
    return {**net, "meta": meta, "edges": edges}
