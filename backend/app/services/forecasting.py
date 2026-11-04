"""Расчёт прогноза для сценария, сводные метрики и сравнение «до/после».

Порядок расчёта (POST /scenarios/{id}/forecast):
  1. берётся исходный граф и список изменений сценария (в порядке seq);
  2. вызывается ml/predict.py: predict(network, changes) -> {edge_id: прогноз};
     GNN обучена на данных SUMO с обычными (жёсткими) светофорами;
  3. если выбран режим «ИИ-агенты» (rl) или «адаптивные» (actuated), к
     прогнозу применяется поправка по результатам agents/evaluate.py:
     для участков перед регулируемыми перекрёстками «лишнее» время
     (сверх времени свободного проезда) умножается на отношение средних
     задержек на этом перекрёстке k = W_mode / W_fixed. Это позволяет
     показать эффект агентов на тепловой карте, не переобучая GNN под
     каждый режим (в перспективе режим можно подать в GNN как признак);
  4. результаты по каждому участку пишутся в forecast_results, сводка — в forecasts.summary.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from ..models import Forecast, ForecastResult, NNModel, Scenario
from ..stubs import predict_stub
from .agents import evaluate_modes
from .changes import apply_changes
from .integrations import integrations
from .network import store

CONGESTED_OCCUPANCY = 0.3
CONGESTED_SPEED_RATIO = 0.5
SUMMARY_KEYS = ("mean_speed", "mean_trip_time", "mean_edge_travel_time", "total_travel_time",
                "mean_occupancy", "congested_share")


# ------------------------------------------------------------------ нормализация ответа модели
def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def normalize_results(net: dict, raw: dict) -> tuple[dict[str, dict], list[str]]:
    """Приводит ответ predict() к {edge_id: {occupancy, mean_speed, travel_time, closed}}."""
    warnings: list[str] = []
    raw = {str(k): v for k, v in (raw or {}).items()}
    occ_values = [_num(v.get("occupancy")) for v in raw.values() if isinstance(v, dict)]
    percent = bool(occ_values) and max((o or 0) for o in occ_values) > 1.5
    if percent:
        warnings.append("occupancy от модели похоже в процентах — переведено в доли (0..1)")
    out: dict[str, dict] = {}
    missing = 0
    for e in net["edges"]:
        eid = e["id"]
        if e.get("closed"):
            out[eid] = {"occupancy": 0.0, "mean_speed": 0.0, "travel_time": None, "closed": True}
            continue
        r = raw.get(eid)
        if not isinstance(r, dict):
            missing += 1
            continue
        occ = _num(r.get("occupancy")) or 0.0
        occ = occ / 100.0 if percent else occ
        speed = _num(r.get("mean_speed"))
        tt = _num(r.get("travel_time"))
        if tt is None or tt <= 0:
            tt = e["length"] / max(speed or e["speed"], 0.1)
        if speed is None or speed <= 0:
            speed = e["length"] / tt
        out[eid] = {"occupancy": min(max(occ, 0.0), 1.0), "mean_speed": max(speed, 0.0),
                    "travel_time": tt, "closed": False}
    if missing:
        warnings.append(f"Модель не вернула прогноз для {missing} участков")
    return out, warnings


# ------------------------------------------------------------------ поправка на режим светофоров
def apply_signal_correction(net: dict, results: dict[str, dict], fixed: dict, mode_metrics: dict) -> dict[str, float]:
    """Корректирует прогноз для участков перед светофорами. -> {node_id: k}."""

    def ratio(a, b) -> float | None:
        a, b = _num(a), _num(b)
        return a / b if a is not None and b and b > 0 else None

    per_f = fixed.get("per_tls") or {}
    per_m = mode_metrics.get("per_tls") or {}
    k_global = ratio(mode_metrics.get("mean_waiting_time"), fixed.get("mean_waiting_time")) or \
        ratio(mode_metrics.get("mean_travel_time"), fixed.get("mean_travel_time")) or 1.0
    nodes = {n["id"]: n for n in net["nodes"]}
    factors: dict[str, float] = {}
    for e in net["edges"]:
        r = results.get(e["id"])
        if r is None or r["closed"] or nodes.get(e["to"], {}).get("type") != "traffic_light":
            continue
        node = e["to"]
        if node not in factors:
            k = ratio((per_m.get(node) or {}).get("mean_waiting_time"), (per_f.get(node) or {}).get("mean_waiting_time"))
            factors[node] = float(np.clip(k if k is not None else k_global, 0.2, 3.0))
        k = factors[node]
        free = e["length"] / max(e["speed"], 0.1)
        tt = r["travel_time"]
        new_tt = free + max(tt - free, 0.0) * k
        if new_tt <= 0 or tt <= 0:
            continue
        r["occupancy"] = float(min(1.0, r["occupancy"] * new_tt / tt))
        r["travel_time"] = new_tt
        r["mean_speed"] = e["length"] / new_tt
    return factors


# ------------------------------------------------------------------ сводные метрики
def compute_summary(net: dict, results: dict[str, dict]) -> dict:
    edges = {e["id"]: e for e in net["edges"]}
    open_ids = [eid for eid, r in results.items() if not r["closed"]]
    lengths = np.array([edges[i]["length"] for i in open_ids])
    tt = np.array([results[i]["travel_time"] for i in open_ids])
    occ = np.array([results[i]["occupancy"] for i in open_ids])
    speed = np.array([results[i]["mean_speed"] for i in open_ids])
    limit = np.array([edges[i]["speed"] for i in open_ids])
    if not len(open_ids):
        return {"edges_count": 0, "closed_count": len(results)}
    congested = (occ >= CONGESTED_OCCUPANCY) | (speed < CONGESTED_SPEED_RATIO * limit)

    # Среднее время в пути: кратчайшие по прогнозному времени пути между фиксированными парами узлов
    trips = store.trip_sample()
    graph = store.edge_costs({i: results[i]["travel_time"] for i in open_ids})
    trip_times, unreachable = [], 0
    if len(trips.origins):
        dist = dijkstra(graph, directed=True, indices=trips.origins)
        for k, dests in enumerate(trips.destinations):
            if len(dests):
                d = dist[k, dests]
                trip_times.extend(d[np.isfinite(d)].tolist())
                unreachable += int((~np.isfinite(d)).sum())

    return {
        "mean_speed": float(lengths.sum() / tt.sum()),  # м/с, средняя по сети (взвешенная по длине)
        "mean_trip_time": float(np.mean(trip_times)) if trip_times else None,  # с
        "trips_count": len(trip_times),
        "unreachable_trips": unreachable,
        "mean_edge_travel_time": float(tt.mean()),  # с
        "total_travel_time": float(tt.sum()),  # с, сумма по всем участкам
        "mean_occupancy": float((occ * lengths).sum() / lengths.sum()),
        "congested_share": float(lengths[congested].sum() / lengths.sum()),
        "edges_count": len(open_ids),
        "closed_count": len(results) - len(open_ids),
    }


# ------------------------------------------------------------------ основной сценарий
def get_or_create_model(session: Session, use_stub: bool = False) -> NNModel:
    """Запись о модели в nn_models (создаётся при первом использовании; последняя — активная)."""
    info = dict(predict_stub.MODEL_INFO) if use_stub else integrations.model_info()
    model = session.scalars(
        select(NNModel).where(NNModel.name == info["name"], NNModel.version == str(info["version"]))
    ).first()
    if model is None:
        model = NNModel(name=info["name"], version=str(info["version"]), architecture=info.get("architecture"),
                        weights_path=info.get("weights_path"), metrics=info.get("metrics"),
                        is_stub=bool(info.get("is_stub")), is_active=True)
        for other in session.scalars(select(NNModel).where(NNModel.is_active.is_(True))):
            other.is_active = False
        session.add(model)
        session.flush()
        session.commit()
    return model


def run_forecast(session: Session, scenario: Scenario, signal_mode: str = "fixed") -> Forecast:
    base = store.load()
    changes = [c.to_contract() for c in scenario.changes]
    net = apply_changes(base, changes)
    t0 = time.perf_counter()
    raw, predictor, warnings = integrations.predict(base, changes)
    results, w = normalize_results(net, raw)
    warnings += w

    agent_metrics = None
    if signal_mode != "fixed":
        metrics, w, _ = evaluate_modes(session, scenario, ["fixed", signal_mode])
        warnings += w
        factors = apply_signal_correction(net, results, metrics["fixed"], metrics[signal_mode])
        agent_metrics = {
            "fixed": {k: v for k, v in metrics["fixed"].items() if k != "per_tls"},
            signal_mode: {k: v for k, v in metrics[signal_mode].items() if k != "per_tls"},
            "tls_factors": factors,
        }

    summary = compute_summary(net, results)
    summary["warnings"] = warnings
    model = get_or_create_model(session, use_stub=(predictor == "stub"))
    forecast = Forecast(
        scenario_id=scenario.id, model_id=model.id if model else None, signal_mode=signal_mode, status="done",
        changes_snapshot=changes, summary=summary, agent_metrics=agent_metrics, network_hash=store.hash,
        predictor=predictor, duration_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    session.add(forecast)
    session.flush()
    rows = [{"forecast_id": forecast.id, "edge_id": eid, "occupancy": r["occupancy"], "mean_speed": r["mean_speed"],
             "travel_time": r["travel_time"], "is_closed": r["closed"]} for eid, r in results.items()]
    for i in range(0, len(rows), 2000):
        session.execute(insert(ForecastResult), rows[i:i + 2000])
    session.commit()
    return forecast


def results_dict(session: Session, forecast_id: int) -> dict[str, dict]:
    rows = session.execute(
        select(ForecastResult.edge_id, ForecastResult.occupancy, ForecastResult.mean_speed,
               ForecastResult.travel_time, ForecastResult.is_closed).where(ForecastResult.forecast_id == forecast_id)
    )
    return {r.edge_id: {"occupancy": r.occupancy, "mean_speed": r.mean_speed, "travel_time": r.travel_time,
                        "closed": r.is_closed} for r in rows}


# ------------------------------------------------------------------ сравнение
def compare(session: Session, fa: Forecast, fb: Forecast, top: int = 10) -> dict:
    ra, rb = results_dict(session, fa.id), results_dict(session, fb.id)
    edges = {}
    for eid in sorted(set(ra) | set(rb)):
        a, b = ra.get(eid), rb.get(eid)
        item = {"a": a, "b": b}
        if a and b:
            item["d_occupancy"] = b["occupancy"] - a["occupancy"]
            item["d_mean_speed"] = b["mean_speed"] - a["mean_speed"]
            if a["travel_time"] is not None and b["travel_time"] is not None:
                item["d_travel_time"] = b["travel_time"] - a["travel_time"]
                item["rel_travel_time"] = b["travel_time"] / a["travel_time"] - 1 if a["travel_time"] else None
        edges[eid] = item

    sa, sb = fa.summary or {}, fb.summary or {}
    delta = {}
    for key in SUMMARY_KEYS:
        va, vb = _num(sa.get(key)), _num(sb.get(key))
        if va is not None and vb is not None:
            delta[key] = {"a": va, "b": vb, "abs": vb - va, "pct": (vb - va) / va * 100 if va else None}
    ranked = [(v["d_travel_time"], eid) for eid, v in edges.items() if v.get("d_travel_time") is not None]
    ranked.sort()
    return {
        "a": {"forecast_id": fa.id, "scenario_id": fa.scenario_id, "signal_mode": fa.signal_mode, "summary": sa},
        "b": {"forecast_id": fb.id, "scenario_id": fb.scenario_id, "signal_mode": fb.signal_mode, "summary": sb},
        "delta": delta,
        "edges": edges,
        "top_improved": [{"edge_id": e, "d_travel_time": d} for d, e in ranked[:top] if d < 0],
        "top_worsened": [{"edge_id": e, "d_travel_time": d} for d, e in reversed(ranked[-top:]) if d > 0],
    }
