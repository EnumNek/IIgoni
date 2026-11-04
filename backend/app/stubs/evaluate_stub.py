"""Заглушка agents/evaluate.py с тем же интерфейсом.

    evaluate(network_path, mode: "fixed"|"actuated"|"rl") -> метрики

Настоящая часть agents/ прогоняет SUMO с RL-агентами (DQN) на каждом
светофоре. Здесь — аналитическая оценка при одинаковом спросе и маршрутах:
  * fixed    — жёсткое регулирование: цикл 90 с, две фазы, зелёное поровну,
               коэффициент HCM k = 0,5;
  * actuated — адаптивное: цикл и доли зелёного по Вебстеру
               C0 = (1,5L + 5)/(1 − Y), зелёное ∝ критическим нагрузкам фаз, k = 0,3;
  * rl       — УСЛОВНАЯ оценка «идеального» адаптивного управления
               (как actuated, но потерянное время 3 с/фаза и k = 0,15).
               Это не результат обучения RL — цифры нужны только, чтобы
               проверить интерфейс и интерфейс пользователя, до появления agents/.
Все результаты помечаются "is_stub": true.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import traffic_model as tm

MODES = ("fixed", "actuated", "rl")
MODE_PARAMS = {
    "fixed": {"k": 0.5, "lost": tm.LOST_TIME_PER_PHASE, "adaptive": False},
    "actuated": {"k": 0.3, "lost": tm.LOST_TIME_PER_PHASE, "adaptive": True},
    "rl": {"k": 0.15, "lost": 3.0, "adaptive": True},
}


def _load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _phase_groups(bearings: np.ndarray, flows: np.ndarray) -> np.ndarray:
    """0/1 — номер фазы: подходы, соосные с самым загруженным, идут в одну фазу."""
    dom = bearings[int(np.argmax(flows))]
    diff = np.abs(((bearings - dom + 90) % 180) - 90)
    return (diff >= 45).astype(int)


def signal_plan(g: tm.Graph, flow: np.ndarray, mode: str) -> tuple[tm.SignalPlan, dict]:
    params = MODE_PARAMS[mode]
    m = len(g.edge_ids)
    green = np.full(m, tm.fixed_signal_ratio())
    cycle = np.full(m, tm.FIXED_CYCLE)
    plans: dict[str, dict] = {}
    approaches_by_node: dict[int, list[int]] = {}
    for i in np.where(g.tls_end & ~g.closed)[0].tolist():
        approaches_by_node.setdefault(int(g.v[i]), []).append(i)
    for node, idx in approaches_by_node.items():
        idx_arr = np.array(idx)
        phases = _phase_groups(g.end_bearing[idx_arr], flow[idx_arr] + 1e-6)
        n_ph = len(set(phases.tolist()))
        lost = params["lost"] * max(n_ph, 2)
        if params["adaptive"]:
            y = flow[idx_arr] / (tm.SAT_FLOW * g.lanes[idx_arr])
            y_p = np.array([max(y[phases == p].max(), 0.05) if (phases == p).any() else 0.0 for p in (0, 1)])
            big_y = float(y_p.sum())
            c = float(np.clip((1.5 * lost + 5) / (1 - big_y), 40, 120)) if big_y < 0.9 else 120.0
            g_p = (c - lost) * y_p / y_p.sum()
            for k, a in enumerate(idx):
                green[a] = g_p[phases[k]] / c
                cycle[a] = c
        else:
            c = tm.FIXED_CYCLE
            if n_ph == 1:  # все подходы соосны — вторая фаза нужна только пешеходам
                green[idx_arr] = (c - lost) * 0.6 / c
        plans[g.node_ids[node]] = {"cycle": round(float(cycle[idx[0]]), 1), "approaches": idx}
    return tm.SignalPlan(green_ratio=green, cycle=cycle, k=params["k"]), plans


def _demand_for(net: dict) -> tm.Demand:
    """Спрос берётся с базовой сети (meta.base_network), чтобы все сценарии сравнивались при равном спросе."""
    base_path = (net.get("meta") or {}).get("base_network")
    if base_path and Path(base_path).exists():
        base = _load(base_path)
        if [n["id"] for n in base["nodes"]] == [n["id"] for n in net["nodes"]]:
            return tm.calibrated_demand(base)
    return tm.calibrated_demand(net)


def evaluate(network_path, mode: str = "fixed") -> dict:
    if mode not in MODES:
        raise ValueError(f"mode должен быть одним из {MODES}")
    net = _load(network_path)
    g = tm.build_graph(net)
    demand = _demand_for(net)
    base = tm.assign(g, demand)  # маршруты/потоки при жёстком регулировании
    flow = base.flow
    plan, plans = signal_plan(g, flow, mode)
    t_run, d_sig = tm.link_costs(g, flow, plan)
    tt = t_run + d_sig

    served = max(base.served, 1e-6)
    veh_time = float((flow * tt).sum())  # авт·с/ч
    tls_idx = np.where(g.tls_end & ~g.closed)[0]
    tls_flow = flow[tls_idx]
    per_tls = {}
    for node_id, info in plans.items():
        a = np.array(info["approaches"])
        q = flow[a]
        wait = float((q * d_sig[a]).sum() / max(q.sum(), 1e-6))
        sat = float((q / np.maximum(tm.SAT_FLOW * g.lanes[a] * plan.green_ratio[a], 1)).max()) if len(a) else 0.0
        per_tls[node_id] = {
            "mean_waiting_time": round(wait, 2),
            "cycle": info["cycle"],
            "flow": round(float(q.sum()), 1),
            "max_saturation": round(sat, 3),
        }
    return {
        "mode": mode,
        "mean_waiting_time": round(float((tls_flow * d_sig[tls_idx]).sum() / max(tls_flow.sum(), 1e-6)), 2),
        "mean_travel_time": round(veh_time / served, 2),
        "mean_speed": round(float((flow * g.length).sum()) / max(veh_time, 1e-6), 3),
        "throughput": round(float(base.served), 1),
        "total_delay_veh_h": round(float((flow * d_sig).sum()) / 3600.0, 2),
        "mean_queue_length": round(float((tls_flow / 3600.0 * d_sig[tls_idx]).mean()) if len(tls_idx) else 0.0, 2),
        "tls_count": len(per_tls),
        "per_tls": per_tls,
        "is_stub": True,
        "evaluator": "stub-hcm",
    }
