"""Заглушка ml/predict.py с тем же интерфейсом.

    predict(network: dict, changes: list) -> dict[edge_id, {occupancy, mean_speed, travel_time}]

Вместо графовой нейросети используется аналитическая модель (traffic_model.py):
равновесное распределение синтетического спроса по сети с функцией BPR и
задержками на светофорах по HCM. Модель детерминирована и считает весь район
(~900 участков) примерно за секунду, поэтому подходит для отладки
веб-приложения и как «наивный» эталон (baseline) при сравнении с GNN.
"""
from __future__ import annotations

import numpy as np

from ..services.changes import apply_changes
from . import traffic_model as tm

MODEL_INFO = {
    "name": "stub-bpr-assignment",
    "version": "1.0",
    "architecture": "Аналитическая модель: MSA-равновесие + BPR + HCM (заглушка вместо GNN)",
    "is_stub": True,
}


def run_model(network: dict, changes: list) -> tuple[tm.Graph, tm.Assignment]:
    demand = tm.calibrated_demand(network)
    changed = apply_changes(network, changes or [])
    g = tm.build_graph(changed)
    return g, tm.assign(g, demand)


def predict(network: dict, changes: list) -> dict[str, dict]:
    g, res = run_model(network, changes)
    tt = res.travel_time
    occ = tm.occupancy(g, res.flow, tt)
    speed = g.length / np.maximum(tt, 1e-6)
    out: dict[str, dict] = {}
    for i, eid in enumerate(g.edge_ids):
        if g.closed[i]:
            out[eid] = {"occupancy": 0.0, "mean_speed": 0.0, "travel_time": None, "closed": True}
        else:
            out[eid] = {
                "occupancy": round(float(occ[i]), 4),
                "mean_speed": round(float(speed[i]), 3),
                "travel_time": round(float(tt[i]), 2),
                "flow": round(float(res.flow[i]), 1),  # авт/ч — дополнительное поле, вне контракта
            }
    return out
