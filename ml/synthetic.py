"""Синтетический датасет в формате sim/data/*.parquet (заглушка части sim/).

Формат (его же ожидает загрузчик ml/dataset.py от части sim/):

  scenarios.parquet — один сценарий на строку
      scenario_id : int64
      changes     : string — JSON-список изменений по контракту
                    [{"type": "close", "target_id": "...", "value": null}, ...];
                    пустой список "[]" — базовый прогон без изменений
      seed        : int64   — seed прогона (в SUMO — seed маршрутов)
      n_changes   : int64

  edge_metrics.parquet — метрики каждого ребра в каждом сценарии
      scenario_id : int64
      edge_id     : string  — id ребра из network.json
      occupancy   : float   — доля 0..1
      mean_speed  : float   — м/с
      travel_time : float   — с
      volume      : float   — авт/ч (необязательная колонка)
  Закрытые в сценарии рёбра в edge_metrics отсутствуют (как и в edgeData SUMO
  для удалённых рёбер).

  network.json — копия сети, на которой сгенерированы данные.

Сценарии генерируются случайно: 1–4 изменения разных типов, «пользователь»
чаще трогает крупные улицы, перекрытие улицы с вероятностью 1/2 закрывает
оба направления, иногда закрытое ребро тут же открывается (проверка
семантики "open").

Запуск:  python -m ml.synthetic [--network auto|osm|grid] [--n 3000] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import SEED, SYNTH_DIR
from ml.macro import MacroModel, MacroParams
from ml.network import State, Topology, apply_changes, parse_network
from ml.stub_network import resolve_network

log = logging.getLogger(__name__)

SPEED_CHOICES = (8.33, 11.11, 13.89, 16.67, 19.44, 22.22)  # 30..80 км/ч
N_CHANGES_P = {1: 0.35, 2: 0.30, 3: 0.20, 4: 0.15}
TYPE_P = {"close": 0.35, "lanes": 0.30, "speed": 0.20, "traffic_light": 0.15}


def _reverse_edges(topo: Topology) -> np.ndarray:
    pair = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(topo.src, topo.dst))}
    return np.array([pair.get((int(b), int(a)), -1) for a, b in zip(topo.src, topo.dst)])


def random_changes(topo: Topology, base: State, rng: np.random.Generator,
                   rev: np.ndarray | None = None) -> list[dict]:
    """Случайный набор изменений, похожий на действия пользователя."""
    rev = _reverse_edges(topo) if rev is None else rev
    E = topo.n_edges
    # Пользователь чаще меняет заметные улицы: вес ~ полосы * sqrt(длины)
    w = base.lanes * np.sqrt(topo.length) * (1 + topo.priority.clip(0) / 10)
    w = w / w.sum()
    in_deg = np.bincount(topo.dst, minlength=topo.n_nodes)
    out_deg = np.bincount(topo.src, minlength=topo.n_nodes)
    junctions = np.nonzero((in_deg >= 2) & (out_deg >= 2) & ~topo.fringe)[0]

    state = base.copy()
    changes: list[dict] = []
    n = rng.choice(list(N_CHANGES_P), p=list(N_CHANGES_P.values()))
    for _ in range(n):
        ctype = rng.choice(list(TYPE_P), p=list(TYPE_P.values()))
        e = int(rng.choice(E, p=w)) if rng.random() < 0.6 else int(rng.integers(E))
        eid = topo.edge_ids[e]
        if ctype == "close":
            if state.closed[e]:
                continue
            changes.append({"type": "close", "target_id": eid, "value": None})
            state.closed[e] = True
            r = rev[e]
            if r >= 0 and not state.closed[r] and rng.random() < 0.5:
                changes.append({"type": "close", "target_id": topo.edge_ids[r], "value": None})
                state.closed[r] = True
            if rng.random() < 0.05:  # передумал — открыл обратно
                changes.append({"type": "open", "target_id": eid, "value": None})
                state.closed[e] = False
        elif ctype == "lanes":
            cur = int(state.lanes[e])
            opts = [v for v in (cur - 1, cur + 1, cur + 2) if 1 <= v <= 4 and v != cur]
            v = int(rng.choice(opts))
            changes.append({"type": "lanes", "target_id": eid, "value": v})
            state.lanes[e] = v
        elif ctype == "speed":
            opts = [s for s in SPEED_CHOICES if abs(s - state.speed[e]) > 0.5]
            v = float(rng.choice(opts))
            changes.append({"type": "speed", "target_id": eid, "value": v})
            state.speed[e] = v
        else:
            if len(junctions) == 0:
                continue
            j = int(rng.choice(junctions))
            v = not bool(state.tl[j])
            changes.append({"type": "traffic_light", "target_id": topo.node_ids[j], "value": v})
            state.tl[j] = v
    return changes


# --- параллельная генерация --------------------------------------------------------
_W: dict = {}


def _init_worker(network: dict, model_seed: int, demand: float, params: MacroParams):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    topo, base = parse_network(network)
    model = MacroModel(topo, base, params, seed=model_seed)
    model.demand = demand
    _W.update(topo=topo, base=base, model=model)


def _run_chunk(tasks: list[tuple[int, list, int]]) -> list[tuple[int, dict]]:
    topo, base, model = _W["topo"], _W["base"], _W["model"]
    out = []
    for sid, changes, seed in tasks:
        state = apply_changes(topo, base, changes)
        res = model.assign(state, model.sample_noise(np.random.default_rng(seed)))
        out.append((sid, {k: res[k] for k in ("occupancy", "mean_speed", "travel_time", "volume")}))
    return out


def generate(network: dict, out_dir: Path = SYNTH_DIR, n_scenarios: int = 3000, n_base: int = 10,
             seed: int = SEED, workers: int | None = None) -> Path:
    t_start = time.time()
    topo, base = parse_network(network)
    params = MacroParams()
    model = MacroModel(topo, base, params, seed=seed)
    demand = model.calibrate()
    log.info("Калибровка спроса: %.0f авт/ч, зон %d", demand, len(model.zones))

    rng = np.random.default_rng(seed)
    rev = _reverse_edges(topo)
    tasks = [(i, [], int(rng.integers(2**31))) for i in range(n_base)]
    for i in range(n_base, n_base + n_scenarios):
        ch = []
        while not ch:
            ch = random_changes(topo, base, rng, rev)
        tasks.append((i, ch, int(rng.integers(2**31))))

    workers = workers or min(4, os.cpu_count() or 1)
    chunk = max(1, len(tasks) // (workers * 8))
    chunks = [tasks[i:i + chunk] for i in range(0, len(tasks), chunk)]
    results: dict[int, dict] = {}
    t_sim = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker,
                                 initargs=(network, seed, demand, params)) as ex:
            for k, part in enumerate(ex.map(_run_chunk, chunks)):
                results.update(part)
                if (k + 1) % max(1, len(chunks) // 10) == 0:
                    log.info("  сценариев: %d / %d", len(results), len(tasks))
    else:
        _init_worker(network, seed, demand, params)
        for part in map(_run_chunk, chunks):
            results.update(part)
    sim_time = time.time() - t_sim

    rows = []
    edge_ids = np.asarray(topo.edge_ids, dtype=object)
    for sid in sorted(results):
        r = results[sid]
        ok = np.isfinite(r["travel_time"])
        rows.append(pd.DataFrame({
            "scenario_id": np.full(ok.sum(), sid, dtype=np.int64),
            "edge_id": edge_ids[ok],
            "occupancy": r["occupancy"][ok].astype(np.float32),
            "mean_speed": r["mean_speed"][ok].astype(np.float32),
            "travel_time": r["travel_time"][ok].astype(np.float32),
            "volume": r["volume"][ok].astype(np.float32),
        }))
    metrics = pd.concat(rows, ignore_index=True)
    scen = pd.DataFrame({
        "scenario_id": np.array([t[0] for t in tasks], dtype=np.int64),
        "changes": [json.dumps(t[1], ensure_ascii=False) for t in tasks],
        "seed": np.array([t[2] for t in tasks], dtype=np.int64),
        "n_changes": np.array([len(t[1]) for t in tasks], dtype=np.int64),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out_dir / "edge_metrics.parquet", index=False)
    scen.to_parquet(out_dir / "scenarios.parquet", index=False)
    with open(out_dir / "network.json", "w", encoding="utf-8") as f:
        json.dump(network, f, ensure_ascii=False)
    meta = {
        "generator": "ml/macro.py (BPR + MSA)", "n_scenarios": n_scenarios, "n_base": n_base,
        "seed": seed, "demand_veh_h": demand, "n_zones": int(len(model.zones)),
        "n_edges": topo.n_edges, "n_nodes": topo.n_nodes,
        "network_source": network.get("meta", {}).get("source", "?"),
        "sim_seconds_total": sim_time, "sim_seconds_per_scenario": sim_time / len(tasks) * workers,
        "workers": workers, "total_seconds": time.time() - t_start,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info("Датасет: %s — %d сценариев, %d строк, %.1f с", out_dir, len(tasks), len(metrics),
             time.time() - t_start)
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="auto", choices=["auto", "osm", "grid"])
    ap.add_argument("--n", type=int, default=3000, help="число сценариев с изменениями")
    ap.add_argument("--n-base", type=int, default=10, help="число базовых прогонов")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=SYNTH_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    network, path = resolve_network(args.network)
    log.info("Сеть: %s", path)
    generate(network, args.out, args.n, args.n_base, args.seed, args.workers)


if __name__ == "__main__":
    main()
