"""Шаг 4. Сборка датасета в Parquet, сценарии считаются параллельно на всех ядрах.

    python -m sim.build_dataset --n-scenarios 300            # все ядра
    python -m sim.build_dataset --n-scenarios 1000 --resume  # дорастить датасет
    python -m sim.build_dataset --n-scenarios 8 --workers 2 --out /tmp/ds   # быстрый пробный прогон

Результат (sim/data/dataset/):
    edges.parquet      — по строке на (сценарий, ребро): состояние ребра после изменений
                         и метрики SUMO (целевые переменные для ml/);
    nodes.parquet      — по строке на (сценарий, узел): режим управления перекрёстком;
    scenarios.parquet  — по строке на сценарий: список изменений (JSON), сводка
                         прогона, время счёта;
    meta.json          — параметры генерации, версии, описание колонок.
Сценарий 0 — базовая сеть без изменений (точка отсчёта «до»).

Параллелизм: multiprocessing.Pool, один процесс = один сценарий целиком
(netconvert + SUMO). SUMO однопоточен (--threads 1), поэтому распараллеливание
по сценариям даёт почти линейное ускорение. Каждый сценарий сразу пишется
в parts/ — прерванную сборку можно продолжить флагом --resume.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sim import config
from sim.scenarios import (ChangeSampler, ScenarioSpec, edge_adjacency, load_baseline_flows,
                           load_network, make_specs, run_scenario)
from sim.sumo_tools import sumo_version

EDGE_DTYPES = {
    "scenario_id": "int32", "lanes": "int8", "speed_limit": "float32",
    "occupancy": "float32", "mean_speed": "float32", "travel_time": "float32",
    "density": "float32", "waiting_time": "float32", "time_loss": "float32",
    "entered": "int32", "left": "int32", "sampled_seconds": "float32",
}

COLUMNS_DOC = {
    "edges.parquet": {
        "scenario_id": "номер сценария (0 — базовая сеть)",
        "edge_id": "id ребра (как в network.json)",
        "lanes": "число полос после изменений",
        "speed_limit": "разрешённая скорость после изменений, м/с",
        "closed": "ребро закрыто для движения",
        "changed": "состояние ребра отличается от базового",
        "occupancy": "ЦЕЛЬ: доля занятой длины ребра, 0..1 (для закрытых — 0)",
        "mean_speed": "ЦЕЛЬ: средняя скорость машин, м/с (для закрытых — NaN)",
        "travel_time": "ЦЕЛЬ: среднее время проезда ребра, с (для закрытых — NaN)",
        "density": "плотность, авт/км",
        "waiting_time": "суммарное время стоянки машин на ребре, с",
        "time_loss": "суммарные потери времени относительно свободного движения, с",
        "entered": "число въехавших на ребро машин",
        "left": "число съехавших с ребра машин",
        "sampled_seconds": "машино-секунды на ребре (объём работы ребра)",
    },
    "nodes.parquet": {
        "scenario_id": "номер сценария",
        "node_id": "id узла (как в network.json)",
        "type": "тип узла SUMO после изменений (traffic_light, priority, ...)",
        "tl_type": "тип программы светофора: static / actuated / null",
        "changed": "режим управления узлом отличается от базового",
    },
}

# --- рабочий процесс -----------------------------------------------------------
_STATE: dict = {}


def _init_worker(network: dict, base_net: str, trips: str, parts_dir: str, work_root: str | None) -> None:
    _STATE.update(network=network, base_net=Path(base_net), trips=Path(trips),
                  parts_dir=Path(parts_dir), work_root=Path(work_root) if work_root else None)


def _part_paths(parts_dir: Path, sid: int) -> tuple[Path, Path, Path]:
    return (parts_dir / f"edges_{sid:05d}.parquet", parts_dir / f"nodes_{sid:05d}.parquet",
            parts_dir / f"summary_{sid:05d}.json")


def _worker(spec: ScenarioSpec) -> dict:
    edges_p, nodes_p, summary_p = _part_paths(_STATE["parts_dir"], spec.scenario_id)
    try:
        edge_rows, node_rows, summary = run_scenario(
            spec, _STATE["network"], base_net=_STATE["base_net"], trips=_STATE["trips"],
            work_root=_STATE["work_root"])
        pd.DataFrame(edge_rows).astype(EDGE_DTYPES).to_parquet(edges_p, index=False)
        pd.DataFrame(node_rows).astype({"scenario_id": "int32"}).to_parquet(nodes_p, index=False)
        summary.update(status="ok", error=None, worker_pid=os.getpid())
    except Exception as exc:  # сценарий не должен ронять всю сборку
        summary = {"scenario_id": spec.scenario_id, "seed": spec.seed, "n_changes": len(spec.changes),
                   "changes": json.dumps(spec.changes, ensure_ascii=False), "status": "error",
                   "error": f"{type(exc).__name__}: {exc}"[-2000:], "worker_pid": os.getpid()}
        traceback.print_exc()
    summary_p.write_text(json.dumps(summary, ensure_ascii=False, default=float), encoding="utf-8")
    return summary


# --- сборка --------------------------------------------------------------------

def merge_parts(parts_dir: Path, out_dir: Path, scenario_ids: list[int]) -> dict:
    summaries = []
    edge_frames, node_frames = [], []
    for sid in sorted(scenario_ids):
        edges_p, nodes_p, summary_p = _part_paths(parts_dir, sid)
        if not summary_p.exists():
            continue
        s = json.loads(summary_p.read_text(encoding="utf-8"))
        summaries.append(s)
        if s.get("status") == "ok":
            edge_frames.append(pd.read_parquet(edges_p))
            node_frames.append(pd.read_parquet(nodes_p))
    scen = pd.DataFrame(summaries).sort_values("scenario_id").reset_index(drop=True)
    edges = pd.concat(edge_frames, ignore_index=True)
    nodes = pd.concat(node_frames, ignore_index=True)
    edges.to_parquet(out_dir / "edges.parquet", index=False, compression="zstd")
    nodes.to_parquet(out_dir / "nodes.parquet", index=False, compression="zstd")
    scen.to_parquet(out_dir / "scenarios.parquet", index=False, compression="zstd")
    return {"scenarios": len(scen), "ok": int((scen["status"] == "ok").sum()),
            "edge_rows": len(edges), "node_rows": len(nodes)}


def build_dataset(n_scenarios: int, workers: int, seed: int, out_dir: Path, resume: bool = False) -> dict:
    for path, hint in ((config.NETWORK_JSON, "sim.build_network"), (config.TRIPS_FILE, "sim.generate_traffic")):
        if not path.exists():
            raise SystemExit(f"Нет {path} — сначала запустите python -m {hint}")
    network = load_network()
    flows = load_baseline_flows()
    if not flows:
        print("[build_dataset] нет базового прогона — выбор рёбер будет равновероятным")
    sampler = ChangeSampler(network, edge_adjacency(), flows)
    specs = make_specs(sampler, n_scenarios, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    if parts_dir.exists() and not resume:
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(exist_ok=True)
    done = {s.scenario_id for s in specs if _part_paths(parts_dir, s.scenario_id)[2].exists()} if resume else set()
    todo = [s for s in specs if s.scenario_id not in done]
    work_root = Path(os.environ.get("SIM_WORK_DIR", "")) if os.environ.get("SIM_WORK_DIR") else None

    print(f"[build_dataset] сценариев: {len(specs)} (уже готово {len(done)}), процессов: {workers}")
    t0 = time.perf_counter()
    errors = 0
    initargs = (network, str(config.BASE_NET), str(config.TRIPS_FILE), str(parts_dir),
                str(work_root) if work_root else None)
    # Самые долгие (с большим числом изменений) — первыми: равномернее загрузка ядер в конце.
    todo.sort(key=lambda s: -len(s.changes))
    with mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=initargs) as pool:
        for i, summary in enumerate(pool.imap_unordered(_worker, todo, chunksize=1), 1):
            errors += summary.get("status") != "ok"
            if i % max(1, len(todo) // 20) == 0 or i == len(todo):
                elapsed = time.perf_counter() - t0
                eta = elapsed / i * (len(todo) - i)
                print(f"[build_dataset] {i}/{len(todo)}  прошло {elapsed / 60:.1f} мин, "
                      f"осталось ≈{eta / 60:.1f} мин, ошибок {errors}", flush=True)
    wall = time.perf_counter() - t0

    counts = merge_parts(parts_dir, out_dir, [s.scenario_id for s in specs])
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_scenarios_requested": n_scenarios, "master_seed": seed, **counts,
        "workers": workers, "cpu_count": os.cpu_count(),
        "wall_seconds_this_run": round(wall, 1), "scenarios_this_run": len(todo),
        "sumo_version": sumo_version(),
        "network": network["meta"],
        "simulation": {"begin": config.SIM_BEGIN, "end": config.SIM_END, "warmup": config.WARMUP,
                       "insertion_rate_veh_h": config.INSERTION_RATE, "sumo_seed": config.SUMO_SEED,
                       "time_to_teleport": config.TIME_TO_TELEPORT,
                       "rerouting_period": config.REROUTING_PERIOD},
        "generator": {"min_changes": config.MIN_CHANGES, "max_changes": config.MAX_CHANGES,
                      "type_weights": config.CHANGE_TYPE_WEIGHTS,
                      "lanes_range": [config.MIN_LANES, config.MAX_LANES],
                      "speed_choices_kmh": list(config.SPEED_CHOICES_KMH)},
        "columns": COLUMNS_DOC,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build_dataset] готово за {wall / 60:.1f} мин: {counts} → {out_dir}")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-scenarios", type=int, default=300, help="число сценариев с изменениями")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="процессов (по умолчанию все ядра)")
    parser.add_argument("--seed", type=int, default=2026, help="главное зерно генератора сценариев")
    parser.add_argument("--out", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--resume", action="store_true", help="не пересчитывать готовые сценарии из parts/")
    args = parser.parse_args(argv)
    build_dataset(args.n_scenarios, args.workers, args.seed, args.out, args.resume)


if __name__ == "__main__":
    main()
