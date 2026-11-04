"""Шум симуляции: базовая сеть с разными зёрнами SUMO.

    python -m sim.noise_floor --seeds 8

Микросимуляция стохастична (моменты смены полос, «нерешительность» водителей
и т.п.). Даже без изменений сети метрики ребра немного «гуляют» от зерна к
зерну. Разброс между прогонами с разными зёрнами — нижняя граница ошибки,
которой в принципе может достичь любая модель-предсказатель (ml/): точнее,
чем симулятор согласуется сам с собой, предсказывать бессмысленно.

Результат: sim/data/dataset/seed_noise.parquet (seed, edge_id, метрики).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from sim import config
from sim.simulate import run_sumo


def _run(seed: int) -> list[dict]:
    work = Path(tempfile.mkdtemp(prefix=f"noise{seed}_"))
    try:
        res = run_sumo(config.BASE_NET, config.TRIPS_FILE, work, seed=seed)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return [{"seed": seed, "edge_id": eid, **{k: m[k] for k in ("occupancy", "mean_speed", "travel_time", "entered")}}
            for eid, m in res["edges"].items()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=config.DATASET_DIR / "seed_noise.parquet")
    args = parser.parse_args(argv)
    seeds = [1000 + i for i in range(args.seeds)]
    with mp.get_context("fork").Pool(min(args.workers, len(seeds))) as pool:
        rows = [r for part in pool.map(_run, seeds) for r in part]
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    std = df.groupby("edge_id")[["occupancy", "mean_speed", "travel_time"]].std().mean()
    print(f"[noise_floor] {len(seeds)} зёрен → {args.out}; средний по рёбрам σ: "
          + ", ".join(f"{k}={v:.4f}" for k, v in std.items()))


if __name__ == "__main__":
    main()
