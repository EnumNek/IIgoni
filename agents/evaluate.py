"""Оценка режима управления светофорами — интерфейс по контракту.

    from agents.evaluate import evaluate
    metrics = evaluate("sim/data/network.json", mode="rl")   # или "fixed" | "actuated"

Что делает: прогоняет симуляцию SUMO (1 ч спроса + 10 мин разгрузки) на
нескольких вариантах спроса, которые агенты НЕ видели при обучении, и
усредняет метрики:

* ``avg_travel_time`` — среднее время в пути ТС, с (tripinfo ``duration``;
  для не успевших доехать — время до конца симуляции);
* ``avg_waiting_time`` — среднее суммарное время ожидания ТС (скорость
  < 0.1 м/с), с;
* ``avg_queue_length`` — средняя длина очереди на перекрёсток, авт.:
  среднее по времени (шаг 5 с) и по светофорам числа стоящих ТС на подходах;
* ``avg_time_loss`` — средние потери времени относительно свободного
  движения, с;
* ``avg_depart_delay`` — средняя задержка въезда в сеть (очередь «за
  границей» района), с;
* ``p95_waiting_time`` — 95-й процентиль времени ожидания ТС, с (мера
  «справедливости»: не ждут ли отдельные направления слишком долго);
* ``throughput`` — число завершённых поездок; ``teleports`` — число
  «телепортаций» застрявших ТС (признак затора).

``guard_rate`` — доля решений RL-агентов, скорректированных защитным
правилом максимального ожидания (env.MaxWaitGuard), для fixed/actuated 0;
``per_tls`` — средняя очередь по каждому светофору (для карты),
``runs`` — метрики каждого прогона, ``std`` — стандартные отклонения.

Командная строка::

    python -m agents.evaluate --network grid4x4 --mode rl
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import config  # noqa: E402

import numpy as np  # noqa: E402

from agents.env import MODES, MaxWaitGuard, make_env  # noqa: E402
from agents.networks import Scenario, get_scenario  # noqa: E402

METRIC_KEYS = ("avg_travel_time", "avg_waiting_time", "avg_queue_length", "avg_time_loss",
               "avg_depart_delay", "p95_waiting_time", "throughput", "teleports", "vehicles")
UNITS = {"avg_travel_time": "s", "avg_waiting_time": "s", "avg_queue_length": "veh",
         "p95_waiting_time": "s",
         "avg_time_loss": "s", "avg_depart_delay": "s", "throughput": "veh",
         "teleports": "veh", "vehicles": "veh", "guard_rate": "share"}


def _parse_outputs(tripinfo: Path, statistics: Path) -> dict:
    trips = list(ET.parse(tripinfo).getroot().iter("tripinfo"))
    stats = ET.parse(statistics).getroot()

    def mean(attr: str) -> float:
        return float(np.mean([float(t.get(attr)) for t in trips])) if trips else 0.0

    vehicles = stats.find("vehicles").attrib
    return {
        "avg_travel_time": mean("duration"),
        "avg_waiting_time": mean("waitingTime"),
        "avg_time_loss": mean("timeLoss"),
        "avg_depart_delay": mean("departDelay"),
        "p95_waiting_time": float(np.percentile([float(t.get("waitingTime")) for t in trips], 95))
        if trips else 0.0,
        "throughput": sum(1 for t in trips if float(t.get("arrival")) >= 0),
        "teleports": int(stats.find("teleports").get("total")),
        "vehicles": int(vehicles["loaded"]),
        "not_inserted": int(vehicles.get("waiting", 0)),
        "travel_times": [float(t.get("duration")) for t in trips],
    }


def run_episode(sc: Scenario, route_file: Path, mode: str, seed: int = 0, agents=None,
                keep_series: bool = False, use_gui: bool = False) -> dict:
    """Один прогон симуляции в заданном режиме; возвращает метрики."""
    if mode == "rl" and agents is None:
        from agents.dqn import IndependentDQN

        agents = IndependentDQN.load(sc.models_dir)
    with tempfile.TemporaryDirectory(prefix="agents_eval_") as tmp:
        tripinfo, stats = Path(tmp) / "tripinfo.xml", Path(tmp) / "stats.xml"
        extra = ["--tripinfo-output", str(tripinfo), "--tripinfo-output.write-unfinished", "true",
                 "--statistic-output", str(stats)]
        # RL: среда с теми же параметрами (min_green и т. п.), что при обучении
        env_params = agents.meta.get("env_params") if mode == "rl" else None
        env = make_env(sc, route_file, mode, seed=seed, extra_args=extra, use_gui=use_gui,
                       env_params=env_params)
        t0 = time.time()
        try:
            obs = env.reset()
            guard = MaxWaitGuard(env)
            if mode == "rl":
                agents.check_compatible(env)
            ts_ids = list(env.ts_ids)
            queue_sum = {ts: 0.0 for ts in ts_ids}
            series = []
            steps = 0
            done = False
            while not done:
                actions = guard(agents.act(obs, deterministic=True)) if mode == "rl" else {}
                obs, _, dones, _ = env.step(actions)
                done = dones["__all__"]
                q = {ts: env.traffic_signals[ts].get_total_queued() for ts in ts_ids}
                for ts in ts_ids:
                    queue_sum[ts] += q[ts]
                steps += 1
                if keep_series:
                    series.append((env.sim_step, sum(q.values())))
        finally:
            env.close()  # SUMO дописывает tripinfo/statistics при закрытии
        res = _parse_outputs(tripinfo, stats)
    per_tls = {ts: queue_sum[ts] / max(steps, 1) for ts in ts_ids}
    res.update({
        "avg_queue_length": float(np.mean(list(per_tls.values()))) if per_tls else 0.0,
        "per_tls_queue": per_tls,
        "guard_rate": guard.rate,  # доля решений RL, скорректированных защитным правилом
        "wall_time": time.time() - t0,
    })
    if keep_series:
        res["queue_series"] = series
    return res


def evaluate(network_path: str | Path, mode: str, *, demand: str = "nominal",
             seeds: tuple[int, ...] | list[int] | None = None,
             route_files: list[str | Path] | None = None,
             models_dir: str | Path | None = None) -> dict:
    """Метрики режима управления светофорами на сети (контракт agents/).

    network_path — ``grid2x2``/``grid4x4``/``sim``, путь к .net.xml или к
    network.json (формат контракта). mode — ``fixed`` | ``actuated`` | ``rl``.
    demand — ``low`` | ``nominal`` | ``high``; seeds — семена спроса
    (по умолчанию config.EVAL_SEEDS); route_files — свои файлы маршрутов
    SUMO вместо сгенерированных; models_dir — другая папка моделей
    (по умолчанию agents/models/<сеть>/).
    """
    if mode not in MODES:
        raise ValueError(f"mode должен быть одним из {MODES}, получено: {mode!r}")
    sc = get_scenario(network_path)
    seeds = list(seeds or config.EVAL_SEEDS)
    routes = [Path(r) for r in route_files] if route_files else [sc.eval_route(demand, s) for s in seeds]
    agents = None
    if mode == "rl":
        from agents.dqn import IndependentDQN

        agents = IndependentDQN.load(Path(models_dir) if models_dir else sc.models_dir)

    runs = []
    for i, route in enumerate(routes):
        r = run_episode(sc, route, mode, seed=seeds[i] if i < len(seeds) else i, agents=agents)
        r.pop("travel_times", None)
        r["route_file"] = route.name
        runs.append(r)

    result = {"network": sc.key, "net_file": str(sc.net_file), "mode": mode, "demand": demand,
              "n_runs": len(runs), "units": UNITS}
    for k in METRIC_KEYS:
        vals = [r[k] for r in runs]
        result[k] = float(np.mean(vals))
    result["std"] = {k: float(np.std([r[k] for r in runs])) for k in METRIC_KEYS}
    tls = runs[0]["per_tls_queue"].keys() if runs else []
    result["per_tls"] = {ts: {"avg_queue_length": float(np.mean([r["per_tls_queue"][ts] for r in runs]))}
                         for ts in tls}
    result["guard_rate"] = float(np.mean([r["guard_rate"] for r in runs])) if runs else 0.0
    result["runs"] = [{k: r[k] for k in (*METRIC_KEYS, "guard_rate", "route_file", "wall_time")} for r in runs]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Оценка режима управления светофорами")
    ap.add_argument("--network", default="grid4x4", help="grid2x2 | grid4x4 | sim | путь к сети")
    ap.add_argument("--mode", default="rl", choices=MODES)
    ap.add_argument("--demand", default="nominal", choices=list(config.EVAL_DEMAND_LEVELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--models-dir", default=None, help="папка моделей (по умолчанию agents/models/<сеть>)")
    args = ap.parse_args()
    res = evaluate(args.network, args.mode, demand=args.demand, seeds=args.seeds, models_dir=args.models_dir)
    res.pop("per_tls")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
