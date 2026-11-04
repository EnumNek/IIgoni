"""Замер времени инференса: сеть района, масштабируемость и сравнение с симуляцией.

* district — ``predict()`` целиком на сети, на которой обучена модель, для
  случайных наборов изменений (медиана и 95-й перцентиль), отдельно — только
  прямой проход GNN, и «холодный старт» (загрузка весов + первый вызов).
* scaling — та же модель на решётках n×n растущего размера (модель
  индуктивна, поэтому применима к сети любого размера; качество прогноза для
  чужих сетей здесь не оценивается, измеряется только время).
* reference — для сравнения: время макромодели (генератор синтетики) на один
  сценарий и, если установлен SUMO, время одного прогона SUMO на той же сети
  (1 час модельного времени). Именно такую симуляцию GNN заменяет в рабочем
  режиме.

Запуск:  python -m ml.benchmark [--no-sumo]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ml.config import REPORTS_DIR, STUB_DIR, SYNTH_DIR, WEIGHTS_DIR
from ml.network import parse_network
from ml.predict import Predictor
from ml.stub_network import build_grid_python, sumo_home, sumo_tool
from ml.synthetic import random_changes

log = logging.getLogger(__name__)


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip() + f" (ядер: {os.cpu_count()})"
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _time_network(pred: Predictor, network: dict, n_runs: int, seed: int = 0) -> dict:
    topo, base = parse_network(network)
    rng = np.random.default_rng(seed)
    change_sets = [random_changes(topo, base, rng) for _ in range(n_runs)]
    pred.predict(network, change_sets[0])  # прогрев (подготовка line graph кэшируется)
    t_pred, t_model = [], []
    for ch in change_sets:
        t = time.perf_counter()
        pred.predict(network, ch)
        t_pred.append((time.perf_counter() - t) * 1000)
    _, _, _, fb, _ = pred._prepare(network)
    x = fb.features(base)[None]
    for _ in range(n_runs):
        t = time.perf_counter()
        pred.runner.raw_delta(x, fb.graphs(pred.runner.dilations))
        t_model.append((time.perf_counter() - t) * 1000)
    return {"n_edges": topo.n_edges, "n_nodes": topo.n_nodes, "runs": n_runs,
            "predict_ms_median": float(np.median(t_pred)), "predict_ms_p95": float(np.percentile(t_pred, 95)),
            "model_ms_median": float(np.median(t_model)), "model_ms_p95": float(np.percentile(t_model, 95))}


def time_sumo(net_xml: Path, end: int = 3600, period: float = 1.0, seed: int = 42) -> dict | None:
    """Один прогон SUMO на сети: randomTrips -> duarouter -> sumo (замеряется только sumo)."""
    sumo, home = sumo_tool("sumo"), sumo_home()
    if sumo is None or home is None or not net_xml.exists():
        return None
    rt = home / "tools" / "randomTrips.py"
    if not rt.exists():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        routes = Path(tmp) / "routes.rou.xml"
        env = dict(os.environ, SUMO_HOME=str(home))
        t = time.perf_counter()
        subprocess.run([sys.executable, str(rt), "-n", str(net_xml), "-r", str(routes), "-e", str(end),
                        "-p", str(period), "--fringe-factor", "5", "--seed", str(seed),
                        "--validate", "-o", str(Path(tmp) / "trips.xml")],
                       check=True, capture_output=True, env=env)
        t_routes = time.perf_counter() - t
        n_veh = open(routes, encoding="utf-8").read().count("<vehicle ")
        t = time.perf_counter()
        subprocess.run([sumo, "-n", str(net_xml), "-r", str(routes), "--end", str(end),
                        "--no-step-log", "--no-warnings", "--time-to-teleport", "300",
                        "--seed", str(seed)], check=True, capture_output=True, env=env)
        t_sumo = time.perf_counter() - t
    return {"sumo_ms_per_run": t_sumo * 1000, "sumo_routing_ms": t_routes * 1000,
            "sumo_vehicles": n_veh, "sumo_sim_seconds": end, "sumo_period_s": period}


def run(weights_dir: Path = WEIGHTS_DIR, reports_dir: Path = REPORTS_DIR, network: dict | None = None,
        synth_meta: dict | None = None, quick: bool = False, with_sumo: bool = True) -> dict:
    if network is None:
        network = json.loads((weights_dir / "network.json").read_text(encoding="utf-8"))
    if synth_meta is None:
        p = SYNTH_DIR / "meta.json"
        synth_meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    t = time.perf_counter()
    pred = Predictor(weights_dir)
    topo, base = parse_network(network)
    pred.predict(network, random_changes(topo, base, np.random.default_rng(1)))
    cold = (time.perf_counter() - t) * 1000

    n_runs = 5 if quick else 50
    district = _time_network(pred, network, n_runs) | {"cold_start_ms": cold}
    log.info("Район: predict %.1f мс (медиана), GNN %.1f мс", district["predict_ms_median"],
             district["model_ms_median"])
    scaling = []
    pred._warned_unknown = True  # для чужих решёток базовое состояние заведомо неизвестно
    for n in ((4, 8) if quick else (4, 8, 12, 16, 24, 32, 48)):
        r = _time_network(pred, build_grid_python(n), 3 if quick else 15) | {"grid": f"{n}x{n}"}
        scaling.append(r)
        log.info("Решётка %dx%d: %d рёбер, predict %.1f мс, GNN %.1f мс", n, n, r["n_edges"],
                 r["predict_ms_median"], r["model_ms_median"])
    scaling.sort(key=lambda r: r["n_edges"])

    reference = {}
    if synth_meta.get("sim_seconds_per_scenario"):
        reference["macro_ms_per_scenario"] = synth_meta["sim_seconds_per_scenario"] * 1000
    if with_sumo and not quick:
        src = str(network.get("meta", {}).get("source", ""))
        net_xml = STUB_DIR / ("osm" if src.startswith("OSM") else "grid") / "network.net.xml"
        try:
            res = time_sumo(net_xml)
            if res:
                reference.update(res)
                log.info("SUMO: %.1f с на прогон (%d ТС)", res["sumo_ms_per_run"] / 1000, res["sumo_vehicles"])
        except subprocess.CalledProcessError as e:
            log.warning("SUMO не отработал: %s", (e.stderr or b"")[-500:])
    out = {"env": {"cpu": _cpu_name(), "torch_threads": torch.get_num_threads(),
                   "torch": torch.__version__, "python": platform.python_version()},
           "district": district, "scaling": scaling, "reference": reference}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "inference_time.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    from ml import report

    report._style()
    report.plot_inference(out, reports_dir / "inference_time.png")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Замер времени инференса")
    ap.add_argument("--no-sumo", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    run(quick=args.quick, with_sumo=not args.no_sumo)
    from ml import report

    report.write_markdown(REPORTS_DIR)


if __name__ == "__main__":
    main()
