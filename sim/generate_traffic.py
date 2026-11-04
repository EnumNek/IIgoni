"""Шаг 2. Генерация трафика (randomTrips) и базовая симуляция на 1 час.

    python -m sim.generate_traffic                     # спрос по config.INSERTION_RATE
    python -m sim.generate_traffic --rate 3000         # другой уровень спроса
    python -m sim.generate_traffic --calibrate 1200,2400,3600   # подбор уровня спроса

Результат:
    sim/data/trips.xml           — 1 час поездок между перекрёстками;
    sim/data/base.sumocfg        — готовая конфигурация (для agents/ и sumo-gui);
    sim/data/baseline/edges.json — метрики каждого ребра в исходной сети;
    sim/data/baseline/summary.json — сводка прогона.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from sim import config
from sim.simulate import run_sumo, sumo_options
from sim.sumo_tools import run_python_tool


def generate_trips(net_file: Path = config.BASE_NET, out_file: Path = config.TRIPS_FILE,
                   rate: float = config.INSERTION_RATE, seed: int = config.TRIPS_SEED) -> Path:
    """Случайные поездки на интервале [SIM_BEGIN, SIM_END).

    * --insertion-rate — интенсивность въезда, авт/ч (пуассоновский поток, --random-depart);
    * --fringe-factor  — поездки чаще начинаются и заканчиваются на границе района:
      это транзитный поток через район, характерный для городской сети;
    * --lanes          — вероятность выбора ребра пропорциональна числу полос
      (на крупных улицах больше источников и стоков поездок);
    * --min-distance   — отсекаем слишком короткие поездки;
    * --junction-taz   — поездки между перекрёстками, а не рёбрами (см. simulate.py);
    * --validate       — duarouter проверяет, что маршрут существует.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_file.resolve()
    # randomTrips/duarouter оставляют промежуточные файлы в рабочей папке —
    # запускаем во временной, чтобы не засорять sim/data.
    with tempfile.TemporaryDirectory() as tmp:
        _random_trips(net_file, out_file, rate, seed, Path(tmp))
    return out_file


def _random_trips(net_file: Path, out_file: Path, rate: float, seed: int, cwd: Path) -> None:
    run_python_tool("randomTrips.py", [
        "-n", str(Path(net_file).resolve()), "-o", str(out_file),
        "-b", str(config.SIM_BEGIN), "-e", str(config.SIM_END),
        "--insertion-rate", f"{rate:g}", "--random-depart",
        "--fringe-factor", str(config.FRINGE_FACTOR),
        "--lanes",
        "--min-distance", f"{config.MIN_TRIP_DISTANCE:g}",
        "--junction-taz", "--validate", "--remove-loops",
        "--seed", str(seed), "--prefix", "veh",
        "--trip-attributes", 'departLane="best" departSpeed="max"',
    ], cwd=cwd)


def write_sumocfg(path: Path = config.BASE_SUMOCFG, net_file: Path = config.BASE_NET,
                  trips_file: Path = config.TRIPS_FILE) -> Path:
    """Конфигурация базового прогона — те же опции, что и в датасете."""
    opts = sumo_options()
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<configuration>",
             "    <input>",
             f'        <net-file value="{net_file.name}"/>',
             f'        <route-files value="{trips_file.name}"/>',
             "    </input>", "    <processing>"]
    i = 0
    while i < len(opts):
        key = opts[i].lstrip("-")
        if i + 1 < len(opts) and not opts[i + 1].startswith("--"):
            lines.append(f'        <{key} value="{opts[i + 1]}"/>')
            i += 2
        else:
            lines.append(f'        <{key} value="true"/>')
            i += 1
    lines += ["    </processing>", "</configuration>", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_baseline(net_file: Path = config.BASE_NET, trips_file: Path = config.TRIPS_FILE,
                 out_dir: Path = config.BASELINE_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    res = run_sumo(net_file, trips_file, work)
    (out_dir / "edges.json").write_text(json.dumps(res["edges"], separators=(",", ":")), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(res["summary"], ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    return res


def describe(res: dict) -> str:
    s = res["summary"]
    occ = sorted(e["occupancy"] for e in res["edges"].values())
    n = len(occ)
    return (f"вставлено {s.get('veh_inserted')} из {s.get('veh_loaded')}, завершили {s.get('trips_completed')}, "
            f"телепортов {s.get('teleports')}, ср. поездка {s.get('trip_duration_mean', 0):.0f} с, "
            f"ср. потери {s.get('trip_time_loss_mean', 0):.0f} с | occupancy: медиана {occ[n // 2]:.3f}, "
            f"p90 {occ[int(n * .9)]:.3f}, max {occ[-1]:.3f} | SUMO {s['sumo_seconds']:.1f} с")


def calibrate(rates: list[float]) -> None:
    """Прогоняет базовую сеть на нескольких уровнях спроса (для выбора INSERTION_RATE)."""
    tmp = config.DATA_DIR / "calibration"
    tmp.mkdir(parents=True, exist_ok=True)
    for rate in rates:
        trips = generate_trips(out_file=tmp / f"trips_{int(rate)}.xml", rate=rate)
        res = run_sumo(config.BASE_NET, trips, tmp / f"work_{int(rate)}")
        print(f"[calibrate] {rate:>6.0f} авт/ч: {describe(res)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rate", type=float, default=config.INSERTION_RATE, help="интенсивность, авт/ч")
    parser.add_argument("--seed", type=int, default=config.TRIPS_SEED)
    parser.add_argument("--calibrate", type=str, help="список интенсивностей через запятую")
    args = parser.parse_args(argv)
    if not config.BASE_NET.exists():
        raise SystemExit("Нет sim/data/base.net.xml — сначала запустите python -m sim.build_network")
    if args.calibrate:
        calibrate([float(x) for x in args.calibrate.split(",")])
        return
    t0 = time.perf_counter()
    generate_trips(rate=args.rate, seed=args.seed)
    write_sumocfg()
    print(f"[generate_traffic] поездки: {config.TRIPS_FILE} ({time.perf_counter() - t0:.1f} с)")
    res = run_baseline()
    print(f"[generate_traffic] базовый прогон: {describe(res)}")


if __name__ == "__main__":
    main()
