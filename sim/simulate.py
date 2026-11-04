"""Запуск SUMO и сбор метрик по каждому ребру (edgeData).

Используется и базовым прогоном (generate_traffic.py), и генератором
сценариев (scenarios.py), чтобы все прогоны были настроены одинаково.

Метрики ребра по контракту:
    occupancy   — доля длины ребра, занятая машинами, усреднённая по времени (0..1);
                  в edgeData SUMO это поле occupancy в процентах, делим на 100;
    mean_speed  — средняя скорость машин на ребре, м/с (поле speed);
    travel_time — среднее время проезда ребра, с (поле traveltime = длина / mean_speed).
Для рёбер, по которым никто не проехал, SUMO с excludeEmpty="defaults"
пишет значения свободного движения (скорость = разрешённой, время = длина/скорость).
"""
from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from sim import config
from sim.sumo_tools import binary, run

EDGE_METRICS = ("occupancy", "mean_speed", "travel_time")
EXTRA_METRICS = ("density", "waiting_time", "time_loss", "entered", "left", "sampled_seconds")


def sumo_options(seed: int = config.SUMO_SEED, begin: int = config.SIM_BEGIN,
                 end: int = config.SIM_END) -> list[str]:
    """Общие опции SUMO для всех прогонов (их же пишем в base.sumocfg)."""
    return [
        "--begin", str(begin), "--end", str(end),
        "--seed", str(seed),
        # Поездки заданы между перекрёстками (fromJunction/toJunction): закрытие
        # одного ребра не делает поездку невозможной, SUMO выберет другой выезд.
        "--junction-taz",
        # Маршрутизация по текущей обстановке: каждая машина пересчитывает путь
        # по измеренным временам проезда. Без этого расширение дороги не
        # притягивало бы поток — выбор маршрута шёл бы по свободному движению.
        "--device.rerouting.probability", "1",
        "--device.rerouting.period", str(config.REROUTING_PERIOD),
        "--device.rerouting.adaptation-interval", str(config.REROUTING_ADAPTATION_INTERVAL),
        "--time-to-teleport", str(config.TIME_TO_TELEPORT),
        "--ignore-route-errors",
        "--no-step-log", "--no-warnings",
        "--duration-log.statistics",  # средние показатели поездок в statistic-output
        "--threads", "1",            # параллелим на уровне сценариев, не внутри SUMO
    ]


def edgedata_additional(out_file: str, begin: int, end: int) -> str:
    return (
        "<additional>\n"
        f'    <edgeData id="metrics" file="{out_file}" begin="{begin}" end="{end}" '
        'excludeEmpty="defaults" withInternal="false"/>\n'
        "</additional>\n"
    )


def run_sumo(net_file: Path, trips_file: Path, work_dir: Path, *,
             seed: int = config.SUMO_SEED, begin: int = config.SIM_BEGIN,
             end: int = config.SIM_END, warmup: int = config.WARMUP,
             demand_scale: float = 1.0, timeout: float = config.SCENARIO_TIMEOUT) -> dict:
    """Прогоняет SUMO и возвращает {"edges": {edge_id: метрики}, "summary": {...}}."""
    work_dir.mkdir(parents=True, exist_ok=True)
    add_file = work_dir / "edgedata.add.xml"
    add_file.write_text(edgedata_additional("edgedata.xml", begin + warmup, end), encoding="utf-8")
    stats_file = work_dir / "stats.xml"
    cmd = [binary("sumo"), "-n", Path(net_file).resolve(), "-r", Path(trips_file).resolve(),
           "-a", add_file.name, "--statistic-output", stats_file.name,
           "--scale", f"{demand_scale:g}", *sumo_options(seed, begin, end)]
    t0 = time.perf_counter()
    run(cmd, cwd=work_dir, timeout=timeout)
    sumo_seconds = time.perf_counter() - t0
    edges = parse_edgedata(work_dir / "edgedata.xml")
    summary = parse_statistics(stats_file)
    summary["sumo_seconds"] = round(sumo_seconds, 3)
    return {"edges": edges, "summary": summary}


def _f(el: ET.Element, name: str, default: float = math.nan) -> float:
    v = el.get(name)
    return float(v) if v is not None else default


def parse_edgedata(path: Path) -> dict[str, dict]:
    """edgeData (один интервал) → {edge_id: {occupancy, mean_speed, travel_time, ...}}."""
    result: dict[str, dict] = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "edge":
            continue
        result[el.get("id")] = {
            "occupancy": min(max(_f(el, "occupancy", 0.0) / 100.0, 0.0), 1.0),
            "mean_speed": _f(el, "speed"),
            "travel_time": _f(el, "traveltime"),
            "density": _f(el, "density", 0.0),          # авт/км
            "waiting_time": _f(el, "waitingTime", 0.0),  # суммарное время стоянки, с
            "time_loss": _f(el, "timeLoss", 0.0),        # суммарные потери времени, с
            "entered": int(_f(el, "entered", 0.0)),      # въехало машин
            "left": int(_f(el, "left", 0.0)),
            "sampled_seconds": _f(el, "sampledSeconds", 0.0),  # машино-секунды на ребре
        }
        el.clear()
    return result


def parse_statistics(path: Path) -> dict:
    """Сводка прогона из --statistic-output."""
    root = ET.parse(path).getroot()
    out: dict = {}
    veh = root.find("vehicles")
    if veh is not None:
        out["veh_loaded"] = int(veh.get("loaded", 0))
        out["veh_inserted"] = int(veh.get("inserted", 0))
        out["veh_running"] = int(veh.get("running", 0))
        out["veh_waiting"] = int(veh.get("waiting", 0))
    tel = root.find("teleports")
    if tel is not None:
        out["teleports"] = int(tel.get("total", 0))
        out["teleports_jam"] = int(tel.get("jam", 0))
    trips = root.find("vehicleTripStatistics")
    if trips is not None:
        out["trips_completed"] = int(float(trips.get("count", 0)))
        for src, dst in (("duration", "trip_duration_mean"), ("routeLength", "trip_length_mean"),
                         ("speed", "trip_speed_mean"), ("waitingTime", "trip_waiting_mean"),
                         ("timeLoss", "trip_time_loss_mean"), ("departDelay", "depart_delay_mean")):
            if trips.get(src) is not None:
                out[dst] = float(trips.get(src))
        out["total_time_loss_h"] = round(out.get("trip_time_loss_mean", 0.0)
                                         * out["trips_completed"] / 3600, 3)
    return out
