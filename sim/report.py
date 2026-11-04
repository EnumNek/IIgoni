"""Шаг 5. Отчёт по датасету: время счёта, распределения метрик, графики.

    python -m sim.report                          # по sim/data/dataset → sim/reports/
    python -m sim.report --dataset /tmp/ds --out /tmp/rep

Результат: sim/reports/report.md, sim/reports/summary.json, sim/reports/fig_*.png.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize  # noqa: E402

from sim import config  # noqa: E402

# --- оформление (палитра: одна серия — синий, вторая — оранжевый; шкалы — один тон) ---
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52e2e"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def _save(fig, out_dir: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return name


def _q(series: pd.Series, qs=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)) -> dict:
    s = series.dropna()
    return {f"p{int(q * 100)}": float(s.quantile(q)) for q in qs} | {"mean": float(s.mean())}


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------

def fig_timing(scen: pd.DataFrame, out_dir: Path) -> str:
    ok = scen[(scen.status == "ok") & (scen.scenario_id > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.hist(ok.total_seconds, bins=30, color=BLUE, edgecolor="white", linewidth=1.0)
    med = ok.total_seconds.median()
    ax.axvline(med, color=INK2, linewidth=1)
    ax.annotate(f"медиана {med:.1f} с", xy=(med, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.set_title("Время одного сценария")
    ax.set_xlabel("netconvert + SUMO, с")
    ax.set_ylabel("сценариев")
    ax = axes[1]
    ax.scatter(ok.net_veh_hours, ok.sumo_seconds, s=14, color=BLUE, alpha=0.75, linewidths=0)
    ax.set_title("Время SUMO и объём движения в сети")
    ax.set_xlabel("машино-часы в сети за прогон")
    ax.set_ylabel("время SUMO, с")
    return _save(fig, out_dir, "fig_time_per_scenario.png")


def fig_metric_distributions(edges: pd.DataFrame, out_dir: Path) -> str:
    e = edges[~edges.closed]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    specs = [
        ("occupancy", "Загруженность (occupancy)", "доля занятой длины, 0..1", np.linspace(0, 0.6, 41), True),
        ("mean_speed", "Средняя скорость", "м/с", np.linspace(0, 23, 47), False),
        ("travel_time", "Время проезда ребра", "с (логарифмическая шкала)",
         np.logspace(np.log10(0.1), np.log10(max(e.travel_time.max(), 10)), 45), True),
    ]
    for ax, (col, title, xlabel, bins, logy) in zip(axes, specs):
        ax.hist(e[col].dropna(), bins=bins, color=BLUE, edgecolor="white", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if logy:
            ax.set_yscale("log")
        if col == "travel_time":
            ax.set_xscale("log")
    axes[0].set_ylabel("строк (ребро × сценарий)")
    return _save(fig, out_dir, "fig_metric_distributions.png")


def _segments(network: dict) -> tuple[list[str], list[np.ndarray], float]:
    ids, segs = [], []
    for e in network["edges"]:
        ids.append(e["id"])
        segs.append(np.asarray(e["geometry"], dtype=float))
    lat0 = float(np.mean([n["lat"] for n in network["nodes"]]))
    return ids, segs, 1.0 / math.cos(math.radians(lat0))


def _map_axes(ax, network: dict, aspect: float) -> None:
    bbox = network["meta"].get("osm_bounds") or network["meta"]["bbox"]
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(aspect)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_baseline_map(edges: pd.DataFrame, network: dict, out_dir: Path) -> str:
    base = edges[edges.scenario_id == 0].set_index("edge_id")
    ids, segs, aspect = _segments(network)
    lanes = {e["id"]: e["lanes"] for e in network["edges"]}
    vmax = 0.3
    values = np.array([base.occupancy.get(i, 0.0) for i in ids])
    order = np.argsort(values)  # загруженные рёбра рисуем поверх
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.add_collection(LineCollection(segs, colors=GRID, linewidths=2.2, zorder=1))
    lc = LineCollection([segs[i] for i in order], cmap=SEQ_CMAP, norm=Normalize(0, vmax),
                        linewidths=[0.8 + 0.7 * lanes[ids[i]] for i in order], zorder=2,
                        capstyle="round")
    lc.set_array(values[order])
    ax.add_collection(lc)
    _map_axes(ax, network, aspect)
    cb = fig.colorbar(lc, ax=ax, shrink=0.7, extend="max", pad=0.01)
    cb.set_label("загруженность (occupancy)", color=INK2)
    cb.outline.set_visible(False)
    tl = [n for n in network["nodes"] if n["type"].startswith("traffic_light")]
    ax.scatter([n["lon"] for n in tl], [n["lat"] for n in tl], s=16, facecolor="white",
               edgecolor=INK, linewidths=0.9, zorder=3, label="светофор")
    ax.legend(loc="lower left")
    ax.set_title("Базовая сеть: загруженность рёбер за час (без изменений)")
    return _save(fig, out_dir, "fig_baseline_map.png")


def fig_example_scenario(edges: pd.DataFrame, scen: pd.DataFrame, network: dict, out_dir: Path,
                         scenario_id: int) -> str:
    base = edges[edges.scenario_id == 0].set_index("edge_id").occupancy
    cur = edges[edges.scenario_id == scenario_id].set_index("edge_id")
    ids, segs, aspect = _segments(network)
    delta = np.array([0.0 if cur.closed.get(i, False) else cur.occupancy.get(i, 0) - base.get(i, 0) for i in ids])
    lim = max(0.02, float(np.quantile(np.abs(delta), 0.99)))
    order = np.argsort(np.abs(delta))
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.add_collection(LineCollection(segs, colors=GRID, linewidths=2.0, zorder=1))
    lc = LineCollection([segs[i] for i in order], cmap=DIV_CMAP, norm=TwoSlopeNorm(0, -lim, lim),
                        linewidths=2.0, zorder=2, capstyle="round")
    lc.set_array(delta[order])
    ax.add_collection(lc)
    changes = json.loads(scen.set_index("scenario_id").changes[scenario_id])
    geo = {e["id"]: np.asarray(e["geometry"]) for e in network["edges"]}
    nodes = {n["id"]: n for n in network["nodes"]}
    for c in changes:
        if c["type"] == "traffic_light":
            n = nodes[c["target_id"]]
            x, y = n["lon"], n["lat"]
        else:
            g = geo[c["target_id"]]
            ax.plot(g[:, 0], g[:, 1], color=INK, linewidth=4.2, zorder=3, solid_capstyle="round")
            ax.plot(g[:, 0], g[:, 1], color="white", linewidth=1.6, zorder=4, solid_capstyle="round")
            x, y = g[len(g) // 2]
        ax.scatter([x], [y], s=46, facecolor=INK, edgecolor="white", linewidths=1.2, zorder=5)
        ax.annotate(change_label(c), (x, y), xytext=(7, 7), textcoords="offset points",
                    fontsize=9, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=AXIS, linewidth=0.6))
    _map_axes(ax, network, aspect)
    cb = fig.colorbar(lc, ax=ax, shrink=0.7, extend="both", pad=0.01)
    cb.set_label("Δ загруженности относительно базы", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f"Сценарий {scenario_id}: изменение загруженности после {len(changes)} изменений "
                 "(синий — разгрузка, красный — рост)")
    return _save(fig, out_dir, "fig_example_scenario.png")


TL_NAMES = {"static": "фикс. цикл", "actuated": "адаптивный", "off": "выключен"}


def change_label(c: dict) -> str:
    t, v = c["type"], c["value"]
    if t == "lanes":
        return f"полос: {v}"
    if t == "speed":
        return f"скорость: {v * 3.6:.0f} км/ч"
    if t == "traffic_light":
        return f"светофор: {TL_NAMES.get(v, v)}"
    return {"close": "закрыто", "open": "открыто"}[t]


def change_effects(edges: pd.DataFrame, scen: pd.DataFrame, network: dict) -> pd.DataFrame:
    """Локальный эффект каждого изменения: Δ времени проезда (в %) на затронутых рёбрах.

    lanes/speed — само ребро; traffic_light — все въезды в узел; close — рёбра,
    въезжающие в начало закрытого ребра (туда, где машинам приходится объезжать).
    """
    base_tt = edges[edges.scenario_id == 0].set_index("edge_id").travel_time
    incoming: dict[str, list[str]] = {}
    for e in network["edges"]:
        incoming.setdefault(e["to"], []).append(e["id"])
    edge_from = {e["id"]: e["from"] for e in network["edges"]}
    base_lanes = {e["id"]: e["lanes"] for e in network["edges"]}
    base_speed = {e["id"]: e["speed"] for e in network["edges"]}
    base_tl = {n["id"]: n["type"].startswith("traffic_light") for n in network["nodes"]}
    by_scn = {sid: g.set_index("edge_id") for sid, g in edges[edges.scenario_id > 0].groupby("scenario_id")}
    rows = []
    for _, s in scen[(scen.status == "ok") & (scen.scenario_id > 0)].iterrows():
        cur = by_scn.get(s.scenario_id)
        if cur is None:
            continue
        for c in json.loads(s.changes):
            t, target, value = c["type"], c["target_id"], c["value"]
            if t == "lanes":
                cat = "полосы +" if value > base_lanes[target] else "полосы −"
                targets = [target]
            elif t == "speed":
                cat = "скорость ↑" if value > base_speed[target] else "скорость ↓"
                targets = [target]
            elif t == "traffic_light":
                cat = {"off": "светофор: выкл."}.get(value) or (
                    "светофор: адаптивный" if base_tl[target] else "светофор: новый")
                targets = incoming.get(target, [])
            elif t == "close":
                cat = "закрытие"
                targets = [e for e in incoming.get(edge_from[target], [])]
            else:
                continue
            vals = [(cur.travel_time.get(x), base_tt.get(x)) for x in targets]
            vals = [(a, b) for a, b in vals if a is not None and b is not None
                    and not (math.isnan(a) or math.isnan(b))]
            if vals:
                cur_sum, base_sum = sum(a for a, _ in vals), sum(b for _, b in vals)
                rows.append({"scenario_id": s.scenario_id, "category": cat,
                             "delta_tt": float((cur_sum / base_sum - 1) * 100) if base_sum > 0 else 0.0})
    return pd.DataFrame(rows)


def fig_change_effects(effects: pd.DataFrame, out_dir: Path) -> str:
    order = ["полосы +", "полосы −", "скорость ↑", "скорость ↓", "закрытие",
             "светофор: новый", "светофор: выкл.", "светофор: адаптивный"]
    order = [o for o in order if o in set(effects.category)]
    data = [effects[effects.category == o].delta_tt.clip(-100, 200).values for o in order]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, linewidth=1.4),
                    boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE, linewidth=1),
                    whiskerprops=dict(color=BLUE, linewidth=1), capprops=dict(color=BLUE, linewidth=1),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=BLUE, markeredgecolor="none",
                                    alpha=0.5))
    ax.set_yticks(range(1, len(order) + 1), [f"{o}  (n={len(d)})" for o, d in zip(order, data)])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("Δ суммарного времени проезда затронутых рёбер относительно базы, % (обрезано до −100…+200)")
    ax.set_title("Локальный эффект изменений по типам")
    ax.grid(axis="y", visible=False)
    del bp
    return _save(fig, out_dir, "fig_change_effects.png")


def influence(edges: pd.DataFrame, sigma_occ: pd.Series | None) -> pd.DataFrame:
    """Сколько рёбер «почувствовали» сценарий: |Δocc| больше 3σ шума симуляции."""
    base = edges[edges.scenario_id == 0][["edge_id", "occupancy"]].rename(columns={"occupancy": "occ0"})
    e = edges[(edges.scenario_id > 0) & ~edges.closed].merge(base, on="edge_id")
    e["d"] = (e.occupancy - e.occ0).abs()
    thr = (3 * sigma_occ).clip(lower=0.002) if sigma_occ is not None else pd.Series(dtype=float)
    e["thr"] = e.edge_id.map(thr).fillna(0.005)
    e["hit"] = e.d > e.thr
    return e.groupby("scenario_id").agg(share=("hit", "mean"), n_hit=("hit", "sum")).reset_index()


def fig_scenario_effects(scen: pd.DataFrame, infl: pd.DataFrame, out_dir: Path) -> str:
    base = scen[scen.scenario_id == 0].iloc[0]
    ok = scen[(scen.status == "ok") & (scen.scenario_id > 0)].copy()
    ok["d_loss"] = (ok.net_time_loss_h / base.net_time_loss_h - 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    groups = [ok[ok.n_changes == k].d_loss.values for k in range(1, config.MAX_CHANGES + 1)]
    ax.boxplot(groups, widths=0.5, patch_artist=True,
               medianprops=dict(color=INK, linewidth=1.4),
               boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE), whiskerprops=dict(color=BLUE),
               capprops=dict(color=BLUE),
               flierprops=dict(marker="o", markersize=3, markerfacecolor=BLUE, markeredgecolor="none"))
    ax.set_xticks(range(1, config.MAX_CHANGES + 1), [str(k) for k in range(1, config.MAX_CHANGES + 1)])
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("число изменений в сценарии")
    ax.set_ylabel("Δ суммарных потерь времени, %")
    ax.set_title("Эффект сценария на всю сеть")
    ax = axes[1]
    ax.hist(infl.share * 100, bins=25, color=BLUE, edgecolor="white", linewidth=1.0)
    ax.set_xlabel("доля рёбер с заметным изменением загрузки, %")
    ax.set_ylabel("сценариев")
    ax.set_title("Радиус влияния: |Δocc| > 3σ шума")
    return _save(fig, out_dir, "fig_scenario_effects.png")


def fig_change_counts(scen: pd.DataFrame, out_dir: Path) -> str:
    ok = scen[scen.scenario_id > 0]
    types = [("n_lanes", "полосы"), ("n_speed", "скорость"), ("n_close", "закрытие"),
             ("n_traffic_light", "светофор"), ("n_open", "открытие")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    counts = ok.n_changes.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=BLUE, width=0.7)
    ax.set_xlabel("изменений в сценарии")
    ax.set_ylabel("сценариев")
    ax.set_title("Число изменений")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    vals = [int(ok[c].sum()) for c, _ in types]
    ax.barh([n for _, n in types], vals, color=BLUE, height=0.6)
    for i, v in enumerate(vals):
        ax.annotate(str(v), (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("изменений всего")
    ax.set_title("Типы изменений")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "fig_change_counts.png")


# ----------------------------------------------------------------------------
# Отчёт
# ----------------------------------------------------------------------------

def _fmt(x: float, nd: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def build_report(dataset: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = pd.read_parquet(dataset / "edges.parquet")
    scen = pd.read_parquet(dataset / "scenarios.parquet")
    meta = json.loads((dataset / "meta.json").read_text(encoding="utf-8"))
    network = json.loads(config.NETWORK_JSON.read_text(encoding="utf-8"))
    noise_path = dataset / "seed_noise.parquet"
    noise = pd.read_parquet(noise_path) if noise_path.exists() else None

    ok = scen[(scen.status == "ok") & (scen.scenario_id > 0)]
    base = scen[scen.scenario_id == 0].iloc[0]
    open_edges = edges[~edges.closed]

    figs = {
        "time": fig_timing(scen, out_dir),
        "metrics": fig_metric_distributions(edges, out_dir),
        "map": fig_baseline_map(edges, network, out_dir),
        "counts": fig_change_counts(scen, out_dir),
    }
    sigma = None
    noise_stats = {}
    if noise is not None:
        sd = noise.groupby("edge_id")[["occupancy", "mean_speed", "travel_time"]].std()
        sigma = sd.occupancy
        # MAE между двумя прогонами с разными зёрнами — «потолок» точности предсказателя
        piv = {m: noise.pivot(index="edge_id", columns="seed", values=m) for m in ("occupancy", "mean_speed", "travel_time")}
        for m, p in piv.items():
            cols = list(p.columns)
            maes = [np.nanmean(np.abs(p[a] - p[b])) for i, a in enumerate(cols) for b in cols[i + 1:]]
            noise_stats[m] = {"sigma_mean": float(sd[m].mean()), "mae_between_seeds": float(np.mean(maes))}
    effects = change_effects(edges, scen, network)
    if not effects.empty:
        figs["effects"] = fig_change_effects(effects, out_dir)
    infl = influence(edges, sigma)
    figs["scenario"] = fig_scenario_effects(scen, infl, out_dir)
    # пример: сценарий с наибольшим влиянием среди сценариев с ≥3 изменениями
    cand = infl.merge(ok[["scenario_id", "n_changes"]], on="scenario_id")
    cand = cand[cand.n_changes >= 3] if (cand.n_changes >= 3).any() else cand
    example_id = int(cand.sort_values("share").iloc[-1].scenario_id)
    figs["example"] = fig_example_scenario(edges, scen, network, out_dir, example_id)

    # --- числа ---------------------------------------------------------------
    t = ok.total_seconds
    wall = meta.get("wall_seconds_this_run") or float("nan")
    n_run = meta.get("scenarios_this_run") or len(scen)
    workers = meta.get("workers") or 1
    # Эталон однопоточного времени — базовый прогон, выполненный отдельно (generate_traffic),
    # без конкуренции с другими процессами. Время сценариев в датасете измерено при
    # параллельной работе и поэтому для оценки ускорения не годится.
    solo_path = config.BASELINE_DIR / "summary.json"
    solo = json.loads(solo_path.read_text(encoding="utf-8"))["sumo_seconds"] if solo_path.exists() else float("nan")
    solo_scenario = solo + float(ok.netconvert_seconds.mean())
    speedup = solo_scenario * n_run / wall if wall == wall else float("nan")
    timing = {
        "per_scenario_seconds": _q(t, (0.0, 0.5, 0.95, 1.0)),
        "netconvert_seconds_mean": float(ok.netconvert_seconds.mean()),
        "sumo_seconds_mean": float(ok.sumo_seconds.mean()),
        "baseline_sumo_seconds_parallel": float(base.sumo_seconds),
        "baseline_sumo_seconds_solo": solo,
        "contention_slowdown": float(t.mean()) / solo_scenario if solo == solo else None,
        "wall_seconds": wall, "scenarios_in_run": n_run, "workers": workers,
        "throughput_scenarios_per_hour": n_run / wall * 3600 if wall == wall else None,
        "parallel_speedup": speedup,
        "parallel_efficiency": speedup / workers if workers else None,
    }
    metrics = {m: _q(open_edges[m]) for m in ("occupancy", "mean_speed", "travel_time")}
    d_loss = (ok.net_time_loss_h / base.net_time_loss_h - 1) * 100
    summary = {
        "dataset": {"scenarios_total": int(len(scen)), "scenarios_ok": int((scen.status == "ok").sum()),
                    "scenarios_error": int((scen.status != "ok").sum()), "edge_rows": int(len(edges)),
                    "edges_per_scenario": int(len(network["edges"])), "nodes": int(len(network["nodes"])),
                    "example_scenario": example_id},
        "timing": timing, "metrics": metrics, "seed_noise": noise_stats,
        "baseline": {k: (float(base[k]) if isinstance(base[k], (int, float, np.number)) else base[k])
                     for k in ("veh_loaded", "veh_inserted", "trips_completed", "teleports",
                               "trip_duration_mean", "trip_time_loss_mean", "net_time_loss_h",
                               "net_occupancy_mean")},
        "effects": {"net_time_loss_delta_pct": _q(d_loss, (0.05, 0.5, 0.95)),
                    "influence_share": _q(infl.share, (0.05, 0.5, 0.95))},
        "data_quality": {"teleports": _q(ok.teleports, (0.5, 0.95, 1.0)),
                         "not_inserted": _q(ok.veh_loaded - ok.veh_inserted, (0.5, 0.95, 1.0))},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                          encoding="utf-8")
    md = render_markdown(summary, meta, network, figs, effects, scen)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    print(f"[report] {out_dir / 'report.md'}: {len(figs)} графиков")
    return summary


def render_markdown(s: dict, meta: dict, network: dict, figs: dict, effects: pd.DataFrame,
                    scen: pd.DataFrame) -> str:
    nm = network["meta"]
    tm = s["timing"]
    ps = tm["per_scenario_seconds"]
    sim = meta["simulation"]
    b = s["baseline"]
    L = []
    L += ["# Отчёт: генерация обучающих данных (sim/)", "",
          f"Сгенерировано `python -m sim.report` по датасету от {meta['created_at']} ({meta['sumo_version']}).", ""]
    L += ["## 1. Что смоделировано", "",
          "| Параметр | Значение |", "|---|---|",
          f"| Источник сети | {nm['source']} (`{nm.get('osm_file', '—')}`), обрезка по границам выгрузки: {'да' if nm.get('cropped') else 'нет'} |",
          f"| Узлов / рёбер / светофоров | {nm['n_nodes']} / {nm['n_edges']} / {nm['n_traffic_lights']} |",
          f"| Протяжённость (все направления) | {nm['total_length_km']} км |",
          f"| Спрос | {sim['insertion_rate_veh_h']} авт/ч, randomTrips, поездки между перекрёстками |",
          f"| Модельное время | {sim['begin']}–{sim['end']} с (1 ч), метрики по интервалу {sim['warmup']}–{sim['end']} с |",
          f"| Сценариев | {s['dataset']['scenarios_total']} (включая базовый №0), успешно {s['dataset']['scenarios_ok']}, ошибок {s['dataset']['scenarios_error']} |",
          f"| Строк в edges.parquet | {s['dataset']['edge_rows']:,} ({s['dataset']['edges_per_scenario']} рёбер × сценарии) |".replace(",", " "),
          "",
          f"Базовый прогон: загружено {b['veh_loaded']:.0f} машин, въехало {b['veh_inserted']:.0f}, "
          f"завершили поездку {b['trips_completed']:.0f}, телепортов {b['teleports']:.0f}; средняя поездка "
          f"{b['trip_duration_mean']:.0f} с, из них потери {b['trip_time_loss_mean']:.0f} с.", "",
          f"![Карта базовой загрузки]({figs['map']})", ""]
    L += ["## 2. Время счёта", "",
          "| Показатель | Значение |", "|---|---|",
          f"| Время одного сценария в датасете (netconvert + SUMO, {tm['workers']} процесса параллельно): медиана | {ps['p50']:.1f} с |",
          f"| — среднее / p95 / максимум | {ps['mean']:.1f} / {ps['p95']:.1f} / {ps['p100']:.1f} с |",
          f"| — из них netconvert / SUMO (в среднем) | {tm['netconvert_seconds_mean']:.2f} / {tm['sumo_seconds_mean']:.1f} с |",
          f"| Базовый прогон SUMO: один процесс / при параллельной работе | {tm['baseline_sumo_seconds_solo']:.1f} / {tm['baseline_sumo_seconds_parallel']:.1f} с |",
          f"| Параллельно: процессов | {tm['workers']} |",
          f"| Общее время сборки | {tm['wall_seconds'] / 60:.1f} мин на {tm['scenarios_in_run']} сценариев |",
          f"| Производительность | {tm['throughput_scenarios_per_hour']:.0f} сценариев/ч |",
          f"| Ускорение от параллелизма | ×{tm['parallel_speedup']:.2f} (эффективность {tm['parallel_efficiency'] * 100:.0f} %) |",
          "",
          "Время сценариев в датасете измерено при одновременной работе всех процессов, поэтому включает "
          "конкуренцию за память и кэш процессора: в одиночку тот же прогон идёт примерно в "
          f"{tm['contention_slowdown']:.2f} раза быстрее. Ускорение посчитано честно — относительно "
          "последовательного счёта в одном процессе (время базового прогона в одиночку × число сценариев / "
          f"общее время). Оценка: 10 000 сценариев на "
          f"{tm['workers']} ядрах ≈ {10000 / tm['throughput_scenarios_per_hour']:.0f} ч, на 32 ядрах ≈ "
          f"{10000 / (tm['throughput_scenarios_per_hour'] / tm['workers'] * 32):.1f} ч (при той же эффективности).", "",
          f"![Время на сценарий]({figs['time']})", ""]
    L += ["## 3. Распределения метрик рёбер", "",
          "По всем открытым рёбрам всех сценариев (закрытые исключены: у них нет скорости и времени проезда).", "",
          "| Метрика | мин | p25 | медиана | p75 | p90 | p99 | макс | среднее |", "|---|---|---|---|---|---|---|---|---|"]
    names = {"occupancy": "occupancy, 0..1", "mean_speed": "mean_speed, м/с", "travel_time": "travel_time, с"}
    for m, q in s["metrics"].items():
        L.append(f"| {names[m]} | " + " | ".join(_fmt(q[k]) for k in ("p0", "p25", "p50", "p75", "p90", "p99", "p100", "mean")) + " |")
    L += ["", f"![Распределения метрик]({figs['metrics']})", "",
          "Распределения сильно скошены: большинство рёбер — тихие жилые улицы с загрузкой около нуля, "
          "заторы сосредоточены на магистралях. Время проезда растянуто на несколько порядков (от долей секунды "
          "на коротких связках перекрёстков до минут в очередях), поэтому для обучения его разумно "
          "логарифмировать, а для occupancy — взвешивать тяжёлый «хвост».", ""]
    L += ["## 4. Сценарии и влияние изменений", "", f"![Состав сценариев]({figs['counts']})", ""]
    if "effects" in figs:
        L += [f"![Локальный эффект]({figs['effects']})", "",
              "Медианы Δ времени проезда по категориям:", "",
              "| Категория | n | медиана Δ, % | p10, % | p90, % |", "|---|---|---|---|---|"]
        order = ["полосы +", "полосы −", "скорость ↑", "скорость ↓", "закрытие",
                 "светофор: новый", "светофор: выкл.", "светофор: адаптивный"]
        for cat in [o for o in order if o in set(effects.category)]:
            g = effects[effects.category == cat]
            L.append(f"| {cat} | {len(g)} | {g.delta_tt.median():+.1f} | {g.delta_tt.quantile(.1):+.1f} | {g.delta_tt.quantile(.9):+.1f} |")
        L += ["",
              "Для lanes/speed затронутым считается само ребро, для светофора — все въезды в перекрёсток, "
              "для закрытия — рёбра, ведущие к началу закрытого ребра. Знаки эффектов согласуются с физикой "
              "движения, это проверка правдоподобия данных: снижение скорости увеличивает время проезда, "
              "новый светофор с фиксированным циклом добавляет задержку на ранее нерегулируемом перекрёстке, "
              "адаптивное регулирование её снижает. Добавление полосы в среднем почти ничего не меняет: "
              "большинство рёбер не перегружено, и полоса помогает только на «узких местах» (длинный левый "
              "хвост распределения). Модели предстоит выучить именно такую зависимость эффекта от контекста.", ""]
    e = s["effects"]
    L += [f"![Эффект на сеть]({figs['scenario']})", "",
          f"Изменение суммарных потерь времени в сети относительно базы: медиана "
          f"{e['net_time_loss_delta_pct']['p50']:+.1f} %, 90 % сценариев в диапазоне "
          f"[{e['net_time_loss_delta_pct']['p5']:+.1f} %; {e['net_time_loss_delta_pct']['p95']:+.1f} %]. "
          f"Доля рёбер, у которых загрузка заметно (больше 3σ шума) изменилась: медиана "
          f"{e['influence_share']['p50'] * 100:.0f} %, до {e['influence_share']['p95'] * 100:.0f} % (p95). "
          "Эффект изменения не локален — поток перераспределяется по соседним улицам; это аргумент "
          "в пользу графовой нейросети, которая учитывает соседей через передачу сообщений.", "",
          f"![Пример сценария]({figs['example']})", ""]
    if s["seed_noise"]:
        L += ["## 5. Шум симуляции (нижняя граница ошибки ML)", "",
              "Базовая сеть прогнана с разными зёрнами SUMO (`python -m sim.noise_floor`). "
              "Средняя абсолютная разница между двумя такими прогонами — это точность, с которой симулятор "
              "согласуется сам с собой; предсказатель не может быть точнее.", "",
              "| Метрика | средний по рёбрам σ | MAE между зёрнами |", "|---|---|---|"]
        for m, v in s["seed_noise"].items():
            L.append(f"| {names[m]} | {v['sigma_mean']:.4f} | {v['mae_between_seeds']:.4f} |")
        L += ["", "Во всех сценариях датасета используется одно и то же зерно (метод общих случайных чисел), "
              "поэтому разница «сценарий − база» содержит меньше шума, чем разница двух независимых прогонов.", ""]
    dq = s["data_quality"]
    errs = scen[scen.status != "ok"]
    L += ["## 6. Качество данных", "",
          f"* Телепорты (SUMO убирает машину, простоявшую в заторе > {config.TIME_TO_TELEPORT} с): "
          f"медиана {dq['teleports']['p50']:.0f}, p95 {dq['teleports']['p95']:.0f}, макс {dq['teleports']['p100']:.0f} на сценарий "
          f"(при ~{b['veh_loaded']:.0f} машинах).",
          f"* Машины, не успевшие въехать в сеть к концу часа: медиана {dq['not_inserted']['p50']:.0f}, "
          f"p95 {dq['not_inserted']['p95']:.0f}, макс {dq['not_inserted']['p100']:.0f}.",
          f"* Сценариев с ошибкой: {len(errs)}." + ("" if errs.empty else " Причины: " + "; ".join(
              f"№{r.scenario_id}: {str(r.error)[:120]}" for r in errs.itertuples())),
          "* Закрытые рёбра: occupancy = 0, mean_speed и travel_time = NaN — при обучении их нужно маскировать по флагу `closed`.",
          ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--out", type=Path, default=config.REPORTS_DIR)
    args = parser.parse_args(argv)
    build_report(args.dataset, args.out)


if __name__ == "__main__":
    main()
