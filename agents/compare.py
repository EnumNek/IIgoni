"""Сравнение режимов управления: фиксированный цикл / actuated / RL-агенты.

    python -m agents.compare                    # все сети, все уровни спроса
    python -m agents.compare --plots-only       # перерисовать графики из results.csv

Для каждой сети, уровня спроса (низкий / номинальный / высокий) и
тестового семени спроса (101–103, агенты их не видели) прогоняются все три
режима на ОДНОМ И ТОМ ЖЕ файле маршрутов и с одним семенем SUMO — режимы
отличаются только логикой светофоров.

Результаты (agents/reports/):
* results.csv — метрики каждого прогона;
* summary.csv, summary.md — итоговая таблица (среднее ± ст. откл.,
  изменение относительно фиксированного цикла);
* fig_metrics_nominal.png — три метрики при номинальном спросе;
* fig_demand_levels.png — как метрики меняются с ростом нагрузки;
* fig_queue_timeseries.png — суммарная очередь в сети во времени;
* fig_training.png — кривые обучения;
* fig_tls_map_<сеть>.png — средняя очередь на каждом перекрёстке.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import config  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sumolib  # noqa: E402

from agents.env import MODES  # noqa: E402
from agents.evaluate import run_episode  # noqa: E402
from agents.networks import default_scenarios, get_scenario  # noqa: E402

REPORTS = config.REPORTS_DIR
MODE_NAMES = {"fixed": "Фиксированный цикл", "actuated": "Actuated (SUMO)", "rl": "RL-агенты (DQN)"}
LEVEL_NAMES = {"low": "низкий", "nominal": "номинальный", "high": "высокий"}
METRICS = {
    "avg_travel_time": ("Среднее время в пути", "с"),
    "avg_waiting_time": ("Среднее время ожидания", "с"),
    "avg_queue_length": ("Средняя очередь на перекрёсток", "авт."),
    "avg_time_loss": ("Средние потери времени", "с"),
    "avg_depart_delay": ("Средняя задержка въезда", "с"),
    "p95_waiting_time": ("95-й процентиль ожидания", "с"),
    "throughput": ("Завершённых поездок", "авт."),
    "teleports": ("Телепортаций (заторов)", "шт."),
}
MAIN_METRICS = ("avg_travel_time", "avg_waiting_time", "avg_queue_length")

# Оформление (палитра проверена на различимость при дальтонизме:
# первые три категориальных цвета, светлая тема)
COLORS = {"fixed": "#2a78d6", "actuated": "#eb6834", "rl": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


# ---------------------------------------------------------------------------
# Прогоны
# ---------------------------------------------------------------------------
def run_all(specs: list[str], levels: list[str], seeds: list[int]) -> pd.DataFrame:
    from agents.dqn import IndependentDQN

    rows, series = [], []
    for spec in specs:
        sc = get_scenario(spec)
        agents = IndependentDQN.load(sc.models_dir)
        for level in levels:
            for seed in seeds:
                route = sc.eval_route(level, seed)
                for mode in MODES:
                    keep = level == "nominal" and seed == seeds[0]
                    r = run_episode(sc, route, mode, seed=seed, agents=agents if mode == "rl" else None,
                                    keep_series=keep)
                    row = {"scenario": sc.key, "demand": level, "seed": seed, "mode": mode,
                           **{k: r[k] for k in METRICS}, "vehicles": r["vehicles"],
                           "p95_travel_time": float(np.percentile(r["travel_times"], 95)),
                           "guard_rate": r["guard_rate"],
                           "wall_time_s": round(r["wall_time"], 1)}
                    rows.append(row)
                    for ts, q in r["per_tls_queue"].items():
                        series.append({"scenario": sc.key, "demand": level, "seed": seed, "mode": mode,
                                       "kind": "tls", "key": ts, "value": q})
                    if keep:
                        series += [{"scenario": sc.key, "demand": level, "seed": seed, "mode": mode,
                                    "kind": "time", "key": t, "value": v} for t, v in r["queue_series"]]
                    print(f"[{sc.key}] {level:8s} s{seed} {mode:8s}: в пути {r['avg_travel_time']:6.1f} с, "
                          f"ожидание {r['avg_waiting_time']:6.1f} с, очередь {r['avg_queue_length']:5.2f}, "
                          f"телепорт. {r['teleports']}", flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "results.csv", index=False)
    pd.DataFrame(series).to_csv(REPORTS / "details.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Таблицы
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [*METRICS, "p95_travel_time", "guard_rate"]
    g = df.groupby(["scenario", "demand", "mode"])[metrics]
    mean, std = g.mean(), g.std(ddof=0)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std")).reset_index()
    for m in metrics:
        base = out[out["mode"] == "fixed"].set_index(["scenario", "demand"])[f"{m}_mean"]
        ref = np.asarray(out.set_index(["scenario", "demand"]).index.map(base.get), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{m}_vs_fixed_pct"] = np.where(ref > 0, (out[f"{m}_mean"] - ref) / ref * 100, np.nan)
    order = {"low": 0, "nominal": 1, "high": 2}
    out["_d"] = out["demand"].map(order)
    out["_m"] = out["mode"].map({m: i for i, m in enumerate(MODES)})
    out = out.sort_values(["scenario", "_d", "_m"]).drop(columns=["_d", "_m"])
    out.to_csv(REPORTS / "summary.csv", index=False)
    return out


def _fmt_pct(v: float) -> str:
    return "—" if not np.isfinite(v) or abs(v) < 0.05 else f"{v:+.1f}%"


def write_markdown(summary: pd.DataFrame, df: pd.DataFrame) -> None:
    lines = ["# Сравнение режимов управления светофорами", "",
             "Сгенерировано `python -m agents.compare`. Среднее ± стандартное отклонение по "
             f"{df['seed'].nunique()} тестовым вариантам спроса (семена "
             f"{', '.join(map(str, sorted(df['seed'].unique())))}; агенты их не видели при обучении). "
             "В скобках — изменение относительно фиксированного цикла (меньше — лучше).", ""]
    for sc_key, part in summary.groupby("scenario", sort=False):
        lines += [f"## Сеть `{sc_key}`", "",
                  "| Спрос | Режим | Время в пути, с | Ожидание, с | Очередь, авт. | Потери времени, с "
                  "| 95-й процентиль ожидания, с | Поездок | Телепорт. |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in part.iterrows():
            def cell(m: str, digits: int = 1) -> str:
                txt = f"{r[f'{m}_mean']:.{digits}f} ± {r[f'{m}_std']:.{digits}f}"
                if r["mode"] != "fixed":
                    txt += f" ({_fmt_pct(r[f'{m}_vs_fixed_pct'])})"
                return txt
            name = MODE_NAMES[r["mode"]]
            name = f"**{name}**" if r["mode"] == "rl" else name
            lines.append(f"| {LEVEL_NAMES.get(r['demand'], r['demand'])} | {name} | {cell('avg_travel_time')} "
                         f"| {cell('avg_waiting_time')} | {cell('avg_queue_length', 2)} | {cell('avg_time_loss')} "
                         f"| {cell('p95_waiting_time')} | {r['throughput_mean']:.0f} | {r['teleports_mean']:.1f} |")
        lines.append("")
    # Краткий вывод по номинальному спросу
    lines += ["## Итог при номинальном спросе", ""]
    nom = summary[summary["demand"] == "nominal"]
    for sc_key, part in nom.groupby("scenario", sort=False):
        p = part.set_index("mode")
        lines.append(
            f"* `{sc_key}`: RL-агенты сокращают среднее время в пути на "
            f"{-p.loc['rl', 'avg_travel_time_vs_fixed_pct']:.1f}% и ожидание на "
            f"{-p.loc['rl', 'avg_waiting_time_vs_fixed_pct']:.1f}% относительно фиксированного цикла "
            f"(actuated: {-p.loc['actuated', 'avg_travel_time_vs_fixed_pct']:.1f}% и "
            f"{-p.loc['actuated', 'avg_waiting_time_vs_fixed_pct']:.1f}%); "
            f"средняя очередь {p.loc['fixed', 'avg_queue_length_mean']:.2f} → "
            f"{p.loc['rl', 'avg_queue_length_mean']:.2f} авт.")
    rl = summary[summary["mode"] == "rl"]
    lines += ["", "Защитное правило максимального ожидания (90 с) скорректировало решения RL-агентов: "
              + "; ".join(f"`{r['scenario']}` / {LEVEL_NAMES.get(r['demand'], r['demand'])} — "
                          f"{100 * r['guard_rate_mean']:.3f}%" for _, r in rl.iterrows()) + "."]
    # Обучение
    lines += ["", "## Обучение", "", "| Сеть | Агентов | Эпизодов | Лучший эпизод | Время обучения, мин |",
              "|---|---|---|---|---|"]
    for sc_key in summary["scenario"].unique():
        meta_f = config.MODELS_DIR / sc_key / "meta.json"
        log_f = config.MODELS_DIR / sc_key / "training_log.csv"
        if meta_f.exists() and log_f.exists():
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            log = pd.read_csv(log_f)
            minutes = log["episode_time_s"].sum() / 60
            lines.append(f"| `{sc_key}` | {len(meta['agents'])} | {meta['episodes']} | {meta['best_episode']} "
                         f"| {minutes:.0f} (без валидации) |")
    lines += ["", "Графики: `fig_metrics_nominal.png`, `fig_demand_levels.png`, `fig_queue_timeseries.png`, "
              "`fig_training.png`, `fig_tls_map_<сеть>.png`.", ""]
    (REPORTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------
def _layout_with_legend(fig, title: str | None = None, modes=MODES) -> None:
    """Заголовок и легенда — отдельной полосой над графиками, без наложений."""
    band = (0.75 if title else 0.45) / fig.get_figheight()  # высота полосы: дюймы -> доля фигуры
    fig.tight_layout(rect=(0, 0, 1, 1 - band))
    handles = [plt.Line2D([], [], color=COLORS[m], lw=6, solid_capstyle="butt") for m in modes]
    fig.legend(handles, [MODE_NAMES[m] for m in modes], loc="lower left", ncol=len(modes),
               bbox_to_anchor=(0.0, 1 - band + 0.02 / fig.get_figheight()), fontsize=10, labelcolor=INK2, handlelength=1.2)
    if title:
        fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)


def plot_metrics_nominal(summary: pd.DataFrame) -> None:
    nom = summary[summary["demand"] == "nominal"]
    scenarios = list(nom["scenario"].unique())
    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=(4.2 * len(MAIN_METRICS), 3.9))
    width = 0.24
    for ax, m in zip(np.atleast_1d(axes), MAIN_METRICS):
        x = np.arange(len(scenarios))
        top = 0.0
        for j, mode in enumerate(MODES):
            vals = [nom[(nom.scenario == s) & (nom["mode"] == mode)][f"{m}_mean"].iloc[0] for s in scenarios]
            errs = [nom[(nom.scenario == s) & (nom["mode"] == mode)][f"{m}_std"].iloc[0] for s in scenarios]
            pos = x + (j - 1) * (width + 0.02)
            ax.bar(pos, vals, width, color=COLORS[mode], zorder=2)
            ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=0, zorder=3)
            for p, v, e in zip(pos, vals, errs):
                ax.text(p, v + e + 0.01 * max(vals), f"{v:.1f}" if v < 100 else f"{v:.0f}",
                        ha="center", va="bottom", fontsize=8, color=INK2)
            top = max(top, max(v + e for v, e in zip(vals, errs)))
        ax.set_ylim(0, top * 1.15)
        ax.set_xticks(x, [s.replace("grid", "решётка ") for s in scenarios])
        ax.tick_params(axis="x", length=0)
        title, unit = METRICS[m]
        ax.set_title(f"{title}, {unit}")
    _layout_with_legend(fig, "Номинальный спрос: меньше — лучше")
    fig.savefig(REPORTS / "fig_metrics_nominal.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_demand_levels(summary: pd.DataFrame) -> None:
    scenarios = list(summary["scenario"].unique())
    levels = [lv for lv in ("low", "nominal", "high") if lv in set(summary["demand"])]
    metrics = ("avg_waiting_time", "avg_travel_time")
    fig, axes = plt.subplots(len(metrics), len(scenarios), figsize=(5.2 * len(scenarios), 3.3 * len(metrics)),
                             squeeze=False)
    for i, m in enumerate(metrics):
        for j, s in enumerate(scenarios):
            ax = axes[i][j]
            ends = []
            for mode in MODES:
                part = summary[(summary.scenario == s) & (summary["mode"] == mode)].set_index("demand").loc[levels]
                y = part[f"{m}_mean"].values
                ax.plot(range(len(levels)), y, color=COLORS[mode], marker="o", ms=6,
                        mec=SURFACE, mew=2, zorder=3)
                ends.append(y[-1])
            ax.set_xticks(range(len(levels)), [LEVEL_NAMES[lv] for lv in levels])
            ax.set_xlim(-0.2, len(levels) - 0.6)
            ax.set_ylim(bottom=0)
            # подписи концов линий — только если они не слипаются (иначе
            # значения несут легенда и таблица summary.md)
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            for k, v in enumerate(ends):
                if all(abs(v - w) > 0.05 * span for n, w in enumerate(ends) if n != k):
                    ax.annotate(f"{v:.0f}", (len(levels) - 1, v), xytext=(8, 0), textcoords="offset points",
                                va="center", fontsize=8, color=INK2)
            ax.tick_params(axis="x", length=0)
            title, unit = METRICS[m]
            ax.set_title(f"{s.replace('grid', 'Решётка ')}: {title.lower()}, {unit}")
    _layout_with_legend(fig, "Метрики при разном уровне спроса: меньше — лучше")
    fig.savefig(REPORTS / "fig_demand_levels.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_queue_timeseries(details: pd.DataFrame) -> None:
    ts = details[details["kind"] == "time"].copy()
    if ts.empty:
        return
    ts["key"] = ts["key"].astype(float)
    scenarios = list(ts["scenario"].unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.0 * len(scenarios), 3.6), squeeze=False)
    for ax, s in zip(axes[0], scenarios):
        for mode in MODES:
            part = ts[(ts.scenario == s) & (ts["mode"] == mode)].sort_values("key")
            # скользящее среднее за 2.5 мин, чтобы читался тренд, а не шум
            y = part["value"].rolling(30, min_periods=1, center=True).mean()
            ax.plot(part["key"] / 60, y, color=COLORS[mode])
        ax.axvline(config.DEMAND_SECONDS / 60, color=AXIS, lw=1)
        ax.text(config.DEMAND_SECONDS / 60, ax.get_ylim()[1] * 0.97, " конец спроса", fontsize=8,
                color=MUTED, va="top")
        ax.set_xlabel("время симуляции, мин")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{s.replace('grid', 'Решётка ')}: стоящие ТС на подходах, авт.")
    _layout_with_legend(fig, "Суммарная очередь в сети во времени (номинальный спрос, "
                             "скользящее среднее 2.5 мин)")
    fig.savefig(REPORTS / "fig_queue_timeseries.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training(scenarios: list[str]) -> None:
    logs = [(s, config.MODELS_DIR / s / "training_log.csv", config.MODELS_DIR / s / "meta.json")
            for s in scenarios]
    logs = [x for x in logs if x[1].exists() and x[2].exists()]
    if not logs:
        return
    fig, axes = plt.subplots(1, len(logs), figsize=(6.0 * len(logs), 3.6), squeeze=False)
    for ax, (s, log_f, meta_f) in zip(axes[0], logs):
        log = pd.read_csv(log_f)
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        base = meta.get("baseline_validation", {})
        ax.plot(log["episode"], log["train_avg_queue"], color=COLORS["rl"], alpha=0.35, lw=1.5)
        val = log.dropna(subset=["val_avg_queue"])
        ax.plot(val["episode"], val["val_avg_queue"], color=COLORS["rl"], marker="o", ms=5, mec=SURFACE, mew=1.5)
        for mode in ("fixed", "actuated"):
            if mode in base:
                v = base[mode]["avg_queue_length"]
                ax.axhline(v, color=COLORS[mode], lw=1.2)
                ax.text(log["episode"].max(), v, f" {MODE_NAMES[mode]}", va="bottom", ha="right",
                        fontsize=8, color=INK2)
        ax.set_yscale("log")
        ticks = [t for t in (0.5, 1, 2, 3, 5, 10, 20, 30, 50) if t <= max(log["train_avg_queue"].max(), 1) * 1.2]
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("эпизод обучения")
        ax.set_title(f"{s.replace('grid', 'Решётка ')}: средняя очередь на перекрёсток, авт.")
        ax.text(0.99, 0.97, "светлая линия — обучение (с исследованием ε)\nточки — валидация (ε = 0); лог. шкала",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_training.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_tls_maps(details: pd.DataFrame) -> None:
    tls = details[(details["kind"] == "tls") & (details["demand"] == "nominal")]
    for s in tls["scenario"].unique():
        sc = get_scenario(s)
        net = sumolib.net.readNet(str(sc.net_file))
        part = tls[tls.scenario == s].groupby(["mode", "key"])["value"].mean()
        vmax = float(part.max()) or 1.0
        fig, axes = plt.subplots(1, len(MODES), figsize=(4.0 * len(MODES), 4.2))
        for ax, mode in zip(axes, MODES):
            for e in net.getEdges(withInternal=False):
                xs, ys = zip(*e.getShape())
                ax.plot(xs, ys, color=GRID, lw=2, zorder=1, solid_capstyle="butt")
            pts = [(net.getNode(ts).getCoord(), part.get((mode, ts), np.nan))
                   for ts in part.loc[mode].index if ts in {n.getID() for n in net.getNodes()}]
            if pts:
                (xy, vals) = zip(*pts)
                xs, ys = zip(*xy)
                sca = ax.scatter(xs, ys, c=vals, cmap="Blues", vmin=0, vmax=vmax, s=340,
                                 edgecolors=SURFACE, linewidths=2, zorder=3)
                if len(pts) <= 25:
                    for x, y, v in zip(xs, ys, vals):
                        ax.text(x, y, f"{v:.1f}", ha="center", va="center", fontsize=7,
                                color="#ffffff" if v > 0.55 * vmax else INK, zorder=4)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(MODE_NAMES[mode])
        fig.colorbar(sca, ax=axes, shrink=0.7, label="средняя очередь, авт.")
        fig.suptitle(f"{s.replace('grid', 'Решётка ')}: средняя очередь на каждом перекрёстке, авт. (номинальный спрос)", x=0.01, ha="left",
                     fontsize=12, fontweight="bold", color=INK)
        fig.savefig(REPORTS / f"fig_tls_map_{s}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def make_reports(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(REPORTS / "results.csv")
    details_f = REPORTS / "details.csv"
    details = pd.read_csv(details_f) if details_f.exists() else pd.DataFrame(columns=["kind"])
    summary = summarize(df)
    write_markdown(summary, df)
    plot_metrics_nominal(summary)
    plot_demand_levels(summary)
    plot_queue_timeseries(details)
    plot_training(list(summary["scenario"].unique()))
    plot_tls_maps(details)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Сравнение fixed / actuated / RL и отчёты")
    ap.add_argument("--scenario", nargs="+", default=None)
    ap.add_argument("--levels", nargs="+", default=list(config.EVAL_DEMAND_LEVELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.EVAL_SEEDS))
    ap.add_argument("--plots-only", action="store_true", help="только перерисовать отчёты из results.csv")
    args = ap.parse_args()
    df = None if args.plots_only else run_all(args.scenario or default_scenarios(), args.levels, args.seeds)
    make_reports(df)
    print((REPORTS / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
