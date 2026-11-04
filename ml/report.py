"""Графики и таблицы для ml/reports/ (идут в пояснительную записку).

Оформление: светлая тема под печать, системный sans (DejaVu Sans — с кириллицей),
тонкие линии и сетка-«волосок», категориальные цвета в фиксированном порядке
(синий, оранжевый, бирюзовый; палитра проверена на различимость при
дальтонизме), акцент — синим на выбранной GNN, остальные модели — серым.
Карты изменений — дивергентная шкала синий (снижение) ↔ серый (нет
изменений) ↔ красный (рост).

Пересобрать графики без переобучения:  python -m ml.report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from ml.config import ML_DATA_DIR, REPORTS_DIR, TARGET_UNITS, TARGETS  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY_BAR = "#c3c2b7"
ROAD = "#e1e0d9"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ef9a99", "#a32a2a"])

TARGET_TITLES = {"occupancy": "Занятость", "mean_speed": "Средняя скорость",
                 "travel_time": "Время проезда"}
CHANGE_TITLES = {"close": "Закрытие", "lanes": "Число полос", "speed": "Скорость",
                 "traffic_light": "Светофор", "mixed": "Комбинация"}


def short_title(title: str) -> str:
    """«GNN: итоговая (GraphSAGE + …)» -> «GNN: итоговая» — для заголовков панелей."""
    return title.split(" (")[0] if title.startswith("GNN: итоговая") else title


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 160,
        "lines.linewidth": 2.0,
    })


def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --- обучение ------------------------------------------------------------------------
def plot_training_curves(history: dict, titles: dict, out: Path):
    """(а) абляция: ошибка на валидации от времени обучения (равный бюджет);
    (б) итоговая модель: обучающая и валидационная ошибка по эпохам;
    (в) MAE занятости на валидации от времени."""
    variants = [k for k in history if k != "final"]
    colors = dict(zip(variants, [BLUE, ORANGE, AQUA, "#eda100"]))
    final = history.get("final")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    def switch_epoch(h):
        st = [r.get("stage", "ablation") for r in h]
        return next((h[i] for i in range(len(h)) if st[i] == "final"), None)

    for key in variants:
        h = history[key]
        tmin = [r["time_s"] / 60 for r in h]
        for ax, field, sc in ((axes[0], "val_loss", 1.0), (axes[2], "val_mae_occupancy", 100.0)):
            ax.plot(tmin, [r[field] * sc for r in h], color=colors[key], label=titles.get(key, key))
    if final:
        sw = switch_epoch(final)
        base_key = next((k for k in variants if final[0]["epoch"] == history[k][0]["epoch"]
                         and final[0]["val_loss"] == history[k][0]["val_loss"]), variants[0])
        tail = [r for r in final if r.get("stage") == "final"]
        for ax, field, sc in ((axes[0], "val_loss", 1.0), (axes[2], "val_mae_occupancy", 100.0)):
            prev = history[base_key][-1]
            ax.plot([prev["time_s"] / 60] + [r["time_s"] / 60 for r in tail],
                    [prev[field] * sc] + [r[field] * sc for r in tail], color=colors[base_key])
            if sw:
                ax.axvline(sw["time_s"] / 60, color=AXIS, lw=0.8)
                ax.annotate("начало дообучения\nитоговой модели", (sw["time_s"] / 60, 0),
                            xycoords=("data", "axes fraction"), xytext=(5, 6), textcoords="offset points",
                            va="bottom", fontsize=8.5, color=INK2)
        ax = axes[1]
        ep = [r["epoch"] for r in final]
        ax.plot(ep, [r["train_loss"] for r in final], color=MUTED, label="обучающая выборка")
        ax.plot(ep, [r["val_loss"] for r in final], color=colors[base_key], label="валидационная выборка")
        if sw:
            ax.axvline(sw["epoch"], color=AXIS, lw=0.8)
        best = min(final, key=lambda r: r["val_loss"])
        ax.plot(best["epoch"], best["val_loss"], "o", ms=8, color=colors[base_key], mec="white", mew=2)
        ax.annotate(f"лучшая эпоха {best['epoch']}: {best['val_loss']:.3f}", (best["epoch"], best["val_loss"]),
                    xytext=(-8, 12), textcoords="offset points", ha="right", fontsize=8.5, color=INK)
        ax.set_yscale("log")
        ax.set_title("Итоговая модель: функция потерь")
        ax.set_xlabel("эпоха")
        ax.set_ylabel("MSE стандартизованной Δ")
        ax.legend(loc="upper right")
    else:
        axes[1].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].set_title("Абляция: ошибка на валидации")
    axes[0].set_xlabel("время обучения, мин (CPU)")
    axes[0].set_ylabel("MSE стандартизованной Δ")
    axes[0].legend(loc="upper right")
    for ax in axes[:2]:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[2].set_title("MAE занятости на валидации")
    axes[2].set_xlabel("время обучения, мин (CPU)")
    axes[2].set_ylabel("MAE, п.п. занятости")
    fig.tight_layout()
    _save(fig, out)


# --- сравнение моделей ---------------------------------------------------------------
def _order(metrics: dict, best_gnn: str) -> list[str]:
    base = [k for k in ("nochange", "linear", "gb") if k in metrics]
    gnns = [k for k in metrics if k not in base]
    gnns = sorted(gnns, key=lambda k: k != best_gnn)
    return base + gnns


def plot_comparison(metrics: dict, titles: dict, best_gnn: str, out: Path):
    keys = _order(metrics, best_gnn)
    cols = [("mae", "MAE"), ("rmse", "RMSE"), ("delta_r2", "R² изменения (Δ)")]
    fig, axes = plt.subplots(len(TARGETS), len(cols), figsize=(13, 2.0 + 1.55 * len(TARGETS) * len(keys) / 4),
                             squeeze=False)
    scale = {"occupancy": 100.0, "mean_speed": 1.0, "travel_time": 1.0}
    unit = {"occupancy": "п.п.", "mean_speed": "м/с", "travel_time": "с"}
    y = np.arange(len(keys))[::-1]
    for r, t in enumerate(TARGETS):
        for c, (mkey, mtitle) in enumerate(cols):
            ax = axes[r, c]
            vals = np.array([metrics[k][t][mkey] for k in keys], float)
            if mkey != "delta_r2":
                vals = vals * scale[t]
            colors = [BLUE if k == best_gnn else GRAY_BAR for k in keys]
            ax.barh(y, vals, height=0.62, color=colors, edgecolor="white", linewidth=2)
            ax.set_yticks(y)
            ax.set_yticklabels([titles.get(k, k) for k in keys] if c == 0 else [])
            ax.grid(axis="y", visible=False)
            lo = min(0.0, np.nanmin(vals))
            hi = max(np.nanmax(vals), 0.0)
            span = (hi - lo) or 1.0
            ax.set_xlim(lo - 0.02 * span if lo < 0 else 0, hi + 0.28 * span)
            for yi, v in zip(y, vals):
                fmt = f"{v:.3f}" if mkey == "delta_r2" or abs(v) < 10 else f"{v:.1f}"
                ax.text(max(v, 0) + 0.02 * span, yi, fmt, va="center", fontsize=8.5, color=INK)
            ax.axvline(0, color=AXIS, lw=0.8)
            suffix = "" if mkey == "delta_r2" else f", {unit[t]}"
            ax.set_title(f"{TARGET_TITLES[t]} — {mtitle}{suffix}", fontsize=9.5)
    fig.suptitle("Сравнение моделей на тестовых сценариях (меньше MAE/RMSE и больше R² — лучше)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, out)


def plot_by_change_type(metrics: dict, titles: dict, best_gnn: str, best_base: str, out: Path):
    groups = [g for g in CHANGE_TITLES if g in metrics[best_gnn].get("by_change_type", {})]
    series = [("nochange", GRAY_BAR), (best_base, ORANGE), (best_gnn, BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 + 0.62 * len(groups) * 1.4))
    h = 0.26
    y = np.arange(len(groups))[::-1]
    for ax, (t, sc, label) in zip(axes, [("occupancy", 100, "MAE занятости, п.п."),
                                         ("travel_time", 1, "MAE времени проезда, с")]):
        for s, (key, color) in enumerate(series):
            vals = [metrics[key]["by_change_type"][g][t]["near_mae"] * sc for g in groups]
            ax.barh(y + (1 - s) * h, vals, height=h, color=color, edgecolor="white", linewidth=1.5,
                    label=titles.get(key, key))
        ax.set_yticks(y)
        ax.set_yticklabels([f"{CHANGE_TITLES[g]} (n={metrics[best_gnn]['by_change_type'][g]['n']})"
                            for g in groups])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(label)
        ax.set_title(f"{label.split(',')[0]} рядом с изменениями (≤3 перехода)", fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out)


# --- прогноз vs факт --------------------------------------------------------------
def plot_scatter(cache: dict, titles: dict, out: Path):
    best_gnn, best_base = cache["best_gnn"], cache["best_baseline"]
    tlist = [("occupancy", 100, "Δ занятости, п.п."), ("travel_time", 1, "Δ времени проезда, с")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.2))
    for r, (t, sc, label) in enumerate(tlist):
        yt = cache[f"true_delta_{t}"] * sc
        lim = np.percentile(np.abs(yt), 99.7) * 1.15 or 1.0
        for c, key in enumerate((best_gnn, best_base)):
            ax = axes[r, c]
            yp = cache[f"pred_delta_{t}_{key}"] * sc
            hb = ax.hexbin(yt, yp, gridsize=70, extent=(-lim, lim, -lim, lim), cmap=SEQ_BLUE,
                           norm=LogNorm(), mincnt=1, linewidths=0)
            ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ss_res = np.sum((yp - yt) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            ax.set_title(f"{short_title(titles.get(key, key))}\nR² изменения = {1 - ss_res / ss_tot:.3f}",
                         fontsize=9.5)
            ax.set_xlabel(f"фактическое {label}")
            ax.set_ylabel(f"прогноз {label}")
            cb = fig.colorbar(hb, ax=ax, shrink=0.8)
            cb.set_label("число рёбер × сценариев", color=INK2)
            cb.outline.set_visible(False)
    fig.suptitle("Прогноз изменения против факта (тестовые сценарии; диагональ — идеальный прогноз)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


# --- карта примера ------------------------------------------------------------------
def describe_change(c: dict) -> str:
    t, tid, v = c["type"], c["target_id"], c.get("value")
    if t == "close":
        return f"закрыто {tid}"
    if t == "open":
        return f"открыто {tid}"
    if t == "lanes":
        return f"{tid}: полос {v}"
    if t == "speed":
        return f"{tid}: {float(v) * 3.6:.0f} км/ч"
    if t == "traffic_light":
        on = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "add", "on")
        return f"светофор {tid} {'установлен' if on else 'снят'}"
    return f"{t} {tid}"


def _segments(network: dict):
    nodes = {n["id"]: (n["lon"], n["lat"]) for n in network["nodes"]}
    segs = []
    for e in network["edges"]:
        g = e.get("geometry") or [nodes[e["from"]], nodes[e["to"]]]
        segs.append(np.asarray([p[:2] for p in g], float))
    return segs, nodes


def _offset(segs, d):
    """Сдвиг вправо по ходу движения, чтобы встречные направления не перекрывались."""
    out = []
    for s in segs:
        if len(s) < 2:
            out.append(s)
            continue
        lat0 = np.deg2rad(np.mean(s[:, 1]))
        xy = s.copy()
        xy[:, 0] *= np.cos(lat0)
        v = np.diff(xy, axis=0)
        n = np.stack([v[:, 1], -v[:, 0]], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        nn = np.vstack([n[:1], (n[1:] + n[:-1]) / 2, n[-1:]]) if len(n) > 1 else np.vstack([n, n])
        xy = xy + d * nn
        xy[:, 0] /= np.cos(lat0)
        out.append(xy)
    return out


def plot_example(network: dict, cache: dict, titles: dict, out: Path):
    ex = cache["example"]
    best_gnn, best_base = cache["best_gnn"], cache["best_baseline"]
    segs, nodes = _segments(network)
    span = max(np.ptp(np.concatenate(segs)[:, 0]), np.ptp(np.concatenate(segs)[:, 1]))
    segs = _offset(segs, span * 0.0025)
    base = np.asarray(ex["base"])[:, 0]
    true = np.asarray(ex["true"])[:, 0]
    closed = np.asarray(ex["closed"], bool)
    panels = [("Базовая занятость (без изменений)", base * 100, "seq"),
              ("Факт: изменение занятости", (true - base) * 100, "div"),
              (f"{short_title(titles.get(best_gnn))}: прогноз изменения", (np.asarray(ex["pred"][best_gnn])[:, 0] - base) * 100, "div"),
              (f"{short_title(titles.get(best_base))}: прогноз изменения", (np.asarray(ex["pred"][best_base])[:, 0] - base) * 100, "div")]
    dmax = np.nanpercentile(np.abs(np.nan_to_num(panels[1][1])), 99.5) or 1.0
    edge_ids = [e["id"] for e in network["edges"]]
    eidx = {e: i for i, e in enumerate(edge_ids)}
    changed_edges = {eidx[c["target_id"]] for c in ex["changes"] if c["target_id"] in eidx}
    tl_nodes = [c["target_id"] for c in ex["changes"] if c["type"] == "traffic_light"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    for ax, (title, vals, kind) in zip(axes.ravel(), panels):
        ax.add_collection(LineCollection(segs, colors=ROAD, linewidths=2.2, zorder=1))
        v = np.nan_to_num(vals)
        if kind == "seq":
            norm = plt.Normalize(0, max(np.nanpercentile(v, 99), 1e-6))
            cmap = SEQ_BLUE
        else:
            norm = TwoSlopeNorm(0, -dmax, dmax)
            cmap = DIVERGING
        ok = ~closed
        lc = LineCollection([s for s, o in zip(segs, ok) if o], array=v[ok], cmap=cmap, norm=norm,
                            linewidths=1.8, zorder=2)
        ax.add_collection(lc)
        for i in changed_edges:
            s = segs[i]
            ax.plot(s[:, 0], s[:, 1], color=INK, lw=3.2 if closed[i] else 2.4, zorder=3,
                    solid_capstyle="round")
            if closed[i]:
                mid = s[len(s) // 2] if len(s) > 2 else s.mean(0)
                ax.plot(*mid, marker="X", ms=9, color=INK, mec="white", mew=1.2, zorder=4)
        for nid in tl_nodes:
            if nid in nodes:
                ax.plot(*nodes[nid], marker="^", ms=9, color=INK, mec="white", mew=1.2, zorder=4)
        ax.autoscale()
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean([p[1] for p in nodes.values()]))))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=10)
        cb = fig.colorbar(lc, ax=ax, shrink=0.7, pad=0.01)
        cb.set_label("занятость, %" if kind == "seq" else "Δ занятости, п.п.", color=INK2)
        cb.outline.set_visible(False)
    desc = "; ".join(describe_change(c) for c in ex["changes"])
    handles = [Line2D([], [], color=INK, lw=2.4, label="изменённое ребро"),
               Line2D([], [], color=INK, marker="X", lw=0, ms=9, mec="white", label="закрыто")]
    if tl_nodes:
        handles.append(Line2D([], [], color=INK, marker="^", lw=0, ms=9, mec="white",
                              label="изменён светофор"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Тестовый сценарий {ex['scenario_id']}: {desc}", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out)


# --- инференс ---------------------------------------------------------------------
def plot_inference(bench: dict, out: Path):
    sc = bench["scaling"]
    n = [r["n_edges"] for r in sc]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n, [r["predict_ms_median"] for r in sc], color=BLUE, marker="o", ms=8, mec="white",
            mew=2, label="predict() целиком (признаки + GNN + ответ)")
    ax.plot(n, [r["model_ms_median"] for r in sc], color=ORANGE, marker="o", ms=8, mec="white",
            mew=2, label="только прямой проход GNN")
    d = bench["district"]
    ax.plot([d["n_edges"]], [d["predict_ms_median"]], marker="o", ms=12, color=BLUE, mec=INK, mew=1.2,
            lw=0, zorder=5)
    ax.annotate(f"сеть района: {d['n_edges']} рёбер,\n{d['predict_ms_median']:.0f} мс",
                (d["n_edges"], d["predict_ms_median"]), xytext=(-14, 14), textcoords="offset points",
                ha="right", color=INK, fontsize=9)
    ref = bench.get("reference", {})
    top = max(r["predict_ms_median"] for r in sc)
    for key, label in (("macro_ms_per_scenario", "макромодель (генератор данных), 1 сценарий"),
                       ("sumo_ms_per_run", "SUMO, 1 ч модельного времени")):
        if ref.get(key):
            top = max(top, ref[key])
            ax.axhline(ref[key], color=MUTED, lw=1.0)
            val = f"{ref[key] / 1000:.1f} с" if ref[key] >= 1000 else f"{ref[key]:.0f} мс"
            ax.annotate(f"{label}: {val}", (1.0, ref[key]), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom",
                        color=INK2, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    bottom = min(r["model_ms_median"] for r in sc)
    ax.set_ylim(bottom / 2, top * 3)
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("число участков дорог (рёбер сети)")
    ax.set_ylabel("время на один сценарий, мс (CPU)")
    ax.set_title("Время прогноза в зависимости от размера сети")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out)


# --- markdown -----------------------------------------------------------------------
def write_markdown(rep: Path = REPORTS_DIR):
    s = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
    metrics, titles, best = s["metrics"], s["titles"], s["best_gnn"]
    keys = _order(metrics, best)
    ds = s["dataset"]
    L = ["# Результаты обучения и сравнения моделей", "",
         "_Файл сгенерирован автоматически (`python -m ml.train`)._", "",
         "## Данные", "",
         f"* Источник датасета: **{ds['source']}** (`{ds['path']}`)"
         + (f", генератор: {ds['generator']}" if ds.get("generator") else ""),
         f"* Сеть: {ds.get('network_source') or '—'}; {ds['n_nodes']} узлов, {ds['n_edges']} рёбер, "
         f"{ds['line_graph_edges']} связей line graph",
         f"* Сценариев: {ds['n_scenarios']} (обучение {ds['n_train']} / валидация {ds['n_val']} / "
         f"тест {ds['n_test']}; разбиение по сценариям)", "",
         "## Метрики на тестовых сценариях", "",
         "MAE/RMSE — в физических единицах по всем открытым рёбрам; «R² Δ» — R² *изменения* "
         "относительно базового состояния (насколько модель объясняет эффект правок); "
         "«рядом» — рёбра не дальше 3 переходов от изменений.", ""]
    if s.get("noise_floor_mae"):
        nf = s["noise_floor_mae"]
        L += ["«Шумовой пол» — средняя абсолютная разница между отдельным базовым прогоном и средним "
              "остальных (стохастичность симуляции): ниже этого уровня MAE принципиально не опускается. "
              + ", ".join(f"{TARGET_TITLES[t]}: {nf[t]:.4f}" for t in TARGETS) + ".", ""]
    for t in TARGETS:
        unit = TARGET_UNITS[t]
        L += [f"### {TARGET_TITLES[t]} ({t}, {unit})", "",
              "| Модель | MAE | RMSE | R² | R² Δ | MAE рядом | R² Δ рядом |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for k in keys:
            m = metrics[k][t]
            name = f"**{titles[k]}**" if k == best else titles[k]
            L.append(f"| {name} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} | {m['delta_r2']:.3f} "
                     f"| {m.get('near_mae', float('nan')):.4f} | {m.get('near_delta_r2', float('nan')):.3f} |")
        L.append("")
    L += ["### MAE занятости рядом с изменениями по типу изменения (п.п.)", "",
          "| Тип изменения | n | " + " | ".join(titles[k] for k in keys) + " |",
          "|---|---:|" + "---:|" * len(keys)]
    for g, gt in CHANGE_TITLES.items():
        if g not in metrics[best].get("by_change_type", {}):
            continue
        n = metrics[best]["by_change_type"][g]["n"]
        vals = " | ".join(f"{100 * metrics[k]['by_change_type'][g]['occupancy']['near_mae']:.3f}" for k in keys)
        L.append(f"| {gt} | {n} | {vals} |")
    L += ["", "## Модели", ""]
    for c, g in s["gnn"].items():
        cfg = g["config"]
        dil = ", ".join(map(str, cfg.get("dilations") or [1] * cfg["layers"]))
        L.append(f"* {titles[c]}: {cfg['conv']}, {cfg['layers']} слоёв (связи через {dil} перехода), "
                 f"hidden {cfg['hidden']}, параметров {g['n_params']:,}; "
                 f"эпох {g['epochs']}, обучение {g['train_seconds'] / 60:.1f} мин; "
                 f"val loss {g['val_loss']:.4f}")
    for k, b in s["baselines"].items():
        L.append(f"* {titles[k]}: обучение {b['train_seconds']:.1f} с на {b['train_rows']:,} строках")
    L += [f"* Для predict.py выбрана **{titles[best]}** (минимальная ошибка на валидации); "
          f"поправка нуля: {'да' if s['zero_correction'] else 'нет'}.", ""]
    bench_p = rep / "inference_time.json"
    if bench_p.exists():
        b = json.loads(bench_p.read_text(encoding="utf-8"))
        d = b["district"]
        L += ["## Время инференса (CPU)", "",
              f"Процессор: {b['env']['cpu']}, потоков torch: {b['env']['torch_threads']}.", "",
              f"* Сеть района ({d['n_edges']} рёбер): `predict()` — медиана **{d['predict_ms_median']:.1f} мс**, "
              f"95-й перцентиль {d['predict_ms_p95']:.1f} мс; только GNN — {d['model_ms_median']:.1f} мс; "
              f"холодный старт (загрузка весов + первый вызов) — {d['cold_start_ms']:.0f} мс.",
              ]
        ref = b.get("reference", {})
        if ref.get("macro_ms_per_scenario"):
            L.append(f"* Макромодель-генератор синтетических данных: {ref['macro_ms_per_scenario']:.0f} мс на сценарий.")
        if ref.get("sumo_ms_per_run"):
            L.append(f"* SUMO на той же сети, 1 ч модельного времени ({ref.get('sumo_vehicles', '?')} ТС): "
                     f"{ref['sumo_ms_per_run'] / 1000:.1f} с — GNN быстрее примерно в "
                     f"{ref['sumo_ms_per_run'] / d['predict_ms_median']:.0f} раз.")
        L += ["", "| Рёбер | predict(), мс (медиана) | GNN, мс (медиана) |", "|---:|---:|---:|"]
        for r in b["scaling"]:
            L.append(f"| {r['n_edges']} | {r['predict_ms_median']:.1f} | {r['model_ms_median']:.1f} |")
        L.append("")
    L += ["## Графики", "",
          "* `training_curves.png` — кривые обучения;",
          "* `comparison.png` — сравнение моделей;",
          "* `scatter.png` — прогноз изменения против факта;",
          "* `by_change_type.png` — ошибка по типам изменений;",
          "* `example_scenario.png` — карта тестового сценария (факт и прогнозы);",
          "* `inference_time.png` — время инференса и масштабируемость.", ""]
    (rep / "metrics.md").write_text("\n".join(L), encoding="utf-8")


def load_cache(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    cache = {k: z[k] for k in z.files if k != "meta"}
    cache.update(json.loads(str(z["meta"])))
    return cache


def make_all(rep: Path = REPORTS_DIR, network: dict | None = None,
             cache_path: Path = ML_DATA_DIR / "report_cache.npz"):
    _style()
    s = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
    hist = json.loads((rep / "history.json").read_text(encoding="utf-8"))
    titles, best = s["titles"], s["best_gnn"]
    plot_training_curves(hist, titles, rep / "training_curves.png")
    plot_comparison(s["metrics"], titles, best, rep / "comparison.png")
    if cache_path.exists():
        cache = load_cache(cache_path)
        plot_scatter(cache, titles, rep / "scatter.png")
        plot_by_change_type(s["metrics"], titles, best, cache["best_baseline"], rep / "by_change_type.png")
        if network is not None:
            plot_example(network, cache, titles, rep / "example_scenario.png")
    bench = rep / "inference_time.json"
    if bench.exists():
        plot_inference(json.loads(bench.read_text(encoding="utf-8")), rep / "inference_time.png")


def main():
    from ml.config import WEIGHTS_DIR

    net_path = WEIGHTS_DIR / "network.json"
    network = json.loads(net_path.read_text(encoding="utf-8")) if net_path.exists() else None
    make_all(REPORTS_DIR, network)
    write_markdown(REPORTS_DIR)
    print(f"Отчёты обновлены: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
