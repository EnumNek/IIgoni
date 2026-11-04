"""Обучение GNN и бейзлайнов, оценка на тестовой выборке, сохранение весов и отчётов.

Запуск (из корня репозитория):

    python -m ml.train                 # полный цикл (~1,5 ч на 4 ядрах CPU)
    python -m ml.train --models sage   # без абляции: только основной вариант
    python -m ml.train --quick         # быстрая проверка конвейера (несколько эпох)

Обучение в два этапа:
  1. Абляция: варианты GNN (VARIANTS) обучаются с одинаковым бюджетом времени
     (--max-minutes) — честное сравнение архитектур.
  2. Итоговая модель: лучший по валидации вариант дообучается ещё
     --final-minutes с косинусным затуханием шага обучения; она сохраняется
     в ml/weights/ и используется в predict.py.

Результаты:
    ml/weights/gnn.pt, ml/weights/meta.json — веса лучшей GNN и всё нужное для predict.py
    ml/reports/ — метрики (metrics.json, metrics.md), история обучения, графики,
                  замер времени инференса.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ml import report
from ml.baselines import GBBaseline, LinearBaseline, NoChange, tabular
from ml.config import ML_DATA_DIR, REPO_ROOT, REPORTS_DIR, SEED, SIM_DATA_DIR, SYNTH_DIR, TARGETS, WEIGHTS_DIR
from ml.core import GNNRunner, Normalizer, build_tensors, evaluate, to_physical
from ml.dataset import load_dataset
from ml.features import CONTEXT_NAMES, FEATURE_NAMES, FeatureBuilder
from ml.model import TrafficGNN, batched_graphs

log = logging.getLogger("ml.train")

# Варианты GNN (абляция при равном бюджете времени): тип свёртки и расширения связей по слоям.
VARIANTS = {
    "sage": ("sage", None, "GNN: GraphSAGE + расширенные связи"),
    "sage_local": ("sage", [1, 1, 1, 1, 1, 1], "GNN: GraphSAGE, только соседи"),
    "gat_local": ("gat", [1, 1, 1, 1, 1, 1], "GNN: GATv2, только соседи"),
}
CONV_TITLES = {k: v[2] for k, v in VARIANTS.items()}
FINAL = "final"
CHANGE_GROUPS = {"close": "Закрытие", "lanes": "Число полос", "speed": "Скорость",
                 "traffic_light": "Светофор", "mixed": "Комбинация"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto", choices=["auto", "sim", "synthetic"])
    ap.add_argument("--synth-dir", type=Path, default=SYNTH_DIR)
    ap.add_argument("--models", default="sage,sage_local,gat_local",
                    help=f"варианты GNN через запятую: {', '.join(VARIANTS)}")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=6, help="используется, если --dilations пусто")
    ap.add_argument("--dilations", default="1,1,2,2,4,4",
                    help="расширения связей по слоям (через сколько переходов берутся соседи)")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--max-minutes", type=float, default=15.0,
                    help="бюджет времени обучения на один вариант GNN в абляции (одинаковый для всех)")
    ap.add_argument("--final-minutes", type=float, default=30.0,
                    help="дообучение лучшего варианта (косинусное затухание шага); 0 — не дообучать")
    ap.add_argument("--reuse-ablation", action="store_true",
                    help="не переобучать варианты абляции, если их веса уже есть в ml/data/ablation")
    ap.add_argument("--gb-rows", type=int, default=1_000_000, help="строк для обучения бустинга")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    ap.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    ap.add_argument("--cache-path", type=Path, default=ML_DATA_DIR / "report_cache.npz",
                    help="кэш данных для графиков (не коммитится)")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--quick", action="store_true", help="несколько эпох, без GAT — проверка конвейера")
    args = ap.parse_args(argv)
    args.dilations = [int(d) for d in args.dilations.split(",") if d.strip()] or None
    if args.quick:
        args.models = "sage"
        args.epochs, args.max_minutes, args.gb_rows, args.final_minutes = 3, 2.0, 50_000, 0.05
    return args


# --- обучение GNN -------------------------------------------------------------------
def build_model(name: str, F: int, args) -> TrafficGNN:
    conv_type, dil, _ = VARIANTS[name]
    return TrafficGNN(F, args.hidden, args.layers, conv_type, args.heads, args.dropout,
                      dilations=dil or args.dilations)


def train_gnn(tag: str, model: TrafficGNN, Xn, Dn, M, R_occ, Y_occ, fb: FeatureBuilder, d_std0: float,
              tr: np.ndarray, va: np.ndarray, args, minutes: float, schedule: str = "plateau",
              lr: float | None = None, epoch_offset: int = 0, time_offset: float = 0.0,
              stage: str = "ablation") -> tuple[TrafficGNN, list[dict]]:
    """Обучает model (в т.ч. продолжает обучение) не дольше minutes минут.

    schedule: plateau — ReduceLROnPlateau + ранняя остановка (абляция);
              cosine — шаг плавно уменьшается от lr до 1e-5 за отведённое время
              (дообучение итоговой модели).
    Возвращает модель с лучшими по валидации весами и историю по эпохам."""
    torch.manual_seed(SEED + epoch_offset)
    rng = np.random.default_rng(SEED + epoch_offset)
    S, E, F = Xn.shape
    lr = lr or args.lr
    log.info("[%s] параметров: %d, расширения связей по слоям: %s, бюджет %.0f мин, %s",
             tag, model.n_params(), model.dilations, minutes, schedule)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-5)
    graphs = fb.graphs(model.dilations)
    ei_cache: dict[int, dict[int, dict]] = {}

    def eib(b: int) -> dict[int, dict]:
        if b not in ei_cache:
            ei_cache[b] = batched_graphs(graphs, E, b)
        return ei_cache[b]

    def run(idx):
        out = model(Xn[idx].reshape(-1, F), eib(len(idx)), len(idx)).view(len(idx), E, 3)
        m = M[idx]
        loss = (((out - Dn[idx]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        return out, loss

    @torch.no_grad()
    def validate():
        model.eval()
        tot, n, mae = 0.0, 0.0, 0.0
        for a in range(0, len(va), 32):
            idx = va[a:a + 32]
            out, loss = run(idx)
            m = M[idx]
            tot += float(loss) * float(m.sum())
            n += float(m.sum())
            occ = (R_occ[idx] + out[..., 0] * d_std0).clamp(0, 1)
            mae += float(((occ - Y_occ[idx]).abs() * m).sum())
        return tot / max(n, 1), mae / max(n, 1)

    budget = minutes * 60
    best, best_state, bad, history = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    if epoch_offset:  # продолжение: стартовая точка тоже кандидат в «лучшие»
        best = validate()[0]
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(tr)
        tl, tn = 0.0, 0
        for a in range(0, len(perm), args.batch):
            if schedule == "cosine":
                frac = min((time.time() - t_start) / budget, 1.0)
                for gr in opt.param_groups:
                    gr["lr"] = 1e-5 + 0.5 * (lr - 1e-5) * (1 + np.cos(np.pi * frac))
            idx = perm[a:a + args.batch]
            _, loss = run(idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss) * len(idx)
            tn += len(idx)
        vl, vmae = validate()
        elapsed = time.time() - t_start
        history.append({"epoch": epoch_offset + epoch, "train_loss": tl / tn, "val_loss": vl,
                        "val_mae_occupancy": vmae, "lr": opt.param_groups[0]["lr"],
                        "time_s": time_offset + elapsed, "stage": stage})
        log.info("[%s] эпоха %3d  train %.4f  val %.4f  val MAE occ %.5f  lr %.1e  %.0f с",
                 tag, epoch_offset + epoch, tl / tn, vl, vmae, opt.param_groups[0]["lr"], elapsed)
        if schedule == "plateau":
            sched.step(vl)
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if schedule == "plateau" and bad >= args.patience:
            log.info("[%s] ранняя остановка: %d эпох без улучшения", tag, bad)
            break
        if elapsed > budget:
            log.info("[%s] исчерпан бюджет времени %.0f мин", tag, minutes)
            break
    model.load_state_dict(best_state)
    return model.eval(), history


def change_group(changes: list[dict]) -> str:
    types = {"close" if c["type"] == "open" else c["type"] for c in changes}
    return types.pop() if len(types) == 1 else "mixed"


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.set_num_threads(args.threads)
    t_total = time.time()

    # --- данные ---------------------------------------------------------------------
    ds = load_dataset(args.source, synth_dir=args.synth_dir)
    topo, base = ds.topo, ds.base_state
    log.info("Датасет: %s, сценариев %d (базовых %d), рёбер %d", ds.source, ds.n_scenarios,
             int(ds.is_base.sum()), topo.n_edges)
    fb = FeatureBuilder(topo, base, ds.base_metrics)
    t = time.time()
    T = build_tensors(fb, ds.changes, ds.Y)
    log.info("Признаки: %s за %.1f с", T.X.shape, time.time() - t)
    tr, va, te = ds.split()
    D = np.nan_to_num(T.Yn - T.R)
    norm = Normalizer.fit(T.X[tr], D[tr], T.mask[tr])
    X_base = fb.features(base)
    C_base = fb.context(base)
    near = np.minimum(T.C[..., CONTEXT_NAMES.index("hops_to_change_down")],
                      T.C[..., CONTEXT_NAMES.index("hops_to_change_up")]) <= 3

    Xn = torch.from_numpy(norm.x(T.X))
    Dn = torch.from_numpy((D / norm.d_std).astype(np.float32))
    M = torch.from_numpy(T.mask.astype(np.float32))
    R_occ = torch.from_numpy(T.R[..., 0].copy())
    Y_occ = torch.from_numpy(np.nan_to_num(T.Yn[..., 0]))

    def val_loss_norm(delta_norm: np.ndarray, idx: np.ndarray) -> float:
        """MSE стандартизованной Δ на сценариях idx (как функция потерь GNN)."""
        err = (delta_norm - D[idx]) / norm.d_std
        m = T.mask[idx]
        return float(((err ** 2).mean(-1) * m).sum() / m.sum())

    # --- GNN ------------------------------------------------------------------------
    gnn = {}
    histories = {}
    F = T.X.shape[-1]
    d0 = float(norm.d_std[0])
    abl_dir = ML_DATA_DIR / "ablation"

    def register(key: str, model: TrafficGNN, hist: list[dict], seconds: float):
        runner = GNNRunner(model, norm, zero_correction=False)
        g = fb.graphs(runner.dilations)
        d_va = runner.raw_delta(T.X[va], g)
        dz = runner.raw_delta(X_base[None], g)[0]
        v_plain, v_corr = val_loss_norm(d_va, va), val_loss_norm(d_va - dz, va)
        runner.zero_correction = v_corr <= v_plain
        histories[key] = hist
        gnn[key] = {"model": model, "runner": runner, "val_loss": min(v_plain, v_corr),
                    "val_loss_plain": v_plain, "val_loss_zero_corr": v_corr,
                    "train_seconds": seconds, "epochs": len(hist)}
        log.info("[%s] val loss %.4f (с поправкой нуля %.4f), обучение %.0f с", key, v_plain, v_corr, seconds)

    # Этап 1: абляция — все варианты с одинаковым бюджетом времени
    for name in [c.strip() for c in args.models.split(",") if c.strip()]:
        if name not in VARIANTS:
            raise SystemExit(f"Неизвестный вариант модели {name!r}; допустимы: {', '.join(VARIANTS)}")
        model = build_model(name, F, args)
        ckpt = abl_dir / f"{name}.pt"
        tag = {"config": model.config, "minutes": args.max_minutes, "n_scenarios": ds.n_scenarios,
               "features": FEATURE_NAMES}
        if args.reuse_ablation and ckpt.exists():
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
            if saved["tag"] == tag:
                model.load_state_dict(saved["state_dict"])
                log.info("[%s] веса абляции загружены из %s", name, ckpt)
                register(name, model.eval(), saved["history"], saved["seconds"])
                continue
        t = time.time()
        model, hist = train_gnn(name, model, Xn, Dn, M, R_occ, Y_occ, fb, d0, tr, va, args,
                                minutes=args.max_minutes)
        register(name, model, hist, time.time() - t)
        abl_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"tag": tag, "state_dict": model.state_dict(), "history": hist,
                    "seconds": time.time() - t}, ckpt)
    best_variant = min(gnn, key=lambda c: gnn[c]["val_loss"])
    log.info("Лучший вариант абляции по валидации: %s", best_variant)

    # Этап 2: дообучение лучшего варианта с косинусным затуханием шага
    best_conv = best_variant
    if args.final_minutes > 0:
        model = build_model(best_variant, F, args)
        model.load_state_dict(gnn[best_variant]["model"].state_dict())
        h0 = histories[best_variant]
        t = time.time()
        model, hist = train_gnn(FINAL, model, Xn, Dn, M, R_occ, Y_occ, fb, d0, tr, va, args,
                                minutes=args.final_minutes, schedule="cosine", lr=args.lr / 2,
                                epoch_offset=h0[-1]["epoch"], time_offset=h0[-1]["time_s"],
                                stage="final")
        register(FINAL, model, h0 + hist, gnn[best_variant]["train_seconds"] + time.time() - t)
        best_conv = FINAL

    # --- бейзлайны ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    Xtab_tr = tabular(norm.x(T.X[tr]), T.C[tr])
    ytab_tr = (D[tr] / norm.d_std).reshape(-1, 3)
    rows = np.nonzero(T.mask[tr].ravel())[0]
    tab_base = tabular(norm.x(X_base[None]), C_base[None])
    baselines = {"nochange": NoChange(), "linear": LinearBaseline(), "gb": GBBaseline()}
    base_info = {}
    for key, mdl in baselines.items():
        t = time.time()
        r = rows if key != "gb" or len(rows) <= args.gb_rows else np.sort(rng.choice(rows, args.gb_rows, replace=False))
        mdl.fit(Xtab_tr[r], ytab_tr[r])
        base_info[key] = {"train_seconds": time.time() - t, "train_rows": int(len(r))}
        log.info("Бейзлайн %s: обучен за %.1f с на %d строках", mdl.name, time.time() - t, len(r))
    del Xtab_tr, ytab_tr

    def tab_delta(mdl, idx):
        d = mdl.predict(tabular(norm.x(T.X[idx]), T.C[idx])).reshape(len(idx), topo.n_edges, 3) * norm.d_std
        if gnn[best_conv]["runner"].zero_correction and not isinstance(mdl, NoChange):
            d = d - (mdl.predict(tab_base) * norm.d_std)[None]
        return d

    # --- оценка на тесте -------------------------------------------------------------
    models_delta: dict[str, np.ndarray] = {}
    inference_s: dict[str, float] = {}
    for key, mdl in baselines.items():
        t = time.time()
        models_delta[key] = tab_delta(mdl, te)
        inference_s[key] = (time.time() - t) / len(te)
    for conv, g in gnn.items():
        t = time.time()
        models_delta[conv] = g["runner"].predict_norm(T.X[te], np.zeros_like(T.R[te]),
                                                      fb.graphs(g["runner"].dilations), X_base)
        inference_s[conv] = (time.time() - t) / len(te)

    final_title = (f"GNN: итоговая ({VARIANTS[best_variant][2].split(': ', 1)[1]}, "
                   f"+{args.final_minutes:g} мин)")
    titles = {"nochange": NoChange.name, "linear": LinearBaseline.name, "gb": GBBaseline.name,
              **{c: CONV_TITLES.get(c, c) for c in gnn if c != FINAL}, FINAL: final_title}
    Y_te, mask_te, near_te = ds.Y[te].astype(np.float64), T.mask[te], near[te]
    groups = np.array([change_group(ds.changes[i]) for i in te])
    metrics, phys = {}, {}
    for key, d in models_delta.items():
        if key == "nochange":  # чистый бейзлайн: метрики ровно как в базовом прогоне
            phys[key] = np.broadcast_to(ds.base_metrics, (len(te), topo.n_edges, 3)).copy()
        else:
            phys[key] = to_physical(T.R[te] + d, topo, T, te, base)
        metrics[key] = evaluate(Y_te, phys[key], mask_te, ds.base_metrics, near_te)
        by_group = {}
        for gname in CHANGE_GROUPS:
            sel = groups == gname
            if sel.sum() == 0:
                continue
            gm = evaluate(Y_te[sel], phys[key][sel], mask_te[sel], ds.base_metrics, near_te[sel])
            by_group[gname] = {"n": int(sel.sum()), **{t: gm[t] for t in TARGETS}}
        metrics[key]["by_change_type"] = by_group
        log.info("%-22s occ MAE %.4f R² %.3f ΔR² %.3f | speed MAE %.3f | tt MAE %.2f ΔR² %.3f",
                 titles[key], metrics[key]["occupancy"]["mae"], metrics[key]["occupancy"]["r2"],
                 metrics[key]["occupancy"]["delta_r2"], metrics[key]["mean_speed"]["mae"],
                 metrics[key]["travel_time"]["mae"], metrics[key]["travel_time"]["delta_r2"])

    # --- сохранение весов -----------------------------------------------------------
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    best = gnn[best_conv]
    torch.save({"state_dict": best["model"].state_dict(), "config": best["model"].config},
               args.weights_dir / "gnn.pt")

    def rel(p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    meta = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"variant": best_variant, "conv": VARIANTS[best_variant][0],
                  "title": titles[best_conv], "final_minutes": args.final_minutes,
                  "config": best["model"].config, "n_params": best["model"].n_params()},
        "feature_names": FEATURE_NAMES,
        "targets": list(TARGETS),
        "normalizer": norm.to_dict(),
        "zero_correction": bool(best["runner"].zero_correction),
        "dataset": {"source": ds.source, "path": rel(ds.path), "n_scenarios": int(ds.n_scenarios),
                    "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                    "generator": ds.meta.get("generator"),
                    "network_source": ds.network.get("meta", {}).get("source")},
        # Эталонная сеть, на которой обучена модель: predict() сравнивает с ней
        # переданную сеть, а базовые метрики берёт отсюда.
        "reference": {
            "signature": topo.signature(),
            "edge_ids": topo.edge_ids,
            "lanes": base.lanes.tolist(), "speed": base.speed.tolist(),
            "closed": base.closed.astype(int).tolist(),
            "base_metrics": np.round(ds.base_metrics, 5).tolist(),
            "node_ids": topo.node_ids, "tl": base.tl.astype(int).tolist(),
        },
        "test_metrics": {k: {t: metrics[best_conv][t] for t in TARGETS} for k in [best_conv]},
    }
    with open(args.weights_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    # копия сети, на которой обучена модель (CLI predict и тесты работают «из коробки»)
    with open(args.weights_dir / "network.json", "w", encoding="utf-8") as f:
        json.dump(ds.network, f, ensure_ascii=False)
    log.info("Веса сохранены: %s", args.weights_dir)

    # --- отчёты ---------------------------------------------------------------------
    rep = args.reports_dir
    rep.mkdir(parents=True, exist_ok=True)
    # «Шумовой пол»: насколько отдельный базовый прогон отличается от среднего остальных
    # (стохастичность симуляции) — ниже этой ошибки прогноз опуститься не может.
    noise = None
    Yb = ds.Y[ds.is_base].astype(np.float64)
    if len(Yb) >= 3:
        errs = [np.nanmean(np.abs(Yb[i] - np.nanmean(np.delete(Yb, i, 0), 0)), 0) for i in range(len(Yb))]
        noise = dict(zip(TARGETS, np.mean(errs, 0).tolist()))
    summary = {
        "noise_floor_mae": noise,
        "dataset": meta["dataset"] | {"n_edges": topo.n_edges, "n_nodes": topo.n_nodes,
                                      "line_graph_edges": int(fb.lg.shape[1]),
                                      "dilated_edges": {str(d): int(ei.shape[1]) for d, ei in
                                                        fb.graphs(gnn[best_conv]["runner"].dilations).items()},
                                      "sim_dir": rel(SIM_DATA_DIR)},
        "best_gnn": best_conv,
        "best_variant": best_variant,
        "zero_correction": meta["zero_correction"],
        "titles": titles,
        "metrics": metrics,
        "gnn": {c: {k: v for k, v in g.items() if k not in ("model", "runner")} | {
            "n_params": g["model"].n_params(), "config": g["model"].config} for c, g in gnn.items()},
        "baselines": base_info,
        "batch_inference_seconds_per_scenario": inference_s,
        "hyperparameters": {k: (rel(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "total_seconds": time.time() - t_total,
    }
    with open(rep / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(rep / "history.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=1)

    # данные для графиков (кэш, не коммитится): Δ по тесту и пример сценария
    best_base = min(("linear", "gb"), key=lambda k: metrics[k]["occupancy"]["mae"])
    cache = {"best_gnn": best_conv, "best_baseline": best_base}
    sel = mask_te
    bm = np.broadcast_to(ds.base_metrics, Y_te.shape)
    sub = rng.choice(int(sel.sum()), size=min(int(sel.sum()), 300_000), replace=False)
    for j, tname in enumerate(TARGETS):
        cache[f"true_delta_{tname}"] = (Y_te[..., j] - bm[..., j])[sel][sub]
        for key in (best_conv, best_base):
            cache[f"pred_delta_{tname}_{key}"] = (phys[key][..., j] - bm[..., j])[sel][sub]
    # пример: сценарий с закрытием, где эффект на сеть наибольший
    effect = np.nansum(np.abs(Y_te[..., 0] - ds.base_metrics[None, :, 0]) * mask_te, axis=1)
    cand = [k for k, i in enumerate(te) if groups[k] in ("close", "mixed")
            and any(c["type"] == "close" for c in ds.changes[i]) and len(ds.changes[i]) <= 3]
    k_ex = max(cand, key=lambda k: effect[k]) if cand else int(np.argmax(effect))
    cache["example"] = {
        "scenario_id": int(ds.scenario_ids[te[k_ex]]), "changes": ds.changes[te[k_ex]],
        "true": Y_te[k_ex].tolist(), "base": ds.base_metrics.tolist(),
        "pred": {key: phys[key][k_ex].tolist() for key in (best_conv, best_base)},
        "closed": T.closed[te[k_ex]].astype(int).tolist(),
    }
    cache_path = args.cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{k: v for k, v in cache.items() if isinstance(v, np.ndarray)},
                        meta=json.dumps({k: v for k, v in cache.items() if not isinstance(v, np.ndarray)}))
    try:
        report.make_all(rep, ds.network, cache_path)
    except Exception:  # графики не должны ронять обучение — веса уже сохранены
        log.exception("Не удалось построить графики")

    if not args.skip_benchmark:
        try:
            from ml import benchmark
            benchmark.run(weights_dir=args.weights_dir, reports_dir=rep, network=ds.network,
                          synth_meta=ds.meta, quick=args.quick)
        except Exception:
            log.exception("Не удалось выполнить замер инференса")
    try:
        report.write_markdown(rep)
    except Exception:
        log.exception("Не удалось записать metrics.md")
    log.info("Готово за %.1f мин", (time.time() - t_total) / 60)


if __name__ == "__main__":
    main()
