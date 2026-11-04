"""Общая логика обучения и инференса: подготовка тензоров, нормализация, метрики.

Используется и в train.py, и в predict.py — чтобы преобразования признаков
при обучении и в рабочем режиме гарантированно совпадали.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ml.features import BINARY_FEATURES, FEATURE_NAMES, FeatureBuilder, inverse_targets, transform_targets
from ml.model import TrafficGNN, batched_graphs
from ml.network import State, Topology, apply_changes


@dataclass
class Normalizer:
    """Стандартизация признаков и масштаб целевых Δ (оцениваются на train)."""

    x_mean: np.ndarray
    x_std: np.ndarray
    d_std: np.ndarray  # (3,) std изменения нормированных целей

    @classmethod
    def fit(cls, X: np.ndarray, D: np.ndarray, mask: np.ndarray) -> "Normalizer":
        rows = X[mask]
        mean, std = rows.mean(0), rows.std(0)
        for i, name in enumerate(FEATURE_NAMES):
            if name in BINARY_FEATURES:
                mean[i], std[i] = 0.0, 1.0
        std = np.where(std < 1e-6, 1.0, std)
        d = D[mask]
        d_std = np.where(d.std(0) < 1e-6, 1.0, d.std(0))
        return cls(mean.astype(np.float32), std.astype(np.float32), d_std.astype(np.float32))

    def x(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.x_mean) / self.x_std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(), "d_std": self.d_std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(np.asarray(d["x_mean"], np.float32), np.asarray(d["x_std"], np.float32),
                   np.asarray(d["d_std"], np.float32))


@dataclass
class ScenarioTensors:
    """Все сценарии датасета в виде плотных массивов (у всех одна топология)."""

    X: np.ndarray  # (S, E, F) признаки
    R: np.ndarray  # (S, E, 3) прогноз «без изменений» (норм. пространство)
    Yn: np.ndarray  # (S, E, 3) истинные цели (норм. пространство)
    mask: np.ndarray  # (S, E) есть ли истинное значение (и ребро открыто)
    C: np.ndarray  # (S, E, K) контекст для табличных моделей
    speed: np.ndarray  # (S, E) разрешённая скорость после изменений
    lanes: np.ndarray  # (S, E)
    closed: np.ndarray  # (S, E)

    def state(self, i: int, base: State) -> State:
        return State(self.lanes[i].astype(np.float64), self.speed[i].astype(np.float64),
                     self.closed[i].copy(), base.tl.copy())


def build_tensors(fb: FeatureBuilder, changes: list[list[dict]], Y: np.ndarray | None = None) -> ScenarioTensors:
    topo, base = fb.topo, fb.base
    S, E = len(changes), topo.n_edges
    F, K = len(FEATURE_NAMES), 6
    X = np.zeros((S, E, F), np.float32)
    R = np.zeros((S, E, 3), np.float32)
    Yn = np.full((S, E, 3), np.nan, np.float32)
    C = np.zeros((S, E, K), np.float32)
    speed = np.zeros((S, E), np.float32)
    lanes = np.zeros((S, E), np.float32)
    closed = np.zeros((S, E), bool)
    for i, ch in enumerate(changes):
        st = apply_changes(topo, base, ch)
        X[i] = fb.features(st)
        R[i] = fb.residual_base(st)
        C[i] = fb.context(st)
        speed[i], lanes[i], closed[i] = st.speed, st.lanes, st.closed
        if Y is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                Yn[i] = transform_targets(Y[i], topo, st)
    mask = np.isfinite(Yn).all(-1) & ~closed
    return ScenarioTensors(X, R, Yn, mask, C, speed, lanes, closed)


class GNNRunner:
    """Модель + нормализация + граф: прогноз для набора сценариев одной сети."""

    def __init__(self, model: TrafficGNN, norm: Normalizer, zero_correction: bool = True):
        self.model = model.eval()
        self.norm = norm
        self.zero_correction = zero_correction
        self._ei_cache: dict[tuple, dict[int, dict]] = {}

    @property
    def dilations(self) -> list[int]:
        return self.model.dilations

    def _graphs(self, graphs: dict[int, np.ndarray], E: int, B: int) -> dict[int, dict]:
        key = (id(graphs), E, B)
        if key not in self._ei_cache:
            self._ei_cache = {k: v for k, v in self._ei_cache.items() if k[0] == id(graphs)}
            self._ei_cache[key] = batched_graphs(graphs, E, B)
        return self._ei_cache[key]

    @torch.no_grad()
    def raw_delta(self, X: np.ndarray, graphs: dict[int, np.ndarray], batch_size: int = 32) -> np.ndarray:
        """X: (S, E, F) -> Δ в нормированном пространстве целей, (S, E, 3).
        graphs — FeatureBuilder.graphs(runner.dilations)."""
        S, E, F = X.shape
        out = np.zeros((S, E, 3), np.float32)
        Xn = self.norm.x(X)
        for a in range(0, S, batch_size):
            xb = torch.from_numpy(Xn[a:a + batch_size]).reshape(-1, F)
            B = xb.shape[0] // E
            d = self.model(xb, self._graphs(graphs, E, B), B).reshape(B, E, 3).numpy()
            out[a:a + B] = d * self.norm.d_std
        return out

    def predict_norm(self, X: np.ndarray, R: np.ndarray, graphs: dict[int, np.ndarray],
                     X_base: np.ndarray | None = None, batch_size: int = 32) -> np.ndarray:
        """Прогноз в нормированном пространстве: база + Δ.

        zero_correction: из Δ вычитается Δ, которую модель выдаёт для сети без
        изменений. Тогда predict(network, []) в точности равен базовому
        состоянию, а систематический сдвиг модели не «рисует» изменения на
        далёких от правок улицах при сравнении «до/после».
        """
        delta = self.raw_delta(X, graphs, batch_size)
        if self.zero_correction and X_base is not None:
            delta = delta - self.raw_delta(X_base[None], graphs)[0][None]
        return R + delta


def to_physical(norm_pred: np.ndarray, topo: Topology, T: ScenarioTensors, idx: np.ndarray,
                base: State) -> np.ndarray:
    """(n, E, 3) нормированные прогнозы сценариев idx -> физические единицы."""
    out = np.zeros_like(norm_pred, dtype=np.float64)
    for k, i in enumerate(idx):
        out[k] = inverse_targets(norm_pred[k], topo, T.state(i, base))
    return out


# --- метрики ------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


def evaluate(Y_true: np.ndarray, Y_pred: np.ndarray, mask: np.ndarray, base_phys: np.ndarray,
             near: np.ndarray | None = None, targets=("occupancy", "mean_speed", "travel_time")) -> dict:
    """Метрики по каждой целевой величине в физических единицах.

    * mae / rmse / r2 — по всем открытым рёбрам тестовых сценариев;
    * delta_r2 — R² *изменения* (прогноз − база vs факт − база): насколько
      модель объясняет эффект правок, а не просто воспроизводит базовую карту;
    * near_* — только рёбра в радиусе 3 переходов от изменений.
    """
    res = {}
    for j, t in enumerate(targets):
        yt, yp = Y_true[..., j][mask], Y_pred[..., j][mask]
        b = np.broadcast_to(base_phys[..., j], Y_true[..., j].shape)[mask]
        m = regression_metrics(yt, yp)
        m["delta_r2"] = regression_metrics(yt - b, yp - b)["r2"]
        if near is not None:
            nm = mask & near
            ytn, ypn = Y_true[..., j][nm], Y_pred[..., j][nm]
            bn = np.broadcast_to(base_phys[..., j], Y_true[..., j].shape)[nm]
            mn = regression_metrics(ytn, ypn)
            m["near_mae"] = mn["mae"]
            m["near_rmse"] = mn["rmse"]
            m["near_delta_r2"] = regression_metrics(ytn - bn, ypn - bn)["r2"]
        res[t] = m
    return res
