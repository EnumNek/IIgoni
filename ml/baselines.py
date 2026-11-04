"""Бейзлайны для сравнения с GNN.

Все модели решают одну и ту же задачу — предсказывают Δ нормированных метрик
ребра относительно базового состояния — и оцениваются одинаково.

1. ``NoChange`` — «прогноз = без изменений»: Δ = 0, т.е. после правок
   метрики считаются такими же, как в базовом прогоне. Нижняя граница:
   любая полезная модель обязана её превзойти.
2. ``LinearBaseline`` — гребневая регрессия (Ridge) на табличных признаках.
3. ``GBBaseline`` — градиентный бустинг (HistGradientBoostingRegressor из
   scikit-learn; по качеству сопоставим с LightGBM, но без доп. зависимостей).

Табличные модели не видят граф, поэтому им, кроме тех же признаков ребра,
что и у GNN, даётся инженерный контекст сценария (features.CONTEXT_NAMES):
число изменений каждого типа и расстояние в переходах до ближайшего
изменения выше/ниже по потоку. Это делает сравнение строже: преимущество GNN
не объясняется тем, что бейзлайнам «не сказали», где были правки.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ml.config import SEED


class NoChange:
    name = "Без изменений"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros((X.shape[0], 3), np.float32)


class LinearBaseline:
    name = "Линейная регрессия"

    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X)).astype(np.float32)


class GBBaseline:
    name = "Градиентный бустинг"

    def __init__(self, max_iter: int = 500, learning_rate: float = 0.08, max_leaf_nodes: int = 63):
        self.models = [HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=40, l2_regularization=1e-3, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=25, random_state=SEED) for _ in range(3)]

    def fit(self, X, y):
        for j, m in enumerate(self.models):
            m.fit(X, y[:, j])
        return self

    def predict(self, X):
        return np.stack([m.predict(X) for m in self.models], axis=1).astype(np.float32)


def tabular(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(…, F) признаки + (…, K) контекст -> плоская таблица (n, F+K)."""
    return np.concatenate([X, C], axis=-1).reshape(-1, X.shape[-1] + C.shape[-1])
