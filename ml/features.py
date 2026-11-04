"""Line graph дорожной сети, признаки рёбер и преобразование целевых метрик.

Почему line graph. Прогноз нужен *по каждому участку дороги* (ребру), а
стандартные GNN (GraphSAGE, GAT) делают предсказания для узлов графа. Поэтому
сеть переводится в line graph: каждое ребро дороги становится узлом, а
направленная связь «ребро a -> ребро b» проводится, если с участка a можно
проехать на участок b (a.to == b.from). Так сообщения GNN распространяются
вдоль реальных маневров: затор на участке передаётся вверх по потоку,
перераспределённый после закрытия поток — вниз.

Развороты (b — обратное к a ребро) исключаются, если у a есть другие
продолжения: в городской сети разворот на перекрёстке скорее исключение,
а лишние связи только «размывают» сообщения.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from ml.network import State, Topology

# --- Признаки, общие для GNN и табличных бейзлайнов --------------------------------
FEATURE_NAMES = [
    # структура (не меняется)
    "log_length", "priority", "from_in_degree", "to_out_degree", "is_entry", "is_exit",
    # итоговое состояние после изменений
    "lanes", "speed", "log_free_time", "tl_at_end", "tl_at_start", "closed",
    # разница с базовой сетью
    "d_lanes", "d_speed", "d_tl_end", "d_tl_start", "changed",
    # базовое состояние (прогон без изменений), в нормированном виде
    "base_occupancy", "base_speed_ratio", "base_log_delay", "base_log_flow", "base_known",
]
# Признаки, которые не стандартизуются (уже в разумном масштабе / бинарные)
BINARY_FEATURES = {"is_entry", "is_exit", "tl_at_end", "tl_at_start", "closed", "changed",
                   "base_known"}

# Дополнительный контекст сценария — только для табличных бейзлайнов, которые
# «не видят» граф: число изменений и расстояние (в переходах) до ближайшего изменения.
CONTEXT_NAMES = ["n_closed", "n_lanes", "n_speed", "n_tl", "hops_to_change_down",
                 "hops_to_change_up"]
MAX_HOPS = 12

# Ограничения при обратном преобразовании прогноза в физические единицы
VEHICLE_SPACE = 7.5  # м на машину (длина + дистанция) — для оценки потока по формуле Литтла
MAX_SPEED_RATIO = 1.3  # средняя скорость может немного превышать лимит (speedFactor в SUMO)
MIN_LOG_DELAY = -0.3


def line_graph(topo: Topology) -> np.ndarray:
    """Рёбра line graph: массив (2, M) пар (a, b) — «с участка a можно выехать на b»."""
    out_edges: list[list[int]] = [[] for _ in range(topo.n_nodes)]
    for e in range(topo.n_edges):
        out_edges[topo.src[e]].append(e)
    pairs_a, pairs_b = [], []
    for a in range(topo.n_edges):
        nxt = out_edges[topo.dst[a]]
        non_uturn = [b for b in nxt if not (topo.dst[b] == topo.src[a] and b != a)]
        for b in (non_uturn or nxt):
            if b != a:
                pairs_a.append(a)
                pairs_b.append(b)
    return np.asarray([pairs_a, pairs_b], dtype=np.int64).reshape(2, -1)


DEFAULT_DILATIONS = (1, 1, 2, 2, 4, 4)


def dilated_graphs(lg: np.ndarray, n: int, dilations) -> dict[int, np.ndarray]:
    """Связи «ровно через d переходов» для каждого d из dilations.

    Эффект изменения распространяется на 8–12 переходов (см. README), а
    каждый слой GNN «видит» только соседей по своим связям. Вместо того чтобы
    делать сеть из 12+ слоёв, слои получают расширенные (dilated) связи:
    при схеме 1, 1, 2, 2, 4, 4 рецептивное поле — 14 переходов при 6 слоях.
    Кольца считаются разреженными степенями матрицы смежности line graph,
    поэтому метод масштабируется на большие сети (без матрицы N×N).
    """
    A = csr_matrix((np.ones(lg.shape[1], np.int32), (lg[0], lg[1])), shape=(n, n))
    A.data[:] = 1
    eye = csr_matrix((np.ones(n, np.int32), (np.arange(n), np.arange(n))), shape=(n, n))
    reach = (A + eye).astype(bool).astype(np.int32)  # достижимо не более чем за 1 переход
    out = {1: lg.astype(np.int64)}
    for d in range(2, max(dilations) + 1):
        nxt = (reach @ A + reach).astype(bool).astype(np.int32)  # не более чем за d
        ring = (nxt - reach).tocoo()
        ring.eliminate_zeros()
        reach = nxt
        if d in dilations:
            keep = ring.data > 0
            out[d] = np.vstack([ring.row[keep], ring.col[keep]]).astype(np.int64)
    return {d: out[d] for d in sorted(set(dilations))}


class FeatureBuilder:
    """Строит признаки для одной сети (кэширует всё, что не зависит от сценария)."""

    def __init__(self, topo: Topology, base_state: State, base_metrics: np.ndarray | None = None):
        """base_metrics — (E, 3) occupancy/mean_speed/travel_time базового прогона;
        NaN в строке означает «неизвестно» — тогда берётся свободный поток."""
        self.topo = topo
        self.base = base_state
        self.lg = line_graph(topo)
        E = topo.n_edges
        in_deg = np.bincount(topo.dst, minlength=topo.n_nodes)
        out_deg = np.bincount(topo.src, minlength=topo.n_nodes)
        self.static = np.stack([
            np.log(topo.length),
            topo.priority,
            in_deg[topo.src].astype(float),
            out_deg[topo.dst].astype(float),
            topo.fringe[topo.src].astype(float),
            topo.fringe[topo.dst].astype(float),
        ], axis=1)

        if base_metrics is None:
            base_metrics = np.full((E, 3), np.nan)
        base_metrics = np.asarray(base_metrics, dtype=np.float64)
        known = np.all(np.isfinite(base_metrics), axis=1) & ~base_state.closed
        free = free_flow_metrics(topo, base_state)
        self.base_metrics = np.where(known[:, None], base_metrics, free)
        self.base_known = known.astype(float)
        self.base_norm_base = transform_targets(self.base_metrics, topo, base_state)
        # Интенсивность потока в базе (авт/ч) по формуле Литтла: N = occ·L·полосы / L_авт,
        # q = N / t. Показывает, сколько машин придётся перераспределить при закрытии.
        n_veh = self.base_metrics[:, 0] * topo.length * base_state.lanes / VEHICLE_SPACE
        self.base_log_flow = np.log1p(3600.0 * n_veh / np.maximum(self.base_metrics[:, 2], 1e-3))
        n = self.lg.shape[1]
        self._adj = csr_matrix((np.ones(n), (self.lg[0], self.lg[1])), shape=(E, E))
        self._graphs: dict[tuple, dict[int, np.ndarray]] = {}

    def graphs(self, dilations=DEFAULT_DILATIONS) -> dict[int, np.ndarray]:
        """Рёбра line graph для каждого расширения (кэшируется)."""
        key = tuple(sorted(set(dilations)))
        if key not in self._graphs:
            self._graphs[key] = dilated_graphs(self.lg, self.topo.n_edges, key)
        return self._graphs[key]

    # -- признаки для GNN ------------------------------------------------------------
    def features(self, state: State) -> np.ndarray:
        t, b = self.topo, self.base
        tl_end, tl_start = state.tl[t.dst], state.tl[t.src]
        d_tl_end = tl_end.astype(float) - b.tl[t.dst]
        d_tl_start = tl_start.astype(float) - b.tl[t.src]
        d_lanes = state.lanes - b.lanes
        d_speed = state.speed - b.speed
        changed = ((d_lanes != 0) | (d_speed != 0) | (state.closed != b.closed)
                   | (d_tl_end != 0) | (d_tl_start != 0))
        dyn = np.stack([
            state.lanes, state.speed, np.log(t.length / state.speed),
            tl_end, tl_start, state.closed,
            d_lanes, d_speed, d_tl_end, d_tl_start, changed,
        ], axis=1).astype(float)
        base = np.column_stack([self.base_norm_base, self.base_log_flow, self.base_known])
        return np.concatenate([self.static, dyn, base], axis=1)

    def residual_base(self, state: State) -> np.ndarray:
        """Прогноз «без изменений» в нормированном пространстве целей для данного
        состояния (относительно *новой* разрешённой скорости)."""
        return transform_targets(self.base_metrics, self.topo, state)

    # -- контекст для табличных моделей ---------------------------------------------
    def context(self, state: State) -> np.ndarray:
        t, b = self.topo, self.base
        changed_edges = np.nonzero((state.lanes != b.lanes) | (state.speed != b.speed)
                                   | (state.closed != b.closed)
                                   | (state.tl[t.dst] != b.tl[t.dst])
                                   | (state.tl[t.src] != b.tl[t.src]))[0]
        E = t.n_edges
        n_closed = float(np.sum(state.closed & ~b.closed))
        n_lanes = float(np.sum(state.lanes != b.lanes))
        n_speed = float(np.sum(state.speed != b.speed))
        n_tl = float(np.sum(state.tl != b.tl))
        if len(changed_edges):
            # down: сколько переходов от ребра e вниз по потоку до изменённого ребра
            d_up = shortest_path(self._adj, directed=True, unweighted=True, indices=changed_edges)
            hops_up = d_up.min(axis=0)  # изменение выше по потоку, чем e
            d_down = shortest_path(self._adj.T.tocsr(), directed=True, unweighted=True,
                                   indices=changed_edges)
            hops_down = d_down.min(axis=0)
        else:
            hops_up = hops_down = np.full(E, np.inf)
        hops_up = np.minimum(hops_up, MAX_HOPS)
        hops_down = np.minimum(hops_down, MAX_HOPS)
        cols = [np.full(E, n_closed), np.full(E, n_lanes), np.full(E, n_speed), np.full(E, n_tl),
                hops_down, hops_up]
        return np.stack(cols, axis=1)


def free_flow_metrics(topo: Topology, state: State) -> np.ndarray:
    """Метрики пустой сети: занятость 0, скорость = лимит, время = длина / лимит."""
    tt = topo.length / state.speed
    return np.stack([np.zeros(topo.n_edges), state.speed.copy(), tt], axis=1)


def transform_targets(metrics: np.ndarray, topo: Topology, state: State) -> np.ndarray:
    """Физические единицы -> нормированное пространство, в котором учится модель.

    * occupancy -> как есть (0..1);
    * mean_speed -> mean_speed / разрешённая скорость (доля от лимита);
    * travel_time -> log(travel_time / (длина / лимит)) — логарифм индекса задержки.
    Логарифм убирает тяжёлый хвост времени проезда на заторных участках, а
    нормировка на лимит делает цели сопоставимыми для улиц разных категорий.
    """
    m = np.asarray(metrics, dtype=np.float64)
    t0 = topo.length / state.speed
    occ = m[:, 0]
    spd = m[:, 1] / state.speed
    tt = np.log(np.maximum(m[:, 2], 1e-3) / t0)
    return np.stack([occ, spd, tt], axis=1)


def inverse_targets(norm: np.ndarray, topo: Topology, state: State) -> np.ndarray:
    """Нормированный прогноз -> физические единицы (с ограничением диапазонов)."""
    t0 = topo.length / state.speed
    occ = np.clip(norm[:, 0], 0.0, 1.0)
    spd = np.clip(norm[:, 1], 0.0, MAX_SPEED_RATIO) * state.speed
    tt = np.exp(np.maximum(norm[:, 2], MIN_LOG_DELAY)) * t0
    return np.stack([occ, spd, tt], axis=1)
