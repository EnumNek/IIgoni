"""Аналитическая модель транспортных потоков — общий «движок» заглушек.

Нужна, чтобы до появления обученной GNN (ml/) и RL-агентов (agents/)
веб-приложение показывало правдоподобную реакцию сети на изменения:
закрытие улицы перераспределяет поток на соседние, добавление полосы
снижает загрузку, светофор добавляет задержку и т.д.

Используются классические методы транспортного планирования:
  * спрос — синтетическая гравитационная матрица корреспонденций между
    «зонами» (въезды на границе района + внутренние узлы), масштаб которой
    калибруется по базовой сети (90-й перцентиль загрузки ≈ 0,85);
  * распределение потоков — равновесие по Вардропу методом последовательных
    средних (MSA) с поиском кратчайших путей алгоритмом Дейкстры (scipy);
  * время проезда перегона — функция BPR: t = t0·(1 + 0,15·(q/c)^4);
  * задержка на регулируемом перекрёстке — формула HCM 2000
    (равномерная d1 + случайная/перенасыщения d2).
Это НЕ замена SUMO и GNN: модель детерминирована, не учитывает
взаимодействие полос, перестроения, блокировку перекрёстков и т.п.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

SAT_FLOW = 1800.0  # авт/ч на полосу — насыщенный поток
VEH_LENGTH = 5.0  # м — длина легкового авто в SUMO (для расчёта occupancy как в edgeData)
FIXED_CYCLE = 90.0  # с — типовой цикл жёсткого регулирования
LOST_TIME_PER_PHASE = 4.0  # с
ANALYSIS_PERIOD_H = 0.25  # ч — период анализа HCM
TARGET_P90_SATURATION = 0.85
MSA_ITERATIONS = 20


@dataclass
class Graph:
    node_ids: list[str]
    node_index: dict[str, int]
    lon: np.ndarray
    lat: np.ndarray
    edge_ids: list[str]
    u: np.ndarray
    v: np.ndarray
    length: np.ndarray
    lanes: np.ndarray
    speed: np.ndarray
    closed: np.ndarray
    tls_end: np.ndarray  # участок заканчивается регулируемым перекрёстком
    end_bearing: np.ndarray  # азимут последнего отрезка (для группировки подходов по фазам)
    node_type: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.node_ids)

    @property
    def free_time(self) -> np.ndarray:
        return self.length / np.maximum(self.speed, 0.5)


def _bearing(a: list[float], b: list[float]) -> float:
    dx = (b[0] - a[0]) * math.cos(math.radians((a[1] + b[1]) / 2))
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dx, dy)) % 360


def build_graph(network: dict) -> Graph:
    nodes = network["nodes"]
    node_ids = [str(n["id"]) for n in nodes]
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    node_type = [n.get("type", "priority") for n in nodes]
    lon = np.array([float(n["lon"]) for n in nodes])
    lat = np.array([float(n["lat"]) for n in nodes])
    edges = [e for e in network["edges"] if str(e["from"]) in node_index and str(e["to"]) in node_index]
    u = np.array([node_index[str(e["from"])] for e in edges], dtype=np.int64)
    v = np.array([node_index[str(e["to"])] for e in edges], dtype=np.int64)
    bearings = []
    for e in edges:
        geom = e.get("geometry") or []
        if len(geom) >= 2:
            bearings.append(_bearing(geom[-2], geom[-1]))
        else:
            a, b = node_index[str(e["from"])], node_index[str(e["to"])]
            bearings.append(_bearing([lon[a], lat[a]], [lon[b], lat[b]]))
    is_tls = np.array([t == "traffic_light" for t in node_type], dtype=bool)
    return Graph(
        node_ids=node_ids, node_index=node_index, lon=lon, lat=lat,
        edge_ids=[str(e["id"]) for e in edges], u=u, v=v,
        length=np.array([max(float(e.get("length") or 1.0), 1.0) for e in edges]),
        lanes=np.array([max(int(e.get("lanes") or 1), 1) for e in edges], dtype=float),
        speed=np.array([max(float(e.get("speed") or 13.89), 0.5) for e in edges]),
        closed=np.array([bool(e.get("closed")) for e in edges], dtype=bool),
        tls_end=is_tls[v] if len(v) else np.zeros(0, dtype=bool),
        end_bearing=np.array(bearings), node_type=node_type,
    )


def fingerprint(network: dict) -> str:
    h = hashlib.sha1()
    for e in network["edges"]:
        h.update(f"{e['id']}|{e['from']}|{e['to']}|{e.get('length')}|{e.get('lanes')}|{e.get('speed')};".encode())
    for n in network["nodes"]:
        h.update(f"{n['id']}|{n.get('type')};".encode())
    return h.hexdigest()


# ------------------------------------------------------------ задержки на перекрёстке
def hcm_delay(q: np.ndarray, c: np.ndarray, green_ratio: np.ndarray, cycle: np.ndarray, k: float) -> np.ndarray:
    """Средняя задержка, с/авт (HCM 2000, ф-ла 16-9): d = d1 + d2.

    q — поток, авт/ч; c — пропускная способность подхода, авт/ч;
    k — коэффициент типа регулирования (0,5 — жёсткое, меньше — адаптивное).
    """
    c = np.maximum(c, 1.0)
    x = q / c
    d1 = 0.5 * cycle * (1 - green_ratio) ** 2 / np.maximum(1 - np.minimum(x, 1.0) * green_ratio, 1e-3)
    t = ANALYSIS_PERIOD_H
    d2 = 900 * t * ((x - 1) + np.sqrt((x - 1) ** 2 + 8 * k * x / (c * t)))
    return d1 + d2


def fixed_signal_ratio() -> float:
    return (FIXED_CYCLE - 2 * LOST_TIME_PER_PHASE) / 2 / FIXED_CYCLE


@dataclass
class SignalPlan:
    """Параметры регулирования для каждого ребра, заканчивающегося светофором."""

    green_ratio: np.ndarray
    cycle: np.ndarray
    k: float = 0.5


def fixed_plan(g: Graph) -> SignalPlan:
    m = len(g.edge_ids)
    return SignalPlan(green_ratio=np.full(m, fixed_signal_ratio()), cycle=np.full(m, FIXED_CYCLE), k=0.5)


def link_costs(g: Graph, q: np.ndarray, plan: SignalPlan) -> tuple[np.ndarray, np.ndarray]:
    """Время движения по перегону (BPR) и задержка на светофоре в конце участка, с."""
    cap_mid = SAT_FLOW * g.lanes * np.where(g.speed >= 13.0, 1.0, 0.8)
    t_run = g.free_time * (1 + 0.15 * (q / cap_mid) ** 4)
    d_sig = np.zeros_like(q)
    if g.tls_end.any():
        idx = g.tls_end
        cap_sig = SAT_FLOW * g.lanes[idx] * plan.green_ratio[idx]
        d_sig[idx] = hcm_delay(q[idx], cap_sig, plan.green_ratio[idx], plan.cycle[idx], plan.k)
    return t_run, d_sig


# ------------------------------------------------------------ спрос
@dataclass
class Demand:
    zones: np.ndarray  # индексы узлов-зон
    matrix: np.ndarray  # авт/ч, zones × zones


def build_demand(g: Graph, total: float = 1.0, seed: int = 7) -> Demand:
    """Гравитационная матрица: q_ij ~ w_i · w_j · exp(-d_ij / 1,5 км)."""
    rng = np.random.default_rng(seed + g.n)
    degree = np.bincount(np.concatenate([g.u, g.v]), minlength=g.n)
    # «Вес» узла — сумма полос примыкающих участков, взвешенная по скорости (магистрали весомее)
    weight = np.zeros(g.n)
    np.add.at(weight, g.u, g.lanes * g.speed)
    np.add.at(weight, g.v, g.lanes * g.speed)
    lon_span = g.lon.max() - g.lon.min() + 1e-9
    lat_span = g.lat.max() - g.lat.min() + 1e-9
    edge_dist = np.minimum.reduce([
        (g.lon - g.lon.min()) / lon_span, (g.lon.max() - g.lon) / lon_span,
        (g.lat - g.lat.min()) / lat_span, (g.lat.max() - g.lat) / lat_span,
    ])
    # Въезды/выезды района: тупиковые узлы у границы (обрезанные улицы)
    boundary = np.where((degree <= 2) & (edge_dist < 0.04))[0]
    boundary = boundary[np.argsort(-weight[boundary])][:30]
    internal_pool = np.setdiff1d(np.where(degree >= 3)[0], boundary)
    n_internal = min(len(internal_pool), max(10, 70 - len(boundary)))
    if n_internal and len(internal_pool):
        p = weight[internal_pool] / weight[internal_pool].sum()
        internal = rng.choice(internal_pool, size=n_internal, replace=False, p=p)
    else:
        internal = np.array([], dtype=np.int64)
    zones = np.concatenate([boundary, internal]).astype(np.int64)
    if len(zones) < 2:  # вырожденная сеть — берём все узлы
        zones = np.arange(g.n)
    w = weight[zones] * np.where(np.isin(zones, boundary), 2.0, 1.0)
    x = g.lon[zones] * 111_320 * math.cos(math.radians(float(g.lat.mean())))
    y = g.lat[zones] * 110_540
    dist = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    m = np.outer(w, w) * np.exp(-dist / 1500.0)
    np.fill_diagonal(m, 0.0)
    m[dist < 150] = 0.0  # слишком короткие поездки на авто не моделируем
    m *= total / max(m.sum(), 1e-9)
    return Demand(zones=zones, matrix=m)


# ------------------------------------------------------------ распределение потоков
def _all_or_nothing(g: Graph, cost: np.ndarray, open_mask: np.ndarray, demand: Demand) -> tuple[np.ndarray, float]:
    n = g.n
    sel_all = np.where(open_mask)[0]
    key = g.u[sel_all] * n + g.v[sel_all]
    order = np.lexsort((cost[sel_all], key))
    k_sorted = key[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = k_sorted[1:] != k_sorted[:-1]
    sel = sel_all[order[first]]  # из параллельных рёбер берём самое «дешёвое»
    mat = csr_matrix((cost[sel] + 1e-6, (g.u[sel], g.v[sel])), shape=(n, n))
    lookup = dict(zip((g.u[sel] * n + g.v[sel]).tolist(), sel.tolist()))
    dist, pred = dijkstra(mat, directed=True, indices=demand.zones, return_predecessors=True)
    load = [0.0] * len(g.edge_ids)
    served = 0.0
    for oi, origin in enumerate(demand.zones.tolist()):
        reach = np.isfinite(dist[oi])
        node_flow = np.zeros(n)
        row = demand.matrix[oi] * reach[demand.zones]
        np.add.at(node_flow, demand.zones, row)
        served += float(row.sum())
        nodes_with_flow = np.where(reach)[0]
        order_nodes = nodes_with_flow[np.argsort(-dist[oi][nodes_with_flow])].tolist()
        flow = node_flow.tolist()
        p = pred[oi].tolist()
        for nd in order_nodes:
            f = flow[nd]
            if f <= 0.0 or nd == origin:
                continue
            pn = p[nd]
            if pn < 0:
                continue
            load[lookup[pn * n + nd]] += f
            flow[pn] += f
    return np.array(load), served


@dataclass
class Assignment:
    flow: np.ndarray  # авт/ч по каждому ребру
    t_run: np.ndarray
    d_sig: np.ndarray
    served: float  # авт/ч, для которых нашёлся путь
    demand_total: float

    @property
    def travel_time(self) -> np.ndarray:
        return self.t_run + self.d_sig


def assign(g: Graph, demand: Demand, plan: SignalPlan | None = None, iterations: int = MSA_ITERATIONS) -> Assignment:
    plan = plan or fixed_plan(g)
    open_mask = ~g.closed
    q = np.zeros(len(g.edge_ids))
    served = 0.0
    for it in range(iterations):
        t_run, d_sig = link_costs(g, q, plan)
        y, served = _all_or_nothing(g, t_run + d_sig, open_mask, demand)
        q = y if it == 0 else q + (y - q) / (it + 1)
    q[g.closed] = 0.0
    t_run, d_sig = link_costs(g, q, plan)
    return Assignment(flow=q, t_run=t_run, d_sig=d_sig, served=served, demand_total=float(demand.matrix.sum()))


_CALIBRATION: dict[str, float] = {}


def calibrated_demand(base_network: dict) -> Demand:
    """Спрос, откалиброванный на базовой (неизменённой) сети; кэшируется по отпечатку сети."""
    g0 = build_graph(base_network)
    key = fingerprint(base_network)
    if key not in _CALIBRATION:
        trial_total = 5000.0
        res = assign(g0, build_demand(g0, trial_total), iterations=12)
        sat = res.flow / (SAT_FLOW * g0.lanes * np.where(g0.tls_end, fixed_signal_ratio(), 0.9))
        used = sat[res.flow > 1.0]
        p90 = float(np.percentile(used, 90)) if len(used) else 1.0
        _CALIBRATION[key] = trial_total * TARGET_P90_SATURATION / max(p90, 1e-6)
    return build_demand(g0, _CALIBRATION[key])


def occupancy(g: Graph, flow: np.ndarray, travel_time: np.ndarray) -> np.ndarray:
    """Доля длины участка, занятая автомобилями (аналог occupancy SUMO edgeData / 100).

    По формуле Литтла: число авто на полосе = поток_полосы · время_проезда.
    """
    vehicles_per_lane = flow / 3600.0 / g.lanes * travel_time
    return np.clip(vehicles_per_lane * VEH_LENGTH / g.length, 0.0, 1.0)
