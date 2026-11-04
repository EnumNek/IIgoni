"""Упрощённая макроскопическая транспортная модель — «учитель» для синтетического датасета.

Пока часть sim/ не выдала датасет SUMO, ml/ отлаживается на данных, которые
генерирует эта модель. Требования к ней: быстро считать тысячи сценариев и
воспроизводить *сетевые* эффекты, ради которых нужна GNN (закрытие улицы
перераспределяет поток на соседние, снижение пропускной способности
вызывает затор и объезд). Для этого используется классическая схема:

1. Спрос — матрица корреспонденций (OD) между зонами: граничные узлы района
   (въезд/выезд транзита) и часть внутренних узлов (местные поездки).
2. Время проезда ребра — функция BPR (Bureau of Public Roads):
       t = t0 * (1 + 0.15 * (v / c)^4),
   где t0 — время свободного проезда плюс задержка на перекрёстке в конце
   ребра, c — пропускная способность: полосы * 1800 авт/ч * доля зелёного
   (светофор) или коэффициент второстепенного направления (приоритетный
   перекрёсток).
3. Потоки — приближённое равновесие по Вардропу методом последовательных
   усреднений (MSA): на каждой итерации все корреспонденции загружаются на
   кратчайшие по текущему времени пути («всё или ничего»), затем потоки
   усредняются с шагом 1/k.
4. Выход по каждому ребру в формате контракта:
       travel_time = t (с), mean_speed = длина / t (м/с),
       occupancy = N * L_авт / (длина * полосы), где N = v * t / 3600 —
       число машин на ребре по формуле Литтла.
5. Стохастичность SUMO (разные seed маршрутов) имитируется случайным
   возмущением матрицы OD в каждом сценарии.

Модель заведомо проще SUMO (нет поворотных ограничений, очередей,
spillback), но этого достаточно, чтобы проверить весь конвейер обучения и
корректность сравнения GNN с бейзлайнами.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from ml.network import State, Topology


@dataclass
class MacroParams:
    sat_flow: float = 1800.0  # насыщенный поток, авт/ч на полосу
    tl_green: float = 0.45  # доля эффективного зелёного на светофоре
    tl_cycle: float = 90.0  # цикл светофора, с
    major_factor: float = 0.95  # главное направление нерегулируемого перекрёстка
    minor_factor: float = 0.55  # второстепенное направление
    minor_delay: float = 6.0  # с
    equal_factor: float = 0.65  # равнозначный перекрёсток (right_before_left и т.п.)
    equal_delay: float = 4.0
    bpr_alpha: float = 0.15
    bpr_beta: float = 4.0
    veh_length: float = 7.5  # длина машины + дистанция, м
    msa_iters: int = 30
    max_delay_factor: float = 15.0  # ограничение роста времени (в SUMO — телепорт)
    od_noise: float = 0.30  # σ логнормального шума корреспонденций
    global_noise: float = 0.03  # σ общего масштаба спроса в сценарии
    n_boundary_zones: int = 24
    n_inner_zones: int = 60
    target_vc_p95: float = 1.0  # калибровка: 95-й перцентиль v/c в базе
    bg_flow: tuple[float, float] = (20.0, 120.0)  # фоновый местный трафик, авт/ч на полосу


class MacroModel:
    def __init__(self, topo: Topology, base: State, params: MacroParams | None = None, seed: int = 0):
        self.t = topo
        self.base = base
        self.p = params or MacroParams()
        rng = np.random.default_rng(seed)
        V = topo.n_nodes
        # --- зоны спроса -------------------------------------------------------------
        inc_lanes = np.bincount(topo.src, weights=base.lanes, minlength=V) + \
            np.bincount(topo.dst, weights=base.lanes, minlength=V)
        cap_w = inc_lanes * (1 + np.bincount(topo.src, weights=topo.priority.clip(0), minlength=V))
        # Зоны берутся только из наибольшей сильно связной компоненты: из неё
        # можно доехать куда угодно и вернуться (в OSM-выгрузке бывают «обрывки»
        # односторонних улиц на краю карты).
        ok = ~base.closed & (topo.src != topo.dst)
        g = csr_matrix((np.ones(ok.sum()), (topo.src[ok], topo.dst[ok])), shape=(V, V))
        _, comp = connected_components(g, directed=True, connection="strong")
        main = comp == np.bincount(comp).argmax()
        fringe = np.nonzero(topo.fringe & (inc_lanes > 0) & main)[0]
        inner = np.nonzero(~topo.fringe & (inc_lanes > 0) & main)[0]
        # Граница: узлы, через которые в район входят самые «крупные» дороги
        fringe = fringe[np.argsort(-cap_w[fringe], kind="stable")][: self.p.n_boundary_zones]
        if len(inner) > self.p.n_inner_zones:
            pr = cap_w[inner] / cap_w[inner].sum()
            inner = np.sort(rng.choice(inner, self.p.n_inner_zones, replace=False, p=pr))
        self.zones = np.concatenate([fringe, inner]).astype(np.int64)
        is_boundary = np.r_[np.ones(len(fringe), bool), np.zeros(len(inner), bool)]
        Z = len(self.zones)
        if Z < 2:
            raise ValueError("Слишком мало узлов для построения матрицы спроса")
        # --- матрица корреспонденций: гравитационная модель ------------------------
        w = rng.lognormal(0.0, 0.5, size=Z) * np.where(is_boundary, 2.0, 1.0)
        x, y = self._xy()
        dx = x[self.zones][:, None] - x[self.zones][None, :]
        dy = y[self.zones][:, None] - y[self.zones][None, :]
        dist = np.hypot(dx, dy)
        scale_len = max(np.nanmedian(dist), 1.0)
        od = np.outer(w, w) * np.exp(-0.3 * dist / scale_len)
        # транзит граница-граница заметнее местных поездок
        od *= np.where(is_boundary[:, None] & is_boundary[None, :], 1.5, 1.0)
        np.fill_diagonal(od, 0.0)
        self.od = od / od.sum()
        self.demand = 1000.0  # авт/ч, уточняется калибровкой
        # Фоновый местный трафик (подъезд к домам, парковки): не перераспределяется,
        # благодаря ему даже второстепенные улицы не «пустые», как и в SUMO с randomTrips.
        self.bg_per_lane = rng.uniform(*self.p.bg_flow, size=topo.n_edges)

    def _xy(self):
        lon, lat = self.t.node_lon, self.t.node_lat
        if np.all(np.isfinite(lon)) and np.all(np.isfinite(lat)):
            lat0 = np.deg2rad(np.mean(lat))
            return (lon - lon.mean()) * 111320 * np.cos(lat0), (lat - lat.mean()) * 110540
        V = self.t.n_nodes  # без координат — «расстояние» по индексу (только для весов OD)
        return np.arange(V, dtype=float), np.zeros(V)

    # --- характеристики рёбер для заданного состояния --------------------------------
    def link_params(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает (t0, capacity) для каждого ребра."""
        t, p = self.t, self.p
        E, V = t.n_edges, t.n_nodes
        open_ = ~state.closed
        max_pr = np.full(V, -np.inf)
        np.maximum.at(max_pr, t.dst[open_], t.priority[open_])
        factor = np.ones(E)
        delay = np.zeros(E)
        tl_end = state.tl[t.dst]
        end_fringe = t.fringe[t.dst]
        equal = np.array([t.node_type[j] in ("right_before_left", "allway_stop", "left_before_right")
                          for j in t.dst])
        major = t.priority >= max_pr[t.dst]
        # нерегулируемый перекрёсток с приоритетом
        factor = np.where(major, p.major_factor, p.minor_factor)
        delay = np.where(major, 1.0, p.minor_delay)
        factor = np.where(equal, p.equal_factor, factor)
        delay = np.where(equal, p.equal_delay, delay)
        # светофор: доля зелёного и средняя задержка Вебстера (равномерная часть)
        g = p.tl_green
        factor = np.where(tl_end, g, factor)
        delay = np.where(tl_end, 0.5 * p.tl_cycle * (1 - g) ** 2, delay)
        # выезд из района — без задержки
        factor = np.where(end_fringe, 1.0, factor)
        delay = np.where(end_fringe, 0.0, delay)
        t0 = t.length / state.speed + delay
        cap = state.lanes * p.sat_flow * factor
        return t0, cap

    def _cost(self, flow, t0, cap):
        p = self.p
        vc = flow / cap
        return t0 * np.minimum(1.0 + p.bpr_alpha * vc ** p.bpr_beta, p.max_delay_factor)

    # --- загрузка «всё или ничего» ------------------------------------------------------
    def _aon(self, cost: np.ndarray, closed: np.ndarray, od_abs: np.ndarray) -> tuple[np.ndarray, float]:
        t = self.t
        E, V = t.n_edges, t.n_nodes
        idx = np.nonzero(~closed & (t.src != t.dst))[0]
        # из параллельных рёбер между парой узлов маршрут идёт по самому быстрому
        order = np.lexsort((cost[idx], t.dst[idx], t.src[idx]))
        oi = idx[order]
        first = np.ones(len(oi), bool)
        first[1:] = (t.src[oi][1:] != t.src[oi][:-1]) | (t.dst[oi][1:] != t.dst[oi][:-1])
        best = oi[first]
        g = csr_matrix((cost[best], (t.src[best], t.dst[best])), shape=(V, V))
        pair_edge = {}
        pair_edge_arr = np.full(V * V, -1, dtype=np.int64) if V <= 6000 else None
        if pair_edge_arr is not None:
            pair_edge_arr[t.src[best] * V + t.dst[best]] = best
        else:  # очень большие сети
            pair_edge = {(int(a), int(b)): int(e) for a, b, e in zip(t.src[best], t.dst[best], best)}

        dist, pred = dijkstra(g, directed=True, indices=self.zones, return_predecessors=True)
        Z = len(self.zones)
        demand = np.zeros((Z, V))
        reach = np.isfinite(dist[:, self.zones])
        demand[:, self.zones] = np.where(reach, od_abs, 0.0)
        unserved = float(od_abs[~reach].sum())
        # поток через узлы дерева кратчайших путей: x = d + T x (T — дерево, нильпотентна)
        zz, vv = np.nonzero(pred >= 0)
        parent = pred[zz, vv]
        child_g = zz * V + vv
        parent_g = zz * V + parent
        T = csr_matrix((np.ones(len(zz)), (parent_g, child_g)), shape=(Z * V, Z * V))
        x = demand.ravel().copy()
        y = x.copy()
        for _ in range(V):
            y = T @ y
            if not y.any():
                break
            x += y
        if pair_edge_arr is not None:
            eidx = pair_edge_arr[parent * V + vv]
        else:
            eidx = np.array([pair_edge[(int(a), int(b))] for a, b in zip(parent, vv)], dtype=np.int64)
        flow = np.bincount(eidx, weights=x[child_g], minlength=E)
        return flow, unserved

    def assign(self, state: State, demand_scale: np.ndarray | float = 1.0) -> dict:
        """Равновесное распределение (MSA). demand_scale — множитель матрицы OD."""
        t0, cap = self.link_params(state)
        scale = np.asarray(demand_scale, dtype=float)
        od_abs = self.od * self.demand * scale
        bg = np.where(state.closed, 0.0, self.bg_per_lane * state.lanes * float(np.mean(scale)))
        flow, unserved = self._aon(self._cost(bg, t0, cap), state.closed, od_abs)
        for k in range(2, self.p.msa_iters + 1):
            aux, unserved = self._aon(self._cost(flow + bg, t0, cap), state.closed, od_abs)
            flow += (aux - flow) / k
        flow = flow + bg
        tt = self._cost(flow, t0, cap)
        L = self.t.length
        speed = L / tt
        n_veh = flow * tt / 3600.0  # формула Литтла
        raw = n_veh * self.p.veh_length / (L * state.lanes)
        # Мягкое насыщение: на очень коротких рёбрах перед светофором очередь
        # физически «уходит» на соседние участки, занятость не может превысить 1.
        occ = 1.0 - np.exp(-raw)
        res = {"occupancy": occ, "mean_speed": speed, "travel_time": tt, "volume": flow,
               "vc": flow / cap, "unserved": unserved}
        for key in ("occupancy", "mean_speed", "travel_time", "volume", "vc"):
            res[key] = np.where(state.closed, np.nan, res[key])
        return res

    def calibrate(self, rounds: int = 5) -> float:
        """Подбирает общий объём спроса так, чтобы сеть была умеренно загружена
        (95-й перцентиль v/c ≈ target_vc_p95): иначе изменения почти ни на что не влияют
        или вся сеть стоит в пробке."""
        for _ in range(rounds):
            res = self.assign(self.base)
            vc = res["vc"][np.isfinite(res["vc"]) & (res["volume"] > 0)]
            p95 = float(np.percentile(vc, 95)) if len(vc) else 0.0
            if p95 <= 0:
                break
            self.demand *= (self.p.target_vc_p95 / p95) ** 0.9
        return self.demand

    def sample_noise(self, rng: np.random.Generator) -> np.ndarray:
        """Случайный множитель матрицы OD — аналог другого seed маршрутов в SUMO."""
        Z = len(self.zones)
        g = rng.lognormal(0.0, self.p.global_noise)
        return g * rng.lognormal(-0.5 * self.p.od_noise ** 2, self.p.od_noise, size=(Z, Z))
