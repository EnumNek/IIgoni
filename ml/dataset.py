"""Загрузчик датасета: sim/data/*.parquet (SUMO) или синтетический датасет ml/.

Источник выбирается так (режим ``auto``):
  * если в sim/data/ есть *.parquet и network.json — используется датасет SUMO;
  * иначе — синтетический датасет ml/data/synthetic/ (если его нет, он
    генерируется автоматически командой из ml/synthetic.py).

Загрузчик сознательно терпим к деталям формата, о которых части системы ещё
не договорились окончательно:
  * метрики и описание сценариев могут лежать в одном файле или в разных
    (файл с колонками scenario_id + edge_id + метрики — метрики; файл с
    scenario_id + changes — сценарии; колонка changes может быть и в файле метрик);
  * поддерживаются синонимы колонок из edgeData SUMO (speed, traveltime, edge, ...);
  * occupancy в процентах (как в edgeData SUMO) автоматически переводится в доли;
  * несколько строк на пару (сценарий, ребро) — например, интервалы
    агрегации edgeData — усредняются;
  * строки с рёбрами, которых нет в network.json (внутренние ':'-рёбра
    перекрёстков SUMO), отбрасываются.
Базовое состояние сети — среднее по сценариям с пустым списком изменений.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import SEED, SIM_DATA_DIR, SYNTH_DIR, TARGETS
from ml.network import State, Topology, load_network, parse_network

log = logging.getLogger(__name__)

COLUMN_ALIASES = {
    "scenario": "scenario_id", "scenarioid": "scenario_id", "run_id": "scenario_id",
    "edge": "edge_id", "edgeid": "edge_id",
    "speed": "mean_speed", "meanspeed": "mean_speed",
    "traveltime": "travel_time", "travel_time_s": "travel_time",
    "occupancy_pct": "occupancy",
}


@dataclass
class TrafficDataset:
    network: dict
    topo: Topology
    base_state: State
    scenario_ids: np.ndarray  # (S,)
    changes: list[list[dict]]  # (S,)
    Y: np.ndarray  # (S, E, 3) метрики в физических единицах, NaN — нет данных
    base_metrics: np.ndarray  # (E, 3)
    is_base: np.ndarray  # (S,) bool — прогон без изменений
    source: str
    path: Path
    meta: dict = field(default_factory=dict)

    @property
    def n_scenarios(self) -> int:
        return len(self.scenario_ids)

    def split(self, val: float = 0.15, test: float = 0.15, seed: int = SEED):
        """Разбиение сценариев (не строк!) на train/val/test. Базовые прогоны
        исключены — они служат эталоном «до» и не несут информации об изменениях."""
        idx = np.nonzero(~self.is_base)[0]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(idx)
        n_test = int(round(len(idx) * test))
        n_val = int(round(len(idx) * val))
        return (np.sort(idx[n_test + n_val:]), np.sort(idx[n_test:n_test + n_val]),
                np.sort(idx[:n_test]))


def _norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        key = COLUMN_ALIASES.get(key.replace("_", ""), COLUMN_ALIASES.get(key, key))
        ren[c] = key
    return df.rename(columns=ren)


def _parse_changes(v) -> list[dict]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (bytes, str)):
        v = json.loads(v) if str(v).strip() else []
    if isinstance(v, dict):
        v = [v]
    return [dict(c) for c in list(v)]


def find_sim_dataset(sim_dir: Path = SIM_DATA_DIR) -> list[Path]:
    if not (sim_dir / "network.json").exists():
        return []
    return sorted(sim_dir.glob("*.parquet"))


def read_parquet_dataset(paths: list[Path], network: dict, source: str = "sim") -> TrafficDataset:
    metrics_parts, scen_parts = [], []
    for p in paths:
        df = _norm_columns(pd.read_parquet(p))
        cols = set(df.columns)
        if "edge_id" not in cols and "id" in cols and cols & set(TARGETS):
            df = df.rename(columns={"id": "edge_id"})  # edgeData SUMO: <edge id=...>
            cols = set(df.columns)
        if "scenario_id" not in cols:
            log.warning("%s: нет колонки scenario_id — файл пропущен", p.name)
            continue
        if "edge_id" in cols and cols & set(TARGETS):
            metrics_parts.append(df)
            if "changes" in cols:
                scen_parts.append(df[["scenario_id", "changes"]].drop_duplicates("scenario_id"))
        elif "changes" in cols:
            scen_parts.append(df[["scenario_id", "changes"]])
        else:
            log.warning("%s: не похоже ни на метрики, ни на сценарии — пропущен", p.name)
    if not metrics_parts:
        raise ValueError(f"Не найдено таблиц с метриками рёбер среди {[p.name for p in paths]}")
    if not scen_parts:
        raise ValueError("Не найдено описание сценариев (колонка changes)")

    m = pd.concat(metrics_parts, ignore_index=True)
    for t in TARGETS:
        if t not in m.columns:
            raise ValueError(f"В метриках нет колонки {t}")
    m["edge_id"] = m["edge_id"].astype(str)
    m["scenario_id"] = m["scenario_id"].astype(np.int64)
    m = m.groupby(["scenario_id", "edge_id"], as_index=False)[list(TARGETS)].mean()
    if np.nanmax(m["occupancy"].to_numpy()) > 1.0 + 1e-6:
        log.warning("occupancy > 1 — считаем, что она в процентах (edgeData SUMO), делим на 100")
        m["occupancy"] = m["occupancy"] / 100.0

    scen = pd.concat(scen_parts, ignore_index=True).drop_duplicates("scenario_id")
    scen["scenario_id"] = scen["scenario_id"].astype(np.int64)
    scen = scen.sort_values("scenario_id").reset_index(drop=True)
    scen = scen[scen["scenario_id"].isin(m["scenario_id"].unique())]

    topo, base_state = parse_network(network)
    e_idx = pd.Index(topo.edge_ids).get_indexer(m["edge_id"])
    unknown = e_idx < 0
    if unknown.any():
        log.info("Отброшено %d строк с рёбрами вне network.json (напр. %s)", int(unknown.sum()),
                 m.loc[unknown, "edge_id"].iloc[0])
    s_ids = scen["scenario_id"].to_numpy()
    s_idx = pd.Index(s_ids).get_indexer(m["scenario_id"])
    keep = (~unknown) & (s_idx >= 0)
    S, E = len(s_ids), topo.n_edges
    Y = np.full((S, E, 3), np.nan, dtype=np.float32)
    Y[s_idx[keep], e_idx[keep]] = m.loc[keep, list(TARGETS)].to_numpy(np.float32)

    changes = [_parse_changes(v) for v in scen["changes"]]
    is_base = np.array([len(c) == 0 for c in changes])
    if is_base.any():
        base_metrics = np.nanmean(Y[is_base], axis=0)
    else:
        log.warning("Нет базовых прогонов (changes=[]) — база оценена медианой по всем сценариям")
        base_metrics = np.nanmedian(Y, axis=0)
    return TrafficDataset(network=network, topo=topo, base_state=base_state, scenario_ids=s_ids,
                          changes=changes, Y=Y, base_metrics=base_metrics.astype(np.float64),
                          is_base=is_base, source=source,
                          path=paths[0].parent)


def load_dataset(source: str = "auto", synth_dir: Path = SYNTH_DIR, sim_dir: Path = SIM_DATA_DIR,
                 n_scenarios: int = 3000, network: str = "auto") -> TrafficDataset:
    """source: auto | sim | synthetic."""
    if source in ("auto", "sim"):
        paths = find_sim_dataset(sim_dir)
        if paths:
            log.info("Датасет SUMO: %s (%d файлов)", sim_dir, len(paths))
            return read_parquet_dataset(paths, load_network(sim_dir / "network.json"), "sim")
        if source == "sim":
            raise FileNotFoundError(f"В {sim_dir} нет network.json и *.parquet")
        log.info("Датасета sim/ ещё нет — используется синтетический (%s)", synth_dir)
    if not (synth_dir / "edge_metrics.parquet").exists():
        from ml.stub_network import resolve_network
        from ml.synthetic import generate

        log.info("Синтетический датасет не найден — генерируем (%d сценариев)…", n_scenarios)
        net, _ = resolve_network(network)
        generate(net, synth_dir, n_scenarios)
    ds = read_parquet_dataset(sorted(synth_dir.glob("*.parquet")),
                              load_network(synth_dir / "network.json"), "synthetic")
    meta_path = synth_dir / "meta.json"
    if meta_path.exists():
        ds.meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return ds
