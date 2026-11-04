import json

import numpy as np
import pandas as pd
import pytest

from ml.dataset import load_dataset, read_parquet_dataset


def test_synthetic_format(tiny_dataset):
    scen = pd.read_parquet(tiny_dataset / "scenarios.parquet")
    met = pd.read_parquet(tiny_dataset / "edge_metrics.parquet")
    assert {"scenario_id", "changes", "seed", "n_changes"} <= set(scen.columns)
    assert {"scenario_id", "edge_id", "occupancy", "mean_speed", "travel_time"} <= set(met.columns)
    assert met["occupancy"].between(0, 1).all()
    for ch in scen["changes"]:
        for c in json.loads(ch):
            assert set(c) == {"type", "target_id", "value"}


def test_load_and_split(tiny_dataset):
    ds = load_dataset("synthetic", synth_dir=tiny_dataset)
    assert ds.Y.shape == (63, 80, 3) and ds.is_base.sum() == 3
    tr, va, te = ds.split()
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert not ds.is_base[np.concatenate([tr, va, te])].any()
    assert len(tr) + len(va) + len(te) == 60


def test_sumo_like_layout(tmp_path, grid_network):
    """Один файл в стиле edgeData SUMO: синонимы колонок, occupancy в %, интервалы,
    внутренние рёбра ':...', колонка changes прямо в таблице метрик."""
    edges = [e["id"] for e in grid_network["edges"]]
    rows = []
    for sid, changes in [(0, []), (1, []), (2, [{"type": "close", "target_id": edges[0], "value": None}])]:
        for begin in (0, 900):  # два интервала агрегации
            for k, eid in enumerate(edges + [":A0_0"]):
                if sid == 2 and eid == edges[0]:
                    continue
                rows.append({"scenario": sid, "id": eid, "speed": 10.0 + sid, "traveltime": 20.0,
                             "occupancy": 10.0 + begin / 900, "begin": begin, "changes": json.dumps(changes)})
    pd.DataFrame(rows).to_parquet(tmp_path / "edgedata.parquet")
    ds = read_parquet_dataset([tmp_path / "edgedata.parquet"], grid_network)
    assert ds.Y.shape == (3, len(edges), 3)
    assert ds.Y[0, 0, 0] == pytest.approx(0.105)  # % -> доля, среднее по интервалам
    assert np.isnan(ds.Y[2, 0]).all()  # закрытое ребро без данных
    assert ds.base_metrics[0, 1] == pytest.approx(10.5)  # среднее двух базовых прогонов
    assert list(ds.is_base) == [True, True, False]
