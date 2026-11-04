"""Сквозной тест: обучение (--quick) -> веса -> predict по контракту."""
import subprocess
import sys

import numpy as np
import pytest

from ml.network import parse_network
from ml.predict import Predictor

from .conftest import ROOT, load_json


@pytest.fixture(scope="module")
def predictor(trained_weights):
    return Predictor(trained_weights / "weights")


@pytest.fixture(scope="module")
def ref_network(trained_weights):
    return load_json(trained_weights / "weights" / "network.json")


def test_reports_written(trained_weights):
    rep = trained_weights / "reports"
    for name in ("metrics.json", "metrics.md", "history.json", "comparison.png", "training_curves.png"):
        assert (rep / name).exists(), name
    m = load_json(rep / "metrics.json")["metrics"]
    assert {"nochange", "linear", "gb", "sage"} <= set(m)
    for key in m:
        assert {"mae", "rmse", "r2", "delta_r2"} <= set(m[key]["occupancy"])


def test_predict_contract(predictor, ref_network):
    out = predictor.predict(ref_network, [{"type": "close", "target_id": "B1B2", "value": None},
                                          {"type": "lanes", "target_id": "A0A1", "value": 2}])
    assert set(out) == {e["id"] for e in ref_network["edges"]}
    assert out["B1B2"] == {"occupancy": 0.0, "mean_speed": 0.0, "travel_time": None, "closed": True}
    for eid, v in out.items():
        if eid == "B1B2":
            continue
        assert set(v) == {"occupancy", "mean_speed", "travel_time", "closed"}
        assert 0 <= v["occupancy"] <= 1 and v["mean_speed"] > 0 and v["travel_time"] > 0
        assert v["closed"] is False


def test_no_changes_equals_base(predictor, ref_network):
    out = predictor.predict(ref_network, [])
    ref = predictor.meta["reference"]
    base = dict(zip(ref["edge_ids"], ref["base_metrics"]))
    for eid, v in out.items():
        assert v["occupancy"] == pytest.approx(base[eid][0], abs=1e-3)
        assert v["travel_time"] == pytest.approx(base[eid][2], rel=1e-3)


def test_network_with_embedded_changes(predictor, ref_network):
    """Сеть, в которую правки уже внесены, даёт тот же прогноз, что эталон + changes."""
    import copy

    changes = [{"type": "lanes", "target_id": "C2D2", "value": 1},
               {"type": "speed", "target_id": "B0B1", "value": 8.33}]
    a = predictor.predict(ref_network, changes)
    net2 = copy.deepcopy(ref_network)
    for e in net2["edges"]:
        if e["id"] == "C2D2":
            e["lanes"] = 1
        if e["id"] == "B0B1":
            e["speed"] = 8.33
    b = predictor.predict(net2, [])
    assert a == b


def test_invalid_change_raises(predictor, ref_network):
    with pytest.raises(ValueError):
        predictor.predict(ref_network, [{"type": "teleport", "target_id": "A0A1", "value": 1}])


def test_other_network_works(predictor):
    from ml.stub_network import build_grid_python

    net = build_grid_python(6)  # другая сеть: модель индуктивна
    out = predictor.predict(net, [{"type": "close", "target_id": "C2C3", "value": None}])
    assert len(out) == len(net["edges"])
    vals = np.array([v["occupancy"] for v in out.values()])
    assert np.isfinite(vals).all()


def test_predict_cli(trained_weights, tmp_path, monkeypatch):
    """predict.py запускается и как модуль, и как скрипт."""
    net = trained_weights / "weights" / "network.json"
    code = ("import sys, ml.predict as p; from pathlib import Path; "
            f"p.WEIGHTS_DIR = Path(r'{trained_weights / 'weights'}'); "
            f"p._DEFAULT = p.Predictor(p.WEIGHTS_DIR); "
            f"p.main(['--network', r'{net}', '--changes', '[{{\"type\":\"close\",\"target_id\":\"B1B2\",\"value\":null}}]', "
            f"'--out', r'{tmp_path / 'f.json'}', '--top', '3'])")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "f.json").exists()
    r = subprocess.run([sys.executable, str(ROOT / "ml" / "predict.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
