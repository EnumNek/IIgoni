"""Проверка весов, закоммиченных в ml/weights (то, что получит backend)."""
import pytest

from ml.config import WEIGHTS_DIR

from .conftest import load_json

pytestmark = pytest.mark.skipif(not (WEIGHTS_DIR / "gnn.pt").exists(), reason="веса ещё не обучены")


def test_default_predict():
    from ml.predict import model_info, predict

    net = load_json(WEIGHTS_DIR / "network.json")
    eid = net["edges"][0]["id"]
    out = predict(net, [{"type": "close", "target_id": eid, "value": None}])
    assert len(out) == len(net["edges"]) and out[eid]["closed"]
    assert model_info()["model"]["conv"] in ("sage", "gat")
    assert model_info()["model"]["variant"]
