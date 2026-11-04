"""Подключение настоящих ml/predict.py и agents/evaluate.py и откат на заглушки."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.services.integrations import Integrations

NET = {
    "nodes": [{"id": "a", "lon": 69.10, "lat": 54.86, "type": "priority"},
              {"id": "b", "lon": 69.11, "lat": 54.86, "type": "traffic_light"}],
    "edges": [{"id": "ab", "from": "a", "to": "b", "length": 640, "lanes": 1, "speed": 13.9, "priority": 5,
               "geometry": [[69.10, 54.86], [69.11, 54.86]]},
              {"id": "ba", "from": "b", "to": "a", "length": 640, "lanes": 1, "speed": 13.9, "priority": 5,
               "geometry": [[69.11, 54.86], [69.10, 54.86]]}],
}


def _settings(tmp_path: Path, mode: str = "auto") -> Settings:
    s = Settings()
    s.ml_dir = tmp_path / "ml"
    s.agents_dir = tmp_path / "agents"
    s.integration_mode = mode
    return s


def test_real_modules_are_loaded(tmp_path):
    (tmp_path / "ml").mkdir()
    # собственный импорт модуля из своей папки должен работать
    (tmp_path / "ml" / "helper.py").write_text("OCC = 0.42\n")
    (tmp_path / "ml" / "predict.py").write_text(
        "from helper import OCC\n"
        "MODEL_INFO = {'name': 'GNN-test', 'version': '7', 'metrics': {'mae': 0.01}}\n"
        "def predict(network, changes):\n"
        "    return {e['id']: {'occupancy': OCC, 'mean_speed': 10.0, 'travel_time': 64.0} for e in network['edges']}\n"
    )
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "evaluate.py").write_text(
        "import json\n"
        "def evaluate(network_path, mode):\n"
        "    net = json.load(open(network_path))\n"
        "    return {'mode': mode, 'edges': len(net['edges']), 'mean_waiting_time': 5.0}\n"
    )
    integ = Integrations(cfg=_settings(tmp_path))
    out, source, warnings = integ.predict(NET, [])
    assert source == "ml" and not warnings and out["ab"]["occupancy"] == 0.42
    info = integ.model_info()
    assert info["name"] == "GNN-test" and info["is_stub"] is False

    p = tmp_path / "net.json"
    p.write_text(json.dumps(NET))
    metrics, source, _ = integ.evaluate(p, "rl")
    assert source == "agents" and metrics == {"mode": "rl", "edges": 2, "mean_waiting_time": 5.0}


def test_fallback_to_stub_on_error(tmp_path):
    (tmp_path / "ml").mkdir()
    (tmp_path / "ml" / "predict.py").write_text("def predict(network, changes):\n    raise RuntimeError('нет весов')\n")
    integ = Integrations(cfg=_settings(tmp_path))
    out, source, warnings = integ.predict(NET, [])
    assert source == "stub" and "нет весов" in warnings[0]
    assert set(out) == {"ab", "ba"}


def test_missing_modules_use_stubs(tmp_path):
    integ = Integrations(cfg=_settings(tmp_path))
    assert integ.predictor.source == "stub" and integ.evaluator.source == "stub"
    _, _, warnings = integ.predict(NET, [])
    assert "не найден" in warnings[0]
