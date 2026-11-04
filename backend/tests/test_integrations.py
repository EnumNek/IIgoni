"""Подключение настоящих ml/predict.py и agents/evaluate.py и откат на заглушки."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.services.changes import ChangeError, apply_changes, normalize_change
from app.services.forecasting import apply_signal_correction
from app.services.integrations import Integrations, normalize_agent_metrics

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


# ---------------------------------------------------------------- форматы соседних частей
AGENTS_FIXED = {  # так отвечает agents/evaluate.py (ключи avg_*, per_tls — id светофоров SUMO)
    "mode": "fixed", "avg_travel_time": 268.0, "avg_waiting_time": 97.0, "avg_queue_length": 2.2,
    "avg_speed": 7.0, "per_tls": {"b": {"avg_queue_length": 4.0}, "joined_c": {"avg_queue_length": 1.0}},
}
AGENTS_RL = {
    "mode": "rl", "avg_travel_time": 230.0, "avg_waiting_time": 60.0, "avg_queue_length": 1.1,
    "avg_speed": 8.1, "per_tls": {"b": {"avg_queue_length": 2.0}, "joined_c": {"avg_queue_length": 1.5}},
}


def test_agent_metrics_are_normalized():
    net = {"nodes": [{"id": "b", "tl_id": "b"}, {"id": "c1", "tl_id": "joined_c"}, {"id": "c2", "tl_id": "joined_c"}]}
    out = normalize_agent_metrics(AGENTS_FIXED, net)
    assert out["mean_waiting_time"] == 97.0 and out["mean_travel_time"] == 268.0
    assert out["mean_queue_length"] == 2.2 and out["mean_speed"] == 7.0
    assert out["avg_waiting_time"] == 97.0  # исходные ключи сохраняются
    # светофор объединённого перекрёстка -> все его узлы
    assert set(out["per_tls"]) == {"b", "c1", "c2"}
    assert out["per_tls"]["c2"]["mean_queue_length"] == 1.0
    # формат заглушки не меняется
    stub = {"mean_waiting_time": 5.0, "per_tls": {"b": {"mean_waiting_time": 3.0}}}
    assert normalize_agent_metrics(stub, net) == stub


def test_signal_correction_prefers_delay_rate():
    fixed = {**AGENTS_FIXED, "per_tls": {"b": {"avg_queue_length": 4.0, "avg_delay_rate": 6.0}}}
    rl = {**AGENTS_RL, "per_tls": {"b": {"avg_queue_length": 0.4, "avg_delay_rate": 3.0}}}
    results = {"ab": {"occupancy": 0.2, "mean_speed": 5.0, "travel_time": 128.0, "closed": False}}
    factors = apply_signal_correction(NET, results, normalize_agent_metrics(fixed, NET),
                                      normalize_agent_metrics(rl, NET))
    assert factors == {"b": pytest.approx(0.5)}  # по задержке, а не по очереди (0,1)


def test_signal_correction_uses_agents_queue_per_tls():
    """По светофорам agents/ даёт очередь, а не ожидание: при одинаковом спросе их отношения
    совпадают (формула Литтла), поэтому поправка считается по очереди."""
    results = {"ab": {"occupancy": 0.2, "mean_speed": 5.0, "travel_time": 128.0, "closed": False},
               "ba": {"occupancy": 0.1, "mean_speed": 10.0, "travel_time": 64.0, "closed": False}}
    factors = apply_signal_correction(NET, results, normalize_agent_metrics(AGENTS_FIXED, NET),
                                      normalize_agent_metrics(AGENTS_RL, NET))
    assert factors == {"b": pytest.approx(0.5)}
    free = 640 / 13.9
    assert results["ab"]["travel_time"] == pytest.approx(free + (128.0 - free) * 0.5)
    assert results["ba"]["travel_time"] == 64.0  # ребро ведёт к узлу без светофора


def test_model_info_from_ml_predict(tmp_path):
    """ml/predict.py отдаёт сведения о модели функцией model_info() в своём формате."""
    (tmp_path / "ml" / "weights").mkdir(parents=True)
    (tmp_path / "ml" / "weights" / "gnn.pt").write_bytes(b"")
    (tmp_path / "ml" / "predict.py").write_text(
        "from pathlib import Path\n"
        "WEIGHTS_DIR = Path(__file__).parent / 'weights'\n"
        "def predict(network, changes):\n    return {}\n"
        "def model_info():\n"
        "    return {'model': {'variant': 'sage', 'conv': 'sage', 'title': 'GraphSAGE'},\n"
        "            'dataset': {'source': 'sim', 'n_scenarios': 301},\n"
        "            'created_utc': '2026-11-04T12:00:00+00:00', 'test_metrics': {'sage': {'occupancy': {'mae': 0.003}}}}\n"
    )
    info = Integrations(cfg=_settings(tmp_path)).model_info()
    assert info["name"] == "GNN sage" and info["version"] == "2026-11-04T12:00:00+00:00"
    assert info["architecture"] == "GraphSAGE" and info["metrics"]["dataset"]["source"] == "sim"
    assert info["weights_path"].endswith("gnn.pt") and info["is_stub"] is False


def test_evaluate_seeds_passed_only_if_supported(tmp_path):
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "evaluate.py").write_text(
        "def evaluate(network_path, mode, *, seeds=None):\n"
        "    return {'mode': mode, 'avg_waiting_time': 10.0, 'seeds': list(seeds or [])}\n"
    )
    cfg = _settings(tmp_path)
    cfg.agents_eval_seeds = [101]
    p = tmp_path / "net.json"
    p.write_text(json.dumps(NET))
    metrics, source, _ = Integrations(cfg=cfg).evaluate(p, "fixed", network=NET)
    assert source == "agents" and metrics["seeds"] == [101] and metrics["mean_waiting_time"] == 10.0


@pytest.mark.parametrize("value,expected", [
    (True, True), ("static", True), ("on", True), ("add", True), ("actuated", "actuated"),
    (False, False), ("off", False), ("none", False), (None, False),
])
def test_traffic_light_values_follow_sim_contract(value, expected):
    nodes = {"b": NET["nodes"][1]}
    ch = normalize_change({"type": "traffic_light", "target_id": "b", "value": value}, {}, nodes)
    assert ch["value"] == expected
    out = apply_changes(NET, [ch])
    assert out["nodes"][1]["type"] == ("priority" if expected is False else "traffic_light")


def test_lanes_range_follows_sim_contract():
    edges = {e["id"]: e for e in NET["edges"]}
    assert normalize_change({"type": "lanes", "target_id": "ab", "value": 6}, edges, {})["value"] == 6
    with pytest.raises(ChangeError):
        normalize_change({"type": "lanes", "target_id": "ab", "value": 7}, edges, {})
