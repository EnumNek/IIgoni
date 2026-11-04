"""Быстрые проверки части agents/ (запуск: python -m pytest agents/tests -q).

Полный прогон — около минуты: каждая симуляция 2×2 с libsumo занимает
несколько секунд.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents import config  # noqa: E402
from agents.dqn import IndependentDQN  # noqa: E402
from agents.env import ApproachObservation, MaxWaitGuard, make_env  # noqa: E402
from agents.evaluate import evaluate, run_episode  # noqa: E402
from agents.networks import get_scenario  # noqa: E402


@pytest.fixture(scope="module")
def grid2x2():
    return get_scenario("grid2x2")


def test_network_json_contract(grid2x2):
    data = json.loads(grid2x2.network_json.read_text(encoding="utf-8"))
    assert {"nodes", "edges"} <= set(data)
    assert {"id", "lon", "lat", "type"} <= set(data["nodes"][0])
    assert {"id", "from", "to", "length", "lanes", "speed", "priority", "geometry"} <= set(data["edges"][0])
    assert sum(n["type"] == "traffic_light" for n in data["nodes"]) == 4


def test_scenario_resolution(grid2x2):
    assert get_scenario(grid2x2.network_json).key == "grid2x2"
    assert get_scenario(grid2x2.net_file).key == "grid2x2"


def test_actuated_programs(grid2x2):
    text = grid2x2.actuated_file.read_text(encoding="utf-8")
    assert text.count('type="actuated"') == 4
    assert 'minDur="5' in text and 'maxDur="50' in text


def test_observation(grid2x2):
    env = make_env(grid2x2, grid2x2.eval_route("nominal", config.EVAL_SEEDS[0]), "rl", seed=1)
    try:
        obs = env.reset()
        assert set(obs) == {"A0", "A1", "B0", "B1"}
        for ts, o in obs.items():
            fn: ApproachObservation = env.traffic_signals[ts].observation_fn
            # 4 зелёные фазы + флаг + 4 стороны × 2 группы полос × (очередь, ожидание)
            assert o.shape == (4 + 1 + 16,)
            assert len(fn.layout()) == o.shape[0]
            # у каждой стороны решётки есть и «прямые», и левоповоротные полосы
            assert all(len(g) > 0 for g in fn._groups)
        for _ in range(60):  # 5 минут симуляции
            obs, rewards, dones, _ = env.step({ts: 0 for ts in obs})
        o = np.stack(list(obs.values()))
        assert (o >= 0).all() and (o <= 1).all()
        assert all(r <= 0 for r in rewards.values())  # награда = минус очередь
        assert o[:, 5:].sum() > 0  # машины видны в наблюдении
    finally:
        env.close()


def test_max_wait_guard(grid2x2):
    """Если «агент» держит одну фазу бесконечно, правило обязано переключить её."""
    env = make_env(grid2x2, grid2x2.eval_route("nominal", config.EVAL_SEEDS[0]), "rl", seed=1)
    try:
        obs = env.reset()
        guard = MaxWaitGuard(env, max_wait=60)
        for _ in range(60):  # 5 минут «вечного» зелёного фазе 0
            obs, _, _, _ = env.step(guard({ts: 0 for ts in obs}))
        assert guard.interventions > 0
        assert 0 < guard.rate < 1
        # отключённое правило (max_wait=None) действия не меняет
        assert MaxWaitGuard(env, max_wait=None)({"A0": 0}) == {"A0": 0}
    finally:
        env.close()


def test_independent_dqn_save_load(tmp_path, grid2x2):
    from gymnasium import spaces

    obs_space = spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32)
    act_space = spaces.Discrete(3)
    params = dict(config.DQN_PARAMS, learning_starts=10, batch_size=8)
    agents = IndependentDQN.create({"a": obs_space, "b": obs_space}, {"a": act_space, "b": act_space},
                                   total_timesteps=50, params=params)
    rng = np.random.default_rng(0)
    obs = {k: rng.random(7, dtype=np.float32) for k in ("a", "b")}
    for _ in range(30):
        act = agents.act(obs, deterministic=False)
        nxt = {k: rng.random(7, dtype=np.float32) for k in obs}
        agents.observe(obs, act, {k: -1.0 for k in obs}, nxt, done=False)
        obs = nxt
    assert agents.models["a"].num_timesteps == 30
    agents.save(tmp_path, meta={"scenario": "test"})
    loaded = IndependentDQN.load(tmp_path)
    assert loaded.act(obs) == agents.act(obs)


def test_evaluate_fixed_contract(grid2x2):
    res = evaluate("grid2x2", "fixed", seeds=[config.EVAL_SEEDS[0]])
    for key in ("avg_travel_time", "avg_waiting_time", "avg_queue_length", "p95_waiting_time",
                "throughput", "per_tls", "guard_rate"):
        assert key in res
    assert res["avg_travel_time"] > 0 and res["throughput"] > 0
    assert set(res["per_tls"]) == {"A0", "A1", "B0", "B1"}


def test_actuated_differs_from_fixed(grid2x2):
    route = grid2x2.eval_route("nominal", config.EVAL_SEEDS[0])
    fixed = run_episode(grid2x2, route, "fixed", seed=1)
    actuated = run_episode(grid2x2, route, "actuated", seed=1)
    assert fixed["vehicles"] == actuated["vehicles"]  # один и тот же спрос
    assert fixed["avg_waiting_time"] != actuated["avg_waiting_time"]


@pytest.mark.skipif(not (config.MODELS_DIR / "grid2x2" / "meta.json").exists(),
                    reason="модели grid2x2 не обучены")
def test_evaluate_rl(grid2x2):
    res = evaluate(grid2x2.network_json, "rl", seeds=[config.EVAL_SEEDS[0]])
    assert res["mode"] == "rl" and res["avg_travel_time"] > 0
    assert res["teleports"] == 0


def test_env_params_override(grid2x2):
    """Параметры среды модели (из meta.json) переопределяют config.ENV_PARAMS."""
    env = make_env(grid2x2, grid2x2.eval_route("nominal", config.EVAL_SEEDS[0]), "rl", seed=1,
                   env_params={"min_green": 10})
    try:
        assert env.min_green == 10 and env.delta_time == config.ENV_PARAMS["delta_time"]
    finally:
        env.close()


def test_evaluate_reports_avg_speed(grid2x2):
    res = evaluate("grid2x2", "fixed", seeds=[config.EVAL_SEEDS[0]])
    assert 0 < res["avg_speed"] < 14 and res["rl_fallback_tls"] == []
    assert all(v["avg_delay_rate"] >= v["avg_queue_length"] * 0.5 for v in res["per_tls"].values())


@pytest.mark.skipif(not (config.MODELS_DIR / "grid2x2" / "meta.json").exists(),
                    reason="модели grid2x2 не обучены")
def test_rl_falls_back_to_fixed_cycle_without_model(tmp_path, grid2x2):
    """Светофор без подходящей модели (например, новый) работает по циклу программы сети,
    остальными управляют агенты."""
    import shutil

    src = config.MODELS_DIR / "grid2x2"
    for f in src.iterdir():
        shutil.copy(f, tmp_path / f.name)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    del meta["agents"]["B1"]
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    res = evaluate("grid2x2", "rl", seeds=[config.EVAL_SEEDS[0]], models_dir=tmp_path)
    assert res["rl_fallback_tls"] == ["B1"]
    assert res["avg_travel_time"] > 0 and res["throughput"] > 0


SIM_JSON = config.SIM_DATA_DIR / "network.json"


@pytest.mark.skipif(not SIM_JSON.exists(), reason="сети sim/ нет")
def test_sim_variant_from_backend_json(tmp_path):
    """Сеть сценария от backend/ (network.json + meta.base_network/changes) собирается из
    base.net.xml патчем sim/: id светофоров те же, модели и спрос — от сети sim/."""
    import sumolib

    from agents.networks import find_sim_network

    base = json.loads(SIM_JSON.read_text(encoding="utf-8"))
    edge = next(e["id"] for e in base["edges"] if e["lanes"] == 1)
    payload = {**base, "meta": {**base["meta"], "base_network": str(SIM_JSON),
                                "changes": [{"type": "close", "target_id": edge, "value": None}]}}
    path = tmp_path / "network_scenario.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    sc = get_scenario(path)
    assert sc.key.startswith("sim-") and sc.models_dir == config.MODELS_DIR / "sim" and sc.demand == "sim"
    assert "--junction-taz" in sc.sumo_args
    net = sumolib.net.readNet(str(sc.net_file))
    assert not net.getEdge(edge).allows("passenger")  # закрыт, но не удалён
    base_tls = {t.getID() for t in sumolib.net.readNet(str(find_sim_network())).getTrafficLights()}
    assert {t.getID() for t in net.getTrafficLights()} == base_tls
    assert get_scenario(path).key == sc.key  # повторный вызов — та же сеть (кэш)
    # без изменений — базовая сеть sim/
    payload["meta"]["changes"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert get_scenario(path).key == "sim"
