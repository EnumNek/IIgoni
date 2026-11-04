"""Сквозной тест конвейера на синтетической решётке 4×4 (нужен SUMO, ~10–20 с).

Проверяет: netgenerate → network.json → randomTrips → сценарий с изменениями
всех типов → netconvert-патч → SUMO → метрики рёбер по контракту.
"""
import json
import math
import random

import pytest

pytest.importorskip("sumo")

from sim.build_network import build  # noqa: E402
from sim.generate_traffic import generate_trips  # noqa: E402
from sim.scenarios import ChangeSampler, ScenarioSpec, edge_adjacency, run_scenario  # noqa: E402


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    d = tmp_path_factory.mktemp("grid")
    net_file, json_file = d / "grid.net.xml", d / "network.json"
    network = build(force_grid=True, out_net=net_file, out_json=json_file)
    trips = generate_trips(net_file, d / "trips.xml", rate=900)
    return network, net_file, trips, json_file


def test_network_json_contract(grid):
    network, _, _, json_file = grid
    on_disk = json.loads(json_file.read_text(encoding="utf-8"))
    assert on_disk["meta"]["source"] == "netgenerate"
    node_ids = {n["id"] for n in on_disk["nodes"]}
    for n in on_disk["nodes"]:
        assert {"id", "lon", "lat", "type"} <= n.keys()
    for e in on_disk["edges"]:
        assert {"id", "from", "to", "length", "lanes", "speed", "priority", "geometry"} <= e.keys()
        assert e["from"] in node_ids and e["to"] in node_ids
        assert len(e["geometry"]) >= 2 and all(len(p) == 2 for p in e["geometry"])
    assert on_disk["meta"]["n_traffic_lights"] == 16


def test_scenario_end_to_end(grid):
    network, net_file, trips, _ = grid
    edges = [e for e in network["edges"]]
    tl_node = next(n["id"] for n in network["nodes"] if n["type"] == "traffic_light")
    inner = [e for e in edges if e["from"] in {n["id"] for n in network["nodes"] if n["type"] == "traffic_light"}
             and e["to"] in {n["id"] for n in network["nodes"] if n["type"] == "traffic_light"}]
    changes = [
        {"type": "lanes", "target_id": inner[0]["id"], "value": 1},
        {"type": "speed", "target_id": inner[1]["id"], "value": 8.33},
        {"type": "close", "target_id": inner[2]["id"], "value": None},
        {"type": "traffic_light", "target_id": tl_node, "value": "actuated"},
    ]
    edge_rows, node_rows, summary = run_scenario(ScenarioSpec(1, changes), network,
                                                 base_net=net_file, trips=trips)
    assert len(edge_rows) == len(edges)
    assert summary["veh_inserted"] > 0
    rows = {r["edge_id"]: r for r in edge_rows}
    closed = rows[inner[2]["id"]]
    assert closed["closed"] and closed["occupancy"] == 0 and math.isnan(closed["mean_speed"])
    assert rows[inner[0]["id"]]["lanes"] == 1 and rows[inner[1]["id"]]["speed_limit"] == 8.33
    for r in edge_rows:
        if not r["closed"]:
            assert 0.0 <= r["occupancy"] <= 1.0
            assert r["mean_speed"] >= 0 and r["travel_time"] > 0
    assert sum(r["changed"] for r in edge_rows) == 3
    assert any(n["changed"] and n["tl_type"] == "actuated" for n in node_rows)


def test_sampler_produces_valid_changes(grid):
    network, net_file, _, _ = grid
    sampler = ChangeSampler(network, edge_adjacency(net_file))
    for seed in range(20):
        changes = sampler.sample(random.Random(seed))
        assert 1 <= len(changes) <= 5
        closed = {c["target_id"] for c in changes if c["type"] == "close"}
        reopened = {c["target_id"] for c in changes if c["type"] == "open"}
        assert sampler.keeps_connectivity(closed - reopened)
