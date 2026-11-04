"""Юнит-тесты контракта изменений (без SUMO, выполняются за доли секунды)."""
import copy

import pytest

from sim.changes import ChangeError, apply_changes, diff_states, to_netconvert_patches, validate_changes


@pytest.fixture()
def network():
    return {
        "meta": {},
        "nodes": [
            {"id": "A", "lon": 0, "lat": 0, "type": "priority", "tl_id": None, "tl_type": None},
            {"id": "B", "lon": 1, "lat": 0, "type": "traffic_light", "tl_id": "B", "tl_type": "static"},
            {"id": "C", "lon": 2, "lat": 0, "type": "dead_end", "tl_id": None, "tl_type": None},
        ],
        "edges": [
            {"id": "AB", "from": "A", "to": "B", "length": 100.0, "lanes": 1, "speed": 13.89,
             "priority": 3, "geometry": [[0, 0], [1, 0]], "closed": False},
            {"id": "BC", "from": "B", "to": "C", "length": 100.0, "lanes": 2, "speed": 16.67,
             "priority": 3, "geometry": [[1, 0], [2, 0]], "closed": False},
        ],
    }


def test_apply_does_not_mutate_input(network):
    original = copy.deepcopy(network)
    apply_changes(network, [{"type": "lanes", "target_id": "AB", "value": 3}])
    assert network == original


def test_each_change_type(network):
    out = apply_changes(network, [
        {"type": "lanes", "target_id": "AB", "value": 3},
        {"type": "speed", "target_id": "BC", "value": 8.33},
        {"type": "close", "target_id": "BC", "value": None},
        {"type": "traffic_light", "target_id": "A", "value": "actuated"},
        {"type": "traffic_light", "target_id": "B", "value": "off"},
    ])
    e = {x["id"]: x for x in out["edges"]}
    n = {x["id"]: x for x in out["nodes"]}
    assert e["AB"]["lanes"] == 3 and e["AB"]["changed"]
    assert e["BC"]["speed"] == 8.33 and e["BC"]["closed"]
    assert n["A"]["type"] == "traffic_light" and n["A"]["tl_type"] == "actuated" and n["A"]["changed"]
    assert n["B"]["type"] == "priority" and n["B"]["tl_type"] is None


def test_close_then_open_is_noop(network):
    out = apply_changes(network, [{"type": "close", "target_id": "AB", "value": None},
                                  {"type": "open", "target_id": "AB", "value": None}])
    edge = next(x for x in out["edges"] if x["id"] == "AB")
    assert edge["closed"] is False and "changed" not in edge
    assert diff_states(network, out) == {"edges": {}, "nodes": {}}
    assert to_netconvert_patches(diff_states(network, out)) == (None, None)


def test_open_edge_closed_in_input_network(network):
    network["edges"][0]["closed"] = True
    out = apply_changes(network, [{"type": "open", "target_id": "AB", "value": None}])
    assert diff_states(network, out)["edges"] == {"AB": {"closed": False}}
    edg, _ = to_netconvert_patches(diff_states(network, out))
    assert 'id="AB"' in edg and "allow=" in edg


@pytest.mark.parametrize("value,expected", [(True, "static"), ("on", "static"), (False, "off"),
                                            (None, "off"), ("none", "off"), ("ACTUATED", "actuated")])
def test_traffic_light_aliases(network, value, expected):
    assert validate_changes(network, [{"type": "traffic_light", "target_id": "A", "value": value}])[0]["value"] == expected


@pytest.mark.parametrize("change", [
    {"type": "unknown", "target_id": "AB", "value": 1},
    {"type": "lanes", "target_id": "XX", "value": 2},
    {"type": "lanes", "target_id": "AB", "value": 0},
    {"type": "lanes", "target_id": "AB", "value": 2.5},
    {"type": "lanes", "target_id": "AB", "value": True},
    {"type": "speed", "target_id": "AB", "value": 100},
    {"type": "speed", "target_id": "AB", "value": "fast"},
    {"type": "traffic_light", "target_id": "C", "value": "static"},   # тупик
    {"type": "traffic_light", "target_id": "AB", "value": "static"},  # это ребро, а не узел
    {"type": "traffic_light", "target_id": "A", "value": "blinking"},
])
def test_invalid_changes_rejected(network, change):
    with pytest.raises(ChangeError):
        validate_changes(network, [change])


def test_patches(network):
    out = apply_changes(network, [
        {"type": "lanes", "target_id": "AB", "value": 2},
        {"type": "speed", "target_id": "AB", "value": 11.111},
        {"type": "close", "target_id": "BC", "value": None},
        {"type": "traffic_light", "target_id": "A", "value": "static"},
        {"type": "traffic_light", "target_id": "B", "value": "actuated"},
    ])
    edg, nod = to_netconvert_patches(diff_states(network, out))
    assert '<edge id="AB" numLanes="2" speed="11.11"/>' in edg
    assert '<edge id="BC" disallow="all"/>' in edg
    assert '<node id="A" type="traffic_light" tlType="static"/>' in nod
    assert '<node id="B" type="traffic_light" tlType="actuated"/>' in nod


def test_same_state_produces_no_diff(network):
    out = apply_changes(network, [{"type": "lanes", "target_id": "AB", "value": 1},
                                  {"type": "traffic_light", "target_id": "B", "value": "static"},
                                  {"type": "traffic_light", "target_id": "A", "value": "off"}])
    assert diff_states(network, out) == {"edges": {}, "nodes": {}}
