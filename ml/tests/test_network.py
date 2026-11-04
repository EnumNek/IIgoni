import numpy as np
import pytest

from ml.network import apply_changes, parse_network


def test_parse_grid(grid_network):
    topo, state = parse_network(grid_network)
    assert topo.n_edges == 80 and topo.n_nodes == 32
    assert state.tl.sum() == 7  # магистрали: столбец B и строка 2
    assert topo.fringe.sum() == 16  # подъезды по периметру
    assert set(np.unique(state.lanes)) == {1.0, 2.0}


def test_parse_fills_missing_fields():
    net = {"nodes": [{"id": "a", "lon": 69.0, "lat": 54.0}],
           "edges": [{"id": "e", "from": "a", "to": "b"}]}  # узла b нет, длины/полос нет
    topo, state = parse_network(net)
    assert topo.n_nodes == 2 and topo.length[0] > 0
    assert state.lanes[0] == 1 and state.speed[0] > 0


def test_apply_changes_semantics(grid_network):
    topo, base = parse_network(grid_network)
    st = apply_changes(topo, base, [
        {"type": "lanes", "target_id": "A0A1", "value": 3},
        {"type": "speed", "target_id": "A1A2", "value": 60},  # км/ч -> м/с
        {"type": "close", "target_id": "A2A3", "value": None},
        {"type": "close", "target_id": "A3A2", "value": None},
        {"type": "open", "target_id": "A3A2", "value": None},
        {"type": "traffic_light", "target_id": "A0", "value": True},
        {"type": "traffic_light", "target_id": "B1", "value": "remove"},
        {"type": "traffic_light", "target_id": "C0", "value": None},  # переключение
    ])
    e = topo.edge_pos
    assert st.lanes[e["A0A1"]] == 3
    assert st.speed[e["A1A2"]] == pytest.approx(60 / 3.6)
    assert st.closed[e["A2A3"]] and not st.closed[e["A3A2"]]
    n = topo.node_pos
    assert st.tl[n["A0"]] and not st.tl[n["B1"]] and st.tl[n["C0"]] != base.tl[n["C0"]]
    # базовое состояние не изменилось
    assert not base.closed.any() and base.lanes[e["A0A1"]] == 1


@pytest.mark.parametrize("change", [
    {"type": "widen", "target_id": "A0A1", "value": 2},
    {"type": "close", "target_id": "нет-такого", "value": None},
    {"type": "lanes", "target_id": "A0A1", "value": 0},
    {"type": "speed", "target_id": "A0A1", "value": "быстро"},
    {"type": "traffic_light", "target_id": "A0A1", "value": True},  # это ребро, а не узел
    {"type": "close", "value": None},
])
def test_invalid_changes(grid_network, change):
    topo, base = parse_network(grid_network)
    with pytest.raises(ValueError):
        apply_changes(topo, base, [change])
