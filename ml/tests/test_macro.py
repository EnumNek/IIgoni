import numpy as np

from ml.macro import MacroModel
from ml.network import apply_changes, parse_network


def test_closure_reroutes_traffic(grid_network):
    topo, base = parse_network(grid_network)
    m = MacroModel(topo, base, seed=0)
    m.calibrate()
    r0 = m.assign(base)
    e = int(np.nanargmax(r0["volume"]))
    st = apply_changes(topo, base, [{"type": "close", "target_id": topo.edge_ids[e], "value": None}])
    r1 = m.assign(st)
    assert np.isnan(r1["occupancy"][e])  # закрытое ребро не измеряется
    assert r1["unserved"] == 0  # сеть осталась связной — весь спрос обслужен
    others = np.arange(topo.n_edges) != e
    assert np.nanmax(r1["volume"][others] - r0["volume"][others]) > 50  # поток ушёл в объезд
    for r in (r0, r1):
        ok = np.isfinite(r["occupancy"])
        assert (r["occupancy"][ok] >= 0).all() and (r["occupancy"][ok] <= 1).all()
        assert (r["travel_time"][ok] >= topo.length[ok] / st.speed[ok] - 1e-9).all()
