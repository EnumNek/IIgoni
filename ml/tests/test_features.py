import numpy as np

from ml.features import FEATURE_NAMES, FeatureBuilder, inverse_targets, line_graph, transform_targets
from ml.network import apply_changes, parse_network


def test_line_graph_no_uturns(grid_network):
    topo, _ = parse_network(grid_network)
    lg = line_graph(topo)
    a, b = lg
    assert (topo.dst[a] == topo.src[b]).all()  # с a можно выехать на b
    uturn = (topo.dst[b] == topo.src[a]) & (topo.src[b] == topo.dst[a])
    # разворот допускается только там, где нет другого продолжения (тупик на границе)
    assert (topo.fringe[topo.dst[a[uturn]]]).all()


def test_target_roundtrip(grid_network):
    topo, base = parse_network(grid_network)
    rng = np.random.default_rng(0)
    m = np.stack([rng.uniform(0, 0.5, topo.n_edges), base.speed * rng.uniform(0.3, 1, topo.n_edges),
                  topo.length / base.speed * rng.uniform(1, 3, topo.n_edges)], 1)
    back = inverse_targets(transform_targets(m, topo, base), topo, base)
    assert np.allclose(back, m, rtol=1e-6)


def test_features_and_context(grid_network):
    topo, base = parse_network(grid_network)
    fb = FeatureBuilder(topo, base)  # базовые метрики неизвестны -> свободный поток
    st = apply_changes(topo, base, [{"type": "close", "target_id": "B1B2", "value": None}])
    x = fb.features(st)
    assert x.shape == (topo.n_edges, len(FEATURE_NAMES)) and np.isfinite(x).all()
    i = topo.edge_pos["B1B2"]
    assert x[i, FEATURE_NAMES.index("closed")] == 1 and x[i, FEATURE_NAMES.index("changed")] == 1
    c = fb.context(st)
    assert c[i, 4] == 0 and c[i, 5] == 0  # само изменённое ребро — 0 переходов
    assert c[:, 0].max() == 1  # одно закрытие в сценарии


def test_dilated_rings_exact(grid_network):
    """Связи «ровно через d переходов» совпадают с кратчайшими расстояниями в line graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    from ml.features import dilated_graphs

    topo, _ = parse_network(grid_network)
    lg = line_graph(topo)
    E = topo.n_edges
    D = shortest_path(csr_matrix((np.ones(lg.shape[1]), (lg[0], lg[1])), shape=(E, E)), unweighted=True)
    rings = dilated_graphs(lg, E, (1, 2, 4))
    for d, ei in rings.items():
        assert (D[ei[0], ei[1]] == d).all()
        assert ei.shape[1] == int((D == d).sum())
