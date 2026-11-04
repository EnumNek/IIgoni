"""Графовая нейросеть для прогноза метрик рёбер дорожной сети.

Архитектура (узлы графа — участки дорог, см. features.py):

    x (признаки участка) ─► MLP-кодировщик ─► K × DirectionalBlock ─► голова ─► Δ (3 метрики)

DirectionalBlock — слой GraphSAGE (или GATv2) с *раздельными* весами для
сообщений от участков выше по потоку (откуда на участок въезжают) и ниже по
потоку (куда с него выезжают):

    m = W_self·h + SAGE_up(h, A) + SAGE_down(h, Aᵀ) + W_g·mean(h по графу)
    h ← h + Dropout(GELU(LayerNorm(m)))

* Направление важно: затор распространяется против потока, а
  перераспределённый поток — по потоку; общий агрегатор их бы смешал.
* Остаточные связи и LayerNorm позволяют сделать сеть глубокой, не теряя
  локальную информацию.
* Расширенные (dilated) связи: слои по очереди смотрят на соседей через
  1, 1, 2, 2, 4, 4 перехода — рецептивное поле 14 переходов при 6 слоях
  (эффект закрытия улицы распространяется на 8–12 переходов).
* Глобальное среднее по графу — дешёвый «виртуальный узел»: даёт каждому
  участку сведения о сценарии в целом (например, сколько улиц закрыто).
* Модель предсказывает *изменение* нормированных метрик относительно
  базового состояния (Δ); итог = база + Δ. Так модель с самого начала не
  хуже бейзлайна «без изменений» и тратит ёмкость только на эффект изменений.

Сеть индуктивна: веса не зависят от размера графа, поэтому одна и та же
модель применяется к сети любого размера (это используется при замере
масштабируемости инференса).
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, SAGEConv


class DirectionalBlock(nn.Module):
    def __init__(self, dim: int, conv: str = "sage", heads: int = 4, dropout: float = 0.1,
                 global_ctx: bool = True):
        super().__init__()
        self.conv = conv
        if conv == "sage":
            self.up = SAGEConv(dim, dim, aggr="mean", root_weight=False)
            self.down = SAGEConv(dim, dim, aggr="mean", root_weight=False)
        elif conv == "gat":
            assert dim % heads == 0, "hidden должен делиться на число голов внимания"
            self.up = GATv2Conv(dim, dim // heads, heads=heads, add_self_loops=False)
            self.down = GATv2Conv(dim, dim // heads, heads=heads, add_self_loops=False)
        else:
            raise ValueError(f"Неизвестный тип свёртки: {conv}")
        self.self_lin = nn.Linear(dim, dim)
        self.glob = nn.Linear(dim, dim, bias=False) if global_ctx else None
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h, pack: dict, n_graphs: int):
        if self.conv == "sage":  # усреднение — разреженным умножением (в ~7 раз быстрее на CPU)
            m = self.self_lin(h) + self.up(h, pack["adj_in"]) + self.down(h, pack["adj_out"])
        else:
            m = self.self_lin(h) + self.up(h, pack["ei"]) + self.down(h, pack["ei_rev"])
        if self.glob is not None:
            # все графы в батче — копии одной сети, поэтому среднее по графу
            # считается через reshape (заметно быстрее scatter на CPU)
            N, D = h.shape
            g = self.glob(h.view(n_graphs, N // n_graphs, D).mean(1))
            m = (m.view(n_graphs, N // n_graphs, D) + g[:, None, :]).view(N, D)
        return h + self.drop(self.act(self.norm(m)))


class TrafficGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 6, conv: str = "sage",
                 heads: int = 4, dropout: float = 0.1, out_dim: int = 3, global_ctx: bool = True,
                 dilations: list[int] | tuple[int, ...] | None = (1, 1, 2, 2, 4, 4)):
        """dilations — для каждого слоя «через сколько переходов» берутся соседи
        (см. features.dilated_graphs); None — все слои по обычным связям."""
        super().__init__()
        dilations = list(dilations) if dilations else [1] * layers
        layers = len(dilations)
        self.dilations = dilations
        self.config = dict(in_dim=in_dim, hidden=hidden, layers=layers, conv=conv, heads=heads,
                           dropout=dropout, out_dim=out_dim, global_ctx=global_ctx, dilations=dilations)
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(
            DirectionalBlock(hidden, conv, heads, dropout, global_ctx) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden + in_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, out_dim))
        # Нулевая инициализация последнего слоя: на старте Δ = 0, т.е. модель
        # совпадает с бейзлайном «без изменений».
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, graphs: dict[int, dict], n_graphs: int = 1) -> torch.Tensor:
        """x: (N, F) — признаки участков; батч — n_graphs копий одной сети по E = N / n_graphs
        участков подряд; graphs: {d: graph_pack(...)} — связи line graph «с участка a за d
        переходов можно попасть на b» для каждого расширения d (см. batched_graphs)."""
        assert x.shape[0] % n_graphs == 0
        h = self.enc(x)
        for d, blk in zip(self.dilations, self.blocks):
            h = blk(h, graphs[d], n_graphs)
        return self.head(torch.cat([h, x], dim=-1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def batched_edge_index(edge_index: torch.Tensor, n_nodes: int, n_graphs: int) -> torch.Tensor:
    """Склеивает n_graphs копий одного графа (у всех сценариев одна топология сети)."""
    offs = (torch.arange(n_graphs, device=edge_index.device) * n_nodes).repeat_interleave(
        edge_index.shape[1])
    return edge_index.repeat(1, n_graphs) + offs


def graph_pack(edge_index: torch.Tensor, n_nodes: int) -> dict:
    """Связи в двух представлениях: edge_index (для GATv2) и разреженные матрицы
    смежности CSR (для SAGEConv: агрегация = разреженное умножение матриц).
    adj_in[i, j] = 1, если j -> i (соседи «сверху»), adj_out — наоборот."""
    def csr(rows, cols):
        return torch.sparse_coo_tensor(torch.stack([rows, cols]), torch.ones(rows.numel()),
                                       (n_nodes, n_nodes)).coalesce().to_sparse_csr()

    return {"ei": edge_index, "ei_rev": edge_index.flip(0),
            "adj_in": csr(edge_index[1], edge_index[0]), "adj_out": csr(edge_index[0], edge_index[1])}


def batched_graphs(graphs: dict, n_nodes: int, n_graphs: int) -> dict[int, dict]:
    """{d: edge_index одной сети (numpy или tensor)} -> {d: graph_pack} для батча из n_graphs копий."""
    out = {}
    for d, ei in graphs.items():
        ei = torch.as_tensor(ei, dtype=torch.long)
        out[d] = graph_pack(batched_edge_index(ei, n_nodes, n_graphs), n_nodes * n_graphs)
    return out
