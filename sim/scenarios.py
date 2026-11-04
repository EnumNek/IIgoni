"""Шаг 3. Генератор сценариев: случайные изменения → сеть → SUMO → метрики рёбер.

    python -m sim.scenarios --seed 7                 # один случайный сценарий, печать сводки
    python -m sim.scenarios --changes '[{"type":"close","target_id":"<edge>","value":null}]'

Сценарий = список из 1–5 изменений по контракту. Конвейер одного сценария:
    1. changes.apply_changes(network.json, changes)  → итоговое состояние графа;
    2. changes.diff_states + to_netconvert_patches   → патчи *.edg.xml / *.nod.xml;
    3. netconvert -s base.net.xml -e патч -n патч     → сеть сценария;
    4. SUMO (тот же спрос trips.xml, то же зерно)      → edgeData по каждому ребру.

Как выбираются изменения (ChangeSampler):
* тип — по весам config.CHANGE_TYPE_WEIGHTS;
* ребро — с вероятностью ∝ sqrt(поток в базовом прогоне + 10): изменения на
  нагруженных улицах встречаются чаще (они интереснее для обучения), но и
  тихие улицы не исключаются;
* lanes — ±1 полоса (на однополосных +1 или +2), в пределах 1..4;
* speed — значение из городского ряда 20…80 км/ч, отличное от текущего;
* close — только если сеть остаётся сильно связной: все рёбра, которые в
  базовой сети взаимно достижимы, остаются взаимно достижимыми (проверка по
  графу «ребро → ребро» с учётом реальных разрешённых поворотов). Иначе
  часть поездок стала бы невозможной и датасет мерил бы потерю спроса,
  а не перераспределение потока;
* traffic_light — либо существующий светофор (выключить или сделать
  адаптивным), либо новый светофор на нерегулируемом перекрёстке с ≥3 въездами;
* open — снять закрытие с ребра, закрытого ранее в этом же сценарии
  (в базовой сети закрытых рёбер нет). Такие пары проверяют, что
  семантика контракта применяется по порядку.
Одно ребро/узел меняется в сценарии не более одного раза (кроме пары close→open).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from sim import config
from sim.changes import apply_changes, diff_states, to_netconvert_patches
from sim.simulate import run_sumo
from sim.sumo_tools import binary, run


@dataclass
class ScenarioSpec:
    scenario_id: int
    changes: list[dict]
    seed: int = 0
    demand_scale: float = 1.0
    meta: dict = field(default_factory=dict)


def load_network(path: Path = config.NETWORK_JSON) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_baseline_flows(path: Path = config.BASELINE_DIR / "edges.json") -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {eid: float(m.get("entered", 0)) for eid, m in data.items()}


def edge_adjacency(net_file: Path = config.BASE_NET) -> dict[str, list[str]]:
    """Граф «ребро → ребро» по разрешённым соединениям полос в сети SUMO."""
    import sumolib
    net = sumolib.net.readNet(str(net_file), withInternal=False, withPrograms=False)
    return {e.getID(): sorted({o.getID() for o in e.getOutgoing()}) for e in net.getEdges()}


class ChangeSampler:
    def __init__(self, network: dict, adjacency: dict[str, list[str]],
                 base_flows: dict[str, float] | None = None):
        self.network = network
        self.edges = {e["id"]: e for e in network["edges"]}
        self.nodes = {n["id"]: n for n in network["nodes"]}
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.edges)
        self.graph.add_edges_from((a, b) for a, outs in adjacency.items() for b in outs
                                  if a in self.edges and b in self.edges)
        self.core = max(nx.strongly_connected_components(self.graph), key=len)
        flows = base_flows or {}
        self.edge_ids = sorted(self.edges)
        self.edge_weights = [math.sqrt(flows.get(e, 0.0) + 10.0) for e in self.edge_ids]
        incoming: dict[str, int] = {}
        for e in network["edges"]:
            incoming[e["to"]] = incoming.get(e["to"], 0) + 1
        self.tl_nodes = sorted(n["id"] for n in network["nodes"] if n["type"].startswith("traffic_light"))
        self.tl_candidates = sorted(n["id"] for n in network["nodes"]
                                    if not n["type"].startswith("traffic_light")
                                    and n["type"] != "dead_end" and incoming.get(n["id"], 0) >= 3)

    # --- проверки ---------------------------------------------------------
    def keeps_connectivity(self, closed: set[str]) -> bool:
        g = self.graph.subgraph(n for n in self.graph if n not in closed)
        need = self.core - closed
        if not need:
            return False
        start = next(iter(need))
        reach_fwd = nx.descendants(g, start) | {start}
        if not need <= reach_fwd:
            return False
        reach_bwd = nx.ancestors(g, start) | {start}
        return need <= reach_bwd

    # --- генерация --------------------------------------------------------
    def _pick_edge(self, rng: random.Random, used: set[str]) -> str | None:
        for _ in range(50):
            eid = rng.choices(self.edge_ids, weights=self.edge_weights, k=1)[0]
            if eid not in used:
                return eid
        return None

    def sample(self, rng: random.Random, n_changes: int | None = None) -> list[dict]:
        n = n_changes or rng.randint(config.MIN_CHANGES, config.MAX_CHANGES)
        changes: list[dict] = []
        used_edges: set[str] = set()
        used_nodes: set[str] = set()
        closed: set[str] = set()
        reopened: set[str] = set()
        types = list(config.CHANGE_TYPE_WEIGHTS)
        attempts = 0
        while len(changes) < n and attempts < 200:
            attempts += 1
            weights = [config.CHANGE_TYPE_WEIGHTS[t] if (t != "open" or closed - reopened) else 0.0
                       for t in types]
            ctype = rng.choices(types, weights=weights, k=1)[0]
            change = getattr(self, f"_sample_{ctype}")(rng, used_edges, used_nodes, closed, reopened)
            if change is not None:
                changes.append(change)
        return changes

    def _sample_lanes(self, rng, used_edges, *_):
        eid = self._pick_edge(rng, used_edges)
        if eid is None:
            return None
        cur = self.edges[eid]["lanes"]
        if cur <= config.MIN_LANES:
            new = cur + (2 if rng.random() < 0.2 else 1)
        elif cur >= config.MAX_LANES:
            new = cur - 1
        else:
            new = cur + rng.choice((-1, 1))
        new = max(config.MIN_LANES, min(config.MAX_LANES, new))
        if new == cur:
            return None
        used_edges.add(eid)
        return {"type": "lanes", "target_id": eid, "value": new}

    def _sample_speed(self, rng, used_edges, *_):
        eid = self._pick_edge(rng, used_edges)
        if eid is None:
            return None
        cur_kmh = round(self.edges[eid]["speed"] * 3.6)
        options = [v for v in config.SPEED_CHOICES_KMH if abs(v - cur_kmh) >= 5]
        kmh = rng.choice(options)
        used_edges.add(eid)
        return {"type": "speed", "target_id": eid, "value": round(kmh / 3.6, 2)}

    def _sample_close(self, rng, used_edges, used_nodes, closed, reopened):
        for _ in range(10):
            eid = self._pick_edge(rng, used_edges)
            if eid is None:
                return None
            if self.keeps_connectivity((closed - reopened) | {eid}):
                used_edges.add(eid)
                closed.add(eid)
                return {"type": "close", "target_id": eid, "value": None}
        return None

    def _sample_open(self, rng, used_edges, used_nodes, closed, reopened):
        candidates = sorted(closed - reopened)
        if not candidates:
            return None
        eid = rng.choice(candidates)
        reopened.add(eid)
        return {"type": "open", "target_id": eid, "value": None}

    def _sample_traffic_light(self, rng, used_edges, used_nodes, *_):
        if self.tl_nodes and (rng.random() < 0.5 or not self.tl_candidates):
            pool = [n for n in self.tl_nodes if n not in used_nodes]
            if not pool:
                return None
            nid = rng.choice(pool)
            cur = self.nodes[nid].get("tl_type") or "static"
            value = "off" if rng.random() < 0.5 else ("actuated" if cur != "actuated" else "static")
        else:
            pool = [n for n in self.tl_candidates if n not in used_nodes]
            if not pool:
                return None
            nid = rng.choice(pool)
            value = "static" if rng.random() < 0.7 else "actuated"
        used_nodes.add(nid)
        return {"type": "traffic_light", "target_id": nid, "value": value}


def scenario_seed(master_seed: int, scenario_id: int) -> int:
    return (master_seed * 1_000_003 + scenario_id * 7919) % (2 ** 31 - 1)


def make_specs(sampler: ChangeSampler, n_scenarios: int, master_seed: int,
               include_baseline: bool = True) -> list[ScenarioSpec]:
    specs = [ScenarioSpec(0, [], seed=0, meta={"baseline": True})] if include_baseline else []
    for sid in range(1, n_scenarios + 1):
        seed = scenario_seed(master_seed, sid)
        specs.append(ScenarioSpec(sid, sampler.sample(random.Random(seed)), seed=seed))
    return specs


# --------------------------------------------------------------------------
# Прогон одного сценария
# --------------------------------------------------------------------------

def build_scenario_net(base_network: dict, changes: list[dict], work_dir: Path,
                       base_net: Path = config.BASE_NET) -> tuple[dict, Path, float]:
    """Применяет изменения и собирает сеть сценария. Возвращает (граф, путь к .net.xml, время netconvert)."""
    final = apply_changes(base_network, changes)
    edg, nod = to_netconvert_patches(diff_states(base_network, final))
    if edg is None and nod is None:
        return final, base_net, 0.0
    cmd = [binary("netconvert"), "-s", Path(base_net).resolve(), "-o", "scenario.net.xml", "--no-warnings"]
    if edg:
        (work_dir / "patch.edg.xml").write_text(edg, encoding="utf-8")
        cmd += ["-e", "patch.edg.xml"]
    if nod:
        (work_dir / "patch.nod.xml").write_text(nod, encoding="utf-8")
        cmd += ["-n", "patch.nod.xml"]
    t0 = time.perf_counter()
    run(cmd, cwd=work_dir)
    return final, work_dir / "scenario.net.xml", time.perf_counter() - t0


def collect_rows(spec: ScenarioSpec, final: dict, sim: dict) -> tuple[list[dict], list[dict], dict]:
    """Строки таблиц edges/nodes и сетевые агрегаты сценария."""
    nan = math.nan
    edge_rows = []
    for e in final["edges"]:
        m = sim["edges"].get(e["id"])
        closed = bool(e.get("closed"))
        row = {"scenario_id": spec.scenario_id, "edge_id": e["id"], "lanes": e["lanes"],
               "speed_limit": e["speed"], "closed": closed, "changed": bool(e.get("changed"))}
        if closed or m is None:
            # Закрытое ребро: машин нет, скорость и время проезда не определены (NaN).
            row.update(occupancy=0.0 if closed else nan, mean_speed=nan, travel_time=nan,
                       density=0.0, waiting_time=0.0, time_loss=0.0, entered=0, left=0,
                       sampled_seconds=0.0)
        else:
            row.update({k: m[k] for k in ("occupancy", "mean_speed", "travel_time", "density",
                                          "waiting_time", "time_loss", "entered", "left",
                                          "sampled_seconds")})
        edge_rows.append(row)
    node_rows = [{"scenario_id": spec.scenario_id, "node_id": n["id"], "type": n["type"],
                  "tl_type": n.get("tl_type"), "changed": bool(n.get("changed"))}
                 for n in final["nodes"]]
    lengths = {e["id"]: e["length"] for e in final["edges"]}
    open_rows = [r for r in edge_rows if not r["closed"] and not math.isnan(r["occupancy"])]
    total_len = sum(lengths[r["edge_id"]] for r in open_rows) or 1.0
    aggregates = {
        "net_occupancy_mean": sum(r["occupancy"] * lengths[r["edge_id"]] for r in open_rows) / total_len,
        "net_veh_hours": sum(r["sampled_seconds"] for r in open_rows) / 3600,
        "net_time_loss_h": sum(r["time_loss"] for r in open_rows) / 3600,
        "net_speed_mean": (sum(r["mean_speed"] * r["sampled_seconds"] for r in open_rows)
                           / max(sum(r["sampled_seconds"] for r in open_rows), 1e-9)),
    }
    return edge_rows, node_rows, aggregates


def run_scenario(spec: ScenarioSpec, base_network: dict, *, base_net: Path = config.BASE_NET,
                 trips: Path = config.TRIPS_FILE, work_root: Path | None = None,
                 keep_files: bool = False) -> tuple[list[dict], list[dict], dict]:
    """Полный прогон сценария. Возвращает (строки рёбер, строки узлов, сводка)."""
    t0 = time.perf_counter()
    work_dir = Path(tempfile.mkdtemp(prefix=f"scn{spec.scenario_id:05d}_", dir=work_root))
    try:
        final, net_file, t_netconvert = build_scenario_net(base_network, spec.changes, work_dir, base_net)
        sim = run_sumo(net_file, trips, work_dir, demand_scale=spec.demand_scale)
        edge_rows, node_rows, aggregates = collect_rows(spec, final, sim)
    finally:
        if not keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)
    counts = {f"n_{t}": sum(1 for c in spec.changes if c["type"] == t)
              for t in ("lanes", "speed", "close", "open", "traffic_light")}
    summary = {
        "scenario_id": spec.scenario_id, "seed": spec.seed, "demand_scale": spec.demand_scale,
        "n_changes": len(spec.changes), **counts,
        "changes": json.dumps(spec.changes, ensure_ascii=False),
        "n_edges_changed": sum(1 for r in edge_rows if r["changed"]),
        "n_nodes_changed": sum(1 for r in node_rows if r["changed"]),
        **sim["summary"], **aggregates,
        "netconvert_seconds": round(t_netconvert, 3),
        "total_seconds": round(time.perf_counter() - t0, 3),
    }
    return edge_rows, node_rows, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1, help="зерно случайных изменений")
    parser.add_argument("--n-changes", type=int, help="число изменений (по умолчанию 1..5)")
    parser.add_argument("--changes", type=str, help="JSON-список изменений вместо случайных")
    parser.add_argument("--keep-files", action="store_true", help="не удалять рабочую папку сценария")
    args = parser.parse_args(argv)

    network = load_network()
    if args.changes:
        changes = json.loads(args.changes)
    else:
        sampler = ChangeSampler(network, edge_adjacency(), load_baseline_flows())
        changes = sampler.sample(random.Random(args.seed), args.n_changes)
    print("[scenarios] изменения:", json.dumps(changes, ensure_ascii=False))
    work_root = config.DATA_DIR / "work" if args.keep_files else None
    if work_root:
        work_root.mkdir(parents=True, exist_ok=True)
    edge_rows, _, summary = run_scenario(ScenarioSpec(1, changes, seed=args.seed), network,
                                         work_root=work_root, keep_files=args.keep_files)
    shown = {k: v for k, v in summary.items() if k != "changes"}
    print("[scenarios] сводка:", json.dumps(shown, ensure_ascii=False, default=float, indent=1))
    changed = [r for r in edge_rows if r["changed"]]
    for r in changed:
        print(f"  {r['edge_id']}: lanes={r['lanes']} speed={r['speed_limit']} closed={r['closed']} "
              f"occ={r['occupancy']:.3f} v={r['mean_speed']:.2f} tt={r['travel_time']:.1f}")


if __name__ == "__main__":
    main()
