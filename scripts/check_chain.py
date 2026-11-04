"""Сквозная проверка цепочки frontend → backend → ml/predict.py и agents/evaluate.py
на реальной сети района из sim/.

Запуск из корня репозитория (нужны зависимости всех частей, см. README.md):

    python scripts/check_chain.py
    python scripts/check_chain.py --report check_chain.md   # отчёт ещё и в файл

Скрипт поднимает backend (FastAPI в этом же процессе, временная SQLite) в режиме
INTEGRATION_MODE=real — заглушки запрещены, любая ошибка ml/ или agents/ считается
провалом — и выполняет те же HTTP-запросы, что frontend (frontend/src/api.ts:
префикс /api, «#» в id участков кодируется как %23):

 1. GET  /api/network        — граф sim/data/network.json; id рёбер совпадают с сетью,
                               на которой обучена GNN (ml/weights/meta.json), id
                               светофоров — с моделями agents/models/sim/;
 2. GET  /api/health         — прогноз считает ml/, оценку светофоров — agents/;
 3. прогноз базового сценария (его считает backend при старте);
 4. POST /api/scenarios + /changes — закрыть оживлённую улицу в обе стороны,
                               добавить полосу на другой, поставить светофор на
                               нерегулируемом перекрёстке;
 5. POST /api/scenarios/{id}/forecast (fixed) и GET /api/compare — «до/после»;
 6. POST /api/scenarios/{id}/forecast (rl) — agents/evaluate.py на сети сценария
                               (sim/data/base.net.xml + патч изменений), поправка
                               задержек на подходах к светофорам, согласованная с SUMO;
 7. POST /api/agents/evaluate — сравнение трёх режимов светофоров.

Число прогонов SUMO на оценку режима — переменная AGENTS_EVAL_SEEDS (по умолчанию
один прогон, семя 101). Код возврата 0 — все проверки пройдены.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return bool(ok)

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if not ok)


def _setup_env(tmp: Path) -> None:
    os.environ.update({
        "DATABASE_URL": f"sqlite:///{tmp / 'chain.db'}",
        "INTEGRATION_MODE": "real",
        "NETWORK_PATH": str(ROOT / "sim" / "data" / "network.json"),
        "STUB_NETWORK_PATH": str(tmp / "network_stub.json"),
        "WORK_DIR": str(tmp / "work"),
        "REPO_ROOT": str(ROOT),
        "SEED_BASE_FORECAST": "1",
    })
    os.environ.setdefault("AGENTS_EVAL_SEEDS", "101")
    sys.path.insert(0, str(ROOT / "backend"))
    sys.path.insert(0, str(ROOT))


def _pick_changes(net: dict, base: dict) -> list[dict]:
    """Изменения «как у пользователя»: закрыть оживлённую двустороннюю улицу, добавить
    полосу на другой оживлённой, поставить светофор на нагруженном нерегулируемом перекрёстке."""
    edges = {e["id"]: e for e in net["edges"]}
    pair = {(e["from"], e["to"]): e["id"] for e in net["edges"]}
    occ = {eid: r["occupancy"] for eid, r in base.items() if not r["closed"]}
    ranked = sorted(occ, key=occ.get, reverse=True)
    close = next(eid for eid in ranked if (edges[eid]["to"], edges[eid]["from"]) in pair)
    reverse = pair[(edges[close]["to"], edges[close]["from"])]
    used_nodes = {edges[close]["from"], edges[close]["to"]}
    widen = next(eid for eid in ranked if eid not in (close, reverse) and edges[eid]["lanes"] < 6
                 and not {edges[eid]["from"], edges[eid]["to"]} & used_nodes)
    incoming: dict[str, list[str]] = {}
    for e in net["edges"]:
        incoming.setdefault(e["to"], []).append(e["id"])
    candidates = [n for n in net["nodes"] if n["type"] in ("priority", "right_before_left")
                  and len(incoming.get(n["id"], [])) >= 3 and n["id"] not in used_nodes]
    tl_node = max(candidates, key=lambda n: sum(occ.get(e, 0.0) for e in incoming[n["id"]]))["id"]
    return [
        {"type": "close", "target_id": close, "value": None},
        {"type": "close", "target_id": reverse, "value": None},
        {"type": "lanes", "target_id": widen, "value": edges[widen]["lanes"] + 1},
        {"type": "traffic_light", "target_id": tl_node, "value": True},
    ]


def run(report: Path | None) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="check_chain_"))
    _setup_env(tmp)
    from fastapi.testclient import TestClient

    from app.main import app

    check = Checks()
    facts: list[tuple[str, str]] = []
    t_start = time.perf_counter()
    with TestClient(app) as client:
        def call(method: str, path: str, **kw):
            t0 = time.perf_counter()
            r = client.request(method, "/api" + path, **kw)
            dt = time.perf_counter() - t0
            if r.status_code >= 400:
                raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
            return r.json(), dt

        # 1. сеть
        net, _ = call("GET", "/network")
        sim_net = json.loads((ROOT / "sim" / "data" / "network.json").read_text(encoding="utf-8"))
        tls_nodes = sorted(n["id"] for n in net["nodes"] if n["type"] == "traffic_light")
        check("GET /api/network отдаёт граф sim/data/network.json",
              net["meta"].get("source") == "sim" and len(net["edges"]) == len(sim_net["edges"])
              and len(net["nodes"]) == len(sim_net["nodes"]),
              f"{len(net['nodes'])} узлов, {len(net['edges'])} участков, {len(tls_nodes)} светофоров")
        ml_meta = json.loads((ROOT / "ml" / "weights" / "meta.json").read_text(encoding="utf-8"))
        ref = set(ml_meta["reference"]["edge_ids"])
        check("GNN обучена на этой сети (ml/weights/meta.json)", ref == {e["id"] for e in net["edges"]},
              f"источник датасета: {ml_meta['dataset']['source']}, сценариев {ml_meta['dataset']['n_scenarios']}; "
              f"совпадает {len(ref & {e['id'] for e in net['edges']})} из {len(net['edges'])} рёбер")
        agents_meta_path = ROOT / "agents" / "models" / "sim" / "meta.json"
        agents_tls = sorted(json.loads(agents_meta_path.read_text(encoding="utf-8"))["agents"]) \
            if agents_meta_path.exists() else []
        check("RL-агенты обучены на этой сети (agents/models/sim)", agents_tls == tls_nodes,
              f"моделей {len(agents_tls)}, светофоров в сети {len(tls_nodes)}")

        # 2. подключение модулей
        health, _ = call("GET", "/health")
        integ = health["integrations"]
        check("backend подключил ml/predict.py и agents/evaluate.py",
              integ["predictor"]["source"] == "ml" and integ["evaluator"]["source"] == "agents", json.dumps(integ))

        # 3. базовый прогноз
        scenarios, _ = call("GET", "/scenarios")
        base_sc = next(s for s in scenarios if s["is_base"])
        base_full, _ = call("GET", f"/scenarios/{base_sc['id']}")
        base_f, _ = call("GET", f"/forecasts/{base_full['forecasts'][0]['id']}")
        check("прогноз базового сценария посчитан GNN", base_f["predictor"] == "ml"
              and not base_f["model"]["is_stub"] and len(base_f["results"]) == len(net["edges"])
              and not base_f["summary"]["warnings"],
              f"модель {base_f['model']['name']} {base_f['model']['version']}, {base_f['duration_ms']:.0f} мс")
        baseline = json.loads((ROOT / "sim" / "data" / "baseline" / "edges.json").read_text(encoding="utf-8"))
        diff = max(abs(base_f["results"][e]["occupancy"] - baseline[e]["occupancy"]) for e in baseline)
        check("«до» = базовый прогон SUMO (sim/data/baseline)", diff < 0.02, f"макс. |Δ occupancy| = {diff:.4f}")

        # 4. сценарий
        changes = _pick_changes(net, base_f["results"])
        sc, _ = call("POST", "/scenarios", json={"name": "Проверка цепочки", "changes": changes[:1]})
        sc, _ = call("POST", f"/scenarios/{sc['id']}/changes", json=changes[1:])
        check("изменения приняты (id с «#» и без)", [c["type"] for c in sc["changes"]] == [c["type"] for c in changes],
              "; ".join(f"{c['type']} {c['target_id']}" + (f"={c['value']}" if c["value"] not in (None,) else "")
                        for c in sc["changes"]))
        edge_id = changes[0]["target_id"]
        detail, _ = call("GET", f"/edges/{quote(edge_id, safe='')}")
        check("GET /api/edges/{id} с закодированным id", detail["edge"]["id"] == edge_id, edge_id)

        # 5. прогноз fixed + сравнение
        f_fixed, dt_fixed = call("POST", f"/scenarios/{sc['id']}/forecast", json={"signal_mode": "fixed"})
        res = f_fixed["results"]
        closed_ok = all(res[c["target_id"]]["closed"] and res[c["target_id"]]["travel_time"] is None
                        for c in changes if c["type"] == "close")
        check("прогноз сценария (fixed) посчитан GNN", f_fixed["predictor"] == "ml" and closed_ok
              and not f_fixed["summary"]["warnings"], f"{dt_fixed * 1000:.0f} мс на запрос")
        cmp_, _ = call("GET", f"/compare?a={base_f['id']}&b={f_fixed['id']}")
        moved = sum(1 for v in cmp_["edges"].values() if abs(v.get("d_occupancy") or 0) > 0.005)
        check("сравнение «до/после»: эффект распространяется по сети",
              moved > 4 and cmp_["top_worsened"] and cmp_["top_improved"],
              f"участков с |Δ occupancy| > 0,5 п.п.: {moved}")
        d = cmp_["delta"]
        facts += [("Средняя скорость в сети, км/ч: до → после",
                   f"{d['mean_speed']['a'] * 3.6:.1f} → {d['mean_speed']['b'] * 3.6:.1f}"),
                  ("Среднее время в пути (пары точек района), с: до → после",
                   f"{d['mean_trip_time']['a']:.0f} → {d['mean_trip_time']['b']:.0f}"),
                  ("Доля перегруженных участков: до → после",
                   f"{d['congested_share']['a']:.1%} → {d['congested_share']['b']:.1%}")]

        # 6. прогноз с ИИ-агентами
        f_rl, dt_rl = call("POST", f"/scenarios/{sc['id']}/forecast", json={"signal_mode": "rl"})
        am = f_rl["agent_metrics"] or {}
        factors = am.get("tls_factors") or {}
        new_tl = changes[-1]["target_id"]
        check("прогноз rl: agents/evaluate.py на сети сценария",
              f_rl["predictor"] == "ml" and am.get("rl", {}).get("evaluator") == "agents"
              and am.get("fixed", {}).get("evaluator") == "agents",
              f"{dt_rl:.0f} с на запрос (fixed + rl в SUMO)")
        check("поправка задержек по светофорам из метрик agents/",
              len(factors) >= len(tls_nodes) and any(abs(k - 1) > 0.02 for k in factors.values()),
              f"светофоров с поправкой: {len(factors)}, k от {min(factors.values()):.2f} до {max(factors.values()):.2f}"
              if factors else "нет поправок")
        cal = am.get("calibration") or {}
        check("эффект ИИ-агентов на карте не больше, чем в SUMO",
              cal.get("map_ratio") is not None and cal.get("sumo_ratio") is not None
              and cal["map_ratio"] >= cal["sumo_ratio"] - 0.01,
              f"время в пути rl/fixed: карта {cal.get('map_ratio', 0):.3f}, SUMO {cal.get('sumo_ratio', 0):.3f}, "
              f"α = {cal.get('alpha')}")
        fb = am.get("rl", {}).get("rl_fallback_tls") or []
        check("новый светофор без RL-модели — фиксированный цикл, остальные — агенты",
              new_tl in fb and len(fb) <= 3, f"rl_fallback_tls = {fb} из {len(tls_nodes) + 1} светофоров")
        facts.append(("Сценарий, ожидание у светофоров (SUMO, agents/): fixed → rl, с",
                      f"{am['fixed']['mean_waiting_time']:.1f} → {am['rl']['mean_waiting_time']:.1f}"))
        facts.append(("Сценарий, время в пути rl / fixed: SUMO → карта (после согласования)",
                      f"{cal.get('sumo_ratio', 0):.3f} → {cal.get('map_ratio', 0):.3f}"))

        # 7. сравнение режимов (панель «Светофоры»)
        ev, dt_ev = call("POST", "/agents/evaluate", json={"scenario_id": None, "modes": ["fixed", "actuated", "rl"]})
        r = ev["results"]
        check("POST /api/agents/evaluate: три режима на базовой сети",
              all(r[m]["evaluator"] == "agents" and r[m].get("mean_waiting_time") is not None
                  and r[m].get("mean_speed") is not None for m in ("fixed", "actuated", "rl"))
              and not ev["warnings"], f"{dt_ev:.0f} с")
        for m, label in (("fixed", "фиксированный цикл"), ("actuated", "actuated"), ("rl", "RL-агенты")):
            facts.append((f"Базовая сеть, {label}: ожидание / время в пути / скорость",
                          f"{r[m]['mean_waiting_time']:.1f} с / {r[m]['mean_travel_time']:.1f} с / "
                          f"{r[m]['mean_speed'] * 3.6:.1f} км/ч"))

    total = time.perf_counter() - t_start
    lines = ["# Проверка цепочки frontend → backend → ml/ и agents/ на сети sim/", "",
             f"Режим INTEGRATION_MODE=real, прогонов SUMO на оценку режима: "
             f"{len(os.environ['AGENTS_EVAL_SEEDS'].replace(',', ' ').split())}, общее время {total:.0f} с.", "",
             "| Проверка | Итог | Подробности |", "|---|---|---|"]
    lines += [f"| {n} | {'OK' if ok else '**FAIL**'} | {d} |" for n, ok, d in check.rows]
    lines += ["", "| Показатель | Значение |", "|---|---|"] + [f"| {k} | {v} |" for k, v in facts]
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if report:
        report.write_text(text, encoding="utf-8")
    return 1 if check.failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=None, help="записать отчёт (Markdown) в файл")
    sys.exit(run(ap.parse_args().report))


if __name__ == "__main__":
    main()
