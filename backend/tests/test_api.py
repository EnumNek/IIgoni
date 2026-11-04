"""Сквозные тесты API: сеть -> сценарий -> изменения -> прогноз -> сравнение -> агенты."""
from __future__ import annotations

from urllib.parse import quote

import pytest


def _busiest_edge(client, network):
    base = client.get("/scenarios").json()[0]
    assert base["is_base"]
    fid = client.get(f"/scenarios/{base['id']}").json()["forecasts"][0]["id"]
    res = client.get(f"/forecasts/{fid}").json()["results"]
    return max(res, key=lambda k: res[k]["occupancy"]), fid


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["integrations"]["predictor"]["source"] == "stub"
    assert body["network"]["edges"] > 0


def test_network_contract(network):
    assert {"nodes", "edges", "meta"} <= network.keys()
    node = network["nodes"][0]
    assert {"id", "lon", "lat", "type"} <= node.keys()
    edge = network["edges"][0]
    assert {"id", "from", "to", "length", "lanes", "speed", "priority", "geometry"} <= edge.keys()
    assert len(edge["geometry"]) >= 2 and len(edge["geometry"][0]) == 2
    ids = {n["id"] for n in network["nodes"]}
    assert all(e["from"] in ids and e["to"] in ids for e in network["edges"])
    assert any(n["type"] == "traffic_light" for n in network["nodes"])


def test_network_bbox_and_geojson(client, network):
    minlon, minlat, maxlon, maxlat = network["meta"]["bounds"]
    midlon = (minlon + maxlon) / 2
    part = client.get("/network", params={"bbox": f"{minlon},{minlat},{midlon},{maxlat}"}).json()
    assert 0 < len(part["edges"]) < len(network["edges"])
    assert client.get("/network", params={"bbox": "1,2,3"}).status_code == 422
    gj = client.get("/network.geojson").json()
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == len(network["edges"]) + len(network["nodes"])


def test_base_scenario_has_forecast_and_is_readonly(client, network):
    base = client.get("/scenarios").json()[0]
    sc = client.get(f"/scenarios/{base['id']}").json()
    assert sc["is_base"] and sc["forecasts"]
    edge = network["edges"][0]["id"]
    r = client.post(f"/scenarios/{base['id']}/changes", json={"type": "close", "target_id": edge})
    assert r.status_code == 409


def test_change_validation(client, network):
    sc = client.post("/scenarios", json={"name": "валидация"}).json()
    edge = network["edges"][0]["id"]
    node = next(n["id"] for n in network["nodes"] if n["type"] != "dead_end")
    bad = [
        {"type": "lanes", "target_id": "нет-такого", "value": 2},
        {"type": "lanes", "target_id": edge, "value": 0},
        {"type": "lanes", "target_id": edge, "value": 7},  # диапазон контракта sim/: 1..6
        {"type": "speed", "target_id": edge, "value": "быстро"},
        {"type": "traffic_light", "target_id": edge, "value": True},  # светофор ставится на узел
        {"type": "traffic_light", "target_id": node, "value": "мигающий"},
        {"type": "teleport", "target_id": edge},
    ]
    dead_end = next((n["id"] for n in network["nodes"] if n["type"] == "dead_end"), None)
    if dead_end is not None:  # как в sim/changes.py: светофор в тупике невозможен
        bad.append({"type": "traffic_light", "target_id": dead_end, "value": True})
    for ch in bad:
        assert client.post(f"/scenarios/{sc['id']}/changes", json=ch).status_code == 422, ch
    ok = client.post(f"/scenarios/{sc['id']}/changes", json=[
        {"type": "speed", "target_id": edge, "value": 8.33},
        {"type": "traffic_light", "target_id": node, "value": "add"},
    ])
    assert ok.status_code == 201
    changes = ok.json()["changes"]
    assert [c["seq"] for c in changes] == [1, 2]
    assert changes[1]["value"] is True
    r = client.delete(f"/scenarios/{sc['id']}/changes/{changes[0]['id']}")
    assert [c["type"] for c in r.json()["changes"]] == ["traffic_light"]
    assert client.delete(f"/scenarios/{sc['id']}").status_code == 204
    assert client.get(f"/scenarios/{sc['id']}").status_code == 404


def test_forecast_close_edge_and_compare(client, network):
    busiest, base_fid = _busiest_edge(client, network)
    sc = client.post("/scenarios", json={
        "name": "Закрытие самого загруженного участка",
        "changes": [{"type": "close", "target_id": busiest}],
    })
    assert sc.status_code == 201, sc.text
    sid = sc.json()["id"]
    f = client.post(f"/scenarios/{sid}/forecast", json={"signal_mode": "fixed"})
    assert f.status_code == 201, f.text
    fc = f.json()
    assert fc["results"][busiest]["closed"] is True
    assert fc["results"][busiest]["travel_time"] is None
    assert len(fc["results"]) == len(network["edges"])
    for r in fc["results"].values():
        assert 0 <= r["occupancy"] <= 1 and r["mean_speed"] >= 0
    assert fc["model"]["is_stub"] is True
    assert client.get(f"/forecasts/{fc['id']}").json()["summary"] == fc["summary"]

    cmp = client.get("/compare", params={"a": base_fid, "b": fc["id"]})
    assert cmp.status_code == 200
    body = cmp.json()
    assert body["edges"][busiest]["b"]["closed"] is True
    assert {"mean_speed", "mean_trip_time", "mean_occupancy"} <= body["delta"].keys()
    # перекрытие загруженного участка увеличивает нагрузку где-то ещё
    assert body["top_worsened"]
    assert client.get("/compare", params={"a": base_fid, "b": 999999}).status_code == 404


def test_more_lanes_reduce_occupancy(client, network):
    busiest, base_fid = _busiest_edge(client, network)
    base_res = client.get(f"/forecasts/{base_fid}").json()["results"][busiest]
    lanes = next(e["lanes"] for e in network["edges"] if e["id"] == busiest)
    sc = client.post("/scenarios", json={"name": "Полоса", "changes": [
        {"type": "lanes", "target_id": busiest, "value": lanes + 1}]}).json()
    res = client.post(f"/scenarios/{sc['id']}/forecast").json()["results"][busiest]
    assert res["occupancy"] < base_res["occupancy"]


def test_rl_mode_and_agents_evaluate(client, network):
    sc = client.post("/scenarios", json={"name": "ИИ-агенты"}).json()
    fixed = client.post(f"/scenarios/{sc['id']}/forecast", json={"signal_mode": "fixed"}).json()
    rl = client.post(f"/scenarios/{sc['id']}/forecast", json={"signal_mode": "rl"}).json()
    assert rl["signal_mode"] == "rl"
    assert {"fixed", "rl"} <= rl["agent_metrics"].keys()
    assert rl["summary"]["mean_speed"] >= fixed["summary"]["mean_speed"]

    ev = client.post("/agents/evaluate", json={"scenario_id": sc["id"]})
    assert ev.status_code == 200
    res = ev.json()["results"]
    assert set(res) == {"fixed", "actuated", "rl"}
    for m in res.values():
        assert {"mean_waiting_time", "mean_travel_time", "mean_speed"} <= m.keys()
    assert res["fixed"]["cached"] is True  # уже посчитан при прогнозе в режиме rl
    assert res["rl"]["mean_waiting_time"] <= res["fixed"]["mean_waiting_time"]
    assert client.post("/agents/evaluate", json={"mode": "rl"}).json()["results"].keys() == {"rl"}
    assert client.post("/agents/evaluate", json={"mode": "magic"}).status_code == 422


def test_add_traffic_light_increases_delay(client, network):
    nodes = {n["id"]: n for n in network["nodes"]}
    incoming: dict[str, list[str]] = {}
    for e in network["edges"]:
        incoming.setdefault(e["to"], []).append(e["id"])
    _, base_fid = _busiest_edge(client, network)
    base_res = client.get(f"/forecasts/{base_fid}").json()["results"]
    node = max((nid for nid, n in nodes.items() if n["type"] == "priority" and len(incoming.get(nid, [])) >= 3),
               key=lambda nid: sum(base_res[e]["occupancy"] for e in incoming[nid]))
    sc = client.post("/scenarios", json={"name": "Светофор", "changes": [
        {"type": "traffic_light", "target_id": node, "value": True}]}).json()
    res = client.post(f"/scenarios/{sc['id']}/forecast").json()["results"]
    before = sum(base_res[e]["travel_time"] for e in incoming[node])
    after = sum(res[e]["travel_time"] for e in incoming[node])
    assert after > before


def test_users_models_edges(client, network):
    assert client.post("/users", json={"username": "ivanov", "full_name": "Иванов И.И."}).status_code == 201
    assert client.post("/users", json={"username": "ivanov"}).status_code == 409
    assert {u["username"] for u in client.get("/users").json()} >= {"demo", "ivanov"}
    models = client.get("/models").json()
    assert models and models[0]["is_stub"]
    edge = network["edges"][0]["id"]
    r = client.get(f"/edges/{quote(edge, safe='')}")  # id вида "123#0": символ # нужно кодировать
    assert r.status_code == 200 and r.json()["edge"]["id"] == edge
    assert client.get("/edges/нет").status_code == 404


@pytest.mark.parametrize("value,expected", [(True, "traffic_light"), (False, "priority")])
def test_apply_changes_order(value, expected):
    from app.services.changes import apply_changes

    net = {"nodes": [{"id": "a", "lon": 0, "lat": 0, "type": "priority"},
                     {"id": "b", "lon": 0, "lat": 0.001, "type": "traffic_light"}],
           "edges": [{"id": "e", "from": "a", "to": "b", "length": 100, "lanes": 1, "speed": 10, "priority": 1}]}
    out = apply_changes(net, [
        {"type": "close", "target_id": "e", "value": None},
        {"type": "open", "target_id": "e", "value": None},
        {"type": "lanes", "target_id": "e", "value": 2},
        {"type": "lanes", "target_id": "e", "value": 3},
        {"type": "traffic_light", "target_id": "b", "value": value},
    ])
    assert out["edges"][0]["lanes"] == 3 and out["edges"][0]["closed"] is False
    assert out["nodes"][1]["type"] == expected
    assert net["edges"][0]["lanes"] == 1  # исходная сеть не изменена


def test_api_prefix_and_encoded_ids(client, network):
    """Фронтенд ходит через /api/* без переписывания пути; id с «#» передаются как %23."""
    edge = next(e["id"] for e in network["edges"] if "#" in e["id"]) if any("#" in e["id"] for e in network["edges"]) \
        else network["edges"][0]["id"]
    r = client.get(f"/api/edges/{quote(edge, safe='')}")
    assert r.status_code == 200 and r.json()["edge"]["id"] == edge
    assert client.get("/api/health").json()["status"] == "ok"


def test_signal_correction_is_calibrated_to_sumo(client, network):
    """Поправка на режим светофоров не преувеличивает эффект: если в SUMO время в пути
    сократилось на 2 %, карта показывает те же 2 %, а не больше."""
    from app.services.forecasting import calibrate_to_sumo, compute_summary, results_dict
    from app.db import SessionLocal

    _, fid = _busiest_edge(client, network)
    with SessionLocal() as session:
        raw = results_dict(session, fid)
    tls = {n["id"] for n in network["nodes"] if n["type"] == "traffic_light"}
    factors = {n: 0.3 for n in tls}
    t0 = compute_summary(network, raw)["mean_trip_time"]
    out, k, info = calibrate_to_sumo(network, raw, factors, {"mean_travel_time": 100.0}, {"mean_travel_time": 98.0})
    assert 0 < info["alpha"] < 1 and all(0.3 < v < 1 for v in k.values())
    assert compute_summary(network, out)["mean_trip_time"] / t0 == pytest.approx(0.98, abs=2e-3)
    # SUMO даёт эффект сильнее карты — поправка не усиливается
    out, k, info = calibrate_to_sumo(network, raw, factors, {"mean_travel_time": 100.0}, {"mean_travel_time": 10.0})
    assert info["alpha"] == 1.0 and k == factors
    # по SUMO режим хуже — эффект на карте не показывается
    out, k, info = calibrate_to_sumo(network, raw, factors, {"mean_travel_time": 100.0}, {"mean_travel_time": 105.0})
    assert info["alpha"] == 0.0 and out == raw
