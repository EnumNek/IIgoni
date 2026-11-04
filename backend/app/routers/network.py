"""GET /network — граф района; GET /edges/{id} — участок с измерениями."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import RoadSegment, TrafficMeasurement
from ..schemas import EdgeDetailOut, NetworkOut
from ..services.integrations import integrations
from ..services.network import store

router = APIRouter(tags=["Сеть"])


def _parse_bbox(bbox: str | None) -> list[float] | None:
    if not bbox:
        return None
    try:
        parts = [float(x) for x in bbox.split(",")]
    except ValueError:
        raise HTTPException(422, "bbox: ожидается minlon,minlat,maxlon,maxlat") from None
    if len(parts) != 4:
        raise HTTPException(422, "bbox: ожидается 4 числа")
    return parts


@router.get("/network", response_model=NetworkOut, summary="Граф сети (узлы и участки)")
def get_network(
    bbox: str | None = Query(None, description="minlon,minlat,maxlon,maxlat — только участки в прямоугольнике"),
    session: Session = Depends(get_session),
):
    data = store.query_db(session, _parse_bbox(bbox))
    data["meta"]["integrations"] = integrations.status()
    return data


@router.get("/network.geojson", summary="Граф сети в GeoJSON (для QGIS и т.п.)")
def get_network_geojson(session: Session = Depends(get_session)):
    net = store.query_db(session)
    features = [
        {"type": "Feature", "id": e["id"], "geometry": {"type": "LineString", "coordinates": e["geometry"]},
         "properties": {k: v for k, v in e.items() if k != "geometry"} | {"kind": "edge"}}
        for e in net["edges"]
    ] + [
        {"type": "Feature", "id": n["id"], "geometry": {"type": "Point", "coordinates": [n["lon"], n["lat"]]},
         "properties": {"id": n["id"], "type": n["type"], "kind": "node"}}
        for n in net["nodes"]
    ]
    return {"type": "FeatureCollection", "features": features}


@router.post("/network/reload", response_model=dict, summary="Перечитать network.json с диска")
def reload_network(session: Session = Depends(get_session)):
    store.load(force=True)
    version = store.sync_to_db(session)
    return {"hash": version.hash, "source": version.source, "nodes": version.nodes_count, "edges": version.edges_count}


@router.get("/edges/{edge_id}", response_model=EdgeDetailOut, summary="Участок дороги и измерения трафика")
def get_edge(edge_id: str, limit: int = 20, session: Session = Depends(get_session)):
    seg = session.get(RoadSegment, edge_id)
    if seg is None:
        raise HTTPException(404, f"Участок {edge_id!r} не найден")
    e = store.edges.get(edge_id) or {}
    edge = {"id": seg.id, "from": seg.from_node_id, "to": seg.to_node_id, "length": seg.length, "lanes": seg.lanes,
            "speed": seg.speed, "priority": seg.priority, "geometry": e.get("geometry", []), "name": seg.name,
            "type": seg.road_type}
    ms = session.scalars(
        select(TrafficMeasurement).where(TrafficMeasurement.edge_id == edge_id)
        .order_by(TrafficMeasurement.id.desc()).limit(limit)
    ).all()
    return {"edge": edge, "measurements": ms}
