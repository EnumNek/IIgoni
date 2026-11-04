"""Прогнозы и сравнение «до/после»."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Forecast
from ..schemas import CompareOut, ForecastOut, ModelOut
from ..services.forecasting import compare, results_dict

router = APIRouter(tags=["Прогнозы"])


def forecast_out(session: Session, f: Forecast) -> dict:
    return {
        "id": f.id, "scenario_id": f.scenario_id, "signal_mode": f.signal_mode, "status": f.status,
        "predictor": f.predictor, "model": ModelOut.model_validate(f.model) if f.model else None,
        "changes_snapshot": f.changes_snapshot, "summary": f.summary, "agent_metrics": f.agent_metrics,
        "network_hash": f.network_hash, "duration_ms": f.duration_ms, "created_at": f.created_at,
        "results": results_dict(session, f.id),
    }


def _get_forecast(session: Session, forecast_id: int) -> Forecast:
    f = session.get(Forecast, forecast_id)
    if f is None:
        raise HTTPException(404, f"Прогноз {forecast_id} не найден")
    return f


@router.get("/forecasts/{forecast_id}", response_model=ForecastOut, summary="Прогноз по каждому участку")
def get_forecast(forecast_id: int, session: Session = Depends(get_session)):
    return forecast_out(session, _get_forecast(session, forecast_id))


@router.get("/compare", response_model=CompareOut, summary="Сравнение двух прогнозов (a — «до», b — «после»)")
def compare_forecasts(
    a: int = Query(..., description="id прогноза «до»"),
    b: int = Query(..., description="id прогноза «после»"),
    top: int = Query(10, ge=1, le=100),
    session: Session = Depends(get_session),
):
    return compare(session, _get_forecast(session, a), _get_forecast(session, b), top=top)
