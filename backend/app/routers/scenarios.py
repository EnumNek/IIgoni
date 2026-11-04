"""Сценарии, их изменения и запуск прогноза."""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Change, Forecast, Scenario, User, utcnow
from ..schemas import ChangeIn, ForecastIn, ForecastOut, ScenarioBrief, ScenarioIn, ScenarioOut
from ..services.changes import ChangeError, normalize_change
from ..services.forecasting import run_forecast
from ..services.network import store
from ..services.seed import get_demo_user
from .forecasts import forecast_out

router = APIRouter(tags=["Сценарии"])


def _get_scenario(session: Session, scenario_id: int) -> Scenario:
    sc = session.get(Scenario, scenario_id)
    if sc is None:
        raise HTTPException(404, f"Сценарий {scenario_id} не найден")
    return sc


def _add_changes(session: Session, sc: Scenario, items: list[ChangeIn]) -> None:
    if sc.is_base:
        raise HTTPException(409, "Базовый сценарий нельзя изменять — создайте новый сценарий")
    seq = (session.scalar(select(func.max(Change.seq)).where(Change.scenario_id == sc.id)) or 0)
    for item in items:
        try:
            ch = normalize_change(item.model_dump(), store.edges, store.nodes)
        except ChangeError as exc:
            raise HTTPException(422, str(exc)) from None
        seq += 1
        sc.changes.append(Change(seq=seq, **ch))
    sc.updated_at = utcnow()


@router.get("/scenarios", response_model=list[ScenarioBrief], summary="Список сценариев")
def list_scenarios(session: Session = Depends(get_session)):
    changes = dict(session.execute(select(Change.scenario_id, func.count()).group_by(Change.scenario_id)).all())
    forecasts = dict(session.execute(select(Forecast.scenario_id, func.count()).group_by(Forecast.scenario_id)).all())
    return [
        ScenarioBrief(id=s.id, name=s.name, is_base=s.is_base, created_at=s.created_at,
                      changes_count=changes.get(s.id, 0), forecasts_count=forecasts.get(s.id, 0))
        for s in session.scalars(select(Scenario).order_by(Scenario.is_base.desc(), Scenario.id))
    ]


@router.post("/scenarios", response_model=ScenarioOut, status_code=201, summary="Создать сценарий")
def create_scenario(body: ScenarioIn, session: Session = Depends(get_session)):
    if body.user_id is not None and session.get(User, body.user_id) is None:
        raise HTTPException(422, f"Пользователь {body.user_id} не найден")
    sc = Scenario(name=body.name, description=body.description, user_id=body.user_id or get_demo_user(session).id)
    session.add(sc)
    session.flush()
    if body.changes:
        _add_changes(session, sc, body.changes)
    session.commit()
    session.refresh(sc)
    return sc


@router.get("/scenarios/{scenario_id}", response_model=ScenarioOut, summary="Сценарий с изменениями и прогнозами")
def get_scenario(scenario_id: int, session: Session = Depends(get_session)):
    return _get_scenario(session, scenario_id)


@router.delete("/scenarios/{scenario_id}", status_code=204, summary="Удалить сценарий")
def delete_scenario(scenario_id: int, session: Session = Depends(get_session)):
    sc = _get_scenario(session, scenario_id)
    if sc.is_base:
        raise HTTPException(409, "Базовый сценарий удалить нельзя")
    session.delete(sc)
    session.commit()
    return Response(status_code=204)


@router.post("/scenarios/{scenario_id}/changes", response_model=ScenarioOut, status_code=201,
             summary="Добавить изменение (или список изменений) в сценарий")
def add_changes(
    scenario_id: int,
    body: ChangeIn | list[ChangeIn] = Body(..., examples=[{"type": "lanes", "target_id": "123#0", "value": 3}]),
    session: Session = Depends(get_session),
):
    sc = _get_scenario(session, scenario_id)
    _add_changes(session, sc, body if isinstance(body, list) else [body])
    session.commit()
    session.refresh(sc)
    return sc


@router.delete("/scenarios/{scenario_id}/changes/{change_id}", response_model=ScenarioOut,
               summary="Удалить изменение")
def delete_change(scenario_id: int, change_id: int, session: Session = Depends(get_session)):
    sc = _get_scenario(session, scenario_id)
    ch = session.get(Change, change_id)
    if ch is None or ch.scenario_id != sc.id:
        raise HTTPException(404, f"Изменение {change_id} не найдено в сценарии {scenario_id}")
    sc.changes.remove(ch)
    sc.updated_at = utcnow()
    session.commit()
    session.refresh(sc)
    return sc


@router.delete("/scenarios/{scenario_id}/changes", response_model=ScenarioOut, summary="Удалить все изменения")
def clear_changes(scenario_id: int, session: Session = Depends(get_session)):
    sc = _get_scenario(session, scenario_id)
    sc.changes.clear()
    sc.updated_at = utcnow()
    session.commit()
    session.refresh(sc)
    return sc


@router.post("/scenarios/{scenario_id}/forecast", response_model=ForecastOut, status_code=201,
             summary="Рассчитать прогноз загруженности для сценария")
def create_forecast(scenario_id: int, body: ForecastIn | None = None, session: Session = Depends(get_session)):
    sc = _get_scenario(session, scenario_id)
    forecast = run_forecast(session, sc, (body or ForecastIn()).signal_mode)
    return forecast_out(session, forecast)
