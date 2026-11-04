"""Начальное наполнение БД при старте приложения."""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Forecast, Scenario, User
from .forecasting import get_or_create_model, run_forecast
from .integrations import integrations
from .network import store

log = logging.getLogger(__name__)

BASE_SCENARIO_NAME = "Базовое состояние"


def get_demo_user(session: Session) -> User:
    user = session.scalars(select(User).where(User.username == "demo")).first()
    if user is None:
        user = User(username="demo", full_name="Демонстрационный пользователь", role="analyst")
        session.add(user)
        session.commit()
    return user


def get_base_scenario(session: Session) -> Scenario:
    sc = session.scalars(select(Scenario).where(Scenario.is_base.is_(True)).order_by(Scenario.id)).first()
    if sc is None:
        sc = Scenario(name=BASE_SCENARIO_NAME, description="Исходная сеть без изменений (состояние «до»)",
                      is_base=True, user_id=get_demo_user(session).id)
        session.add(sc)
        session.commit()
    return sc


def seed(session: Session) -> None:
    store.load()
    store.sync_to_db(session)
    get_demo_user(session)
    get_or_create_model(session, use_stub=integrations.predictor.source == "stub")
    base = get_base_scenario(session)
    if settings.seed_base_forecast:
        has = session.scalars(
            select(Forecast.id).where(Forecast.scenario_id == base.id, Forecast.network_hash == store.hash,
                                      Forecast.signal_mode == "fixed", Forecast.status == "done").limit(1)
        ).first()
        if has is None:
            f = run_forecast(session, base, "fixed")
            log.info("Рассчитан прогноз базового сценария #%d за %.0f мс", f.id, f.duration_ms or 0)
