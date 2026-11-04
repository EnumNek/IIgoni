"""ORM-модели (сущности БД).

Обязательные сущности из технического задания:
    Пользователь          -> User              (users)
    Сценарий              -> Scenario          (scenarios)
    Изменение             -> Change            (changes)
    Участок дороги        -> RoadSegment       (road_segments)
    Узел сети             -> NetworkNode       (network_nodes)
    Измерение трафика     -> TrafficMeasurement(traffic_measurements)
    Модель нейросети      -> NNModel           (nn_models)
    Прогноз               -> Forecast          (forecasts)
    Результат прогноза    -> ForecastResult    (forecast_results)
Служебные сущности:
    NetworkVersion  — журнал загрузок графа сети (какой network.json и когда загружен);
    AgentEvaluation — кэш результатов agents/evaluate.py (оценка светофорных агентов
                      дорогая — там прогоняется SUMO, поэтому результаты сохраняются).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base
from .geo import Geometry

# В PostgreSQL используем JSONB (индексируемый двоичный JSON), в SQLite — обычный JSON
JSONType = JSON().with_variant(JSONB(), "postgresql")

CHANGE_TYPES = ("lanes", "speed", "close", "open", "traffic_light")
SIGNAL_MODES = ("fixed", "actuated", "rl")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ Пользователь
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    full_name: Mapped[str | None] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    # Аутентификация в прототипе не реализована (см. README), столбец заложен под неё
    password_hash: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="analyst")  # admin | analyst | viewer
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    scenarios: Mapped[list[Scenario]] = relationship(back_populates="user")


# ------------------------------------------------------------------ Граф сети
class NetworkVersion(Base):
    __tablename__ = "network_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hash: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(32))  # sim | stub-osm | stub-grid
    path: Mapped[str | None] = mapped_column(Text)
    nodes_count: Mapped[int] = mapped_column(Integer)
    edges_count: Mapped[int] = mapped_column(Integer)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class NetworkNode(Base):
    """Узел сети — перекрёсток, тупик или точка соединения участков."""

    __tablename__ = "network_nodes"

    # Идентификатор берётся из network.json (в SUMO — id junction), поэтому строковый
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    type: Mapped[str] = mapped_column(String(32), default="priority")  # priority | traffic_light | dead_end ...
    lon: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    geom: Mapped[str] = mapped_column(Geometry("POINT"))

    __table_args__ = (Index("ix_network_nodes_geom", "geom", postgresql_using="gist").ddl_if(dialect="postgresql"),)


class RoadSegment(Base):
    """Участок дороги — направленное ребро графа (в SUMO — edge)."""

    __tablename__ = "road_segments"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    from_node_id: Mapped[str] = mapped_column(ForeignKey("network_nodes.id", ondelete="CASCADE"), index=True)
    to_node_id: Mapped[str] = mapped_column(ForeignKey("network_nodes.id", ondelete="CASCADE"), index=True)
    length: Mapped[float] = mapped_column(Float)  # м
    lanes: Mapped[int] = mapped_column(Integer)
    speed: Mapped[float] = mapped_column(Float)  # разрешённая скорость, м/с
    priority: Mapped[int] = mapped_column(Integer, default=0)
    name: Mapped[str | None] = mapped_column(String(255))
    road_type: Mapped[str | None] = mapped_column(String(64))  # класс дороги (highway.primary и т.п.)
    geom: Mapped[str] = mapped_column(Geometry("LINESTRING"))

    __table_args__ = (
        Index("ix_road_segments_geom", "geom", postgresql_using="gist").ddl_if(dialect="postgresql"),
        CheckConstraint("lanes >= 1", name="ck_road_segments_lanes"),
        CheckConstraint("length >= 0", name="ck_road_segments_length"),
    )


class TrafficMeasurement(Base):
    """Измерение трафика на участке: из SUMO (edgeData), датчиков или синтетическое."""

    __tablename__ = "traffic_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    edge_id: Mapped[str] = mapped_column(ForeignKey("road_segments.id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String(32))  # sumo | sensor | synthetic
    run_label: Mapped[str | None] = mapped_column(String(128))  # вариант сети / прогон SUMO
    measured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    interval_begin: Mapped[float | None] = mapped_column(Float)  # с от начала моделирования
    interval_end: Mapped[float | None] = mapped_column(Float)
    occupancy: Mapped[float | None] = mapped_column(Float)  # 0..1
    mean_speed: Mapped[float | None] = mapped_column(Float)  # м/с
    travel_time: Mapped[float | None] = mapped_column(Float)  # с
    flow: Mapped[float | None] = mapped_column(Float)  # авт/ч
    density: Mapped[float | None] = mapped_column(Float)  # авт/км


# ------------------------------------------------------------------ Модель нейросети
class NNModel(Base):
    __tablename__ = "nn_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(64))
    architecture: Mapped[str | None] = mapped_column(String(128))
    weights_path: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict | None] = mapped_column(JSONType)  # MAE/RMSE на валидации и т.п.
    is_stub: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_nn_models_name_version"),)


# ------------------------------------------------------------------ Сценарии
class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    # Базовый сценарий = исходная сеть без изменений («до»); редактировать его нельзя
    is_base: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped[User | None] = relationship(back_populates="scenarios")
    changes: Mapped[list[Change]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan", order_by="Change.seq"
    )
    forecasts: Mapped[list[Forecast]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan", order_by="Forecast.id"
    )


class Change(Base):
    """Изменение сети в сценарии: {type, target_id, value}.

    target_id ссылается либо на участок (lanes/speed/close/open), либо на узел
    (traffic_light), поэтому внешнего ключа нет — проверка выполняется в сервисе.
    """

    __tablename__ = "changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)  # порядок применения
    type: Mapped[str] = mapped_column(String(16))
    target_id: Mapped[str] = mapped_column(String(128))
    value: Mapped[object | None] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    scenario: Mapped[Scenario] = relationship(back_populates="changes")

    __table_args__ = (
        CheckConstraint(
            "type IN ('lanes', 'speed', 'close', 'open', 'traffic_light')", name="ck_changes_type"
        ),
    )

    def to_contract(self) -> dict:
        return {"type": self.type, "target_id": self.target_id, "value": self.value}


# ------------------------------------------------------------------ Прогнозы
class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"), index=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("nn_models.id", ondelete="SET NULL"))
    signal_mode: Mapped[str] = mapped_column(String(16), default="fixed")
    status: Mapped[str] = mapped_column(String(16), default="done")  # done | failed
    # Снимок списка изменений на момент расчёта: сценарий могут потом отредактировать
    changes_snapshot: Mapped[list | None] = mapped_column(JSONType)
    summary: Mapped[dict | None] = mapped_column(JSONType)
    agent_metrics: Mapped[dict | None] = mapped_column(JSONType)
    network_hash: Mapped[str | None] = mapped_column(String(64))
    predictor: Mapped[str | None] = mapped_column(String(32))  # ml | stub
    duration_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    scenario: Mapped[Scenario] = relationship(back_populates="forecasts")
    model: Mapped[NNModel | None] = relationship()
    results: Mapped[list[ForecastResult]] = relationship(
        back_populates="forecast", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint("signal_mode IN ('fixed', 'actuated', 'rl')", name="ck_forecasts_signal_mode"),
    )


class ForecastResult(Base):
    """Прогноз по одному участку: {occupancy, mean_speed, travel_time}."""

    __tablename__ = "forecast_results"

    forecast_id: Mapped[int] = mapped_column(ForeignKey("forecasts.id", ondelete="CASCADE"), primary_key=True)
    edge_id: Mapped[str] = mapped_column(ForeignKey("road_segments.id", ondelete="CASCADE"), primary_key=True)
    occupancy: Mapped[float] = mapped_column(Float)  # 0..1
    mean_speed: Mapped[float] = mapped_column(Float)  # м/с
    travel_time: Mapped[float | None] = mapped_column(Float)  # с; NULL для закрытого участка
    is_closed: Mapped[bool] = mapped_column(Boolean, default=False)

    forecast: Mapped[Forecast] = relationship(back_populates="results")

    __table_args__ = (
        CheckConstraint("occupancy >= 0 AND occupancy <= 1", name="ck_forecast_results_occupancy"),
    )


class AgentEvaluation(Base):
    __tablename__ = "agent_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int | None] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"), index=True)
    # Хэш «сеть + изменения»: одинаковые конфигурации не пересчитываются
    config_hash: Mapped[str] = mapped_column(String(64), index=True)
    mode: Mapped[str] = mapped_column(String(16))
    evaluator: Mapped[str] = mapped_column(String(32))  # agents | stub
    metrics: Mapped[dict] = mapped_column(JSONType)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (CheckConstraint("mode IN ('fixed', 'actuated', 'rl')", name="ck_agent_evaluations_mode"),)
