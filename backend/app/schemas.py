"""Pydantic-схемы запросов и ответов API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChangeType = Literal["lanes", "speed", "close", "open", "traffic_light"]
SignalMode = Literal["fixed", "actuated", "rl"]


# ------------------------------------------------------------------ сеть
class NodeOut(BaseModel):
    id: str
    lon: float
    lat: float
    type: str


class EdgeOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    from_: str = Field(alias="from")
    to: str
    length: float
    lanes: int
    speed: float
    priority: int
    geometry: list[list[float]]
    name: str | None = None
    type: str | None = None


class NetworkOut(BaseModel):
    meta: dict[str, Any]
    nodes: list[NodeOut]
    edges: list[EdgeOut]


class MeasurementOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    run_label: str | None
    interval_begin: float | None
    interval_end: float | None
    occupancy: float | None
    mean_speed: float | None
    travel_time: float | None
    flow: float | None


class EdgeDetailOut(BaseModel):
    edge: EdgeOut
    measurements: list[MeasurementOut]


# ------------------------------------------------------------------ пользователи и модели
class UserIn(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    full_name: str | None = None
    email: str | None = None
    role: Literal["admin", "analyst", "viewer"] = "analyst"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    full_name: str | None
    email: str | None
    role: str
    created_at: datetime


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    name: str
    version: str
    architecture: str | None
    weights_path: str | None
    metrics: dict | None
    is_stub: bool
    is_active: bool
    created_at: datetime


# ------------------------------------------------------------------ сценарии и изменения
class ChangeIn(BaseModel):
    type: ChangeType
    target_id: str
    value: Any = None


class ChangeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    seq: int
    type: str
    target_id: str
    value: Any
    created_at: datetime


class ForecastBrief(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    signal_mode: str
    status: str
    predictor: str | None
    created_at: datetime
    summary: dict | None


class ScenarioIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    user_id: int | None = None
    changes: list[ChangeIn] = []


class ScenarioOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    user_id: int | None
    is_base: bool
    created_at: datetime
    updated_at: datetime
    changes: list[ChangeOut]
    forecasts: list[ForecastBrief]


class ScenarioBrief(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    is_base: bool
    created_at: datetime
    changes_count: int = 0
    forecasts_count: int = 0


# ------------------------------------------------------------------ прогнозы
class ForecastIn(BaseModel):
    signal_mode: SignalMode = "fixed"


class EdgeForecast(BaseModel):
    occupancy: float
    mean_speed: float
    travel_time: float | None
    closed: bool = False


class ForecastOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    scenario_id: int
    signal_mode: str
    status: str
    predictor: str | None
    model: ModelOut | None
    changes_snapshot: list | None
    summary: dict | None
    agent_metrics: dict | None
    network_hash: str | None
    duration_ms: float | None
    created_at: datetime
    results: dict[str, EdgeForecast]


class CompareOut(BaseModel):
    a: dict
    b: dict
    delta: dict
    edges: dict[str, dict]
    top_improved: list[dict]
    top_worsened: list[dict]


# ------------------------------------------------------------------ агенты
class EvaluateIn(BaseModel):
    scenario_id: int | None = None
    mode: SignalMode | None = None
    modes: list[SignalMode] | None = None


class EvaluateOut(BaseModel):
    scenario_id: int | None
    config_hash: str
    results: dict[str, dict]
    warnings: list[str]
