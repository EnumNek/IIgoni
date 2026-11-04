"""Настройки бэкенда.

Все параметры читаются из переменных окружения, чтобы один и тот же код
работал и в docker-compose (PostgreSQL/PostGIS), и локально (SQLite).
Значения по умолчанию подобраны так, чтобы `uvicorn app.main:app` запускался
без какой-либо настройки.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
# корень репозитория (рядом лежат sim/, ml/, agents/, frontend/);
# в docker-compose репозиторий монтируется в /repo и задаётся через REPO_ROOT
REPO_ROOT = Path(os.getenv("REPO_ROOT") or BACKEND_DIR.parent).resolve()


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass
class Settings:
    # Строка подключения SQLAlchemy. В docker-compose:
    # postgresql+psycopg://transport:transport@db:5432/transport
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'data' / 'transport.db'}"
        )
    )
    # Граф сети по контракту, который формирует часть sim/
    network_path: Path = field(
        default_factory=lambda: _env_path("NETWORK_PATH", REPO_ROOT / "sim" / "data" / "network.json")
    )
    # Где искать OSM-выгрузку района для построения графа-заглушки
    osm_candidates: list[Path] = field(
        default_factory=lambda: [
            _env_path("OSM_PATH", REPO_ROOT / "sim" / "data" / "map.osm"),
            REPO_ROOT / "sim" / "data" / "map.osm",
            REPO_ROOT / "map.osm",
        ]
    )
    # Кэш графа-заглушки (строится один раз, если sim/data/network.json ещё нет)
    stub_network_path: Path = field(
        default_factory=lambda: _env_path("STUB_NETWORK_PATH", BACKEND_DIR / "data" / "network_stub.json")
    )
    # Рабочий каталог: сюда пишутся сети с применёнными изменениями для agents/evaluate.py
    work_dir: Path = field(default_factory=lambda: _env_path("WORK_DIR", BACKEND_DIR / "data" / "work"))
    ml_dir: Path = field(default_factory=lambda: _env_path("ML_DIR", REPO_ROOT / "ml"))
    agents_dir: Path = field(default_factory=lambda: _env_path("AGENTS_DIR", REPO_ROOT / "agents"))
    # auto  — использовать ml/predict.py и agents/evaluate.py, если они есть, иначе заглушки;
    # stub  — всегда заглушки (удобно для тестов и демонстрации без GPU/SUMO);
    # real  — только настоящие модули, ошибка, если их нет.
    integration_mode: str = field(default_factory=lambda: os.getenv("INTEGRATION_MODE", "auto").lower())
    cors_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
    )
    # Посчитать прогноз базового сценария при старте (чтобы фронтенду сразу было что показать)
    seed_base_forecast: bool = field(
        default_factory=lambda: os.getenv("SEED_BASE_FORECAST", "1") not in {"0", "false", "no"}
    )

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")


settings = Settings()
