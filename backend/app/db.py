"""Подключение к БД и инициализация схемы."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        db_file = url.split("///", 1)[-1]
        if db_file and db_file != ":memory:":
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        eng = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(eng, "connect")
        def _sqlite_on_connect(dbapi_conn, _record):  # pragma: no cover - инфраструктура
            # «Прозрачные» аналоги функций PostGIS: геометрия в SQLite — это просто WKT
            dbapi_conn.create_function("ST_GeomFromText", 2, lambda wkt, _srid: wkt, deterministic=True)
            dbapi_conn.create_function("ST_AsText", 1, lambda g: g, deterministic=True)
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        return eng
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


engine: Engine = _make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """Зависимость FastAPI: одна сессия на запрос."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Создаёт расширение PostGIS (если нужно) и все таблицы.

    Для прототипа используется `create_all`; для промышленной эксплуатации
    схему следует вести миграциями (Alembic) — DDL для PostGIS можно
    получить командой `python -m app.cli ddl`.
    """
    from . import models  # noqa: F401  — регистрация моделей в метаданных

    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
    log.info("Схема БД готова (%s)", engine.dialect.name)
