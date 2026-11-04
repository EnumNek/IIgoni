"""Общие фикстуры тестов.

По умолчанию тесты идут на временной SQLite-базе и всегда на заглушках
(INTEGRATION_MODE=stub), чтобы не зависеть от состояния ml/ и agents/.
Проверка на PostGIS:  TEST_DATABASE_URL=postgresql+psycopg://... pytest
"""
from __future__ import annotations

import os
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="backend-tests-")
os.environ["DATABASE_URL"] = os.getenv("TEST_DATABASE_URL", f"sqlite:///{_tmp}/test.db")
os.environ["INTEGRATION_MODE"] = "stub"
os.environ["NETWORK_PATH"] = f"{_tmp}/no-network.json"  # граф строится заглушкой из map.osm (или решётка)
os.environ["STUB_NETWORK_PATH"] = f"{_tmp}/network_stub.json"
os.environ["WORK_DIR"] = f"{_tmp}/work"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.db import Base, engine
    from app.main import app

    if engine.dialect.name == "postgresql":  # чистая схема для повторяемости
        from app import models  # noqa: F401
        Base.metadata.drop_all(engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def network(client):
    return client.get("/network").json()
