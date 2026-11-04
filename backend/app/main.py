"""Точка входа FastAPI: `uvicorn app.main:app --reload` (из папки backend/)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import SessionLocal, init_db
from .routers import agents, forecasts, meta, network, scenarios
from .services.seed import seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    with SessionLocal() as session:
        seed(session)
    yield


app = FastAPI(
    title="Интеллектуальная система моделирования городской транспортной сети",
    description="API прототипа: граф района, сценарии изменений, прогноз загруженности (GNN), "
                "сравнение «до/после» и оценка светофорных RL-агентов.",
    version="0.1.0",
    lifespan=lifespan,
)


class StripApiPrefix:
    """Позволяет обращаться к API и как /network, и как /api/network.

    Фронтенд (Vite dev-сервер и nginx) проксирует /api/* на бэкенд БЕЗ переписывания
    пути: при переписывании прокси декодирует URL, и id участков SUMO вида "123#0"
    (в URL — "123%230") ломаются. Префикс снимается здесь, до маршрутизации.
    """

    def __init__(self, app, prefix: str = "/api"):
        self.app = app
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            if path == self.prefix or path.startswith(self.prefix + "/"):
                scope = dict(scope)
                scope["path"] = path[len(self.prefix):] or "/"
                raw = scope.get("raw_path")
                if raw and raw.startswith(self.prefix.encode()):
                    scope["raw_path"] = raw[len(self.prefix):] or b"/"
        await self.app(scope, receive, send)


app.add_middleware(StripApiPrefix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
for r in (network.router, scenarios.router, forecasts.router, agents.router, meta.router):
    app.include_router(r)
