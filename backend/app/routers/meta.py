"""Служебные эндпоинты: состояние, пользователи, модели нейросети."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import NNModel, User
from ..schemas import ModelOut, UserIn, UserOut
from ..services.integrations import integrations
from ..services.network import store

router = APIRouter(tags=["Служебное"])


@router.get("/health", summary="Состояние сервиса")
def health(session: Session = Depends(get_session)):
    return {
        "status": "ok",
        "database": session.get_bind().dialect.name,
        "network": {"source": store.source, "path": str(store.path), "hash": store.hash,
                    "edges": len(store.edges), "nodes": len(store.nodes)},
        "integrations": integrations.status(),
    }


@router.get("/models", response_model=list[ModelOut], summary="Зарегистрированные модели нейросети")
def list_models(session: Session = Depends(get_session)):
    return session.scalars(select(NNModel).order_by(NNModel.id.desc())).all()


@router.get("/users", response_model=list[UserOut], summary="Пользователи")
def list_users(session: Session = Depends(get_session)):
    return session.scalars(select(User).order_by(User.id)).all()


@router.post("/users", response_model=UserOut, status_code=201, summary="Создать пользователя")
def create_user(body: UserIn, session: Session = Depends(get_session)):
    if session.scalars(select(User).where(User.username == body.username)).first():
        raise HTTPException(409, f"Пользователь {body.username!r} уже существует")
    user = User(**body.model_dump())
    session.add(user)
    session.commit()
    return user
