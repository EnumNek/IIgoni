"""POST /agents/evaluate — оценка светофорного управления (обычное / адаптивное / RL)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import SIGNAL_MODES, Scenario
from ..schemas import EvaluateIn, EvaluateOut
from ..services.agents import evaluate_modes

router = APIRouter(tags=["ИИ-агенты"])


@router.post("/agents/evaluate", response_model=EvaluateOut, summary="Оценить режимы управления светофорами")
def evaluate(body: EvaluateIn, session: Session = Depends(get_session)):
    scenario = None
    if body.scenario_id is not None:
        scenario = session.get(Scenario, body.scenario_id)
        if scenario is None:
            raise HTTPException(404, f"Сценарий {body.scenario_id} не найден")
    modes = body.modes or ([body.mode] if body.mode else list(SIGNAL_MODES))
    results, warnings, chash = evaluate_modes(session, scenario, list(dict.fromkeys(modes)))
    return {"scenario_id": body.scenario_id, "config_hash": chash, "results": results, "warnings": warnings}
