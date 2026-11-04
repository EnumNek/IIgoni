"""Оценка светофорного управления (обычное / адаптивное / RL-агенты).

Контракт agents/evaluate.py принимает ПУТЬ к сети, поэтому для сценария с
изменениями бэкенд записывает изменённую сеть в backend/data/work/
(закрытые участки удалены, тип узлов обновлён) и передаёт этот путь.
В meta файла — base_network (исходный network.json) и changes: по ним
agents/ собирает сеть сценария из sim/data/base.net.xml тем же патчем
netconvert, что и датасет sim/ (id и фазы светофоров сохраняются, подходят
обученные модели). Ответ приводится к единому виду (mean_*, per_tls по id
узлов) в integrations.normalize_agent_metrics.
Результаты кэшируются в таблице agent_evaluations по хэшу «сеть + изменения»:
настоящая оценка прогоняет SUMO и может занимать минуты.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import SIGNAL_MODES, AgentEvaluation, Scenario
from .changes import apply_changes, export_for_simulation
from .integrations import agent_warnings, integrations
from .network import store


def config_hash(changes: list[dict]) -> str:
    payload = json.dumps({"network": store.hash, "changes": changes}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_network_file(changes: list[dict], chash: str) -> Path:
    """Сеть с применёнными изменениями -> JSON-файл (тот же формат, что sim/data/network.json)."""
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    path = settings.work_dir / f"network_{chash[:16]}.json"
    if not path.exists():
        net = export_for_simulation(apply_changes(store.load(), changes))
        net["meta"]["base_network"] = str(store.path)
        net["meta"]["changes"] = changes
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(net, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    return path


def evaluate_modes(session: Session, scenario: Scenario | None, modes: list[str]) -> tuple[dict, list[str], str]:
    """-> ({mode: метрики}, предупреждения, config_hash)."""
    changes = [c.to_contract() for c in scenario.changes] if scenario else []
    chash = config_hash(changes)
    results: dict[str, dict] = {}
    warnings: list[str] = []
    path: Path | None = None
    for mode in modes:
        if mode not in SIGNAL_MODES:
            raise ValueError(f"Неизвестный режим {mode!r}")
        cached = session.scalars(
            select(AgentEvaluation)
            .where(AgentEvaluation.config_hash == chash, AgentEvaluation.mode == mode)
            .order_by(AgentEvaluation.id.desc())
            .limit(1)
        ).first()
        if cached is not None and cached.evaluator == integrations.evaluator.source:
            results[mode] = {**cached.metrics, "evaluator": cached.evaluator, "cached": True}
            warnings += [w for w in agent_warnings(cached.metrics) if w not in warnings]
            continue
        path = path or write_network_file(changes, chash)
        t0 = time.perf_counter()
        metrics, source, warn = integrations.evaluate(path, mode, network=store.load())
        warnings += [w for w in warn if w not in warnings]
        rec = AgentEvaluation(scenario_id=scenario.id if scenario else None, config_hash=chash, mode=mode,
                              evaluator=source, metrics=metrics, duration_ms=(time.perf_counter() - t0) * 1000)
        session.add(rec)
        session.commit()
        results[mode] = {**metrics, "evaluator": source, "cached": False}
    return results, warnings, chash
