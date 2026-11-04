"""Подключение соседних частей системы: ml/predict.py и agents/evaluate.py.

Бэкенд не импортирует их как пакеты, а загружает файл по пути
(importlib.util.spec_from_file_location): так не нужны __init__.py в чужих
папках, а каталог модуля временно добавляется в sys.path, чтобы его
собственные импорты вида `from model import GNN` работали.

Режимы (переменная INTEGRATION_MODE):
  auto — настоящий модуль, если файл есть и импортируется; при ошибке вызова —
         откат на заглушку с предупреждением (warnings в ответе API);
  stub — всегда заглушки из app/stubs (тесты, демонстрация без GPU/SUMO);
  real — только настоящие модули; ошибка, если их нет.
"""
from __future__ import annotations

import importlib.util
import logging
import math
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import REPO_ROOT, Settings, settings
from ..stubs import evaluate_stub, predict_stub

log = logging.getLogger(__name__)


@dataclass
class Provider:
    fn: Callable
    source: str  # ml | agents | stub
    module: Any = None
    load_error: str | None = None


def _load_function(path: Path, func_name: str, module_name: str) -> tuple[Callable, Any]:
    for p in (str(path.parent), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise ImportError(f"В {path} нет функции {func_name}()")
    return fn, module


def to_jsonable(obj: Any) -> Any:
    """numpy-типы, Path, NaN/inf -> JSON-совместимые значения."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


@dataclass
class Integrations:
    cfg: Settings = field(default_factory=lambda: settings)
    _predictor: Provider | None = None
    _evaluator: Provider | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -------------------------------------------------- загрузка
    def _provider(self, path: Path, func: str, module_name: str, stub_fn: Callable, real_source: str) -> Provider:
        mode = self.cfg.integration_mode
        if mode == "stub":
            return Provider(fn=stub_fn, source="stub")
        if path.exists():
            try:
                fn, module = _load_function(path, func, module_name)
                log.info("Подключён %s", path)
                return Provider(fn=fn, source=real_source, module=module)
            except Exception as exc:  # noqa: BLE001 — любая ошибка импорта чужого модуля
                if mode == "real":
                    raise
                err = f"{type(exc).__name__}: {exc}"
                log.warning("Не удалось импортировать %s (%s) — используется заглушка", path, err)
                return Provider(fn=stub_fn, source="stub", load_error=err)
        if mode == "real":
            raise FileNotFoundError(f"{path} не найден, а INTEGRATION_MODE=real")
        return Provider(fn=stub_fn, source="stub", load_error=f"{path} не найден")

    @property
    def predictor(self) -> Provider:
        with self._lock:
            if self._predictor is None:
                self._predictor = self._provider(self.cfg.ml_dir / "predict.py", "predict", "ml_predict",
                                                 predict_stub.predict, "ml")
            return self._predictor

    @property
    def evaluator(self) -> Provider:
        with self._lock:
            if self._evaluator is None:
                self._evaluator = self._provider(self.cfg.agents_dir / "evaluate.py", "evaluate",
                                                 "agents_evaluate", evaluate_stub.evaluate, "agents")
            return self._evaluator

    def reset(self) -> None:
        with self._lock:
            self._predictor = self._evaluator = None

    # -------------------------------------------------- сведения о модели нейросети
    def model_info(self) -> dict:
        p = self.predictor
        if p.source == "stub":
            return dict(predict_stub.MODEL_INFO)
        info = getattr(p.module, "MODEL_INFO", None)
        getter = getattr(p.module, "get_model_info", None)
        if info is None and callable(getter):
            try:
                info = getter()
            except Exception:  # noqa: BLE001
                info = None
        info = dict(info or {})
        if "weights_path" not in info:
            weights = sorted(
                (f for ext in ("*.pt", "*.pth", "*.ckpt") for f in self.cfg.ml_dir.rglob(ext)),
                key=lambda f: f.stat().st_mtime, reverse=True,
            )
            if weights:
                info["weights_path"] = str(weights[0].relative_to(REPO_ROOT))
                info.setdefault("version", weights[0].stem)
        info.setdefault("name", "GNN")
        info.setdefault("version", "unversioned")
        info.setdefault("architecture", "Graph Neural Network (PyTorch Geometric)")
        info["is_stub"] = False
        return info

    # -------------------------------------------------- вызовы
    def predict(self, network: dict, changes: list[dict]) -> tuple[dict, str, list[str]]:
        """-> (прогноз по рёбрам, кто считал: ml|stub, предупреждения)."""
        p = self.predictor
        warnings: list[str] = []
        if p.load_error and self.cfg.integration_mode == "auto":
            warnings.append(f"ml/predict.py недоступен ({p.load_error}) — прогноз выполнен заглушкой")
        try:
            return p.fn(network, changes), p.source, warnings
        except Exception as exc:  # noqa: BLE001
            if p.source == "stub" or self.cfg.integration_mode == "real":
                raise
            log.error("ml/predict.py упал: %s", traceback.format_exc())
            warnings.append(f"ml/predict.py завершился ошибкой ({type(exc).__name__}: {exc}) — использована заглушка")
            return predict_stub.predict(network, changes), "stub", warnings

    def evaluate(self, network_path: Path, mode: str) -> tuple[dict, str, list[str]]:
        p = self.evaluator
        warnings: list[str] = []
        if p.load_error and self.cfg.integration_mode == "auto":
            warnings.append(f"agents/evaluate.py недоступен ({p.load_error}) — оценка выполнена заглушкой")
        try:
            metrics = p.fn(str(network_path), mode)
            source = p.source
        except Exception as exc:  # noqa: BLE001
            if p.source == "stub" or self.cfg.integration_mode == "real":
                raise
            log.error("agents/evaluate.py упал: %s", traceback.format_exc())
            warnings.append(f"agents/evaluate.py завершился ошибкой ({type(exc).__name__}: {exc}) — использована заглушка")
            metrics, source = evaluate_stub.evaluate(network_path, mode), "stub"
        metrics = to_jsonable(metrics if isinstance(metrics, dict) else {"value": metrics})
        return metrics, source, warnings

    def status(self) -> dict:
        p, e = self.predictor, self.evaluator
        return {
            "mode": self.cfg.integration_mode,
            "predictor": {"source": p.source, "error": p.load_error},
            "evaluator": {"source": e.source, "error": e.load_error},
        }


integrations = Integrations()
