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
import inspect
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


# ------------------------------------------------------------------ формат ответа agents/evaluate.py
# agents/evaluate.py отдаёт метрики с префиксом avg_ (avg_waiting_time, ...), а бэкенд,
# заглушка и фронтенд используют mean_*. Ответ приводится к одному виду: недостающее
# имя mean_* заполняется из синонима, исходные ключи сохраняются.
AGENT_METRIC_ALIASES = {
    "mean_travel_time": ("avg_travel_time",),
    "mean_waiting_time": ("avg_waiting_time",),
    "mean_queue_length": ("avg_queue_length",),
    "mean_speed": ("avg_speed",),
    "mean_time_loss": ("avg_time_loss",),
    "mean_delay_rate": ("avg_delay_rate",),  # по светофору: Σ(1 − v/v_max) на подходах, авт.·с/с
}


def _with_aliases(metrics: dict) -> dict:
    out = dict(metrics)
    for canon, aliases in AGENT_METRIC_ALIASES.items():
        if out.get(canon) is None:
            value = next((out[a] for a in aliases if out.get(a) is not None), None)
            if value is not None:
                out[canon] = value
    return out


def normalize_agent_metrics(metrics: dict, network: dict | None = None) -> dict:
    """Метрики evaluate() -> единый вид бэкенда.

    * mean_waiting_time / mean_travel_time / mean_queue_length / mean_speed
      из avg_* (формат agents/evaluate.py);
    * per_tls — по id УЗЛОВ сети: agents/ отдаёт id светофоров SUMO, которые у
      объединённых перекрёстков могут отличаться от id узла (поле tl_id в
      network.json); у каждого светофора те же синонимы метрик.
    """
    out = _with_aliases(metrics)
    per_tls = out.get("per_tls")
    if isinstance(per_tls, dict):
        tl_nodes: dict[str, list[str]] = {}
        for n in (network or {}).get("nodes", []):
            if n.get("tl_id") is not None and str(n["tl_id"]) != str(n["id"]):
                tl_nodes.setdefault(str(n["tl_id"]), []).append(str(n["id"]))
        mapped: dict[str, dict] = {}
        for tls_id, m in per_tls.items():
            m = _with_aliases(m) if isinstance(m, dict) else {"value": m}
            for node_id in tl_nodes.get(str(tls_id), [str(tls_id)]):
                mapped[node_id] = m
        out["per_tls"] = mapped
    return out


def agent_warnings(metrics: dict) -> list[str]:
    """Предупреждения по ответу agents/evaluate.py (и для свежих, и для кэшированных метрик)."""
    fallback = metrics.get("rl_fallback_tls") or []
    if not fallback:
        return []
    return [f"ИИ-агенты: для {len(fallback)} светофоров нет обученной модели (новый светофор или "
            "изменились фазы) — они работают по фиксированному циклу"]


def _model_info_from_ml(raw: dict, weights_dir: Path | None) -> dict:
    """ml/predict.py: model_info() -> {model: {...}, dataset, created_utc, test_metrics}
    -> поля таблицы nn_models (name, version, architecture, metrics, weights_path)."""
    model = raw.get("model") or {}
    dataset = raw.get("dataset") or {}
    info = {
        "name": f"GNN {model.get('variant') or model.get('conv') or ''}".strip(),
        "version": str(raw.get("created_utc") or "unversioned"),
        "architecture": model.get("title") or "Graph Neural Network (PyTorch Geometric)",
        "metrics": {"test": raw.get("test_metrics"), "dataset": dataset},
    }
    if weights_dir is not None and (Path(weights_dir) / "gnn.pt").exists():
        try:
            info["weights_path"] = str((Path(weights_dir) / "gnn.pt").resolve().relative_to(REPO_ROOT))
        except ValueError:
            info["weights_path"] = str(Path(weights_dir) / "gnn.pt")
    return info


@dataclass
class Integrations:
    cfg: Settings = field(default_factory=lambda: settings)
    _predictor: Provider | None = None
    _evaluator: Provider | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # agents/ гоняет SUMO через libsumo: одна симуляция на процесс — вызовы по очереди
    _eval_lock: threading.Lock = field(default_factory=threading.Lock)

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
        getter = getattr(p.module, "get_model_info", None) or getattr(p.module, "model_info", None)
        if info is None and callable(getter):
            try:
                info = getter()
            except Exception:  # noqa: BLE001
                log.warning("model_info() модуля ml/ завершился ошибкой: %s", traceback.format_exc())
                info = None
        info = dict(info or {})
        if isinstance(info.get("model"), dict):  # формат ml/predict.model_info()
            info = _model_info_from_ml(info, getattr(p.module, "WEIGHTS_DIR", None))
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

    def _evaluate_kwargs(self, fn: Callable) -> dict:
        """Необязательные параметры evaluate(): число прогонов SUMO (AGENTS_EVAL_SEEDS)."""
        seeds = self.cfg.agents_eval_seeds
        if not seeds:
            return {}
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return {}
        return {"seeds": seeds} if "seeds" in params else {}

    def evaluate(self, network_path: Path, mode: str, network: dict | None = None) -> tuple[dict, str, list[str]]:
        """-> (метрики в едином виде, кто считал: agents|stub, предупреждения).

        network — граф, по которому per_tls переводится в id узлов (см. normalize_agent_metrics).
        """
        p = self.evaluator
        warnings: list[str] = []
        if p.load_error and self.cfg.integration_mode == "auto":
            warnings.append(f"agents/evaluate.py недоступен ({p.load_error}) — оценка выполнена заглушкой")
        try:
            if p.source == "stub":
                metrics = p.fn(str(network_path), mode)
            else:
                with self._eval_lock:
                    metrics = p.fn(str(network_path), mode, **self._evaluate_kwargs(p.fn))
            source = p.source
        except Exception as exc:  # noqa: BLE001
            if p.source == "stub" or self.cfg.integration_mode == "real":
                raise
            log.error("agents/evaluate.py упал: %s", traceback.format_exc())
            warnings.append(f"agents/evaluate.py завершился ошибкой ({type(exc).__name__}: {exc}) — использована заглушка")
            metrics, source = evaluate_stub.evaluate(network_path, mode), "stub"
        metrics = metrics if isinstance(metrics, dict) else {"value": metrics}
        warnings += agent_warnings(metrics)
        metrics = to_jsonable(normalize_agent_metrics(metrics, network))
        return metrics, source, warnings

    def status(self) -> dict:
        p, e = self.predictor, self.evaluator
        return {
            "mode": self.cfg.integration_mode,
            "predictor": {"source": p.source, "error": p.load_error},
            "evaluator": {"source": e.source, "error": e.load_error},
        }


integrations = Integrations()
