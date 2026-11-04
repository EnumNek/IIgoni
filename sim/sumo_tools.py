"""Поиск установленного SUMO и запуск его утилит.

SUMO ставится через `pip install eclipse-sumo`: пакет кладёт бинарники
(sumo, netconvert, netgenerate) и инструменты (randomTrips.py) внутрь
site-packages. Переменная SUMO_HOME, если задана, имеет приоритет — так
можно использовать и системную установку.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


class SumoError(RuntimeError):
    """Ошибка запуска одной из утилит SUMO."""


@lru_cache(maxsize=None)
def sumo_home() -> Path:
    env = os.environ.get("SUMO_HOME")
    if env and Path(env).is_dir():
        return Path(env)
    try:
        import sumo  # пакет eclipse-sumo
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise SumoError(
            "SUMO не найден: установите `pip install eclipse-sumo` или задайте SUMO_HOME"
        ) from exc
    return Path(sumo.SUMO_HOME)


@lru_cache(maxsize=None)
def binary(name: str) -> str:
    """Путь к бинарнику SUMO (sumo, netconvert, netgenerate...)."""
    candidate = sumo_home() / "bin" / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise SumoError(f"Не найден исполняемый файл SUMO: {name}")


def tool(relative: str) -> Path:
    """Путь к python-инструменту из SUMO_HOME/tools (например, randomTrips.py)."""
    path = sumo_home() / "tools" / relative
    if not path.exists():
        raise SumoError(f"Не найден инструмент SUMO: {path}")
    return path


def sumo_env() -> dict[str, str]:
    env = dict(os.environ)
    env["SUMO_HOME"] = str(sumo_home())
    return env


def run(cmd: list[str], *, cwd: Path | None = None, timeout: float | None = None,
        log_file: Path | None = None) -> subprocess.CompletedProcess:
    """Запускает команду, при ошибке бросает SumoError с хвостом stderr."""
    try:
        proc = subprocess.run(
            [str(c) for c in cmd], cwd=cwd, env=sumo_env(), capture_output=True,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SumoError(f"Тайм-аут {timeout} с: {' '.join(map(str, cmd[:3]))} ...") from exc
    if log_file is not None:
        log_file.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
        raise SumoError(f"Команда завершилась с кодом {proc.returncode}: {cmd[0]}\n{tail}")
    return proc


def run_python_tool(relative: str, args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return run([sys.executable, str(tool(relative)), *args], **kwargs)


@lru_cache(maxsize=None)
def sumo_version() -> str:
    out = run([binary("sumo"), "--version"]).stdout
    return out.splitlines()[0].strip() if out else "unknown"
