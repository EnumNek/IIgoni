"""Пути и общие константы модуля ml/.

Все пути вычисляются относительно корня репозитория, поэтому скрипты можно
запускать из любой директории. Каталог с данными симуляции можно
переопределить переменной окружения IIGONI_SIM_DATA (удобно для тестов
и для запуска на другом наборе данных).
"""
from __future__ import annotations

import os
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parent

# Данные, которые готовит часть sim/ (контракт: network.json + *.parquet).
SIM_DATA_DIR = Path(os.environ.get("IIGONI_SIM_DATA", REPO_ROOT / "sim" / "data"))

# Собственные (генерируемые) данные ml/: заглушка сети и синтетический датасет.
ML_DATA_DIR = ML_DIR / "data"
STUB_DIR = ML_DATA_DIR / "stub"
SYNTH_DIR = ML_DATA_DIR / "synthetic"

# Артефакты, которые коммитятся в репозиторий.
WEIGHTS_DIR = ML_DIR / "weights"
REPORTS_DIR = ML_DIR / "reports"

# Целевые метрики по каждому ребру (контракт прогноза).
TARGETS = ("occupancy", "mean_speed", "travel_time")
TARGET_UNITS = {"occupancy": "доля 0..1", "mean_speed": "м/с", "travel_time": "с"}

SEED = 42
