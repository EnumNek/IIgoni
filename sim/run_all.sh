#!/usr/bin/env bash
# Полный конвейер части sim/: сеть → спрос → базовый прогон → датасет → отчёт.
# Запуск из корня репозитория:  bash sim/run_all.sh [число_сценариев]
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-300}"
python -m sim.build_network
python -m sim.generate_traffic
python -m sim.noise_floor --seeds 8
python -m sim.build_dataset --n-scenarios "$N"
python -m sim.report
