"""Командная строка бэкенда (запуск из папки backend/):

    python -m app.cli init                       # создать схему и заполнить начальные данные
    python -m app.cli ddl                        # вывести DDL схемы для PostgreSQL/PostGIS
    python -m app.cli import-edgedata FILE.xml   # загрузить измерения из SUMO edgeData
    python -m app.cli import-csv FILE.csv        # загрузить измерения из CSV
    python -m app.cli seed-measurements          # синтетические измерения для демонстрации
    python -m app.cli rebuild-stub-network       # пересобрать граф-заглушку из map.osm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ddl() -> str:
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex, CreateTable

    from . import models  # noqa: F401
    from .db import Base

    dialect = postgresql.dialect()
    parts = ["CREATE EXTENSION IF NOT EXISTS postgis;"]
    for table in Base.metadata.sorted_tables:
        parts.append(str(CreateTable(table).compile(dialect=dialect)).strip() + ";")
        for idx in sorted(table.indexes, key=lambda i: i.name or ""):
            parts.append(str(CreateIndex(idx).compile(dialect=dialect)).strip() + ";")
    return "\n\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p_ddl = sub.add_parser("ddl")
    p_ddl.add_argument("-o", "--output", type=Path)
    p_ed = sub.add_parser("import-edgedata")
    p_ed.add_argument("file", type=Path)
    p_ed.add_argument("--label")
    p_csv = sub.add_parser("import-csv")
    p_csv.add_argument("file", type=Path)
    sub.add_parser("seed-measurements")
    sub.add_parser("rebuild-stub-network")
    args = parser.parse_args(argv)

    if args.cmd == "ddl":
        text = _ddl()
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0

    if args.cmd == "rebuild-stub-network":
        from .config import settings
        settings.stub_network_path.unlink(missing_ok=True)
        from .services.network import store
        net = store.load(force=True)
        print(f"Граф-заглушка: {store.path} ({store.source}), узлов {len(net['nodes'])}, участков {len(net['edges'])}")
        return 0

    from sqlalchemy import select

    from .db import SessionLocal, init_db
    from .services import measurements
    from .services.seed import seed

    init_db()
    with SessionLocal() as session:
        seed(session)
        if args.cmd == "init":
            print("БД инициализирована")
        elif args.cmd == "import-edgedata":
            print(f"Загружено измерений: {measurements.import_sumo_edgedata(session, args.file, run_label=args.label)}")
        elif args.cmd == "import-csv":
            print(f"Загружено измерений: {measurements.import_csv(session, args.file)}")
        elif args.cmd == "seed-measurements":
            from .models import Forecast
            from .services.forecasting import results_dict
            from .services.seed import get_base_scenario

            base = get_base_scenario(session)
            fid = session.scalars(select(Forecast.id).where(Forecast.scenario_id == base.id)
                                  .order_by(Forecast.id.desc()).limit(1)).first()
            if fid is None:
                print("Нет прогноза базового сценария")
                return 1
            print(f"Создано синтетических измерений: {measurements.seed_synthetic(session, results_dict(session, fid))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
