"""Геометрический тип столбца, совместимый с PostGIS и SQLite.

Основная СУБД проекта — PostgreSQL + PostGIS: геометрия участков дорог и
узлов хранится в столбцах `geometry(LINESTRING,4326)` / `geometry(POINT,4326)`
с GiST-индексом, что позволяет делать пространственные запросы
(ST_Intersects, ST_DWithin и т.д.).

Чтобы прототип можно было запустить и проверить без Docker, поддержан
резервный режим SQLite: там геометрия хранится как текст WKT. Приложение
всегда обменивается с БД строками WKT, а преобразование выполняет сама СУБД:
  * запись:  ST_GeomFromText(:wkt, 4326)
  * чтение:  ST_AsText(geom)
Для SQLite эти две функции регистрируются как «прозрачные» (см. db.py),
поэтому один и тот же SQL работает в обеих СУБД.

Почему не GeoAlchemy2: без расширения SpatiaLite он не создаёт таблицы в
SQLite, а нам нужен запуск «из коробки». Собственный тип занимает ~30 строк
и не мешает перейти на GeoAlchemy2 позже.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from sqlalchemy import func
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import UserDefinedType

SRID = 4326  # WGS 84 — система координат OSM и network.json (lon, lat)


class Geometry(UserDefinedType):
    """PostGIS geometry(<type>, 4326); в SQLite — TEXT с WKT."""

    cache_ok = True

    def __init__(self, geometry_type: str = "GEOMETRY", srid: int = SRID):
        self.geometry_type = geometry_type.upper()
        self.srid = srid

    def get_col_spec(self, **kw) -> str:
        return f"geometry({self.geometry_type},{self.srid})"

    def bind_expression(self, bindvalue):
        return func.ST_GeomFromText(bindvalue, self.srid, type_=self)

    def column_expression(self, col):
        return func.ST_AsText(col, type_=self)


@compiles(Geometry, "sqlite")
def _compile_geometry_sqlite(type_, compiler, **kw):  # pragma: no cover - тривиально
    return "TEXT"


# ---------------------------------------------------------------- WKT-утилиты

def _fmt(v: float) -> str:
    return f"{v:.7f}".rstrip("0").rstrip(".")


def point_wkt(lon: float, lat: float) -> str:
    return f"POINT({_fmt(lon)} {_fmt(lat)})"


def linestring_wkt(coords: Iterable[Sequence[float]]) -> str:
    pts = ", ".join(f"{_fmt(c[0])} {_fmt(c[1])}" for c in coords)
    return f"LINESTRING({pts})"


_NUM_PAIR = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_wkt_coords(wkt: str | None) -> list[list[float]]:
    """Достаёт список [lon, lat] из WKT POINT/LINESTRING (в т.ч. EWKT с SRID=...)."""
    if not wkt:
        return []
    body = wkt.split(";", 1)[-1]
    return [[float(x), float(y)] for x, y in _NUM_PAIR.findall(body)]
