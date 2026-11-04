# backend/ — серверная часть (FastAPI + PostgreSQL/PostGIS)

Серверная часть прототипа «Интеллектуальная система моделирования и оптимизации
городской транспортной сети». Она связывает все остальные части системы:

```
            ┌───────────── офлайн (обучение) ─────────────┐
 map.osm ─▶ │ sim/ (SUMO) ─▶ датасет ─▶ ml/ (GNN, обучение)│
            │                └──────▶ agents/ (DQN-агенты) │
            └──────────────────────────────────────────────┘
                     │ network.json      │ predict()      │ evaluate()
                     ▼                   ▼                ▼
 frontend/ (React) ◀──REST──▶  backend/ (FastAPI) ◀──▶ PostgreSQL + PostGIS
```

В рабочем режиме SUMO не запускается: прогноз за доли секунды даёт графовая
нейросеть (`ml/predict.py`), а бэкенд хранит граф, сценарии, изменения и
результаты, считает сводные метрики и сравнение «до/после».

## Что сделано

* **REST API** (FastAPI, автодокументация Swagger — `/docs`): все эндпоинты из
  задания + вспомогательные (список сценариев, удаление изменений, участок с
  измерениями, GeoJSON-выгрузка графа, пользователи, модели).
* **Схема БД под PostGIS**: 9 сущностей из задания + 2 служебные; геометрия
  узлов и участков в `geometry(POINT/LINESTRING, 4326)` с GiST-индексами,
  пространственный фильтр `GET /network?bbox=` через `ST_Intersects`.
  Готовый DDL — [`db/schema_postgis.sql`](db/schema_postgis.sql).
* **Резервный режим SQLite** — для запуска без Docker (геометрия хранится как
  WKT-текст, тот же код).
* **Интеграция с соседними частями по контракту** с автоматическим
  переключением на заглушки, если части ещё нет (`ml/predict.py`,
  `agents/evaluate.py`, `sim/data/network.json`).
* **Заглушки с тем же интерфейсом**: граф района из `map.osm` (чистый Python,
  без SUMO), аналитическая модель потоков вместо GNN, аналитическая оценка
  светофорного регулирования вместо RL-агентов.
* **Сводные метрики и сравнение**: средняя скорость, среднее время в пути
  (по кратчайшим путям между фиксированными парами точек), средняя
  загруженность, доля перегруженных участков, топ ухудшений/улучшений.
* **Режим «ИИ-агенты»**: прогноз GNN корректируется по результатам оценки
  агентов (см. «Принятые решения»).
* **Импорт измерений трафика** из SUMO edgeData (XML) и CSV.
* **Тесты** (pytest, 16 шт.): сквозной сценарий API, валидация изменений,
  физический смысл прогноза (закрытие/полоса/светофор), подключение внешних
  модулей и откат на заглушки. Проходят и на SQLite, и на PostGIS.
* **docker-compose**: PostGIS + бэкенд + фронтенд (nginx).

## Структура

```
backend/
├── app/
│   ├── main.py            # приложение FastAPI, CORS, префикс /api
│   ├── config.py          # настройки из переменных окружения
│   ├── db.py              # подключение к БД, создание схемы
│   ├── geo.py             # тип столбца Geometry (PostGIS / SQLite) и WKT
│   ├── models.py          # ORM-сущности
│   ├── schemas.py         # Pydantic-схемы API
│   ├── cli.py             # командная строка: init, ddl, импорт измерений
│   ├── routers/           # эндпоинты: network, scenarios, forecasts, agents, meta
│   ├── services/
│   │   ├── network.py     # загрузка графа, синхронизация с БД, выборки
│   │   ├── changes.py     # проверка и применение изменений
│   │   ├── integrations.py# подключение ml/predict.py и agents/evaluate.py
│   │   ├── forecasting.py # расчёт прогноза, поправка на режим светофоров, метрики, сравнение
│   │   ├── agents.py      # оценка режимов светофоров с кэшированием
│   │   ├── measurements.py# импорт измерений (SUMO edgeData, CSV)
│   │   └── seed.py        # начальные данные
│   └── stubs/             # заглушки соседних частей (тот же интерфейс)
│       ├── network_builder.py  # граф из OSM / решётка 4×4
│       ├── traffic_model.py    # MSA + BPR + HCM
│       ├── predict_stub.py     # = ml/predict.py
│       └── evaluate_stub.py    # = agents/evaluate.py
├── db/schema_postgis.sql  # DDL схемы (генерируется: python -m app.cli ddl)
├── tests/                 # pytest
├── Dockerfile, Dockerfile.dockerignore, docker-compose.yml
├── requirements.txt
└── .env.example
```

## Как запустить

### Вариант 1. Docker Compose (PostGIS + бэкенд + фронтенд)

```bash
cd backend
docker compose up --build
```

* веб-приложение — <http://localhost:8080>
* API и Swagger — <http://localhost:8000/docs>
* PostgreSQL/PostGIS — `localhost:5432`, БД/пользователь/пароль `transport`

Репозиторий монтируется в контейнер только для чтения (`/repo`), поэтому
появившиеся `sim/data/network.json`, `ml/predict.py`, `agents/evaluate.py`
подхватываются после перезапуска без пересборки. Чтобы в образ бэкенда
установились зависимости нейросети и агентов (PyTorch Geometric, SUMO):

```bash
WITH_ML=1 docker compose up --build
```

### Вариант 2. Локально с PostgreSQL/PostGIS

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d db          # или свой PostgreSQL с расширением postgis
export DATABASE_URL=postgresql+psycopg://transport:transport@localhost:5432/transport
uvicorn app.main:app --reload    # http://localhost:8000/docs
```

### Вариант 3. Без Docker и PostgreSQL (SQLite)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload    # БД создаётся в backend/data/transport.db
```

При первом старте бэкенд: создаёт схему, загружает граф (или строит
заглушку из `map.osm`), записывает его в БД, регистрирует модель
нейросети, создаёт пользователя `demo` и сценарий «Базовое состояние» и
считает для него прогноз.

### Тесты

```bash
cd backend
pytest                                         # SQLite, заглушки
TEST_DATABASE_URL=postgresql+psycopg://transport:transport@localhost:5432/transport_test pytest   # PostGIS
```

### Командная строка

```bash
python -m app.cli init                         # схема + начальные данные
python -m app.cli ddl -o db/schema_postgis.sql # DDL для PostGIS
python -m app.cli import-edgedata edgedata.xml --label base  # измерения из SUMO
python -m app.cli import-csv sensors.csv       # измерения из CSV
python -m app.cli seed-measurements            # синтетические измерения (только для демонстрации)
python -m app.cli rebuild-stub-network         # пересобрать граф-заглушку
```

### Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `DATABASE_URL` | `sqlite:///backend/data/transport.db` | строка подключения SQLAlchemy |
| `INTEGRATION_MODE` | `auto` | `auto` — настоящие модули при наличии, иначе заглушки; `stub`; `real` |
| `REPO_ROOT` | папка над `backend/` | корень репозитория (в Docker — `/repo`) |
| `NETWORK_PATH` | `$REPO_ROOT/sim/data/network.json` | граф по контракту |
| `OSM_PATH` | `$REPO_ROOT/sim/data/map.osm` | OSM для графа-заглушки (также ищется `$REPO_ROOT/map.osm`) |
| `STUB_NETWORK_PATH` | `backend/data/network_stub.json` | кэш графа-заглушки |
| `WORK_DIR` | `backend/data/work` | изменённые сети для `agents/evaluate.py` |
| `CORS_ORIGINS` | `*` | разрешённые источники |
| `SEED_BASE_FORECAST` | `1` | считать прогноз базового сценария при старте |

## API

Все пути доступны и с префиксом `/api` (так к ним обращается фронтенд).

| Метод и путь | Назначение |
|---|---|
| `GET /network[?bbox=minlon,minlat,maxlon,maxlat]` | граф по контракту `{meta, nodes, edges}`; bbox — `ST_Intersects` в PostGIS |
| `GET /network.geojson` | граф в GeoJSON (для QGIS) |
| `POST /network/reload` | перечитать `network.json` с диска |
| `GET /edges/{id}` | участок дороги и последние измерения трафика |
| `GET /scenarios` | список сценариев |
| `POST /scenarios` | создать сценарий `{name, description?, user_id?, changes?[]}` |
| `GET /scenarios/{id}` | сценарий с изменениями и прогнозами |
| `DELETE /scenarios/{id}` | удалить сценарий (кроме базового) |
| `POST /scenarios/{id}/changes` | добавить изменение `{type, target_id, value}` или список |
| `DELETE /scenarios/{id}/changes[/{change_id}]` | удалить одно / все изменения |
| `POST /scenarios/{id}/forecast` | рассчитать прогноз `{signal_mode: "fixed"\|"actuated"\|"rl"}` |
| `GET /forecasts/{id}` | прогноз: сводка + `results: {edge_id: {occupancy, mean_speed, travel_time, closed}}` |
| `GET /compare?a=&b=` | сравнение прогнозов a («до») и b («после»): дельты по сети и по участкам |
| `POST /agents/evaluate` | оценка режимов светофоров `{scenario_id?, mode? \| modes?[]}` |
| `GET /health`, `GET /models`, `GET/POST /users` | служебные |

Пример:

```bash
curl -X POST localhost:8000/scenarios -H 'Content-Type: application/json' \
     -d '{"name": "Закрыть участок", "changes": [{"type": "close", "target_id": "165240480#0"}]}'
curl -X POST localhost:8000/scenarios/2/forecast -H 'Content-Type: application/json' -d '{"signal_mode": "rl"}'
curl 'localhost:8000/compare?a=1&b=2'
```

> id участков в стиле SUMO содержат `#` — в URL его нужно кодировать (`%23`).

### Изменения сети (контракт)

| `type` | `target_id` | `value` |
|---|---|---|
| `lanes` | участок | число полос 1…8 |
| `speed` | участок | разрешённая скорость, м/с (1…40) |
| `close` | участок | — |
| `open` | участок | — (отменяет `close`) |
| `traffic_light` | узел | `true` — поставить светофор, `false` — убрать |

Изменения применяются строго в порядке добавления (поле `seq`).

## Модель данных

```mermaid
erDiagram
    users ||--o{ scenarios : "создаёт"
    scenarios ||--o{ changes : "содержит"
    scenarios ||--o{ forecasts : "имеет"
    scenarios ||--o{ agent_evaluations : "оценки агентов"
    nn_models ||--o{ forecasts : "рассчитан моделью"
    forecasts ||--o{ forecast_results : "по участкам"
    road_segments ||--o{ forecast_results : ""
    road_segments ||--o{ traffic_measurements : "измерения"
    network_nodes ||--o{ road_segments : "from / to"

    users { int id PK  string username  string role }
    network_nodes { string id PK  string type  geometry geom "POINT 4326" }
    road_segments { string id PK  string from_node_id FK  string to_node_id FK  float length  int lanes  float speed  int priority  geometry geom "LINESTRING 4326" }
    traffic_measurements { int id PK  string edge_id FK  string source  float occupancy  float mean_speed  float travel_time  float flow }
    nn_models { int id PK  string name  string version  jsonb metrics  bool is_stub  bool is_active }
    scenarios { int id PK  string name  int user_id FK  bool is_base }
    changes { int id PK  int scenario_id FK  int seq  string type  string target_id  jsonb value }
    forecasts { int id PK  int scenario_id FK  int model_id FK  string signal_mode  jsonb summary  jsonb changes_snapshot  string network_hash }
    forecast_results { int forecast_id PK  string edge_id PK  float occupancy  float mean_speed  float travel_time  bool is_closed }
    agent_evaluations { int id PK  int scenario_id FK  string config_hash  string mode  jsonb metrics }
    network_versions { int id PK  string hash  string source  int edges_count }
```

| Сущность (ТЗ) | Таблица | Комментарий |
|---|---|---|
| Пользователь | `users` | автор сценариев; аутентификация в прототипе не реализована, есть пользователь `demo` |
| Сценарий | `scenarios` | набор изменений; `is_base` — исходная сеть («до»), её менять нельзя |
| Изменение | `changes` | `{type, target_id, value}` + порядок `seq`; `target_id` — участок или узел, поэтому без внешнего ключа, проверка в сервисе |
| Участок дороги | `road_segments` | направленное ребро графа, геометрия `LINESTRING` |
| Узел сети | `network_nodes` | перекрёсток/тупик, тип (`priority`, `traffic_light`, `dead_end`…), геометрия `POINT` |
| Измерение трафика | `traffic_measurements` | замеры из SUMO edgeData, датчиков или синтетические (поле `source`) |
| Модель нейросети | `nn_models` | версия, путь к весам, метрики качества; одна активная |
| Прогноз | `forecasts` | расчёт для сценария: модель, режим светофоров, снимок изменений, сводка |
| Результат прогноза | `forecast_results` | прогноз по каждому участку; составной ключ (прогноз, участок) |
| *служебная* | `network_versions` | журнал загрузок графа (SHA-256 файла) |
| *служебная* | `agent_evaluations` | кэш результатов `agents/evaluate.py` по хэшу «сеть + изменения» |

## Интеграция с соседними частями

| Часть | Что ожидается | Если нет |
|---|---|---|
| `sim/` | `sim/data/network.json` по контракту | граф строится из `map.osm` (`sim/data/` или корень репозитория) парсером OSM на Python; если нет и его — решётка 4×4 |
| `ml/` | `ml/predict.py: predict(network, changes) -> {edge_id: {occupancy, mean_speed, travel_time}}` | `app/stubs/predict_stub.py` |
| `agents/` | `agents/evaluate.py: evaluate(network_path, mode) -> метрики` | `app/stubs/evaluate_stub.py` |

* Модули загружаются по пути файла (`importlib`), каталог модуля добавляется
  в `sys.path` — внутри `ml/` можно писать `from model import GNN`.
* Если `ml/predict.py` экспортирует `MODEL_INFO = {name, version, metrics, …}`
  (или функцию `get_model_info()`), эти сведения попадут в таблицу `nn_models`;
  иначе бэкенд найдёт последний файл весов `*.pt/*.pth/*.ckpt` в `ml/`.
* `agents/evaluate.py` получает путь к JSON-файлу сети **с уже применёнными
  изменениями** (формат `network.json`, закрытые участки удалены, в
  `meta.closed_edges` — их список, в `meta.base_network` — путь к исходной сети).
  Желательно, чтобы метрики содержали `mean_waiting_time`, `mean_travel_time`,
  `mean_speed` и, по возможности, `per_tls: {node_id: {mean_waiting_time}}` —
  тогда поправка прогноза делается для каждого перекрёстка отдельно.
* Ответ модели нормализуется: значения обрезаются до допустимых диапазонов,
  `occupancy` в процентах переводится в доли (с предупреждением), для
  отсутствующих участков выдаётся предупреждение.
* В режиме `auto` ошибка внешнего модуля не роняет API: расчёт выполняется
  заглушкой, а в `summary.warnings` прогноза пишется причина. Какой модуль
  реально считал, видно в полях `predictor` / `evaluator` и в `GET /health`.

## Как устроены заглушки

Заглушки нужны, чтобы веб-приложение показывало правдоподобную реакцию сети
на изменения до готовности GNN и RL-агентов. Это классическая
макроскопическая модель транспортного планирования:

1. **Граф** (`network_builder.py`): из OSM берутся проезжие дороги
   (`primary…residential`, без `service`), обрезаются по границе выгрузки,
   режутся на участки по перекрёсткам. Учитываются `oneway`, `lanes`,
   `lanes:forward/backward`, `maxspeed` (по умолчанию 60 км/ч, во дворах
   40 км/ч). Светофоры (`highway=traffic_signals`) привязываются к ближайшему
   перекрёстку в радиусе 40 м. Для `map.osm` (Петропавловск) получается
   378 узлов, 927 участков (~101 км), 29 регулируемых перекрёстков.
2. **Спрос**: гравитационная матрица корреспонденций между ~70 «зонами»
   (въезды на границе района и внутренние узлы):
   q<sub>ij</sub> ∝ w<sub>i</sub>·w<sub>j</sub>·exp(−d<sub>ij</sub>/1,5 км).
   Масштаб калибруется на базовой сети так, чтобы 90-й перцентиль загрузки
   был ≈ 0,85, и остаётся постоянным для всех сценариев.
3. **Распределение потоков**: равновесие по Вардропу методом
   последовательных средних (MSA, 20 итераций), кратчайшие пути — Дейкстра
   (`scipy.sparse.csgraph`).
4. **Время проезда**: функция BPR t = t<sub>0</sub>·(1 + 0,15·(q/c)<sup>4</sup>)
   плюс задержка на светофоре по HCM 2000:
   d = 0,5·C·(1−g/C)²/(1 − min(1,x)·g/C) + 900·T·[(x−1) + √((x−1)² + 8kIx/(cT))].
5. **Загруженность** (как `occupancy` в SUMO edgeData): по формуле Литтла
   число автомобилей на полосе = поток·время проезда; occupancy = N·5 м / длина.
6. **Оценка светофоров** (`evaluate_stub.py`) при одинаковых потоках:
   `fixed` — цикл 90 с, зелёное поровну, k = 0,5; `actuated` — цикл и доли
   зелёного по Вебстеру C<sub>0</sub> = (1,5L+5)/(1−Y), k = 0,3; `rl` —
   **условная** оценка (как actuated, потерянное время 3 с/фазу, k = 0,15).
   Это не результат обучения RL, все такие метрики помечены `is_stub: true`,
   а в интерфейсе — значком «заглушка».

Заглушка прогноза считает весь район за ~0,2–0,4 с и может служить
наивным эталоном (baseline) при оценке качества GNN.

## Принятые решения и почему

* **FastAPI** — асинхронный, автоматическая OpenAPI-документация, Pydantic-валидация;
  вычислительные эндпоинты объявлены синхронными (`def`) и выполняются в пуле
  потоков, не блокируя цикл событий.
* **PostgreSQL + PostGIS** — пространственные типы и индексы (GiST) для
  геометрии улиц, выборка по области карты, совместимость с QGIS.
  **SQLite** оставлен как резервный вариант, чтобы прототип запускался без
  Docker. Вместо GeoAlchemy2 написан свой тип столбца (`geo.py`, ~30 строк):
  обмен с БД в WKT через `ST_GeomFromText`/`ST_AsText`, а в SQLite эти функции
  регистрируются как «прозрачные» — один код для обеих СУБД.
* **Строковые id узлов и участков** — берутся из `network.json` (id SUMO),
  чтобы прогнозы GNN, данные SUMO и записи БД ссылались на одни и те же объекты.
* **Граф хранится и в памяти, и в БД**: в `predict()` уходит dict по контракту
  (без лишних запросов), а БД обеспечивает API, пространственные запросы и
  ссылочную целостность результатов. Синхронизация — по SHA-256 файла.
* **Снимок изменений в прогнозе** (`changes_snapshot`): сценарий можно
  редактировать дальше, а прогноз остаётся воспроизводимым; фронтенд по нему
  определяет, что прогноз устарел.
* **«До» = базовый сценарий с обычными светофорами**, «после» = сценарий с
  выбранным режимом: так видны и эффект изменений, и эффект ИИ-агентов.
* **Поправка прогноза на режим светофоров.** GNN обучается на данных SUMO с
  обычным регулированием и не знает о режиме. Для режимов `rl`/`actuated`
  на участках перед регулируемыми перекрёстками «лишнее» время (сверх
  свободного проезда) умножается на k = W<sub>режим</sub>/W<sub>обычный</sub> —
  отношение средних задержек на этом перекрёстке по оценке `agents/evaluate.py`;
  скорость и загруженность пересчитываются согласованно. Это не требует
  обучать отдельную GNN на каждый режим; в перспективе режим управления
  можно подать в GNN как признак узла.
* **Кэширование оценки агентов** по хэшу «сеть + изменения» — настоящая
  оценка гоняет SUMO и может занимать минуты.
* **Среднее время в пути** считается по кратчайшим (по прогнозному времени)
  путям между фиксированным набором пар точек района на расстоянии ≥ 500 м —
  метрика чувствительна к объездам при закрытии улиц, чего не даёт простое
  среднее по участкам.
* **Префикс `/api`** снимается в самом бэкенде, а прокси (Vite, nginx)
  передают путь без переписывания: иначе прокси декодирует `%23` в id SUMO.
* **`create_all` вместо миграций** — для прототипа проще; DDL выгружается
  командой `ddl`, для эксплуатации стоит подключить Alembic.

## Ограничения и развитие

* Нет аутентификации (заложен столбец `password_hash`, роли) — следующий шаг: JWT.
* Прогноз и оценка агентов выполняются синхронно в запросе; для долгой
  оценки в SUMO стоит вынести их в очередь задач (Celery/RQ) со статусом прогноза.
* Заглушки не моделируют перестроения, блокировку перекрёстков, пешеходов,
  общественный транспорт и т.п. — их цифры нельзя выдавать за результат GNN/RL.
* При смене графа старые результаты прогнозов по удалённым участкам удаляются
  каскадно; прогнозы помечены `network_hash`, по которому видна их версия сети.

## Что проверено

* `pytest` — 16 тестов, зелёные на SQLite и на PostgreSQL 16 + PostGIS 3.4.
* Ручной прогон API на PostGIS: граф 927 участков, прогноз базового
  сценария ~0,4 с, сценарий с закрытием, режим `rl`, `/compare`, `/agents/evaluate`.
* `db/schema_postgis.sql` применяется к чистой БД PostGIS без ошибок.
* Сервис `db` из `docker-compose.yml` (образ `postgis/postgis:16-3.4`)
  поднимается, бэкенд с ним работает. Полная сборка образов `backend` и
  `frontend` в среде разработки не проверена: Docker Hub ограничивал число
  загрузок (HTTP 429), а у контейнеров сборки не было доступа в интернет.
