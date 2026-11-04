"""Среда обучения на базе sumo-rl: наблюдение, награда, фабрика среды.

Каждый светофор — отдельный агент. Агент раз в ``delta_time`` = 5 с
выбирает, какая ЗЕЛЁНАЯ фаза будет включена следующей (действие —
номер фазы). Если фаза меняется, sumo-rl сам вставляет жёлтый сигнал
длительностью ``yellow_time`` и не даёт переключиться раньше
``min_green`` (это гарантирует безопасность: агент не может «мигать»).

Наблюдение (вектор для одного перекрёстка), см. :class:`ApproachObservation`:

    [ one-hot текущей зелёной фазы | можно_переключить |
      очередь[С,В,Ю,З] × {прямо/направо, налево} |
      ожидание[С,В,Ю,З] × {прямо/направо, налево} ]

Награда: ``-REWARD_SCALE · (число стоящих ТС на всех подходах)``, т. е.
минус суммарная очередь (масштаб 0.1 не меняет оптимальную политику,
только делает Q-значения численно удобнее для нейросети).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import config  # noqa: E402  (до sumo_rl: SUMO_HOME и libsumo)

import numpy as np  # noqa: E402
import sumolib  # noqa: E402
import traci  # noqa: E402
from gymnasium import spaces  # noqa: E402
from sumo_rl import SumoEnvironment  # noqa: E402
from sumo_rl.environment.observations import ObservationFunction  # noqa: E402

from agents.networks import Scenario  # noqa: E402

SECTORS = ("N", "E", "S", "W")
LANE_GROUPS = ("through", "left")
MODES = ("fixed", "actuated", "rl")


# ---------------------------------------------------------------------------
# Наблюдение
# ---------------------------------------------------------------------------
def _approach_sector(shape: list[tuple[float, float]]) -> int:
    """Сторона света, С КОТОРОЙ полоса подходит к перекрёстку.

    По последнему отрезку полосы (у стоп-линии) находим направление
    движения; сторона подхода противоположна ему. Например, машины,
    едущие на восток, подъезжают с запада. Индекс в порядке С, В, Ю, З.
    """
    (x1, y1), (x2, y2) = shape[-2], shape[-1]
    heading = math.degrees(math.atan2(y2 - y1, x2 - x1))   # 0° = восток, 90° = север
    from_deg = (heading + 180.0) % 360.0
    ccw = int(round(from_deg / 90.0)) % 4                   # 0=В, 1=С, 2=З, 3=Ю
    return {1: 0, 0: 1, 3: 2, 2: 3}[ccw]


def _is_left_lane(sumo, lane: str) -> bool:
    """Полоса только для левого поворота (или разворота)."""
    dirs = set()
    for link in sumo.lane.getLinks(lane):
        dirs.add(link[6])  # кортеж связи: (…, state, direction, length)
    return bool(dirs) and dirs <= {"l", "L", "t"}


class ApproachObservation(ObservationFunction):
    """Очереди и время ожидания по подходам + текущая фаза.

    Почему так, а не «по полосам», как в sumo-rl по умолчанию:

    * подходы приведены к 4 сторонам света, поэтому размер вектора не
      зависит от числа полос — модель переживает изменение числа полос
      (тип изменения ``lanes`` в веб-приложении);
    * полосы левого поворота выделены в отдельную группу, т. к. для них
      есть своя (защищённая) фаза, и агенту важно видеть эту очередь;
    * признаки нормированы в [0, 1]: очередь — доля ёмкости подхода
      (ёмкость = длина / 7.5 м), ожидание — среднее время простоя стоящих
      машин, делённое на 120 с.
    """

    def __init__(self, ts):
        super().__init__(ts)
        self._groups: list[list[str]] | None = None
        self._capacity: np.ndarray | None = None

    # sumo-rl создаёт функцию наблюдения ДО того, как у светофора появится
    # список полос, поэтому группы строятся лениво.
    def _build(self) -> None:
        sumo = self.ts.sumo
        groups: list[list[str]] = [[] for _ in range(config.N_SECTORS * len(LANE_GROUPS))]
        for lane in self.ts.lanes:
            sector = _approach_sector(sumo.lane.getShape(lane))
            g = sector * len(LANE_GROUPS) + (1 if _is_left_lane(sumo, lane) else 0)
            groups[g].append(lane)
        self._groups = groups
        self._capacity = np.array(
            [max(1.0, sum(sumo.lane.getLength(l) for l in g) / config.VEHICLE_SPACE_M) for g in groups],
            dtype=np.float32,
        )

    @property
    def size(self) -> int:
        return self.ts.num_green_phases + 1 + 2 * config.N_SECTORS * len(LANE_GROUPS)

    def layout(self) -> list[str]:
        """Имена признаков — для документации и отладки."""
        names = [f"phase_{i}" for i in range(self.ts.num_green_phases)] + ["can_switch"]
        for feat in ("queue", "wait"):
            names += [f"{feat}_{s}_{g}" for s in SECTORS for g in LANE_GROUPS]
        return names

    def __call__(self) -> np.ndarray:
        if self._groups is None:
            self._build()
        sumo = self.ts.sumo
        ts = self.ts
        phase = np.zeros(ts.num_green_phases, dtype=np.float32)
        if ts.green_phase < ts.num_green_phases:
            phase[ts.green_phase] = 1.0
        can_switch = float(ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time)

        n = len(self._groups)
        halting = np.zeros(n, dtype=np.float32)
        waiting = np.zeros(n, dtype=np.float32)
        for i, lanes in enumerate(self._groups):
            for lane in lanes:
                halting[i] += sumo.lane.getLastStepHaltingNumber(lane)
                waiting[i] += sumo.lane.getWaitingTime(lane)
        queue = np.minimum(1.0, halting / self._capacity)
        mean_wait = np.minimum(1.0, waiting / np.maximum(halting, 1.0) / config.WAIT_NORM_S)
        return np.concatenate([phase, [can_switch], queue, mean_wait]).astype(np.float32)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(self.size,), dtype=np.float32)


# ---------------------------------------------------------------------------
# Защитное правило максимального ожидания
# ---------------------------------------------------------------------------
class MaxWaitGuard:
    """Защитный слой поверх политики агента («максимальный красный»).

    Если какая-то машина на подходе стоит дольше ``max_wait`` секунд, а
    выбранная агентом фаза эту полосу не обслуживает, действие заменяется
    на фазу, дающую этой полосе зелёный (предпочтительно главный «G»,
    а не «g» с уступанием). Так делают и в реальных адаптивных системах:
    обучаемая логика работает внутри жёстких ограничений безопасности и
    справедливости.

    Зачем: при обучении на награде «минус очередь» выгода от переключения
    ради ОДНОЙ машины мала (0.1 за шаг) и тонет в ошибке аппроксимации
    Q-сети, поэтому в редких состояниях (почти пустая сеть) агент мог
    держать зелёный пустому направлению минутами. Правило срабатывает
    редко; число срабатываний выводится в метриках (``guard_rate``).
    """

    def __init__(self, env, max_wait: float | None = config.MAX_WAIT_GUARD_S):
        self.env = env
        self.max_wait = max_wait
        self.decisions = 0
        self.interventions = 0
        self._serving: dict[str, dict[str, tuple[list[int], list[int]]]] = {}

    def _serving_map(self, ts) -> dict[str, tuple[list[int], list[int]]]:
        """Полоса -> (фазы с «G», фазы с «g»), индексы зелёных фаз sumo-rl."""
        if ts.id not in self._serving:
            links = ts.sumo.trafficlight.getControlledLinks(ts.id)
            serving: dict[str, tuple[list[int], list[int]]] = {}
            for i, phase in enumerate(ts.green_phases):
                for li, conns in enumerate(links):
                    if li >= len(phase.state) or phase.state[li] not in "Gg":
                        continue
                    for conn in conns:
                        major, minor = serving.setdefault(conn[0], ([], []))
                        target = major if phase.state[li] == "G" else minor
                        if i not in target:
                            target.append(i)
            self._serving[ts.id] = serving
        return self._serving[ts.id]

    def __call__(self, actions: dict[str, int]) -> dict[str, int]:
        if not self.max_wait:
            return actions
        out = dict(actions)
        for ts_id, action in actions.items():
            ts = self.env.traffic_signals[ts_id]
            self.decisions += 1
            if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time:
                continue  # переключиться всё равно нельзя
            sumo = ts.sumo
            worst_lane, worst = None, self.max_wait
            for lane in ts.lanes:
                # сумма ожиданий на полосе >= ожидания любой машины — дешёвый фильтр
                if sumo.lane.getWaitingTime(lane) <= self.max_wait:
                    continue
                w = max((sumo.vehicle.getWaitingTime(v) for v in sumo.lane.getLastStepVehicleIDs(lane)), default=0.0)
                if w > worst:
                    worst_lane, worst = lane, w
            if worst_lane is None:
                continue
            major, minor = self._serving_map(ts).get(worst_lane, ([], []))
            good = major or minor
            if good and action not in good:
                out[ts_id] = good[0]
                self.interventions += 1
        return out

    @property
    def rate(self) -> float:
        return self.interventions / self.decisions if self.decisions else 0.0


# ---------------------------------------------------------------------------
# Награда
# ---------------------------------------------------------------------------
def queue_reward(ts) -> float:
    """Минус суммарная очередь (стоящие ТС, v < 0.1 м/с) на подходах перекрёстка."""
    return -config.REWARD_SCALE * ts.get_total_queued()


# ---------------------------------------------------------------------------
# Среда
# ---------------------------------------------------------------------------
class TrafficEnv(SumoEnvironment):
    """SumoEnvironment с двумя доработками.

    1. Дополнительные аргументы SUMO передаются списком (в sumo-rl —
       строкой, которая режется по пробелам и ломается на путях с
       пробелами, например «Мои документы»).
    2. Отключён вывод прогресса SUMO в консоль.
    """

    def __init__(self, *args, extra_args: list[str] | None = None, **kwargs):
        self._extra_args = list(extra_args or [])
        super().__init__(*args, **kwargs)

    def _start_simulation(self):
        cmd = [
            self._sumo_binary,
            "-n", self._net,
            "-r", self._route,
            "--max-depart-delay", str(self.max_depart_delay),
            "--waiting-time-memory", str(self.waiting_time_memory),
            "--time-to-teleport", str(self.time_to_teleport),
            "--no-step-log", "true",
            "--duration-log.disable", "true",
        ]
        if self.begin_time > 0:
            cmd += ["-b", str(self.begin_time)]
        cmd += ["--random"] if self.sumo_seed == "random" else ["--seed", str(self.sumo_seed)]
        if not self.sumo_warnings:
            cmd += ["--no-warnings", "true"]
        cmd += self._extra_args
        if self.use_gui:
            cmd += ["--start", "--quit-on-end"]
        if "LIBSUMO_AS_TRACI" in os.environ:
            traci.start(cmd)
            self.sumo = traci
        else:
            traci.start(cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)


def make_env(sc: Scenario, route_file: Path, mode: str = "rl", seed: int = 0,
             extra_args: list[str] | None = None, use_gui: bool = False,
             num_seconds: int | None = None) -> TrafficEnv:
    """Создаёт многоагентную среду sumo-rl для сценария.

    mode:
      * ``rl`` — фазами управляют агенты (действия из step());
      * ``fixed`` — исходная статическая программа сети, действия игнорируются;
      * ``actuated`` — то же, но с actuated-программой из tls_actuated.add.xml.
    """
    if mode not in MODES:
        raise ValueError(f"mode должен быть одним из {MODES}, получено: {mode!r}")
    args = list(extra_args or [])
    if mode == "actuated":
        args = ["--additional-files", str(sc.actuated_file)] + args
    return TrafficEnv(
        net_file=str(sc.net_file),
        route_file=str(route_file),
        use_gui=use_gui,
        num_seconds=num_seconds or config.EPISODE_SECONDS,
        delta_time=config.ENV_PARAMS["delta_time"],
        yellow_time=config.ENV_PARAMS["yellow_time"],
        min_green=config.ENV_PARAMS["min_green"],
        max_green=config.ENV_PARAMS["max_green"],
        time_to_teleport=config.ENV_PARAMS["time_to_teleport"],
        reward_fn=queue_reward,
        observation_class=ApproachObservation,
        add_system_info=False,
        add_per_agent_info=False,
        sumo_seed=seed,
        fixed_ts=(mode != "rl"),
        sumo_warnings=False,
        single_agent=False,
        extra_args=args,
    )


def tls_ids(net_file: Path) -> list[str]:
    """Светофоры сети (без запуска SUMO)."""
    net = sumolib.net.readNet(str(net_file))
    return sorted(tl.getID() for tl in net.getTrafficLights())
