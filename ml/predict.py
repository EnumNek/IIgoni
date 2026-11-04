"""Прогноз загруженности сети после изменений — интерфейс ml/ для backend/.

Контракт:

    from ml.predict import predict
    forecast = predict(network, changes)
    # network — dict в формате sim/data/network.json
    # changes — [{"type": "lanes"|"speed"|"close"|"open"|"traffic_light",
    #             "target_id": "...", "value": ...}, ...]
    # forecast — {edge_id: {"occupancy": 0..1, "mean_speed": м/с,
    #                       "travel_time": с, "closed": bool}}

Для закрытого ребра возвращается occupancy = 0, mean_speed = 0,
travel_time = None (проезд невозможен) и closed = True. Поле ``closed`` —
дополнение к контракту, у открытых рёбер оно False.

Как считается прогноз:
  1. Веса и метаданные загружаются из ml/weights/ один раз (ленивый кэш).
  2. Переданная сеть сравнивается с эталонной сетью, на которой обучена
     модель (meta.json). Если backend хранит сеть уже с внесёнными правками,
     разница с эталоном будет учтена как изменения автоматически.
  3. К сети применяются changes, строятся признаки рёбер line graph,
     GNN предсказывает Δ относительно базового состояния.
  4. Если итоговое состояние совпадает с эталонным (изменений нет),
     возвращается базовое состояние без вызова модели: predict(сеть, [])
     в точности равен «до».

Неизвестные модели рёбра (другая сеть) получают базовое состояние
«свободный поток»; прогноз для них работоспособен, но для точности модель
нужно переобучить на датасете этой сети (python -m ml.train).

CLI:
    python -m ml.predict --changes '[{"type": "close", "target_id": "E1", "value": null}]'
    python -m ml.predict --network sim/data/network.json --changes-file changes.json --out f.json
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # запуск как скрипта: python ml/predict.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import logging
import threading
from collections import OrderedDict

import numpy as np
import torch

from ml.config import REPO_ROOT, WEIGHTS_DIR
from ml.core import GNNRunner, Normalizer
from ml.features import FEATURE_NAMES, FeatureBuilder, inverse_targets
from ml.model import TrafficGNN
from ml.network import State, apply_changes, parse_network

log = logging.getLogger(__name__)

_CACHE_SIZE = 8


class Predictor:
    """Загруженная модель. Потокобезопасна: инференс идёт без градиентов,
    общий кэш подготовленных сетей защищён блокировкой."""

    def __init__(self, weights_dir: str | Path = WEIGHTS_DIR):
        weights_dir = Path(weights_dir)
        meta_path, ckpt_path = weights_dir / "meta.json", weights_dir / "gnn.pt"
        if not meta_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"Нет весов модели в {weights_dir}. Обучите модель: python -m ml.train")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("feature_names") != FEATURE_NAMES:
            raise RuntimeError("Веса обучены с другим набором признаков — переобучите модель")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = TrafficGNN(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        self.runner = GNNRunner(model.eval(), Normalizer.from_dict(self.meta["normalizer"]),
                                zero_correction=bool(self.meta.get("zero_correction", True)))
        ref = self.meta["reference"]
        self.ref_edge = {eid: i for i, eid in enumerate(ref["edge_ids"])}
        self.ref_lanes = np.asarray(ref["lanes"], float)
        self.ref_speed = np.asarray(ref["speed"], float)
        self.ref_closed = np.asarray(ref["closed"], bool)
        self.ref_metrics = np.asarray(ref["base_metrics"], float)
        self.ref_node = {nid: i for i, nid in enumerate(ref["node_ids"])}
        self.ref_tl = np.asarray(ref["tl"], bool)
        self._prepared: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()
        self._warned_unknown = False

    # -- подготовка сети (кэшируется) --------------------------------------------------
    def _prepare(self, network: dict):
        topo, passed = parse_network(network)
        h = hashlib.sha1(topo.signature().encode())
        for arr in (passed.lanes, passed.speed, passed.closed, passed.tl):
            h.update(np.ascontiguousarray(arr).tobytes())
        key = h.hexdigest()
        with self._lock:
            if key in self._prepared:
                self._prepared.move_to_end(key)
                topo_c, ref_state, fb, x_base = self._prepared[key]
                return topo_c, passed, ref_state, fb, x_base

        ei = np.array([self.ref_edge.get(e, -1) for e in topo.edge_ids])
        known = ei >= 0
        ni = np.array([self.ref_node.get(n, -1) for n in topo.node_ids])
        nk = ni >= 0
        if known.mean() < 0.5 and not self._warned_unknown:
            log.warning("Сеть совпадает с эталонной моделью лишь на %.0f%% рёбер: для остальных "
                        "базовое состояние неизвестно (свободный поток). Переобучите модель на "
                        "датасете этой сети: python -m ml.train", 100 * known.mean())
            self._warned_unknown = True
        ref_state = State(
            lanes=np.where(known, self.ref_lanes[ei], passed.lanes),
            speed=np.where(known, self.ref_speed[ei], passed.speed),
            closed=np.where(known, self.ref_closed[ei], passed.closed),
            tl=np.where(nk, self.ref_tl[ni], passed.tl),
        )
        base_metrics = np.where(known[:, None], self.ref_metrics[ei], np.nan)
        fb = FeatureBuilder(topo, ref_state, base_metrics)
        x_base = fb.features(ref_state)
        with self._lock:
            self._prepared[key] = (topo, ref_state, fb, x_base)
            while len(self._prepared) > _CACHE_SIZE:
                self._prepared.popitem(last=False)
        return topo, passed, ref_state, fb, x_base

    # -- прогноз -----------------------------------------------------------------------
    def predict_arrays(self, network: dict, changes: list | None):
        """Прогноз в виде массивов: (topo, state, metrics (E, 3))."""
        topo, passed, ref, fb, x_base = self._prepare(network)
        state = apply_changes(topo, passed, changes or [])
        r = fb.residual_base(state)[None]
        if _same_state(state, ref):
            # Изменений относительно эталона нет — это и есть базовое состояние.
            # Модель не вызывается: сценариев «без изменений» нет в обучении, и
            # её выход на них — экстраполяция; так «до» совпадает с базой в точности.
            return topo, state, inverse_targets(r[0], topo, state)
        x = fb.features(state)[None]
        norm_pred = self.runner.predict_norm(x, r, fb.graphs(self.runner.dilations), x_base)[0]
        return topo, state, inverse_targets(norm_pred, topo, state)

    def predict(self, network: dict, changes: list | None) -> dict[str, dict]:
        topo, state, m = self.predict_arrays(network, changes)
        out: dict[str, dict] = {}
        for i, eid in enumerate(topo.edge_ids):
            if state.closed[i]:
                out[eid] = {"occupancy": 0.0, "mean_speed": 0.0, "travel_time": None, "closed": True}
            else:
                out[eid] = {"occupancy": round(float(m[i, 0]), 4), "mean_speed": round(float(m[i, 1]), 3),
                            "travel_time": round(float(m[i, 2]), 2), "closed": False}
        return out

    def info(self) -> dict:
        return {"model": self.meta["model"], "dataset": self.meta["dataset"],
                "created_utc": self.meta.get("created_utc"),
                "test_metrics": self.meta.get("test_metrics")}


def _same_state(a: State, b: State) -> bool:
    return (np.array_equal(a.lanes, b.lanes) and np.array_equal(a.speed, b.speed)
            and np.array_equal(a.closed, b.closed) and np.array_equal(a.tl, b.tl))


_DEFAULT: Predictor | None = None
_DEFAULT_LOCK = threading.Lock()


def load_predictor(weights_dir: str | Path | None = None) -> Predictor:
    """Модель по умолчанию загружается один раз на процесс."""
    global _DEFAULT
    if weights_dir is not None:
        return Predictor(weights_dir)
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = Predictor()
        return _DEFAULT


def predict(network: dict, changes: list) -> dict:
    """Прогноз по каждому ребру: {edge_id: {occupancy, mean_speed, travel_time, closed}}.

    Бросает ValueError при некорректном изменении (неизвестный тип или
    target_id, недопустимое значение) — backend может вернуть это как HTTP 400.
    """
    return load_predictor().predict(network, changes)


def model_info() -> dict:
    """Сведения о загруженной модели (для отображения в интерфейсе/логах)."""
    return load_predictor().info()


def default_network_path() -> Path:
    """Сеть по умолчанию для CLI: sim/data/network.json, иначе сеть, на которой обучена модель."""
    for p in (REPO_ROOT / "sim" / "data" / "network.json", WEIGHTS_DIR / "network.json"):
        if p.exists():
            return p
    raise FileNotFoundError("Не найдена сеть: укажите --network")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Прогноз GNN для сети с изменениями")
    ap.add_argument("--network", type=Path, default=None, help="network.json (по умолчанию — авто)")
    ap.add_argument("--changes", default="[]", help="JSON-список изменений")
    ap.add_argument("--changes-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="куда сохранить прогноз (JSON)")
    ap.add_argument("--top", type=int, default=10, help="показать N рёбер с наибольшим ростом занятости")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    net_path = args.network or default_network_path()
    network = json.loads(net_path.read_text(encoding="utf-8"))
    changes = json.loads(args.changes_file.read_text(encoding="utf-8") if args.changes_file else args.changes)
    p = load_predictor()
    before = p.predict(network, [])
    try:
        after = p.predict(network, changes)
    except ValueError as e:
        raise SystemExit(f"Ошибка в изменениях: {e}") from None
    if args.out:
        args.out.write_text(json.dumps(after, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Прогноз сохранён: {args.out}")
    print(f"Сеть: {net_path} ({len(after)} рёбер), изменений: {len(changes)}; модель: "
          f"{p.meta['model']['title']}")
    deltas = sorted(((after[e]["occupancy"] - before[e]["occupancy"], e) for e in after
                     if not after[e]["closed"]), reverse=True)
    print(f"{'ребро':<28}{'занятость до':>14}{'после':>9}{'скорость, м/с':>16}{'время, с':>11}")
    for d, e in deltas[: args.top]:
        a, b = after[e], before[e]
        print(f"{e:<28}{b['occupancy']:>14.3f}{a['occupancy']:>9.3f}"
              f"{b['mean_speed']:>8.2f}→{a['mean_speed']:<7.2f}{b['travel_time']:>6.1f}→{a['travel_time']:.1f}")


if __name__ == "__main__":
    main()
