"""Обучение независимых DQN-агентов светофоров.

    python -m agents.train                         # сеть sim/ или решётки 2×2 и 4×4
    python -m agents.train --scenario grid4x4 --episodes 40

Схема обучения:

1. Эпизод = 70 мин симуляции (1 ч спроса + 10 мин разгрузки), решение
   каждые 5 с → 840 шагов на агента.
2. Каждый эпизод — новый вариант спроса из 10 обучающих (семена 0–9,
   масштаб интенсивности случайный в 0.6–1.3 от номинала): агент учится
   на разных нагрузках, а не запоминает один сценарий.
3. Все агенты действуют одновременно в одной симуляции; каждый хранит
   свой опыт и обучает свою сеть (см. agents/dqn.py).
4. Поверх политики действует защитное правило максимального ожидания
   (env.MaxWaitGuard, 90 с) — и при обучении, и при оценке.
5. Каждые VALIDATE_EVERY эпизодов — проверка без исследования (ε = 0) на
   отдельном валидационном спросе (семя 50, не совпадает ни с обучающими,
   ни с тестовыми). Сохраняется лучший по средней очереди набор моделей.

Результат: agents/models/<сеть>/ — tls_<id>.zip (модели SB3), meta.json
(описание наблюдения, гиперпараметры, лучший эпизод), training_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import config  # noqa: E402

import numpy as np  # noqa: E402

from agents.dqn import IndependentDQN  # noqa: E402
from agents.env import MaxWaitGuard, make_env  # noqa: E402
from agents.evaluate import run_episode  # noqa: E402
from agents.networks import default_scenarios, get_scenario  # noqa: E402


def train(spec: str, episodes: int = config.TRAIN_EPISODES, seed: int = 0,
          validate_every: int = config.VALIDATE_EVERY, out_dir: Path | None = None) -> dict:
    sc = get_scenario(spec)
    out_dir = Path(out_dir or sc.models_dir)
    val_route = sc.eval_route("nominal", config.VALIDATION_SEED)

    # Пространства наблюдений/действий берём из самой среды
    probe = make_env(sc, sc.train_route(config.TRAIN_SEEDS[0]), "rl", seed=seed)
    obs_spaces = {ts: probe.observation_spaces(ts) for ts in probe.ts_ids}
    act_spaces = {ts: probe.action_spaces(ts) for ts in probe.ts_ids}
    probe.close()
    steps_per_episode = config.EPISODE_SECONDS // config.ENV_PARAMS["delta_time"]
    agents = IndependentDQN.create(obs_spaces, act_spaces, total_timesteps=episodes * steps_per_episode,
                                   seed=seed)
    print(f"[{sc.key}] агентов: {len(agents.models)}; эпизодов: {episodes} × {steps_per_episode} шагов")

    # Метрики режимов-ориентиров на валидационном спросе — для лога
    baseline = {m: run_episode(sc, val_route, m, seed=config.VALIDATION_SEED) for m in ("fixed", "actuated")}
    print(f"[{sc.key}] валидация: fixed — очередь {baseline['fixed']['avg_queue_length']:.2f}, "
          f"ожидание {baseline['fixed']['avg_waiting_time']:.1f} с; actuated — очередь "
          f"{baseline['actuated']['avg_queue_length']:.2f}, ожидание {baseline['actuated']['avg_waiting_time']:.1f} с")

    log_rows = []
    best = {"avg_queue_length": float("inf"), "episode": None}
    obs_layout = None
    t_start = time.time()
    for ep in range(1, episodes + 1):
        route_seed = config.TRAIN_SEEDS[(ep - 1) % len(config.TRAIN_SEEDS)]
        env = make_env(sc, sc.train_route(route_seed), "rl", seed=seed * 1000 + ep)
        t0 = time.time()
        obs = env.reset()
        if obs_layout is None:
            obs_layout = {ts: env.traffic_signals[ts].observation_fn.layout() for ts in env.ts_ids}
        rewards_sum, losses, done = 0.0, [], False
        eps_start = agents.epsilon()
        guard = MaxWaitGuard(env)
        while not done:
            # в буфер опыта попадает действие, которое реально выполнено
            actions = guard(agents.act(obs, deterministic=False))
            next_obs, rewards, dones, _ = env.step(actions)
            done = dones["__all__"]
            losses += agents.observe(obs, actions, rewards, next_obs, done)
            rewards_sum += sum(rewards.values())
            obs = next_obs
        env.close()
        n_tls = len(agents.models)
        row = {
            "episode": ep, "route_seed": route_seed,
            "epsilon": round(eps_start, 4),
            # награда = -0.1·очередь → средняя очередь на перекрёсток за эпизод
            "train_avg_queue": round(-rewards_sum / config.REWARD_SCALE / steps_per_episode / n_tls, 3),
            "train_reward": round(rewards_sum, 2),
            "loss": round(float(np.mean(losses)), 5) if losses else None,
            "guard_rate": round(guard.rate, 4),
            "episode_time_s": round(time.time() - t0, 1),
        }
        if ep % validate_every == 0 or ep == episodes:
            val = run_episode(sc, val_route, "rl", seed=config.VALIDATION_SEED, agents=agents)
            row.update({"val_avg_queue": round(val["avg_queue_length"], 3),
                        "val_avg_waiting_time": round(val["avg_waiting_time"], 2),
                        "val_avg_travel_time": round(val["avg_travel_time"], 2),
                        "val_guard_rate": round(val["guard_rate"], 4)})
            if val["avg_queue_length"] < best["avg_queue_length"]:
                best = {"avg_queue_length": val["avg_queue_length"], "episode": ep,
                        "avg_waiting_time": val["avg_waiting_time"], "avg_travel_time": val["avg_travel_time"]}
                agents.save(out_dir, meta={
                    "scenario": sc.key, "net_file": str(sc.net_file.relative_to(config.REPO_ROOT))
                    if sc.net_file.is_relative_to(config.REPO_ROOT) else str(sc.net_file),
                    "best_episode": ep, "episodes": episodes, "seed": seed,
                    "validation": best,
                    "baseline_validation": {m: {k: baseline[m][k] for k in
                                                ("avg_queue_length", "avg_waiting_time", "avg_travel_time")}
                                            for m in baseline},
                    "env_params": config.ENV_PARAMS, "reward_scale": config.REWARD_SCALE,
                    "max_wait_guard_s": config.MAX_WAIT_GUARD_S,
                    "dqn_params": {k: v for k, v in config.DQN_PARAMS.items()},
                    "observation_layout": obs_layout,
                })
                row["saved"] = 1
        log_rows.append(row)
        val_txt = (f" | вал.: очередь {row['val_avg_queue']:.2f}, ожидание {row['val_avg_waiting_time']:.1f} с"
                   + (" *" if row.get("saved") else "")) if "val_avg_queue" in row else ""
        print(f"[{sc.key}] эп. {ep:3d}/{episodes}  ε={row['epsilon']:.2f}  очередь {row['train_avg_queue']:.2f}"
              f"  loss {row['loss']}  {row['episode_time_s']:.0f} с{val_txt}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["episode", "route_seed", "epsilon", "train_avg_queue", "train_reward", "loss", "guard_rate",
              "val_avg_queue", "val_avg_waiting_time", "val_avg_travel_time", "val_guard_rate", "saved",
              "episode_time_s"]
    with open(out_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(log_rows)
    total = time.time() - t_start
    print(f"[{sc.key}] готово за {total / 60:.1f} мин; лучший эпизод {best['episode']} "
          f"(очередь {best['avg_queue_length']:.2f}); модели: {out_dir}")
    return {"scenario": sc.key, "best": best, "train_minutes": total / 60, "models_dir": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение независимых DQN-агентов светофоров")
    ap.add_argument("--scenario", nargs="+", default=None,
                    help="grid2x2 grid4x4 sim или путь к сети (по умолчанию — sim/ или решётки)")
    ap.add_argument("--episodes", type=int, default=config.TRAIN_EPISODES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate-every", type=int, default=config.VALIDATE_EVERY)
    # Для экспериментов (абляции): параметры записываются в meta.json модели
    # и автоматически применяются при её оценке.
    ap.add_argument("--min-green", type=int, default=None, help="минимальный зелёный, с (по умолчанию 5)")
    ap.add_argument("--gamma", type=float, default=None, help="коэффициент дисконтирования DQN (по умолчанию 0.95)")
    ap.add_argument("--out-dir", default=None, help="куда сохранить модели (по умолчанию agents/models/<сеть>)")
    args = ap.parse_args()
    if args.min_green is not None:
        config.ENV_PARAMS["min_green"] = args.min_green
    if args.gamma is not None:
        config.DQN_PARAMS["gamma"] = args.gamma
    specs = args.scenario or default_scenarios()
    if args.out_dir and len(specs) > 1:
        ap.error("--out-dir можно указать только для одной сети")
    for spec in specs:
        train(spec, episodes=args.episodes, seed=args.seed, validate_every=args.validate_every,
              out_dir=Path(args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    main()
