"""Независимые DQN-агенты (Independent DQN) поверх stable-baselines3.

Идея: у каждого светофора своя нейросеть Q(s, a), свой буфер опыта и
свой ε-жадный выбор действий. Агенты не обмениваются параметрами и
сообщениями; влияние соседей агент видит только через изменение
собственных очередей. Это простейшая и хорошо масштабируемая схема
многоагентного RL (IQL, Tan 1993; для светофоров — напр. Chu et al. 2019).

Почему не ``model.learn()``: в SB3 ``learn()`` сам «крутит» среду, а у нас
одна симуляция SUMO на всех агентов — все светофоры должны сделать ход
одновременно. Поэтому цикл «наблюдение → действия всех агентов → шаг
симуляции» написан вручную, а от SB3 используются готовые компоненты:
сеть и оптимизатор (``DQN.train``), буфер опыта, расписание ε,
обновление целевой сети (``DQN._on_step``) и сохранение/загрузка.
Шаги в точности повторяют ``OffPolicyAlgorithm.collect_rollouts``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.logger import Logger

from agents import config

# Сети маленькие (2×64) — многопоточность torch только мешает.
torch.set_num_threads(int(os.environ.get("AGENTS_TORCH_THREADS", "1")))


class _SpacesOnlyEnv(gym.Env):
    """«Паспорт» среды: SB3 требует env при создании модели ради пространств."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):  # pragma: no cover - среда только для пространств
        raise NotImplementedError("Шаги делает общая симуляция SUMO, а не эта заглушка")


def _safe_name(ts_id: str) -> str:
    return re.sub(r"[^\w.-]", "_", ts_id)


class IndependentDQN:
    """Набор независимых DQN-агентов: ``{tls_id: DQN}``."""

    def __init__(self, models: dict[str, DQN], meta: dict | None = None):
        self.models = models
        self.meta = meta or {}

    # ------------------------------------------------------------------ создание
    @classmethod
    def create(cls, obs_spaces: dict, act_spaces: dict, total_timesteps: int,
               seed: int = 0, params: dict | None = None) -> "IndependentDQN":
        params = dict(config.DQN_PARAMS if params is None else params)
        models = {}
        for i, ts in enumerate(sorted(obs_spaces)):
            model = DQN("MlpPolicy", _SpacesOnlyEnv(obs_spaces[ts], act_spaces[ts]),
                        seed=seed + i, device="cpu", verbose=0, **params)
            model.set_logger(Logger(folder=None, output_formats=[]))
            model._total_timesteps = total_timesteps  # нужно для расписания ε
            models[ts] = model
        return cls(models)

    # ------------------------------------------------------------------ действия
    def epsilon(self) -> float:
        m = next(iter(self.models.values()))
        if m.num_timesteps < m.learning_starts:
            return 1.0
        return float(m.exploration_schedule(m._current_progress_remaining))

    def act(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> dict[str, int]:
        """Выбор фазы каждым агентом. deterministic=False — ε-жадно (обучение)."""
        actions = {}
        eps = 0.0 if deterministic else self.epsilon()
        for ts, o in obs.items():
            model = self.models[ts]
            if not deterministic and np.random.rand() < eps:
                actions[ts] = int(model.action_space.sample())
            else:
                a, _ = model.predict(o, deterministic=True)
                actions[ts] = int(a)
        return actions

    # ------------------------------------------------------------------ обучение
    def observe(self, obs: dict, actions: dict, rewards: dict, next_obs: dict, done: bool) -> list[float]:
        """Сохраняет переход каждого агента и делает шаг обучения.

        Эпизод в sumo-rl заканчивается по времени, а не из-за «терминального»
        состояния, поэтому переход помечается как усечённый
        (``TimeLimit.truncated``): SB3 тогда продолжает бутстрэп Q-значения,
        а не считает будущее нулевым.
        """
        losses = []
        for ts, model in self.models.items():
            if ts not in actions or ts not in next_obs:
                continue
            model.replay_buffer.add(
                obs[ts][None, :], next_obs[ts][None, :],
                np.array([actions[ts]]), np.array([rewards[ts]], dtype=np.float32),
                np.array([done]), [{"TimeLimit.truncated": done}],
            )
            model.num_timesteps += 1
            model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
            model._on_step()  # ε по расписанию + обновление целевой сети
            freq = model.train_freq.frequency if hasattr(model.train_freq, "frequency") else int(model.train_freq)
            if model.num_timesteps > model.learning_starts and model.num_timesteps % freq == 0:
                model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)
                loss = model.logger.name_to_value.get("train/loss")
                if loss is not None:
                    losses.append(float(loss))
        return losses

    # ------------------------------------------------------------------ сохранение
    def save(self, out_dir: Path, meta: dict) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for ts, model in self.models.items():
            name = f"tls_{_safe_name(ts)}.zip"
            model.save(out_dir / name)
            files[ts] = {
                "file": name,
                "obs_dim": int(model.observation_space.shape[0]),
                "n_actions": int(model.action_space.n),
            }
        meta = dict(meta, agents=files)
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.meta = meta

    @classmethod
    def load(cls, models_dir: Path) -> "IndependentDQN":
        meta_file = Path(models_dir) / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"Нет обученных моделей в {models_dir}. Обучите: python -m agents.train --scenario <сеть>")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        models = {ts: DQN.load(Path(models_dir) / info["file"], device="cpu")
                  for ts, info in meta["agents"].items()}
        return cls(models, meta)

    def compatible(self, env) -> tuple[list[str], list[str]]:
        """-> (светофоры среды, к которым подходит модель; описание остальных)."""
        ok, problems = [], []
        for ts in env.ts_ids:
            if ts not in self.models:
                problems.append(f"нет модели для светофора {ts}")
                continue
            m = self.models[ts]
            obs_dim = env.observation_spaces(ts).shape[0]
            n_act = env.action_spaces(ts).n
            if m.observation_space.shape[0] != obs_dim or m.action_space.n != n_act:
                problems.append(f"{ts}: модель ожидает obs={m.observation_space.shape[0]}, "
                                f"фаз={m.action_space.n}; в сети obs={obs_dim}, фаз={n_act}")
            else:
                ok.append(ts)
        return ok, problems

    def check_compatible(self, env) -> None:
        """Проверяет, что модели подходят ко всем светофорам среды (id, размерности)."""
        _, problems = self.compatible(env)
        if problems:
            raise ValueError("Модели не подходят к сети (сеть изменилась — нужно дообучение):\n  "
                             + "\n  ".join(problems))
