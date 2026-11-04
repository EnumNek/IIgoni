"""agents/ — RL-агенты светофоров (независимые DQN на базе sumo-rl и stable-baselines3).

Главная точка входа для других частей системы::

    from agents.evaluate import evaluate
    metrics = evaluate("sim/data/network.json", mode="rl")  # "fixed" | "actuated" | "rl"

Подробности — в agents/README.md.
"""
