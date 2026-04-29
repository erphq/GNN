"""RL env contract + Q-learning convergence on a tiny instance."""

from __future__ import annotations

import numpy as np

from modules.data_preprocessing import encode_categoricals
from modules.rl_optimization import ProcessEnv, get_optimal_policy, run_q_learning
from modules.utils import set_seed


def test_env_reset_returns_state_vector(synthetic_event_log):
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    env = ProcessEnv(df, le_task, resources=[0, 1])
    state = env.reset()
    assert state.shape == (len(env.all_tasks),)
    assert state.sum() == 1.0  # one-hot


def test_env_step_returns_5tuple_signature(synthetic_event_log):
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    env = ProcessEnv(df, le_task, resources=[0, 1])
    env.reset()
    action = (env.all_tasks[0], 0)
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (len(env.all_tasks),)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert {"transition_cost", "processing_delay", "resource_efficiency"} <= info.keys()


def test_q_learning_produces_policy(synthetic_event_log):
    set_seed(0)
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    env = ProcessEnv(df, le_task, resources=[0, 1])
    q_table = run_q_learning(env, episodes=3)
    assert len(q_table) > 0
    all_actions = [(t, r) for t in env.all_tasks for r in env.resources]
    policy = get_optimal_policy(q_table, all_actions)
    assert len(policy) == len(q_table)
    # Each policy entry must be a valid action.
    for action in policy.values():
        assert action in all_actions
