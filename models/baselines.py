"""Null and Markov baselines for next-task prediction.

A trained model's accuracy is uninterpretable on its own — process logs
have one mega-frequent task in many domains, and a "always predict the
mode" classifier already lands well above 0.5 on those. The two
baselines here are the standard scientific floor:

* **Most-common baseline.** Predict ``mode(next_task)`` for every event,
  ignoring context. This is the literal null model.
* **Markov baseline (1-st order).** Look up the most common next-task
  for the *current* task in the training set, predict that. This is
  the simplest model that uses any process structure at all.

If a deep model isn't beating the Markov baseline by a meaningful
margin, the gain is from class imbalance, not from learning the
process. Surfacing both numbers in every run forces the comparison.
"""

from __future__ import annotations


import numpy as np
import pandas as pd


def fit_most_common(train_df: pd.DataFrame, label_col: str = "next_task") -> int:
    """Return the modal next-task label in the training data."""
    return int(train_df[label_col].mode().iloc[0])


def predict_most_common(val_df: pd.DataFrame, mode_label: int) -> np.ndarray:
    """Predict the same modal label for every val event."""
    return np.full(len(val_df), mode_label, dtype=np.int64)


def fit_markov(
    train_df: pd.DataFrame,
    cur_col: str = "task_id",
    next_col: str = "next_task",
) -> dict[int, int]:
    """Per current-task lookup: most common next-task seen in training."""
    table: dict[int, int] = {}
    for cur, sub in train_df.groupby(cur_col):
        table[int(cur)] = int(sub[next_col].mode().iloc[0])
    return table


def predict_markov(
    val_df: pd.DataFrame,
    table: dict[int, int],
    fallback: int,
    cur_col: str = "task_id",
) -> np.ndarray:
    """Apply the Markov table to val events; fall back to ``fallback`` when
    a current-task wasn't seen in training (otherwise we'd lose val rows)."""
    return np.array(
        [table.get(int(c), fallback) for c in val_df[cur_col].values],
        dtype=np.int64,
    )


def evaluate_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str = "next_task",
) -> dict:
    """Fit + score both baselines. Returns a metrics dict ready to dump
    as ``baseline_metrics.json``."""
    mode_label = fit_most_common(train_df, label_col=label_col)
    mc_pred = predict_most_common(val_df, mode_label)
    mc_acc = float((mc_pred == val_df[label_col].values).mean())

    markov_table = fit_markov(train_df, next_col=label_col)
    mk_pred = predict_markov(val_df, markov_table, fallback=mode_label)
    mk_acc = float((mk_pred == val_df[label_col].values).mean())

    coverage = float(val_df["task_id"].isin(set(markov_table.keys())).mean())

    return {
        "most_common_label": mode_label,
        "most_common_accuracy": mc_acc,
        "markov_accuracy": mk_acc,
        "markov_coverage": coverage,
        "num_val_events": int(len(val_df)),
    }
