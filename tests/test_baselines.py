"""Null and Markov baselines.

Two invariants matter:
1. On a deterministic chain (A → B → A → B → …), the Markov baseline
   nails it (accuracy = 1.0) while the most-common baseline only gets
   the half it agrees with by chance.
2. The Markov coverage drops below 1 if the val split contains a
   current-task that never appears as a current-task in train.
"""

from __future__ import annotations

import pandas as pd

from models.baselines import (
    evaluate_baselines,
    fit_markov,
    fit_most_common,
    predict_markov,
    predict_most_common,
)


def _toy_chain(n_cases=10, n_per_case=6) -> pd.DataFrame:
    """A,B,A,B,... per case. Every transition A→B and B→A is deterministic."""
    rows = []
    for cid in range(n_cases):
        for i in range(n_per_case):
            tid = i % 2
            next_tid = (i + 1) % 2 if i < n_per_case - 1 else None
            if next_tid is None:
                continue  # last event in case has no next_task; mirror dropna
            rows.append({"case_id": f"c{cid}", "task_id": tid, "next_task": next_tid})
    return pd.DataFrame(rows)


def test_markov_perfect_on_deterministic_chain():
    df = _toy_chain()
    # Full df as both train and val — for a deterministic process, the
    # Markov baseline must hit 1.0.
    result = evaluate_baselines(df, df)
    assert result["markov_accuracy"] == 1.0
    # Most-common predicts a single label so on an alternating chain it
    # gets only the rows whose label happens to match the mode. Crucial
    # property: the trivial baseline is *strictly* worse than Markov.
    assert result["most_common_accuracy"] < result["markov_accuracy"]


def test_markov_coverage_lt_one_when_val_has_unseen_current_task():
    train = pd.DataFrame({
        "case_id": ["a", "a"],
        "task_id":  [0, 1],
        "next_task": [1, 0],
    })
    val = pd.DataFrame({
        "case_id": ["b", "b"],
        # task_id=2 was never seen in train as a current-task.
        "task_id":  [0, 2],
        "next_task": [1, 0],
    })
    result = evaluate_baselines(train, val)
    assert result["markov_coverage"] == 0.5
    # The unseen task fell back to the most-common label, which may or
    # may not be right — that's the cost of the fallback, not a bug.


def test_baseline_metrics_dict_shape():
    df = _toy_chain()
    result = evaluate_baselines(df, df)
    assert {
        "most_common_label",
        "most_common_accuracy",
        "markov_accuracy",
        "markov_coverage",
        "num_val_events",
    } <= result.keys()
