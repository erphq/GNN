"""Hypothesis-driven property tests for the preprocessing layer.

Property-based testing complements the example-based suite: instead of
hand-writing one fixture, we describe the *shape* of valid input and
let hypothesis search for counter-examples to invariants. The
preprocessing pipeline is the right place for this — its invariants are
crisp and the failure modes (split leakage, drop-row asymmetry, encoder
mis-alignment) are exactly the bugs the v0.2 audit caught the hard way.

Properties locked in here:

1. ``encode_categoricals`` is **deterministic** — same input twice
   gives identical output. (Catches an entire class of "label encoder
   shuffles between runs" bugs.)
2. ``split_cases`` produces **disjoint case sets** for any seed and
   any val_frac. The case-isolation invariant is the v0.2 audit's
   load-bearing fix.
3. ``split_cases`` is **size-monotonic** in val_frac: larger val_frac
   ⇒ more val cases, fewer train cases.
4. ``build_graph_data`` preserves the per-case event count (graph
   nodes == events with a next_task), regardless of case structure.
"""

from __future__ import annotations

import datetime as dt

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings

from modules.data_preprocessing import (
    apply_feature_scaler,
    build_graph_data,
    encode_categoricals,
    fit_feature_scaler,
    split_cases,
)


def _build_log(n_cases: int, events_per_case: int, n_tasks: int, seed: int) -> pd.DataFrame:
    """Build a synthetic event log with deterministic structure."""
    rng = np.random.default_rng(seed)
    tasks = [f"task_{i}" for i in range(n_tasks)]
    rows = []
    base = dt.datetime(2025, 1, 1)
    for c in range(n_cases):
        ts = base + dt.timedelta(hours=int(c))
        chosen = list(rng.choice(tasks, size=events_per_case))
        for ev_idx, t in enumerate(chosen):
            ts = ts + dt.timedelta(minutes=int(rng.integers(5, 90)))
            rows.append({
                "case_id": f"c{c:03d}",
                "task_name": t,
                "timestamp": ts,
                "resource": str(rng.choice(["alice", "bob"])),
                "amount": float(rng.uniform(10, 1000)),
            })
    return pd.DataFrame(rows)


@given(
    n_cases=st.integers(min_value=4, max_value=30),
    events_per_case=st.integers(min_value=3, max_value=10),
    n_tasks=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=30, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_encode_categoricals_is_deterministic(n_cases, events_per_case, n_tasks, seed):
    """Same input → identical encoded output."""
    df = _build_log(n_cases, events_per_case, n_tasks, seed)
    a, le_a, _ = encode_categoricals(df)
    b, le_b, _ = encode_categoricals(df)
    assert list(a.columns) == list(b.columns)
    assert (a["task_id"].to_numpy() == b["task_id"].to_numpy()).all()
    assert (a["next_task"].to_numpy() == b["next_task"].to_numpy()).all()
    assert list(le_a.classes_) == list(le_b.classes_)


@given(
    n_cases=st.integers(min_value=4, max_value=40),
    events_per_case=st.integers(min_value=2, max_value=8),
    val_frac=st.floats(min_value=0.1, max_value=0.5),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=30, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_split_cases_always_disjoint(n_cases, events_per_case, val_frac, seed):
    """The case-isolation invariant must hold for any (val_frac, seed)."""
    df = _build_log(n_cases, events_per_case, 4, seed)
    df, _, _ = encode_categoricals(df)
    train, val = split_cases(df, val_frac=val_frac, seed=seed)
    train_cases = set(train["case_id"])
    val_cases = set(val["case_id"])
    assert train_cases.isdisjoint(val_cases)
    assert train_cases | val_cases == set(df["case_id"])


@given(
    n_cases=st.integers(min_value=10, max_value=50),
    events_per_case=st.integers(min_value=3, max_value=8),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=20, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_split_cases_size_monotonic_in_val_frac(n_cases, events_per_case, seed):
    """Larger val_frac → at-least-as-many val cases."""
    df = _build_log(n_cases, events_per_case, 3, seed)
    df, _, _ = encode_categoricals(df)
    _, val_a = split_cases(df, val_frac=0.1, seed=seed)
    _, val_b = split_cases(df, val_frac=0.4, seed=seed)
    assert val_a["case_id"].nunique() <= val_b["case_id"].nunique()


@given(
    n_cases=st.integers(min_value=4, max_value=20),
    events_per_case=st.integers(min_value=3, max_value=8),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=20, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
def test_build_graph_preserves_event_count(n_cases, events_per_case, seed):
    """For each case, len(graph.x) equals the count of events with a next_task."""
    df = _build_log(n_cases, events_per_case, 3, seed)
    df, _, _ = encode_categoricals(df)
    scaler, _ = fit_feature_scaler(df)
    df = apply_feature_scaler(df, scaler)
    graphs = build_graph_data(df)
    cases = list(df.groupby("case_id"))
    assert len(graphs) == len(cases)
    for g, (_, sub) in zip(graphs, cases, strict=True):
        assert g.x.shape[0] == len(sub)
        # `y` is per-node next_task; same length as x.
        assert g.y.shape[0] == g.x.shape[0]
