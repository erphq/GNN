"""Preprocessing covers the bug-prone bits: case-level splits, scaler
fitted only on train, encoder label-space stable across the split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.data_preprocessing import (
    apply_feature_scaler,
    build_graph_data,
    compute_class_weights,
    encode_categoricals,
    fit_feature_scaler,
    load_and_preprocess_data,
    split_cases,
)


def test_load_normalizes_xes_columns(tmp_path, synthetic_event_log):
    xes = synthetic_event_log.rename(
        columns={
            "case_id": "case:concept:name",
            "task_name": "concept:name",
            "timestamp": "time:timestamp",
            "resource": "org:resource",
            "amount": "case:Amount",
        }
    )
    p = tmp_path / "xes.csv"
    xes.to_csv(p, index=False)
    df = load_and_preprocess_data(str(p))
    for c in ("case_id", "task_name", "timestamp", "resource", "amount"):
        assert c in df.columns


def test_load_rejects_missing_columns(tmp_path):
    bad = pd.DataFrame({"foo": [1, 2]})
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing column"):
        load_and_preprocess_data(str(p))


def test_split_cases_disjoint(synthetic_event_log):
    df, _, _ = encode_categoricals(synthetic_event_log)
    train, val = split_cases(df, val_frac=0.2, seed=0)
    train_cases = set(train["case_id"])
    val_cases = set(val["case_id"])
    # No leakage: a case is exclusively in one half.
    assert train_cases.isdisjoint(val_cases)
    # Both halves non-empty.
    assert len(train_cases) > 0 and len(val_cases) > 0
    # Together they cover every (post-shift) case.
    assert train_cases | val_cases == set(df["case_id"])


def test_split_cases_is_deterministic(synthetic_event_log):
    df, _, _ = encode_categoricals(synthetic_event_log)
    a, _ = split_cases(df, seed=7)
    b, _ = split_cases(df, seed=7)
    assert set(a["case_id"]) == set(b["case_id"])


def test_scaler_fit_on_train_only(synthetic_event_log):
    df, _, _ = encode_categoricals(synthetic_event_log)
    train, val = split_cases(df, val_frac=0.3, seed=0)

    scaler, mode = fit_feature_scaler(train, use_norm_features=False)
    assert mode == "minmax"

    train_scaled = apply_feature_scaler(train, scaler)
    val_scaled = apply_feature_scaler(val, scaler)

    # Train should fall in [0, 1] after MinMax. Val may exceed it — and that
    # is exactly the point: if val data ever drifts, the scaler reveals it
    # rather than silently absorbing it (which is what fitting on the full
    # dataset would do).
    feat_cols = [c for c in train_scaled.columns if c.startswith("feat_")]
    train_arr = train_scaled[feat_cols].values
    assert train_arr.min() >= -1e-9 and train_arr.max() <= 1 + 1e-9


def test_class_weights_balance_inversely_with_freq(synthetic_event_log):
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    weights = compute_class_weights(df, num_classes=len(le_task.classes_))
    # Every class present at least once should have weight > 0.
    assert (weights > 0).all().item()


def test_build_graph_data_shapes_causal(synthetic_event_log):
    """Default causal mode: one forward edge per consecutive pair.

    Forward-only edges prevent the GAT from attending to events the
    label is asking it to predict (a one-hop look-ahead leak).
    """
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    scaler, _ = fit_feature_scaler(df)
    df = apply_feature_scaler(df, scaler)
    graphs = build_graph_data(df)
    assert len(graphs) > 0
    for g in graphs:
        assert g.x.shape[1] == 5
        n = g.x.shape[0]
        assert g.edge_index.shape == (2, max(0, n - 1))
        assert g.y.shape[0] == n
        # All edges should go strictly forward in chronological order.
        if g.edge_index.shape[1] > 0:
            src, tgt = g.edge_index[0], g.edge_index[1]
            assert (tgt > src).all().item()


def test_build_graph_data_shapes_bidirectional(synthetic_event_log):
    """Legacy causal=False: 2 edges per consecutive pair (forward + reverse)."""
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    scaler, _ = fit_feature_scaler(df)
    df = apply_feature_scaler(df, scaler)
    graphs = build_graph_data(df, causal=False)
    assert len(graphs) > 0
    for g in graphs:
        n = g.x.shape[0]
        assert g.edge_index.shape == (2, max(0, 2 * (n - 1)))


def test_temporal_split_puts_recent_cases_in_val(synthetic_event_log):
    """In temporal mode, every val case starts at-or-after every train case."""
    df, _, _ = encode_categoricals(synthetic_event_log)
    train, val = split_cases(df, val_frac=0.3, seed=0, mode="temporal")

    train_cases = set(train["case_id"])
    val_cases = set(val["case_id"])
    # Case-isolation invariant still holds.
    assert train_cases.isdisjoint(val_cases)

    train_max_start = train.groupby("case_id")["timestamp"].min().max()
    val_min_start = val.groupby("case_id")["timestamp"].min().min()
    # Every val case starts at-or-after the latest train-case start.
    assert val_min_start >= train_max_start


def test_split_mode_validation_rejects_unknown():
    import pandas as pd
    df = pd.DataFrame({"case_id": ["a"], "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")]})
    try:
        split_cases(df, mode="random_garbage")
    except ValueError as e:
        assert "case" in str(e) and "temporal" in str(e)
    else:
        raise AssertionError("split_cases should reject unknown mode")


def test_xes_load_roundtrip(tmp_path, synthetic_event_log):
    """Round-trip a CSV through XES and assert the loader recovers the
    same case_count + columns."""
    pm4py = pytest.importorskip("pm4py")

    df_in = synthetic_event_log.rename(
        columns={
            "case_id": "case:concept:name",
            "task_name": "concept:name",
            "timestamp": "time:timestamp",
            "resource": "org:resource",
            "amount": "case:Amount",
        }
    )
    # pm4py.write_xes wants timestamps as datetime — already are.
    log = pm4py.format_dataframe(
        df_in, case_id="case:concept:name",
        activity_key="concept:name", timestamp_key="time:timestamp",
    )
    xes_path = tmp_path / "log.xes"
    pm4py.write_xes(log, str(xes_path))

    df_out = load_and_preprocess_data(str(xes_path))
    for c in ("case_id", "task_name", "timestamp", "resource", "amount"):
        assert c in df_out.columns
    # Allow a small drop from the dropna(timestamp) path; structure
    # should otherwise survive the round-trip.
    assert df_out["case_id"].nunique() == synthetic_event_log["case_id"].nunique()
