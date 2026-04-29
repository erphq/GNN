"""Smoke tests for the GAT and LSTM modules — forward pass, gradient flow,
prefix split is case-isolated."""

from __future__ import annotations

import pytest
import torch

from modules.data_preprocessing import (
    apply_feature_scaler,
    build_graph_data,
    encode_categoricals,
    fit_feature_scaler,
    split_cases,
)
from models.gat_model import NextTaskGAT, compute_graph_label
from models.lstm_model import (
    NextActivityLSTM,
    make_padded_dataset,
    prepare_sequence_data,
)


def test_gat_forward_runs_node_level(synthetic_event_log):
    """Default (node-level) head emits one logit row per event."""
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    scaler, _ = fit_feature_scaler(df)
    df = apply_feature_scaler(df, scaler)
    graphs = build_graph_data(df)
    g = graphs[0]
    model = NextTaskGAT(
        input_dim=5, hidden_dim=8, output_dim=len(le_task.classes_),
        num_layers=2, heads=2, dropout=0.0,
    )
    batch = torch.zeros(g.x.shape[0], dtype=torch.long)
    out = model(g.x, g.edge_index, batch)
    assert out.shape == (g.x.shape[0], len(le_task.classes_))


def test_gat_forward_runs_graph_level(synthetic_event_log):
    """Legacy v0.2 path: with node_level=False, output is one row per graph."""
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    scaler, _ = fit_feature_scaler(df)
    df = apply_feature_scaler(df, scaler)
    graphs = build_graph_data(df)
    g = graphs[0]
    model = NextTaskGAT(
        input_dim=5, hidden_dim=8, output_dim=len(le_task.classes_),
        num_layers=2, heads=2, dropout=0.0,
        node_level=False,
    )
    batch = torch.zeros(g.x.shape[0], dtype=torch.long)
    out = model(g.x, g.edge_index, batch)
    assert out.shape == (1, len(le_task.classes_))


def test_gat_backward_flows_node_level(synthetic_event_log):
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    scaler, _ = fit_feature_scaler(df)
    df = apply_feature_scaler(df, scaler)
    graphs = build_graph_data(df)
    g = graphs[0]
    model = NextTaskGAT(
        5, 8, len(le_task.classes_), num_layers=1, heads=2, dropout=0.0,
    )
    batch = torch.zeros(g.x.shape[0], dtype=torch.long)
    out = model(g.x, g.edge_index, batch)
    # Per-node labels come straight from g.y (already per-event).
    loss = torch.nn.functional.cross_entropy(out, g.y.long())
    loss.backward()
    for name, p in model.named_parameters():
        if (p.requires_grad and "convs.0" in name) or "fc" in name:
            assert p.grad is not None, name


def test_gat_predict_time_returns_tuple_and_supervises(synthetic_event_log):
    """Multi-task head: forward returns (logits, dt_pred); MSE backprop works."""
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    scaler, _ = fit_feature_scaler(df)
    df = apply_feature_scaler(df, scaler)
    graphs = build_graph_data(df)
    g = graphs[0]
    assert hasattr(g, "dt"), "build_graph_data must attach per-node dt when present"
    model = NextTaskGAT(
        5, 8, len(le_task.classes_),
        num_layers=1, heads=2, dropout=0.0,
        node_level=True, predict_time=True,
    )
    batch = torch.zeros(g.x.shape[0], dtype=torch.long)
    out = model(g.x, g.edge_index, batch)
    assert isinstance(out, tuple) and len(out) == 2
    logits, dt_pred = out
    assert dt_pred.shape == (g.x.shape[0],)
    loss = (
        torch.nn.functional.cross_entropy(logits, g.y.long())
        + torch.nn.functional.mse_loss(dt_pred, g.dt)
    )
    loss.backward()
    assert model.dt_head.weight.grad is not None


def test_gat_predict_time_requires_node_level(synthetic_event_log):
    """predict_time on the graph-level head is a configuration error."""
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    with pytest.raises(ValueError, match="node_level=True"):
        NextTaskGAT(
            5, 8, len(le_task.classes_),
            num_layers=1, heads=2, dropout=0.0,
            node_level=False, predict_time=True,
        )


def test_gat_backward_flows_graph_level(synthetic_event_log):
    """Regression test for the legacy graph-level head."""
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    scaler, _ = fit_feature_scaler(df)
    df = apply_feature_scaler(df, scaler)
    graphs = build_graph_data(df)
    g = graphs[0]
    model = NextTaskGAT(
        5, 8, len(le_task.classes_), num_layers=1, heads=2, dropout=0.0,
        node_level=False,
    )
    batch = torch.zeros(g.x.shape[0], dtype=torch.long)
    out = model(g.x, g.edge_index, batch)
    label = compute_graph_label(g.y, batch).long()
    loss = torch.nn.functional.cross_entropy(out, label)
    loss.backward()
    for name, p in model.named_parameters():
        if (p.requires_grad and "convs.0" in name) or "fc" in name:
            assert p.grad is not None, name


def test_lstm_forward_runs(synthetic_event_log):
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    train_seq, _ = prepare_sequence_data(df, val_frac=0.2, seed=0)
    Xp, Xl, y, _ = make_padded_dataset(train_seq, num_cls=len(le_task.classes_))
    model = NextActivityLSTM(num_cls=len(le_task.classes_), emb_dim=8, hidden_dim=8)
    out = model(Xp[:4], Xl[:4])
    assert out.shape == (4, len(le_task.classes_))


def test_prefixes_are_case_isolated(synthetic_event_log):
    """The most important regression test in this repo.

    `prepare_sequence_data` must never produce a (train_seq, val_seq) pair
    where any case contributes prefixes to both halves — that's the data
    leak the original implementation had.
    """
    df, _, _ = encode_categoricals(synthetic_event_log)
    train, val = split_cases(df, val_frac=0.2, seed=0)
    train_seq, val_seq = prepare_sequence_data(df, train_df=train, val_df=val)

    # Reconstruct membership: first task in each prefix uniquely identifies
    # the case under our synthetic generator (case_id-stratified by start
    # time). For a more general check, we use case row presence directly.
    train_cases = set(train["case_id"])
    val_cases = set(val["case_id"])
    assert train_cases.isdisjoint(val_cases)

    # And the prefix counts should be reasonable: each case contributes
    # (n_events - 1) prefixes.
    train_expected = sum(
        max(0, len(g) - 1) for _, g in train.groupby("case_id")
    )
    assert len(train_seq) == train_expected
