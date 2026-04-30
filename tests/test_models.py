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
from models.gat_model import (
    NextTaskGAT,
    compute_graph_label,
    expected_calibration_error,
)
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


def test_ece_zero_when_perfectly_calibrated():
    """If confidence == accuracy, ECE = 0. Use a one-hot prob matrix that's
    correct everywhere — confidence is 1.0, accuracy is 1.0, ECE = 0."""
    n, k = 20, 4
    y_true = torch.arange(n) % k
    y_prob = torch.zeros(n, k)
    y_prob[torch.arange(n), y_true] = 1.0
    assert expected_calibration_error(y_true, y_prob) == 0.0


def test_ece_high_when_uniformly_overconfident():
    """All predictions wrong but at probability 1 → ECE = 1 (max)."""
    n, k = 20, 4
    y_true = torch.zeros(n, dtype=torch.long)
    # Always predict class 1 with probability 1, but truth is class 0.
    y_prob = torch.zeros(n, k)
    y_prob[:, 1] = 1.0
    ece = expected_calibration_error(y_true, y_prob)
    assert ece > 0.99


def test_lstm_predict_time_returns_tuple(synthetic_event_log):
    """Multi-task LSTM: forward returns (logits, dt_pred); both heads
    backprop through the shared LSTM hidden state."""
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    train_seq, _ = prepare_sequence_data(df, val_frac=0.2, seed=0)
    Xp, Xl, y, _ = make_padded_dataset(train_seq, num_cls=len(le_task.classes_))
    assert hasattr(Xp, "dt_targets"), \
        "make_padded_dataset must attach dt_targets when prefixes carry dt"

    model = NextActivityLSTM(
        num_cls=len(le_task.classes_), emb_dim=8, hidden_dim=8,
        predict_time=True,
    )
    out = model(Xp[:4], Xl[:4])
    assert isinstance(out, tuple) and len(out) == 2
    logits, dt_pred = out
    assert dt_pred.shape == (4,)


def test_lstm_temperature_calibration_returns_positive_scalar(synthetic_event_log):
    """fit_temperature_lstm must return a finite positive temperature."""
    from models.lstm_model import fit_temperature_lstm

    df, le_task, _ = encode_categoricals(synthetic_event_log)
    train_seq, _ = prepare_sequence_data(df, val_frac=0.2, seed=0)
    Xp, Xl, y, _ = make_padded_dataset(train_seq, num_cls=len(le_task.classes_))
    model = NextActivityLSTM(num_cls=len(le_task.classes_), emb_dim=8, hidden_dim=8)
    # No training — just exercise the calibration loop on noise.
    T = fit_temperature_lstm(model, Xp[:32], Xl[:32], y[:32], batch_size=16, device=torch.device("cpu"))
    assert T > 0 and T == T  # finite, positive


def test_per_class_metrics_shape():
    """per_class_metrics carries one row per class plus macro / weighted F1."""
    from gnn_cli.stages import per_class_metrics
    import numpy as np

    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    out = per_class_metrics(y_true, y_pred, class_names=["A", "B", "C"])
    assert set(out["per_class"].keys()) == {"A", "B", "C"}
    for v in out["per_class"].values():
        assert {"precision", "recall", "f1", "support"} <= v.keys()
    assert 0.0 <= out["macro_f1"] <= 1.0
    assert 0.0 <= out["weighted_f1"] <= 1.0


def test_top_k_accuracy_perfect_when_truth_first():
    """If argmax = truth on every row, top-1 = top-3 = top-5 = 1.0."""
    from models.gat_model import top_k_accuracy
    n, k = 50, 8
    y_true = torch.arange(n) % k
    y_prob = torch.zeros(n, k)
    y_prob[torch.arange(n), y_true] = 1.0
    assert top_k_accuracy(y_true, y_prob, 1) == 1.0
    assert top_k_accuracy(y_true, y_prob, 3) == 1.0
    assert top_k_accuracy(y_true, y_prob, 5) == 1.0


def test_top_k_accuracy_recovers_with_higher_k():
    """When truth is consistently 2nd-best, top-1=0, top-3=1.0."""
    from models.gat_model import top_k_accuracy
    n, k = 20, 4
    y_true = torch.zeros(n, dtype=torch.long)
    y_prob = torch.zeros(n, k)
    y_prob[:, 1] = 0.9   # argmax → class 1 (wrong)
    y_prob[:, 0] = 0.1   # 2nd best → class 0 (correct)
    assert top_k_accuracy(y_true, y_prob, 1) == 0.0
    assert top_k_accuracy(y_true, y_prob, 2) == 1.0
    assert top_k_accuracy(y_true, y_prob, 3) == 1.0


def test_mrr_perfect_and_random():
    """MRR=1 when truth is always rank-1; MRR≈1/k for uniform priors."""
    from models.gat_model import mean_reciprocal_rank
    n, k = 100, 5
    y_true = torch.arange(n) % k
    # Perfect: truth always rank 1.
    perfect = torch.zeros(n, k)
    perfect[torch.arange(n), y_true] = 1.0
    assert mean_reciprocal_rank(y_true, perfect) == 1.0
    # Worst: truth always rank k. With k=5 → 1/5 = 0.2.
    worst = torch.ones(n, k)
    worst[torch.arange(n), y_true] = 0.0
    assert abs(mean_reciprocal_rank(y_true, worst) - 0.2) < 1e-6


def test_transformer_forward_shape(synthetic_event_log):
    """Transformer returns logits of shape (batch, num_cls)."""
    from models.transformer_model import NextActivityTransformer

    df, le_task, _ = encode_categoricals(synthetic_event_log)
    train_seq, _ = prepare_sequence_data(df, val_frac=0.2, seed=0)
    Xp, Xl, y, _ = make_padded_dataset(train_seq, num_cls=len(le_task.classes_))
    model = NextActivityTransformer(
        num_cls=len(le_task.classes_),
        emb_dim=8, hidden_dim=8, num_layers=1, num_heads=2,
        max_len=int(Xp.shape[1]) + 4,
    )
    out = model(Xp[:4], Xl[:4])
    assert out.shape == (4, len(le_task.classes_))


def test_transformer_predict_time_returns_tuple(synthetic_event_log):
    """With predict_time=True, forward returns (logits, dt_pred)."""
    from models.transformer_model import NextActivityTransformer

    df, le_task, _ = encode_categoricals(synthetic_event_log)
    train_seq, _ = prepare_sequence_data(df, val_frac=0.2, seed=0)
    Xp, Xl, y, _ = make_padded_dataset(train_seq, num_cls=len(le_task.classes_))
    model = NextActivityTransformer(
        num_cls=len(le_task.classes_),
        emb_dim=8, hidden_dim=8, num_layers=1, num_heads=2,
        predict_time=True, max_len=int(Xp.shape[1]) + 4,
    )
    out = model(Xp[:4], Xl[:4])
    assert isinstance(out, tuple) and len(out) == 2
    logits, dt_pred = out
    assert dt_pred.shape == (4,)


def test_transformer_emb_dim_must_divide_heads():
    """Constructor rejects emb_dim not divisible by num_heads."""
    from models.transformer_model import NextActivityTransformer

    with pytest.raises(ValueError, match="divisible"):
        NextActivityTransformer(num_cls=4, emb_dim=10, num_heads=4)


def test_bootstrap_ci_brackets_point_estimate():
    """The 95% bootstrap CI must contain the point estimate."""
    import numpy as np
    from models.gat_model import bootstrap_ci

    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.integers(0, 4, size=n)
    y_pred = y_true.copy()
    y_pred[:60] = (y_pred[:60] + 1) % 4  # 80% accuracy

    def acc(yt, yp):
        return float((yt == yp).mean())

    point = acc(y_true, y_pred)
    lo, hi = bootstrap_ci(y_true, y_pred, acc, n_resamples=500, seed=0)
    assert lo <= point <= hi
    # CI should be tight (well below 0.10 wide on n=300, p=0.8).
    assert hi - lo < 0.10


def test_bootstrap_ci_widens_on_small_n():
    """CI widens when val set shrinks — sampling variance grows."""
    import numpy as np
    from models.gat_model import bootstrap_ci

    rng = np.random.default_rng(0)

    def acc(yt, yp):
        return float((yt == yp).mean())

    # Small n.
    n_small = 30
    y = rng.integers(0, 4, size=n_small)
    yp = y.copy()
    yp[:6] = (yp[:6] + 1) % 4
    lo_s, hi_s = bootstrap_ci(y, yp, acc, n_resamples=500, seed=0)
    width_small = hi_s - lo_s

    # Large n.
    n_big = 3000
    y = rng.integers(0, 4, size=n_big)
    yp = y.copy()
    yp[:600] = (yp[:600] + 1) % 4
    lo_b, hi_b = bootstrap_ci(y, yp, acc, n_resamples=500, seed=0)
    width_big = hi_b - lo_b

    assert width_small > width_big * 3  # roughly sqrt(100) wider
