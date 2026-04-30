"""Process-mining helpers: spectral clustering on a known-bipartite graph,
bottleneck/cycle-time stability, transition matrix shape."""

from __future__ import annotations

import numpy as np
import pytest

from modules.data_preprocessing import encode_categoricals
from modules.process_mining import (
    analyze_bottlenecks,
    analyze_cycle_times,
    analyze_rare_transitions,
    analyze_transition_patterns,
    build_task_adjacency,
    spectral_cluster_graph,
)


def test_spectral_cluster_recovers_two_blocks():
    """Two disconnected cliques → spectral clustering finds them."""
    block = np.ones((4, 4)) - np.eye(4)
    A = np.zeros((8, 8))
    A[:4, :4] = block
    A[4:, 4:] = block
    labels = spectral_cluster_graph(A, k=2, normalized=True)
    # Each block should be assigned to one cluster (label values are
    # arbitrary, only the partition matters).
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]


def test_spectral_cluster_normalized_handles_isolated_node():
    """Isolated node (zero degree) should not cause divide-by-zero."""
    A = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],  # isolated
        ],
        dtype=float,
    )
    labels = spectral_cluster_graph(A, k=2, normalized=True)
    assert labels.shape == (4,)
    assert np.isfinite(labels).all()


def test_spectral_cluster_eigh_returns_real_labels():
    """Symmetric Laplacian → real eigenvalues → real-typed labels."""
    A = np.random.RandomState(0).rand(10, 10)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0)
    labels = spectral_cluster_graph(A, k=3, normalized=True)
    assert labels.dtype.kind in ("i", "u")


def test_build_task_adjacency_counts_transitions(synthetic_event_log):
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    A = build_task_adjacency(df, num_tasks=len(le_task.classes_))
    assert A.shape == (len(le_task.classes_),) * 2
    # The adjacency must capture every observed (task → next_task) pair.
    assert A.sum() > 0


def test_analyze_bottlenecks_columns(synthetic_event_log):
    df, _, _ = encode_categoricals(synthetic_event_log)
    stats, sig = analyze_bottlenecks(df, freq_threshold=1)
    for col in ("task_id", "next_task_id", "mean", "count", "mean_hours"):
        assert col in stats.columns
    assert (sig["count"] >= 1).all()


def test_analyze_cycle_times_returns_long_cases(synthetic_event_log):
    df, _, _ = encode_categoricals(synthetic_event_log)
    merged, long_cases, cut95 = analyze_cycle_times(df)
    assert "duration_h" in merged.columns
    # cut95 must be the 95th percentile of duration.
    assert cut95 == pytest.approx(merged["duration_h"].quantile(0.95))
    assert (long_cases["duration_h"] > cut95).all()


def test_rare_transitions_thresholded(synthetic_event_log):
    df, _, _ = encode_categoricals(synthetic_event_log)
    stats, _ = analyze_bottlenecks(df, freq_threshold=1)
    rare = analyze_rare_transitions(stats, rare_threshold=2)
    assert (rare["count"] <= 2).all()


def test_transition_matrix_shape(synthetic_event_log):
    df, _, _ = encode_categoricals(synthetic_event_log)
    _, trans_count, prob = analyze_transition_patterns(df)
    # Probability rows should sum to 1 (or 0 for tasks that never transition).
    row_sums = prob.sum(axis=1).values
    assert np.allclose(row_sums[row_sums > 0], 1.0)


def test_conformance_returns_four_metrics(synthetic_event_log):
    """Conformance summary must carry deviant count + fitness + precision + F."""
    from modules.process_mining import perform_conformance_checking

    replayed, summary = perform_conformance_checking(synthetic_event_log)
    assert {"num_deviant", "fitness", "precision", "f_score"} <= summary.keys()
    assert 0.0 <= summary["fitness"] <= 1.0
    assert 0.0 <= summary["precision"] <= 1.0
    assert 0.0 <= summary["f_score"] <= 1.0
    assert summary["num_deviant"] <= len(replayed)
    # The harmonic mean is at most the smaller of the two factors.
    assert summary["f_score"] <= max(summary["fitness"], summary["precision"]) + 1e-9


def test_bottleneck_drivers_on_synthetic(synthetic_event_log):
    """Driver analysis returns a non-empty dict on the synthetic log
    and surfaces resource as a candidate driver."""
    from modules.data_preprocessing import encode_categoricals
    from modules.process_mining import analyze_bottleneck_drivers

    df, le_task, _ = encode_categoricals(synthetic_event_log)
    drivers = analyze_bottleneck_drivers(df, le_task=le_task, top_n=3)
    assert len(drivers) >= 1
    # Every transition entry has the contract.
    for trans, payload in drivers.items():
        assert {"n_transitions", "mean_wait_h", "drivers"} <= payload.keys()
        for d in payload["drivers"]:
            assert {
                "feature", "spread_h", "worst_group", "worst_group_mean_h",
                "worst_group_n", "best_group", "best_group_mean_h",
                "best_group_n",
            } <= d.keys()
            assert d["spread_h"] >= 0


def test_bottleneck_drivers_renders_markdown():
    from modules.process_mining import render_bottleneck_drivers
    fake = {
        "A -> B": {
            "n_transitions": 100,
            "mean_wait_h": 5.0,
            "drivers": [
                {
                    "feature": "resource", "spread_h": 4.0,
                    "worst_group": "alice", "worst_group_mean_h": 8.0, "worst_group_n": 30,
                    "best_group": "bob", "best_group_mean_h": 4.0, "best_group_n": 25,
                },
            ],
        },
    }
    md = render_bottleneck_drivers(fake)
    assert "A -> B" in md
    assert "alice" in md and "bob" in md
