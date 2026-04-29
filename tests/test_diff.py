"""Diff report between two run directories."""

from __future__ import annotations

import json
from pathlib import Path

from gnn_cli.diff import diff_runs


def _make_run(tmp_path: Path, name: str, metrics: dict):
    run = tmp_path / name
    (run / "metrics").mkdir(parents=True)
    for fname, payload in metrics.items():
        (run / "metrics" / fname).write_text(json.dumps(payload))
    return run


def test_diff_surfaces_numeric_changes(tmp_path):
    a = _make_run(tmp_path, "a", {"gat_metrics.json": {"accuracy": 0.5, "mcc": 0.1}})
    b = _make_run(tmp_path, "b", {"gat_metrics.json": {"accuracy": 0.7, "mcc": 0.1}})
    report = diff_runs(str(a), str(b))
    assert "accuracy" in report
    assert "0.5" in report and "0.7" in report
    # Unchanged metrics should not appear.
    assert "mcc" not in report


def test_diff_handles_missing_files(tmp_path):
    a = _make_run(tmp_path, "a", {
        "gat_metrics.json": {"accuracy": 0.5},
        "extra.json": {"x": 1},
    })
    b = _make_run(tmp_path, "b", {"gat_metrics.json": {"accuracy": 0.5}})
    report = diff_runs(str(a), str(b))
    assert "extra.json" in report  # listed under "only in A"


def test_diff_clustering_lists_moved_tasks(tmp_path):
    a = _make_run(tmp_path, "a", {
        "clustering_results.json": {"task_clusters": {"A": 0, "B": 0, "C": 1}},
    })
    b = _make_run(tmp_path, "b", {
        "clustering_results.json": {"task_clusters": {"A": 0, "B": 1, "C": 1}},
    })
    report = diff_runs(str(a), str(b))
    # B moved from cluster 0 to 1 — must appear; A unchanged — must not.
    assert "B" in report
    assert "Clustering" in report


def test_no_changes_message(tmp_path):
    a = _make_run(tmp_path, "a", {"gat_metrics.json": {"accuracy": 0.5}})
    b = _make_run(tmp_path, "b", {"gat_metrics.json": {"accuracy": 0.5}})
    report = diff_runs(str(a), str(b))
    assert "No differences" in report
