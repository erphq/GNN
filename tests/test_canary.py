"""Reference-metrics canary.

Runs ``gnn smoke --device cpu --seed 42 --num-cases 50 …`` in a
subprocess and asserts every metric in
``tests/golden_smoke_metrics.json`` is within a per-key tolerance of
the saved golden. This catches silent regressions in the full
pipeline — a model accuracy drop, an unexpected ECE explosion, a
conformance-precision change — without needing every contributor to
remember to inspect run output.

To regenerate the golden after an intentional behavior change::

    .venv/bin/gnn smoke --device cpu --seed 42 --num-cases 50 \
        --epochs-gat 2 --epochs-lstm 2 --rl-episodes 5 \
        --out-dir /tmp/golden
    python -c '...'  # see capture script in commit 5a2c... or simply
                     # copy the metrics/*.json files into
                     # tests/golden_smoke_metrics.json'

Per-key tolerances are deliberately generous so this test stays green
across CPU vendors / BLAS implementations; tighten them if drift
becomes a concern.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden_smoke_metrics.json"

# Per-metric tolerances. Floats are absolute deltas; the value picks the
# tightest tolerance still robust to BLAS / device / version drift.
TOLERANCES_FLOAT = {
    "accuracy": 0.05,
    "macro_f1": 0.05,
    "weighted_f1": 0.05,
    "mcc": 0.10,
    "precision": 0.10,
    "recall": 0.10,
    "f1": 0.10,
    "lift_over_markov": 0.05,
    "lift_over_most_common": 0.05,
    "ece_before_calibration": 0.05,
    "ece_after_calibration": 0.05,
    "temperature": 1.0,
    "most_common_accuracy": 0.05,
    "markov_accuracy": 0.05,
    "markov_coverage": 0.05,
    "conformance_fitness": 0.10,
    "conformance_precision": 0.20,
    "conformance_f_score": 0.20,
    "cycle_time_95th_percentile_h": 5.0,
    "dt_mae_hours": 5.0,
    "time_loss_weight": 0.0,
    "seed": 0.0,
}
TOLERANCES_INT = {
    "num_val_events": 5,
    "num_long_cases": 5,
    "total_traces": 5,
    "num_deviant_traces": 5,
    "num_rare_transitions": 5,
    "num_tasks": 0,
    "num_resources": 0,
    "num_cases_total": 0,
    "num_cases_train": 5,
    "num_cases_val": 5,
    "support": 5,
    "most_common_label": 0,
    "num_states": 50,
    "num_actions": 0,
}


def _compare(label: str, actual, golden):
    """Recursive comparison with per-key tolerance lookup."""
    if isinstance(golden, dict):
        assert isinstance(actual, dict), f"{label}: expected dict, got {type(actual)}"
        # Every golden key must appear in actual; drift to a missing key
        # would silently bypass the canary, so we require coverage.
        for k, gv in golden.items():
            assert k in actual, f"{label}: missing key {k!r}"
            _compare(f"{label}.{k}", actual[k], gv)
    elif isinstance(golden, list):
        # Lists are tested for length only — order/values may shift.
        assert isinstance(actual, list), f"{label}: expected list"
        assert len(actual) == len(golden), \
            f"{label}: list length {len(actual)} vs golden {len(golden)}"
    elif isinstance(golden, bool):
        assert actual == golden, f"{label}: {actual} vs {golden}"
    elif isinstance(golden, int):
        key = label.split(".")[-1]
        tol = TOLERANCES_INT.get(key, 5)
        assert abs(int(actual) - golden) <= tol, \
            f"{label}: {actual} vs golden {golden} (int tol {tol})"
    elif isinstance(golden, float):
        key = label.split(".")[-1]
        tol = TOLERANCES_FLOAT.get(key, 0.10)
        assert abs(float(actual) - golden) <= tol, \
            f"{label}: {actual} vs golden {golden} (float tol {tol})"
    else:
        assert str(actual) == str(golden), f"{label}: {actual!r} vs {golden!r}"


@pytest.mark.canary
def test_smoke_metrics_match_golden(tmp_path):
    """Run the canary smoke and compare every metric to the committed golden.

    Marked with @pytest.mark.canary so it can be skipped via
    ``pytest -m 'not canary'`` on slow machines or when iterating.
    """
    if not GOLDEN.exists():
        pytest.skip("tests/golden_smoke_metrics.json not committed")

    out_dir = tmp_path / "smoke"
    cmd = [
        sys.executable, "-m", "gnn_cli", "smoke",
        "--device", "cpu",
        "--seed", "42",
        "--num-cases", "50",
        "--epochs-gat", "2",
        "--epochs-lstm", "2",
        "--rl-episodes", "5",
        "--out-dir", str(out_dir),
    ]
    # PYTHONPATH=repo_root so `python -m gnn_cli` resolves regardless of
    # whether the package was installed (CI does `pip install -e .`,
    # but local dev may not have).
    repo_root = str(Path(__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": repo_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env,
    )
    assert result.returncode == 0, (
        f"smoke failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    run_dirs = list(out_dir.glob("run_*"))
    assert len(run_dirs) == 1, f"expected 1 run dir, got {run_dirs}"
    metrics_dir = run_dirs[0] / "metrics"

    golden = json.loads(GOLDEN.read_text())
    for fname, golden_metrics in golden.items():
        path = metrics_dir / fname
        assert path.exists(), f"missing metric file: {fname}"
        actual = json.loads(path.read_text())
        if fname == "rl_results.json":
            actual = {k: v for k, v in actual.items() if k != "policy"}
        if fname == "preprocessing_info.json":
            actual.pop("date_range", None)
        _compare(fname, actual, golden_metrics)
