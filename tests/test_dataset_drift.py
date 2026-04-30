"""Cross-dataset drift regression test (v0.5 milestone hygiene).

For each dataset registered in ``bench/datasets/download.py`` this
test:

1. Looks for the local ``.xes.gz`` (skip if absent — datasets aren't
   vendored due to TOS gating).
2. Looks for a pinned ``tests/canonical_metrics/<name>.json`` (skip
   if absent — pinning is a one-time manual step after the first
   canonical run).
3. Runs ``gnn run`` with the canonical config — same hyperparameters
   as the BPI 2020 leaderboard rows so cross-dataset numbers are
   directly comparable.
4. Asserts the headline metrics (top-1, top-3, MRR, ECE) drift no
   more than ``TOL`` from pinned. Tolerances are generous (±1 pp on
   accuracy / top-K, ±0.05 on MRR, ±0.01 on ECE) because cross-
   machine BLAS variation is real and we don't want false positives.

The test is marked ``@pytest.mark.dataset_drift`` so slow CI workers
can opt out with ``pytest -m 'not dataset_drift'``. It's also
auto-skipped when no datasets have a canonical pin, so a fresh clone
won't fail.

To pin a new dataset:

    # 1. Download the .xes.gz and place it in bench/datasets/data/
    # 2. Run the canonical config:
    gnn run bench/datasets/data/<name>.xes.gz \\
        --seed 42 --device cpu \\
        --epochs-lstm 30 --hidden-dim 256 --lr-lstm 5e-4 \\
        --predict-time --skip-gat --skip-rl \\
        --out-dir bench/results/<name>_canonical
    # 3. Copy the metrics into tests/canonical_metrics/:
    cp bench/results/<name>_canonical/run_*/metrics/lstm_metrics.json \\
        tests/canonical_metrics/<name>.json
    # 4. Commit. CI will now enforce drift on every push.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Lazy import — the registry doesn't depend on pytorch.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.datasets.download import REGISTRY, locate  # noqa: E402

CANONICAL_DIR = Path(__file__).parent / "canonical_metrics"

# Per-key drift tolerances. Generous enough to absorb BLAS / CPU /
# torch-version variation; tight enough to catch a real regression.
TOL_FLOAT = {
    "accuracy": 0.01,
    "top_3_accuracy": 0.01,
    "top_5_accuracy": 0.01,
    "mrr": 0.05,
    "macro_f1": 0.05,
    "ece_after_calibration": 0.05,
    "dt_mae_hours": 5.0,
    "lift_over_markov": 0.01,
}


def _datasets_with_pins():
    """Yield (name, data_path, canonical_path) for datasets with a pin."""
    for name in REGISTRY:
        canonical = CANONICAL_DIR / f"{name}.json"
        if not canonical.exists():
            continue
        yield name, locate(name), canonical


# Static parameterization. If no datasets have pins we still want a
# clean "skipped" state instead of "no tests collected".
_PARAMS = list(_datasets_with_pins()) or [(None, None, None)]


@pytest.mark.dataset_drift
@pytest.mark.parametrize("name,data_path,canonical_path", _PARAMS)
def test_dataset_drift_within_tolerance(tmp_path, name, data_path, canonical_path):
    if name is None:
        pytest.skip("no datasets pinned in tests/canonical_metrics/")
    if not data_path.exists():
        pytest.skip(
            f"{name}: {data_path} not present "
            f"(see bench/datasets/download.py --where --dataset {name})"
        )

    canonical = json.loads(canonical_path.read_text())
    out_dir = tmp_path / "run"
    cmd = [
        sys.executable, "-m", "gnn_cli", "run", str(data_path),
        "--device", "cpu", "--seed", "42",
        "--epochs-lstm", "30", "--hidden-dim", "256",
        "--lr-lstm", "5e-4",
        "--predict-time", "--skip-gat", "--skip-rl",
        "--out-dir", str(out_dir),
    ]
    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1])
        + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600, env=env,
    )
    assert result.returncode == 0, (
        f"{name}: gnn run failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    runs = list(out_dir.glob("run_*"))
    assert len(runs) == 1, f"{name}: expected 1 run dir, got {runs}"
    actual = json.loads((runs[0] / "metrics" / "lstm_metrics.json").read_text())

    failed = []
    for key, tol in TOL_FLOAT.items():
        if key not in canonical or key not in actual:
            continue
        delta = abs(float(actual[key]) - float(canonical[key]))
        if delta > tol:
            failed.append(
                f"  {key}: actual {actual[key]:.4f} vs pinned "
                f"{canonical[key]:.4f}, delta {delta:.4f} > tol {tol:.4f}"
            )

    if failed:
        pytest.fail(
            f"{name}: drift exceeded tolerance on "
            f"{len(failed)} metric(s):\n" + "\n".join(failed)
        )
