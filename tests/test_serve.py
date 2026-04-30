"""FastAPI inference endpoint.

Two contracts:
1. /predict returns ranked candidates for a known prefix; probabilities sum to ≤ 1
   and are sorted descending.
2. /predict_suffix returns up to ``beam`` continuations; their normalized
   probabilities sum to 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _train_quick_model(tmp_path, csv_path):
    """Run a 1-epoch smoke pipeline and return the run_dir + the CSV path
    so the serve app has something to load."""
    import subprocess
    import sys

    out_dir = tmp_path / "run"
    cmd = [
        sys.executable, "-m", "gnn_cli", "run", str(csv_path),
        "--device", "cpu", "--seed", "42",
        "--epochs-gat", "1", "--epochs-lstm", "1", "--rl-episodes", "1",
        "--hidden-dim", "16", "--skip-rl", "--skip-analyze", "--skip-viz",
        "--skip-cluster",
        "--out-dir", str(out_dir),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    runs = list(out_dir.glob("run_*"))
    assert runs, "no run dir produced"
    return runs[0]


def test_predict_endpoint(tmp_path, csv_path):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("starlette")
    from fastapi.testclient import TestClient

    from gnn_cli.serve import build_app

    run_dir = _train_quick_model(tmp_path, csv_path)
    app = build_app(
        run_dir=str(run_dir), data_path=str(csv_path),
        seq_arch="lstm", hidden_dim=16, predict_time=False, seed=42,
    )
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["seq_arch"] == "lstm"

    # Pick a known activity from the synthetic log.
    import pandas as pd
    df = pd.read_csv(csv_path)
    first_activity = str(df["task_name"].iloc[0])

    resp = client.post("/predict", json={"prefix": [first_activity], "k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["candidates"]) == 3
    probs = [c["probability"] for c in data["candidates"]]
    # Sorted descending; sum ≤ 1.
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0 + 1e-6


def test_predict_suffix_endpoint(tmp_path, csv_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from gnn_cli.serve import build_app

    run_dir = _train_quick_model(tmp_path, csv_path)
    app = build_app(
        run_dir=str(run_dir), data_path=str(csv_path),
        seq_arch="lstm", hidden_dim=16, predict_time=False, seed=42,
    )
    client = TestClient(app)

    import pandas as pd
    df = pd.read_csv(csv_path)
    first_activity = str(df["task_name"].iloc[0])

    resp = client.post(
        "/predict_suffix",
        json={"prefix": [first_activity], "beam": 3, "max_steps": 3},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 1 <= len(data["completions"]) <= 3
    probs = [c["prob"] for c in data["completions"]]
    assert abs(sum(probs) - 1.0) < 1e-5


def test_unknown_activity_rejected(tmp_path, csv_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from gnn_cli.serve import build_app

    run_dir = _train_quick_model(tmp_path, csv_path)
    app = build_app(
        run_dir=str(run_dir), data_path=str(csv_path),
        seq_arch="lstm", hidden_dim=16, predict_time=False, seed=42,
    )
    client = TestClient(app)
    resp = client.post("/predict", json={"prefix": ["NOT_AN_ACTIVITY"], "k": 3})
    assert resp.status_code == 400
    assert "unknown activity" in resp.json()["detail"].lower()
