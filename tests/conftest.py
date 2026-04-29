"""Shared fixtures: a tiny synthetic event log with three tasks, two
resources, deterministic timestamps, and enough cases to exercise the
case-level split (50 cases, 4-7 events each)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_event_log() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    base_ts = pd.Timestamp("2026-01-01", tz="UTC")
    tasks = ["receive", "review", "approve", "pay"]
    resources = ["alice", "bob"]

    for cid in range(50):
        n_events = int(rng.integers(4, 8))
        cur = base_ts + pd.Timedelta(hours=int(cid))
        # Always start with "receive"; remaining events are random.
        chosen = ["receive"] + list(rng.choice(tasks[1:], size=n_events - 1))
        for ev_idx, t in enumerate(chosen):
            cur = cur + pd.Timedelta(minutes=int(rng.integers(5, 90)))
            rows.append(
                {
                    "case_id": f"c{cid:03d}",
                    "task_name": t,
                    "timestamp": cur,
                    "resource": str(rng.choice(resources)),
                    "amount": float(rng.uniform(10, 1000)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def csv_path(tmp_path, synthetic_event_log) -> str:
    p = tmp_path / "log.csv"
    synthetic_event_log.to_csv(p, index=False)
    return str(p)
