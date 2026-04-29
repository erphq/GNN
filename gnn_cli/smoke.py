"""Synthetic event-log generator for end-to-end smoke testing.

Produces a CSV with the columns the pipeline expects (case_id, task_name,
timestamp, resource, amount). The generated process is a small
linear/branching workflow with realistic-ish noise — enough to exercise
every stage without external data.
"""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timedelta
from typing import List

TASKS = [
    "Submit",
    "Review",
    "Approve",
    "Process",
    "Notify",
    "Archive",
]

RESOURCES = ["alice", "bob", "carol", "dave", "erin"]


def generate_synthetic_csv(
    out_path: str,
    num_cases: int = 80,
    seed: int = 42,
) -> str:
    """Write a synthetic event log to `out_path` and return the path.

    Each case is a randomized walk through TASKS with skips and a small
    chance of revisiting Review (so the graph isn't strictly a DAG).
    """
    rng = random.Random(seed)
    base = datetime(2025, 1, 1, 9, 0, 0)
    rows: List[dict] = []

    for cid in range(num_cases):
        case_id = f"case_{cid:04d}"
        ts = base + timedelta(hours=rng.randint(0, 24 * 30))
        amount = round(rng.uniform(50, 5000), 2)

        idx = 0
        while idx < len(TASKS):
            task = TASKS[idx]
            ts += timedelta(minutes=rng.randint(15, 240))
            rows.append(
                {
                    "case_id": case_id,
                    "task_name": task,
                    "timestamp": ts.isoformat(),
                    "resource": rng.choice(RESOURCES),
                    "amount": amount,
                }
            )
            if task == "Review" and rng.random() < 0.15:
                continue
            if rng.random() < 0.05 and idx < len(TASKS) - 1:
                idx += 2
            else:
                idx += 1

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "task_name", "timestamp", "resource", "amount"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path
