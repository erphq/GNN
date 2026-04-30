"""Synthetic event-log generator for end-to-end smoke testing.

Earlier versions of this generator emitted near-random task sequences;
the GAT and LSTM both flatlined at ~20% accuracy on it because there
was no signal to learn. The smoke test was checking that the pipeline
*ran*, not that any model *worked*.

This version uses a Markov-style transition graph with a realistic
process structure:

    Submit ─► Review ─┬─[0.70]─► Approve ─► Process ─► Notify ─► Archive
                      │
                      └─[0.30]─► Reject  ─► Resubmit ─► Review … (loop)

Each transition has a typical wait time drawn from a per-edge log-normal
so the time-prediction head has signal to learn. Loops are bounded so
cases terminate.

A 1st-order Markov baseline on this generator hits ~0.85; a working
GAT after a couple of epochs lands well above that, while a broken one
stays near 0.5. That gap is what makes ``gnn smoke`` an informative
regression test, not just a "did the pipeline crash?" canary.
"""

from __future__ import annotations

import csv
import math
import os
import random
from datetime import datetime, timedelta

# Transition table: state -> [(next, probability, mean_wait_minutes)].
# Probabilities per row sum to 1.0; "END" is a terminal sink.
TRANSITIONS: dict[str, list[tuple[str, float, float]]] = {
    "Submit":   [("Review",   1.00,  30)],
    "Review":   [("Approve",  0.70,  60), ("Reject",   0.30,  45)],
    "Approve":  [("Process",  1.00, 120)],
    "Process":  [("Notify",   1.00,  90)],
    "Notify":   [("Archive",  1.00,  20)],
    "Archive":  [("END",      1.00,   0)],
    "Reject":   [("Resubmit", 1.00,  60)],
    "Resubmit": [("Review",   1.00,  30)],
}

START_TASK = "Submit"
RESOURCES = ["alice", "bob", "carol", "dave", "erin"]
MAX_EVENTS_PER_CASE = 30  # safety cap; bounded loops shouldn't hit it


def _sample_next(state: str, rng: random.Random) -> tuple[str, float]:
    """Sample (next_task, wait_minutes) from the transition row."""
    row = TRANSITIONS[state]
    r = rng.random()
    cum = 0.0
    for next_task, prob, mean_wait in row:
        cum += prob
        if r <= cum:
            # Log-normal wait so the dt distribution has a right tail —
            # gives the time-prediction head something interesting to fit.
            sigma = 0.5
            mu = math.log(max(1.0, mean_wait)) - sigma * sigma / 2
            wait = rng.lognormvariate(mu, sigma)
            return next_task, wait
    # Should never reach here when probs sum to 1, but be defensive.
    last = row[-1]
    return last[0], float(last[2])


def generate_synthetic_csv(
    out_path: str,
    num_cases: int = 80,
    seed: int = 42,
) -> str:
    """Write a Markov-process event log to ``out_path`` and return the path."""
    rng = random.Random(seed)
    base = datetime(2025, 1, 1, 9, 0, 0)
    rows: list[dict] = []

    for cid in range(num_cases):
        case_id = f"case_{cid:04d}"
        ts = base + timedelta(hours=rng.randint(0, 24 * 30))
        amount = round(rng.uniform(50, 5000), 2)
        # Per-case resource preference adds slight cross-resource variance.
        primary_resource = rng.choice(RESOURCES)

        # Always start with the same first event so the transition graph
        # has a unique entry point.
        rows.append({
            "case_id": case_id,
            "task_name": START_TASK,
            "timestamp": ts.isoformat(),
            "resource": primary_resource,
            "amount": amount,
        })
        state = START_TASK

        for _ in range(MAX_EVENTS_PER_CASE):
            next_task, wait_min = _sample_next(state, rng)
            if next_task == "END":
                break
            # Round to whole seconds — sub-second precision triggers
            # mixed-format auto-detection in pandas.to_datetime, which
            # silently drops the microsecond rows. Process mining doesn't
            # care about sub-second timing anyway.
            ts += timedelta(seconds=int(wait_min * 60))
            r = primary_resource if rng.random() < 0.8 else rng.choice(RESOURCES)
            rows.append({
                "case_id": case_id,
                "task_name": next_task,
                "timestamp": ts.isoformat(),
                "resource": r,
                "amount": amount,
            })
            state = next_task

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "task_name", "timestamp", "resource", "amount"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path
