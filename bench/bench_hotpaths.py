"""Benchmark the Rust hot paths against their pure-Python equivalents.

Generates a synthetic event log of N cases, runs each kernel `repeats`
times, prints the median wall-clock per implementation and the speedup.

Run from the repo root:

    python bench/bench_hotpaths.py             # default 5000 cases
    python bench/bench_hotpaths.py --num-cases 50000 --repeats 5

Requires `pm_fast` to be installed (build via `maturin develop --release`
from `pm_fast/`). Falls back to "skipped" if it isn't.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import pandas as pd

from modules._fast import (
    AVAILABLE,
    build_padded_prefixes_fast,
    build_task_adjacency_fast,
)
from modules.data_preprocessing import encode_categoricals


def make_synthetic_log(num_cases: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tasks = ["receive", "review", "approve", "pay", "audit", "close"]
    resources = ["alice", "bob", "carol", "dave"]
    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for cid in range(num_cases):
        n = int(rng.integers(4, 14))
        chosen = rng.choice(tasks, size=n)
        cur = base + pd.Timedelta(hours=int(cid))
        for t in chosen:
            cur = cur + pd.Timedelta(minutes=int(rng.integers(5, 90)))
            rows.append(
                {
                    "case_id": f"c{cid:06d}",
                    "task_name": t,
                    "timestamp": cur,
                    "resource": str(rng.choice(resources)),
                    "amount": float(rng.uniform(10, 1000)),
                }
            )
    return pd.DataFrame(rows)


def time_call(fn, *args, repeats: int = 5) -> float:
    """Median wall-clock seconds across `repeats` runs."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def py_build_task_adjacency(df: pd.DataFrame, num_tasks: int) -> np.ndarray:
    A = np.zeros((num_tasks, num_tasks), dtype=np.float32)
    for _cid, cdata in df.groupby("case_id"):
        cdata = cdata.sort_values("timestamp")
        seq = cdata["task_id"].values
        for i in range(len(seq) - 1):
            A[seq[i], seq[i + 1]] += 1.0
    return A


def py_build_padded_prefixes(df: pd.DataFrame):
    samples = []
    for _cid, cdata in df.groupby("case_id"):
        cdata = cdata.sort_values("timestamp")
        seq = cdata["task_id"].tolist()
        for i in range(1, len(seq)):
            samples.append((seq[:i], seq[i]))
    if not samples:
        return np.empty((0, 0), np.int64), np.empty(0, np.int64), np.empty(0, np.int64), 0
    max_len = max(len(s[0]) for s in samples)
    X = np.zeros((len(samples), max_len), dtype=np.int64)
    L = np.zeros(len(samples), dtype=np.int64)
    Y = np.zeros(len(samples), dtype=np.int64)
    for r, (pfx, nxt) in enumerate(samples):
        for c, t in enumerate(pfx):
            X[r, c] = t + 1  # +1 padding shift
        L[r] = len(pfx)
        Y[r] = nxt
    return X, L, Y, max_len


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-cases", type=int, default=5000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Generating synthetic log: {args.num_cases} cases ...", flush=True)
    df = make_synthetic_log(args.num_cases, seed=args.seed)
    df, le_task, _ = encode_categoricals(df)
    num_tasks = len(le_task.classes_)
    print(f"  → {len(df):,} rows, {num_tasks} unique tasks")

    print(f"\nMethod                              | median wall-clock ({args.repeats} runs)")
    print(f"{'-' * 36}-+-{'-' * 35}")

    print(f"{'build_task_adjacency (Python)':<36} | ", end="", flush=True)
    py_t = time_call(py_build_task_adjacency, df, num_tasks, repeats=args.repeats)
    print(f"{py_t * 1000:>9.2f} ms")

    if AVAILABLE:
        print(f"{'build_task_adjacency (Rust)':<36} | ", end="", flush=True)
        rs_t = time_call(build_task_adjacency_fast, df, num_tasks, repeats=args.repeats)
        print(f"{rs_t * 1000:>9.2f} ms     ({py_t / rs_t:.1f}x speedup)")

        # Correctness check.
        py_adj = py_build_task_adjacency(df, num_tasks)
        rs_adj = build_task_adjacency_fast(df, num_tasks)
        assert np.allclose(py_adj, rs_adj), "Rust adjacency disagrees with Python"
    else:
        print(f"{'build_task_adjacency (Rust)':<36} | (skipped — pm_fast not installed)")

    print(f"{'build_padded_prefixes (Python)':<36} | ", end="", flush=True)
    py_t = time_call(py_build_padded_prefixes, df, repeats=args.repeats)
    print(f"{py_t * 1000:>9.2f} ms")

    if AVAILABLE:
        print(f"{'build_padded_prefixes (Rust)':<36} | ", end="", flush=True)
        rs_t = time_call(build_padded_prefixes_fast, df, repeats=args.repeats)
        print(f"{rs_t * 1000:>9.2f} ms     ({py_t / rs_t:.1f}x speedup)")

        # Correctness check (sort prefix lists since order is implementation-defined).
        pyX, pyL, pyY, py_max = py_build_padded_prefixes(df)
        rsX, rsL, rsY, rs_max = build_padded_prefixes_fast(df)
        assert py_max == rs_max
        # Both produce prefixes in case_id traversal order; should match exactly.
        assert np.array_equal(pyX, rsX), "X mismatch"
        assert np.array_equal(pyL, rsL), "L mismatch"
        assert np.array_equal(pyY, rsY), "Y mismatch"
    else:
        print(f"{'build_padded_prefixes (Rust)':<36} | (skipped — pm_fast not installed)")


if __name__ == "__main__":
    main()
