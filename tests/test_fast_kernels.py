"""Parity between the Rust kernels in `pm_fast` and the pure-Python
implementations they replace. Skipped entirely if the extension is not
installed (e.g. on a toolchain-free environment)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules._fast import AVAILABLE, build_padded_prefixes_fast, build_task_adjacency_fast
from modules.data_preprocessing import encode_categoricals
from modules.process_mining import build_task_adjacency

pytestmark = pytest.mark.skipif(
    not AVAILABLE, reason="pm_fast extension not installed"
)


def _py_padded_prefixes(df: pd.DataFrame):
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
            X[r, c] = t + 1
        L[r] = len(pfx)
        Y[r] = nxt
    return X, L, Y, max_len


def test_adjacency_parity(synthetic_event_log):
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    num_tasks = len(le_task.classes_)

    # Force the Python path by bypassing the wrapper.
    py_adj = np.zeros((num_tasks, num_tasks), dtype=np.float32)
    for _cid, cdata in df.groupby("case_id"):
        cdata = cdata.sort_values("timestamp")
        seq = cdata["task_id"].values
        for i in range(len(seq) - 1):
            py_adj[seq[i], seq[i + 1]] += 1.0

    rs_adj = build_task_adjacency_fast(df, num_tasks)
    assert np.allclose(py_adj, rs_adj)


def test_padded_prefixes_parity(synthetic_event_log):
    df, _, _ = encode_categoricals(synthetic_event_log)

    pyX, pyL, pyY, py_max = _py_padded_prefixes(df)
    rsX, rsL, rsY, rs_max = build_padded_prefixes_fast(df)
    assert py_max == rs_max
    assert np.array_equal(pyX, rsX)
    assert np.array_equal(pyL, rsL)
    assert np.array_equal(pyY, rsY)


def test_wrapper_uses_fast_path(synthetic_event_log):
    """`modules.process_mining.build_task_adjacency` should pick up the
    Rust kernel transparently via the `_fast` bridge."""
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    A = build_task_adjacency(df, num_tasks=len(le_task.classes_))
    A_direct = build_task_adjacency_fast(df, num_tasks=len(le_task.classes_))
    assert np.array_equal(A, A_direct)
