"""Rust hot-path kernels for the gnn pipeline.

Pure-Python wrappers that import the maturin-built `_native` extension.
The wrappers handle pandas → numpy conversion so callers can pass dataframes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from . import _native

__version__ = _native.__version__


def _factorize_case_ids(df: pd.DataFrame) -> np.ndarray:
    """case_id → contiguous int64. The Rust loops only need equality, not
    the original string identity, so factorization is the cheapest stable
    representation."""
    codes, _ = pd.factorize(df["case_id"], sort=False)
    return codes.astype(np.int64, copy=False)


def build_task_adjacency(df: pd.DataFrame, num_tasks: int) -> np.ndarray:
    """Same contract as `modules.process_mining.build_task_adjacency` —
    expects `df` sorted by (case_id, timestamp), with `task_id` populated."""
    case_codes = _factorize_case_ids(df)
    task_ids = df["task_id"].to_numpy(dtype=np.int64, copy=False)
    return _native.build_task_adjacency(case_codes, task_ids, num_tasks)


def build_padded_prefixes(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Replacement for `_build_prefixes` + `make_padded_dataset` fused.

    Expects `df` sorted by (case_id, timestamp). Returns
    `(X_padded, X_lens, Y, max_len)` as numpy arrays so callers can wrap
    them in `torch.from_numpy(...)` without copying."""
    case_codes = _factorize_case_ids(df)
    task_ids = df["task_id"].to_numpy(dtype=np.int64, copy=False)
    return _native.build_padded_prefixes(case_codes, task_ids)
