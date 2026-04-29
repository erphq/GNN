#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing for process mining.

Loads event-log CSVs, encodes categoricals, builds per-case PyG graphs,
and computes class weights. The feature-scaling helpers are split into
fit / apply so callers can fit on a train subset only and transform the
val subset with the train-fitted scaler (no leakage).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from torch_geometric.data import Data

DEFAULT_REQUIRED_COLS: Tuple[str, ...] = (
    "case_id",
    "task_name",
    "timestamp",
    "resource",
    "amount",
)

FEATURE_COLS: Tuple[str, ...] = (
    "task_id",
    "resource_id",
    "amount",
    "day_of_week",
    "hour_of_day",
)

FEAT_OUT_COLS: Tuple[str, ...] = tuple(f"feat_{c}" for c in FEATURE_COLS)


def load_and_preprocess_data(
    data_path: str,
    required_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a process-mining event log from CSV, normalize column names, and
    sort by (case_id, timestamp). Drops rows with unparseable timestamps.

    Accepts both XES-style (`case:concept:name`, `concept:name`, …) and
    snake_case (`case_id`, `task_name`, …) column conventions.
    """
    if required_cols is None:
        required_cols = DEFAULT_REQUIRED_COLS

    df = pd.read_csv(data_path)
    df.rename(
        columns={
            "case:id": "case_id",
            "case:concept:name": "case_id",
            "concept:name": "task_name",
            "time:timestamp": "timestamp",
            "org:resource": "resource",
            "case:Amount": "amount",
        },
        inplace=True,
        errors="ignore",
    )

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"Missing column '{c}' in CSV. Found: {df.columns.tolist()}"
            )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values(["case_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def encode_categoricals(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Add `task_id`, `resource_id`, `next_task`, and time features.

    Encoders are fit on the full dataframe — that is intentional, because
    the val split must use the same label space. Numeric *scaling* is a
    separate step (see `fit_feature_scaler` / `apply_feature_scaler`) and
    must be fit on training rows only.
    """
    df = df.copy()
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour

    le_task = LabelEncoder()
    le_resource = LabelEncoder()
    df["task_id"] = le_task.fit_transform(df["task_name"])
    df["resource_id"] = le_resource.fit_transform(df["resource"].astype(str))

    df["next_task"] = df.groupby("case_id")["task_id"].shift(-1)
    df.dropna(subset=["next_task"], inplace=True)
    df["next_task"] = df["next_task"].astype(int)
    return df, le_task, le_resource


def fit_feature_scaler(
    train_df: pd.DataFrame,
    use_norm_features: bool = True,
) -> Tuple[object, str]:
    """Fit a feature scaler on training rows only.

    Returns (fitted_estimator, mode) where mode is 'norm' or 'minmax'. The
    caller passes both back to `apply_feature_scaler`.
    """
    raw = train_df[list(FEATURE_COLS)].values
    if use_norm_features:
        est = Normalizer(norm="l2").fit(raw)
        return est, "norm"
    est = MinMaxScaler().fit(raw)
    return est, "minmax"


def apply_feature_scaler(
    df: pd.DataFrame,
    scaler: object,
) -> pd.DataFrame:
    """Add `feat_*` columns to `df` using a previously fitted scaler."""
    df = df.copy()
    raw = df[list(FEATURE_COLS)].values
    transformed = scaler.transform(raw)
    for i, out_col in enumerate(FEAT_OUT_COLS):
        df[out_col] = transformed[:, i]
    return df


def create_feature_representation(
    df: pd.DataFrame,
    use_norm_features: bool = True,
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Backwards-compatible one-shot: encode + fit + apply on the same df.

    New code should prefer `encode_categoricals` + a case-level split +
    `fit_feature_scaler(train)` + `apply_feature_scaler(both)` to avoid
    leaking val statistics into the scaler.
    """
    df, le_task, le_resource = encode_categoricals(df)
    scaler, _ = fit_feature_scaler(df, use_norm_features=use_norm_features)
    df = apply_feature_scaler(df, scaler)
    return df, le_task, le_resource


def split_cases(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (train_df, val_df) by sampling whole *case_ids*.

    Critical for any next-event prediction: events from one case must not
    appear in both halves, or future leaks into past during evaluation.
    """
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    case_ids = np.array(sorted(df["case_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(case_ids)
    n_val = max(1, int(round(len(case_ids) * val_frac)))
    val_cases = set(case_ids[:n_val].tolist())
    val_mask = df["case_id"].isin(val_cases)
    return df.loc[~val_mask].copy(), df.loc[val_mask].copy()


def build_graph_data(df: pd.DataFrame) -> List[Data]:
    """Convert a preprocessed event log into one PyG Data object per case.

    Each case becomes a graph with one node per event, edges between
    chronologically adjacent events (bidirectional), node features from
    `feat_*` columns, and per-node `next_task` labels.
    """
    feat_cols = list(FEAT_OUT_COLS)
    graphs: List[Data] = []
    for _cid, cdata in df.groupby("case_id", sort=False):
        cdata = cdata.sort_values("timestamp")
        x = torch.tensor(cdata[feat_cols].values, dtype=torch.float)
        n = len(cdata)
        if n > 1:
            src = list(range(n - 1))
            tgt = list(range(1, n))
            edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.tensor(cdata["next_task"].values, dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


def compute_class_weights(
    df: pd.DataFrame,
    num_classes: int,
    label_col: str = "next_task",
) -> torch.Tensor:
    """Sklearn-style balanced class weights for `label_col` over `num_classes`.

    Classes never observed in `df` get weight 1.0 (they cannot be in any
    loss term anyway, so the value is inert).
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = df[label_col].values
    weights = np.ones(num_classes, dtype=np.float32)
    present = np.unique(labels)
    if len(present) > 1:
        cw = compute_class_weight("balanced", classes=present, y=labels)
        for c, w in zip(present, cw):
            weights[c] = float(w)
    return torch.tensor(weights, dtype=torch.float32)
