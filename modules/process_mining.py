#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Analysis Module
Includes bottleneck analysis, conformance checking, and cycle time analysis
"""

import pandas as pd
import numpy as np

# pm4py is imported lazily inside `perform_conformance_checking` because it's
# a heavy optional dependency (~1 GB with transitive deps). Importing it at
# module-load time forced every consumer — including unit tests — to install
# the full PM stack just to use `analyze_bottlenecks` or `spectral_cluster`.

def analyze_bottlenecks(df, freq_threshold=5):
    """
    Analyze process bottlenecks based on waiting times between activities
    """
    df = df.copy()
    df["next_task_id"] = df.groupby("case_id")["task_id"].shift(-1)
    df["next_timestamp"] = df.groupby("case_id")["timestamp"].shift(-1)
    transitions = df.dropna(subset=["next_task_id"]).copy()
    transitions["wait_sec"] = (transitions["next_timestamp"] - transitions["timestamp"]).dt.total_seconds()
    
    bottleneck_stats = transitions.groupby(["task_id","next_task_id"])["wait_sec"].agg([
        "mean","count"
    ]).reset_index()
    
    bottleneck_stats["mean_hours"] = bottleneck_stats["mean"]/3600.0
    bottleneck_stats.sort_values("mean_hours", ascending=False, inplace=True)
    
    # Filter by frequency threshold
    significant_bottlenecks = bottleneck_stats[bottleneck_stats["count"] >= freq_threshold]
    
    return bottleneck_stats, significant_bottlenecks

def analyze_cycle_times(df):
    """
    Analyze process cycle times
    """
    case_grouped = df.groupby("case_id")["timestamp"].agg(["min","max"])
    case_grouped["cycle_time_hours"] = (
        case_grouped["max"] - case_grouped["min"]
    ).dt.total_seconds()/3600.0
    case_grouped.reset_index(inplace=True)
    
    df_feats = df.groupby("case_id").agg({
        "amount": "mean",
        "task_id": "count"
    }).rename(columns={
        "amount": "mean_amount",
        "task_id": "num_events"
    }).reset_index()
    
    case_merged = pd.merge(case_grouped, df_feats, on="case_id", how="left")
    case_merged["duration_h"] = case_merged["cycle_time_hours"]
    
    # Identify long-running cases (95th percentile)
    cut95 = case_merged["duration_h"].quantile(0.95)
    long_cases = case_merged[case_merged["duration_h"] > cut95]
    
    return case_merged, long_cases, cut95

def analyze_rare_transitions(bottleneck_stats, rare_threshold=2):
    """
    Identify rare transitions in the process
    """
    rare_trans = bottleneck_stats[bottleneck_stats["count"] <= rare_threshold]
    return rare_trans

def perform_conformance_checking(df):
    """Inductive miner + token replay conformance check (requires pm4py)."""
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.objects.conversion.process_tree import converter as pt_converter
    from pm4py.objects.log.util import dataframe_utils

    df_pm = df[["case_id", "task_name", "timestamp"]].rename(
        columns={
            "case_id": "case:concept:name",
            "task_name": "concept:name",
            "timestamp": "time:timestamp",
        }
    )
    df_pm = dataframe_utils.convert_timestamp_columns_in_df(df_pm)
    event_log = log_converter.apply(df_pm)

    process_tree = inductive_miner.apply(event_log)
    net, im, fm = pt_converter.apply(process_tree)

    replayed = token_replay.apply(event_log, net, im, fm)
    n_deviant = sum(1 for t in replayed if not t["trace_is_fit"])
    return replayed, n_deviant

def analyze_transition_patterns(df):
    """
    Analyze transition patterns and compute transition matrix
    """
    transitions = df.copy()
    transitions["next_task_id"] = transitions.groupby("case_id")["task_id"].shift(-1)
    trans_count = transitions.groupby(["task_id","next_task_id"]).size().unstack(fill_value=0)
    prob_matrix = trans_count.div(trans_count.sum(axis=1), axis=0)
    
    return transitions, trans_count, prob_matrix

def spectral_cluster_graph(adj_matrix, k=2, normalized=True, random_state=42):
    """
    Spectral clustering on a (possibly directed) process-task graph.

    Symmetrizes the adjacency, then uses the *normalized* Laplacian
    L_sym = I - D^{-1/2} A D^{-1/2} (Shi-Malik / Ng-Jordan-Weiss). Falls
    back to the unnormalized Laplacian when `normalized=False`. Eigen-
    decomposition uses `numpy.linalg.eigh` since the Laplacian is real
    and symmetric — `eig` returns complex values in floating point and
    is meaningfully slower on this shape.
    """
    from sklearn.cluster import KMeans

    A = np.asarray(adj_matrix, dtype=np.float64)
    A = 0.5 * (A + A.T)  # symmetrize (process graphs are inherently directed)
    degrees = A.sum(axis=1)

    if normalized:
        # Avoid division-by-zero on isolated nodes.
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = np.diag(degrees) - A

    # eigh: real, ascending eigenvalues, orthonormal eigenvectors.
    eigenvals, eigenvecs = np.linalg.eigh(L)

    if k == 2:
        # Fiedler vector = eigenvector of the 2nd-smallest eigenvalue.
        fiedler = eigenvecs[:, 1]
        return (fiedler >= 0).astype(int)

    # Multi-cluster: rows of the bottom-k non-trivial eigenvectors → k-means.
    embedding = eigenvecs[:, 1:k]
    return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(embedding).labels_

def build_task_adjacency(df, num_tasks):
    """Build adjacency matrix weighted by transition frequencies.

    Prefers the Rust kernel when `pm_fast` is installed; falls back to
    a pure-Python groupby loop otherwise. Both implementations require
    `df` to be sorted by (case_id, timestamp), which matches the output
    of `load_and_preprocess_data`.
    """
    from modules._fast import build_task_adjacency_fast

    if build_task_adjacency_fast is not None:
        return build_task_adjacency_fast(df, num_tasks)

    A = np.zeros((num_tasks, num_tasks), dtype=np.float32)
    for _cid, cdata in df.groupby("case_id"):
        cdata = cdata.sort_values("timestamp")
        tasks_seq = cdata["task_id"].values
        for i in range(len(tasks_seq) - 1):
            A[tasks_seq[i], tasks_seq[i + 1]] += 1.0
    return A 