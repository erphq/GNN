#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end pipeline: preprocess → train GAT + LSTM → process-mine →
visualize → spectral-cluster → tabular RL. Hyperparameters are exposed
via argparse; sensible defaults reproduce the original paper's setup.

Usage:
    python main.py <input.csv> [--epochs-gat 20] [--epochs-lstm 5] ...
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
from torch_geometric.loader import DataLoader

from modules.data_preprocessing import (
    apply_feature_scaler,
    build_graph_data,
    compute_class_weights,
    encode_categoricals,
    fit_feature_scaler,
    load_and_preprocess_data,
    split_cases,
)
from modules.process_mining import (
    analyze_bottlenecks,
    analyze_cycle_times,
    analyze_rare_transitions,
    analyze_transition_patterns,
    build_task_adjacency,
    perform_conformance_checking,
    spectral_cluster_graph,
)
from modules.rl_optimization import ProcessEnv, get_optimal_policy, run_q_learning
from modules.utils import pick_device, set_seed
from models.gat_model import NextTaskGAT, evaluate_gat_model, train_gat_model
from models.lstm_model import (
    NextActivityLSTM,
    evaluate_lstm_model,
    make_padded_dataset,
    prepare_sequence_data,
    train_lstm_model,
)
from visualization.process_viz import (
    create_sankey_diagram,
    plot_confusion_matrix,
    plot_cycle_time_distribution,
    plot_process_flow,
    plot_transition_heatmap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process mining with GNN + LSTM + RL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("data_path", help="Path to event-log CSV.")
    p.add_argument("--out-dir", default="results", help="Root directory for run output.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="Override device (cpu / cuda / mps).")
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of cases held out for validation.")
    p.add_argument("--batch-size-gat", type=int, default=32)
    p.add_argument("--batch-size-lstm", type=int, default=64)
    p.add_argument("--epochs-gat", type=int, default=20)
    p.add_argument("--epochs-lstm", type=int, default=5)
    p.add_argument("--lr-gat", type=float, default=5e-4)
    p.add_argument("--lr-lstm", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--gat-heads", type=int, default=4)
    p.add_argument("--gat-layers", type=int, default=2)
    p.add_argument("--rl-episodes", type=int, default=30)
    p.add_argument("--clusters", type=int, default=3)
    return p.parse_args()


def setup_results_dir(out_root: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, out_root) if not os.path.isabs(out_root) else out_root
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    for sub in ("models", "visualizations", "metrics", "analysis", "policies"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def save_metrics(metrics_dict: dict, run_dir: str, filename: str) -> None:
    with open(os.path.join(run_dir, "metrics", filename), "w") as f:
        json.dump(metrics_dict, f, indent=4, default=str)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"Using device: {device}")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")

    run_dir = setup_results_dir(args.out_dir)
    print(f"Results will be saved in: {run_dir}")

    # 1. Load + encode (encoders fit on full df — we need a stable label space).
    print("\n1. Loading and preprocessing data...")
    df = load_and_preprocess_data(args.data_path)
    df, le_task, le_resource = encode_categoricals(df)

    # 2. Case-level split (no future-event leakage across train/val).
    train_df, val_df = split_cases(df, val_frac=args.val_frac, seed=args.seed)

    # 3. Fit scaler on train only, transform both halves with that fitted scaler.
    scaler, mode = fit_feature_scaler(train_df, use_norm_features=True)
    train_df = apply_feature_scaler(train_df, scaler)
    val_df = apply_feature_scaler(val_df, scaler)

    save_metrics(
        {
            "num_tasks": int(len(le_task.classes_)),
            "num_resources": int(len(le_resource.classes_)),
            "num_cases_total": int(df["case_id"].nunique()),
            "num_cases_train": int(train_df["case_id"].nunique()),
            "num_cases_val": int(val_df["case_id"].nunique()),
            "scaler_mode": mode,
            "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
            "seed": args.seed,
        },
        run_dir,
        "preprocessing_info.json",
    )

    # 4. Build graphs and data loaders.
    print("\n2. Building graph data...")
    train_graphs = build_graph_data(train_df)
    val_graphs = build_graph_data(val_df)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size_gat, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size_gat, shuffle=False)

    # 5. Train GAT.
    print("\n3. Training GAT model...")
    num_classes = len(le_task.classes_)
    class_weights = compute_class_weights(train_df, num_classes).to(device)

    gat_model = NextTaskGAT(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_layers=args.gat_layers,
        heads=args.gat_heads,
        dropout=0.5,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        gat_model.parameters(), lr=args.lr_gat, weight_decay=5e-4
    )
    gat_model_path = os.path.join(run_dir, "models", "best_gnn_model.pth")
    gat_model = train_gat_model(
        gat_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=args.epochs_gat,
        model_path=gat_model_path,
    )

    # 6. Evaluate GAT.
    print("\n4. Evaluating GAT model...")
    y_true, y_pred, _y_prob = evaluate_gat_model(gat_model, val_loader, device)
    plot_confusion_matrix(
        y_true,
        y_pred,
        le_task.classes_,
        os.path.join(run_dir, "visualizations", "gat_confusion_matrix.png"),
    )
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    save_metrics(
        {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        },
        run_dir,
        "gat_metrics.json",
    )

    # 7. Train + evaluate LSTM (case-level split passed in explicitly).
    print("\n5. Training LSTM model...")
    train_seq, val_seq = prepare_sequence_data(
        df, train_df=train_df, val_df=val_df, seed=args.seed
    )
    X_train_pad, X_train_len, y_train_lstm, _ = make_padded_dataset(
        train_seq, num_classes
    )
    X_val_pad, X_val_len, y_val_lstm, _ = make_padded_dataset(val_seq, num_classes)

    lstm_model = NextActivityLSTM(
        num_classes, emb_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_layers=1
    ).to(device)
    lstm_model_path = os.path.join(run_dir, "models", "lstm_next_activity.pth")
    lstm_model = train_lstm_model(
        lstm_model,
        X_train_pad,
        X_train_len,
        y_train_lstm,
        device,
        batch_size=args.batch_size_lstm,
        epochs=args.epochs_lstm,
        model_path=lstm_model_path,
    )
    lstm_preds, _ = evaluate_lstm_model(
        lstm_model, X_val_pad, X_val_len, args.batch_size_lstm, device
    )
    save_metrics(
        {
            "accuracy": float(accuracy_score(y_val_lstm.numpy(), lstm_preds)),
        },
        run_dir,
        "lstm_metrics.json",
    )

    # 8. Process mining analysis (uses full df — descriptive, not predictive).
    print("\n6. Performing process mining analysis...")
    bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
    case_merged, long_cases, cut95 = analyze_cycle_times(df)
    rare_trans = analyze_rare_transitions(bottleneck_stats)
    replayed, n_deviant = perform_conformance_checking(df)
    save_metrics(
        {
            "num_long_cases": int(len(long_cases)),
            "cycle_time_95th_percentile_h": float(cut95),
            "num_rare_transitions": int(len(rare_trans)),
            "num_deviant_traces": int(n_deviant),
            "total_traces": int(len(replayed)),
        },
        run_dir,
        "process_analysis.json",
    )

    # 9. Visualizations.
    print("\n7. Creating visualizations...")
    viz_dir = os.path.join(run_dir, "visualizations")
    plot_cycle_time_distribution(
        case_merged["duration_h"].values,
        os.path.join(viz_dir, "cycle_time_distribution.png"),
    )
    plot_process_flow(
        bottleneck_stats,
        le_task,
        significant_bottlenecks.head(),
        os.path.join(viz_dir, "process_flow_bottlenecks.png"),
    )
    transitions, _trans_count, _prob_matrix = analyze_transition_patterns(df)
    plot_transition_heatmap(
        transitions,
        le_task,
        os.path.join(viz_dir, "transition_probability_heatmap.png"),
    )
    create_sankey_diagram(
        transitions, le_task, os.path.join(viz_dir, "process_flow_sankey.html")
    )

    # 10. Spectral clustering.
    print("\n8. Performing spectral clustering...")
    adj_matrix = build_task_adjacency(df, num_classes)
    cluster_labels = spectral_cluster_graph(
        adj_matrix, k=args.clusters, normalized=True, random_state=args.seed
    )
    save_metrics(
        {
            "task_clusters": {
                le_task.inverse_transform([t_id])[0]: int(lbl)
                for t_id, lbl in enumerate(cluster_labels)
            }
        },
        run_dir,
        "clustering_results.json",
    )

    # 11. Tabular RL.
    print("\n9. Training RL agent...")
    dummy_resources = [0, 1]
    env = ProcessEnv(df, le_task, dummy_resources)
    q_table = run_q_learning(env, episodes=args.rl_episodes)
    all_actions = [(t, r) for t in env.all_tasks for r in env.resources]
    policy = get_optimal_policy(q_table, all_actions)
    save_metrics(
        {
            "num_states": len(policy),
            "num_actions": len(all_actions),
            "policy": {
                str(state): {"task": int(a[0]), "resource": int(a[1])}
                for state, a in policy.items()
            },
        },
        run_dir,
        "rl_results.json",
    )

    print(f"\nDone! Results saved in {run_dir}")


if __name__ == "__main__":
    main()
