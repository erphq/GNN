"""Pipeline stages, each callable in isolation.

The full `run` command threads them together. Standalone subcommands
(`analyze`, `cluster`) call only the subset they need. Functions take
plain arguments (no argparse Namespace) so they are reusable from
notebooks or future Rust orchestrators that shell into Python.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch_geometric.loader import DataLoader

from models.gat_model import (
    NextTaskGAT,
    bootstrap_ci,
    evaluate_gat_model,
    expected_calibration_error,
    fit_temperature,
    mean_reciprocal_rank,
    top_k_accuracy,
    train_gat_model,
)
from models.lstm_model import (
    NextActivityLSTM,
    evaluate_lstm_model,
    fit_temperature_lstm,
    make_padded_dataset,
    prepare_sequence_data,
    train_lstm_model,
)
from models.transformer_model import NextActivityTransformer
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
    analyze_bottleneck_drivers,
    analyze_bottlenecks,
    analyze_cycle_times,
    analyze_rare_transitions,
    analyze_transition_patterns,
    build_task_adjacency,
    perform_conformance_checking,
    render_bottleneck_drivers,
    spectral_cluster_graph,
)
from modules.rl_optimization import ProcessEnv, get_optimal_policy, run_q_learning
from visualization.process_viz import (
    create_sankey_diagram,
    plot_confusion_matrix,
    plot_cycle_time_distribution,
    plot_process_flow,
    plot_transition_heatmap,
)


@dataclass
class RunConfig:
    """Hyperparameters and skip flags for the full pipeline."""

    out_dir: str = "results"
    seed: int = 42
    device: str | None = None
    val_frac: float = 0.2
    batch_size_gat: int = 32
    batch_size_lstm: int = 64
    epochs_gat: int = 20
    epochs_lstm: int = 5
    lr_gat: float = 5e-4
    lr_lstm: float = 1e-3
    hidden_dim: int = 64
    gat_heads: int = 4
    gat_layers: int = 2
    rl_episodes: int = 30
    clusters: int = 3
    gat_node_level: bool = True
    gat_causal: bool = True
    gat_predict_time: bool = False
    time_loss_weight: float = 0.5
    calibrate: bool = True
    split_mode: str = "case"
    seq_arch: str = "lstm"  # "lstm" | "transformer"
    transformer_layers: int = 4
    transformer_heads: int = 4
    compile_models: bool = False
    use_resource: bool = False
    use_temporal: bool = False
    use_dt_input: bool = False
    # When non-empty, the LSTM time head emits one log-seconds prediction
    # per quantile and is trained with pinball loss. Only meaningful when
    # gat_predict_time (which the LSTM aliases for its own time head) is
    # also True. Default (empty) keeps the legacy single-value MSE head.
    time_quantiles: tuple[float, ...] = ()

    skip_gat: bool = False
    skip_lstm: bool = False
    skip_analyze: bool = False
    skip_viz: bool = False
    skip_cluster: bool = False
    skip_rl: bool = False

    rl_resources: tuple[int, ...] = field(default_factory=lambda: (0, 1))


def setup_results_dir(out_root: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_root if os.path.isabs(out_root) else os.path.join(os.getcwd(), out_root)
    run_dir = os.path.join(base, f"run_{timestamp}")
    for sub in ("models", "visualizations", "metrics", "analysis", "policies"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def save_metrics(metrics: dict, run_dir: str, filename: str) -> None:
    with open(os.path.join(run_dir, "metrics", filename), "w") as f:
        json.dump(metrics, f, indent=4, default=str)


def maybe_compile(model, *, enabled: bool, name: str = "model"):
    """Apply ``torch.compile`` when requested, with a graceful fallback.

    PyTorch 2's compiler (``torch.compile`` / ``dynamo``) can fail on
    PyG's ``GATConv`` and on ``pack_padded_sequence`` paths the LSTM
    uses. The fallback prints a warning and returns the original model
    so the run still succeeds — the cost is just losing the speedup,
    not the run.

    Performance characteristics (Apple Silicon, observed):
    - Transformer: 1.5-2x speedup, clean compile.
    - LSTM (packed): often falls back; pack_padded_sequence triggers
      a graph break that nullifies most of the gain.
    - GAT (PyG): often falls back; some GATConv paths use ``scatter_``
      operations that aren't supported.
    """
    if not enabled:
        return model
    try:
        return torch.compile(model, mode="reduce-overhead", dynamic=False)
    except Exception as e:  # noqa: BLE001 — torch.compile errors aren't a fixed type
        import warnings as _w
        _w.warn(
            f"torch.compile failed for {name}: {e}. "
            f"Falling back to eager mode — run continues but without speedup.",
            RuntimeWarning,
            stacklevel=2,
        )
        return model


def stage_preprocess(
    data_path: str, val_frac: float, seed: int, split_mode: str = "case"
):
    """Load CSV → encode → case-level split → fit+apply scaler.

    Returns (df, train_df, val_df, le_task, le_resource, scaler, mode).
    """
    df = load_and_preprocess_data(data_path)
    df, le_task, le_resource = encode_categoricals(df)
    train_df, val_df = split_cases(
        df, val_frac=val_frac, seed=seed, mode=split_mode
    )
    scaler, mode = fit_feature_scaler(train_df, use_norm_features=True)
    train_df = apply_feature_scaler(train_df, scaler)
    val_df = apply_feature_scaler(val_df, scaler)
    return df, train_df, val_df, le_task, le_resource, scaler, mode


def stage_train_gat(
    train_df, val_df, le_task, cfg: RunConfig, device, run_dir: str,
    baseline: dict | None = None,
):
    train_loader = DataLoader(
        build_graph_data(train_df, causal=cfg.gat_causal),
        batch_size=cfg.batch_size_gat, shuffle=True,
    )
    val_loader = DataLoader(
        build_graph_data(val_df, causal=cfg.gat_causal),
        batch_size=cfg.batch_size_gat, shuffle=False,
    )
    num_classes = len(le_task.classes_)
    class_weights = compute_class_weights(train_df, num_classes).to(device)

    model = NextTaskGAT(
        input_dim=5,
        hidden_dim=cfg.hidden_dim,
        output_dim=num_classes,
        num_layers=cfg.gat_layers,
        heads=cfg.gat_heads,
        dropout=0.5,
        node_level=cfg.gat_node_level,
        predict_time=cfg.gat_predict_time,
    ).to(device)
    model = maybe_compile(model, enabled=cfg.compile_models, name="GAT")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr_gat, weight_decay=5e-4
    )
    model_path = os.path.join(run_dir, "models", "best_gnn_model.pth")
    model = train_gat_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=cfg.epochs_gat, model_path=model_path,
        time_loss_weight=cfg.time_loss_weight,
    )
    # Uncalibrated pass — for accuracy / MCC and the pre-cal ECE baseline.
    y_true, y_pred, y_prob = evaluate_gat_model(model, val_loader, device)
    plot_confusion_matrix(
        y_true, y_pred, le_task.classes_,
        os.path.join(run_dir, "visualizations", "gat_confusion_matrix.png"),
    )
    from sklearn.metrics import accuracy_score, matthews_corrcoef

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top_3_accuracy": top_k_accuracy(y_true, y_prob, 3),
        "top_5_accuracy": top_k_accuracy(y_true, y_prob, 5),
        "mrr": mean_reciprocal_rank(y_true, y_prob),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    add_bootstrap_cis(metrics, y_true, y_pred, y_prob)
    metrics.update(per_class_metrics(y_true, y_pred, le_task.classes_))
    if baseline is not None:
        # Lift over Markov is what tells you if the GAT learned anything
        # beyond the trivial 1-st order baseline. A near-zero lift means
        # the gain is class-imbalance, not process structure.
        metrics["lift_over_markov"] = float(
            metrics["accuracy"] - baseline["markov_accuracy"]
        )
        metrics["lift_over_most_common"] = float(
            metrics["accuracy"] - baseline["most_common_accuracy"]
        )
    if cfg.gat_predict_time:
        metrics["dt_mae_hours"] = float(getattr(model, "last_dt_mae_hours", float("nan")))
        metrics["time_loss_weight"] = float(cfg.time_loss_weight)

    if cfg.calibrate:
        ece_before = expected_calibration_error(y_true, y_prob)
        T = fit_temperature(model, val_loader, device)
        # Re-evaluate with the calibrated temperature so reported probs
        # are honest. Accuracy / MCC are unchanged (T does not change
        # argmax) but ECE typically drops sharply.
        _, _, y_prob_cal = evaluate_gat_model(model, val_loader, device, temperature=T)
        ece_after = expected_calibration_error(y_true, y_prob_cal)
        metrics["temperature"] = float(T)
        metrics["ece_before_calibration"] = float(ece_before)
        metrics["ece_after_calibration"] = float(ece_after)
        # Persist T alongside the model so inference can reload both.
        torch.save(
            {"temperature": float(T)},
            os.path.join(run_dir, "models", "gat_calibration.pt"),
        )

    save_metrics(metrics, run_dir, "gat_metrics.json")


def stage_train_lstm(
    df, train_df, val_df, num_classes, cfg: RunConfig, device, run_dir: str,
    baseline: dict | None = None,
    le_task=None,
):
    # The Rust hot path doesn't carry dt / resource / continuous-feature
    # tensors, so it's only safe when none of the rich-feature flags are
    # set. Fall back to the Python prefix builder when any is on; it
    # threads everything through as tensor attributes.
    use_rust_path = (
        not cfg.gat_predict_time
        and not cfg.use_resource
        and not cfg.use_temporal
        and not cfg.use_dt_input
    )
    dt_train = dt_val = None

    # Build the list of continuous-feature columns to thread through.
    # Step 2: cyclic encoding of day-of-week / hour-of-day. Step 3
    # (dt-since-prev as input feature) requires a new column in
    # encode_categoricals — tracked separately.
    cont_cols: list[str] = []
    if cfg.use_temporal:
        cont_cols += ["dow_sin", "dow_cos", "hod_sin", "hod_cos"]

    if use_rust_path:
        from modules._fast import build_padded_prefixes_fast
        if build_padded_prefixes_fast is not None:
            Xt, lt, yt, _ = build_padded_prefixes_fast(train_df)
            Xv, lv, yv, _ = build_padded_prefixes_fast(val_df)
            X_train_pad = torch.from_numpy(Xt)
            X_train_len = torch.from_numpy(lt)
            y_train = torch.from_numpy(yt)
            X_val_pad = torch.from_numpy(Xv)
            X_val_len = torch.from_numpy(lv)
            y_val = torch.from_numpy(yv)
        else:
            use_rust_path = False  # extension not built; fall through

    if not use_rust_path:
        train_seq, val_seq = prepare_sequence_data(
            df, train_df=train_df, val_df=val_df, seed=cfg.seed,
            continuous_features=cont_cols or None,
        )
        X_train_pad, X_train_len, y_train, _ = make_padded_dataset(train_seq, num_classes)
        X_val_pad, X_val_len, y_val, _ = make_padded_dataset(val_seq, num_classes)
        # The prefix builder attaches dt_targets and resource_pad as
        # tensor attributes when those features are present in the df.
        if cfg.gat_predict_time:
            dt_train = getattr(X_train_pad, "dt_targets", None)
            dt_val = getattr(X_val_pad, "dt_targets", None)

    # Type-annotate as ``torch.nn.Module`` so mypy doesn't widen to the
    # first branch's concrete class — the two architectures share the
    # forward(x, seq_len) interface this stage relies on.
    model: torch.nn.Module
    if cfg.seq_arch == "transformer":
        model = NextActivityTransformer(
            num_classes,
            emb_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.transformer_layers,
            num_heads=cfg.transformer_heads,
            predict_time=cfg.gat_predict_time,
            # Pad to the longest possible prefix in this run.
            max_len=int(X_train_pad.shape[1]) + 8,
        ).to(device)
    else:
        # Resource vocabulary size: encoder produces 0..N-1 across the
        # full df, so we read off the train half's max + 1 (sufficient
        # because the encoder was fit on the full df → val resources
        # share the same id space).
        num_resources = (
            int(train_df["resource_id"].max()) + 1
            if cfg.use_resource and "resource_id" in train_df.columns
            else None
        )
        model = NextActivityLSTM(
            num_classes, emb_dim=cfg.hidden_dim, hidden_dim=cfg.hidden_dim, num_layers=1,
            predict_time=cfg.gat_predict_time,
            num_resources=num_resources,
            n_continuous_dims=len(cont_cols),
            time_quantiles=tuple(cfg.time_quantiles) or None,
        ).to(device)
    model = maybe_compile(
        model, enabled=cfg.compile_models, name=f"seq:{cfg.seq_arch}",
    )
    model_path = os.path.join(
        run_dir, "models",
        f"{'transformer' if cfg.seq_arch == 'transformer' else 'lstm'}_next_activity.pth",
    )
    model = train_lstm_model(
        model, X_train_pad, X_train_len, y_train, device,
        batch_size=cfg.batch_size_lstm, epochs=cfg.epochs_lstm,
        model_path=model_path,
        dt_targets=dt_train,
        time_loss_weight=cfg.time_loss_weight,
    )

    # Save architecture metadata alongside the .pth so downstream
    # consumers (gnn export, gnn predict-suffix, gnn serve) can
    # reconstruct the class without the user re-supplying every flag.
    arch_meta = {
        "seq_arch": cfg.seq_arch,
        "num_classes": int(num_classes),
        "emb_dim": int(cfg.hidden_dim),
        "hidden_dim": int(cfg.hidden_dim),
        "num_layers": (
            int(cfg.transformer_layers) if cfg.seq_arch == "transformer" else 1
        ),
        "predict_time": bool(cfg.gat_predict_time),
        "num_resources": (int(num_resources) if num_resources else None),
        "n_continuous_dims": int(len(cont_cols)),
        "continuous_features": list(cont_cols),
        "time_quantiles": list(cfg.time_quantiles),
        "transformer_heads": int(cfg.transformer_heads),
        "max_seq_len": int(X_train_pad.shape[1]),
    }
    arch_path = os.path.join(
        run_dir, "models",
        f"{'transformer' if cfg.seq_arch == 'transformer' else 'lstm'}_arch.json",
    )
    with open(arch_path, "w") as f:
        json.dump(arch_meta, f, indent=2)

    # Uncalibrated pass — accuracy + pre-cal ECE.
    preds, probs = evaluate_lstm_model(
        model, X_val_pad, X_val_len, cfg.batch_size_lstm, device,
        dt_targets=dt_val,
    )
    from sklearn.metrics import accuracy_score

    y_true_lstm = y_val.numpy()
    # The LSTM evaluator returns numpy probs; convert to torch for the
    # ranking metrics so they share an implementation with the GAT.
    import torch as _torch  # local alias to avoid the pyright complaint
    y_true_t = _torch.from_numpy(y_true_lstm).long()
    y_prob_t = _torch.from_numpy(probs)
    lstm_metrics = {
        "model_arch": cfg.seq_arch,
        "accuracy": float(accuracy_score(y_true_lstm, preds)),
        "top_3_accuracy": top_k_accuracy(y_true_t, y_prob_t, 3),
        "top_5_accuracy": top_k_accuracy(y_true_t, y_prob_t, 5),
        "mrr": mean_reciprocal_rank(y_true_t, y_prob_t),
    }
    add_bootstrap_cis(lstm_metrics, y_true_lstm, preds, probs)
    class_names = (
        list(le_task.classes_) if le_task is not None
        else [str(i) for i in range(num_classes)]
    )
    lstm_metrics.update(per_class_metrics(y_true_lstm, preds, class_names))
    if baseline is not None:
        lstm_metrics["lift_over_markov"] = float(
            lstm_metrics["accuracy"] - baseline["markov_accuracy"]
        )
        lstm_metrics["lift_over_most_common"] = float(
            lstm_metrics["accuracy"] - baseline["most_common_accuracy"]
        )
    if cfg.gat_predict_time:
        lstm_metrics["dt_mae_hours"] = float(
            getattr(model, "last_dt_mae_hours", float("nan"))
        )
        lstm_metrics["time_loss_weight"] = float(cfg.time_loss_weight)
        if cfg.time_quantiles:
            lstm_metrics["time_quantiles"] = list(cfg.time_quantiles)
            lstm_metrics["dt_coverage"] = float(
                getattr(model, "last_dt_coverage", float("nan"))
            )
            lstm_metrics["dt_interval_width_hours"] = float(
                getattr(model, "last_dt_interval_width_hours", float("nan"))
            )

    if cfg.calibrate:
        ece_before = expected_calibration_error(
            y_val.long(), torch.from_numpy(probs)
        )
        T = fit_temperature_lstm(
            model, X_val_pad, X_val_len, y_val, cfg.batch_size_lstm, device
        )
        _, probs_cal = evaluate_lstm_model(
            model, X_val_pad, X_val_len, cfg.batch_size_lstm, device,
            temperature=T, dt_targets=dt_val,
        )
        ece_after = expected_calibration_error(
            y_val.long(), torch.from_numpy(probs_cal)
        )
        lstm_metrics["temperature"] = float(T)
        lstm_metrics["ece_before_calibration"] = float(ece_before)
        lstm_metrics["ece_after_calibration"] = float(ece_after)
        torch.save(
            {"temperature": float(T)},
            os.path.join(run_dir, "models", "lstm_calibration.pt"),
        )

    save_metrics(lstm_metrics, run_dir, "lstm_metrics.json")


def add_bootstrap_cis(
    metrics: dict, y_true, y_pred, y_prob,
    n_resamples: int = 500,
):
    """Decorate a metrics dict in place with 95% bootstrap CIs.

    Adds ``<metric>_ci_low`` and ``<metric>_ci_high`` for accuracy,
    macro_f1, top_3_accuracy, top_5_accuracy, and mrr. Kept fast
    (n_resamples=500) so it adds <1s on real-sized val sets — bumping
    to 1000+ is fine for paper-grade reporting but rarely changes the
    headline.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    yt_np = np.asarray(y_true)
    yp_np = np.asarray(y_pred)
    yprob_np = np.asarray(y_prob)
    yt_t = torch.from_numpy(yt_np).long() if not isinstance(y_true, torch.Tensor) else y_true.long()
    yprob_t = torch.from_numpy(yprob_np) if not isinstance(y_prob, torch.Tensor) else y_prob

    # Pin the label set so macro_f1 averages over the same N classes on
    # every resample (otherwise sklearn averages only over classes
    # present in the resample, biasing CI upward when classes are rare).
    num_classes = int(max(yprob_t.shape[1] if yprob_t.ndim == 2 else 0,
                          int(yt_np.max()) + 1, int(yp_np.max()) + 1))
    label_set = list(range(num_classes))

    def _ci(name, fn, score):
        lo, hi = bootstrap_ci(*score, fn, n_resamples=n_resamples)
        metrics[f"{name}_ci_low"] = lo
        metrics[f"{name}_ci_high"] = hi

    _ci("accuracy", lambda y, p: accuracy_score(y, p), (yt_np, yp_np))
    _ci(
        "macro_f1",
        lambda y, p: f1_score(y, p, average="macro", zero_division=0, labels=label_set),
        (yt_np, yp_np),
    )
    _ci("top_3_accuracy", lambda y, p: top_k_accuracy(y, p, 3), (yt_t, yprob_t))
    _ci("top_5_accuracy", lambda y, p: top_k_accuracy(y, p, 5), (yt_t, yprob_t))
    _ci("mrr", mean_reciprocal_rank, (yt_t, yprob_t))


def per_class_metrics(y_true, y_pred, class_names) -> dict:
    """Per-class precision, recall, F1, support + macro / weighted F1.

    Macro F1 averages equally across classes — surfaces rare-class
    performance. Weighted F1 averages by support — closer to overall
    accuracy. Reporting both is the honest move on imbalanced logs.
    """
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    macro_f1 = float(f.mean())
    total = float(s.sum())
    weighted_f1 = float((f * s).sum() / total) if total > 0 else 0.0
    per_class = {
        str(class_names[i]): {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(len(class_names))
    }
    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def stage_baselines(train_df, val_df, run_dir: str) -> dict:
    """Fit + score the null and Markov baselines on the val set.

    Run early in the pipeline so later stages can compute lift over
    Markov when reporting their own accuracy. Returns the metrics dict
    so the caller doesn't have to re-read the JSON.
    """
    from models.baselines import evaluate_baselines

    metrics = evaluate_baselines(train_df, val_df)
    save_metrics(metrics, run_dir, "baseline_metrics.json")
    return metrics


def stage_analyze(df, run_dir: str, le_task=None):
    bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
    case_merged, long_cases, cut95 = analyze_cycle_times(df)
    rare_trans = analyze_rare_transitions(bottleneck_stats)
    replayed, conformance = perform_conformance_checking(df)

    # Root-cause analysis: for the top-5 slowest transitions, which case
    # attributes drive the wait? Surfaces "this stalls when assigned to
    # alice" or "high-amount cases stall here" — actionable signal that
    # the mean-wait ranking alone hides.
    drivers = analyze_bottleneck_drivers(df, le_task=le_task, top_n=5)

    save_metrics(
        {
            "num_long_cases": int(len(long_cases)),
            "cycle_time_95th_percentile_h": float(cut95),
            "num_rare_transitions": int(len(rare_trans)),
            "num_deviant_traces": int(conformance["num_deviant"]),
            "total_traces": int(len(replayed)),
            "conformance_fitness": float(conformance["fitness"]),
            "conformance_precision": float(conformance["precision"]),
            "conformance_f_score": float(conformance["f_score"]),
            "bottleneck_drivers": drivers,
        },
        run_dir, "process_analysis.json",
    )
    # Also write a human-readable markdown report to analysis/.
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    with open(os.path.join(analysis_dir, "bottleneck_drivers.md"), "w") as f:
        f.write(render_bottleneck_drivers(drivers))
    return bottleneck_stats, significant_bottlenecks, case_merged


def stage_viz(df, le_task, bottleneck_stats, significant_bottlenecks, case_merged, run_dir: str):
    viz_dir = os.path.join(run_dir, "visualizations")
    plot_cycle_time_distribution(
        case_merged["duration_h"].values,
        os.path.join(viz_dir, "cycle_time_distribution.png"),
    )
    plot_process_flow(
        bottleneck_stats, le_task, significant_bottlenecks.head(),
        os.path.join(viz_dir, "process_flow_bottlenecks.png"),
    )
    transitions, _, _ = analyze_transition_patterns(df)
    plot_transition_heatmap(
        transitions, le_task,
        os.path.join(viz_dir, "transition_probability_heatmap.png"),
    )
    create_sankey_diagram(
        transitions, le_task,
        os.path.join(viz_dir, "process_flow_sankey.html"),
    )


def stage_cluster(df, le_task, num_classes: int, k: int, seed: int, run_dir: str):
    adj = build_task_adjacency(df, num_classes)
    labels = spectral_cluster_graph(adj, k=k, normalized=True, random_state=seed)
    save_metrics(
        {
            "task_clusters": {
                le_task.inverse_transform([t_id])[0]: int(lbl)
                for t_id, lbl in enumerate(labels)
            }
        },
        run_dir, "clustering_results.json",
    )


def stage_rl(df, le_task, episodes: int, resources, run_dir: str):
    env = ProcessEnv(df, le_task, list(resources))
    q_table = run_q_learning(env, episodes=episodes)
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
        run_dir, "rl_results.json",
    )


def run_full_pipeline(data_path: str, cfg: RunConfig) -> str:
    """End-to-end pipeline. Returns the run directory."""
    from modules.utils import pick_device, set_seed

    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    print(f"Using device: {device}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    run_dir = setup_results_dir(cfg.out_dir)
    print(f"Results will be saved in: {run_dir}")

    print("\n[1/9] Preprocess")
    df, train_df, val_df, le_task, le_resource, _, mode = stage_preprocess(
        data_path, cfg.val_frac, cfg.seed, split_mode=cfg.split_mode
    )
    save_metrics(
        {
            "num_tasks": int(len(le_task.classes_)),
            "num_resources": int(len(le_resource.classes_)),
            "num_cases_total": int(df["case_id"].nunique()),
            "num_cases_train": int(train_df["case_id"].nunique()),
            "num_cases_val": int(val_df["case_id"].nunique()),
            "scaler_mode": mode,
            "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
            "seed": cfg.seed,
        },
        run_dir, "preprocessing_info.json",
    )
    num_classes = len(le_task.classes_)

    # Baselines first so the GAT/LSTM stages can record lift over them.
    print("\n[1.5/9] Baselines (null + Markov)")
    baseline = stage_baselines(train_df, val_df, run_dir)
    print(
        f"  most-common acc={baseline['most_common_accuracy']:.4f}, "
        f"markov acc={baseline['markov_accuracy']:.4f} "
        f"(coverage={baseline['markov_coverage']:.2f})"
    )

    if not cfg.skip_gat:
        print("\n[2/9] Train + eval GAT")
        stage_train_gat(train_df, val_df, le_task, cfg, device, run_dir, baseline=baseline)
    else:
        print("\n[2/9] Train + eval GAT — skipped")

    if not cfg.skip_lstm:
        print("\n[3/9] Train LSTM")
        stage_train_lstm(
            df, train_df, val_df, num_classes, cfg, device, run_dir,
            baseline=baseline, le_task=le_task,
        )
    else:
        print("\n[3/9] Train LSTM — skipped")

    bottleneck_stats = significant_bottlenecks = case_merged = None
    if not cfg.skip_analyze:
        print("\n[4/9] Process-mining analysis")
        bottleneck_stats, significant_bottlenecks, case_merged = stage_analyze(
            df, run_dir, le_task=le_task,
        )
    else:
        print("\n[4/9] Process-mining analysis — skipped")

    if not cfg.skip_viz:
        if bottleneck_stats is None:
            print("\n[5/9] Visualizations — skipped (analyze did not run)")
        else:
            print("\n[5/9] Visualizations")
            stage_viz(df, le_task, bottleneck_stats, significant_bottlenecks, case_merged, run_dir)
    else:
        print("\n[5/9] Visualizations — skipped")

    if not cfg.skip_cluster:
        print("\n[6/9] Spectral clustering")
        stage_cluster(df, le_task, num_classes, cfg.clusters, cfg.seed, run_dir)
    else:
        print("\n[6/9] Spectral clustering — skipped")

    if not cfg.skip_rl:
        print("\n[7/9] Tabular RL")
        stage_rl(df, le_task, cfg.rl_episodes, cfg.rl_resources, run_dir)
    else:
        print("\n[7/9] Tabular RL — skipped")

    print(f"\nDone. Results saved in {run_dir}")
    return run_dir
