"""Argparse-based CLI for the GNN process-mining pipeline.

Subcommands: run, analyze, cluster, smoke, version.

Design notes:
- Kebab-case flags throughout (`--num-epochs`), so a future Rust rewrite
  using clap can keep the same surface.
- Exit codes follow the conventional CLI scheme: 0 = ok, 2 = bad usage,
  3 = data error (missing file, bad columns), 4 = model/runtime error.
- Each subcommand maps to one or two functions in `gnn_cli.stages` so
  the orchestration is easy to port.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from collections.abc import Sequence

from gnn_cli import __version__

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DATA = 3
EXIT_RUNTIME = 4


def _add_run_flags(p: argparse.ArgumentParser) -> None:
    """Hyperparameter flags shared by `run` and `smoke`."""
    p.add_argument(
        "--config", default=None,
        help="Path to a TOML file with run defaults. Keys can be either at "
             "the top level or under a [run] table; flag names use snake_case "
             "(e.g. epochs_gat = 30). CLI flags override TOML; TOML overrides "
             "hardcoded defaults.",
    )
    p.add_argument("--out-dir", default="results", help="Root directory for run output.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="Override device (cpu / cuda / mps).")
    p.add_argument("--val-frac", type=float, default=0.2)
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
    p.add_argument(
        "--gat-graph-label",
        action="store_true",
        help="Use the legacy v0.2 graph-level GAT head "
             "(modal next-task per case) instead of the default node-level head.",
    )
    p.add_argument(
        "--gat-bidirectional",
        action="store_true",
        help="Use legacy bidirectional event-graph edges in the GAT. The "
             "default is causal (forward-only) edges, which prevent the "
             "model from attending to events it is supposed to predict.",
    )
    p.add_argument(
        "--predict-time",
        action="store_true",
        help="Add a regression head to the GAT that predicts time-to-next-event "
             "alongside next-task. Reports MAE in hours on the val set.",
    )
    p.add_argument(
        "--time-loss-weight",
        type=float,
        default=0.5,
        help="Weight on the time-prediction MSE term when --predict-time is set.",
    )
    p.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip post-hoc temperature scaling on the GAT classifier. "
             "Default fits one scalar T on the val NLL so reported "
             "probabilities are calibrated; ECE before/after is recorded.",
    )
    p.add_argument(
        "--split-mode",
        choices=("case", "temporal"),
        default="case",
        help="How to split cases into train/val. 'case' (default) is a "
             "random case-level split. 'temporal' uses the most recent "
             "val_frac of cases as val — closer to production deployment, "
             "and surfaces drift that random splits hide.",
    )
    p.add_argument(
        "--seq-arch",
        choices=("lstm", "transformer"),
        default="lstm",
        help="Sequence-model architecture for the next-activity head. "
             "'lstm' (default) is the production-tuned baseline. "
             "'transformer' is a small causal transformer (4 layers x 4 "
             "heads) that competes with the LSTM head-to-head on long "
             "logs; uses the same calibration and time-head plumbing.",
    )
    p.add_argument("--transformer-layers", type=int, default=4)
    p.add_argument("--transformer-heads", type=int, default=4)
    p.add_argument(
        "--use-resource",
        action="store_true",
        help="Add the per-event resource ID as a parallel embedding to "
             "the LSTM input. The previous default sees only task IDs — "
             "strictly less information than the 1st-order Markov "
             "baseline. Adding resource is the highest-leverage single "
             "feature for closing the gap to Markov.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        dest="compile_models",
        help="Apply torch.compile (PyTorch 2+) to the GAT and the "
             "sequence model. Falls back gracefully on architectures "
             "the compiler can't handle (PyG GATConv, pack_padded LSTM).",
    )

    p.add_argument("--skip-gat", action="store_true")
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--skip-analyze", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--skip-cluster", action="store_true")
    p.add_argument("--skip-rl", action="store_true")


def _cfg_from_args(args: argparse.Namespace):
    from gnn_cli.stages import RunConfig

    return RunConfig(
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
        val_frac=args.val_frac,
        batch_size_gat=args.batch_size_gat,
        batch_size_lstm=args.batch_size_lstm,
        epochs_gat=args.epochs_gat,
        epochs_lstm=args.epochs_lstm,
        lr_gat=args.lr_gat,
        lr_lstm=args.lr_lstm,
        hidden_dim=args.hidden_dim,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        rl_episodes=args.rl_episodes,
        clusters=args.clusters,
        gat_node_level=not args.gat_graph_label,
        gat_causal=not args.gat_bidirectional,
        gat_predict_time=args.predict_time,
        time_loss_weight=args.time_loss_weight,
        calibrate=not args.no_calibrate,
        split_mode=args.split_mode,
        seq_arch=args.seq_arch,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        compile_models=args.compile_models,
        use_resource=args.use_resource,
        skip_gat=args.skip_gat,
        skip_lstm=args.skip_lstm,
        skip_analyze=args.skip_analyze,
        skip_viz=args.skip_viz,
        skip_cluster=args.skip_cluster,
        skip_rl=args.skip_rl,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gnn",
        description="Process mining with Graph Neural Networks, LSTMs, and tabular RL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-V", "--version", action="version", version=f"gnn {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    p_run = sub.add_parser("run", help="Run the full pipeline on an event-log CSV.")
    p_run.add_argument("data_path", help="Path to event-log CSV.")
    _add_run_flags(p_run)

    p_an = sub.add_parser(
        "analyze",
        help="Process-mining stats only (bottlenecks, cycle times, conformance).",
    )
    p_an.add_argument("data_path")
    p_an.add_argument("--out-dir", default="results")
    p_an.add_argument("--seed", type=int, default=42)

    p_cl = sub.add_parser("cluster", help="Spectral clustering of the task adjacency.")
    p_cl.add_argument("data_path")
    p_cl.add_argument("-k", "--clusters", type=int, default=3)
    p_cl.add_argument("--out-dir", default="results")
    p_cl.add_argument("--seed", type=int, default=42)

    p_sx = sub.add_parser(
        "predict-suffix",
        help="Beam-search rollout from a case prefix using the trained "
             "sequence model. Returns top-B continuations ranked by joint "
             "probability, with predicted total remaining cycle time.",
    )
    p_sx.add_argument("data_path")
    p_sx.add_argument("--case-id", required=True)
    p_sx.add_argument(
        "--model", required=True,
        help="Path to a trained sequence model "
             "(lstm_next_activity.pth or transformer_next_activity.pth).",
    )
    p_sx.add_argument(
        "--prefix-len", type=int, default=None,
        help="Take only the first N events of the case as the prefix. "
             "Default: use the full case.",
    )
    p_sx.add_argument("--beam", type=int, default=5)
    p_sx.add_argument("--max-steps", type=int, default=20)
    p_sx.add_argument(
        "--seq-arch", choices=("lstm", "transformer"), default="lstm",
        help="Architecture of the saved model.",
    )
    p_sx.add_argument("--hidden-dim", type=int, default=64)
    p_sx.add_argument("--predict-time", action="store_true",
                      help="Set if the saved model has the time-prediction head.")
    p_sx.add_argument("--out-dir", default="results")
    p_sx.add_argument("--seed", type=int, default=42)
    p_sx.add_argument("--device", default=None)

    p_wi = sub.add_parser(
        "whatif",
        help="Counterfactual rollout — swap a resource on a case and "
             "predict the cycle-time delta from historical (transition, "
             "resource) statistics.",
    )
    p_wi.add_argument("data_path")
    p_wi.add_argument("--case-id", required=True)
    p_wi.add_argument(
        "--swap-resource", required=True,
        help="Format: from=to (e.g. alice=bob).",
    )
    p_wi.add_argument("--out-dir", default="results")
    p_wi.add_argument("--seed", type=int, default=42)

    p_sv = sub.add_parser(
        "serve",
        help="Start a FastAPI inference endpoint backed by a trained "
             "sequence model. Exposes POST /predict and POST /predict_suffix.",
    )
    p_sv.add_argument("data_path", help="Path to the same CSV/XES used for training.")
    p_sv.add_argument("--run-dir", required=True,
                      help="Path to a results/run_<timestamp>/ directory.")
    p_sv.add_argument("--host", default="127.0.0.1")
    p_sv.add_argument("--port", type=int, default=8000)
    p_sv.add_argument("--seq-arch", choices=("lstm", "transformer"), default="lstm")
    p_sv.add_argument("--hidden-dim", type=int, default=64)
    p_sv.add_argument("--predict-time", action="store_true")
    p_sv.add_argument("--seed", type=int, default=42)

    p_df = sub.add_parser(
        "diff",
        help="Compare two run directories' metrics and emit a markdown report.",
    )
    p_df.add_argument("run_a", help="Path to first run_<timestamp>/ dir.")
    p_df.add_argument("run_b", help="Path to second run_<timestamp>/ dir.")
    p_df.add_argument("--out", default=None,
                      help="Write the report to this path instead of stdout.")

    p_ex = sub.add_parser(
        "explain",
        help="Run the trained GAT on a single case and dump per-event "
             "attention weights + a heatmap PNG. Pass --model to use a "
             "previously-trained model; otherwise a quick demo GAT is "
             "trained on the fly.",
    )
    p_ex.add_argument("data_path")
    p_ex.add_argument("--case-id", required=True, help="Case ID to explain.")
    p_ex.add_argument("--model", default=None,
                      help="Path to a previously-trained best_gnn_model.pth.")
    p_ex.add_argument("--out-dir", default="results")
    p_ex.add_argument("--seed", type=int, default=42)
    p_ex.add_argument("--epochs-gat", type=int, default=2,
                      help="Used only when --model is not supplied.")
    p_ex.add_argument("--hidden-dim", type=int, default=64)
    p_ex.add_argument("--gat-heads", type=int, default=4)
    p_ex.add_argument("--gat-layers", type=int, default=2)
    p_ex.add_argument("--device", default=None)

    p_bl = sub.add_parser(
        "baseline",
        help="Score null + Markov baselines on a CSV. The scientific floor "
             "every trained-model accuracy should be compared against.",
    )
    p_bl.add_argument("data_path")
    p_bl.add_argument("--out-dir", default="results")
    p_bl.add_argument("--seed", type=int, default=42)
    p_bl.add_argument("--val-frac", type=float, default=0.2)
    p_bl.add_argument(
        "--split-mode", choices=("case", "temporal"), default="case",
    )

    p_sm = sub.add_parser(
        "smoke",
        help="Generate synthetic data and run an abbreviated pipeline end-to-end.",
    )
    p_sm.add_argument("--num-cases", type=int, default=80)
    p_sm.add_argument("--keep-data", action="store_true",
                      help="Keep the generated CSV instead of using a tempfile.")
    _add_run_flags(p_sm)
    p_sm.set_defaults(epochs_gat=2, epochs_lstm=2, rl_episodes=5)

    sub.add_parser("version", help="Print version and exit.")

    return p


def cmd_run(args: argparse.Namespace) -> int:
    from gnn_cli.stages import run_full_pipeline

    cfg = _cfg_from_args(args)
    if not os.path.exists(args.data_path):
        print(f"error: dataset not found at {args.data_path}", file=sys.stderr)
        return EXIT_DATA
    run_full_pipeline(args.data_path, cfg)
    return EXIT_OK


def cmd_analyze(args: argparse.Namespace) -> int:
    from gnn_cli.stages import (
        save_metrics,
        setup_results_dir,
        stage_analyze,
        stage_preprocess,
    )

    if not os.path.exists(args.data_path):
        print(f"error: dataset not found at {args.data_path}", file=sys.stderr)
        return EXIT_DATA

    run_dir = setup_results_dir(args.out_dir)
    print(f"Results will be saved in: {run_dir}")
    df, _, _, le_task, le_resource, _, _ = stage_preprocess(
        args.data_path, val_frac=0.2, seed=args.seed
    )
    save_metrics(
        {
            "num_tasks": int(len(le_task.classes_)),
            "num_resources": int(len(le_resource.classes_)),
            "num_cases_total": int(df["case_id"].nunique()),
        },
        run_dir, "preprocessing_info.json",
    )
    stage_analyze(df, run_dir, le_task=le_task)
    print(f"Done. Analysis saved in {run_dir}")
    return EXIT_OK


def cmd_predict_suffix(args: argparse.Namespace) -> int:
    import json as _json

    import torch as _torch

    from gnn_cli.stages import setup_results_dir, stage_preprocess
    from gnn_cli.suffix import predict_suffix, render_suffix_report
    from models.lstm_model import NextActivityLSTM
    from models.transformer_model import NextActivityTransformer
    from modules.utils import pick_device, set_seed

    if not os.path.exists(args.data_path):
        print(f"error: dataset not found at {args.data_path}", file=sys.stderr)
        return EXIT_DATA
    if not os.path.exists(args.model):
        print(f"error: model not found at {args.model}", file=sys.stderr)
        return EXIT_DATA

    set_seed(args.seed)
    device = pick_device(args.device)
    run_dir = setup_results_dir(args.out_dir)
    print(f"Results will be saved in: {run_dir}")

    # Reuse the same preprocessing pipeline so encoders match training.
    df, _, _, le_task, _, _, _ = stage_preprocess(
        args.data_path, val_frac=0.2, seed=args.seed
    )
    case_df = df[df["case_id"] == args.case_id].sort_values("timestamp")
    if case_df.empty:
        print(f"error: case_id={args.case_id!r} not found", file=sys.stderr)
        return EXIT_DATA

    full_prefix = case_df["task_id"].astype(int).tolist()
    prefix = full_prefix[: args.prefix_len] if args.prefix_len else full_prefix
    if not prefix:
        print("error: prefix is empty after truncation", file=sys.stderr)
        return EXIT_USAGE

    num_classes = len(le_task.classes_)
    model: _torch.nn.Module
    if args.seq_arch == "transformer":
        model = NextActivityTransformer(
            num_classes, emb_dim=args.hidden_dim, hidden_dim=args.hidden_dim,
            predict_time=args.predict_time,
            max_len=max(args.max_steps + len(prefix), 64) + 8,
        ).to(device)
    else:
        model = NextActivityLSTM(
            num_classes, emb_dim=args.hidden_dim, hidden_dim=args.hidden_dim,
            num_layers=1, predict_time=args.predict_time,
        ).to(device)
    state = _torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state)

    completions = predict_suffix(
        model, prefix,
        beam=args.beam, max_steps=args.max_steps, device=device,
    )
    report = render_suffix_report(completions, le_task, prefix_len=len(prefix))
    print(report)

    out_dir = os.path.join(run_dir, "suffix_predictions")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"suffix_{args.case_id}.json")
    payload = {
        "case_id": args.case_id,
        "prefix": [
            str(le_task.inverse_transform([t])[0]) for t in prefix
        ],
        "completions": [
            {
                "rank": i,
                "tasks": [
                    str(le_task.inverse_transform([t])[0])
                    for t in seq[len(prefix):]
                ],
                "log_prob": logp,
                "prob": prob,
                "total_dt_hours": dt_seconds / 3600.0,
            }
            for i, (seq, logp, dt_seconds, prob) in enumerate(completions, 1)
        ],
    }
    with open(json_path, "w") as f:
        _json.dump(payload, f, indent=2, default=str)
    print(f"Saved {json_path}")
    return EXIT_OK


def cmd_whatif(args: argparse.Namespace) -> int:
    import json as _json

    from gnn_cli.stages import setup_results_dir, stage_preprocess
    from gnn_cli.whatif import predict_whatif, render_whatif_report

    if not os.path.exists(args.data_path):
        print(f"error: dataset not found: {args.data_path}", file=sys.stderr)
        return EXIT_DATA
    if "=" not in args.swap_resource:
        print("error: --swap-resource must be 'from=to' (e.g. alice=bob)", file=sys.stderr)
        return EXIT_USAGE
    from_r, to_r = args.swap_resource.split("=", 1)

    run_dir = setup_results_dir(args.out_dir)
    print(f"Results will be saved in: {run_dir}")
    df, _, _, le_task, _, _, _ = stage_preprocess(
        args.data_path, val_frac=0.2, seed=args.seed
    )

    result = predict_whatif(df, args.case_id, (from_r, to_r), le_task=le_task)
    report = render_whatif_report(result)
    print(report)

    out_dir = os.path.join(run_dir, "whatif")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"whatif_{args.case_id}.json")
    with open(json_path, "w") as f:
        _json.dump(result, f, indent=2, default=str)
    md_path = os.path.join(out_dir, f"whatif_{args.case_id}.md")
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Saved {json_path} and {md_path}")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from gnn_cli.serve import serve
    except ImportError as e:
        print(
            f"error: serve dependencies not installed ({e}). "
            f"Run `pip install fastapi uvicorn` and retry.",
            file=sys.stderr,
        )
        return EXIT_RUNTIME

    if not os.path.exists(args.data_path):
        print(f"error: dataset not found: {args.data_path}", file=sys.stderr)
        return EXIT_DATA
    if not os.path.exists(args.run_dir):
        print(f"error: run dir not found: {args.run_dir}", file=sys.stderr)
        return EXIT_DATA

    serve(
        run_dir=args.run_dir, data_path=args.data_path,
        host=args.host, port=args.port,
        seq_arch=args.seq_arch, hidden_dim=args.hidden_dim,
        predict_time=args.predict_time, seed=args.seed,
    )
    return EXIT_OK


def cmd_diff(args: argparse.Namespace) -> int:
    from gnn_cli.diff import write_diff

    for path in (args.run_a, args.run_b):
        if not os.path.exists(path):
            print(f"error: run dir not found: {path}", file=sys.stderr)
            return EXIT_DATA
    report = write_diff(args.run_a, args.run_b, out_path=args.out)
    if args.out:
        print(f"Wrote diff to {args.out}")
    else:
        print(report)
    return EXIT_OK


def cmd_explain(args: argparse.Namespace) -> int:
    from gnn_cli.explain import explain_case, load_or_train_gat
    from gnn_cli.stages import RunConfig, setup_results_dir, stage_preprocess
    from modules.utils import pick_device, set_seed

    if not os.path.exists(args.data_path):
        print(f"error: dataset not found at {args.data_path}", file=sys.stderr)
        return EXIT_DATA

    set_seed(args.seed)
    device = pick_device(args.device)

    run_dir = setup_results_dir(args.out_dir)
    print(f"Results will be saved in: {run_dir}")
    df, train_df, val_df, le_task, _, _, _ = stage_preprocess(
        args.data_path, val_frac=0.2, seed=args.seed
    )

    cfg = RunConfig(
        epochs_gat=args.epochs_gat,
        hidden_dim=args.hidden_dim,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
    )
    model = load_or_train_gat(
        train_df, val_df, le_task, device, args.model, cfg
    )
    # explain_case calls build_graph_data which needs the feat_* columns.
    # train_df and val_df both have them (from stage_preprocess); concat
    # gives a scaled full df without re-fitting the scaler.
    import pandas as pd

    scaled_df = pd.concat([train_df, val_df], ignore_index=True)
    summary = explain_case(
        scaled_df, args.case_id, model, le_task,
        os.path.join(run_dir, "explanations"), device,
    )
    print(
        f"Explained case {summary['case_id']} "
        f"({summary['num_events']} events). "
        f"Output: {run_dir}/explanations/explain_{args.case_id}.json + .png"
    )
    return EXIT_OK


def cmd_baseline(args: argparse.Namespace) -> int:
    from gnn_cli.stages import (
        save_metrics,
        setup_results_dir,
        stage_baselines,
        stage_preprocess,
    )

    if not os.path.exists(args.data_path):
        print(f"error: dataset not found at {args.data_path}", file=sys.stderr)
        return EXIT_DATA

    run_dir = setup_results_dir(args.out_dir)
    print(f"Results will be saved in: {run_dir}")
    df, train_df, val_df, le_task, le_resource, _, _ = stage_preprocess(
        args.data_path, val_frac=args.val_frac, seed=args.seed,
        split_mode=args.split_mode,
    )
    save_metrics(
        {
            "num_tasks": int(len(le_task.classes_)),
            "num_resources": int(len(le_resource.classes_)),
            "num_cases_total": int(df["case_id"].nunique()),
            "num_cases_train": int(train_df["case_id"].nunique()),
            "num_cases_val": int(val_df["case_id"].nunique()),
        },
        run_dir, "preprocessing_info.json",
    )
    metrics = stage_baselines(train_df, val_df, run_dir)
    print(
        f"most-common acc={metrics['most_common_accuracy']:.4f}, "
        f"markov acc={metrics['markov_accuracy']:.4f} "
        f"(coverage={metrics['markov_coverage']:.2f}, "
        f"n_val_events={metrics['num_val_events']})"
    )
    return EXIT_OK


def cmd_cluster(args: argparse.Namespace) -> int:
    from gnn_cli.stages import (
        setup_results_dir,
        stage_cluster,
        stage_preprocess,
    )

    if not os.path.exists(args.data_path):
        print(f"error: dataset not found at {args.data_path}", file=sys.stderr)
        return EXIT_DATA

    run_dir = setup_results_dir(args.out_dir)
    print(f"Results will be saved in: {run_dir}")
    df, _, _, le_task, _, _, _ = stage_preprocess(
        args.data_path, val_frac=0.2, seed=args.seed
    )
    stage_cluster(df, le_task, len(le_task.classes_), args.clusters, args.seed, run_dir)
    print(f"Done. Clustering saved in {run_dir}")
    return EXIT_OK


def cmd_smoke(args: argparse.Namespace) -> int:
    import contextlib

    from gnn_cli.smoke import generate_synthetic_csv
    from gnn_cli.stages import run_full_pipeline

    if args.keep_data:
        data_path = os.path.join(args.out_dir, "smoke_data.csv")
    else:
        with tempfile.NamedTemporaryFile(
            prefix="gnn_smoke_", suffix=".csv", delete=False,
        ) as tmp:
            data_path = tmp.name

    generate_synthetic_csv(data_path, num_cases=args.num_cases, seed=args.seed)
    print(f"Generated synthetic event log: {data_path} ({args.num_cases} cases)")

    cfg = _cfg_from_args(args)
    try:
        run_full_pipeline(data_path, cfg)
    finally:
        if not args.keep_data:
            with contextlib.suppress(OSError):
                os.unlink(data_path)
    return EXIT_OK


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"gnn {__version__}")
    return EXIT_OK


COMMANDS = {
    "run": cmd_run,
    "analyze": cmd_analyze,
    "cluster": cmd_cluster,
    "baseline": cmd_baseline,
    "explain": cmd_explain,
    "predict-suffix": cmd_predict_suffix,
    "whatif": cmd_whatif,
    "serve": cmd_serve,
    "diff": cmd_diff,
    "smoke": cmd_smoke,
    "version": cmd_version,
}


def _load_toml_run_config(path: str) -> dict:
    """Read run-flag overrides from a TOML file.

    Accepts either a top-level table or a [run] sub-table. Keys must
    match argparse dest names (snake_case). Returns ``{}`` on read
    error rather than crashing — the user gets a clear "config X not
    found" later from the file open above.
    """
    try:
        import tomllib as _toml  # py 3.11+
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as _toml  # noqa: F401  # type: ignore[import-not-found]
    with open(path, "rb") as f:
        data = _toml.load(f)
    return data.get("run", data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()

    # Two-pass parse so a TOML --config can supply defaults that CLI
    # flags override. Pass 1: peek for --config; pass 2: real parse with
    # set_defaults applied. Subparsers all use the same shared flag set
    # so set_defaults walks every subcommand parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.config:
        try:
            overrides = _load_toml_run_config(pre_args.config)
        except FileNotFoundError:
            print(f"error: config file not found: {pre_args.config}", file=sys.stderr)
            return EXIT_DATA
        # Walk every subparser and apply matching defaults; argparse
        # quietly ignores unknown keys when set on the wrong subparser.
        for action in parser._subparsers._group_actions[0].choices.values():  # type: ignore[attr-defined]
            action.set_defaults(**{
                k: v for k, v in overrides.items()
                if any(a.dest == k for a in action._actions)
            })

    args = parser.parse_args(argv)
    handler = COMMANDS[args.command]
    try:
        return handler(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_DATA
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except Exception:
        traceback.print_exc()
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
