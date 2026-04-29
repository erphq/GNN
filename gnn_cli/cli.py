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
from typing import Optional, Sequence

from gnn_cli import __version__

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DATA = 3
EXIT_RUNTIME = 4


def _add_run_flags(p: argparse.ArgumentParser) -> None:
    """Hyperparameter flags shared by `run` and `smoke`."""
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
    stage_analyze(df, run_dir)
    print(f"Done. Analysis saved in {run_dir}")
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
    from gnn_cli.smoke import generate_synthetic_csv
    from gnn_cli.stages import run_full_pipeline

    if args.keep_data:
        data_path = os.path.join(args.out_dir, "smoke_data.csv")
    else:
        tmp = tempfile.NamedTemporaryFile(
            prefix="gnn_smoke_", suffix=".csv", delete=False
        )
        tmp.close()
        data_path = tmp.name

    generate_synthetic_csv(data_path, num_cases=args.num_cases, seed=args.seed)
    print(f"Generated synthetic event log: {data_path} ({args.num_cases} cases)")

    cfg = _cfg_from_args(args)
    try:
        run_full_pipeline(data_path, cfg)
    finally:
        if not args.keep_data:
            try:
                os.unlink(data_path)
            except OSError:
                pass
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
    "smoke": cmd_smoke,
    "version": cmd_version,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
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
