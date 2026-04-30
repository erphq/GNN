"""Cross-seed variance harness.

Runs ``gnn run`` (or any subset thereof via passthrough flags) with N
different seeds in sequence, then aggregates the per-seed metrics
JSONs into ``mean / std / min / max`` per leaf metric. The output
sits alongside the per-seed run directories so a downstream
aggregator can pick it up.

The point: bootstrap CI captures sampling noise (which val rows the
metric was scored on) but not training noise (which initialization,
which mini-batch order). On BPI 2020, the LSTM lands at top-1 ≈ 81.8%
across three feature ablations — but is that 81.8% a robust point
estimate or a single seed's draw? This script answers that.

Usage::

    python bench/seeds.py \\
        --csv input/BPI2020_DomesticDeclarations.csv \\
        --seeds 42 43 44 \\
        --out-root bench/results/bpi2020_seeds \\
        --published-out bench/published/bpi2020_lstm_seeds \\
        -- --epochs-lstm 30 --hidden-dim 256 --lr-lstm 5e-4 \\
        --predict-time --skip-gat --skip-rl --device cpu

Everything after ``--`` is forwarded to ``gnn run`` verbatim.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts to dotted-path leaves; skip lists."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key))
        elif isinstance(v, (int, float, bool)) and not isinstance(v, bool):
            out[key] = v
    return out


def _aggregate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For each numeric leaf metric present in *every* run, compute
    mean / std / min / max / N. Skip metrics that vary in presence."""
    if not per_seed:
        return {}
    flats = [_flatten(d) for d in per_seed]
    common_keys = set(flats[0].keys())
    for f in flats[1:]:
        common_keys &= set(f.keys())

    summary: Dict[str, Any] = {}
    for key in sorted(common_keys):
        values = [f[key] for f in flats]
        if not all(isinstance(v, (int, float)) for v in values):
            continue
        # statistics.stdev raises on N=1; report 0.0 in that case.
        std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        summary[key] = {
            "mean": float(statistics.mean(values)),
            "std": std,
            "min": float(min(values)),
            "max": float(max(values)),
            "n": len(values),
            "values": [float(v) for v in values],
        }
    return summary


def _run_one(csv: str, seed: int, out_root: Path, passthrough: List[str]) -> Path:
    """Run ``gnn run`` for one seed; return the run_<ts>/ dir."""
    out_dir = out_root / f"seed_{seed}"
    cmd = [
        sys.executable, "-m", "gnn_cli", "run", csv,
        "--seed", str(seed),
        "--out-dir", str(out_dir),
        *passthrough,
    ]
    print(f"  [seed {seed}] launching: {' '.join(cmd[3:])}")
    subprocess.run(cmd, check=True)
    runs = sorted(out_dir.glob("run_*"))
    if not runs:
        raise RuntimeError(f"no run_<ts>/ produced under {out_dir}")
    return runs[-1]


def _collect_metrics(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read every metrics/*.json and tag by filename."""
    out = {}
    for path in (run_dir / "metrics").glob("*.json"):
        out[path.stem] = json.loads(path.read_text())
    return out


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Cross-seed variance harness.",
        usage="%(prog)s --csv CSV --seeds N [N ...] --out-root DIR "
              "[--published-out DIR] -- [gnn-run flags...]",
    )
    p.add_argument("--csv", required=True, help="Path to event-log CSV/XES.")
    p.add_argument(
        "--seeds", type=int, nargs="+", required=True,
        help="Space-separated list of seeds (e.g. 42 43 44).",
    )
    p.add_argument(
        "--out-root", required=True,
        help="Root directory for per-seed run outputs.",
    )
    p.add_argument(
        "--published-out", default=None,
        help="Optional path to write the aggregated seed_variance.json "
             "(creates parent dirs as needed).",
    )
    return p.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    print(f"Running {len(args.seeds)} seeds: {args.seeds}")
    print(f"  csv:        {args.csv}")
    print(f"  out_root:   {out_root}")
    print(f"  passthrough: {' '.join(passthrough)}")

    per_file_per_seed: Dict[str, List[Dict[str, Any]]] = {}
    for seed in args.seeds:
        run_dir = _run_one(args.csv, seed, out_root, passthrough)
        for fname, payload in _collect_metrics(run_dir).items():
            per_file_per_seed.setdefault(fname, []).append(payload)

    aggregate: Dict[str, Dict[str, Any]] = {}
    for fname, runs in per_file_per_seed.items():
        aggregate[fname] = _aggregate(runs)

    output = {
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "passthrough_args": passthrough,
        "metrics": aggregate,
    }
    summary_path = out_root / "seed_variance.json"
    summary_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {summary_path}")

    if args.published_out:
        pub = Path(args.published_out)
        pub.mkdir(parents=True, exist_ok=True)
        (pub / "seed_variance.json").write_text(json.dumps(output, indent=2))
        print(f"Wrote {pub / 'seed_variance.json'}")

    # Print a tiny cross-seed report for the most common LSTM keys.
    print("\n=== Cross-seed summary (lstm_metrics) ===")
    lstm = aggregate.get("lstm_metrics", {})
    for key in (
        "accuracy", "top_3_accuracy", "top_5_accuracy", "mrr",
        "macro_f1", "ece_after_calibration", "dt_mae_hours",
        "lift_over_markov",
    ):
        if key not in lstm:
            continue
        m = lstm[key]
        print(
            f"  {key:28s} {m['mean']:.4f} ± {m['std']:.4f}  "
            f"(min {m['min']:.4f}, max {m['max']:.4f}, n={m['n']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
