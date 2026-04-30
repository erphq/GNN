"""Aggregate metrics from one or more `gnn run` output dirs into a
markdown leaderboard suitable for the README.

Usage::

    python bench/eval.py \
        --row "BPI 2020 DomesticDeclarations" bench/results/bpi2020 \
        --row "Synthetic Markov (200 cases)"  bench/results/smoke200 \
        --out bench/leaderboard.md

The script reads ``baseline_metrics.json``, ``gat_metrics.json``,
``lstm_metrics.json``, and ``process_analysis.json`` from each named
run directory (auto-resolves ``run_<timestamp>/``) and emits a single
markdown table comparing models against the Markov baseline floor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple


def _resolve_run_dir(p: str) -> Path:
    """Accept either the parent dir (containing run_*) or the run dir itself."""
    path = Path(p)
    if (path / "metrics").is_dir():
        return path
    runs = sorted(path.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"no run_* directory found under {p!r}")
    return runs[-1]


def _read(run_dir: Path, fname: str) -> dict:
    f = run_dir / "metrics" / fname
    return json.loads(f.read_text()) if f.exists() else {}


def _fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{digits}f}%"


def _fmt(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def _row(name: str, run_dir: Path) -> dict:
    base = _read(run_dir, "baseline_metrics.json")
    gat = _read(run_dir, "gat_metrics.json")
    lstm = _read(run_dir, "lstm_metrics.json")
    pa = _read(run_dir, "process_analysis.json")
    pp = _read(run_dir, "preprocessing_info.json")
    return {
        "name": name,
        "n_cases": pp.get("num_cases_total"),
        "n_tasks": pp.get("num_tasks"),
        "most_common": base.get("most_common_accuracy"),
        "markov": base.get("markov_accuracy"),
        "gat_acc": gat.get("accuracy"),
        "gat_f1": gat.get("macro_f1"),
        "gat_lift": gat.get("lift_over_markov"),
        "gat_ece": gat.get("ece_after_calibration"),
        "gat_T": gat.get("temperature"),
        "gat_dt_mae": gat.get("dt_mae_hours"),
        "lstm_acc": lstm.get("accuracy"),
        "lstm_f1": lstm.get("macro_f1"),
        "lstm_lift": lstm.get("lift_over_markov"),
        "lstm_ece": lstm.get("ece_after_calibration"),
        "lstm_dt_mae": lstm.get("dt_mae_hours"),
        "fitness": pa.get("conformance_fitness"),
        "precision": pa.get("conformance_precision"),
        "f_score": pa.get("conformance_f_score"),
    }


def render_leaderboard(rows: List[dict]) -> str:
    """Two stacked tables: predictive metrics + process-mining metrics."""
    out: List[str] = []
    out.append("### Predictive performance")
    out.append("")
    out.append(
        "| dataset | cases | tasks | "
        "most-common | **Markov** | "
        "LSTM acc / F1 / ΔMarkov | "
        "**GAT acc / F1 / ΔMarkov** | "
        "GAT ECE | GAT T | GAT dt MAE (h) |"
    )
    out.append(
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|"
    )
    for r in rows:
        out.append(
            f"| {r['name']} "
            f"| {r['n_cases'] or '—'} "
            f"| {r['n_tasks'] or '—'} "
            f"| {_fmt_pct(r['most_common'])} "
            f"| **{_fmt_pct(r['markov'])}** "
            f"| {_fmt_pct(r['lstm_acc'])} / {_fmt(r['lstm_f1'])} / "
            f"{_fmt_pct(r['lstm_lift'])} "
            f"| **{_fmt_pct(r['gat_acc'])} / {_fmt(r['gat_f1'])} / "
            f"{_fmt_pct(r['gat_lift'])}** "
            f"| {_fmt(r['gat_ece'])} "
            f"| {_fmt(r['gat_T'], 2)} "
            f"| {_fmt(r['gat_dt_mae'], 2)} |"
        )

    out.append("")
    out.append("### Process-mining quality (PM4Py inductive miner + token replay)")
    out.append("")
    out.append("| dataset | fitness | precision | F-score |")
    out.append("|---|---:|---:|---:|")
    for r in rows:
        out.append(
            f"| {r['name']} "
            f"| {_fmt(r['fitness'])} "
            f"| {_fmt(r['precision'])} "
            f"| {_fmt(r['f_score'])} |"
        )
    return "\n".join(out) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--row", nargs=2, action="append", metavar=("NAME", "RUN_DIR"),
        required=True,
        help="Add a row: NAME shown in table, RUN_DIR is either a "
             "results/run_<ts>/ or its parent.",
    )
    p.add_argument("--out", default=None, help="Write to file (else stdout).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = [_row(name, _resolve_run_dir(rd)) for name, rd in args.row]
    table = render_leaderboard(rows)
    if args.out:
        Path(args.out).write_text(table)
        print(f"Wrote {args.out} ({len(rows)} row(s))")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
