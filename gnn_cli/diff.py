"""Compare two run directories — what changed, by how much.

Hyperparameter exploration usually means running gnn N times and
diffing JSON files by hand. This module formalizes that.

For each ``metrics/*.json`` shared by both runs, the diff:

- For numbers: abs delta + % change.
- For dicts (e.g. ``per_class``): per-key recursive diff.
- For lists: change in length.
- Clustering: which tasks moved between cluster IDs (label permutations
  are valid clusterings, so this is a heuristic — see comment).

Output is a markdown report on stdout (optionally written to a file).
"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt_delta(a, b) -> str:
    """Format a numeric delta as 'a → b (Δ +0.05, +6.3%)'."""
    delta = b - a
    pct = (delta / a * 100.0) if a not in (0, 0.0) else float("inf")
    pct_str = f"{pct:+.1f}%" if pct != float("inf") else "—"
    return f"{a:.4g} → {b:.4g} (Δ {delta:+.4g}, {pct_str})"


def _diff_value(label, a, b, lines, threshold: float = 1e-9):
    """Append a markdown-friendly line per changed leaf value."""
    if isinstance(a, dict) and isinstance(b, dict):
        all_keys = sorted(set(a.keys()) | set(b.keys()))
        for k in all_keys:
            if k not in a:
                lines.append(f"- **{label}.{k}**: added → `{b[k]}`")
            elif k not in b:
                lines.append(f"- **{label}.{k}**: removed (was `{a[k]}`)")
            else:
                _diff_value(f"{label}.{k}", a[k], b[k], lines, threshold)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            lines.append(f"- **{label}**: list length {len(a)} → {len(b)}")
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            lines.append(f"- **{label}**: `{a}` → `{b}`")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(b - a) > threshold:
            lines.append(f"- **{label}**: {_fmt_delta(a, b)}")
    else:
        if a != b:
            lines.append(f"- **{label}**: `{a!r}` → `{b!r}`")


def _diff_clustering(a: dict, b: dict, lines: list):
    """Tasks that moved between cluster IDs.

    A label permutation (cluster 0 ↔ 1) is the same clustering — but
    detecting that requires Hungarian matching + ARI. For v1 we just
    surface which tasks have a different cluster ID; downstream users
    can interpret. If you see *every* task moved, suspect a permutation.
    """
    a_map = a.get("task_clusters", {})
    b_map = b.get("task_clusters", {})
    moved = []
    for task in sorted(set(a_map) | set(b_map)):
        ca, cb = a_map.get(task), b_map.get(task)
        if ca != cb:
            moved.append((task, ca, cb))
    if moved:
        lines.append("\n### Clustering")
        lines.append(f"{len(moved)} of {len(a_map)} tasks have a different cluster ID:\n")
        lines.append("| task | run A | run B |")
        lines.append("|---|---|---|")
        for t, ca, cb in moved:
            lines.append(f"| {t} | {ca} | {cb} |")


def diff_runs(run_a: str, run_b: str) -> str:
    """Build a markdown report comparing two run directories."""
    a, b = Path(run_a), Path(run_b)
    lines = ["# Run diff", f"- A: `{a}`", f"- B: `{b}`", ""]

    a_metrics = {p.name for p in (a / "metrics").glob("*.json")}
    b_metrics = {p.name for p in (b / "metrics").glob("*.json")}

    only_a = sorted(a_metrics - b_metrics)
    only_b = sorted(b_metrics - a_metrics)
    if only_a:
        lines.append(f"### Metrics only in A: {only_a}")
    if only_b:
        lines.append(f"### Metrics only in B: {only_b}")

    for fname in sorted(a_metrics & b_metrics):
        if fname == "clustering_results.json":
            ja = json.loads((a / "metrics" / fname).read_text())
            jb = json.loads((b / "metrics" / fname).read_text())
            _diff_clustering(ja, jb, lines)
            continue
        ja = json.loads((a / "metrics" / fname).read_text())
        jb = json.loads((b / "metrics" / fname).read_text())
        section_lines: list = []
        _diff_value(fname, ja, jb, section_lines)
        if section_lines:
            lines.append(f"\n### {fname}")
            lines.extend(section_lines)

    if len(lines) <= 4:
        lines.append("\n_No differences in any metric file._")
    return "\n".join(lines) + "\n"


def write_diff(run_a: str, run_b: str, out_path: str | None = None) -> str:
    """Convenience wrapper: write report to ``out_path`` if given."""
    report = diff_runs(run_a, run_b)
    if out_path:
        Path(out_path).write_text(report)
    return report
