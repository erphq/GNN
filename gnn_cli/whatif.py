"""Counterfactual rollout — swap a resource on a case, predict the delta.

Honest framing: this is **empirical counterfactual estimation**, not
ML inference. The sequence model takes only task IDs (it doesn't
condition on resource), so swapping resources at the model level is
a no-op. What this module does instead is reuse the same historical
per-transition / per-resource statistics that drive the bottleneck-
driver analysis, and re-score the case under the counterfactual
resource assignment.

Workflow::

    gnn whatif input/log.csv --case-id "case_42" \
        --swap-resource alice=bob

For every transition in the case, the actual resource is replaced
with the swapped value (when the original equals the LHS of the
swap), and the historical mean wait at that (transition, resource)
group is looked up. We sum across the case for the predicted total
cycle time under the counterfactual, and compare to the model's
prediction under the actual resource assignment.

Limitations
-----------
- Reads from history alone — extrapolates the resource-conditioned
  mean wait observed in the log, not what an unseen workload pattern
  might do. If a (transition, resource) cell has zero support, we
  fall back to the transition-wide mean and flag the row.
- Each event is treated independently; cumulative load effects on
  the swapped resource (queueing, fatigue) aren't modeled.
"""

from __future__ import annotations


import pandas as pd


def _resource_wait_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(task_id, next_task_id, resource) mean wait in hours."""
    df = df.copy()
    df["next_task_id"] = df.groupby("case_id")["task_id"].shift(-1)
    df["next_timestamp"] = df.groupby("case_id")["timestamp"].shift(-1)
    df = df.dropna(subset=["next_task_id"]).copy()
    df["wait_h"] = (
        df["next_timestamp"] - df["timestamp"]
    ).dt.total_seconds() / 3600.0
    df["next_task_id"] = df["next_task_id"].astype(int)
    return (
        df.groupby(["task_id", "next_task_id", "resource"])["wait_h"]
        .agg(["mean", "count"])
        .reset_index()
    )


def _transition_wait_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(task_id, next_task_id) mean wait — used when a (resource)
    cell has zero support."""
    df = df.copy()
    df["next_task_id"] = df.groupby("case_id")["task_id"].shift(-1)
    df["next_timestamp"] = df.groupby("case_id")["timestamp"].shift(-1)
    df = df.dropna(subset=["next_task_id"]).copy()
    df["wait_h"] = (
        df["next_timestamp"] - df["timestamp"]
    ).dt.total_seconds() / 3600.0
    df["next_task_id"] = df["next_task_id"].astype(int)
    return (
        df.groupby(["task_id", "next_task_id"])["wait_h"]
        .agg(["mean", "count"])
        .reset_index()
    )


def predict_whatif(
    df: pd.DataFrame,
    case_id: str,
    swap: tuple[str, str],
    *,
    le_task=None,
) -> dict:
    """For each transition in ``case_id``, predict the counterfactual
    wait under ``swap=(from_resource, to_resource)``.

    Returns a dict with per-transition predictions and the summary
    deltas. ``le_task`` is optional; pass it to render task names
    instead of encoded IDs in the output.
    """
    from_r, to_r = swap
    case_df = df[df["case_id"] == case_id].sort_values("timestamp").copy()
    if case_df.empty:
        raise ValueError(f"case_id={case_id!r} not found")

    res_table = _resource_wait_table(df)
    fallback = _transition_wait_fallback(df)

    case_df["next_task_id"] = case_df["task_id"].shift(-1)
    case_df["next_timestamp"] = case_df["timestamp"].shift(-1)
    transitions = case_df.dropna(subset=["next_task_id"]).copy()
    transitions["wait_h"] = (
        transitions["next_timestamp"] - transitions["timestamp"]
    ).dt.total_seconds() / 3600.0
    transitions["next_task_id"] = transitions["next_task_id"].astype(int)

    def _name(t_id: int) -> str:
        if le_task is None:
            return str(t_id)
        return str(le_task.inverse_transform([int(t_id)])[0])

    rows = []
    actual_total = 0.0
    counterfactual_total = 0.0
    fallback_count = 0
    for _, ev in transitions.iterrows():
        src, tgt = int(ev["task_id"]), int(ev["next_task_id"])
        actual_resource = str(ev["resource"])
        cf_resource = to_r if actual_resource == from_r else actual_resource

        # Look up per-resource mean.
        m = res_table[
            (res_table["task_id"] == src)
            & (res_table["next_task_id"] == tgt)
            & (res_table["resource"] == cf_resource)
        ]
        if m.empty:
            f = fallback[
                (fallback["task_id"] == src) & (fallback["next_task_id"] == tgt)
            ]
            cf_mean = float(f["mean"].iloc[0]) if len(f) else float(ev["wait_h"])
            used_fallback = True
            fallback_count += 1
        else:
            cf_mean = float(m["mean"].iloc[0])
            used_fallback = False

        actual_h = float(ev["wait_h"])
        actual_total += actual_h
        counterfactual_total += cf_mean
        rows.append({
            "src_task": _name(src),
            "tgt_task": _name(tgt),
            "actual_resource": actual_resource,
            "cf_resource": cf_resource,
            "actual_wait_h": actual_h,
            "cf_wait_h": cf_mean,
            "delta_h": cf_mean - actual_h,
            "fallback_used": used_fallback,
        })

    return {
        "case_id": case_id,
        "swap": {"from": from_r, "to": to_r},
        "per_event": rows,
        "actual_total_h": actual_total,
        "counterfactual_total_h": counterfactual_total,
        "delta_total_h": counterfactual_total - actual_total,
        "fallback_count": fallback_count,
    }


def render_whatif_report(result: dict) -> str:
    """Render the counterfactual analysis as a markdown table."""
    lines = [
        f"### Counterfactual: case `{result['case_id']}`, "
        f"swap `{result['swap']['from']} → {result['swap']['to']}`",
        "",
        f"Actual total wait: **{result['actual_total_h']:.2f} h**",
        f"Counterfactual total: **{result['counterfactual_total_h']:.2f} h**",
        f"Delta: **{result['delta_total_h']:+.2f} h** "
        f"({result['fallback_count']} of {len(result['per_event'])} "
        f"transitions used fallback estimates)",
        "",
        "| transition | actual resource | cf resource | actual h | cf h | Δ h | fallback |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in result["per_event"]:
        lines.append(
            f"| {r['src_task']} → {r['tgt_task']} "
            f"| {r['actual_resource']} | {r['cf_resource']} "
            f"| {r['actual_wait_h']:.2f} | {r['cf_wait_h']:.2f} "
            f"| {r['delta_h']:+.2f} | {'yes' if r['fallback_used'] else ''} |"
        )
    return "\n".join(lines) + "\n"
