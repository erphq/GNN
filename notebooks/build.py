"""Build ``01_bpi2020_tutorial.ipynb`` from the cell list below.

Source-of-truth lives in this file (Python, diff-able, lint-able). The
notebook is the rendered artifact — regenerate after edits with::

    python notebooks/build.py

The notebook contains no pre-executed outputs by design; readers
either run it themselves (~30 s with published metrics, ~30 min if
they want to retrain) or read the static markdown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

# Cells: list of (cell_type, source). cell_type is "markdown" or "code".
CELLS: List[Tuple[str, str]] = [
    ("markdown", """\
# Tutorial — process mining with BPI 2020, end to end

This notebook walks the full `gnn` workflow on a real industrial event
log. By the end you'll have:

- run a descriptive analysis surfacing bottlenecks and root-cause drivers
- scored the **null + Markov baselines** that any deep model has to beat
- read the published LSTM metrics (or retrained one yourself)
- explained a single prediction with attention saliency
- rolled the model forward from a case prefix
- run a counterfactual resource-swap analysis

The dataset is **BPI Challenge 2020 — Domestic Declarations**: 10,366
cases of expense declaration workflows from a Dutch academic institution
spanning 24 months, with 17 task classes. Already shipped at
`input/BPI2020_DomesticDeclarations.csv`.

> **Time budget.** ~30 seconds if you read the published metrics and
> skip retraining. ~30 minutes if you retrain end-to-end on CPU
> (~8 minutes on Apple silicon MPS).

> **Setup.** From the repo root: `pip install -e .` plus the deps in
> `requirements.txt`. Then `jupyter lab notebooks/`. The notebook
> assumes you launch it from `notebooks/` — `REPO` resolves to the
> parent directory.
"""),
    ("code", """\
import json
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path("..").resolve()
LOG = REPO / "input" / "BPI2020_DomesticDeclarations.csv"
PUBLISHED = REPO / "bench" / "published" / "bpi2020_lstm" / "metrics"

print("Repo root:", REPO)
print("Event log:", LOG, "·", LOG.stat().st_size // 1024, "KB")
print("Published metrics:", PUBLISHED.exists())
"""),
    ("markdown", """\
## 1. First look at the data

The BPI 2020 log uses XES-style column names (`case:concept:name`,
`concept:name`, `time:timestamp`). The pipeline auto-renames these to
the canonical `case_id` / `task_name` / `timestamp` schema, so the
columns below differ from what `gnn` will see internally.
"""),
    ("code", """\
df = pd.read_csv(LOG)
print(f"{len(df):,} events across {df['case:concept:name'].nunique():,} cases")
print(f"{df['concept:name'].nunique()} distinct task types")
print(f"Time span: {df['time:timestamp'].min()} → {df['time:timestamp'].max()}")
df.head()
"""),
    ("markdown", """\
## 2. Descriptive process mining — `gnn analyze`

Before any model training, see what's visible in the log. `gnn analyze`
runs PM4Py's inductive miner + token-replay conformance + a
bottleneck-driver analysis without touching the deep models. Takes ~30
seconds.
"""),
    ("code", """\
out_dir = Path("/tmp/tutorial_analyze")
subprocess.run(
    ["gnn", "analyze", str(LOG), "--out-dir", str(out_dir)],
    check=True,
)
analyze_run = sorted(out_dir.glob("run_*"))[-1]
print("Output dir:", analyze_run)
"""),
    ("code", """\
process_analysis = json.loads(
    (analyze_run / "metrics" / "process_analysis.json").read_text()
)

for key in ("total_traces", "num_long_cases", "cycle_time_95th_percentile_h",
            "num_deviant_traces", "conformance_fitness",
            "conformance_precision", "conformance_f_score"):
    print(f"{key:38s} {process_analysis[key]}")
"""),
    ("markdown", """\
**Read this carefully.**

- `conformance_fitness = 1.000` — every observed event is reproducible
  by the discovered Petri net.
- `conformance_precision = 0.247` — the discovered model also allows a
  *lot* of behavior the log never showed.

This is the **flower-model warning** the v0.4 audit was built to
surface. Before this fix, `num_deviant_traces = 0` looked green and
hid the precision problem entirely. The F-score 0.396 is the real
quality of the discovered model.

Now the *why* — for each top bottleneck transition, which case
attribute drives the wait time?
"""),
    ("code", """\
drivers = process_analysis["bottleneck_drivers"]
for transition, payload in list(drivers.items())[:3]:
    print(f"\\n{transition}")
    print(f"  n={payload['n_transitions']}, mean wait {payload['mean_wait_h']:.1f}h")
    for d in payload["drivers"][:3]:
        print(
            f"    {d['feature']:14s} "
            f"spread {d['spread_h']:>7.1f}h  "
            f"worst {d['worst_group']!r:<8s} ({d['worst_group_mean_h']:.1f}h, n={d['worst_group_n']})"
        )
"""),
    ("markdown", """\
The top hits are usually:

- **`FINAL_APPROVED → REJECTED by MISSING`** with **hour-of-day spread
  ~2332h** — declarations rejected at 5 am wait 50× longer than at
  5 pm. Probably a queue / batch effect.
- **`SUBMITTED → REJECTED by ADMIN`** with **day-of-week spread ~122h**
  — Friday rejections wait three times longer than Saturday. Weekly
  queue dynamics.

These are concrete operational levers a process owner can act on.
The previous version of this analysis only reported "where" the
bottlenecks were; the v0.4 audit added the *why*.

## 3. Set the bar — `gnn baseline`

The 1st-order Markov baseline is the floor any deep model has to beat
on top-1 accuracy. It's simply: for the current task, predict the most
likely next task seen in training.
"""),
    ("code", """\
out_dir = Path("/tmp/tutorial_baseline")
subprocess.run(
    ["gnn", "baseline", str(LOG), "--out-dir", str(out_dir)],
    check=True,
)
baseline_run = sorted(out_dir.glob("run_*"))[-1]
baseline_metrics = json.loads(
    (baseline_run / "metrics" / "baseline_metrics.json").read_text()
)
print(json.dumps(baseline_metrics, indent=2))
"""),
    ("markdown", """\
- **Most-common baseline: 22.0%** — predicting the global mode.
- **Markov 1st-order: 85.4%** — surprisingly high, because most BPI
  2020 transitions are near-deterministic given the current task.

That 85.4% is the bar. Holding the deep model to it is one of the
load-bearing scientific commitments of this repo — every model run
records `lift_over_markov` so you can tell at a glance whether the
deep model is adding signal or just inheriting class imbalance.

## 4. Train the LSTM — or read the published metrics

The full training command is:

```bash
gnn run input/BPI2020_DomesticDeclarations.csv \\
    --seed 42 --device cpu \\
    --epochs-lstm 30 --hidden-dim 256 --lr-lstm 5e-4 \\
    --predict-time --skip-gat --skip-rl
```

~8 minutes on Apple silicon MPS, ~30 minutes on CPU. The output JSON
files are committed under `bench/published/bpi2020_lstm/metrics/` so
you can read the headline numbers without retraining.
"""),
    ("code", """\
lstm_metrics = json.loads((PUBLISHED / "lstm_metrics.json").read_text())

print("Headline metrics on BPI 2020 val set:")
for key in ("accuracy", "top_3_accuracy", "top_5_accuracy", "mrr",
            "ece_after_calibration", "dt_mae_hours",
            "lift_over_markov", "lift_over_most_common"):
    print(f"  {key:28s} {lstm_metrics[key]:.4f}")
"""),
    ("markdown", """\
## 5. Read the numbers honestly

Top-1 alone is the wrong target on a 17-class workflow. The metrics
that match how the model is actually deployed (ranked candidates,
calibrated confidence, time prediction) tell a much stronger story:

| metric | Markov | LSTM | margin |
|---|---:|---:|---:|
| top-1     | **85.4%** | 81.8% | Markov +3.6 |
| top-3     | 85.4% | **97.1%** | **LSTM +11.7** |
| MRR       | 0.854 | **0.898** | **LSTM +0.044** |
| ECE       | — | **0.011** | LSTM has it |
| dt MAE    | — | **48 h** | LSTM has it |

Markov collapses to a single argmax per current-task — its top-3 is
identical to its top-1. The LSTM's top-3 = 97.1% means the right next
event is in its top 3 guesses 97 times out of 100. For a workflow tool
that surfaces a shortlist to a reviewer, that's a single-digit miss
rate vs Markov's 14.6 %.

Three feature ablations (`+resource`, `+temporal`) all left the top-1
unchanged at ~81.8% — see the README for the full ablation table. The
top-1 gap is a property of BPI 2020's deterministic majority
transitions, not a missing-features problem.

## 6. Explain a single prediction — `gnn explain`

The GAT exposes attention weights via `forward_with_attention`. For a
given case, `gnn explain` dumps:

- per-event predicted next-task with probability
- top-5 attended predecessors per event (which past events drove the
  prediction)
- a heatmap PNG of last-layer attention across the case's events

Requires a trained GAT model. The command:

```bash
gnn explain input/BPI2020_DomesticDeclarations.csv \\
    --case-id "<case_id>" \\
    --model results/run_<latest>/models/best_gnn_model.pth \\
    --hidden-dim 64 --gat-heads 4 --gat-layers 2
```

Skip if you don't have a trained GAT — the LSTM is the production
model in this repo, and `gnn explain` is GAT-specific because only
the GAT exposes per-event attention.

## 7. Roll the model forward — `gnn predict-suffix`

Single-event prediction tells you what *might* happen next; the
deployment question is *"given this case in progress, what does the
rest look like?"*. Beam search from any prefix:

```bash
gnn predict-suffix input/BPI2020_DomesticDeclarations.csv \\
    --case-id "<case_id>" \\
    --model results/run_<latest>/models/lstm_next_activity.pth \\
    --hidden-dim 256 --predict-time \\
    --prefix-len 2 --beam 5 --max-steps 8
```

Returns ranked continuations with joint probability and total
predicted cycle time. Top continuation might be `Request Payment →
SUBMITTED → FINAL_APPROVED → ...` with `total_dt_hours = 80.7` at
`p = 0.35`.

## 8. Counterfactual resource swap — `gnn whatif`

Honest framing: this is **empirical** counterfactual estimation, not
causal inference. The sequence model takes only task IDs (it doesn't
condition on resource), so swapping resources at the model level is a
no-op. Instead `whatif` re-scores the case against the per-(transition,
resource) historical mean wait derived from the same log.

```bash
gnn whatif input/BPI2020_DomesticDeclarations.csv \\
    --case-id "declaration 86791" \\
    --swap-resource "STAFF MEMBER=SYSTEM"
```

Output (real example):

```
Actual total wait: 23.75 h
Counterfactual:    98.67 h
Delta:            +74.92 h (2 of 2 transitions used fallback estimates)
```

The "fallback used" flag is honest: when a `(transition, target_resource)`
cell has zero historical support, the tool falls back to the
transition-wide mean and tells you. Decision support, not magic.

## 9. Compare two runs — `gnn diff`

Hyperparameter exploration in two commands:

```bash
gnn run input/log.csv --epochs-lstm 10 --out-dir results/short
gnn run input/log.csv --epochs-lstm 30 --out-dir results/long

gnn diff results/short/run_<ts>/ results/long/run_<ts>/ --out diff.md
```

The markdown report shows every metric that moved with both raw and
percentage deltas. Real example output snippet:

```
- lstm_metrics.json.accuracy:  0.444 → 0.778 (Δ +0.333, +75.0%)
- lstm_metrics.json.macro_f1:  0.155 → 0.489 (Δ +0.335, +216.2%)
- lstm_metrics.json.per_class.Resubmit.f1: 0 → 1 (Δ +1, —)
```

The per-class breakdown reveals which transitions the extra training
unlocked — visible in `gnn diff` but invisible in a single-number
summary.

## 10. Deploy outside Python — `gnn export onnx`

Once a run looks good, serialize the trained sequence model to ONNX
so it can run in any ONNX Runtime environment (Rust, Java, browser,
mobile) without a Python interpreter. Bridge to the v0.7 Rust
orchestrator milestone in GOALS.md.

```bash
gnn export onnx results/run_<ts>/
# Writes:
#   results/run_<ts>/models/lstm.onnx
#   results/run_<ts>/models/lstm.onnx.meta.json   <- input/output schema
```

After export, inference from non-Python:

```python
# This is the same pattern any ONNX Runtime caller uses (Rust /
# Java / browser bind to the same C++ kernels under the hood).
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("results/run_.../models/lstm.onnx")
# Inputs follow the meta.json contract — `x` and `seq_len` here:
prefix = np.array([[1, 2, 3]], dtype=np.int64)        # shape (1, T)
seq_len = np.array([3], dtype=np.int64)               # shape (1,)
logits, dt_pred = sess.run(None, {"x": prefix, "seq_len": seq_len})
```

The export round-trip is asserted in CI: `max(|torch −
onnxruntime|) < 1e-4` per logit on a held-out batch
(`tests/test_export.py::test_onnx_export_roundtrip`).

## 11. What's reproducible

- Every JSON file behind every README claim lives at
  `bench/published/<dataset>/metrics/`.
- `--seed 42` is the canonical seed; the CI canary
  (`tests/test_canary.py`) re-runs `gnn smoke --device cpu --seed 42`
  on every push and asserts every metric is within tolerance of a
  committed `tests/golden_smoke_metrics.json`.
- `gnn diff` lets you compare any two runs by JSON.

## 11. Where to go next

- **Your own log.** `gnn run my_log.csv` or `gnn run my_log.xes`. The
  pipeline accepts both natively.
- **Multi-dataset comparison.** See `bench/datasets/README.md` for the
  registry (BPI 2012/2017/2019, Sepsis); v0.5 milestone is to publish
  cross-dataset rows.
- **The audit history.** `CHANGELOG.md` has the v0.2 → v0.4 audit
  story (case-level splits, calibration, conformance F, etc.) — useful
  if you want to see what was wrong before and why it's right now.

That's the whole `gnn` story end to end. Most of what's not in this
notebook (transformer head, suffix beam search, counterfactual
estimator, `gnn serve`) is in the README's CLI section.
"""),
]


def _cell(cell_type: str, source: str, idx: int) -> dict:
    """Construct a Jupyter cell dict with the given source.

    The ``id`` field is required by nbformat ≥5.1 and warned about by
    earlier versions; we derive it from ``idx`` so the build is
    deterministic.
    """
    base = {
        "cell_type": cell_type,
        "id": f"cell-{idx:02d}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def build_notebook() -> dict:
    return {
        "cells": [_cell(t, s, i) for i, (t, s) in enumerate(CELLS)],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
                "codemirror_mode": {"name": "ipython", "version": 3},
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out = Path(__file__).parent / "01_bpi2020_tutorial.ipynb"
    out.write_text(json.dumps(build_notebook(), indent=1) + "\n")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
