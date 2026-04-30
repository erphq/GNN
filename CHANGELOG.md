# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] — 2026-04-30

Honest scientific framing + richer model surface. The headline change
is that the README now leads with the metrics where the LSTM
*dominates* Markov (top-3 +11.7 pp, MRR +0.044, ECE 0.011) instead of
chasing top-1 — a property of the dataset, not the model. Three
feature ablations (`--use-resource`, `--use-temporal`) plus a
cross-seed variance harness confirmed the gap on top-1 is within seed
noise and not feature-engineerable on BPI 2020.

### Added

- **End-to-end Jupyter tutorial** (`notebooks/01_bpi2020_tutorial.ipynb`)
  walking the full BPI 2020 workflow: load → analyze → baseline →
  read pinned metrics → explain → predict-suffix → whatif → diff. The
  notebook reads `bench/published/bpi2020_lstm/metrics/` so a reader
  gets the headline numbers in 30 seconds without retraining. Source-
  of-truth in `notebooks/build.py`.
- **Resource embedding** in the LSTM (`--use-resource`). Parallel
  embedding concatenated with the task embedding before the LSTM. No-
  op on BPI 2020 (only 2 resources) but lands the infrastructure for
  richer logs.
- **Cyclic temporal features** (`--use-temporal`). Per-event sin/cos
  of day-of-week and hour-of-day as continuous LSTM input. Also no-op
  on BPI 2020 top-1; included for the same infrastructure reasons.
- **Quantile time head** (`--time-quantiles "0.1,0.5,0.9"`). Replaces
  the MSE point estimate with a K-output head trained with pinball
  loss; eval reports interval coverage and width alongside median MAE.
  "Will finish in 80 h with 80 % confidence" instead of "in 80 h."
- **Cross-seed variance harness** (`bench/seeds.py`). Runs N seeds in
  sequence, aggregates per-seed metrics into mean ± std / min / max
  per leaf metric. The number that tells you whether two leaderboard
  rows differ meaningfully or just by training noise.
- **Property tests for the model layer** (`tests/test_properties.py`):
  top-K accuracy is monotone in K; temperature scaling is argmax-
  invariant; ECE is in [0, 1]; LSTM forward shape is invariant under
  feature flags.

### Changed

- README leaderboard reframed: top-3, MRR, ECE, dt MAE as the headline
  numbers; an honest "what we tried, what didn't help" subsection
  documents the BPI 2020 feature ablation.
- README CLI section now documents every v0.4+ flag and the non-`run`
  subcommands (`baseline`, `explain`, `predict-suffix`, `whatif`,
  `diff`, `serve`) plus the `bench/seeds.py` cross-seed harness.

### Notes

- No backwards-incompatible changes. Every new flag defaults to off /
  empty / no-op so existing callers (predict-suffix, serve, the canary)
  are unaffected.
- The cross-seed sweep on BPI 2020 (5 × 30 epochs) lands the canonical
  variance numbers under `bench/published/bpi2020_lstm_seeds/` —
  pinned for reproducibility once the sweep completes.

## [0.4.0] — 2026-04-29

Reproducibility, baselines, and ergonomics. Every accuracy number now
has a baseline to compare against; every full pipeline run is
canary-tested in CI; XES logs work natively; both models can be
calibrated and predict times.

### Added

- `gnn baseline <csv>` — null (most-common-next-task) + 1st-order Markov
  baseline; reports `markov_accuracy`, `markov_coverage`,
  `most_common_accuracy`. The full pipeline now records
  `lift_over_markov` in `gat_metrics.json` and `lstm_metrics.json`
  so model accuracy is compared against the trivial floor.
- `gnn explain <csv> --case-id <id> [--model <path>]` — runs the
  trained GAT on a single case with `forward_with_attention`, dumps
  per-event predictions + attended-predecessor weights as JSON and
  a heatmap PNG. Loads `--model` if given, otherwise trains a small
  demo GAT on the spot.
- `gnn diff <run_a> <run_b>` — markdown delta report for two run
  directories. Numbers, dicts, and clustering changes are surfaced.
- LSTM parity with the GAT: optional time-to-next-event regression
  head, post-hoc temperature scaling, ECE-before-after reporting.
  Multi-task path falls back to the Python prefix builder when
  `--predict-time` is set (Rust hot path doesn't carry dt yet).
- Per-class precision / recall / F1 / support + macro F1 + weighted F1
  in both GAT and LSTM metric files.
- Reference-metrics canary (`tests/test_canary.py`) — runs
  `gnn smoke --device cpu --seed 42` in a subprocess and compares every
  metric to `tests/golden_smoke_metrics.json` with per-key tolerances.
- Native XES ingest (`.xes`, `.xes.gz`) via PM4Py's importer — every
  public BPI-style log now works directly with `gnn run log.xes`.
- Markov-chain smoke generator: a real process graph with branching
  (Approve / Reject) and bounded loops (Reject → Resubmit → Review).
  Markov baseline 96.4% on the new generator, vs the random-skip
  generator that gave both models ~20% with no signal to learn.
- TOML `--config` file with two-pass argparse (CLI > TOML > defaults).
- `gnn version`, `gnn --version`.

### Changed

- `perform_conformance_checking` now returns
  `(replayed, summary_dict)` with `num_deviant`, `fitness`, `precision`,
  `f_score`. Token-replay deviance count alone is misleading on
  flower-model logs; this surfaces the precision gap that calls them
  out (BPI 2020: fitness 1.0, precision 0.247).
- `evaluate_gat_model` accepts a `temperature` kwarg.
- `stage_train_lstm` now takes `le_task` so per-class metric keys are
  real task names instead of integer indices.

### Fixed

- LSTM `evaluate` previously emitted a "torch.tensor of list of numpy
  arrays is extremely slow" warning; stack via numpy first.

## [0.3.0] — 2026-04-29

Methodological audit. Three correctness fixes that affected published
numbers, each shipped with a regression test.

### Added

- Causal forward-only edges in `build_graph_data` (default). Combined
  with default GATConv self-loops, node *j*'s representation depends
  only on `{j-K, …, j-1, j}` after K layers — strictly past + present.
  `--gat-bidirectional` opt-in for v0.2 reproducibility.
- Node-level GAT head (default): predict next-task at every event.
  `--gat-graph-label` opt-in for v0.2 reproducibility.
- Optional time-to-next-event regression head on the GAT
  (`--predict-time`). Joint loss, val-set MAE in hours.
- Post-hoc temperature scaling on the GAT classifier. ECE before/after
  recorded; `--no-calibrate` to skip.
- `--split-mode temporal`: order cases by start time, take last
  `val_frac` as val. Surfaces drift random splits hide.
- `pm_fast/` Rust hot-path kernels (PyO3 + maturin). 588× / 505×
  speedup on `build_task_adjacency` and the LSTM prefix builder.
  Auto-detected; falls back to Python when not built.

### Changed

- `data_preprocessing.encode_categoricals` adds `dt_seconds` and
  `dt_log` columns alongside `next_task`.

### Fixed

- `ProcessEnv._get_state` was sized by `len(all_tasks)` (post-dropna
  source-task subset) but indexed with `current_task` from the full
  label space. Tasks that only appear as terminal events crashed RL.
  Sized by `len(le_task.classes_)` now.
- BPI logs ship both `case:id` and `case:concept:name`; the previous
  rename map collapsed both to `case_id`, producing a duplicate-named
  column. Priority-based rename + collision drop.

## [0.2.0] — 2026-04 (audit-and-improvements branch merged)

First public audit. Three correctness issues that affected published
numbers; full `gnn` CLI as the user-facing surface.

### Added

- `gnn` CLI with subcommands: `run`, `analyze`, `cluster`, `smoke`,
  `version`. `python main.py <csv>` legacy entry point preserved as
  a thin shim.
- Pytest suite (case-isolated split, scaler fit-on-train-only, both
  model forward+backward passes, RL contract, spectral on a known
  bipartite graph). CI on Python 3.10 + 3.11.
- `modules.utils.set_seed` + `pick_device` reproducibility helpers.

### Fixed

- **Case-level train/val split.** The previous LSTM pipeline shuffled
  *prefixes* before the 80/20 split, so future events of the same case
  ended up in both halves. Splits now happen on `case_id` first.
- **Train-only scaler fit.** `MinMaxScaler` / `Normalizer` are fit on
  training rows only.
- **Normalized Laplacian + `eigh`** for spectral clustering. The
  unnormalized Laplacian + `np.linalg.eig` returned complex
  eigenvectors and produced unstable clusters when task degrees were
  skewed.

## [0.1.0]

Initial release. Process mining with GNN + LSTM + RL + spectral
clustering on event logs. Single-script `python main.py <csv>` entry.
