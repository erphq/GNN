# Project Goals

A working draft of where this project is going and why. Edit freely — the
intent is to keep one source of truth that prospective contributors and
future-us can read in a few minutes.

## Mission

Bring rigorous, reproducible process mining to enterprise event logs by
combining classical process-mining algorithms (PM4Py) with modern
graph-based and sequence learning (GAT, LSTM) and lightweight RL —
packaged as a CLI you can drop on real data and trust.

## Why this exists

Off-the-shelf process-mining tools are descriptive but not predictive;
off-the-shelf ML doesn't respect process structure. Bridging the two
requires honest data handling (no future leakage, stable label spaces,
train-only scaler fits) and an engineering surface that makes results
reproducible across runs and machines.

## Questions this should answer for a user

Given an event log on disk, the CLI should answer — concretely, in one
command each — the following:

- *Where are the bottlenecks?* Top-N transitions ranked by mean wait
  time, with confidence and frequency.
- *Which cases are anomalous?* Long-running cases above a configurable
  percentile, plus rare transition paths that distinguish them.
- *What happens next in this case?* Next-activity prediction with
  calibrated probabilities (GAT for structural context, LSTM for
  temporal context).
- *Does reality match the model?* Conformance checking against a
  discovered process model, with per-trace deviation scores.
- *How should resources be allocated?* An RL-derived policy mapping
  process state to a recommended (task, resource) assignment.
- *Are these activities really distinct?* Spectral clustering of the
  task adjacency graph, surfacing tasks that always co-occur.

Every public release should be able to answer all six on the BPI 2020
sample log without manual surgery.

## Near-term goals (v0.x)

1. **A reliable CLI for testing the pipeline.** Every stage runnable in
   isolation; `gnn smoke` lets a new contributor verify the full
   pipeline in under a minute on synthetic data, with no external
   datasets needed.
2. **Native XES ingestion.** Process-mining datasets ship as XES, not
   CSV. `gnn run` should accept `.xes` and `.xes.gz` directly via
   PM4Py's importer, so users don't have to pre-convert.
3. **Honest baselines on public datasets.** Reproduce numbers on BPI
   logs (BPI 2020 already in `input/`, with BPI 2012 / 2017 / 2019 as
   stretch targets), each with a documented seed and split so anyone
   can verify.
4. **Correctness over speed.** Continue closing audit findings
   (case-level splits, train-only scaler fits, normalized Laplacian,
   RL state-space sizing). Tests guard each fix.
5. **Stable I/O contracts.** Run outputs land under
   `results/run_<timestamp>/{models,visualizations,metrics,analysis,policies}`
   with versioned JSON / Parquet schemas, so downstream consumers
   (dashboards, follow-up analyses) don't break across versions.
6. **ONNX export for trained models.** A `gnn export` subcommand that
   serializes the trained GAT / LSTM to ONNX. This is the bridge that
   lets a future Rust orchestrator (or any non-Python consumer) run
   inference without a Python interpreter.

## Quality bar

The codebase commits to:

- **Deterministic runs.** A given `(data, seed, config)` triple
  produces the same metrics on the same hardware, and within a stated
  tolerance across CPUs / GPUs / Apple Silicon. Regressions in
  determinism are bugs.
- **Type-clean public API.** Anything imported from `gnn_cli`,
  `models`, `modules`, or `visualization` carries type hints; mypy
  passes on those packages in strict-ish mode (gradually tightened).
- **Lint-clean tree.** Ruff passes on push; the rule set widens over
  time (currently `E` + `F`, with `B` / `UP` / `SIM` queued).
- **Tests are the documentation of intent.** Every correctness fix
  ships with a regression test; CI runs them on Python 3.10 and 3.11.
- **No silent fallbacks.** Errors fail loudly with a useful message
  and a non-zero exit code; warnings are not the place for
  correctness signals.

## Longer-term goals

1. **Rust orchestration layer.** Once the Python CLI surface
   stabilizes, port the orchestrator (arg parsing, stage scheduling,
   artifact management) to Rust for a single-binary distribution. ML
   kernels stay in Python via subprocess or PyO3 until there's a clear
   case for porting them.
2. **Pluggable miners and learners.** Decouple the pipeline from
   PM4Py / PyG specifics so alternative back-ends (e.g. a Rust process
   miner, a different GNN library) can be swapped in.
3. **Streaming / incremental mode.** Today the pipeline is batch-only;
   production event logs arrive incrementally. The long-term goal is
   window-based incremental mining and online model updates.
4. **Citable benchmarks.** Curate a benchmark suite (BPI + synthetic
   + private when/if available) with leaderboard-style reporting, so
   methodology improvements have measurable impact.

## Non-goals

- Yet another general-purpose deep-learning framework. The point is
  process mining; ML is in service of that.
- Exotic visualization (3D, real-time dashboards). Static PNGs and an
  HTML Sankey are enough for v0.x.
- Cloud-native multi-tenant SaaS. This is a library plus CLI;
  productization is a downstream concern.
- Beating SOTA on next-event prediction in isolation. The framing is
  multi-task — prediction *and* discovery *and* optimization, none in
  isolation.

## Milestones

Concrete checkpoints, in rough dependency order. Tick each off only
when the named acceptance criterion holds.

- **v0.3 — Methodological audit (✅ shipped).** Causal forward-only
  GAT edges, node-level head, time-to-next-event multi-task head,
  post-hoc temperature scaling, conformance F-score, RL state-vec fix,
  XES alias collision fix.
- **v0.4 — Reproducibility, baselines, ergonomics (✅ shipped).**
  Null + Markov baselines with lift-over-baseline reporting, LSTM
  parity (calibration + time head), `gnn explain <case_id>` attention
  saliency, per-class P/R/F1, reference-metrics canary in CI, native
  `.xes` / `.xes.gz` ingest, Markov-chain smoke generator, TOML
  `--config` file, `gnn diff <run_a> <run_b>`.
- **v0.5 — Multi-dataset benchmark (🟡 partial).** BPI 2012 / 2017 /
  2019 each have a one-line invocation, a published metrics table,
  and a regression test that flags drift > 1pp on next-activity
  accuracy. *Status:* registry + downloader live
  (`bench/datasets/`); parameterized drift regression test scaffold
  is in (`tests/test_dataset_drift.py`); awaiting dataset pinning to
  activate. The 4TU portal requires interactive TOS acceptance per
  dataset, so this is a one-time manual step.
- **v0.6 — ONNX export (✅ shipped).** `gnn export onnx <run_dir>`
  produces inference artifacts that load in Rust / Java / browser
  without a Python interpreter. ONNX Runtime round-trip is asserted
  in `tests/test_export.py`.
- **v0.7 — Rust orchestrator (prototype).** Single static binary that
  shells into Python for ML kernels but owns CLI parsing, config
  loading, scheduling, and artifact management. Same surface as
  today's `gnn` CLI; same outputs byte-for-byte.
- **v0.8 — GAT v2: heterogeneous graph.** Today's GAT under-performs
  the LSTM on every dataset we have, but it's the only model that
  exposes per-event attention (used by `gnn explain`). The honest fix
  is to expand the graph from "events as nodes" to a heterogeneous
  graph with **events, resources, and cases as separate node types**
  with typed edges (event → event chronological, event → resource
  assigned-to, event → case belongs-to). PyG's `HeteroData` supports
  it. Acceptance: GAT v2 closes the gap to the LSTM on top-3 / MRR on
  BPI 2020, AND keeps `gnn explain` working with attention over the
  richer graph. Until then GAT v1 stays in the codebase as the
  attention surface; we don't drop it.
- **v1.0 — Stable.** All of the above, type-clean, deterministic,
  documented, and tagged. Breaking changes after v1.0 require a
  deprecation cycle.

## How to evaluate progress

A quarterly check: can a new contributor (a) clone, (b) `pip install
-e .`, (c) run `gnn smoke` and `gnn run input/BPI2020_DomesticDeclarations.csv`,
and (d) get the same numbers we publish — all in under 30 minutes? If
any step regresses, that's the bug to fix first.
