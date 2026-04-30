# gnn-rs

Rust orchestrator for the `gnn` process-mining CLI. **v0.7 milestone
foundation** per [GOALS.md](../GOALS.md).

## What this is

A single static binary (`gnn-rs`) that owns:

- **CLI parsing** — clap-derived subcommand surface mirroring the
  Python `gnn_cli` 1:1.
- **Config loading** — TOML config parser at `src/config.rs`.
- **Subprocess dispatch** — shells into `python -m gnn_cli` for the
  actual ML stages.

What it explicitly does **not** own (yet): training, inference,
artifact serialization. Those stay in Python — porting them is a v0.8+
question, not a v0.7 one. The point of the prototype is to give
downstream callers a Rust binary they can drop into a cargo / shell
environment without re-implementing the orchestration logic.

## Build & install

```bash
cd gnn_rs
cargo build --release
# Binary at: target/release/gnn-rs
```

To use the binary with the Python ML stages, the `gnn_cli` package
needs to be importable somewhere `python -m gnn_cli` can find. The
bridge resolves the interpreter in this order:

1. `GNN_PYTHON` env var (explicit override)
2. `VIRTUAL_ENV/bin/python` if a venv is active
3. `python3` on `$PATH`

## Usage

```text
$ gnn-rs --help
Single static binary that orchestrates the gnn process-mining pipeline.

Usage: gnn-rs <COMMAND>

Commands:
  version          Print the gnn-rs version and the underlying Python gnn_cli version.
  run              Run the full pipeline on an event-log CSV / XES.
  analyze          Process-mining stats only (bottlenecks, conformance, drivers).
  cluster          Spectral clustering on the task adjacency.
  baseline         Score the null + Markov baselines on a CSV / XES.
  explain          Per-case attention saliency for a trained GAT model.
  predict-suffix   Beam-search rollout from a case prefix.
  whatif           Counterfactual resource swap.
  diff             Compare two run directories' metrics.
  export           Export a trained model to ONNX / verify ONNX vs PyTorch.
  serve            FastAPI inference endpoint.
  smoke            Synthetic data + abbreviated full pipeline end-to-end.
```

Every subcommand forwards its trailing args to `python -m gnn_cli`
verbatim — `gnn-rs run input/log.csv --epochs-lstm 30 ...` is
behaviorally identical to `gnn run input/log.csv --epochs-lstm 30 ...`.
For now `gnn-rs <subcmd> --help` does **not** print the Python flag
table; run the Python CLI directly for that.

## Testing

```bash
cargo test
```

Three integration tests assert the binary exposes every subcommand
name and exits zero for `gnn-rs version`. Two unit tests in
`config.rs` assert the TOML parser handles both `[run]` tables and
bare top-level keys.

## Roadmap (v0.7 → v1.0)

Once the prototype is stable and the Python contract is locked, the
v1.0 plan is to incrementally absorb capabilities into Rust:

1. **Run-dir management** in Rust — directory layout, metric JSON
   schemas, atomic writes. Python no longer needs to own
   `setup_results_dir`.
2. **Config validation** in Rust — type-checked merging of TOML +
   CLI flags, surfaces errors before launching any training.
3. **Stage scheduling** in Rust — DAG of stages with parallel
   execution where possible (analyze + cluster don't depend on each
   other).
4. **PyO3 entry points** instead of subprocess for the per-stage
   call — eliminates the per-call Python startup cost.

ML kernels themselves stay in Python through all of this. Porting
torch / pm4py / pyG to Rust is a separate, much larger conversation.
