# Process Mining with Graph Neural Networks

[![ci](https://github.com/erphq/gnn/actions/workflows/ci.yml/badge.svg)](https://github.com/erphq/gnn/actions/workflows/ci.yml)
[![docker](https://github.com/erphq/gnn/actions/workflows/docker.yml/badge.svg)](https://github.com/erphq/gnn/actions/workflows/docker.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![rust](https://img.shields.io/badge/rust-stable-orange.svg)](./pm_fast)

A research codebase combining **Graph Attention Networks**, **LSTMs**, **classical process mining**, and **tabular RL** for next-event prediction and process analysis on event-log data.

## Authors

- **Somesh Misra** — [@mathprobro](https://x.com/mathprobro)
- **Shashank Dixit** — [@protosphinx](https://x.com/protosphinx)
- **Research group:** [ERP.AI Research](https://www.erp.ai)

## What's in here

```
.
├── input/                       # sample event logs (BPI2020 included)
├── gnn_cli/                     # `gnn` CLI: argparse, stage orchestration, smoke generator
│   ├── cli.py                   #   subcommands: run / analyze / cluster / smoke / version
│   ├── stages.py                #   pipeline stages, each callable in isolation
│   └── smoke.py                 #   synthetic event-log generator
├── models/
│   ├── gat_model.py             # Graph Attention Network for next-task prediction
│   └── lstm_model.py            # LSTM next-activity model
├── modules/
│   ├── data_preprocessing.py    # encoders, scalers, case-level split, PyG graphs
│   ├── process_mining.py        # bottlenecks, conformance, transitions, spectral
│   ├── rl_optimization.py       # tabular Q-learning over (task, resource) actions
│   └── utils.py                 # set_seed(), pick_device()
├── visualization/
│   └── process_viz.py           # confusion matrix, Sankey, transition heatmap, ...
├── pm_fast/                     # Rust hot-path kernels (PyO3, optional 500×+ speedup)
├── bench/                       # benchmark scripts (Python vs Rust)
├── tests/                       # pytest suite, synthetic event log fixtures
├── .github/workflows/           # ci.yml (ruff + pytest + Rust), docker.yml (ghcr)
├── pyproject.toml               # project metadata + `gnn` script + ruff + pytest config
└── main.py                      # legacy entry point (delegates to gnn_cli)
```

## Capability surface

- **Process analysis** — bottleneck detection by mean wait time, 95th-percentile cycle-time outlier flagging, rare-transition discovery, conformance checking via inductive miner + token replay.
- **Models** — Graph Attention Networks (PyG `GATConv`) for structural learning over per-case event graphs; LSTM with packed sequences for prefix-conditioned next-activity prediction.
- **Clustering** — spectral clustering on the task-transition adjacency (normalized Laplacian + `eigh`).
- **Optimization** — tabular Q-learning agent choosing `(next_task, resource)` actions in a `ProcessEnv`.
- **Visualization** — confusion matrix, cycle-time histogram, NetworkX process flow with bottleneck overlay, transition-probability heatmap, Plotly Sankey of full process flow.

## Installation

```bash
git clone https://github.com/erphq/gnn
cd gnn

python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch  # or a CUDA wheel
pip install -r requirements.txt
pip install -e .            # installs the `gnn` CLI entry point
```

Optionally, build the Rust hot-path kernels for a 500×+ speedup on the
per-case loops (`build_task_adjacency`, prefix builder for the LSTM):

```bash
pip install maturin
cd pm_fast && maturin develop --release && cd ..
```

The pipeline auto-detects the extension and falls back to pure Python if
it isn't installed, so this step is genuinely optional. See [pm_fast/README.md](./pm_fast/README.md) for details.

## Data format

CSV event log with one row per event:

| column      | type      | description                                  |
|-------------|-----------|----------------------------------------------|
| `case_id`   | string    | process instance identifier                  |
| `task_name` | string    | activity / event name                        |
| `timestamp` | datetime  | event time (any pandas-parseable format)     |
| `resource`  | string    | resource / actor that performed the event    |
| `amount`    | numeric   | numerical attribute, optional but expected   |

XES-style column names (`case:concept:name`, `concept:name`,
`time:timestamp`, `org:resource`, `case:Amount`) are accepted and
auto-renamed.

## Usage

The repo ships a `gnn` CLI with subcommands. After `pip install -e .`:

```bash
gnn smoke                                  # synthetic data, ~1 minute end-to-end (no real data needed)
gnn run input/BPI2020_DomesticDeclarations.csv
gnn analyze input/BPI2020_DomesticDeclarations.csv   # process-mining stats only
gnn cluster input/BPI2020_DomesticDeclarations.csv   # spectral clustering only
```

`gnn run --help` lists every flag. The most useful ones:

```bash
gnn run input/BPI2020_DomesticDeclarations.csv \
  --epochs-gat 30 --epochs-lstm 10 \
  --hidden-dim 128 --gat-heads 8 \
  --val-frac 0.2 --seed 42 \
  --skip-rl                                # any stage can be skipped: --skip-{gat,lstm,analyze,viz,cluster,rl}
```

The legacy entry point still works for backward compatibility:

```bash
python main.py input/BPI2020_DomesticDeclarations.csv
```

Output goes to `results/run_YYYYMMDD_HHMMSS/`:

```
results/run_…/
├── models/             # GAT and LSTM weights (best-val for GAT)
├── visualizations/     # PNGs + HTML Sankey
├── metrics/            # JSON: per-stage metrics, scaler mode, seed, splits
├── analysis/           # process-mining outputs
└── policies/           # learned RL policy
```

### Exit codes

- `0` — success
- `2` — usage error (bad args / value out of range)
- `3` — data error (missing file, bad columns, unparseable timestamps)
- `4` — runtime error (model / training crash)

## Methodology notes (v0.2 audit)

The v0.2 changes fix three correctness issues that affected published
numbers:

1. **Case-level train/val split.** The previous LSTM pipeline shuffled
   *prefixes* before the 80/20 split, so future events of the same case
   ended up in both halves. Splits now happen on `case_id` first; prefixes
   are then built only within each half. See
   `tests/test_models.py::test_prefixes_are_case_isolated`.
2. **Train-only scaler fit.** `MinMaxScaler` / `Normalizer` are now fit
   on the training rows only, then applied to the validation rows. The
   label encoders for tasks / resources still fit on the full dataframe
   (a stable label space is needed across splits).
3. **Normalized Laplacian for spectral clustering.** The unnormalized
   Laplacian + `np.linalg.eig` returned complex eigenvectors and produced
   unstable clusters when task degrees were skewed. We now symmetrize the
   adjacency, build the normalized Laplacian, and use `np.linalg.eigh`
   (real, ascending, faster).

A reproducibility helper `modules.utils.set_seed` seeds `random`,
`numpy`, `torch` (CPU + CUDA), and toggles `cudnn.deterministic`.

## Methodology notes (v0.3 audit)

Three further fixes, each with a regression test:

1. **Node-level GAT head (default).** The GAT now predicts next-task at
   every event (`shape=(total_nodes, num_classes)`) instead of pooling
   to a single graph-level prediction supervised by the modal next-task.
   The legacy graph-level head is kept behind `--gat-graph-label` for
   reproducing v0.2 numbers; see
   `tests/test_models.py::test_gat_forward_runs_node_level` and the
   `_graph_level` companion test.
2. **RL state-vector sizing.** `ProcessEnv._get_state` previously sized
   the one-hot state by `len(all_tasks)` (the subset of task-ids that
   appear as a transition source after `dropna(next_task)`) but indexed
   it with `current_task` from the full label space. When some tasks
   only appear as terminal events the index went out of bounds. Sized
   by `len(le_task.classes_)` now.
3. **Priority-based XES alias rename.** BPI logs ship both `case:id`
   and `case:concept:name`, and the previous rename map collapsed both
   to `case_id`, producing a duplicate-named column that broke
   `df["case_id"]` access. The loader now picks the highest-priority
   alias and drops collisions.

## Performance (Rust hot paths)

`pm_fast` ports the two pure-Python loops that dominated wall-clock time
on real BPI logs (per-case adjacency increment, per-case prefix
expansion). Numbers from `bench/bench_hotpaths.py` on a synthetic
event log of 5,000 cases / 37,651 rows on M-series Apple silicon:

| function                   | Python    | Rust        | speedup |
|----------------------------|-----------|-------------|---------|
| `build_task_adjacency`     | 396.31 ms | **0.67 ms** | **588×** |
| `build_padded_prefixes`    | 396.27 ms | **0.78 ms** | **505×** |

Reproduce: `python bench/bench_hotpaths.py --num-cases 5000 --repeats 5`.

The kernels are wired in transparently — `modules.process_mining.build_task_adjacency`
and the LSTM stage in `gnn_cli.stages` prefer the Rust kernel when
`pm_fast` is importable and fall through to Python otherwise. CI builds
the extension and runs a parity test that asserts byte-identical output
against the reference implementation (`tests/test_fast_kernels.py`).

GAT and LSTM training itself stays in PyTorch — those are already
calling into BLAS / cuDNN, so there's nothing for Rust to win there.
This is the "Option A" in the design discussion: profile, then push
only the genuinely hot Python loops down to native code.

## Tests

```bash
pytest -q
```

Covers preprocessing, splits, scaler fit/apply, both model forward+backward
passes, the RL env contract, and a known-bipartite-graph regression test
for spectral clustering. Synthetic event-log fixture in
`tests/conftest.py`; no external data needed.

## Citation

```bibtex
@software{GNN_ProcessMining,
  author    = {Misra, Somesh and Dixit, Shashank},
  title     = {Process Mining with Graph Neural Networks},
  year      = {2025},
  publisher = {ERP.AI},
  url       = {https://github.com/erphq/gnn}
}
```

## License

MIT — see [LICENSE](./LICENSE).
