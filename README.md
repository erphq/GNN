# Process Mining with Graph Neural Networks

[![ci](https://github.com/erphq/gnn/actions/workflows/ci.yml/badge.svg)](https://github.com/erphq/gnn/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)

A research codebase combining **Graph Attention Networks**, **LSTMs**, **classical process mining**, and **tabular RL** for next-event prediction and process analysis on event-log data.

## Authors

- **Somesh Misra** — [@mathprobro](https://x.com/mathprobro)
- **Shashank Dixit** — [@protosphinx](https://x.com/protosphinx)
- **Research group:** [ERP.AI Research](https://www.erp.ai)

## What's in here

```
.
├── input/                       # sample event logs (BPI2020 included)
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
├── tests/                       # pytest suite, synthetic event log fixtures
├── .github/workflows/ci.yml     # ruff + pytest on push / PR
├── pyproject.toml               # project metadata + ruff + pytest config
└── main.py                      # end-to-end pipeline (CLI: argparse)
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
```

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

```bash
python main.py input/BPI2020_DomesticDeclarations.csv
```

All hyperparameters are exposed via `argparse`:

```bash
python main.py input/BPI2020_DomesticDeclarations.csv \
  --epochs-gat 30 --epochs-lstm 10 --hidden-dim 128 \
  --gat-heads 8 --val-frac 0.2 --seed 42
```

`python main.py --help` for the full list.

Output goes to `results/run_YYYYMMDD_HHMMSS/`:

```
results/run_…/
├── models/             # GAT and LSTM weights (best-val for GAT)
├── visualizations/     # PNGs + HTML Sankey
├── metrics/            # JSON: per-stage metrics, scaler mode, seed, splits
├── analysis/           # process-mining outputs
└── policies/           # learned RL policy
```

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

A known-suboptimal piece preserved for backwards compatibility: the GAT
trains on a *graph-level* label that is the modal next-task across all
events of a case (`compute_graph_label`). Switching to a node-level head
is a meaningful methodology change and is left for a follow-up PR; see
the docstring in `models/gat_model.py` for the rationale.

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
