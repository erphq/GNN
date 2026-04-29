# Contributing

Thanks for picking up an issue or proposing a change.

## Local setup

```bash
git clone https://github.com/erphq/gnn
cd gnn

python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Tests

```bash
pytest -q
```

The suite is fast (synthetic event log, no heavy training). Anything that
touches the pipeline's data layout (preprocessing, splits, graph build)
should ship with a regression test.

The most load-bearing test is
`tests/test_models.py::test_prefixes_are_case_isolated` — it guards the
fix for the original prefix-shuffle data leak. **Do not relax it.**

## Lint

```bash
ruff check .
```

CI runs both on every push / PR.

## Commit messages

- Imperative mood, short subject, optional body.
- No AI-attribution trailers (`Co-Authored-By: Claude`, etc.) — this repo
  doesn't carry tool credit in commit metadata.

## Methodology guardrails

A few non-obvious invariants this codebase relies on:

1. **Case-level splits, never prefix-level.** Splitting prefixes randomly
   leaks future events of a case into training. Use
   `data_preprocessing.split_cases` and pass the resulting halves into
   `prepare_sequence_data`.
2. **Scaler / Normalizer fit on train only.** `fit_feature_scaler(train)`
   then `apply_feature_scaler(both_halves, scaler)`. Encoders
   (`LabelEncoder` for tasks/resources) are fit on the full dataframe so
   the label space is stable across the split.
3. **Spectral clustering uses the normalized Laplacian + `eigh`.** The
   unnormalized Laplacian + `np.linalg.eig` returned complex types and
   degraded clusters when degrees were skewed.
4. **`modules.utils.set_seed` covers `random`, `numpy`, `torch` (CPU +
   CUDA), and `cudnn.deterministic`.** Don't reach for
   `torch.manual_seed` directly — call the helper.
