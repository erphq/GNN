# Notebooks

End-to-end walkthroughs.

## `01_bpi2020_tutorial.ipynb`

The full `gnn` workflow on the BPI 2020 Domestic Declarations log: load
→ analyze → baseline → train → read metrics → explain → predict suffix
→ what-if → diff. ~30 seconds if you read the published metrics, ~30
minutes if you retrain end-to-end.

## Running

From the repo root, after `pip install -e .`:

```bash
pip install jupyter nbformat
jupyter lab notebooks/
```

The notebook expects to be opened from `notebooks/`; `REPO` resolves
to the parent directory.

## Editing

The notebook is **generated** from `build.py`. Edit cell content
there (Python source-of-truth, easy to diff and lint), then rebuild:

```bash
python notebooks/build.py
```

Don't hand-edit the `.ipynb` JSON — your changes will be overwritten
on the next rebuild. This pattern keeps the notebook content reviewable
in PRs without committing notebook execution noise (timestamps, kernel
state, image bytes).

## What the notebook doesn't do

- **No pre-executed outputs.** The committed `.ipynb` has empty cell
  outputs by design; the reader runs the cells themselves and sees
  fresh numbers, or just reads the static markdown rendering on
  GitHub. This keeps PR diffs clean.
- **No retraining by default.** The notebook reads pinned metrics from
  `bench/published/bpi2020_lstm/metrics/` so the headline numbers show
  up without waiting for training. The `gnn run` command is shown as
  a code block (not executed) for users who want the full loop.
