# Canonical metric pins for the dataset drift regression test

`tests/test_dataset_drift.py` looks for `<dataset_name>.json` files
in this directory. Each pin is a copy of `lstm_metrics.json` from a
canonical training run. To add a new dataset:

```bash
# 1. Get the data (see bench/datasets/download.py --where --dataset NAME).
# 2. Run the canonical config:
gnn run bench/datasets/data/<name>.xes.gz \
    --seed 42 --device cpu \
    --epochs-lstm 30 --hidden-dim 256 --lr-lstm 5e-4 \
    --predict-time --skip-gat --skip-rl \
    --out-dir bench/results/<name>_canonical
# 3. Copy the metrics:
cp bench/results/<name>_canonical/run_*/metrics/lstm_metrics.json \
    tests/canonical_metrics/<name>.json
# 4. git add tests/canonical_metrics/<name>.json && git commit
```

Once committed, `pytest -m dataset_drift` will enforce the tolerances
in `tests/test_dataset_drift.py::TOL_FLOAT` on every push.
