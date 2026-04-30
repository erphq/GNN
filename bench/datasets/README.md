# Benchmark datasets

The pipeline ingests CSV (already shipped) and XES (`.xes`, `.xes.gz`)
event logs natively. This directory holds:

- **`download.py`** — fetcher for canonical 4TU mirrors. Run
  `python bench/datasets/download.py --list` to see the registry,
  `--dataset <name>` to fetch one, or `--all` to fetch all.
- **`data/`** (git-ignored) — downloaded XES files. Created on first
  run; the URLs and checksums live in `download.py`.

## Registry

| dataset | cases | tasks | description |
|---|---:|---:|---|
| `bpi_2020_domestic` | 10,366 | 17 | BPI 2020, Domestic Declarations. Shipped as CSV in `input/`; this is the XES form. |
| `bpi_2020_international` | 6,449 | 19 | BPI 2020, International Declarations. Companion to Domestic. |
| `bpi_2012` | 13,087 | 36 | BPI 2012, Loan applications. Classic NE-prediction benchmark. |
| `bpi_2017` | 31,509 | 24 | BPI 2017, Loan applications. Larger, longer cases. |
| `sepsis` | 1,050 | 16 | Hospital ER sepsis cases. Small + fast — ideal for local dev. |

## Reproducing leaderboard rows

The 4TU landing pages require interactive download (terms-of-use
acceptance), so we don't auto-fetch. To add a row:

```bash
# 1. Show the landing page URL.
python bench/datasets/download.py --where --dataset sepsis

# 2. Visit the URL, accept TOS, download the .xes.gz, drop it at
#    bench/datasets/data/sepsis.xes.gz.

# 3. Run as normal — `gnn run` accepts .xes.gz directly.
gnn run bench/datasets/data/sepsis.xes.gz \
    --seed 42 --device cpu \
    --epochs-lstm 30 --hidden-dim 256 --lr-lstm 5e-4 \
    --predict-time --skip-gat --skip-rl
```

To generate the README leaderboard table from multiple runs:

```bash
python bench/eval.py \
    --row "BPI 2020 Domestic" bench/results/bpi2020 \
    --row "Sepsis"            bench/results/sepsis \
    --out bench/leaderboard.md
```

## Licensing & redistribution

All datasets in the registry are public, mirrored on
[4TU.ResearchData](https://data.4tu.nl/). Most are released under
**CC-BY 4.0** but the exact terms vary — verify per dataset on the 4TU
landing page before redistribution. We **do not vendor** the data into
this repo (`data/` is git-ignored); the downloader is the canonical way
to get them locally.
