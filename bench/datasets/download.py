"""Registry + helper for public process-mining benchmark datasets.

Source: 4TU.ResearchData (https://data.4tu.nl/) hosts the canonical
copies of all BPI Challenge logs and several related public event
logs. Each dataset has a landing page that requires interactive
download (terms-of-use acceptance), so we don't try to fetch them
automatically. Instead we publish:

- A registry (``REGISTRY``) of supported datasets with metadata.
- A ``where`` helper that prints the 4TU landing page URL.
- A ``locate`` helper that returns the expected local path under
  ``bench/datasets/data/<name>.xes.gz``.

Workflow::

    python bench/datasets/download.py --list             # see registry
    python bench/datasets/download.py --where sepsis     # print URL
    # → visit URL, download .xes.gz, drop in bench/datasets/data/sepsis.xes.gz
    gnn run bench/datasets/data/sepsis.xes.gz ...        # use as normal

Files land under ``bench/datasets/data/`` (git-ignored). Licensing for
each dataset follows 4TU's terms of use (CC-BY 4.0 in most cases —
verify per dataset before redistribution).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, NamedTuple

DATA_DIR = Path(__file__).parent / "data"


class Dataset(NamedTuple):
    name: str
    landing_url: str
    description: str
    size_mb: float


# Curated subset of public process-mining benchmarks. ``landing_url``
# points to the 4TU dataset landing page where the XES download lives
# behind the terms-of-use button.
REGISTRY: Dict[str, Dataset] = {
    "bpi_2020_domestic": Dataset(
        name="bpi_2020_domestic",
        landing_url="https://data.4tu.nl/articles/dataset/BPI_Challenge_2020_Domestic_Declarations/12692543",
        description="BPI Challenge 2020 — Domestic Declarations (10,366 cases, 17 tasks). "
                    "Already shipped as CSV in input/; downloads the XES variant.",
        size_mb=1.0,
    ),
    "bpi_2020_international": Dataset(
        name="bpi_2020_international",
        landing_url="https://data.4tu.nl/articles/dataset/BPI_Challenge_2020_International_Declarations/12696892",
        description="BPI Challenge 2020 — International Declarations (6,449 cases). "
                    "Companion log to Domestic; same vocabulary, different population.",
        size_mb=1.7,
    ),
    "bpi_2012": Dataset(
        name="bpi_2012",
        landing_url="https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204",
        description="BPI Challenge 2012 — Loan applications (13,087 cases, 36 tasks). "
                    "Classic next-event-prediction benchmark.",
        size_mb=22.0,
    ),
    "bpi_2017": Dataset(
        name="bpi_2017",
        landing_url="https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884",
        description="BPI Challenge 2017 — Loan applications (31,509 cases, 24 tasks). "
                    "Larger, longer cases than 2012.",
        size_mb=66.0,
    ),
    "sepsis": Dataset(
        name="sepsis",
        landing_url="https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639",
        description="Sepsis Cases — Hospital ER admissions (1,050 cases, 16 tasks). "
                    "Small, fast to run; ideal for local development.",
        size_mb=0.3,
    ),
}


def locate(name: str) -> Path:
    """Return the expected local path for dataset ``name``.

    Doesn't check existence — call ``.exists()`` if you need to know
    whether the file's been placed there.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(REGISTRY)}")
    return DATA_DIR / f"{name}.xes.gz"


def status(name: str) -> str:
    """Human-readable status: 'present' / 'missing' for a dataset."""
    p = locate(name)
    if not p.exists():
        return "missing"
    return f"present ({p.stat().st_size / 1e6:.1f} MB)"


def main() -> int:
    p = argparse.ArgumentParser(description="Manage benchmark datasets.")
    p.add_argument("--dataset", choices=sorted(REGISTRY.keys()),
                   help="Operate on a single dataset.")
    p.add_argument("--list", action="store_true",
                   help="Print the registry with local-status and exit.")
    p.add_argument("--where", action="store_true",
                   help="Print the 4TU landing-page URL for --dataset.")
    args = p.parse_args()

    if args.list:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for ds in REGISTRY.values():
            print(f"{ds.name:30s} {ds.size_mb:5.1f} MB  [{status(ds.name)}]  {ds.description}")
        print()
        print(f"local data dir: {DATA_DIR}")
        return 0

    if args.where:
        if not args.dataset:
            p.error("--where requires --dataset NAME")
        ds = REGISTRY[args.dataset]
        print(f"{ds.name}")
        print(f"  landing page: {ds.landing_url}")
        print(f"  drop the .xes.gz at: {locate(args.dataset)}")
        return 0

    p.error("pass --list, or --where --dataset NAME")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
