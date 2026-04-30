"""Cross-seed aggregation logic.

Validates that ``_aggregate`` produces the right mean / std / min /
max / n / values shape, including the edge cases (single seed, mixed
metric presence)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Load bench/seeds.py without making it a package import.
SPEC = importlib.util.spec_from_file_location(
    "bench_seeds", Path(__file__).resolve().parents[1] / "bench" / "seeds.py"
)
seeds_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seeds_mod)  # type: ignore[union-attr]


def test_aggregate_basic_three_seeds():
    payloads = [
        {"accuracy": 0.81, "top_3_accuracy": 0.97, "n": 100},
        {"accuracy": 0.82, "top_3_accuracy": 0.97, "n": 100},
        {"accuracy": 0.83, "top_3_accuracy": 0.96, "n": 100},
    ]
    out = seeds_mod._aggregate(payloads)
    assert set(out.keys()) >= {"accuracy", "top_3_accuracy", "n"}
    assert abs(out["accuracy"]["mean"] - 0.82) < 1e-9
    assert out["accuracy"]["min"] == 0.81
    assert out["accuracy"]["max"] == 0.83
    assert out["accuracy"]["n"] == 3
    # Expected std for [0.81, 0.82, 0.83] is 0.01 by sample std (N-1).
    assert abs(out["accuracy"]["std"] - 0.01) < 1e-9


def test_aggregate_skips_keys_missing_from_some_runs():
    """A metric that's only present in run 0 must not appear in the
    aggregate (we only summarize keys present in *every* run, so std
    is comparable)."""
    payloads = [
        {"accuracy": 0.8, "ece": 0.01},
        {"accuracy": 0.81},  # missing ece
    ]
    out = seeds_mod._aggregate(payloads)
    assert "accuracy" in out
    assert "ece" not in out


def test_aggregate_handles_single_seed():
    """N=1 → std=0.0 instead of crashing on statistics.stdev."""
    out = seeds_mod._aggregate([{"accuracy": 0.5}])
    assert out["accuracy"]["std"] == 0.0
    assert out["accuracy"]["n"] == 1


def test_aggregate_flattens_nested_per_class():
    """per_class.<task>.f1 should appear as a leaf metric with std."""
    payloads = [
        {"per_class": {"Submit": {"f1": 0.9, "support": 100}}},
        {"per_class": {"Submit": {"f1": 0.92, "support": 100}}},
    ]
    out = seeds_mod._aggregate(payloads)
    assert "per_class.Submit.f1" in out
    assert abs(out["per_class.Submit.f1"]["mean"] - 0.91) < 1e-9


def test_aggregate_empty_input():
    assert seeds_mod._aggregate([]) == {}
