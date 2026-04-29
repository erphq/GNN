"""TOML --config overrides hardcoded defaults; CLI flags still win."""

from __future__ import annotations

from pathlib import Path

from gnn_cli.cli import _load_toml_run_config, build_parser


def test_loads_top_level_keys(tmp_path):
    p = tmp_path / "gnn.toml"
    p.write_text("epochs_gat = 30\nhidden_dim = 128\n")
    cfg = _load_toml_run_config(str(p))
    assert cfg["epochs_gat"] == 30
    assert cfg["hidden_dim"] == 128


def test_loads_run_table(tmp_path):
    p = tmp_path / "gnn.toml"
    p.write_text("[run]\nepochs_gat = 7\nseed = 99\n")
    cfg = _load_toml_run_config(str(p))
    assert cfg["epochs_gat"] == 7
    assert cfg["seed"] == 99


def test_main_two_pass_applies_toml_then_overrides_with_cli(tmp_path, monkeypatch):
    """Run main(): TOML supplies seed=7, CLI passes seed=42 → CLI wins."""
    from gnn_cli.cli import main

    p = tmp_path / "gnn.toml"
    p.write_text("[run]\nseed = 7\nepochs_gat = 11\n")

    captured = {}

    def fake_run(args):
        captured["seed"] = args.seed
        captured["epochs_gat"] = args.epochs_gat
        return 0

    monkeypatch.setitem(__import__("gnn_cli.cli").cli.COMMANDS, "run", fake_run)
    rc = main([
        "run", "/tmp/nonexistent.csv",
        "--config", str(p),
        "--seed", "42",  # CLI override
        # epochs_gat unset → TOML value (11) should land
    ])
    assert rc == 0
    assert captured["seed"] == 42
    assert captured["epochs_gat"] == 11
