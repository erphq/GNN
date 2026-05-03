"""Surface tests for ``gnn_cli.cli``.

The full pipeline (`run`, `analyze`, `cluster`, `smoke`) is exercised
end to end by the existing integration tests against the synthetic
event log; this file pins the **CLI shell**: the parser shape, exit
codes, version flag, and the ``--config`` TOML loader's error path.

These tests deliberately do not exercise the heavy ``cmd_*``
handlers (those require torch / torch-geometric and a real event
log; they're already covered elsewhere). What they cover is the
contract operators script against:

- exit codes documented in ``cli.py``'s module docstring
  (``EXIT_OK = 0``, ``EXIT_USAGE = 2``, ``EXIT_DATA = 3``,
  ``EXIT_RUNTIME = 4``)
- the documented subcommand surface (``run``, ``analyze``, etc.)
- the ``--version`` and ``--help`` outputs
- the ``--config`` TOML loader's "file not found" path
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn_cli import __version__
from gnn_cli.cli import (
    EXIT_DATA,
    EXIT_OK,
    EXIT_USAGE,
    build_parser,
    cmd_version,
    main,
)


# ---------------------------------------------------------------------------
# Documented exit codes
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_documented_exit_codes_are_stable_constants(self) -> None:
        # The module docstring documents these exact values; CI / shell
        # scripts trap on them. A renumbering would silently break every
        # caller that relied on, e.g., "exit 3 means data error".
        assert EXIT_OK == 0
        assert EXIT_USAGE == 2
        assert EXIT_DATA == 3


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


class TestParserShape:
    def test_parser_has_a_command_subparser(self) -> None:
        # `args.command` must be present after parsing — main()
        # indexes COMMANDS[args.command].
        p = build_parser()
        args = p.parse_args(["version"])
        assert args.command == "version"

    @pytest.mark.parametrize(
        "subcommand",
        [
            "run",
            "analyze",
            "cluster",
            "smoke",
            "version",
            "predict-suffix",
            "whatif",
            "export",
            "serve",
            "diff",
            "explain",
            "baseline",
        ],
    )
    def test_documented_subcommand_is_registered(self, subcommand: str) -> None:
        # Every README-documented subcommand must be registered with
        # the parser. We probe via `<sub> --help` because some
        # subcommands have required positional arguments that we
        # don't want to model in this test; --help exits cleanly
        # (SystemExit(0)) on a registered subcommand and with code
        # 2 ("invalid choice") on an unregistered one. A rename is
        # a breaking change for shell users.
        p = build_parser()
        with pytest.raises(SystemExit) as excinfo:
            p.parse_args([subcommand, "--help"])
        assert excinfo.value.code == EXIT_OK, (
            f"subcommand {subcommand!r} not registered (got exit "
            f"code {excinfo.value.code})"
        )

    def test_unknown_subcommand_exits_with_usage_error(self) -> None:
        # argparse exits with SystemExit(2) on unknown subcommand;
        # pin that the convention holds (and that EXIT_USAGE matches).
        p = build_parser()
        with pytest.raises(SystemExit) as excinfo:
            p.parse_args(["definitely-not-a-real-command"])
        assert excinfo.value.code == EXIT_USAGE

    def test_no_subcommand_exits_with_usage_error(self) -> None:
        # Bare `python -m gnn_cli` should error out with usage, not
        # silently no-op or run a default.
        p = build_parser()
        with pytest.raises(SystemExit) as excinfo:
            p.parse_args([])
        assert excinfo.value.code == EXIT_USAGE


# ---------------------------------------------------------------------------
# Run-flag defaults — pin so a stealth change to a default fails here
# ---------------------------------------------------------------------------


class TestRunFlagDefaults:
    # `run` requires a `data_path` positional arg; pass a dummy
    # string (we never invoke the handler, just inspect the parsed
    # Namespace).
    def test_seed_defaults_to_42(self) -> None:
        p = build_parser()
        args = p.parse_args(["run", "/tmp/no-such-data.csv"])
        assert args.seed == 42

    def test_out_dir_defaults_to_results(self) -> None:
        p = build_parser()
        args = p.parse_args(["run", "/tmp/no-such-data.csv"])
        assert args.out_dir == "results"

    def test_val_frac_default(self) -> None:
        p = build_parser()
        args = p.parse_args(["run", "/tmp/no-such-data.csv"])
        assert args.val_frac == pytest.approx(0.2)

    def test_epochs_defaults(self) -> None:
        # Two model defaults at once; an accidental swap of the
        # GAT/LSTM defaults would silently change run cost.
        p = build_parser()
        args = p.parse_args(["run", "/tmp/no-such-data.csv"])
        assert args.epochs_gat == 20
        assert args.epochs_lstm == 5


# ---------------------------------------------------------------------------
# version subcommand
# ---------------------------------------------------------------------------


class TestVersionCommand:
    def test_cmd_version_returns_exit_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Direct call (skipping main's two-pass --config parse) to
        # isolate the command behaviour from the parser plumbing.
        p = build_parser()
        args = p.parse_args(["version"])
        rc = cmd_version(args)
        assert rc == EXIT_OK
        out = capsys.readouterr().out
        assert __version__ in out

    def test_main_version_subcommand_returns_exit_ok(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["version"])
        assert rc == EXIT_OK
        # The version string appears on stdout (not stderr).
        out = capsys.readouterr().out
        assert __version__ in out


# ---------------------------------------------------------------------------
# --config TOML loader error path
# ---------------------------------------------------------------------------


class TestConfigPath:
    def test_main_with_missing_config_file_returns_exit_data(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # The two-pass `--config` parse intercepts FileNotFoundError
        # and maps it to EXIT_DATA with a clear message; pin this so
        # a refactor doesn't accidentally swallow the error or surface
        # it as EXIT_RUNTIME (which would mis-classify a config typo
        # as a model failure in operator dashboards).
        missing = tmp_path / "no-such.toml"
        rc = main(["--config", str(missing), "version"])
        assert rc == EXIT_DATA
        err = capsys.readouterr().err
        assert "config file not found" in err
        assert str(missing) in err
