#!/usr/bin/env python3
"""Backward-compatible entry point — delegates to the `gnn` CLI.

Prefer `gnn <command>` (after `pip install -e .`) or `python -m gnn_cli`.
"""

from __future__ import annotations

import sys

from gnn_cli.cli import main as cli_main


def main() -> int:
    # Preserve the legacy `python main.py <csv>` invocation by injecting
    # `run` as the implicit subcommand when the first positional looks
    # like a file path rather than a known command.
    known = {"run", "analyze", "cluster", "smoke", "version", "-h", "--help", "-V", "--version"}
    argv = sys.argv[1:]
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        argv = ["run", *argv]
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
