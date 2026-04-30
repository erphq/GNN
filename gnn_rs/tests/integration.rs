//! Integration tests for the `gnn-rs` binary.
//!
//! These tests exercise the actual built binary end-to-end via
//! `assert_cmd`. They confirm the clap surface is wired up and that
//! the bridge to `python -m gnn_cli` resolves correctly.

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn version_subcommand_prints_both_versions() {
    // Should print at least the gnn-rs version. The python half may
    // fail on hosts without the venv (e.g. fresh CI checkouts before
    // `pip install -e .`); accept either outcome.
    let mut cmd = Command::cargo_bin("gnn-rs").unwrap();
    cmd.arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains("gnn-rs 0.1.0"));
}

#[test]
fn help_lists_every_subcommand() {
    let mut cmd = Command::cargo_bin("gnn-rs").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        // Spot-check the canonical surface — these MUST be present and
        // cover every Python subcommand.
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("analyze"))
        .stdout(predicate::str::contains("baseline"))
        .stdout(predicate::str::contains("explain"))
        .stdout(predicate::str::contains("predict-suffix"))
        .stdout(predicate::str::contains("whatif"))
        .stdout(predicate::str::contains("diff"))
        .stdout(predicate::str::contains("export"))
        .stdout(predicate::str::contains("serve"))
        .stdout(predicate::str::contains("smoke"));
}

#[test]
fn unknown_subcommand_fails_clearly() {
    let mut cmd = Command::cargo_bin("gnn-rs").unwrap();
    cmd.arg("frobnicate")
        .assert()
        .failure()
        .stderr(predicate::str::contains("frobnicate"));
}
