//! `gnn-rs` — Rust orchestrator for the gnn process-mining CLI (v0.7).
//!
//! Owns CLI parsing, config loading, and subprocess dispatch; shells out to
//! the Python `gnn_cli` package for the actual ML stages. The point isn't
//! "rewrite the ML in Rust" — it's to give downstream callers a single
//! static binary they can drop into a Rust / cargo / shell environment
//! without a Python interpreter on $PATH for the orchestration layer.
//!
//! See [`cli`] for the subcommand surface (mirrors the Python CLI 1:1) and
//! [`bridge`] for the subprocess dispatch.

mod bridge;
mod cli;
mod config;

fn main() {
    std::process::exit(cli::run());
}
