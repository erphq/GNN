//! clap-derived subcommand surface.
//!
//! Mirrors the Python `gnn_cli.cli` argparse layout 1:1. Each subcommand
//! either:
//!
//! - dispatches to [`crate::bridge::shell_out`] which forwards to
//!   `python -m gnn_cli <args>` (everything that needs torch / pm4py /
//!   the trained models);
//! - or runs locally in Rust (just `version` for now — anything that
//!   doesn't touch the ML stack).
//!
//! The flag set per subcommand is **deliberately not enumerated** in
//! the Rust struct: clap's `trailing_var_arg` collects everything after
//! the subcommand name and we forward verbatim. This keeps the Rust
//! orchestrator from having to mirror every Python flag (and stay in
//! lockstep with their defaults). The cost is that `gnn-rs run --help`
//! doesn't print the full Python flag table — users run `gnn run --help`
//! for that. Acceptable for a v0.7 prototype.

use clap::{Parser, Subcommand};

/// Single static binary that orchestrates the gnn process-mining pipeline.
///
/// All ML stages shell out to the Python `gnn_cli` package; this binary
/// owns CLI parsing, TOML config loading, and run-dir management.
#[derive(Parser, Debug)]
#[command(name = "gnn-rs", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Print the gnn-rs version and the underlying Python gnn_cli version.
    Version,

    /// Run the full pipeline on an event-log CSV / XES.
    #[command(trailing_var_arg = true, allow_hyphen_values = true)]
    Run {
        /// All args are forwarded to `python -m gnn_cli run ...`.
        args: Vec<String>,
    },

    /// Process-mining stats only (bottlenecks, conformance, drivers).
    #[command(trailing_var_arg = true, allow_hyphen_values = true)]
    Analyze { args: Vec<String> },

    /// Spectral clustering on the task adjacency.
    #[command(trailing_var_arg = true, allow_hyphen_values = true)]
    Cluster { args: Vec<String> },

    /// Score the null + Markov baselines on a CSV / XES.
    #[command(trailing_var_arg = true, allow_hyphen_values = true)]
    Baseline { args: Vec<String> },

    /// Per-case attention saliency for a trained GAT model.
    #[command(trailing_var_arg = true, allow_hyphen_values = true)]
    Explain { args: Vec<String> },

    /// Beam-search rollout from a case prefix.
    #[command(
        name = "predict-suffix",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    PredictSuffix { args: Vec<String> },

    /// Counterfactual resource swap.
    #[command(trailing_var_arg = true, allow_hyphen_values = true)]
    Whatif { args: Vec<String> },

    /// Compare two run directories' metrics.
    #[command(trailing_var_arg = true, allow_hyphen_values = true)]
    Diff { args: Vec<String> },

    /// Export a trained model to ONNX / verify ONNX vs PyTorch.
    #[command(trailing_var_arg = true, allow_hyphen_values = true)]
    Export { args: Vec<String> },

    /// FastAPI inference endpoint.
    #[command(trailing_var_arg = true, allow_hyphen_values = true)]
    Serve { args: Vec<String> },

    /// Synthetic data + abbreviated full pipeline end-to-end.
    #[command(trailing_var_arg = true, allow_hyphen_values = true)]
    Smoke { args: Vec<String> },
}

/// Top-level entry — parse args, dispatch, return the exit code.
pub fn run() -> i32 {
    let cli = Cli::parse();
    match cli.command {
        Command::Version => {
            println!("gnn-rs {}", env!("CARGO_PKG_VERSION"));
            // Bubble up the Python-side version so users see both halves.
            match crate::bridge::shell_out_quiet(&["version".into()]) {
                Ok(stdout) => print!("python {}", stdout),
                Err(_) => println!("python gnn_cli: not on $PATH"),
            }
            0
        }
        Command::Run { args }
        | Command::Analyze { args }
        | Command::Cluster { args }
        | Command::Baseline { args }
        | Command::Explain { args }
        | Command::PredictSuffix { args }
        | Command::Whatif { args }
        | Command::Diff { args }
        | Command::Export { args }
        | Command::Serve { args }
        | Command::Smoke { args } => {
            // The clap subcommand name is consumed; we need to put it
            // back in front of the trailing args before forwarding.
            let subcommand = std::env::args().nth(1).unwrap_or_else(|| "run".to_string());
            let mut full = vec![subcommand];
            full.extend(args);
            crate::bridge::shell_out(&full).unwrap_or_else(|err| {
                eprintln!("gnn-rs: bridge error: {err}");
                4
            })
        }
    }
}
