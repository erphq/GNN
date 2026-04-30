//! TOML config loading.
//!
//! The Python CLI already supports `--config gnn.toml` (CLI flags
//! override TOML). This module exists to give the Rust orchestrator
//! the same capability *before* shelling out — useful when a future
//! version of `gnn-rs` does its own scheduling / artifact management
//! and needs to know what the config says without round-tripping
//! through Python.
//!
//! For v0.7-prototype the Python CLI handles `--config` itself; this
//! module just exposes a parser that callers (and downstream Rust
//! code) can use.

#![allow(dead_code)]
// Phase-D foundation — these types are tested and ready to use, but
// the v0.7 prototype dispatches to Python before consuming them. The
// allow keeps the build clean while the wiring happens incrementally
// (run-dir management + stage scheduling are the next things to land
// in Rust per gnn_rs/README.md roadmap).

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Mirror of the Python ``RunConfig`` dataclass. Optional fields stay
/// `None` when absent from the TOML file; the Python CLI falls back
/// to its own defaults.
#[derive(Debug, Default, Deserialize)]
pub struct RunConfig {
    pub seed: Option<i64>,
    pub device: Option<String>,
    pub val_frac: Option<f64>,
    pub epochs_gat: Option<i64>,
    pub epochs_lstm: Option<i64>,
    pub hidden_dim: Option<i64>,
    pub gat_heads: Option<i64>,
    pub gat_layers: Option<i64>,
    pub seq_arch: Option<String>,
    pub use_resource: Option<bool>,
    pub use_temporal: Option<bool>,
    pub predict_time: Option<bool>,
    pub split_mode: Option<String>,
}

/// Top-level TOML envelope. Allows either bare keys or a `[run]` table.
#[derive(Debug, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub run: RunConfig,
}

impl Config {
    /// Read a TOML file, accepting either bare top-level keys or a
    /// `[run]` table — same convention as the Python loader.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path)
            .with_context(|| format!("reading config: {}", path.display()))?;
        // Try parsing as `[run]` table first; if no `run` section, fall
        // back to treating the whole file as the run config.
        let cfg: Config = match toml::from_str::<Config>(&raw) {
            Ok(c) if !is_empty(&c.run) => c,
            _ => {
                let inner: RunConfig = toml::from_str(&raw)
                    .with_context(|| format!("parsing config: {}", path.display()))?;
                Config { run: inner }
            }
        };
        Ok(cfg)
    }
}

fn is_empty(r: &RunConfig) -> bool {
    r.seed.is_none()
        && r.device.is_none()
        && r.val_frac.is_none()
        && r.epochs_gat.is_none()
        && r.epochs_lstm.is_none()
        && r.hidden_dim.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_run_table() {
        let raw = r#"
[run]
seed = 42
hidden_dim = 256
seq_arch = "transformer"
"#;
        let cfg: Config = toml::from_str(raw).unwrap();
        assert_eq!(cfg.run.seed, Some(42));
        assert_eq!(cfg.run.hidden_dim, Some(256));
        assert_eq!(cfg.run.seq_arch.as_deref(), Some("transformer"));
    }

    #[test]
    fn parses_bare_keys_via_from_file_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("gnn.toml");
        std::fs::write(&p, "seed = 7\nepochs_lstm = 30\n").unwrap();
        let cfg = Config::from_file(&p).unwrap();
        assert_eq!(cfg.run.seed, Some(7));
        assert_eq!(cfg.run.epochs_lstm, Some(30));
    }
}
