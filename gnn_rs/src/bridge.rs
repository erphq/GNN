//! Subprocess dispatch to `python -m gnn_cli`.
//!
//! The Rust orchestrator owns CLI parsing + config loading + run-dir
//! management; everything that touches torch / pm4py / a trained model
//! is delegated to the Python package via subprocess. This module is
//! the single place that knows how to launch it.

use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};

/// Resolve the python interpreter to use for the bridge.
///
/// Order:
/// 1. `GNN_PYTHON` env var (explicit override; deployed by ops).
/// 2. The python from the active venv (`VIRTUAL_ENV/bin/python`).
/// 3. `python3` on `$PATH`.
fn resolve_python() -> Result<String> {
    if let Ok(p) = std::env::var("GNN_PYTHON") {
        return Ok(p);
    }
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        return Ok(format!("{venv}/bin/python"));
    }
    Ok("python3".to_string())
}

/// Forward a fully-formed argv (subcommand + flags) to `python -m gnn_cli`.
///
/// Stdin / stdout / stderr are inherited so the user sees Python's
/// progress bars, error tracebacks, etc. exactly as they would running
/// the Python CLI directly. The exit code is propagated.
pub fn shell_out(args: &[String]) -> Result<i32> {
    let python = resolve_python()?;
    let status = Command::new(&python)
        .arg("-m")
        .arg("gnn_cli")
        .args(args)
        .status()
        .with_context(|| {
            format!(
                "failed to spawn `{python} -m gnn_cli` — \
                 set GNN_PYTHON to override, or `pip install -e .` \
                 in the active venv"
            )
        })?;
    Ok(status.code().unwrap_or(4))
}

/// Like `shell_out` but captures stdout (used for the version probe).
///
/// Stderr is inherited so error messages still surface naturally.
pub fn shell_out_quiet(args: &[String]) -> Result<String> {
    let python = resolve_python()?;
    let mut child = Command::new(&python)
        .arg("-m")
        .arg("gnn_cli")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawn {python} -m gnn_cli"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("no stdout pipe"))?;
    let mut buf = Vec::new();
    let _ = std::io::copy(&mut std::io::BufReader::new(stdout), &mut buf);
    let status = child.wait()?;
    if !status.success() {
        return Err(anyhow!("python exited with code {:?}", status.code()));
    }
    let _ = std::io::stdout().flush();
    Ok(String::from_utf8_lossy(&buf).into_owned())
}
