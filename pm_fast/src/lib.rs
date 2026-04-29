//! Rust hot-path kernels for the gnn pipeline.
//!
//! Two functions ported from Python because they were the hottest pure-Python
//! loops in profiling: a per-case adjacency builder and a per-case padded
//! prefix builder. Everything else stays in PyTorch / pandas where it
//! already calls into BLAS or vectorized C code.
//!
//! Inputs are passed as flat numpy arrays sorted by `(case_id, timestamp)`
//! so we can stream through them without random access. The Python wrapper
//! is responsible for the sort.

// `clippy::useless_conversion` fires inside the `#[pyfunction]` macro
// expansion on every PyResult-returning function in pyo3 0.22; the
// macro inserts an Into::into that becomes a no-op for the same type.
// Suppressing at module scope keeps the explicit `?` calls in the
// function bodies readable.
#![allow(clippy::useless_conversion, clippy::type_complexity)]

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Build the (num_tasks, num_tasks) transition adjacency matrix.
///
/// Increments `A[task_ids[i], task_ids[i+1]]` for every consecutive pair of
/// rows that share a `case_id`. Equivalent to:
///
/// ```python
/// for cid, cdata in df.groupby("case_id"):
///     seq = cdata.sort_values("timestamp")["task_id"].values
///     for i in range(len(seq) - 1):
///         A[seq[i], seq[i+1]] += 1
/// ```
///
/// but the loop is in native code with no Python-level allocations.
#[pyfunction]
fn build_task_adjacency<'py>(
    py: Python<'py>,
    case_ids: PyReadonlyArray1<'py, i64>,
    task_ids: PyReadonlyArray1<'py, i64>,
    num_tasks: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let case_ids = case_ids.as_slice()?;
    let task_ids = task_ids.as_slice()?;
    if case_ids.len() != task_ids.len() {
        return Err(PyValueError::new_err(
            "case_ids and task_ids must have the same length",
        ));
    }

    let mut adj = Array2::<f32>::zeros((num_tasks, num_tasks));
    let n = case_ids.len();
    if n < 2 {
        return Ok(adj.into_pyarray_bound(py));
    }

    for i in 0..n - 1 {
        if case_ids[i] != case_ids[i + 1] {
            continue; // case boundary — no transition edge across it.
        }
        let src = task_ids[i];
        let dst = task_ids[i + 1];
        if src < 0 || dst < 0 {
            continue;
        }
        let (src, dst) = (src as usize, dst as usize);
        if src < num_tasks && dst < num_tasks {
            adj[[src, dst]] += 1.0;
        }
    }
    Ok(adj.into_pyarray_bound(py))
}

/// Build padded prefix tensors for LSTM training in one pass.
///
/// For every case of length `L >= 2`, emits `L - 1` prefixes:
///   prefix[i] = task_ids[..i+1], label[i] = task_ids[i+1]
///
/// Padded to `max_len` columns with zeros; task_ids are shifted by +1 so
/// `0` is reserved for padding (matches `make_padded_dataset`).
///
/// Returns `(X_padded, X_lens, Y, max_len)`.
#[pyfunction]
fn build_padded_prefixes<'py>(
    py: Python<'py>,
    case_ids: PyReadonlyArray1<'py, i64>,
    task_ids: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    usize,
)> {
    let case_ids = case_ids.as_slice()?;
    let task_ids = task_ids.as_slice()?;
    if case_ids.len() != task_ids.len() {
        return Err(PyValueError::new_err(
            "case_ids and task_ids must have the same length",
        ));
    }
    let n = case_ids.len();

    // First pass: per-case lengths to compute the global max prefix length
    // and total prefix count. case_ids must be runs of equal values (i.e.
    // input is sorted by case_id).
    let mut total_prefixes = 0usize;
    let mut max_prefix_len = 0usize;
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && case_ids[j] == case_ids[i] {
            j += 1;
        }
        let case_len = j - i;
        if case_len >= 2 {
            total_prefixes += case_len - 1;
            // Longest prefix from this case is case_len - 1.
            if case_len - 1 > max_prefix_len {
                max_prefix_len = case_len - 1;
            }
        }
        i = j;
    }

    let mut x_padded = Array2::<i64>::zeros((total_prefixes, max_prefix_len));
    let mut x_lens = Array1::<i64>::zeros(total_prefixes);
    let mut y = Array1::<i64>::zeros(total_prefixes);

    // Second pass: fill the padded matrix.
    let mut row = 0usize;
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && case_ids[j] == case_ids[i] {
            j += 1;
        }
        let case = &task_ids[i..j];
        for k in 1..case.len() {
            // Prefix is case[..k], label is case[k].
            for (col, &t) in case[..k].iter().enumerate() {
                // +1 shift so 0 is reserved for padding (matches Python).
                x_padded[[row, col]] = t + 1;
            }
            x_lens[row] = k as i64;
            y[row] = case[k];
            row += 1;
        }
        i = j;
    }

    Ok((
        x_padded.into_pyarray_bound(py),
        x_lens.into_pyarray_bound(py),
        y.into_pyarray_bound(py),
        max_prefix_len,
    ))
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_task_adjacency, m)?)?;
    m.add_function(wrap_pyfunction!(build_padded_prefixes, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
