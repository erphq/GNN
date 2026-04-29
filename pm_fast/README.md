# pm_fast

Rust hot-path kernels for the gnn process-mining pipeline. Built with
[PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs).

Two kernels are exposed today:

| function                  | replaces                                                  |
|---------------------------|-----------------------------------------------------------|
| `build_task_adjacency`    | `modules.process_mining.build_task_adjacency` (per-case loop)  |
| `build_padded_prefixes`   | `_build_prefixes` + `make_padded_dataset` (fused, one pass)    |

Both keep the original Python contracts; the parent package transparently
prefers the Rust kernels and falls back to pure-Python when the extension
isn't installed (so you can still `pip install -r requirements.txt` and
work without a Rust toolchain).

## Build & install (from the repo root)

```bash
pip install maturin
cd pm_fast
maturin develop --release       # builds + installs into the active env
cd .. && python bench/bench_hotpaths.py
```

`maturin develop` is the equivalent of `pip install -e .` for a Rust
extension — fastest dev loop. For wheels (CI / publishing) use
`maturin build --release`.

## Why these two

A `cProfile` run of `gnn run` on BPI2020 puts the per-case Python loops
at >40% of pre-training wall time. The two functions above are the
densest of those loops; both translate to tight integer arithmetic in
Rust with zero Python-level allocations. Numbers in the parent
README's "Performance" section.
