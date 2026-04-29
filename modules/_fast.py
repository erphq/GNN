"""Optional fast-path bridge.

Exposes references to the `pm_fast` Rust kernels if the extension is
installed, otherwise `None`. Consumers decide whether to use the fast
path or fall back to pure Python — keep the Python paths as the
reference implementation so the module graph stays loadable on a
toolchain-free install.
"""

from __future__ import annotations

try:
    import pm_fast as _pm_fast

    build_task_adjacency_fast = _pm_fast.build_task_adjacency
    build_padded_prefixes_fast = _pm_fast.build_padded_prefixes
    AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only on installs without Rust
    build_task_adjacency_fast = None
    build_padded_prefixes_fast = None
    AVAILABLE = False
