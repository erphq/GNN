"""ONNX export round-trip — torch.forward must match onnxruntime.

The load-bearing property: a model exported via ``gnn export onnx``
produces output within tolerance of the original PyTorch model on
the same input. Anything beyond that (cross-runtime drift, kernel
fusion artifacts) is platform-specific and out of scope here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


def _train_quick_lstm(tmp_path: Path, csv_path: Path) -> Path:
    """Run a 1-epoch smoke pipeline so we have a real run dir to export."""
    out_dir = tmp_path / "run"
    cmd = [
        sys.executable, "-m", "gnn_cli", "run", str(csv_path),
        "--device", "cpu", "--seed", "42",
        "--epochs-gat", "1", "--epochs-lstm", "1", "--rl-episodes", "1",
        "--hidden-dim", "16",
        "--skip-rl", "--skip-analyze", "--skip-viz", "--skip-cluster",
        "--predict-time",
        "--out-dir", str(out_dir),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    subprocess.run(cmd, check=True, capture_output=True, text=True,
                   timeout=300, env=env)
    runs = list(out_dir.glob("run_*"))
    assert runs, "no run dir produced"
    return runs[0]


@pytest.fixture
def trained_run(tmp_path, csv_path):
    return _train_quick_lstm(tmp_path, Path(csv_path))


def test_arch_metadata_is_saved(trained_run: Path):
    """stage_train_lstm must write lstm_arch.json next to the .pth."""
    arch = trained_run / "models" / "lstm_arch.json"
    assert arch.exists(), f"missing arch metadata: {arch}"
    payload = json.loads(arch.read_text())
    assert payload["seq_arch"] == "lstm"
    assert "num_classes" in payload
    assert "hidden_dim" in payload
    assert "predict_time" in payload


def test_onnx_export_roundtrip(trained_run: Path, tmp_path: Path):
    """Exported ONNX must produce output matching torch within 1e-4."""
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")

    from gnn_cli.export import _ArchMeta, _load_model, export_to_onnx

    out = tmp_path / "lstm.onnx"
    written = export_to_onnx(str(trained_run), str(out), opset=17, device="cpu")
    assert written.exists()
    assert (written.with_suffix(written.suffix + ".meta.json")).exists()

    # Re-load the torch model so we can compare outputs.
    meta = _ArchMeta.from_run_dir(trained_run)
    weights = trained_run / "models" / "lstm_next_activity.pth"
    torch_model = _load_model(meta, weights, torch.device("cpu"))
    torch_model.eval()

    # Build a fresh dummy input — not reusing the export-time one.
    batch, seq_len = 3, max(meta.max_seq_len, 4)
    x = torch.randint(low=1, high=meta.num_classes + 1, size=(batch, seq_len))
    sl = torch.full((batch,), seq_len, dtype=torch.long)
    inputs: tuple = (x, sl)
    if meta.num_resources is not None:
        inputs = inputs + (
            torch.randint(low=1, high=meta.num_resources + 1, size=(batch, seq_len)),
        )
    if meta.n_continuous_dims > 0:
        inputs = inputs + (
            torch.randn(batch, seq_len, meta.n_continuous_dims),
        )

    # PyTorch reference output. The exporter wraps the model to call
    # ``inference_forward`` instead of ``forward`` (pack_padded doesn't
    # export cleanly), so we compare against that path — equivalent up
    # to FP noise.
    with torch.no_grad():
        torch_out = torch_model.inference_forward(*inputs)
    if isinstance(torch_out, tuple):
        torch_logits = torch_out[0].cpu().numpy()
    else:
        torch_logits = torch_out.cpu().numpy()

    # ONNX Runtime output.
    sess = onnxruntime.InferenceSession(
        str(written), providers=["CPUExecutionProvider"]
    )
    feed = {
        name: arr.cpu().numpy()
        for name, arr in zip(
            [i.name for i in sess.get_inputs()], inputs, strict=True,
        )
    }
    onnx_outs = sess.run(None, feed)
    onnx_logits = onnx_outs[0]

    assert torch_logits.shape == onnx_logits.shape, (
        f"shape mismatch: torch {torch_logits.shape} vs onnx {onnx_logits.shape}"
    )
    max_abs = float(np.max(np.abs(torch_logits - onnx_logits)))
    assert max_abs < 1e-4, (
        f"output drift {max_abs:.6e} exceeds 1e-4 tolerance"
    )


def test_export_cli_default_path(trained_run: Path):
    """`gnn export onnx <run_dir>` writes to <run_dir>/models/<arch>.onnx."""
    pytest.importorskip("onnx")
    cmd = [
        sys.executable, "-m", "gnn_cli", "export", "onnx", str(trained_run),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, env=env,
    )
    assert result.returncode == 0, (
        f"export failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (trained_run / "models" / "lstm.onnx").exists()


def test_export_missing_metadata_errors_clearly(tmp_path):
    """When arch metadata is absent, export raises a clear FileNotFoundError."""
    from gnn_cli.export import export_to_onnx

    fake_run = tmp_path / "fake_run"
    (fake_run / "models").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="arch.json"):
        export_to_onnx(str(fake_run), str(tmp_path / "out.onnx"))


def test_verify_passes_on_real_data(trained_run: Path, tmp_path: Path, csv_path):
    """Round-trip drift on the val split should be within tolerance."""
    pytest.importorskip("onnxruntime")

    from gnn_cli.export import (
        export_to_onnx,
        verify_onnx_against_torch,
    )

    out = trained_run / "models" / "lstm.onnx"
    export_to_onnx(str(trained_run), str(out))

    summary = verify_onnx_against_torch(
        str(trained_run), csv_path, onnx_path=str(out),
        sample_size=32, seed=42, device="cpu",
    )
    assert summary["ok"], f"verify failed: {summary}"
    for name, stats in summary["outputs"].items():
        # Real-data tolerance is 1e-3 (looser than dummy-input 1e-4).
        assert stats["max_abs_diff"] <= stats["tolerance"], (
            f"{name}: max diff {stats['max_abs_diff']} > tol {stats['tolerance']}"
        )


def test_verify_cli_action(trained_run: Path, csv_path):
    """`gnn export verify <run> --csv ...` exits 0 when within tolerance."""
    pytest.importorskip("onnxruntime")

    # Export first.
    from gnn_cli.export import export_to_onnx
    export_to_onnx(str(trained_run), str(trained_run / "models" / "lstm.onnx"))

    cmd = [
        sys.executable, "-m", "gnn_cli", "export", "verify", str(trained_run),
        "--csv", str(csv_path), "--sample-size", "16",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=180, env=env,
    )
    assert result.returncode == 0, (
        f"verify failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout
