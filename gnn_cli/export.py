"""ONNX export — bridge to inference outside Python (v0.6 milestone).

Serializes a trained LSTM (or Transformer) from a ``run_<ts>/`` dir to
``.onnx`` so downstream consumers (Rust orchestrator, Java services,
browser inference, ONNX Runtime) can run the model without a Python
interpreter.

Reconstructs the model class from the architecture metadata
``stage_train_lstm`` writes alongside the weights
(``models/<arch>_arch.json``). On models trained before that metadata
existed, falls back to user-supplied flags.

Round-trip validation
---------------------
The exporter calls ``torch.onnx.export`` against the model's normal
``forward`` method; pack_padded_sequence does export under opset ≥17
but emits a SequenceLength + LSTM op pair. The generated graph
produces output that matches the PyTorch model within 1e-4 on a
shape-matched dummy input, asserted by ``tests/test_export.py``.
Real-world divergence beyond that comes from runtime-specific
floating-point rounding, not export bugs.

Limitations
-----------
- Quantile head and resource embedding are exported when present, but
  callers must feed inputs with the right shapes (see ``input_names``
  and ``dynamic_axes`` in the metadata sidecar).
- The transformer max_seq_len is fixed at export time (uses the
  train-set max). Inference at longer prefixes is undefined behavior;
  re-export with a larger max_seq_len if needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class _ArchMeta:
    seq_arch: str
    num_classes: int
    emb_dim: int
    hidden_dim: int
    num_layers: int
    predict_time: bool
    num_resources: int | None
    n_continuous_dims: int
    continuous_features: list[str]
    time_quantiles: list[float]
    transformer_heads: int
    max_seq_len: int

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "_ArchMeta":
        candidates = [
            run_dir / "models" / "lstm_arch.json",
            run_dir / "models" / "transformer_arch.json",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"no <arch>_arch.json under {run_dir}/models/. The model "
                f"was trained before architecture metadata was saved; "
                f"retrain or pass flags explicitly to gnn export."
            )
        data = json.loads(path.read_text())
        return cls(**data)


def _load_model(meta: _ArchMeta, weights_path: Path, device: torch.device):
    """Reconstruct the model class from arch metadata and load weights."""
    from models.lstm_model import NextActivityLSTM
    from models.transformer_model import NextActivityTransformer

    if meta.seq_arch == "transformer":
        model: torch.nn.Module = NextActivityTransformer(
            num_cls=meta.num_classes,
            emb_dim=meta.emb_dim,
            hidden_dim=meta.hidden_dim,
            num_layers=meta.num_layers,
            num_heads=meta.transformer_heads,
            predict_time=meta.predict_time,
            max_len=max(meta.max_seq_len + 8, 64),
        )
    else:
        model = NextActivityLSTM(
            num_cls=meta.num_classes,
            emb_dim=meta.emb_dim,
            hidden_dim=meta.hidden_dim,
            num_layers=meta.num_layers,
            predict_time=meta.predict_time,
            num_resources=meta.num_resources,
            n_continuous_dims=meta.n_continuous_dims,
            time_quantiles=tuple(meta.time_quantiles) or None,
        )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _make_dummy_inputs(meta: _ArchMeta, batch: int = 1) -> tuple[Any, ...]:
    """Construct a tuple of dummy inputs matching the model's forward signature."""
    seq_len = max(meta.max_seq_len, 4)
    x = torch.randint(low=1, high=meta.num_classes + 1, size=(batch, seq_len))
    sl = torch.full((batch,), seq_len, dtype=torch.long)
    inputs: tuple[Any, ...] = (x, sl)
    if meta.num_resources is not None:
        inputs = inputs + (
            torch.randint(
                low=1, high=meta.num_resources + 1, size=(batch, seq_len)
            ),
        )
    if meta.n_continuous_dims > 0:
        inputs = inputs + (
            torch.randn(batch, seq_len, meta.n_continuous_dims),
        )
    return inputs


def _input_output_names(meta: _ArchMeta) -> tuple[list[str], list[str], dict]:
    """Return (input_names, output_names, dynamic_axes) for ONNX export."""
    input_names = ["x", "seq_len"]
    if meta.num_resources is not None:
        input_names.append("x_resources")
    if meta.n_continuous_dims > 0:
        input_names.append("x_continuous")

    output_names = ["logits"]
    if meta.predict_time:
        output_names.append("dt_pred")

    # Dynamic axes: batch (axis 0) + sequence length (axis 1) for the
    # token-shaped inputs. Shapes:
    #   x, x_resources: (B, T)
    #   x_continuous:   (B, T, C)
    #   seq_len:        (B,)
    dynamic = {
        "x":        {0: "batch", 1: "seq"},
        "seq_len":  {0: "batch"},
        "logits":   {0: "batch"},
    }
    if meta.num_resources is not None:
        dynamic["x_resources"] = {0: "batch", 1: "seq"}
    if meta.n_continuous_dims > 0:
        dynamic["x_continuous"] = {0: "batch", 1: "seq"}
    if meta.predict_time:
        dynamic["dt_pred"] = {0: "batch"}

    return input_names, output_names, dynamic


def export_to_onnx(
    run_dir: str | Path,
    out_path: str | Path,
    *,
    opset: int = 17,
    device: str = "cpu",
) -> Path:
    """Serialize the trained sequence model under ``run_dir`` to ONNX.

    Returns the path to the written ``.onnx`` file. Also writes a
    sidecar ``<out>.meta.json`` with input/output names and arch info
    so downstream consumers don't have to introspect the graph.
    """
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = _ArchMeta.from_run_dir(run_dir)
    weights_path = (
        run_dir / "models"
        / f"{'transformer' if meta.seq_arch == 'transformer' else 'lstm'}"
        f"_next_activity.pth"
    )
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    dev = torch.device(device)
    model = _load_model(meta, weights_path, dev)
    dummy = tuple(t.to(dev) for t in _make_dummy_inputs(meta))
    input_names, output_names, dynamic_axes = _input_output_names(meta)

    # The training-path forward uses pack_padded_sequence for speed,
    # which ONNX's exporter handles poorly. The model exposes
    # ``inference_forward`` that's equivalent up to FP noise but
    # processes the full padded sequence — ONNX-friendly. Wrap so
    # ``torch.onnx.export`` traces that path instead.
    class _InferenceWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args):
            if hasattr(self.m, "inference_forward"):
                return self.m.inference_forward(*args)
            return self.m(*args)

    wrapped = _InferenceWrapper(model)

    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps({
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "arch": meta.__dict__,
            "opset": opset,
        }, indent=2)
    )
    return out_path
