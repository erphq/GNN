"""FastAPI inference endpoint — `gnn serve --run-dir <path>`.

Loads a trained sequence model + the encoders that produced it, and
exposes two endpoints:

- ``POST /predict`` — top-K next-event prediction for a given prefix
  of activity names. Returns ranked candidates with calibrated
  probabilities (after temperature scaling) and a per-step time
  estimate when the model has the time-prediction head.
- ``POST /predict_suffix`` — beam-search rollout from the prefix.
  Returns up to ``beam`` continuations with joint probabilities and
  cumulative predicted cycle time.

Both endpoints accept JSON like::

    {"prefix": ["Submit", "Review"], "k": 3}            # /predict
    {"prefix": ["Submit", "Review"], "beam": 5,
     "max_steps": 20}                                   # /predict_suffix

The serve command depends on ``fastapi`` and ``uvicorn`` — added as
optional extras in ``pyproject.toml`` so the core pipeline stays
deploy-free.
"""

from __future__ import annotations

from pathlib import Path

import torch

try:
    from pydantic import BaseModel, Field

    class PredictRequest(BaseModel):
        prefix: list[str] = Field(..., description="Activity names in order.")
        k: int = Field(3, ge=1, le=50, description="Top-K to return.")

    class PredictSuffixRequest(BaseModel):
        prefix: list[str]
        beam: int = Field(5, ge=1, le=20)
        max_steps: int = Field(20, ge=1, le=100)
        stop_on_self_loop: bool = True
except ImportError:  # pydantic optional — `gnn serve` advertises the install
    PredictRequest = None  # type: ignore[misc]
    PredictSuffixRequest = None  # type: ignore[misc]


def build_app(
    run_dir: str,
    data_path: str,
    *,
    seq_arch: str = "lstm",
    hidden_dim: int = 64,
    predict_time: bool = False,
    seed: int = 42,
):
    """Construct a FastAPI app that loads the trained model from ``run_dir``.

    The encoders aren't separately saved (yet — that's a v0.5 follow-up),
    so we re-run preprocessing on ``data_path`` to recover the same
    label encoder. As long as the dataset is identical to training,
    encoder labels are deterministic.
    """
    from fastapi import Body, FastAPI, HTTPException

    from gnn_cli.stages import stage_preprocess
    from gnn_cli.suffix import predict_suffix
    from models.lstm_model import NextActivityLSTM
    from models.transformer_model import NextActivityTransformer
    from modules.utils import pick_device, set_seed

    set_seed(seed)
    device = pick_device(None)

    # Re-derive encoders from the same data the model was trained on.
    df, _, _, le_task, _, _, _ = stage_preprocess(data_path, val_frac=0.2, seed=seed)
    label_to_id = {str(name): int(i) for i, name in enumerate(le_task.classes_)}
    id_to_label = {i: str(name) for i, name in enumerate(le_task.classes_)}
    num_classes = len(le_task.classes_)

    # Find the model file under run_dir/models/.
    models_dir = Path(run_dir) / "models"
    candidates = [
        models_dir / "transformer_next_activity.pth",
        models_dir / "lstm_next_activity.pth",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"no sequence-model checkpoint under {models_dir}"
        )
    arch_from_path = "transformer" if "transformer" in model_path.name else "lstm"
    if arch_from_path != seq_arch:
        raise ValueError(
            f"--seq-arch {seq_arch} but checkpoint is {arch_from_path}: {model_path}"
        )

    model: torch.nn.Module
    if seq_arch == "transformer":
        model = NextActivityTransformer(
            num_classes, emb_dim=hidden_dim, hidden_dim=hidden_dim,
            predict_time=predict_time, max_len=512,
        ).to(device)
    else:
        model = NextActivityLSTM(
            num_classes, emb_dim=hidden_dim, hidden_dim=hidden_dim,
            num_layers=1, predict_time=predict_time,
        ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load calibrated temperature if present.
    cal_path = models_dir / f"{seq_arch}_calibration.pt"
    temperature = 1.0
    if cal_path.exists():
        temperature = float(
            torch.load(cal_path, weights_only=True).get("temperature", 1.0)
        )

    app = FastAPI(
        title="gnn process-mining inference",
        version="0.4.x",
        description="Next-event + suffix prediction on event-log prefixes.",
    )

    def _encode_prefix(activities: list[str]) -> list[int]:
        try:
            return [label_to_id[a] for a in activities]
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"unknown activity: {e.args[0]}; "
                       f"valid labels: {sorted(label_to_id)[:20]}…",
            ) from e

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "seq_arch": seq_arch,
            "num_classes": num_classes,
            "temperature": temperature,
            "predict_time": predict_time,
            "model_path": str(model_path),
        }

    @app.post("/predict")
    def predict(req: PredictRequest = Body(...)):
        prefix_ids = _encode_prefix(req.prefix)
        if not prefix_ids:
            raise HTTPException(400, detail="prefix must be non-empty")
        x = torch.tensor([[t + 1 for t in prefix_ids]], dtype=torch.long, device=device)
        sl = torch.tensor([len(prefix_ids)], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x, sl)
        if isinstance(out, tuple):
            logits, dt_pred = out
            dt_seconds = float(torch.expm1(dt_pred[0]).item())
        else:
            logits, dt_seconds = out, None
        probs = torch.softmax(logits[0] / temperature, dim=0)
        topk = torch.topk(probs, min(req.k, num_classes))
        candidates = [
            {"activity": id_to_label[int(i)], "probability": float(p)}
            for p, i in zip(topk.values.tolist(), topk.indices.tolist(), strict=True)
        ]
        return {
            "prefix": req.prefix,
            "candidates": candidates,
            "next_dt_hours": dt_seconds / 3600.0 if dt_seconds is not None else None,
        }

    @app.post("/predict_suffix")
    def predict_suffix_endpoint(req: PredictSuffixRequest = Body(...)):
        prefix_ids = _encode_prefix(req.prefix)
        if not prefix_ids:
            raise HTTPException(400, detail="prefix must be non-empty")
        completions = predict_suffix(
            model, prefix_ids,
            beam=req.beam, max_steps=req.max_steps,
            stop_on_self_loop=req.stop_on_self_loop, device=device,
        )
        return {
            "prefix": req.prefix,
            "completions": [
                {
                    "rank": i,
                    "tasks": [id_to_label[t] for t in seq[len(prefix_ids):]],
                    "log_prob": logp,
                    "prob": prob,
                    "total_dt_hours": dt / 3600.0,
                }
                for i, (seq, logp, dt, prob) in enumerate(completions, 1)
            ],
        }

    return app


def serve(
    run_dir: str,
    data_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    seq_arch: str = "lstm",
    hidden_dim: int = 64,
    predict_time: bool = False,
    seed: int = 42,
) -> None:
    """Build the app and start uvicorn."""
    import uvicorn

    app = build_app(
        run_dir, data_path,
        seq_arch=seq_arch, hidden_dim=hidden_dim,
        predict_time=predict_time, seed=seed,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
