"""Multi-step suffix prediction with beam search.

Single-event prediction tells you what *might* happen at step N+1; it
doesn't tell you what the case looks like at termination. This module
rolls the sequence model forward autoregressively from any prefix and
returns the top-B continuations ranked by joint log-probability, along
with cumulative time-to-next-event estimates so the user can read off
the predicted total remaining cycle time.

Two stop conditions:

- ``max_steps``: hard cap on rollout depth (default 20). Bounded loops
  in real workflows shouldn't exceed this; the demo synthetic generator
  caps cases at 30.
- ``stop_on_self_loop``: if the model's argmax for *every* surviving
  beam stays the same task two steps in a row, stop early. Cheap
  heuristic for "the model thinks the case is done" without requiring
  an explicit END token.

Usage::

    from gnn_cli.suffix import predict_suffix
    completions = predict_suffix(model, prefix_task_ids, beam=5, max_steps=20)
    # completions: list[(seq, log_prob, total_dt_seconds, normalized_prob)]
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def _model_output(model, x: torch.Tensor, seq_len: torch.Tensor):
    """Normalize forward() output across LSTM / Transformer / multi-task."""
    out = model(x, seq_len)
    if isinstance(out, tuple):
        # (logits, dt_pred)
        return out[0], out[1]
    return out, None


def predict_suffix(
    model,
    prefix: Sequence[int],
    *,
    beam: int = 5,
    max_steps: int = 20,
    device: torch.device = torch.device("cpu"),
    stop_on_self_loop: bool = True,
) -> list[tuple[list[int], float, float, float]]:
    """Beam-search rollout from a prefix of task_ids.

    Returns a list of up to ``beam`` candidates ordered by joint
    log-probability (best first). Each tuple is
    ``(extended_sequence, log_prob, total_dt_seconds,
    normalized_prob)``. ``extended_sequence`` includes the original
    prefix plus the predicted continuation.

    The probabilities are joint over the entire continuation — interpret
    as "this whole suffix has probability p under the model" — and the
    normalized_prob is the within-beam softmax so values across
    candidates sum to 1.0.

    Time prediction
    ---------------
    When the model has the time-prediction head, the per-step ``dt_pred``
    (in log1p-seconds) is inverted to seconds and summed across the
    rollout, giving an estimate of total remaining cycle time. Without
    the time head, ``total_dt_seconds`` is 0.0 for every beam.
    """
    if beam < 1:
        raise ValueError(f"beam must be >= 1, got {beam}")
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}")
    if not prefix:
        raise ValueError("prefix must contain at least one event")

    model.eval()
    # Each beam: (seq, log_prob, total_dt_seconds, last_argmax)
    beams: list[tuple[list[int], float, float, int]] = [
        (list(prefix), 0.0, 0.0, -1)
    ]

    for _step in range(max_steps):
        candidates: list[tuple[list[int], float, float, int]] = []
        for seq, logp, dt_total, _last in beams:
            # Tokenizer convention from make_padded_dataset: 0 is pad,
            # task_ids shift by +1.
            x = torch.tensor([[t + 1 for t in seq]], dtype=torch.long, device=device)
            seq_len = torch.tensor([len(seq)], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, dt_pred = _model_output(model, x, seq_len)
            log_probs = torch.log_softmax(logits[0], dim=0)
            top = torch.topk(log_probs, min(beam, log_probs.shape[0]))
            step_dt_seconds = (
                float(np.expm1(float(dt_pred[0].item()))) if dt_pred is not None else 0.0
            )
            for lp, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
                candidates.append(
                    (seq + [idx], logp + lp, dt_total + step_dt_seconds, idx)
                )
        # Keep top-B candidates by joint log probability.
        candidates.sort(key=lambda c: -c[1])
        beams = candidates[:beam]

        if stop_on_self_loop and beams:
            # Stop when the top beam's last two tokens are the same
            # task — heuristic for "model thinks it's done". Lower
            # beams may diverge but we already have the best path.
            top_seq = beams[0][0]
            if len(top_seq) >= 2 and top_seq[-1] == top_seq[-2]:
                break

    # Normalize the surviving-beam log probs so callers get a usable
    # within-beam distribution alongside the raw log probs.
    beam_logp = np.array([b[1] for b in beams], dtype=np.float64)
    if beam_logp.size == 0:
        return []
    beam_logp -= beam_logp.max()
    norm = np.exp(beam_logp)
    norm /= norm.sum()
    return [
        (seq, lp, dt, float(p))
        for (seq, lp, dt, _), p in zip(beams, norm, strict=True)
    ]


def render_suffix_report(
    completions: list[tuple[list[int], float, float, float]],
    le_task,
    prefix_len: int,
) -> str:
    """Pretty-print continuations as a markdown table."""
    lines = [
        "| rank | prob | total dt (h) | continuation |",
        "|---:|---:|---:|---|",
    ]
    for i, (seq, _logp, dt_seconds, prob) in enumerate(completions, 1):
        suffix_ids = seq[prefix_len:]
        names = [str(le_task.inverse_transform([t])[0]) for t in suffix_ids]
        lines.append(
            f"| {i} | {prob:.3f} | {dt_seconds / 3600.0:.2f} | "
            f"{' → '.join(names) if names else '—'} |"
        )
    return "\n".join(lines) + "\n"
