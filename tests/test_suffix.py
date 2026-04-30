"""Beam-search suffix prediction.

Two contracts to lock:

1. The first beam is always the greedy (argmax-each-step) trajectory.
2. With ``stop_on_self_loop=True``, a beam that doubles up the same
   task two steps in a row terminates early instead of running to
   ``max_steps``.
3. The within-beam normalized probabilities sum to 1.0.
"""

from __future__ import annotations

import torch

from gnn_cli.suffix import predict_suffix
from models.lstm_model import NextActivityLSTM


def test_beam_one_equals_greedy_argmax(synthetic_event_log):
    """With beam=1, the only surviving path *must* be stepwise argmax —
    that's how a beam-of-one degenerates to greedy decoding."""
    from modules.data_preprocessing import encode_categoricals

    df, le_task, _ = encode_categoricals(synthetic_event_log)
    model = NextActivityLSTM(num_cls=len(le_task.classes_), emb_dim=8, hidden_dim=8)
    completions = predict_suffix(
        model, prefix=[0], beam=1, max_steps=4, stop_on_self_loop=False,
    )
    seq, _, _, _ = completions[0]

    # Reproduce greedy manually.
    greedy = [0]
    model.eval()
    for _ in range(4):
        x = torch.tensor([[t + 1 for t in greedy]], dtype=torch.long)
        sl = torch.tensor([len(greedy)], dtype=torch.long)
        with torch.no_grad():
            logits = model(x, sl)
        greedy.append(int(logits[0].argmax().item()))
    assert seq == greedy


def test_normalized_probabilities_sum_to_one(synthetic_event_log):
    from modules.data_preprocessing import encode_categoricals

    df, le_task, _ = encode_categoricals(synthetic_event_log)
    model = NextActivityLSTM(num_cls=len(le_task.classes_), emb_dim=8, hidden_dim=8)
    completions = predict_suffix(model, prefix=[0, 1], beam=4, max_steps=3)
    total = sum(p for _, _, _, p in completions)
    assert abs(total - 1.0) < 1e-6


def test_stop_on_self_loop_terminates_early(synthetic_event_log):
    """If every beam emits the same task twice in a row we stop early."""
    from modules.data_preprocessing import encode_categoricals

    class StubModel(torch.nn.Module):
        def __init__(self, num_cls):
            super().__init__()
            self.num_cls = num_cls
            self.predict_time = False
            self.node_level = True
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def eval(self):
            return self

        def forward(self, x, seq_len):
            # Always rank class 0 at the top — every beam will repeat 0.
            logits = torch.full((x.shape[0], self.num_cls), -1e9)
            logits[:, 0] = 0.0
            return logits

    df, le_task, _ = encode_categoricals(synthetic_event_log)
    model = StubModel(num_cls=len(le_task.classes_))
    completions = predict_suffix(
        model, prefix=[0], beam=2, max_steps=10, stop_on_self_loop=True,
    )
    # Without self-loop stopping the sequence would run to length 1+10=11.
    # With stopping, it terminates after 2 consecutive zeros → length ≤ 4.
    seq = completions[0][0]
    assert len(seq) <= 4, f"expected early stop, got len={len(seq)}: {seq}"


def test_rejects_empty_prefix():
    import pytest
    model = NextActivityLSTM(num_cls=4, emb_dim=4, hidden_dim=4)
    with pytest.raises(ValueError, match="prefix"):
        predict_suffix(model, prefix=[], beam=2, max_steps=3)
