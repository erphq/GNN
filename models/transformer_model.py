"""Causal transformer for next-activity prediction.

Same interface as ``models.lstm_model.NextActivityLSTM`` so the rest of
the pipeline (training loop, calibration, evaluation, top-K + MRR
metrics) is shared. The only thing that changes is the encoder.

Architecture
------------
- Token embedding (``num_cls + 1``, padding_idx=0) + learned positional
  embedding (capped at ``max_len``).
- Stack of ``nn.TransformerEncoderLayer`` (GELU, batch_first) with both:
    * a causal (lower-triangular) ``attn_mask`` so position *t* never
      attends to ``t' > t`` — the same forward-only invariant the GAT
      now enforces with its edge construction.
    * a per-row key-padding mask derived from ``seq_len`` so attention
      doesn't waste capacity on padding tokens.
- Output: hidden state at position ``seq_len-1`` (the last real token),
  fed to ``fc`` → logits and optionally ``dt_head`` → time scalar.

This is intentionally small (defaults: 4 layers × 4 heads, emb_dim 64).
On BPI logs that's enough to compete with the LSTM without overfitting.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NextActivityTransformer(nn.Module):
    """Causal transformer; drop-in replacement for ``NextActivityLSTM``.

    Parameters mirror the LSTM where they have an obvious analogue:
    ``emb_dim`` is the model dimension, ``hidden_dim`` is the
    feedforward dimension (rendered as ``dim_feedforward = 4 *
    hidden_dim`` per the original Transformer paper), ``num_layers``
    is the encoder stack depth.
    """

    def __init__(
        self,
        num_cls: int,
        emb_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        predict_time: bool = False,
        max_len: int = 512,
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.predict_time = predict_time
        self.max_len = max_len
        self.emb = nn.Embedding(num_cls + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm — much more stable for short stacks
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, num_cls)
        if predict_time:
            self.dt_head = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor):
        bsz, plen = x.shape
        if plen > self.max_len:
            raise ValueError(
                f"sequence length {plen} exceeds transformer max_len={self.max_len}"
            )
        device = x.device

        positions = torch.arange(plen, device=device).unsqueeze(0).expand(bsz, plen)
        h = self.emb(x) + self.pos_emb(positions)

        # Causal mask: float('-inf') above diagonal blocks future attention.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(plen).to(device)
        # Padding mask: True where the token is pad (== position >= seq_len).
        pad_mask = (
            torch.arange(plen, device=device).unsqueeze(0) >= seq_len.to(device).unsqueeze(1)
        )
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=pad_mask)

        # Pool the hidden state at the last *real* position (seq_len - 1).
        last_idx = (seq_len.to(device) - 1).clamp(min=0).long()
        last_hidden = h[torch.arange(bsz, device=device), last_idx]

        logits = self.fc(last_hidden)
        if self.predict_time:
            dt_pred = self.dt_head(last_hidden).squeeze(-1)
            return logits, dt_pred
        return logits
