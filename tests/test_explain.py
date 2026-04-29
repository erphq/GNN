"""Smoke test for the GAT attention-saliency path used by `gnn explain`.

Two invariants worth locking in:

1. ``forward_with_attention`` returns one ``(edge_index, alpha)`` per
   GATConv layer, with ``alpha`` shape ``(num_edges, heads)``.
2. With causal forward-only edges + GATConv self-loops (default), every
   target node attends to itself plus a strictly-earlier-or-equal
   predecessor — never to a future node.
"""

from __future__ import annotations

import torch

from models.gat_model import NextTaskGAT
from modules.data_preprocessing import (
    apply_feature_scaler,
    build_graph_data,
    encode_categoricals,
    fit_feature_scaler,
)


def test_forward_with_attention_shapes(synthetic_event_log):
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    scaler, _ = fit_feature_scaler(df)
    df = apply_feature_scaler(df, scaler)
    g = build_graph_data(df)[0]
    model = NextTaskGAT(
        input_dim=5, hidden_dim=8, output_dim=len(le_task.classes_),
        num_layers=2, heads=2, dropout=0.0,
    )
    batch = torch.zeros(g.x.shape[0], dtype=torch.long)
    logits, attns = model.forward_with_attention(g.x, g.edge_index, batch)

    assert logits.shape == (g.x.shape[0], len(le_task.classes_))
    assert len(attns) == 2  # one per GATConv layer
    for ei, alpha in attns:
        assert ei.shape[0] == 2
        assert alpha.shape == (ei.shape[1], 2)  # heads=2


def test_causal_attention_never_looks_ahead(synthetic_event_log):
    """With causal=True, every attention edge must satisfy src <= tgt."""
    df, le_task, _ = encode_categoricals(synthetic_event_log)
    scaler, _ = fit_feature_scaler(df)
    df = apply_feature_scaler(df, scaler)
    g = build_graph_data(df, causal=True)[0]
    model = NextTaskGAT(
        input_dim=5, hidden_dim=8, output_dim=len(le_task.classes_),
        num_layers=2, heads=2, dropout=0.0,
    )
    batch = torch.zeros(g.x.shape[0], dtype=torch.long)
    _, attns = model.forward_with_attention(g.x, g.edge_index, batch)

    for ei, _alpha in attns:
        # Self-loops (src == tgt) are allowed; any src > tgt is a leak.
        assert (ei[0] <= ei[1]).all().item(), \
            "causal attention leaked information from future to past"
