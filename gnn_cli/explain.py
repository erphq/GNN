"""Single-case explanation: which past events drove each prediction.

For a given (csv, case_id) and a trained GAT, this module:

1. Re-runs preprocessing on the CSV with the same seed, so the label
   space and feature scaler match what the model was trained on.
2. Builds the PyG graph for that case alone.
3. Runs the model with ``forward_with_attention`` to capture per-layer
   attention weights.
4. Writes a JSON summary (per-event task, predicted next task,
   probability, top attended predecessors) and an attention heatmap PNG.

The attention shown is from the **last** GATConv layer, averaged over
heads — that's the layer immediately before classification, so its
attention is closest to "what did the model pay attention to when
deciding this prediction?".
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

from models.gat_model import NextTaskGAT
from modules.data_preprocessing import build_graph_data


def explain_case(
    df,
    case_id: str,
    model: NextTaskGAT,
    le_task,
    out_dir: str,
    device: torch.device,
    causal: bool = True,
    temperature: float = 1.0,
) -> dict:
    """Run the model on a single case and dump JSON + PNG. Returns the JSON dict."""
    case_df = df[df["case_id"] == case_id].copy()
    if case_df.empty:
        raise ValueError(f"case_id={case_id!r} not found in dataframe")

    graphs = build_graph_data(case_df, causal=causal)
    if not graphs:
        raise ValueError(f"case_id={case_id!r} produced no graph data")
    g = graphs[0]
    batch = Batch.from_data_list([g]).to(device)

    model.eval()
    with torch.no_grad():
        out = model.forward_with_attention(batch.x, batch.edge_index, batch.batch)
    if model.predict_time:
        logits, _dt_pred, attentions = out
    else:
        logits, attentions = out

    probs = torch.softmax(logits / temperature, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()

    # Last-layer attention, averaged across heads.
    edge_index_last, alpha_last = attentions[-1]
    alpha_mean = alpha_last.mean(dim=1).numpy()  # (num_edges,)

    # Per-event records, in chronological order.
    events = []
    sorted_case = case_df.sort_values("timestamp").reset_index(drop=True)
    for i in range(len(g.x)):
        # Find edges that point to node i (target=i); record the source
        # nodes and their attention weights as "what the model attended
        # to for predicting after event i".
        mask = edge_index_last[1] == i
        srcs = edge_index_last[0][mask].tolist()
        weights = alpha_mean[mask.numpy()].tolist()
        attended = sorted(zip(srcs, weights, strict=True), key=lambda kv: -kv[1])

        true_next = int(g.y[i].item())
        pred_next = int(preds[i])
        events.append(
            {
                "index": i,
                "task_name": str(sorted_case.iloc[i]["task_name"]),
                "true_next_task": str(le_task.inverse_transform([true_next])[0]),
                "pred_next_task": str(le_task.inverse_transform([pred_next])[0]),
                "pred_probability": float(probs[i, pred_next]),
                "attended_predecessors": [
                    {
                        "src_index": int(s),
                        "src_task": str(sorted_case.iloc[s]["task_name"]),
                        "weight": float(w),
                    }
                    for s, w in attended[:5]
                ],
            }
        )

    summary = {
        "case_id": str(case_id),
        "num_events": int(len(events)),
        "events": events,
        "model_node_level": bool(model.node_level),
        "model_predict_time": bool(model.predict_time),
        "temperature": float(temperature),
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"explain_{case_id}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Heatmap: attention(target → source) for last layer, mean over heads.
    n = len(g.x)
    heat = np.zeros((n, n), dtype=np.float32)
    for src, tgt, w in zip(
        edge_index_last[0].tolist(), edge_index_last[1].tolist(), alpha_mean.tolist(),
        strict=True,
    ):
        heat[tgt, src] = w

    fig, ax = plt.subplots(figsize=(max(4, n * 0.4), max(4, n * 0.4)))
    im = ax.imshow(heat, cmap="viridis", aspect="auto")
    ax.set_xlabel("source event (attended to)")
    ax.set_ylabel("target event (predicting after)")
    ax.set_title(f"Last-layer attention · case {case_id}")
    ticks = list(range(n))
    labels = [str(sorted_case.iloc[i]["task_name"])[:8] for i in range(n)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="attention weight (mean over heads)")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"explain_{case_id}.png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return summary


def load_or_train_gat(
    train_df,
    val_df,
    le_task,
    device,
    model_path: str | None,
    cfg,
) -> NextTaskGAT:
    """Either load weights from disk or train a quick GAT for explanation.

    The CLI offers ``--model <path>`` so a previously-trained model can
    be reused. Without a path, a fresh small GAT is trained for one or
    two epochs purely so ``gnn explain`` works standalone — call it the
    "demo" path. Production users should always pass --model.
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from models.gat_model import train_gat_model
    from modules.data_preprocessing import compute_class_weights

    num_classes = len(le_task.classes_)
    model = NextTaskGAT(
        input_dim=5,
        hidden_dim=cfg.hidden_dim,
        output_dim=num_classes,
        num_layers=cfg.gat_layers,
        heads=cfg.gat_heads,
        dropout=0.5,
        node_level=cfg.gat_node_level,
        predict_time=cfg.gat_predict_time,
    ).to(device)

    if model_path and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model

    # Demo path: train a small GAT on the spot.
    train_loader = PyGDataLoader(
        build_graph_data(train_df, causal=cfg.gat_causal),
        batch_size=cfg.batch_size_gat, shuffle=True,
    )
    val_loader = PyGDataLoader(
        build_graph_data(val_df, causal=cfg.gat_causal),
        batch_size=cfg.batch_size_gat, shuffle=False,
    )
    class_weights = compute_class_weights(train_df, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_gat, weight_decay=5e-4)
    return train_gat_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=max(1, min(cfg.epochs_gat, 3)),
        model_path="/tmp/_gnn_explain_demo.pth",
        time_loss_weight=cfg.time_loss_weight,
    )
