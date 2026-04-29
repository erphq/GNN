#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Graph Attention Network for next-task prediction.

The default mode is **node-level** prediction: each event in a case is a
node, and the model emits a next-task distribution at every node. This
matches the standard next-event-prediction formulation used in the
process-mining literature.

A legacy **graph-level** mode is preserved for reproducing the v0.2
configuration: events are pooled with `global_mean_pool` and a single
prediction is made per case, supervised by the modal next-task across
the case. Pass `node_level=False` to opt in.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class NextTaskGAT(nn.Module):
    """Graph Attention Network for next-task prediction.

    Parameters
    ----------
    input_dim, hidden_dim, output_dim, num_layers, heads, dropout :
        Standard GAT hyperparameters.
    node_level :
        If True (default), forward returns per-node logits with shape
        ``(total_nodes, output_dim)``. If False, forward applies
        ``global_mean_pool`` and returns one logit row per graph
        (legacy v0.2 behavior).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        heads=4,
        dropout=0.5,
        node_level: bool = True,
    ):
        super().__init__()
        self.node_level = node_level
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
            )
        self.fc = nn.Linear(hidden_dim * heads, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.elu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        if not self.node_level:
            x = global_mean_pool(x, batch)
        return self.fc(x)


def train_gat_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=20,
    model_path="best_gnn_model.pth",
):
    """Train the GAT model.

    Label shape depends on ``model.node_level``:
    - node-level: ``batch_data.y`` directly (per-event next-task)
    - graph-level: ``compute_graph_label(batch_data.y, batch_data.batch)``
    """
    best_val_loss = float("inf")
    node_level = getattr(model, "node_level", False)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_data in train_loader:
            out = model(
                batch_data.x.to(device),
                batch_data.edge_index.to(device),
                batch_data.batch.to(device),
            )
            labels = _select_labels(batch_data, node_level).to(device, dtype=torch.long)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                out = model(
                    batch_data.x.to(device),
                    batch_data.edge_index.to(device),
                    batch_data.batch.to(device),
                )
                labels = _select_labels(batch_data, node_level).to(device, dtype=torch.long)
                val_loss += criterion(out, labels).item()
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")

    return model


def _select_labels(batch_data, node_level: bool):
    """Pick the right label tensor for the active head."""
    if node_level:
        return batch_data.y
    return compute_graph_label(batch_data.y, batch_data.batch)


def compute_graph_label(y, batch):
    """Modal next_task across all events of each case → graph-level label.

    Used only by the legacy graph-level head (``node_level=False``).
    """
    unique_batches = batch.unique()
    labels_out = []
    for bidx in unique_batches:
        mask = batch == bidx
        yvals_cpu = y[mask].detach().cpu()
        vals, counts = torch.unique(yvals_cpu, return_counts=True)
        labels_out.append(vals[torch.argmax(counts)])
    return torch.stack(labels_out)


def evaluate_gat_model(model, val_loader, device):
    """Run the model over the val set and return (y_true, y_pred, y_prob).

    Shape is per-node when ``model.node_level`` is True, per-graph
    otherwise. Downstream consumers (confusion matrix, accuracy, MCC)
    only see flat tensors so the same plumbing works for both heads.
    """
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    node_level = getattr(model, "node_level", False)

    with torch.no_grad():
        for batch_data in val_loader:
            logits = model(
                batch_data.x.to(device),
                batch_data.edge_index.to(device),
                batch_data.batch.to(device),
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels = _select_labels(batch_data, node_level)

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            for i in range(logits.size(0)):
                y_pred_all.append(int(preds[i]))
                y_prob_all.append(probs[i])
                y_true_all.append(int(labels[i]))

    return (
        torch.tensor(y_true_all),
        torch.tensor(y_pred_all),
        torch.tensor(y_prob_all),
    )
