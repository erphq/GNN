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

import numpy as np
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
    predict_time :
        If True, attaches a regression head that emits a per-node
        log-seconds prediction for the time-to-next-event. The forward
        pass returns ``(logits, dt_pred)`` instead of just logits.
        Requires ``node_level=True`` because the time target is
        per-event.
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
        predict_time: bool = False,
    ):
        super().__init__()
        if predict_time and not node_level:
            raise ValueError(
                "predict_time=True requires node_level=True — the time "
                "delta target is per-event and cannot be supervised "
                "after global_mean_pool."
            )
        self.node_level = node_level
        self.predict_time = predict_time
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
            )
        self.fc = nn.Linear(hidden_dim * heads, output_dim)
        if predict_time:
            self.dt_head = nn.Linear(hidden_dim * heads, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.elu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        if not self.node_level:
            x = global_mean_pool(x, batch)
        logits = self.fc(x)
        if self.predict_time:
            dt_pred = self.dt_head(x).squeeze(-1)
            return logits, dt_pred
        return logits


def train_gat_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=20,
    model_path="best_gnn_model.pth",
    time_loss_weight: float = 0.5,
):
    """Train the GAT model.

    Label shape depends on ``model.node_level``:
    - node-level: ``batch_data.y`` directly (per-event next-task)
    - graph-level: ``compute_graph_label(batch_data.y, batch_data.batch)``

    When ``model.predict_time`` is True, the loss is the sum of
    cross-entropy on the next-task and ``time_loss_weight`` × MSE on
    the per-node log-seconds-to-next-event target carried on
    ``batch_data.dt``. Total val loss (used for model selection) is
    the same combined quantity, so the saved checkpoint reflects joint
    performance, not task-only.
    """
    best_val_loss = float("inf")
    node_level = getattr(model, "node_level", False)
    predict_time = getattr(model, "predict_time", False)

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
            loss = _combined_loss(
                out, labels, batch_data, criterion, predict_time,
                time_loss_weight, device,
            )

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
                val_loss += _combined_loss(
                    out, labels, batch_data, criterion, predict_time,
                    time_loss_weight, device,
                ).item()
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


def _combined_loss(out, labels, batch_data, criterion, predict_time, weight, device):
    """CE alone, or CE + weight*MSE on per-node log-dt when multi-task."""
    if not predict_time:
        return criterion(out, labels)
    logits, dt_pred = out
    ce = criterion(logits, labels)
    dt_target = batch_data.dt.to(device, dtype=torch.float)
    mse = torch.nn.functional.mse_loss(dt_pred, dt_target)
    return ce + weight * mse


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

    Side effect: when the model has a time-prediction head, populates
    ``model.last_dt_mae_hours`` with the mean absolute error in hours
    on the val set (computed by inverse-transforming the log-seconds
    target). Stages can read it back without changing the return type.
    """
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    dt_pred_all, dt_true_all = [], []
    node_level = getattr(model, "node_level", False)
    predict_time = getattr(model, "predict_time", False)

    with torch.no_grad():
        for batch_data in val_loader:
            out = model(
                batch_data.x.to(device),
                batch_data.edge_index.to(device),
                batch_data.batch.to(device),
            )
            if predict_time:
                logits, dt_pred = out
                dt_pred_all.append(dt_pred.cpu().numpy())
                dt_true_all.append(batch_data.dt.cpu().numpy())
            else:
                logits = out
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels = _select_labels(batch_data, node_level)

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            for i in range(logits.size(0)):
                y_pred_all.append(int(preds[i]))
                y_prob_all.append(probs[i])
                y_true_all.append(int(labels[i]))

    if predict_time and dt_pred_all:
        # Targets and predictions are log1p(seconds). Invert before MAE
        # so the reported number is in hours, which is what a process
        # analyst reads. expm1(x) - expm1(y) is more numerically stable
        # than np.exp(x) - np.exp(y) when targets are near zero.
        pred_hours = np.expm1(np.concatenate(dt_pred_all)) / 3600.0
        true_hours = np.expm1(np.concatenate(dt_true_all)) / 3600.0
        model.last_dt_mae_hours = float(np.mean(np.abs(pred_hours - true_hours)))

    # `torch.tensor(list_of_numpy_arrays)` walks every element through Python;
    # stacking via numpy first is ~10x faster on real-sized validation sets
    # and silences the "extremely slow" warning PyTorch emits.
    return (
        torch.tensor(y_true_all),
        torch.tensor(y_pred_all),
        torch.from_numpy(np.stack(y_prob_all)) if y_prob_all else torch.empty(0),
    )
