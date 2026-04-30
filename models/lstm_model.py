#!/usr/bin/env python3

"""LSTM next-activity model.

Same architecture surface as ``models.gat_model.NextTaskGAT`` for
fair comparison: optional time-to-next-event regression head, and
post-hoc temperature scaling for calibration.

The default mode is single-task (next-activity classification only).
Pass ``predict_time=True`` to attach a regression head; the prefix
builder will then also emit per-prefix log-seconds-to-next-event
targets (requires ``dt_log`` column in the input dataframe, which
``encode_categoricals`` adds automatically).
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


class NextActivityLSTM(nn.Module):
    """LSTM next-activity model with an optional time-prediction head."""

    def __init__(
        self, num_cls, emb_dim=64, hidden_dim=64, num_layers=1,
        predict_time: bool = False,
    ):
        super().__init__()
        self.predict_time = predict_time
        self.emb = nn.Embedding(num_cls + 1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_cls)
        if predict_time:
            self.dt_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, seq_len):
        seq_len_sorted, perm_idx = seq_len.sort(0, descending=True)
        x_sorted = x[perm_idx]
        x_emb = self.emb(x_sorted)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_emb, seq_len_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        _, unperm_idx = perm_idx.sort(0)
        last_hidden = last_hidden[unperm_idx]
        logits = self.fc(last_hidden)
        if self.predict_time:
            dt_pred = self.dt_head(last_hidden).squeeze(-1)
            return logits, dt_pred
        return logits


def _build_prefixes(df):
    """Yield (prefix_task_ids, next_task_id, dt_log_or_None) per case.

    ``dt_log`` is the log1p-seconds from the last event in the prefix
    to the next event — i.e. the regression target for the time head.
    Falls back to None when the column isn't present (legacy data).
    """
    has_dt = "dt_log" in df.columns
    samples = []
    for _cid, cdata in df.groupby("case_id", sort=False):
        cdata = cdata.sort_values("timestamp")
        tasks_list = cdata["task_id"].tolist()
        dt_list = cdata["dt_log"].tolist() if has_dt else None
        for i in range(1, len(tasks_list)):
            dt = float(dt_list[i - 1]) if dt_list is not None else None
            samples.append((tasks_list[:i], tasks_list[i], dt))
    return samples


def prepare_sequence_data(df, val_frac=0.2, seed=42, train_df=None, val_df=None):
    """Build (prefix, next-task, dt_log) samples per split.

    Splits *cases* (not prefixes) so no future event of a case can
    leak into both halves. Callers can pass already-split frames.
    """
    if train_df is not None and val_df is not None:
        train_samples = _build_prefixes(train_df)
        val_samples = _build_prefixes(val_df)
    else:
        case_ids = sorted(df["case_id"].unique())
        rng = random.Random(seed)
        rng.shuffle(case_ids)
        n_val = max(1, int(round(len(case_ids) * val_frac)))
        val_cases = set(case_ids[:n_val])
        train_df_ = df[~df["case_id"].isin(val_cases)]
        val_df_ = df[df["case_id"].isin(val_cases)]
        train_samples = _build_prefixes(train_df_)
        val_samples = _build_prefixes(val_df_)

    random.Random(seed).shuffle(train_samples)
    random.Random(seed + 1).shuffle(val_samples)
    return train_samples, val_samples


def make_padded_dataset(sample_list, num_cls):
    """Pad prefix sequences and return tensors.

    Returns ``(X_padded, X_lens, Y_labels, max_len)`` for backwards
    compatibility, plus a ``DT_targets`` tensor on the dataset object
    when the samples carry dt_log values. Callers that need it can
    pull it via ``X_padded.dt_targets`` (set as an attribute).
    """
    max_len = max(len(s[0]) for s in sample_list)
    X_padded, X_lens, Y_labels, DT = [], [], [], []
    has_dt = sample_list and len(sample_list[0]) >= 3 and sample_list[0][2] is not None

    for sample in sample_list:
        pfx, nxt = sample[0], sample[1]
        dt = sample[2] if len(sample) >= 3 else None
        seqlen = len(pfx)
        X_lens.append(seqlen)
        seq = [(tid + 1) for tid in pfx]  # shift for pad=0
        seq += [0] * (max_len - seqlen)
        X_padded.append(seq)
        Y_labels.append(nxt)
        if has_dt:
            DT.append(float(dt))

    Xp = torch.tensor(X_padded, dtype=torch.long)
    if has_dt:
        # Store dt targets as a separate tensor; the legacy 4-tuple
        # return shape is preserved for callers that don't need it.
        Xp.dt_targets = torch.tensor(DT, dtype=torch.float)
    return (
        Xp,
        torch.tensor(X_lens, dtype=torch.long),
        torch.tensor(Y_labels, dtype=torch.long),
        max_len,
    )


def train_lstm_model(
    model,
    X_train_pad,
    X_train_len,
    y_train,
    device,
    batch_size: int = 64,
    epochs: int = 5,
    model_path: str = "lstm_next_activity.pth",
    dt_targets: torch.Tensor | None = None,
    time_loss_weight: float = 0.5,
):
    """Train the LSTM. When ``dt_targets`` is supplied and the model has
    ``predict_time=True``, train with a joint CE + ``time_loss_weight``×MSE
    loss; otherwise train with CE only."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset_size = X_train_pad.size(0)
    predict_time = getattr(model, "predict_time", False) and dt_targets is not None

    for ep in range(1, epochs + 1):
        model.train()
        indices = np.random.permutation(dataset_size)
        total_loss = 0.0

        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            idx = indices[start:end]

            bx = X_train_pad[idx].to(device)
            blen = X_train_len[idx].to(device)
            by = y_train[idx].to(device)

            optimizer.zero_grad()
            out = model(bx, blen)
            if predict_time:
                logits, dt_pred = out
                bdt = dt_targets[idx].to(device)
                lval = loss_fn(logits, by) + time_loss_weight * nn.functional.mse_loss(
                    dt_pred, bdt
                )
            else:
                lval = loss_fn(out, by)
            lval.backward()
            optimizer.step()
            total_loss += lval.item()

        avg_loss = total_loss / ((dataset_size + batch_size - 1) // batch_size)
        print(f"[LSTM Ep {ep}/{epochs}] Loss={avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    return model


def evaluate_lstm_model(
    model,
    X_test_pad,
    X_test_len,
    batch_size: int,
    device,
    temperature: float = 1.0,
    dt_targets: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the LSTM and return (preds, probs).

    ``temperature`` divides logits before softmax (use the value
    returned by ``fit_temperature_lstm`` for calibrated probabilities).
    When ``dt_targets`` is provided and the model has the time head,
    populates ``model.last_dt_mae_hours`` so the stage can report it
    without changing the return shape.
    """
    model.eval()
    test_size = X_test_pad.size(0)
    logits_list, dt_pred_list = [], []
    predict_time = getattr(model, "predict_time", False) and dt_targets is not None

    with torch.no_grad():
        for start in range(0, test_size, batch_size):
            end = min(start + batch_size, test_size)
            bx = X_test_pad[start:end].to(device)
            blen = X_test_len[start:end].to(device)
            out = model(bx, blen)
            if predict_time:
                logits, dt_pred = out
                dt_pred_list.append(dt_pred.cpu().numpy())
            else:
                logits = out
            logits_list.append(logits.cpu().numpy())

    logits_arr = np.concatenate(logits_list, axis=0) / temperature
    logits_exp = np.exp(logits_arr - np.max(logits_arr, axis=1, keepdims=True))
    probs = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)
    preds = np.argmax(logits_arr, axis=1)

    if predict_time and dt_pred_list:
        pred = np.expm1(np.concatenate(dt_pred_list))
        true = np.expm1(dt_targets.cpu().numpy())
        model.last_dt_mae_hours = float(np.mean(np.abs(pred - true)) / 3600.0)

    return preds, probs


def fit_temperature_lstm(
    model,
    X_val_pad,
    X_val_len,
    y_val,
    batch_size: int,
    device,
    lr: float = 0.01,
    max_iter: int = 50,
) -> float:
    """Calibrate the LSTM classifier with a single positive temperature.

    Same recipe as ``models.gat_model.fit_temperature``: collect val
    logits, optimize ``T = exp(log_T)`` to minimize NLL via LBFGS.
    """
    model.eval()
    test_size = X_val_pad.size(0)
    logits_list = []

    with torch.no_grad():
        for start in range(0, test_size, batch_size):
            end = min(start + batch_size, test_size)
            bx = X_val_pad[start:end].to(device)
            blen = X_val_len[start:end].to(device)
            out = model(bx, blen)
            logits = out[0] if isinstance(out, tuple) else out
            logits_list.append(logits.cpu())

    if not logits_list:
        return 1.0

    logits = torch.cat(logits_list, dim=0)
    labels = y_val.cpu().long()

    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = log_T.exp()
        loss = nn.functional.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.exp().item())
