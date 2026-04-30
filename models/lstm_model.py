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
    """LSTM next-activity model with an optional time-prediction head.

    Optional resource conditioning
    -----------------------------
    The previous version of this model saw only the task-ID sequence —
    *strictly less information* than the 1st-order Markov baseline,
    which directly memorizes per-task transition probabilities. To beat
    Markov on real industrial logs you need features Markov can't see;
    the simplest and highest-leverage one is **resource**.

    Pass ``num_resources=N`` to attach a parallel resource embedding.
    Forward then takes a second padded tensor (``x_resources``) and
    concatenates its embedding with the task embedding before feeding
    to the LSTM. Doubles the LSTM input dim (2*emb_dim).

    With ``num_resources=None`` (default) the model is identical to the
    pre-feature version — important for backwards-compat with saved
    checkpoints and the existing predict-suffix / serve callers.
    """

    def __init__(
        self, num_cls, emb_dim=64, hidden_dim=64, num_layers=1,
        predict_time: bool = False,
        num_resources: int | None = None,
        n_continuous_dims: int = 0,
        time_quantiles: tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.predict_time = predict_time
        self.use_resource = num_resources is not None
        self.n_continuous_dims = n_continuous_dims
        # When ``time_quantiles`` is set (e.g. (0.1, 0.5, 0.9)) the time
        # head emits one log-seconds prediction per quantile and is
        # trained with pinball loss instead of MSE. dt_pred shape is
        # ``(N, K)`` instead of ``(N,)`` — eval reports MAE on the median
        # quantile and ``coverage`` (fraction of true values inside the
        # outer interval).
        if time_quantiles is not None:
            qs = tuple(float(q) for q in time_quantiles)
            if not all(0.0 < q < 1.0 for q in qs):
                raise ValueError(f"all quantiles must be in (0, 1), got {qs}")
            if len(qs) != len(set(qs)):
                raise ValueError(f"quantiles must be unique, got {qs}")
            self.time_quantiles: tuple[float, ...] | None = tuple(sorted(qs))
        else:
            self.time_quantiles = None
        self.emb = nn.Embedding(num_cls + 1, emb_dim, padding_idx=0)
        if self.use_resource:
            self.resource_emb = nn.Embedding(num_resources + 1, emb_dim, padding_idx=0)
        lstm_in = (
            emb_dim
            + (emb_dim if self.use_resource else 0)
            + n_continuous_dims
        )
        self.lstm = nn.LSTM(lstm_in, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_cls)
        if predict_time:
            n_dt_out = (
                len(self.time_quantiles) if self.time_quantiles is not None else 1
            )
            self.dt_head = nn.Linear(hidden_dim, n_dt_out)

    def forward(
        self, x, seq_len,
        x_resources: torch.Tensor | None = None,
        x_continuous: torch.Tensor | None = None,
    ):
        seq_len_sorted, perm_idx = seq_len.sort(0, descending=True)
        x_sorted = x[perm_idx]
        x_emb = self.emb(x_sorted)
        if self.use_resource:
            if x_resources is None:
                raise ValueError(
                    "model has resource embedding but x_resources is None"
                )
            r_sorted = x_resources[perm_idx]
            x_emb = torch.cat([x_emb, self.resource_emb(r_sorted)], dim=-1)
        if self.n_continuous_dims > 0:
            if x_continuous is None:
                raise ValueError(
                    f"model expects {self.n_continuous_dims} continuous dims "
                    f"but x_continuous is None"
                )
            c_sorted = x_continuous[perm_idx]
            x_emb = torch.cat([x_emb, c_sorted], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_emb, seq_len_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        _, unperm_idx = perm_idx.sort(0)
        last_hidden = last_hidden[unperm_idx]
        logits = self.fc(last_hidden)
        if self.predict_time:
            dt_raw = self.dt_head(last_hidden)
            if self.time_quantiles is not None:
                # Shape (N, K). Don't squeeze — caller distinguishes via
                # model.time_quantiles.
                return logits, dt_raw
            return logits, dt_raw.squeeze(-1)
        return logits

    def inference_forward(
        self, x, seq_len,
        x_resources: torch.Tensor | None = None,
        x_continuous: torch.Tensor | None = None,
    ):
        """ONNX-exportable forward — same outputs as ``forward`` but
        without ``pack_padded_sequence``.

        The training path packs the batch by descending length so the
        LSTM only does work on real (non-pad) tokens. ONNX's exporter
        struggles with the dynamic shapes that produces, so for
        inference we run the LSTM over the full padded sequence and
        gather the last *real* hidden state via ``seq_len - 1``.

        Equivalent up to floating-point noise (the LSTM still returns
        the same hidden at the same position; padding tokens just get
        processed but their output is ignored). The export round-trip
        test enforces ``max(|torch - onnx|) < 1e-4``.
        """
        x_emb = self.emb(x)
        if self.use_resource:
            if x_resources is None:
                raise ValueError(
                    "model has resource embedding but x_resources is None"
                )
            x_emb = torch.cat([x_emb, self.resource_emb(x_resources)], dim=-1)
        if self.n_continuous_dims > 0:
            if x_continuous is None:
                raise ValueError(
                    f"model expects {self.n_continuous_dims} continuous dims "
                    f"but x_continuous is None"
                )
            x_emb = torch.cat([x_emb, x_continuous], dim=-1)
        # Full-sequence LSTM; output_seq is (B, T, H), final state is (1, B, H).
        output_seq, _ = self.lstm(x_emb)
        # Gather the hidden state at position seq_len-1 per row. clamp
        # protects against an empty input row (seq_len=0) which would
        # otherwise index -1 → last padded position; pipeline guarantees
        # seq_len >= 1, but defensive.
        last_idx = (seq_len - 1).clamp(min=0).long()
        bsz = output_seq.shape[0]
        last_hidden = output_seq[torch.arange(bsz, device=output_seq.device), last_idx]
        logits = self.fc(last_hidden)
        if self.predict_time:
            dt_raw = self.dt_head(last_hidden)
            if self.time_quantiles is not None:
                return logits, dt_raw
            return logits, dt_raw.squeeze(-1)
        return logits


def _build_prefixes(df, *, continuous_features: list[str] | None = None):
    """Yield 5-tuples
    ``(task_pfx, resource_pfx_or_None, continuous_pfx_or_None, next_task, dt_or_None)``.

    Resource IDs are emitted in parallel with task IDs when
    ``resource_id`` is present in ``df``. ``continuous_features`` is a
    list of column names to pull out as per-event float vectors for the
    LSTM's auxiliary continuous-input path; pass empty / None to skip.

    Stages opt in by setting ``num_resources`` and ``n_continuous_dims``
    on the model.
    """
    has_dt = "dt_log" in df.columns
    has_resource = "resource_id" in df.columns
    cont_cols = list(continuous_features or [])
    has_cont = bool(cont_cols) and all(c in df.columns for c in cont_cols)
    samples = []
    for _cid, cdata in df.groupby("case_id", sort=False):
        cdata = cdata.sort_values("timestamp")
        tasks_list = cdata["task_id"].tolist()
        resources_list = cdata["resource_id"].tolist() if has_resource else None
        dt_list = cdata["dt_log"].tolist() if has_dt else None
        cont_arr = (
            cdata[cont_cols].to_numpy(dtype=float) if has_cont else None
        )
        for i in range(1, len(tasks_list)):
            dt = float(dt_list[i - 1]) if dt_list is not None else None
            r_pfx = resources_list[:i] if resources_list is not None else None
            c_pfx = cont_arr[:i].tolist() if cont_arr is not None else None
            samples.append((tasks_list[:i], r_pfx, c_pfx, tasks_list[i], dt))
    return samples


def prepare_sequence_data(
    df, val_frac=0.2, seed=42, train_df=None, val_df=None,
    continuous_features: list[str] | None = None,
):
    """Build (prefix, next-task, dt_log) samples per split.

    Splits *cases* (not prefixes) so no future event of a case can
    leak into both halves. Callers can pass already-split frames.
    ``continuous_features`` is the list of df columns to thread through
    as per-event float vectors (see :func:`_build_prefixes`).
    """
    if train_df is not None and val_df is not None:
        train_samples = _build_prefixes(train_df, continuous_features=continuous_features)
        val_samples = _build_prefixes(val_df, continuous_features=continuous_features)
    else:
        case_ids = sorted(df["case_id"].unique())
        rng = random.Random(seed)
        rng.shuffle(case_ids)
        n_val = max(1, int(round(len(case_ids) * val_frac)))
        val_cases = set(case_ids[:n_val])
        train_df_ = df[~df["case_id"].isin(val_cases)]
        val_df_ = df[df["case_id"].isin(val_cases)]
        train_samples = _build_prefixes(train_df_, continuous_features=continuous_features)
        val_samples = _build_prefixes(val_df_, continuous_features=continuous_features)

    random.Random(seed).shuffle(train_samples)
    random.Random(seed + 1).shuffle(val_samples)
    return train_samples, val_samples


def make_padded_dataset(sample_list, num_cls):
    """Pad prefix sequences and return tensors.

    Sample shape from :func:`_build_prefixes` is
    ``(task_pfx, resource_pfx_or_None, continuous_pfx_or_None,
    next_task, dt_or_None)``. The return shape stays
    ``(X_padded, X_lens, Y_labels, max_len)`` for backwards
    compatibility; optional ``dt_targets``, ``resource_pad``, and
    ``continuous_pad`` ride along as attributes on the padded tensor so
    no caller breaks if it ignores them.
    """
    if not sample_list:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            0,
        )

    first = sample_list[0]
    # Backwards compatibility: support the old 4-tuple shape too.
    if len(first) == 4:
        # Old shape: (task, resource, next, dt). Insert None for continuous.
        sample_list = [(t, r, None, n, d) for (t, r, n, d) in sample_list]
        first = sample_list[0]

    max_len = max(len(s[0]) for s in sample_list)
    X_padded, X_lens, Y_labels, R_padded, C_padded, DT = [], [], [], [], [], []
    has_resource = first[1] is not None
    has_continuous = first[2] is not None
    n_cont_dims = len(first[2][0]) if has_continuous and first[2] else 0
    has_dt = len(first) >= 5 and first[4] is not None

    for sample in sample_list:
        pfx, r_pfx, c_pfx, nxt, dt = sample
        seqlen = len(pfx)
        X_lens.append(seqlen)
        seq = [(tid + 1) for tid in pfx]  # shift for pad=0
        seq += [0] * (max_len - seqlen)
        X_padded.append(seq)
        Y_labels.append(nxt)
        if has_resource:
            r_seq = [(rid + 1) for rid in r_pfx]
            r_seq += [0] * (max_len - seqlen)
            R_padded.append(r_seq)
        if has_continuous:
            c_seq = list(c_pfx)
            c_seq += [[0.0] * n_cont_dims] * (max_len - seqlen)
            C_padded.append(c_seq)
        if has_dt:
            DT.append(float(dt))

    Xp = torch.tensor(X_padded, dtype=torch.long)
    if has_resource:
        Xp.resource_pad = torch.tensor(R_padded, dtype=torch.long)
    if has_continuous:
        Xp.continuous_pad = torch.tensor(C_padded, dtype=torch.float)
    if has_dt:
        Xp.dt_targets = torch.tensor(DT, dtype=torch.float)
    return (
        Xp,
        torch.tensor(X_lens, dtype=torch.long),
        torch.tensor(Y_labels, dtype=torch.long),
        max_len,
    )


def _pinball_loss(
    pred_quantiles: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...],
) -> torch.Tensor:
    """Quantile (pinball) loss averaged across K quantiles.

    For each quantile q with predicted value q_pred, the per-row loss
    is ``max(q * (target - q_pred), (q - 1) * (target - q_pred))`` —
    asymmetric L1 that converges to the q-th quantile of P(target | x).

    pred_quantiles: ``(N, K)``; target: ``(N,)``; quantiles: tuple of K
    floats in (0, 1). Returns a scalar mean over rows + quantiles.
    """
    target = target.unsqueeze(-1)  # (N, 1) for broadcast
    q_tensor = torch.tensor(
        quantiles, dtype=pred_quantiles.dtype, device=pred_quantiles.device
    )
    err = target - pred_quantiles  # (N, K)
    loss = torch.maximum(q_tensor * err, (q_tensor - 1) * err)
    return loss.mean()


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
    loss; otherwise train with CE only.

    Resource conditioning is auto-detected: if ``model.use_resource`` is
    True and the padded tensor has a ``resource_pad`` attribute (set by
    ``make_padded_dataset``), the parallel resource sequence is passed
    to ``model.forward(...)`` per batch. No flag — the model determines
    whether features are used.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset_size = X_train_pad.size(0)
    predict_time = getattr(model, "predict_time", False) and dt_targets is not None
    use_resource = (
        getattr(model, "use_resource", False)
        and hasattr(X_train_pad, "resource_pad")
    )
    R_train_pad = X_train_pad.resource_pad if use_resource else None
    n_cont = getattr(model, "n_continuous_dims", 0)
    use_continuous = n_cont > 0 and hasattr(X_train_pad, "continuous_pad")
    C_train_pad = X_train_pad.continuous_pad if use_continuous else None

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
            br = R_train_pad[idx].to(device) if use_resource else None
            bc = C_train_pad[idx].to(device) if use_continuous else None

            optimizer.zero_grad()
            kw = {}
            if use_resource:
                kw["x_resources"] = br
            if use_continuous:
                kw["x_continuous"] = bc
            out = model(bx, blen, **kw) if kw else model(bx, blen)
            if predict_time:
                logits, dt_pred = out
                bdt = dt_targets[idx].to(device)
                quantiles = getattr(model, "time_quantiles", None)
                if quantiles is not None:
                    time_loss = _pinball_loss(dt_pred, bdt, quantiles)
                else:
                    time_loss = nn.functional.mse_loss(dt_pred, bdt)
                lval = loss_fn(logits, by) + time_loss_weight * time_loss
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
    use_resource = (
        getattr(model, "use_resource", False)
        and hasattr(X_test_pad, "resource_pad")
    )
    R_test_pad = X_test_pad.resource_pad if use_resource else None
    n_cont = getattr(model, "n_continuous_dims", 0)
    use_continuous = n_cont > 0 and hasattr(X_test_pad, "continuous_pad")
    C_test_pad = X_test_pad.continuous_pad if use_continuous else None

    with torch.no_grad():
        for start in range(0, test_size, batch_size):
            end = min(start + batch_size, test_size)
            bx = X_test_pad[start:end].to(device)
            blen = X_test_len[start:end].to(device)
            br = R_test_pad[start:end].to(device) if use_resource else None
            bc = C_test_pad[start:end].to(device) if use_continuous else None
            kw = {}
            if use_resource:
                kw["x_resources"] = br
            if use_continuous:
                kw["x_continuous"] = bc
            out = model(bx, blen, **kw) if kw else model(bx, blen)
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
        true_log = dt_targets.cpu().numpy()
        true_h = np.expm1(true_log) / 3600.0
        quantiles = getattr(model, "time_quantiles", None)
        if quantiles is not None:
            # Stack quantile predictions: shape (N, K).
            preds_q_log = np.concatenate(dt_pred_list, axis=0)
            preds_q_h = np.expm1(preds_q_log) / 3600.0
            # Use the median quantile (closest to 0.5) for MAE.
            median_idx = int(
                np.argmin(np.abs(np.array(quantiles) - 0.5))
            )
            median_h = preds_q_h[:, median_idx]
            model.last_dt_mae_hours = float(np.mean(np.abs(median_h - true_h)))
            # Coverage: fraction of true values inside the [first,last]
            # quantile interval (the outer pair after sorting). For
            # quantiles (0.1, 0.5, 0.9) that's the 80% interval.
            lo = preds_q_h[:, 0]
            hi = preds_q_h[:, -1]
            coverage = float(np.mean((true_h >= lo) & (true_h <= hi)))
            model.last_dt_coverage = coverage
            # Mean width of the interval (in hours) — useful as a
            # "honest sharpness" metric paired with coverage.
            model.last_dt_interval_width_hours = float(np.mean(hi - lo))
        else:
            preds_h = np.expm1(np.concatenate(dt_pred_list)) / 3600.0
            model.last_dt_mae_hours = float(np.mean(np.abs(preds_h - true_h)))

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
    use_resource = (
        getattr(model, "use_resource", False)
        and hasattr(X_val_pad, "resource_pad")
    )
    R_val_pad = X_val_pad.resource_pad if use_resource else None
    n_cont = getattr(model, "n_continuous_dims", 0)
    use_continuous = n_cont > 0 and hasattr(X_val_pad, "continuous_pad")
    C_val_pad = X_val_pad.continuous_pad if use_continuous else None

    with torch.no_grad():
        for start in range(0, test_size, batch_size):
            end = min(start + batch_size, test_size)
            bx = X_val_pad[start:end].to(device)
            blen = X_val_len[start:end].to(device)
            br = R_val_pad[start:end].to(device) if use_resource else None
            bc = C_val_pad[start:end].to(device) if use_continuous else None
            kw = {}
            if use_resource:
                kw["x_resources"] = br
            if use_continuous:
                kw["x_continuous"] = bc
            out = model(bx, blen, **kw) if kw else model(bx, blen)
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
