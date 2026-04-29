#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Small shared utilities."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline touches.

    `torch.manual_seed` alone is not enough — CUDA, NumPy, and the
    standard-library `random` module all keep their own state, and CuDNN
    can introduce non-determinism through algorithm selection.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best available torch device, with `prefer` as an override."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
