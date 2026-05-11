"""
Logit-Adjusted Training (Menon et al. 2021)
============================================
Train-time prior correction: L = CE(z + tau * log pi_train, y).
Inference: argmax(z) — raw logits, no shift.

Regime A (this implementation):
    log pi is computed from the TRAIN manifest only. Test labels are never
    inspected — clean fairness regime.

Train priors (Dataset_1024_8bit, 8557 patients, from CLAUDE.md):
    BR1=1678, BR2=2754, BR4=1898, BR5=2227
    pi_train = [0.1961, 0.3219, 0.2218, 0.2603]
    log pi   = [-1.6293, -1.1336, -1.5060, -1.3460]

Used by: utils/losses.py MultiHeadLoss when training.logit_adjustment.enabled
is true. Buffer is registered on the loss module and broadcast across the
batch dimension at forward time. outputs["full_logits"] is left unmodified
so metrics and inference paths see raw logits as required.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

# Train patient counts for Dataset_1024_8bit (matches CLAUDE.md table).
# Order: [BR1, BR2, BR4, BR5] — the same index order used everywhere else
# in this repo (class_weights, full_head logits, label_dict["full"]).
TRAIN_COUNTS_4CLASS: tuple[int, int, int, int] = (1678, 2754, 1898, 2227)


def compute_log_prior(counts: Sequence[int]) -> torch.Tensor:
    """
    Returns log(pi) as a 1-D float tensor of length len(counts).

    Args:
        counts: per-class sample counts (positive ints).

    Returns:
        Tensor of shape (C,) with log-probabilities.
    """
    if any(c <= 0 for c in counts):
        raise ValueError(f"counts must be positive, got {counts}")
    total = float(sum(counts))
    log_pi = torch.tensor(
        [math.log(c / total) for c in counts],
        dtype=torch.float32,
    )
    return log_pi


def train_log_prior_4class() -> torch.Tensor:
    """
    Returns log pi_train for the 4-class full head, in [BR1, BR2, BR4, BR5] order.
    """
    return compute_log_prior(TRAIN_COUNTS_4CLASS)
