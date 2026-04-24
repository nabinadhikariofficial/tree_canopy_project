from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = torch.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    return F.mse_loss(pred, target)


def input_change_weight(x1: torch.Tensor, x2: torch.Tensor, gamma: float = 3.0) -> torch.Tensor:
    diff = torch.mean(torch.abs(x1 - x2), dim=(1, 2, 3))
    return torch.exp(-gamma * diff)


def temporal_consistency_loss(f1: torch.Tensor, f2: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
                              gamma: float = 3.0) -> torch.Tensor:
    weight = input_change_weight(x1, x2, gamma=gamma)
    feat_diff = torch.mean((f1 - f2) ** 2, dim=(1, 2, 3))
    return torch.mean(weight * feat_diff)
