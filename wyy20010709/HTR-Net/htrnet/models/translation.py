from __future__ import annotations

import torch
from torch import nn


class LinearTranslation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def cross_loss(x: torch.Tensor, y: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor(0.0, device=x.device if x.numel() else y.device)
    min_n = min(x.size(0), y.size(0))
    x = x[:min_n]
    y = y[:min_n]
    if loss_type == "cosine":
        return (1 - torch.nn.functional.cosine_similarity(x, y, dim=-1)).mean()
    return torch.nn.functional.mse_loss(x, y)
