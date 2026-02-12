from __future__ import annotations

import torch


def batch_retrieval(query: torch.Tensor, bank: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.zeros_like(query)
    sim = torch.matmul(query, bank.T)
    k = min(topk, bank.size(0))
    vals, idx = torch.topk(sim, k=k, dim=-1)
    weights = torch.softmax(vals / max(temperature, 1e-6), dim=-1)
    retrieved = bank[idx]
    return (retrieved * weights.unsqueeze(-1)).sum(dim=1)
