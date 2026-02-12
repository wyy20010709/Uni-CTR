from __future__ import annotations

import torch
from torch import nn


class MMoE(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, expert_dim: int, num_tasks: int):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, expert_dim), nn.ReLU()) for _ in range(num_experts)])
        self.gates = nn.ModuleList([nn.Linear(input_dim, num_experts) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        exp_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        task_reprs = []
        for gate in self.gates:
            weight = torch.softmax(gate(x), dim=-1).unsqueeze(-1)
            task_reprs.append((exp_out * weight).sum(dim=1))
        return task_reprs
