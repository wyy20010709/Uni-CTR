from __future__ import annotations

import torch
from torch import nn

from .mmoe_backbone import MMoE


def mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class BaseInput(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_domains: int, embed_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, embed_dim)
        self.item_emb = nn.Embedding(num_items + 1, embed_dim)
        self.domain_emb = nn.Embedding(num_domains + 1, embed_dim)
        self.out_dim = embed_dim * 3

    def encode(self, batch: dict) -> torch.Tensor:
        return torch.cat([
            self.user_emb(batch["user_id"]),
            self.item_emb(batch["item_id"]),
            self.domain_emb(batch["domain_id"]),
        ], dim=-1)


class SharedBottomModel(BaseInput):
    def __init__(self, num_users: int, num_items: int, num_domains: int, embed_dim: int, hidden: int = 64):
        super().__init__(num_users, num_items, num_domains, embed_dim)
        self.bottom = mlp([self.out_dim, hidden, hidden])
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: dict) -> dict:
        h = self.bottom(self.encode(batch))
        return {"logits": self.head(h).squeeze(-1)}


class MMOEModel(BaseInput):
    def __init__(self, num_users: int, num_items: int, num_domains: int, embed_dim: int, experts: int = 4, hidden: int = 64):
        super().__init__(num_users, num_items, num_domains, embed_dim)
        self.mmoe = MMoE(self.out_dim, experts, hidden, num_tasks=2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: dict) -> dict:
        task_out = self.mmoe(self.encode(batch))
        h = (task_out[0] + task_out[1]) / 2
        return {"logits": self.head(h).squeeze(-1)}


class PLEModel(BaseInput):
    def __init__(self, num_users: int, num_items: int, num_domains: int, embed_dim: int, experts: int = 4, hidden: int = 64):
        super().__init__(num_users, num_items, num_domains, embed_dim)
        self.shared = nn.ModuleList([nn.Sequential(nn.Linear(self.out_dim, hidden), nn.ReLU()) for _ in range(experts)])
        self.task = nn.ModuleList([nn.Sequential(nn.Linear(self.out_dim, hidden), nn.ReLU()) for _ in range(experts)])
        self.gate = nn.Linear(self.out_dim, experts * 2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: dict) -> dict:
        x = self.encode(batch)
        all_exp = torch.stack([m(x) for m in list(self.shared) + list(self.task)], dim=1)
        weight = torch.softmax(self.gate(x), dim=-1).unsqueeze(-1)
        h = (all_exp * weight).sum(dim=1)
        return {"logits": self.head(h).squeeze(-1)}


class STARModel(BaseInput):
    def __init__(self, num_users: int, num_items: int, num_domains: int, embed_dim: int, hidden: int = 64):
        super().__init__(num_users, num_items, num_domains, embed_dim)
        self.shared = nn.Sequential(nn.Linear(self.out_dim, hidden), nn.ReLU())
        self.domain_adaptor = nn.Embedding(num_domains + 1, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: dict) -> dict:
        h = self.shared(self.encode(batch))
        h = h + self.domain_adaptor(batch["domain_id"])
        return {"logits": self.head(h).squeeze(-1)}
