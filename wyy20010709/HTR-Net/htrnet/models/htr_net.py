from __future__ import annotations

import torch
from torch import nn

from .baselines import BaseInput
from .mmoe_backbone import MMoE
from .retrieval import batch_retrieval
from .translation import LinearTranslation, cross_loss


class HTRNet(BaseInput):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_domains: int,
        embed_dim: int,
        source_domain_id: int,
        target_domain_id: int,
        experts: int = 4,
        hidden: int = 64,
        topk: int = 8,
        temperature: float = 0.1,
        alpha: float = 0.1,
        beta_orth: float = 0.0,
        loss_type: str = "mse",
    ):
        super().__init__(num_users, num_items, num_domains, embed_dim)
        self.backbone = MMoE(self.out_dim, experts, hidden, num_tasks=2)
        self.translation = LinearTranslation(hidden)
        self.head_s = nn.Linear(hidden * 2, 1)
        self.head_t = nn.Linear(hidden * 2, 1)
        self.source_domain_id = source_domain_id
        self.target_domain_id = target_domain_id
        self.topk = topk
        self.temperature = temperature
        self.alpha = alpha
        self.beta_orth = beta_orth
        self.loss_type = loss_type

    def forward(self, batch: dict) -> dict:
        x = self.encode(batch)
        s_repr, t_repr = self.backbone(x)
        src_mask = batch["domain_id"] == self.source_domain_id
        tgt_mask = batch["domain_id"] == self.target_domain_id

        src_bank = s_repr[src_mask]
        tgt_bank = t_repr[tgt_mask]

        s_aug = batch_retrieval(s_repr, src_bank, self.topk, self.temperature)
        t_aug = batch_retrieval(t_repr, tgt_bank, self.topk, self.temperature)

        s_logit = self.head_s(torch.cat([s_repr, s_aug], dim=-1)).squeeze(-1)
        t_logit = self.head_t(torch.cat([t_repr, t_aug], dim=-1)).squeeze(-1)

        src_trans = self.translation(t_repr[tgt_mask])
        cross = cross_loss(src_trans, s_repr[src_mask], self.loss_type)
        orth = torch.tensor(0.0, device=x.device)

        logits = torch.where(src_mask, s_logit, t_logit)
        aux_loss = self.alpha * cross + self.beta_orth * orth
        return {"logits": logits, "aux_loss": aux_loss, "cross_loss": cross.detach()}
