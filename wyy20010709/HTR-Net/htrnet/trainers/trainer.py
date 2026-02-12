from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from htrnet.data.dataloader import build_dataloaders
from htrnet.models.baselines import MMOEModel, PLEModel, STARModel, SharedBottomModel
from htrnet.models.htr_net import HTRNet
from htrnet.trainers.metrics import compute_auc_logloss
from htrnet.utils.config import load_config, parse_args
from htrnet.utils.io import save_json
from htrnet.utils.logger import build_logger
from htrnet.utils.seed import set_seed


def build_model(name: str, cfg: dict, meta: dict) -> nn.Module:
    embed_dim = int(cfg["model"].get("embed_dim", 16))
    hidden = int(cfg["model"].get("hidden_dim", 64))
    experts = int(cfg["model"].get("num_experts", 4))
    kwargs = dict(num_users=meta["num_users"], num_items=meta["num_items"], num_domains=meta["num_domains"], embed_dim=embed_dim)

    if name == "sharedbottom":
        return SharedBottomModel(**kwargs, hidden=hidden)
    if name == "mmoe":
        return MMOEModel(**kwargs, experts=experts, hidden=hidden)
    if name == "ple":
        return PLEModel(**kwargs, experts=experts, hidden=hidden)
    if name == "star":
        return STARModel(**kwargs, hidden=hidden)
    source_name = cfg["task"].get("source_domain", cfg["task"].get("domains", ["0", "1"])[0])
    target_name = cfg["task"].get("target_domain", cfg["task"].get("domains", ["0", "1"])[-1])
    return HTRNet(
        **kwargs,
        source_domain_id=meta["domain2idx"].get(str(source_name), 0),
        target_domain_id=meta["domain2idx"].get(str(target_name), min(1, meta["num_domains"] - 1)),
        experts=experts,
        hidden=hidden,
        topk=int(cfg["model"].get("topk", 8)),
        temperature=float(cfg["model"].get("temperature", 0.1)),
        alpha=float(cfg["model"].get("alpha", 0.1)),
        beta_orth=float(cfg["model"].get("beta_orth", 0.0)),
        loss_type=str(cfg["model"].get("loss_type", "mse")),
    )


def run_epoch(model: nn.Module, loader, optimizer=None, device="cpu", max_steps=10_000):
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss()
    ys, ps, losses = [], [], []
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        loss = loss_fn(out["logits"], batch["label"])
        if "aux_loss" in out:
            loss = loss + out["aux_loss"]
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(loss.item())
        ys.append(batch["label"].detach().cpu().numpy())
        ps.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())
    y = np.concatenate(ys) if ys else np.array([0, 1])
    p = np.concatenate(ps) if ps else np.array([0.5, 0.5])
    metric = compute_auc_logloss(y, p)
    metric["loss"] = float(np.mean(losses)) if losses else 0.0
    return metric


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or cfg.get("model", {}).get("name", "htr_net")
    set_seed(int(cfg.get("seed", 2026)))

    train_loader, val_loader, test_loader, meta = build_dataloaders(cfg)
    model = build_model(model_name.lower(), cfg, meta)

    device = "cuda" if (cfg.get("device", "auto") == "auto" and torch.cuda.is_available()) else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("lr", 1e-3)), weight_decay=float(cfg["train"].get("weight_decay", 1e-5)))

    exp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"
    out_dir = Path(args.output_dir) / exp
    logger = build_logger(out_dir / "log.txt")

    best_auc = -1.0
    all_metrics = {"model": model_name, "epochs": []}
    epochs = int(cfg["train"].get("epochs", 1))
    max_steps = int(cfg["train"].get("max_steps", 200))

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, max_steps=max_steps)
        va = run_epoch(model, val_loader, optimizer=None, device=device, max_steps=max_steps)
        logger.info(f"epoch={epoch} train={tr} val={va}")
        all_metrics["epochs"].append({"epoch": epoch, "train": tr, "val": va})
        if va["auc"] >= best_auc:
            best_auc = va["auc"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    te = run_epoch(model, test_loader, optimizer=None, device=device, max_steps=max_steps)
    logger.info(f"test={te}")
    all_metrics["test"] = te
    torch.save(model.state_dict(), out_dir / "last.pt")
    save_json(all_metrics, out_dir / "metrics.json")
    print(json.dumps({"output_dir": str(out_dir), "test": te}, ensure_ascii=False))


if __name__ == "__main__":
    main()
