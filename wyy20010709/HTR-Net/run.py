#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from htrnet.models.baselines import MMOEModel, PLEModel, STARModel, SharedBottomModel
from htrnet.models.htr_net import HTRNet
from htrnet.trainers.metrics import compute_auc_logloss
from htrnet.utils.io import save_json
from htrnet.utils.logger import build_logger
from htrnet.utils.seed import set_seed

MODEL_REGISTRY = {
    "ple": PLEModel,
    "mmoe": MMOEModel,
    "sharedbottom": SharedBottomModel,
    "star": STARModel,
    "htr_net": HTRNet,
}


class CTRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, user2idx: dict[str, int], item2idx: dict[str, int], domain2idx: dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.domain2idx = domain2idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        domain_id = int(row.domain_id) if "domain_id" in self.df.columns else self.domain2idx[str(row.domain)]
        return {
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
            "user_id": torch.tensor(self.user2idx[str(row.user_id)], dtype=torch.long),
            "item_id": torch.tensor(self.item2idx[str(row.item_id)], dtype=torch.long),
            "label": torch.tensor(float(row.label), dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified HTR-Net runner with source/target/model overrides")
    p.add_argument("--config", type=str, default="wyy20010709/HTR-Net/configs/fast.yaml")
    p.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), default=None)
    p.add_argument("--source_domain", type=str, default=None)
    p.add_argument("--target_domain", type=str, default=None)
    p.add_argument("--data_root", type=str, default="wyy20010709/HTR-Net/data")
    p.add_argument("--output_dir", type=str, default="wyy20010709/HTR-Net/outputs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lambda_src", type=float, default=None)
    p.add_argument("--lambda_tgt", type=float, default=None)
    p.add_argument("--target_ratio", type=float, default=None)
    p.add_argument("--stage2_epochs", type=int, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    cfg.setdefault("task", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("runtime", {})

    cfg["runtime"]["data_root"] = args.data_root
    cfg["runtime"]["output_dir"] = args.output_dir

    if args.source_domain is not None:
        cfg["task"]["source_domain"] = args.source_domain
    if args.target_domain is not None:
        cfg["task"]["target_domain"] = args.target_domain
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.lambda_src is not None:
        cfg["train"]["lambda_src"] = args.lambda_src
    if args.lambda_tgt is not None:
        cfg["train"]["lambda_tgt"] = args.lambda_tgt
    if args.target_ratio is not None:
        cfg["train"]["target_ratio"] = args.target_ratio
    if args.stage2_epochs is not None:
        cfg["train"]["stage2_epochs"] = args.stage2_epochs

    cfg.setdefault("seed", 2026)
    cfg["train"].setdefault("batch_size", 128)
    cfg["train"].setdefault("eval_batch_size", 256)
    cfg["train"].setdefault("epochs", 1)
    cfg["train"].setdefault("max_steps", 100)
    cfg["train"].setdefault("lr", 1e-3)
    cfg["train"].setdefault("weight_decay", 1e-5)
    cfg["train"].setdefault("lambda_src", 0.2)
    cfg["train"].setdefault("lambda_tgt", 1.0)
    cfg["train"].setdefault("target_ratio", 0.5)
    cfg["train"].setdefault("stage2_epochs", 1)

    cfg["model"].setdefault("name", "ple")
    cfg["model"].setdefault("embed_dim", 16)
    cfg["model"].setdefault("hidden_dim", 64)
    cfg["model"].setdefault("num_experts", 4)

    if "source_domain" not in cfg["task"] or "target_domain" not in cfg["task"]:
        raise ValueError("source_domain and target_domain must be provided via config or CLI")
    return cfg


def read_split(data_root: Path, domain: str, split: str) -> pd.DataFrame:
    fp = data_root / domain / "split" / f"{split}.tsv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}. Please run scripts/preprocess_and_split.py first.")
    df = pd.read_csv(fp, sep="\t")
    required = ["domain", "user_id", "item_id", "label", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fp} missing columns {missing}")
    return df


def build_model(model_name: str, cfg: dict, meta: dict) -> nn.Module:
    embed_dim = int(cfg["model"].get("embed_dim", 16))
    hidden = int(cfg["model"].get("hidden_dim", 64))
    experts = int(cfg["model"].get("num_experts", 4))
    base_kwargs = {
        "num_users": meta["num_users"],
        "num_items": meta["num_items"],
        "num_domains": meta["num_domains"],
        "embed_dim": embed_dim,
    }
    if model_name == "ple":
        return PLEModel(**base_kwargs, experts=experts, hidden=hidden)
    if model_name == "mmoe":
        return MMOEModel(**base_kwargs, experts=experts, hidden=hidden)
    if model_name == "sharedbottom":
        return SharedBottomModel(**base_kwargs, hidden=hidden)
    if model_name == "star":
        return STARModel(**base_kwargs, hidden=hidden)
    return HTRNet(
        **base_kwargs,
        source_domain_id=meta["domain2idx"][meta["source_domain"]],
        target_domain_id=meta["domain2idx"][meta["target_domain"]],
        experts=experts,
        hidden=hidden,
        topk=int(cfg["model"].get("topk", 8)),
        temperature=float(cfg["model"].get("temperature", 0.1)),
        alpha=float(cfg["model"].get("alpha", 0.1)),
        beta_orth=float(cfg["model"].get("beta_orth", 0.0)),
        loss_type=str(cfg["model"].get("loss_type", "mse")),
    )


def build_weighted_sampler(train_df: pd.DataFrame, target_ratio: float) -> WeightedRandomSampler:
    src_mask = train_df["domain_id"] == 0
    tgt_mask = train_df["domain_id"] == 1
    n_src = int(src_mask.sum())
    n_tgt = int(tgt_mask.sum())
    if n_src == 0 or n_tgt == 0:
        raise ValueError("Both source and target samples are required for weighted sampling")

    target_ratio = min(max(float(target_ratio), 1e-3), 1 - 1e-3)
    w_src = (1.0 - target_ratio) / n_src
    w_tgt = target_ratio / n_tgt

    weights = np.where(src_mask.values, w_src, w_tgt)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_df),
        replacement=True,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer=None,
    device="cpu",
    max_steps=10_000,
    lambda_src: float = 0.2,
    lambda_tgt: float = 1.0,
    split_loss_by_domain: bool = False,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.BCEWithLogitsLoss()
    ys, ps, losses = [], [], []
    src_losses, tgt_losses = [], []

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)

        if split_loss_by_domain:
            src_mask = batch["domain_id"] == 0
            tgt_mask = batch["domain_id"] == 1
            src_loss = torch.tensor(0.0, device=device)
            tgt_loss = torch.tensor(0.0, device=device)
            if src_mask.any():
                src_loss = loss_fn(out["logits"][src_mask], batch["label"][src_mask])
            if tgt_mask.any():
                tgt_loss = loss_fn(out["logits"][tgt_mask], batch["label"][tgt_mask])
            loss = lambda_src * src_loss + lambda_tgt * tgt_loss
            src_losses.append(float(src_loss.detach().cpu().item()))
            tgt_losses.append(float(tgt_loss.detach().cpu().item()))
        else:
            loss = loss_fn(out["logits"], batch["label"])

        if "aux_loss" in out:
            loss = loss + out["aux_loss"]

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        ys.append(batch["label"].detach().cpu().numpy())
        ps.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())
        losses.append(loss.item())

    y = np.concatenate(ys) if ys else np.array([0, 1])
    p = np.concatenate(ps) if ps else np.array([0.5, 0.5])
    metrics = compute_auc_logloss(y, p)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    if split_loss_by_domain:
        metrics["loss_src"] = float(np.mean(src_losses)) if src_losses else 0.0
        metrics["loss_tgt"] = float(np.mean(tgt_losses)) if tgt_losses else 0.0
    return metrics


def main() -> None:
    args = parse_args()
    cfg = merge_cfg(load_config(args.config), args)
    set_seed(int(cfg["seed"]))

    source = cfg["task"]["source_domain"]
    target = cfg["task"]["target_domain"]
    model_name = cfg["model"]["name"]

    data_root = Path(cfg["runtime"]["data_root"])
    s_train, s_val, s_test = (read_split(data_root, source, x) for x in ["train", "val", "test"])
    t_train, t_val, t_test = (read_split(data_root, target, x) for x in ["train", "val", "test"])

    s_train = s_train.copy()
    t_train = t_train.copy()
    s_val = s_val.copy()
    t_val = t_val.copy()
    s_test = s_test.copy()
    t_test = t_test.copy()

    s_train["domain_id"] = 0
    t_train["domain_id"] = 1
    s_val["domain_id"] = 0
    t_val["domain_id"] = 1
    s_test["domain_id"] = 0
    t_test["domain_id"] = 1

    train_df = pd.concat([s_train, t_train], axis=0).reset_index(drop=True)
    val_src_df, val_tgt_df = s_val.reset_index(drop=True), t_val.reset_index(drop=True)
    test_src_df, test_tgt_df = s_test.reset_index(drop=True), t_test.reset_index(drop=True)

    all_df = pd.concat([train_df, val_src_df, val_tgt_df, test_src_df, test_tgt_df], axis=0)
    user2idx = {u: i for i, u in enumerate(all_df.user_id.astype(str).unique())}
    item2idx = {i: j for j, i in enumerate(all_df.item_id.astype(str).unique())}
    domain2idx = {d: i for i, d in enumerate([source, target])}

    train_ds = CTRDataset(train_df, user2idx, item2idx, domain2idx)
    sampler = build_weighted_sampler(train_df, target_ratio=float(cfg["train"].get("target_ratio", 0.5)))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), sampler=sampler)

    target_train_ds = CTRDataset(t_train.reset_index(drop=True), user2idx, item2idx, domain2idx)
    target_train_loader = DataLoader(target_train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True)

    val_src_loader = DataLoader(CTRDataset(val_src_df, user2idx, item2idx, domain2idx), batch_size=int(cfg["train"].get("eval_batch_size", 256)), shuffle=False)
    val_tgt_loader = DataLoader(CTRDataset(val_tgt_df, user2idx, item2idx, domain2idx), batch_size=int(cfg["train"].get("eval_batch_size", 256)), shuffle=False)
    test_src_loader = DataLoader(CTRDataset(test_src_df, user2idx, item2idx, domain2idx), batch_size=int(cfg["train"].get("eval_batch_size", 256)), shuffle=False)
    test_tgt_loader = DataLoader(CTRDataset(test_tgt_df, user2idx, item2idx, domain2idx), batch_size=int(cfg["train"].get("eval_batch_size", 256)), shuffle=False)

    meta = {
        "num_users": len(user2idx),
        "num_items": len(item2idx),
        "num_domains": len(domain2idx),
        "domain2idx": domain2idx,
        "source_domain": source,
        "target_domain": target,
    }

    model = build_model(model_name, cfg, meta)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    exp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_{source}_to_{target}"
    out_dir = Path(cfg["runtime"]["output_dir"]) / exp
    logger = build_logger(out_dir / "log.txt")

    metrics = {"model": model_name, "source": source, "target": target, "epochs": [], "stage2_epochs": []}
    best_tgt_auc = -1.0

    lambda_src = float(cfg["train"].get("lambda_src", 0.2))
    lambda_tgt = float(cfg["train"].get("lambda_tgt", 1.0))

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        tr = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            max_steps=int(cfg["train"]["max_steps"]),
            lambda_src=lambda_src,
            lambda_tgt=lambda_tgt,
            split_loss_by_domain=True,
        )
        va_s = run_epoch(model, val_src_loader, optimizer=None, device=device)
        va_t = run_epoch(model, val_tgt_loader, optimizer=None, device=device)
        logger.info(f"stage1 epoch={epoch} train={tr} | source_val={va_s} | target_val={va_t}")
        metrics["epochs"].append({"epoch": epoch, "train": tr, "source_val": va_s, "target_val": va_t})
        if va_t["auc"] >= best_tgt_auc:
            best_tgt_auc = va_t["auc"]
            torch.save(model.state_dict(), out_dir / "ckpt_best.pt")

    for epoch in range(1, int(cfg["train"].get("stage2_epochs", 1)) + 1):
        tr_tgt = run_epoch(
            model,
            target_train_loader,
            optimizer=optimizer,
            device=device,
            max_steps=int(cfg["train"]["max_steps"]),
            lambda_src=0.0,
            lambda_tgt=1.0,
            split_loss_by_domain=True,
        )
        va_s = run_epoch(model, val_src_loader, optimizer=None, device=device)
        va_t = run_epoch(model, val_tgt_loader, optimizer=None, device=device)
        logger.info(f"stage2 epoch={epoch} target_train={tr_tgt} | source_val={va_s} | target_val={va_t}")
        metrics["stage2_epochs"].append({"epoch": epoch, "target_train": tr_tgt, "source_val": va_s, "target_val": va_t})
        if va_t["auc"] >= best_tgt_auc:
            best_tgt_auc = va_t["auc"]
            torch.save(model.state_dict(), out_dir / "ckpt_best.pt")

    ts_s = run_epoch(model, test_src_loader, optimizer=None, device=device)
    ts_t = run_epoch(model, test_tgt_loader, optimizer=None, device=device)
    logger.info(f"source_test={ts_s} | target_test={ts_t}")
    metrics["source_test"] = ts_s
    metrics["target_test"] = ts_t

    torch.save(model.state_dict(), out_dir / "ckpt_last.pt")
    save_json(metrics, out_dir / "metrics.json")
    print(json.dumps({"output_dir": str(out_dir), "source_test": ts_s, "target_test": ts_t}, ensure_ascii=False))


if __name__ == "__main__":
    main()
