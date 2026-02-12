from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

from .dataset import CTRDataset, build_vocab, load_splits


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    data_dir = Path(cfg["data"]["tiny_path"])
    train_df, val_df, test_df = load_splits(data_dir)
    vocab = build_vocab(train_df, val_df, test_df)

    train_ds = CTRDataset(train_df, vocab)
    val_ds = CTRDataset(val_df, vocab)
    test_ds = CTRDataset(test_df, vocab)

    batch_size = int(cfg["train"].get("batch_size", 256))
    eval_bs = int(cfg["train"].get("eval_batch_size", batch_size))
    num_workers = int(cfg.get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, num_workers=num_workers)

    meta = {
        "num_users": len(vocab.user2idx),
        "num_items": len(vocab.item2idx),
        "num_domains": len(vocab.domain2idx),
        "domain2idx": vocab.domain2idx,
    }
    return train_loader, val_loader, test_loader, meta
