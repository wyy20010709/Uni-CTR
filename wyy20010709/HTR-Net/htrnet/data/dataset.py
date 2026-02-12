from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ["domain_id", "user_id", "item_id", "label"]


@dataclass
class Vocab:
    user2idx: Dict[str, int]
    item2idx: Dict[str, int]
    domain2idx: Dict[str, int]


class CTRDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: Vocab):
        self.frame = frame.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        return {
            "domain_id": torch.tensor(self.vocab.domain2idx[str(row.domain_id)], dtype=torch.long),
            "user_id": torch.tensor(self.vocab.user2idx[str(row.user_id)], dtype=torch.long),
            "item_id": torch.tensor(self.vocab.item2idx[str(row.item_id)], dtype=torch.long),
            "label": torch.tensor(float(row.label), dtype=torch.float32),
        }


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c} in {path}")
    return df


def load_splits(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = _read_table(data_dir / "train.csv")
    val = _read_table(data_dir / "val.csv")
    test = _read_table(data_dir / "test.csv")
    return train, val, test


def build_vocab(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Vocab:
    all_df = pd.concat([train, val, test], axis=0)
    user2idx = {u: i for i, u in enumerate(all_df.user_id.astype(str).unique())}
    item2idx = {it: i for i, it in enumerate(all_df.item_id.astype(str).unique())}
    domain2idx = {d: i for i, d in enumerate(all_df.domain_id.astype(str).unique())}
    return Vocab(user2idx=user2idx, item2idx=item2idx, domain2idx=domain2idx)
