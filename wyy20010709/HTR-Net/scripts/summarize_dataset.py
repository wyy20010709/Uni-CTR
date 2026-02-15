#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--as_json", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    parts = []
    for name in ["train.csv", "val.csv", "test.csv"]:
        fp = data_dir / name
        if fp.exists():
            parts.append(pd.read_csv(fp))
    if not parts:
        raise FileNotFoundError(f"No split files found in {data_dir}")
    df = pd.concat(parts, axis=0)
    stat = {
        "users": int(df.user_id.nunique()),
        "items": int(df.item_id.nunique()),
        "interactions": int(len(df)),
        "domains": int(df.domain_id.nunique()),
    }
    if args.as_json:
        print(json.dumps(stat, ensure_ascii=False, indent=2))
    else:
        print(stat)


if __name__ == "__main__":
    main()
