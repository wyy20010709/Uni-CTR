#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gen_toy(domains: list[str], n_users: int, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in domains:
        for u in range(n_users):
            cnt = int(rng.integers(5, k + 1))
            for _ in range(cnt):
                rows.append({
                    "domain_id": d,
                    "user_id": f"{d}_u{u}",
                    "item_id": f"{d}_i{int(rng.integers(0, n_users * 3))}",
                    "label": int(rng.random() > 0.5),
                    "text": "",
                })
    return pd.DataFrame(rows)


def split_and_save(df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    i1, i2 = int(n * 0.8), int(n * 0.9)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.iloc[:i1].to_csv(out_dir / "train.csv", index=False)
    df.iloc[i1:i2].to_csv(out_dir / "val.csv", index=False)
    df.iloc[i2:].to_csv(out_dir / "test.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--unictr_root", type=str, default="../data")
    p.add_argument("--output_dir", type=str, default="data/tiny/amazon_book_movie")
    p.add_argument("--domains", nargs="+", default=["book", "movie"])
    p.add_argument("--n_users", type=int, default=30)
    p.add_argument("--k_per_user", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--toy_if_missing", action="store_true")
    args = p.parse_args()

    root = Path(args.unictr_root)
    out = Path(args.output_dir)

    found = list(root.glob("**/*.csv")) + list(root.glob("**/*.tsv"))
    if not found:
        if not args.toy_if_missing:
            raise FileNotFoundError(
                f"No Uni-CTR data found under {root}. Pass --toy_if_missing to generate a tiny toy dataset."
            )
        df = gen_toy(args.domains, args.n_users, args.k_per_user, args.seed)
        split_and_save(df, out, args.seed)
        print(f"toy tiny dataset generated at {out}")
        return

    # minimal compatible sampling from detected files that already include required fields
    frames = []
    for f in found:
        try:
            d = pd.read_csv(f, sep="\t" if f.suffix == ".tsv" else ",")
            if all(c in d.columns for c in ["domain_id", "user_id", "item_id", "label"]):
                frames.append(d[["domain_id", "user_id", "item_id", "label"]])
        except Exception:
            continue
    if not frames:
        if args.toy_if_missing:
            df = gen_toy(args.domains, args.n_users, args.k_per_user, args.seed)
            split_and_save(df, out, args.seed)
            print(f"fallback toy tiny dataset generated at {out}")
            return
        raise RuntimeError("Detected files but none match required columns domain_id,user_id,item_id,label")

    df = pd.concat(frames, axis=0)
    sampled = []
    for d in args.domains:
        dd = df[df["domain_id"].astype(str) == d]
        users = dd["user_id"].astype(str).drop_duplicates().head(args.n_users)
        for u in users:
            sampled.append(dd[dd["user_id"].astype(str) == u].head(args.k_per_user))
    out_df = pd.concat(sampled, axis=0) if sampled else gen_toy(args.domains, args.n_users, args.k_per_user, args.seed)
    split_and_save(out_df, out, args.seed)
    print(f"tiny dataset generated at {out}, rows={len(out_df)}")


if __name__ == "__main__":
    main()
