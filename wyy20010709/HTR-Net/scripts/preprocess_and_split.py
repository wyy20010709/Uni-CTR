#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Amazon 5-core JSONL and stratified 8:1:1 split")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split_ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--input_format", type=str, default="jsonl", choices=["jsonl"])
    parser.add_argument("--rating_field", type=str, default="overall")
    parser.add_argument("--user_field", type=str, default="reviewerID")
    parser.add_argument("--item_field", type=str, default="asin")
    parser.add_argument("--time_field", type=str, default="unixReviewTime")
    parser.add_argument("--label_rule", type=str, default="gt3", choices=["gt3"])
    parser.add_argument("--no_deduplicate", action="store_true", help="Disable (user,item) deduplication")
    parser.add_argument("--print_stats", action="store_true")
    return parser.parse_args()


def _list_json_files(raw_dir: str) -> list[str]:
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    files = []
    for name in sorted(os.listdir(raw_dir)):
        fp = os.path.join(raw_dir, name)
        if os.path.isfile(fp) and (name.endswith(".json") or name.endswith(".jsonl")):
            files.append(fp)
    if not files:
        raise FileNotFoundError(f"No .json/.jsonl files found under {raw_dir}")
    return files


def _safe_int(v: Any) -> int | None:
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _build_records(
    file_paths: list[str],
    domain: str,
    user_field: str,
    item_field: str,
    rating_field: str,
    time_field: str,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    raw_lines = 0
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                raw_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if user_field not in row or item_field not in row or rating_field not in row:
                    continue
                try:
                    rating = float(row[rating_field])
                except (TypeError, ValueError):
                    continue
                user = str(row[user_field]).strip()
                item = str(row[item_field]).strip()
                if not user or not item:
                    continue
                ts = _safe_int(row.get(time_field))
                records.append(
                    {
                        "domain": domain,
                        "user_id": user,
                        "item_id": item,
                        "label": 1 if rating > 3 else 0,
                        "timestamp": ts,
                    }
                )
    return records, raw_lines


def _deduplicate_latest(df: pd.DataFrame) -> pd.DataFrame:
    # keep latest by timestamp when available; otherwise keep last seen record
    seq = pd.Series(range(len(df)), name="_seq")
    work = pd.concat([df.reset_index(drop=True), seq], axis=1)
    work["_ts_exists"] = work["timestamp"].notna().astype(int)
    work["_ts_sort"] = work["timestamp"].fillna(-1)
    work = work.sort_values(["user_id", "item_id", "_ts_exists", "_ts_sort", "_seq"], ascending=[True, True, True, True, True])
    dedup = work.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    dedup = dedup.sort_values("_seq").drop(columns=["_seq", "_ts_exists", "_ts_sort"]).reset_index(drop=True)
    return dedup


def _stratified_split(df: pd.DataFrame, seed: int, split_ratio: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(split_ratio) != 3:
        raise ValueError("split_ratio must have 3 values: train val test")
    tr_r, va_r, te_r = split_ratio
    total = tr_r + va_r + te_r
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}")

    if df["label"].nunique() < 2:
        raise ValueError("Need both positive and negative labels for stratified split")

    # NOTE: no balancing/downsampling/upsampling here — preserve original label distribution.
    train_val, test = train_test_split(
        df,
        test_size=te_r,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )

    val_ratio_in_train_val = va_r / (tr_r + va_r)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_in_train_val,
        random_state=seed,
        stratify=train_val["label"],
        shuffle=True,
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _print_split_stats(domain: str, split_name: str, split_df: pd.DataFrame) -> None:
    n = len(split_df)
    pos_cnt = int(split_df["label"].sum())
    neg_cnt = int(n - pos_cnt)
    pos_ratio = (pos_cnt / n) if n > 0 else 0.0
    print(f"[{domain}] {split_name}: n={n}, pos_ratio={pos_ratio:.6f}, pos_cnt={pos_cnt}, neg_cnt={neg_cnt}")


def _save_split(split_df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = split_df[["domain", "user_id", "item_id", "label", "timestamp"]].copy()
    out.to_csv(output_path, sep="\t", index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()

    for domain in args.domains:
        raw_dir = os.path.join(args.data_root, domain, "raw")
        split_dir = os.path.join(args.data_root, domain, "split")

        files = _list_json_files(raw_dir)
        records, raw_lines = _build_records(
            file_paths=files,
            domain=domain,
            user_field=args.user_field,
            item_field=args.item_field,
            rating_field=args.rating_field,
            time_field=args.time_field,
        )
        if not records:
            raise RuntimeError(f"No valid rows parsed for domain={domain} from files={files}")

        df = pd.DataFrame(records)
        parsed_rows = len(df)
        if not args.no_deduplicate:
            df = _deduplicate_latest(df)
        final_rows = len(df)

        train_df, val_df, test_df = _stratified_split(df, seed=args.seed, split_ratio=args.split_ratio)

        _save_split(train_df, os.path.join(split_dir, "train.tsv"))
        _save_split(val_df, os.path.join(split_dir, "val.tsv"))
        _save_split(test_df, os.path.join(split_dir, "test.tsv"))

        print(
            f"processed domain={domain} files={len(files)} raw_lines={raw_lines} "
            f"parsed_rows={parsed_rows} final_rows={final_rows} -> {split_dir}"
        )
        if args.print_stats:
            _print_split_stats(domain, "train", train_df)
            _print_split_stats(domain, "val", val_df)
            _print_split_stats(domain, "test", test_df)


if __name__ == "__main__":
    main()
