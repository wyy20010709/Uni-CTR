from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path


def _write_jsonl(path: Path, domain: str, n_users: int = 50, n_items: int = 40, n_rows: int = 1800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026 + hash(domain) % 1000)
    with open(path, "w", encoding="utf-8") as f:
        for idx in range(n_rows):
            # keep imbalanced label distribution (~30% positive) and duplicates to test dedup path
            rating = 5 if rng.random() < 0.3 else 2
            row = {
                "reviewerID": f"{domain}_u{rng.randrange(n_users)}",
                "asin": f"{domain}_i{rng.randrange(n_items)}",
                "overall": rating,
                "unixReviewTime": 1700000000 + idx,
            }
            f.write(json.dumps(row) + "\n")


def test_preprocess_and_run_ple_with_fixed_source_target() -> None:
    root = Path(__file__).resolve().parents[1]
    _write_jsonl(root / "data/fashion/raw/AMAZON_FASHION_5.json", "fashion")
    _write_jsonl(root / "data/beauty/raw/All_Beauty_5.json", "beauty")
    _write_jsonl(root / "data/gift/raw/Gift_Cards_5.json", "gift")

    subprocess.run(
        [
            "python",
            "wyy20010709/HTR-Net/scripts/preprocess_and_split.py",
            "--data_root",
            "wyy20010709/HTR-Net/data",
            "--domains",
            "fashion",
            "beauty",
            "gift",
            "--seed",
            "2026",
            "--split_ratio",
            "0.8",
            "0.1",
            "0.1",
            "--input_format",
            "jsonl",
            "--rating_field",
            "overall",
            "--user_field",
            "reviewerID",
            "--item_field",
            "asin",
            "--time_field",
            "unixReviewTime",
            "--label_rule",
            "gt3",
        ],
        cwd=root.parents[1],
        check=True,
    )

    subprocess.run(
        [
            "python",
            "wyy20010709/HTR-Net/run.py",
            "--config",
            "wyy20010709/HTR-Net/configs/fast.yaml",
            "--model",
            "ple",
            "--source_domain",
            "beauty",
            "--target_domain",
            "gift",
            "--epochs",
            "1",
            "--max_steps",
            "20",
        ],
        cwd=root.parents[1],
        check=True,
    )
