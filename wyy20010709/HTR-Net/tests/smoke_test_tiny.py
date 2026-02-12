from __future__ import annotations

import subprocess
from pathlib import Path


def test_tiny_smoke_htr_and_baselines() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["python", "scripts/make_tiny_from_unictr.py", "--toy_if_missing", "--output_dir", "data/tiny/amazon_book_movie"],
        cwd=root,
        check=True,
    )
    for model in ["sharedbottom", "mmoe", "ple", "star", "htr_net"]:
        subprocess.run(
            [
                "python",
                "-m",
                "htrnet.trainers.trainer",
                "--config",
                "configs/tiny_amazon_book_movie.yaml",
                "--model",
                model,
            ],
            cwd=root,
            check=True,
        )
