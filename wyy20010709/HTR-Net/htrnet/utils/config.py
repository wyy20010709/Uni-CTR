from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTR-Net trainer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, choices=["htr_net", "sharedbottom", "mmoe", "ple", "star"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()
