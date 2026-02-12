#!/usr/bin/env python
"""Placeholder for heavier Uni-CTR preprocessing.

This script intentionally keeps logic minimal for the smoke stage.
Users can extend this to align full Uni-CTR raw formats into
`domain_id,user_id,item_id,label,text` tables.
"""

from __future__ import annotations

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    print(f"[placeholder] preprocess from {args.input} to {args.output}")
