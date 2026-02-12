#!/usr/bin/env bash
set -euo pipefail

python scripts/make_tiny_from_unictr.py --toy_if_missing --output_dir data/tiny/amazon_book_movie
for m in sharedbottom mmoe ple star htr_net; do
  python -m htrnet.trainers.trainer --config configs/tiny_amazon_book_movie.yaml --model "$m"
done

python scripts/summarize_dataset.py --data_dir data/tiny/amazon_book_movie --as_json
