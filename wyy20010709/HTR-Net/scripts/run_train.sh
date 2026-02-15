#!/usr/bin/env bash
set -euo pipefail

python wyy20010709/HTR-Net/run.py --model "${1:-ple}" --source_domain "${2:-beauty}" --target_domain "${3:-gift}"
