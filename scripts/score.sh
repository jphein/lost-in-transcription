#!/usr/bin/env bash
# Usage: scripts/score.sh <gt.csv> <pred.csv>
cd "$(dirname "$0")/.." && uv run -q runtime/score.py "$(realpath "$1")" --predicted-path "$(realpath "$2")"
