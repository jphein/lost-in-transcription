#!/usr/bin/env bash
# Pack src/*.py + model dirs into a submission zip (weights stored, not deflated).
# Usage: ./pack.sh <out.zip> <model-dir> [<model-dir> ...]   (model dirs under models/)
set -euo pipefail
cd "$(dirname "$0")"
OUT="$(realpath -m "$1")"; shift
mkdir -p "$(dirname "$OUT")"; rm -f "$OUT"
stage=$(mktemp -d -p "$PWD" .stage.XXXX); trap 'rm -rf "$stage"' EXIT
cp src/*.py "$stage/"; mkdir -p "$stage/models"
for M in "$@"; do cp -rL "models/$M" "$stage/models/$M"; rm -rf "$stage/models/$M/.cache"; done
chmod -R a+rX "$stage"
(cd "$stage" && zip -q -r -0 "$OUT" . -x '*.pyc' '__pycache__/*')
unzip -Z1 "$OUT" | grep -Fxq main.py && echo "OK $(du -h "$OUT" | cut -f1) $OUT"
