#!/usr/bin/env bash
# Pack src/main.py + a model dir into a submission zip (weights stored, not deflated).
# Usage: ./pack.sh <model-dir-name> <out.zip>
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-faster-whisper-large-v3}"; OUT="$(realpath -m "${2:-dist/submission.zip}")"
mkdir -p "$(dirname "$OUT")"; rm -f "$OUT"
stage=$(mktemp -d -p "$PWD" .stage.XXXX); trap 'rm -rf "$stage"' EXIT
cp src/*.py "$stage/"; mkdir -p "$stage/models"
cp -rL "models/$MODEL" "$stage/models/$MODEL"
rm -rf "$stage/models/$MODEL/.cache"
(cd "$stage" && zip -q -r -0 "$OUT" . -x '*.pyc')
unzip -Z1 "$OUT" | grep -Fxq main.py && echo "OK $(du -h "$OUT" | cut -f1) $OUT"
