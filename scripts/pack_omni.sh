#!/usr/bin/env bash
# Pack the omni zip (sp-nh and in-jv): the src/ code + models/omni-3b-v2-fp16 + src/nah_vocab.tsv.
# nah_vocab.tsv is generated here from the local dev references (scripts/build_nah_vocab.py) and never committed:
# it is derived from the MDC dev set. Packing refuses without it, because segment.py is a no-op without it.
# Usage: scripts/pack_omni.sh   (env: GT=data/devrt/nahuatl/gt.csv, OUT=dist/omni/submission.zip)
set -euo pipefail
cd "$(dirname "$0")/.."
GT=${GT:-data/devrt/nahuatl/gt.csv}
OUT=$(realpath -m "${OUT:-dist/omni/submission.zip}")
FILES=(main.py omni.py nahuatl.py es_words.txt segment.py nah_vocab.tsv indonesian.py)
uv run --no-project -q --with jiwer --with pandas --with typer python scripts/build_nah_vocab.py "$GT"
stage=$(mktemp -d -p "$PWD" .stage.XXXX); trap 'rm -rf "$stage"' EXIT
for f in "${FILES[@]}"; do cp "src/$f" "$stage/"; done
mkdir -p "$stage/models"
ionice -c3 cp -rL models/omni-3b-v2-fp16 "$stage/models/"; rm -rf "$stage/models/omni-3b-v2-fp16/.cache"
chmod -R a+rX "$stage"            # the weights are 600 on disk; the container needs them readable
mkdir -p "$(dirname "$OUT")"; rm -f "$OUT"
(cd "$stage" && ionice -c3 nice -n 19 zip -q -r -0 "$OUT" . -x '*.pyc' '__pycache__/*')
for f in "${FILES[@]}"; do unzip -Z1 "$OUT" | grep -Fxq "$f" || { echo "MISSING $f in $OUT"; exit 1; }; done
unzip -Z1 "$OUT" | grep -v '^models/'
echo "OK $(du -h "$OUT" | cut -f1) $OUT"
sha256sum "$OUT"
