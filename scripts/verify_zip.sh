#!/usr/bin/env bash
# Re-verify each dist zip in the official runtime image (digest-pinned), GPU on, network off.
# Usage: verify.sh <tag> <data_dir> <zip> [docker -e args...]
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=$1 DATA=$(realpath "$2") ZIP=$(realpath "$3"); shift 3
IMG=lostintranscriptionprodacr.azurecr.io/lost-in-transcription-competition@sha256:7a311286889722f5867c4c44b76b4d5a2416b0a4303210d4f01d9aae0440ae3e
OUT=${VERIFY_OUT:-scratch/verify}; mkdir -p "$OUT"; sub=$(mktemp -d -p "$PWD" .run.XXXX); trap 'rm -rf "$sub"' EXIT
chmod 777 "$sub"; ln "$ZIP" "$sub/submission.zip"
s=$(date +%s.%N)
docker run --rm --gpus all --network none "$@" \
  --mount type=bind,source="$DATA",target=/code_execution/data,readonly \
  --mount type=bind,source="$sub",target=/code_execution/submission \
  "$IMG" > "$OUT/$TAG.docker.log" 2>&1
rc=$?; e=$(date +%s.%N)
cp "$sub/submission.csv" "$OUT/$TAG.csv" 2>/dev/null
python3 - "$DATA/test_metadata.csv" "$OUT/$TAG.csv" "$TAG" "$rc" "$(echo "$e - $s" | bc)" <<'PY'
import csv,sys
man,sub,tag,rc,wall=sys.argv[1:]
m=[r['audio_filename'] for r in csv.DictReader(open(man))]
try:
    rows=list(csv.reader(open(sub,newline='',encoding='utf-8')))
except FileNotFoundError:
    print(f"{tag}: rc={rc} wall={float(wall):.1f}s NO submission.csv"); sys.exit()
hdr,body=rows[0],rows[1:]
ok = hdr==['audio_filename','transcript'] and [r[0] for r in body]==m and all(len(r)==2 for r in body)
empty=sum(1 for r in body if not r[1].strip())
print(f"{tag}: rc={rc} wall={float(wall):.1f}s rows={len(body)}/{len(m)} header+order+2cols={'OK' if ok else 'FAIL'} empty={empty}")
PY
