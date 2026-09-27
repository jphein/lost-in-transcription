#!/usr/bin/env bash
# Run a submission zip in the official image against a data dir, GPU on, network off.
# Usage: scripts/run_local.sh <data_dir> <zip> <out.csv> [-e VAR=val ...]
set -euo pipefail
DATA=$(realpath "$1"); ZIP=$(realpath "$2"); OUTCSV=$(realpath -m "$3"); shift 3
IMG=lostintranscriptionprodacr.azurecr.io/lost-in-transcription-competition:gpu-latest
sub=$(mktemp -d -p "$(realpath "$(dirname "$0")/..")" .run.XXXX); trap 'rm -rf "$sub"' EXIT
chmod 777 "$sub"; ln "$ZIP" "$sub/submission.zip" 2>/dev/null || cp "$ZIP" "$sub/submission.zip"
start=$(date +%s)
docker run --rm --gpus all --network none "$@" \
  --mount type=bind,source="$DATA",target=/code_execution/data,readonly \
  --mount type=bind,source="$sub",target=/code_execution/submission \
  "$IMG" 2>&1 | grep -E '^\[|ERROR|Error|Traceback|exit code' | tail -40
echo "wall: $(( $(date +%s) - start ))s"
mkdir -p "$(dirname "$OUTCSV")"; cp "$sub/submission.csv" "$OUTCSV"; cp "$sub/log.txt" "${OUTCSV%.csv}.log"
