#!/usr/bin/env bash
# Run tests/ inside the official runtime image (it has torch; no GPU needed, network off).
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=lostintranscriptionprodacr.azurecr.io/lost-in-transcription-competition@sha256:7a311286889722f5867c4c44b76b4d5a2416b0a4303210d4f01d9aae0440ae3e
docker run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 \
  --mount type=bind,source="$PWD",target=/repo,readonly \
  --entrypoint bash "$IMG" -c 'cd /code_execution && uv run python -m unittest discover -s /repo/tests -v'
