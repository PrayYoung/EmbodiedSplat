#!/usr/bin/env bash
set -euo pipefail

SCENE=${1:-living_room}
VIDEO=${2:-scenes/${SCENE}/raw/video.mp4}
OUTDIR=${3:-scenes/${SCENE}/processed}

mkdir -p "$(dirname "$VIDEO")" "$OUTDIR"

uv run ns-process-data video \
  --data "$VIDEO" \
  --output-dir "$OUTDIR" \
  --matching-method exhaustive
