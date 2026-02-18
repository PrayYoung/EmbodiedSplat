#!/usr/bin/env bash
set -euo pipefail

SCENE=${1:-living_room}
DATA=${2:-scenes/${SCENE}/processed}
RUNS=${3:-scenes/${SCENE}/runs}

mkdir -p "$RUNS"

uv run ns-train splatfacto --data "$DATA" --output-dir "$RUNS"
