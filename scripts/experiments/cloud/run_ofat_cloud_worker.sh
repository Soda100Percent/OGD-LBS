#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="scripts/experiments/configs/ofat_sweep_config.json"

exec python scripts/experiments/cloud/cloud_shard.py \
  --kind ofat \
  --config "$CONFIG" \
  "$@"
