#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${OGD_STAGE_PYTHON:?Set OGD_STAGE_PYTHON to the approved absolute Python executable}"

case "${OGD_STAGE_PYTHON}" in
  /*) ;;
  *) echo "OGD_STAGE_PYTHON must be absolute" >&2; exit 2 ;;
esac

test -x "${OGD_STAGE_PYTHON}" || {
  echo "OGD_STAGE_PYTHON is not executable" >&2
  exit 2
}

ADAPTER="$(cd -- "${HERE}/.." && pwd -P)/stage_adapter.py"
test -f "${ADAPTER}" || {
  echo "stage_adapter.py not found: ${ADAPTER}" >&2
  exit 2
}

export PYTHONDONTWRITEBYTECODE=1
exec "${OGD_STAGE_PYTHON}" \
  "${ADAPTER}" \
  --stage-id step02 \
  "$@"
