#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Prep 02 full pipeline wrapper:
#   02a: Blender 5.x SDF Grid fusion
#   02b: Blender 4.x import fused OBJ and create Source_<VDB>
#
# This wrapper intentionally stays in scripts/data_prep because it is a CLI
# orchestration layer, not reusable library code.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

BPY5_ENV="ogd-bpy52"
BPY4_ENV="lao-lbs"

INPUT_BLEND=""
CONFIG=""
OUT_DIR=""
OUTPUT_BLEND=""

TARGET_FRAME=""
VOXEL_SIZE=""
OFFSET=""
AUTO_OFFSET_MODE=""   # "", "auto", "no_auto"

MAX_OFFSET_MULTIPLIER=""
BAND_WIDTH_PADDING=""
MIN_COMPONENT_FACES=""
MIN_COMPONENT_FACE_RATIO=""
REQUIRE_WATERTIGHT=0
NO_APPLY_SCALE=0

DRY_RUN=0

print_usage() {
    cat << 'USAGE'
Usage:
  bash scripts/data_prep/prep_02_sdf_pipeline.sh \
      --input_blend /path/to/Asset_prep_01.blend \
      [--config /path/to/prep_02_sdf_merge_config.json] \
      [--out_dir /path/to/sdf_exports] \
      [--output_blend /path/to/Asset_prep_02.blend] \
      [--bpy5_env ogd-bpy52] \
      [--bpy4_env lao-lbs] \
      [Prep 02a options...]

Required:
  --input_blend PATH
      Input Prep 01 .blend.

Environment:
  --bpy5_env NAME
      Conda environment for Prep 02a / Blender 5.x SDF Grid.
      Default: ogd-bpy52

  --bpy4_env NAME
      Conda environment for Prep 02b / Blender 4.x mainline processing.
      Default: lao-lbs

Path options:
  --config PATH
      Prep 02a SDF config.
      If omitted, Prep 02a uses:
          <work_dir>/prep_02_sdf_merge_config.json

  --out_dir PATH
      Prep 02a export directory.
      If omitted, Prep 02a uses:
          <work_dir>/<base_stem>_prep_01_sdf_exports/

  --output_blend PATH
      Prep 02b output blend.
      If omitted, Prep 02b uses:
          <work_dir>/<base_stem>_prep_02.blend

Prep 02a runtime options:
  --target_frame N
  --max_offset_multiplier X
  --band_width_padding N
  --min_component_faces N
  --min_component_face_ratio X
  --require_watertight
  --no_apply_scale

Prep 02a config override options:
  --voxel_size X
  --offset X
  --auto_offset
  --no_auto_offset

Other:
  --dry_run
      Print commands but do not execute.

Examples:

  # Explicit test paths, recommended during migration:
  bash scripts/data_prep/prep_02_sdf_pipeline.sh \
      --input_blend /workspace/data/dataset/Griffin_A2/Blender/Griffin_A2_prep_01_stage02_test.blend \
      --config configs/prep_02_sdf_merge_config_griffin.json \
      --out_dir /workspace/data/dataset/Griffin_A2/Blender/Griffin_A2_prep_01_stage03_sdf_exports_pipeline \
      --output_blend /workspace/data/dataset/Griffin_A2/Blender/Griffin_A2_prep_02_stage03_pipeline_test.blend \
      --offset 15 \
      --no_auto_offset

  # Folder-convention mode:
  # Put prep_02_sdf_merge_config.json next to Asset_prep_01.blend.
  bash scripts/data_prep/prep_02_sdf_pipeline.sh \
      --input_blend /path/to/Asset_prep_01.blend \
      --offset 15 \
      --no_auto_offset
USAGE
}

require_value() {
    local key="$1"
    local value="${2:-}"
    if [[ -z "${value}" ]]; then
        echo "ERROR: ${key} requires a value." >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_blend)
            require_value "$1" "${2:-}"
            INPUT_BLEND="$2"
            shift 2
            ;;
        --config)
            require_value "$1" "${2:-}"
            CONFIG="$2"
            shift 2
            ;;
        --out_dir)
            require_value "$1" "${2:-}"
            OUT_DIR="$2"
            shift 2
            ;;
        --output_blend)
            require_value "$1" "${2:-}"
            OUTPUT_BLEND="$2"
            shift 2
            ;;
        --bpy5_env)
            require_value "$1" "${2:-}"
            BPY5_ENV="$2"
            shift 2
            ;;
        --bpy4_env)
            require_value "$1" "${2:-}"
            BPY4_ENV="$2"
            shift 2
            ;;

        --target_frame)
            require_value "$1" "${2:-}"
            TARGET_FRAME="$2"
            shift 2
            ;;
        --voxel_size)
            require_value "$1" "${2:-}"
            VOXEL_SIZE="$2"
            shift 2
            ;;
        --offset)
            require_value "$1" "${2:-}"
            OFFSET="$2"
            shift 2
            ;;
        --auto_offset)
            AUTO_OFFSET_MODE="auto"
            shift 1
            ;;
        --no_auto_offset)
            AUTO_OFFSET_MODE="no_auto"
            shift 1
            ;;
        --max_offset_multiplier)
            require_value "$1" "${2:-}"
            MAX_OFFSET_MULTIPLIER="$2"
            shift 2
            ;;
        --band_width_padding)
            require_value "$1" "${2:-}"
            BAND_WIDTH_PADDING="$2"
            shift 2
            ;;
        --min_component_faces)
            require_value "$1" "${2:-}"
            MIN_COMPONENT_FACES="$2"
            shift 2
            ;;
        --min_component_face_ratio)
            require_value "$1" "${2:-}"
            MIN_COMPONENT_FACE_RATIO="$2"
            shift 2
            ;;
        --require_watertight)
            REQUIRE_WATERTIGHT=1
            shift 1
            ;;
        --no_apply_scale)
            NO_APPLY_SCALE=1
            shift 1
            ;;
        --dry_run)
            DRY_RUN=1
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo >&2
            print_usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${INPUT_BLEND}" ]]; then
    echo "ERROR: --input_blend is required." >&2
    echo >&2
    print_usage >&2
    exit 2
fi

if [[ ! -f "${INPUT_BLEND}" ]]; then
    echo "ERROR: input blend does not exist: ${INPUT_BLEND}" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found in PATH." >&2
    echo "Please run this from a shell where conda is available." >&2
    exit 1
fi

# Validate environment names early.
if ! conda env list | awk '{print $1}' | grep -Fxq "${BPY5_ENV}"; then
    echo "ERROR: Conda env not found for Blender 5.x / Prep 02a: ${BPY5_ENV}" >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${BPY4_ENV}"; then
    echo "ERROR: Conda env not found for Blender 4.x / Prep 02b: ${BPY4_ENV}" >&2
    exit 1
fi

# Resolve expected manifest path. If --out_dir is explicit, manifest is there.
# Otherwise mirror scripts/data_prep/paths.py base-stem convention.
MANIFEST_PATH="$(
python - "${INPUT_BLEND}" "${OUT_DIR}" << 'PY'
from __future__ import annotations
from pathlib import Path
import sys

input_blend = Path(sys.argv[1]).expanduser().resolve(strict=False)
out_dir_arg = sys.argv[2]

if out_dir_arg:
    print(str((Path(out_dir_arg).expanduser().resolve(strict=False) / "manifest.json")))
    raise SystemExit(0)

suffixes = (
    "_prep_04a_bone_pruning",
    "_prep_03_LBS_baseline",
    "_prep_03",
    "_prep_02",
    "_prep_01",
)

stem = input_blend.stem
for suffix in sorted(suffixes, key=len, reverse=True):
    if stem.endswith(suffix):
        stem = stem[:-len(suffix)]
        break

print(str((input_blend.parent / f"{stem}_prep_01_sdf_exports" / "manifest.json").resolve(strict=False)))
PY
)"

echo "================================================================================"
echo "Prep 02 SDF pipeline"
echo "PROJECT_ROOT        : ${PROJECT_ROOT}"
echo "input_blend         : ${INPUT_BLEND}"
echo "config              : ${CONFIG:-<Prep 02a default>}"
echo "out_dir             : ${OUT_DIR:-<Prep 02a default>}"
echo "manifest_for_02b    : ${MANIFEST_PATH}"
echo "output_blend        : ${OUTPUT_BLEND:-<Prep 02b default>}"
echo "bpy5_env            : ${BPY5_ENV}"
echo "bpy4_env            : ${BPY4_ENV}"
echo "================================================================================"

PREP02A_CMD=(
    conda run --no-capture-output -n "${BPY5_ENV}"
    python scripts/data_prep/prep_02a_sdf_fuse_5x.py
    --input_blend "${INPUT_BLEND}"
)

if [[ -n "${CONFIG}" ]]; then
    PREP02A_CMD+=(--config "${CONFIG}")
fi
if [[ -n "${OUT_DIR}" ]]; then
    PREP02A_CMD+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${TARGET_FRAME}" ]]; then
    PREP02A_CMD+=(--target_frame "${TARGET_FRAME}")
fi
if [[ -n "${VOXEL_SIZE}" ]]; then
    PREP02A_CMD+=(--voxel_size "${VOXEL_SIZE}")
fi
if [[ -n "${OFFSET}" ]]; then
    PREP02A_CMD+=(--offset "${OFFSET}")
fi
if [[ "${AUTO_OFFSET_MODE}" == "auto" ]]; then
    PREP02A_CMD+=(--auto_offset)
elif [[ "${AUTO_OFFSET_MODE}" == "no_auto" ]]; then
    PREP02A_CMD+=(--no_auto_offset)
fi
if [[ -n "${MAX_OFFSET_MULTIPLIER}" ]]; then
    PREP02A_CMD+=(--max_offset_multiplier "${MAX_OFFSET_MULTIPLIER}")
fi
if [[ -n "${BAND_WIDTH_PADDING}" ]]; then
    PREP02A_CMD+=(--band_width_padding "${BAND_WIDTH_PADDING}")
fi
if [[ -n "${MIN_COMPONENT_FACES}" ]]; then
    PREP02A_CMD+=(--min_component_faces "${MIN_COMPONENT_FACES}")
fi
if [[ -n "${MIN_COMPONENT_FACE_RATIO}" ]]; then
    PREP02A_CMD+=(--min_component_face_ratio "${MIN_COMPONENT_FACE_RATIO}")
fi
if [[ "${REQUIRE_WATERTIGHT}" -eq 1 ]]; then
    PREP02A_CMD+=(--require_watertight)
fi
if [[ "${NO_APPLY_SCALE}" -eq 1 ]]; then
    PREP02A_CMD+=(--no_apply_scale)
fi

PREP02B_CMD=(
    conda run --no-capture-output -n "${BPY4_ENV}"
    python scripts/data_prep/prep_02b_import_fused_4x.py
    --input_blend "${INPUT_BLEND}"
    --manifest "${MANIFEST_PATH}"
)

if [[ -n "${OUTPUT_BLEND}" ]]; then
    PREP02B_CMD+=(--output_blend "${OUTPUT_BLEND}")
fi

echo
echo "================================================================================"
echo "Step 02a command:"
printf ' %q' "${PREP02A_CMD[@]}"
echo
echo "================================================================================"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${PREP02A_CMD[@]}"
else
    echo "[DRY_RUN] skipped Prep 02a"
fi

echo
echo "================================================================================"
echo "Checking manifest after Prep 02a:"
echo "  ${MANIFEST_PATH}"
echo "================================================================================"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ ! -f "${MANIFEST_PATH}" ]]; then
        echo "ERROR: manifest was not created or not found: ${MANIFEST_PATH}" >&2
        exit 1
    fi
    ls -lh "${MANIFEST_PATH}"
else
    echo "[DRY_RUN] skipped manifest existence check"
fi

echo
echo "================================================================================"
echo "Step 02b command:"
printf ' %q' "${PREP02B_CMD[@]}"
echo
echo "================================================================================"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${PREP02B_CMD[@]}"
else
    echo "[DRY_RUN] skipped Prep 02b"
fi

echo
echo "================================================================================"
echo "Prep 02 SDF pipeline done."
echo "manifest     : ${MANIFEST_PATH}"
if [[ -n "${OUTPUT_BLEND}" ]]; then
    echo "output_blend : ${OUTPUT_BLEND}"
else
    echo "output_blend : <Prep 02b default>"
fi
echo "================================================================================"
