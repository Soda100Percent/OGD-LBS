#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Prep 02 full pipeline wrapper:
#   02a: Blender 5.x SDF Grid fusion
#   02b: Blender 4.x import fused OBJ and create Source_<VDB>
#
# Staged data-prep behavior:
#
#   --input_dir <data_prep_work_dir>
#       reads : <data_prep_work_dir>/data_prep_01/*_prep_01.blend
#       writes: <data_prep_work_dir>/data_prep_02/
#
#   --input_dir <data_prep_work_dir>/data_prep_01
#       reads : that stage input dir
#       writes: <data_prep_work_dir>/data_prep_02/
#
#   --input_dir <custom_input_tree> --output_dir <custom_output_tree>
#       reads : <custom_input_tree>/*_prep_01.blend
#       writes: <custom_output_tree>/
#
# This wrapper delegates actual work to:
#   scripts/data_prep/prep_02a_sdf_fuse_5x.py
#   scripts/data_prep/prep_02b_import_fused_4x.py
#
# Both sub-steps are called in staged directory mode.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

BPY5_ENV="ogd-bpy52"
BPY4_ENV="lao-lbs"

INPUT_DIR=""
OUTPUT_DIR=""
CONFIG=""

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
      --input_dir /path/to/data_prep_work_dir \
      [--output_dir /path/to/custom_data_prep_02] \
      [--config /path/to/prep_02_sdf_merge_config.json] \
      [--bpy5_env ogd-bpy52] \
      [--bpy4_env lao-lbs] \
      [Prep 02a options...]

Required:
  --input_dir PATH
      Data-prep work_dir or data_prep_01 stage directory.

      If PATH is work_dir:
          input  = PATH/data_prep_01
          output = PATH/data_prep_02

      If PATH is work_dir/data_prep_01:
          input  = PATH
          output = work_dir/data_prep_02

      If --output_dir is explicit and PATH is not a standard stage dir:
          input  = PATH
          output = --output_dir

Environment:
  --bpy5_env NAME
      Conda environment for Prep 02a / Blender 5.x SDF Grid.
      Default: ogd-bpy52

  --bpy4_env NAME
      Conda environment for Prep 02b / Blender 4.x mainline processing.
      Default: lao-lbs

Path options:
  --output_dir PATH
      Optional explicit output tree.
      If omitted, writes to <work_dir>/data_prep_02.

  --config PATH
      Optional Prep 02a SDF config override.
      If omitted, Prep 02a uses:
          <work_dir>/prep_02_sdf_merge_config.json

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

Example:
  bash scripts/data_prep/prep_02_sdf_pipeline.sh \
      --input_dir "$DP" \
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
        --input_dir|-i)
            require_value "$1" "${2:-}"
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_dir|-o)
            require_value "$1" "${2:-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            require_value "$1" "${2:-}"
            CONFIG="$2"
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

if [[ -z "${INPUT_DIR}" ]]; then
    echo "ERROR: --input_dir is required." >&2
    echo >&2
    print_usage >&2
    exit 2
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

# Resolve staged paths using scripts/data_prep/paths.py.
eval "$(
python - "${INPUT_DIR}" "${OUTPUT_DIR}" "${CONFIG}" << 'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

repo = Path.cwd()
src = repo / "src"
data_prep = repo / "scripts" / "data_prep"

for item in [src, data_prep]:
    item_s = str(item)
    if item_s not in sys.path:
        sys.path.insert(0, item_s)

from paths import (  # noqa: E402
    resolve_data_prep_stage_dirs,
    validate_data_prep_stage_resolution,
    discover_unique_blend,
    extract_asset_base_stem,
    prep_02b_output_blend_name,
)

input_dir_arg = sys.argv[1]
output_dir_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].strip() else None
config_arg = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].strip() else None

res = resolve_data_prep_stage_dirs(
    step_id="02",
    input_dir=input_dir_arg,
    output_dir=output_dir_arg,
    step_title="Prep 02: SDF Fusion + Import",
)
validate_data_prep_stage_resolution(res)

input_blend = discover_unique_blend(
    res.input_tree,
    pattern="*_prep_01.blend",
    role="Prep 02 input blend",
)

base_stem = extract_asset_base_stem(input_blend)

if config_arg:
    config_path = Path(config_arg).expanduser().resolve(strict=False)
    config_source = "EXPLICIT_OVERRIDE"
else:
    config_path = (res.work_dir / "prep_02_sdf_merge_config.json").resolve(strict=False)
    config_source = "WORK_DIR_DEFAULT"

out_dir = (res.output_tree / "sdf_exports").resolve(strict=False)
manifest_path = (out_dir / "manifest.json").resolve(strict=False)
output_blend = (res.output_tree / prep_02b_output_blend_name(base_stem)).resolve(strict=False)

items = {
    "RES_STEP_ID": res.step_id,
    "RES_MODE": res.mode,
    "RES_WORK_DIR": str(res.work_dir),
    "RES_INPUT_TREE": str(res.input_tree),
    "RES_OUTPUT_TREE": str(res.output_tree),
    "RES_INPUT_SOURCE": res.input_source,
    "RES_OUTPUT_SOURCE": res.output_source,
    "RES_INPUT_STAGE_DIR": res.input_stage_dir_name,
    "RES_OUTPUT_STAGE_DIR": res.output_stage_dir_name,
    "RES_INPUT_BLEND": str(input_blend),
    "RES_BASE_STEM": str(base_stem),
    "RES_CONFIG": str(config_path),
    "RES_CONFIG_SOURCE": config_source,
    "RES_OUT_DIR": str(out_dir),
    "RES_MANIFEST": str(manifest_path),
    "RES_OUTPUT_BLEND": str(output_blend),
}

for key, value in items.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

if [[ ! -f "${RES_INPUT_BLEND}" ]]; then
    echo "ERROR: resolved input blend does not exist: ${RES_INPUT_BLEND}" >&2
    exit 1
fi

if [[ ! -f "${RES_CONFIG}" ]]; then
    echo "ERROR: resolved config does not exist: ${RES_CONFIG} (${RES_CONFIG_SOURCE})" >&2
    exit 1
fi

mkdir -p "${RES_OUTPUT_TREE}" "${RES_OUT_DIR}"

echo "================================================================================"
echo "Prep 02 SDF pipeline"
echo "PROJECT_ROOT        : ${PROJECT_ROOT}"
echo "step_id             : ${RES_STEP_ID}"
echo "mode                : ${RES_MODE}"
echo "work_dir            : ${RES_WORK_DIR}"
echo "input_tree          : ${RES_INPUT_TREE} (${RES_INPUT_SOURCE})"
echo "output_tree         : ${RES_OUTPUT_TREE} (${RES_OUTPUT_SOURCE})"
echo "input_stage_dir     : ${RES_INPUT_STAGE_DIR}"
echo "output_stage_dir    : ${RES_OUTPUT_STAGE_DIR}"
echo "input_blend         : ${RES_INPUT_BLEND}"
echo "base_stem           : ${RES_BASE_STEM}"
echo "config              : ${RES_CONFIG} (${RES_CONFIG_SOURCE})"
echo "out_dir             : ${RES_OUT_DIR}"
echo "manifest_for_02b    : ${RES_MANIFEST}"
echo "output_blend        : ${RES_OUTPUT_BLEND}"
echo "bpy5_env            : ${BPY5_ENV}"
echo "bpy4_env            : ${BPY4_ENV}"
echo "dry_run             : ${DRY_RUN}"
echo "================================================================================"

PREP02A_CMD=(
    conda run --no-capture-output -n "${BPY5_ENV}"
    python scripts/data_prep/prep_02a_sdf_fuse_5x.py
    --input_dir "${RES_INPUT_TREE}"
    --output_dir "${RES_OUTPUT_TREE}"
    --config "${RES_CONFIG}"
)

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
    --input_dir "${RES_INPUT_TREE}"
    --output_dir "${RES_OUTPUT_TREE}"
)

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
echo "  ${RES_MANIFEST}"
echo "================================================================================"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ ! -f "${RES_MANIFEST}" ]]; then
        echo "ERROR: manifest was not created or not found: ${RES_MANIFEST}" >&2
        exit 1
    fi
    ls -lh "${RES_MANIFEST}"
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
echo "Checking output blend after Prep 02b:"
echo "  ${RES_OUTPUT_BLEND}"
echo "================================================================================"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ ! -f "${RES_OUTPUT_BLEND}" ]]; then
        echo "ERROR: output blend was not created or not found: ${RES_OUTPUT_BLEND}" >&2
        exit 1
    fi
    ls -lh "${RES_OUTPUT_BLEND}"
else
    echo "[DRY_RUN] skipped output blend existence check"
fi

echo
echo "================================================================================"
echo "Prep 02 SDF pipeline done."
echo "work_dir     : ${RES_WORK_DIR}"
echo "input_blend  : ${RES_INPUT_BLEND}"
echo "manifest     : ${RES_MANIFEST}"
echo "output_blend : ${RES_OUTPUT_BLEND}"
echo "================================================================================"
