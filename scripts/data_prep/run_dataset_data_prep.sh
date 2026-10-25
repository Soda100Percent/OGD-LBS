#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Dataset-side Data Prep runner.
#
# Supported Data Prep models:
#   Griffin_A2
#   Winter_Samurai
#   All = Griffin_A2 + Winter_Samurai
#
# Recognized but intentionally unsupported by this runner:
#   Vroid_18
#   Vroid_19
#   Vroid_20
#
# Default dataset root:
#   /workspace/ogd_dataset
#
# This script is intentionally dataset-side / script-level.  It hard-codes
# model names and dataset layout, so it must not be moved into src/ogd_lbs/.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="/workspace/ogd_dataset"
CONDA_ENV="lao-lbs"
BPY5_ENV="ogd-bpy52"
BPY4_ENV="lao-lbs"

MODEL=""
DEBUG_FRAME_MAX=""
TARGET_FRAME_COUNT=""
NLA_GAP="0"
EXTRA_BONE_REDUCE_CONFIG_OVERRIDE=""
USE_MODEL_EXTRA_BONE_REDUCE_CONFIG=1
INCLUDE_UNMODIFIED_TEMPLATES=0
ARCHIVE_EXISTING=0
OVERWRITE_WORK_DIR=0
DRY_RUN=0

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_ROOT="/workspace/ogd_backups/data_prep_work_dir_archives"

print_usage() {
    cat <<'USAGE'
Usage:
  bash scripts/data_prep/run_dataset_data_prep.sh MODEL [options]

Models:
  Griffin_A2
  Winter_Samurai
  All

Recognized but not supported by this Data Prep runner:
  Vroid_18
  Vroid_19
  Vroid_20

Examples:
  bash scripts/data_prep/run_dataset_data_prep.sh Griffin_A2 --dry_run

  bash scripts/data_prep/run_dataset_data_prep.sh Griffin_A2 \
    --debug_frame_max 60 \
    --archive_existing

Options:
  --model MODEL
      Alternative to positional MODEL.

  --dataset_root PATH
      Default: /workspace/ogd_dataset

  --conda_env NAME
      Conda env for Prep 01 / 03 / 04a / 05.
      Default: lao-lbs

  --bpy5_env NAME
      Conda env for Prep 02a / Blender 5.x SDF fusion.
      Default: ogd-bpy52

  --bpy4_env NAME
      Conda env for Prep 02b / Blender 4.x import.
      Default: lao-lbs

  --debug_frame_max N
      Passed to Prep 05. Useful for debug exports.

  --target_frame_count N
      Passed to Prep 05. If > 0, uniformly retain N frames.
      This has higher priority than removal_rate.

  --nla_gap N
      Gap between Actions during Prep 01 NLA concatenation.
      Default: 0

  --extra_bone_reduce_config PATH
      Optional explicit extra bone-reduction config for Prep 04a.
      Only allowed for a single model run.

  --no_model_extra_bone_reduce_config
      Do not auto-use model-level config if present under scripts/data_prep/configs/.

  --include_unmodified_templates
      Passed to Prep 05.

  --archive_existing
      Move an existing non-empty data_prep_work_dir to:
        /workspace/ogd_backups/data_prep_work_dir_archives/

  --overwrite_work_dir
      Safely delete an existing data_prep_work_dir before running.

  --dry_run
      Print resolved paths and commands. Do not modify dataset files.

  -h, --help
      Show this help.
USAGE
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_value() {
    local key="$1"
    local value="${2:-}"
    if [[ -z "$value" ]]; then
        fail "$key requires a value."
    fi
}

path_abs_existing_dir() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        fail "Directory does not exist: $path"
    fi
    (cd "$path" && pwd -P)
}

path_abs_maybe_missing() {
    python - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

dir_is_nonempty() {
    local dir="$1"
    [[ -d "$dir" ]] && [[ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}

check_conda_env() {
    local env_name="$1"
    if ! command -v conda >/dev/null 2>&1; then
        fail "conda command not found in PATH."
    fi
    if ! conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
        fail "Conda env not found: $env_name"
    fi
}

shell_join() {
    printf ' %q' "$@"
    echo
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            require_value "$1" "${2:-}"
            MODEL="$2"
            shift 2
            ;;
        --dataset_root)
            require_value "$1" "${2:-}"
            DATASET_ROOT="$2"
            shift 2
            ;;
        --conda_env)
            require_value "$1" "${2:-}"
            CONDA_ENV="$2"
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
        --debug_frame_max)
            require_value "$1" "${2:-}"
            DEBUG_FRAME_MAX="$2"
            shift 2
            ;;
        --target_frame_count)
            require_value "$1" "${2:-}"
            TARGET_FRAME_COUNT="$2"
            shift 2
            ;;
        --nla_gap)
            require_value "$1" "${2:-}"
            NLA_GAP="$2"
            shift 2
            ;;
        --extra_bone_reduce_config)
            require_value "$1" "${2:-}"
            EXTRA_BONE_REDUCE_CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --no_model_extra_bone_reduce_config)
            USE_MODEL_EXTRA_BONE_REDUCE_CONFIG=0
            shift 1
            ;;
        --include_unmodified_templates)
            INCLUDE_UNMODIFIED_TEMPLATES=1
            shift 1
            ;;
        --archive_existing)
            ARCHIVE_EXISTING=1
            shift 1
            ;;
        --overwrite_work_dir)
            OVERWRITE_WORK_DIR=1
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
        --*)
            fail "Unknown option: $1"
            ;;
        *)
            if [[ -n "$MODEL" ]]; then
                fail "Multiple model arguments were provided: existing=$MODEL, extra=$1"
            fi
            MODEL="$1"
            shift 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    print_usage >&2
    fail "MODEL is required."
fi

if [[ "$ARCHIVE_EXISTING" -eq 1 && "$OVERWRITE_WORK_DIR" -eq 1 ]]; then
    fail "Use only one of --archive_existing or --overwrite_work_dir."
fi

case "$MODEL" in
    Griffin_A2)
        MODELS=("Griffin_A2")
        ;;
    Winter_Samurai)
        MODELS=("Winter_Samurai")
        ;;
    All|all|ALL)
        MODELS=("Griffin_A2" "Winter_Samurai")
        ;;
    Vroid_18|Vroid_19|Vroid_20)
        fail "$MODEL uses an old/preprocessed data0/target layout. This runner only supports Griffin_A2, Winter_Samurai, or All."
        ;;
    *)
        fail "Unsupported model: $MODEL. Expected Griffin_A2, Winter_Samurai, or All."
        ;;
esac

if [[ -n "$EXTRA_BONE_REDUCE_CONFIG_OVERRIDE" && "${#MODELS[@]}" -ne 1 ]]; then
    fail "--extra_bone_reduce_config can only be used with a single model, not All."
fi

DATASET_ROOT="$(path_abs_existing_dir "$DATASET_ROOT")"

check_conda_env "$CONDA_ENV"
check_conda_env "$BPY5_ENV"
check_conda_env "$BPY4_ENV"

resolve_model() {
    local model="$1"

    MODEL_NAME="$model"
    DATASET_DIR="${DATASET_ROOT}/${model}"
    WORK_DIR="${DATASET_DIR}/data_prep_work_dir"

    case "$model" in
        Griffin_A2)
            SOURCE_BLEND="${DATASET_DIR}/Griffin_A2.blend"
            STAGE_INPUT_BLEND_NAME="Griffin_A2.blend"
            CONFIG_PATH="${PROJECT_ROOT}/scripts/data_prep/configs/prep_02_sdf_merge_config_griffin_a2.json"
            MODEL_EXTRA_BONE_REDUCE_CONFIG="${PROJECT_ROOT}/scripts/data_prep/configs/extra_bone_reduce_griffin_a2.json"
            ;;
        Winter_Samurai)
            SOURCE_BLEND="${DATASET_DIR}/Winter_Samurai_Normalized.blend"
            STAGE_INPUT_BLEND_NAME="Winter_Samurai.blend"
            CONFIG_PATH="${PROJECT_ROOT}/scripts/data_prep/configs/prep_02_sdf_merge_config_winter_samurai.json"
            MODEL_EXTRA_BONE_REDUCE_CONFIG="${PROJECT_ROOT}/scripts/data_prep/configs/extra_bone_reduce_winter_samurai.json"
            ;;
        *)
            fail "Internal error: unsupported resolved model: $model"
            ;;
    esac

    STAGE_INPUT_BLEND="${WORK_DIR}/data_prep_00/${STAGE_INPUT_BLEND_NAME}"
    DP_CONFIG="${WORK_DIR}/prep_02_sdf_merge_config.json"
    COPIED_EXTRA_BONE_REDUCE_CONFIG="${WORK_DIR}/extra_bone_reduce_config.json"

    if [[ -n "${EXTRA_BONE_REDUCE_CONFIG_OVERRIDE}" ]]; then
        EXTRA_BONE_REDUCE_CONFIG="$(path_abs_maybe_missing "${EXTRA_BONE_REDUCE_CONFIG_OVERRIDE}")"
    elif [[ "$USE_MODEL_EXTRA_BONE_REDUCE_CONFIG" -eq 1 && -f "$MODEL_EXTRA_BONE_REDUCE_CONFIG" ]]; then
        EXTRA_BONE_REDUCE_CONFIG="$MODEL_EXTRA_BONE_REDUCE_CONFIG"
    else
        EXTRA_BONE_REDUCE_CONFIG=""
    fi

    OUT_01="${WORK_DIR}/data_prep_01/${MODEL_NAME}_prep_01.blend"
    OUT_02_MANIFEST="${WORK_DIR}/data_prep_02/sdf_exports/manifest.json"
    OUT_02_BLEND="${WORK_DIR}/data_prep_02/${MODEL_NAME}_prep_02.blend"
    OUT_03_SURFACE="${WORK_DIR}/data_prep_03/${MODEL_NAME}_prep_03_surface_drive_mode.blend"
    OUT_03_LBS="${WORK_DIR}/data_prep_03/${MODEL_NAME}_prep_03_lbs_drive_mode.blend"
    OUT_04="${WORK_DIR}/data_prep_04/${MODEL_NAME}_prep_04a_cleaned_full.blend"
    OUT_05_SUMMARY="${WORK_DIR}/data_prep_05/export_summary.json"
    OUT_05_VDB_SKEL="${WORK_DIR}/data_prep_05/fitbox_vdb/skeleton_hierarchy_data.pkl"
    OUT_05_VDB_ANIM="${WORK_DIR}/data_prep_05/fitbox_vdb/animation_data.pkl"
}

preflight_model() {
    local model="$1"
    resolve_model "$model"

    echo
    echo "--------------------------------------------------------------------------------"
    echo "Preflight: $MODEL_NAME"
    echo "--------------------------------------------------------------------------------"
    echo "dataset_root        : $DATASET_ROOT"
    echo "dataset_dir         : $DATASET_DIR"
    echo "source_blend        : $SOURCE_BLEND"
    echo "stage_input_blend   : $STAGE_INPUT_BLEND"
    echo "work_dir            : $WORK_DIR"
    echo "config              : $CONFIG_PATH"
    echo "copied config       : $DP_CONFIG"
    echo "model extra config  : $MODEL_EXTRA_BONE_REDUCE_CONFIG"
    echo "extra reduce config : ${EXTRA_BONE_REDUCE_CONFIG:-<unset>}"
    echo "copied extra config : $COPIED_EXTRA_BONE_REDUCE_CONFIG"
    echo "conda_env           : $CONDA_ENV"
    echo "bpy5_env            : $BPY5_ENV"
    echo "bpy4_env            : $BPY4_ENV"
    echo "debug_frame_max     : ${DEBUG_FRAME_MAX:-<unset>}"
    echo "target_frame_count  : ${TARGET_FRAME_COUNT:-<unset>}"
    echo "nla_gap             : $NLA_GAP"
    echo "include templates   : $INCLUDE_UNMODIFIED_TEMPLATES"
    echo "archive_existing    : $ARCHIVE_EXISTING"
    echo "overwrite_work_dir  : $OVERWRITE_WORK_DIR"
    echo "dry_run             : $DRY_RUN"

    [[ -d "$DATASET_DIR" ]] || fail "Dataset model directory not found: $DATASET_DIR"

    if [[ ! -f "$SOURCE_BLEND" ]]; then
        if [[ "$MODEL_NAME" == "Winter_Samurai" ]]; then
            fail "Winter_Samurai source blend not found: $SOURCE_BLEND. Run scripts/custom_model_clean/winter_samurai/run_dataset_cleaning.sh first."
        fi
        fail "Source blend not found: $SOURCE_BLEND"
    fi

    if [[ ! -f "$CONFIG_PATH" ]]; then
        if [[ "$MODEL_NAME" == "Winter_Samurai" ]]; then
            fail "Winter_Samurai Data Prep config is not ready. Expected: $CONFIG_PATH"
        fi
        fail "Data Prep config not found: $CONFIG_PATH"
    fi

    if [[ -n "$EXTRA_BONE_REDUCE_CONFIG" && ! -f "$EXTRA_BONE_REDUCE_CONFIG" ]]; then
        fail "Extra bone-reduction config not found: $EXTRA_BONE_REDUCE_CONFIG"
    fi

    if [[ -L "$WORK_DIR" ]]; then
        fail "Refusing to use symlink work_dir: $WORK_DIR"
    fi

    if [[ -e "$WORK_DIR" && ! -d "$WORK_DIR" ]]; then
        fail "work_dir exists but is not a directory: $WORK_DIR"
    fi

    if [[ "$DRY_RUN" -eq 0 ]]; then
        if dir_is_nonempty "$WORK_DIR" && [[ "$ARCHIVE_EXISTING" -eq 0 && "$OVERWRITE_WORK_DIR" -eq 0 ]]; then
            fail "Existing non-empty work_dir: $WORK_DIR. Use --archive_existing or --overwrite_work_dir."
        fi
    else
        if dir_is_nonempty "$WORK_DIR" && [[ "$ARCHIVE_EXISTING" -eq 0 && "$OVERWRITE_WORK_DIR" -eq 0 ]]; then
            echo "DRY_RUN NOTE: existing non-empty work_dir would fail in a real run unless --archive_existing or --overwrite_work_dir is used."
        fi
    fi
}

safe_remove_work_dir() {
    local model="$1"
    resolve_model "$model"

    local expected="${DATASET_ROOT}/${model}/data_prep_work_dir"
    local work_abs
    local expected_abs
    work_abs="$(path_abs_maybe_missing "$WORK_DIR")"
    expected_abs="$(path_abs_maybe_missing "$expected")"

    if [[ "$work_abs" != "$expected_abs" ]]; then
        fail "Refusing to remove unexpected work_dir: $WORK_DIR (expected $expected)"
    fi

    if [[ -L "$WORK_DIR" ]]; then
        fail "Refusing to remove symlink work_dir: $WORK_DIR"
    fi

    case "$WORK_DIR" in
        ""|"/"|"/workspace"|"/workspace/ogd_dataset"|"$DATASET_DIR")
            fail "Refusing unsafe work_dir removal: $WORK_DIR"
            ;;
    esac

    if [[ -d "$WORK_DIR" ]]; then
        echo "Removing existing work_dir: $WORK_DIR"
        rm -rf "$WORK_DIR"
    fi
}

archive_work_dir() {
    local model="$1"
    resolve_model "$model"

    if [[ ! -d "$WORK_DIR" ]] || ! dir_is_nonempty "$WORK_DIR"; then
        return
    fi

    if [[ -L "$WORK_DIR" ]]; then
        fail "Refusing to archive symlink work_dir: $WORK_DIR"
    fi

    mkdir -p "$ARCHIVE_ROOT"
    local target="${ARCHIVE_ROOT}/${model}_${RUN_TIMESTAMP}"

    if [[ -e "$target" ]]; then
        fail "Archive target already exists: $target"
    fi

    echo "Archiving existing work_dir:"
    echo "  from: $WORK_DIR"
    echo "  to  : $target"
    mv "$WORK_DIR" "$target"
}

write_git_state() {
    local out="$1"
    {
        echo "timestamp: $(date -Is)"
        echo "project_root: $PROJECT_ROOT"
        echo "git_head: $(git rev-parse HEAD 2>/dev/null || true)"
        echo
        echo "[git status --short]"
        git status --short || true
    } > "$out"
}

prepare_model() {
    local model="$1"
    resolve_model "$model"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo
        echo "[DRY_RUN] Would prepare work_dir for $MODEL_NAME:"
        echo "  mkdir -p $WORK_DIR/data_prep_00 $WORK_DIR/logs"
        echo "  cp -a $SOURCE_BLEND $STAGE_INPUT_BLEND"
        echo "  cp -a $CONFIG_PATH $DP_CONFIG"
        if [[ -n "$EXTRA_BONE_REDUCE_CONFIG" ]]; then
            echo "  cp -a $EXTRA_BONE_REDUCE_CONFIG $COPIED_EXTRA_BONE_REDUCE_CONFIG"
        fi
        return
    fi

    if [[ "$OVERWRITE_WORK_DIR" -eq 1 ]]; then
        safe_remove_work_dir "$model"
    elif [[ "$ARCHIVE_EXISTING" -eq 1 ]]; then
        archive_work_dir "$model"
    fi

    mkdir -p "$WORK_DIR/data_prep_00" "$WORK_DIR/logs"

    cp -a "$SOURCE_BLEND" "$STAGE_INPUT_BLEND"
    cp -a "$CONFIG_PATH" "$DP_CONFIG"

    if [[ -n "$EXTRA_BONE_REDUCE_CONFIG" ]]; then
        cp -a "$EXTRA_BONE_REDUCE_CONFIG" "$COPIED_EXTRA_BONE_REDUCE_CONFIG"
    fi

    {
        echo "model: $MODEL_NAME"
        echo "timestamp: $(date -Is)"
        echo "source_config: $CONFIG_PATH"
        echo "copied_config: $DP_CONFIG"
        echo "extra_bone_reduce_config: ${EXTRA_BONE_REDUCE_CONFIG:-<unset>}"
        echo "copied_extra_bone_reduce_config: ${COPIED_EXTRA_BONE_REDUCE_CONFIG}"
        echo "source_blend: $SOURCE_BLEND"
        echo "stage_input_blend: $STAGE_INPUT_BLEND"
        echo "project_root: $PROJECT_ROOT"
        echo "git_head: $(git rev-parse HEAD 2>/dev/null || true)"
    } > "$WORK_DIR/config_source.txt"

    sha256sum "$SOURCE_BLEND" "$STAGE_INPUT_BLEND" > "$WORK_DIR/source_blend_sha256.txt"

    HASH_INPUTS=(
        "scripts/data_prep/run_dataset_data_prep.sh"
        "scripts/data_prep/prep_01_concat_nla.py"
        "scripts/data_prep/prep_02_sdf_pipeline.sh"
        "scripts/data_prep/prep_02a_sdf_fuse_5x.py"
        "scripts/data_prep/prep_02b_import_fused_4x.py"
        "scripts/data_prep/prep_03_surface_deform.py"
        "scripts/data_prep/prep_04a_bone_pruning.py"
        "scripts/data_prep/prep_05_export_fitbox.py"
        "scripts/data_prep/paths.py"
        "scripts/data_prep/manifest.py"
        "scripts/data_prep/_env.py"
        "src/ogd_lbs/blender/blend_io.py"
        "src/ogd_lbs/utils/common.py"
        "src/ogd_lbs/utils/file_state.py"
        "$DP_CONFIG"
    )

    if [[ -n "$EXTRA_BONE_REDUCE_CONFIG" ]]; then
        HASH_INPUTS+=("$COPIED_EXTRA_BONE_REDUCE_CONFIG")
    fi

    sha256sum "${HASH_INPUTS[@]}" > "$WORK_DIR/code_config_sha256.txt"
    write_git_state "$WORK_DIR/git_state.txt"
}

run_step() {
    local label="$1"
    local log_path="$2"
    shift 2
    local cmd=("$@")

    echo
    echo "================================================================================"
    echo "Running step: $label"
    echo "Log: $log_path"
    echo "Command:"
    shell_join "${cmd[@]}"
    echo "================================================================================"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY_RUN] skipped $label"
        return
    fi

    mkdir -p "$(dirname "$log_path")"
    "${cmd[@]}" 2>&1 | tee "$log_path"
}

require_output_file() {
    local path="$1"
    local role="$2"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        return
    fi

    if [[ ! -f "$path" ]]; then
        fail "Missing expected output ${role}: $path"
    fi

    ls -lh "$path"
}

run_model() {
    local model="$1"
    resolve_model "$model"

    echo
    echo "################################################################################"
    echo "Data Prep model: $MODEL_NAME"
    echo "################################################################################"

    prepare_model "$model"

    CMD_01=(
        conda run --no-capture-output -n "$CONDA_ENV"
        python scripts/data_prep/prep_01_concat_nla.py
        --input_dir "$WORK_DIR"
        --gap "$NLA_GAP"
    )
    run_step "Prep 01: NLA concatenation" "$WORK_DIR/logs/01_prep_01_concat_nla.log" "${CMD_01[@]}"
    require_output_file "$OUT_01" "Prep 01 blend"

    CMD_02=(
        bash scripts/data_prep/prep_02_sdf_pipeline.sh
        --input_dir "$WORK_DIR"
        --config "$DP_CONFIG"
        --bpy5_env "$BPY5_ENV"
        --bpy4_env "$BPY4_ENV"
    )
    run_step "Prep 02: SDF fusion + import" "$WORK_DIR/logs/02_prep_02_sdf_pipeline.log" "${CMD_02[@]}"
    require_output_file "$OUT_02_MANIFEST" "Prep 02 manifest"
    require_output_file "$OUT_02_BLEND" "Prep 02 blend"

    CMD_03=(
        conda run --no-capture-output -n "$CONDA_ENV"
        python scripts/data_prep/prep_03_surface_deform.py
        --input_dir "$WORK_DIR"
        --drive_mode BOTH
    )
    run_step "Prep 03: Surface Deform + LBS drive modes" "$WORK_DIR/logs/03_prep_03_surface_deform.log" "${CMD_03[@]}"
    require_output_file "$OUT_03_SURFACE" "Prep 03 surface blend"
    require_output_file "$OUT_03_LBS" "Prep 03 LBS blend"

    CMD_04=(
        conda run --no-capture-output -n "$CONDA_ENV"
        python scripts/data_prep/prep_04a_bone_pruning.py
        --input_dir "$WORK_DIR"
    )

    if [[ -n "$EXTRA_BONE_REDUCE_CONFIG" ]]; then
        CMD_04+=(--extra_bone_reduce_config "$COPIED_EXTRA_BONE_REDUCE_CONFIG")
    fi
    run_step "Prep 04a: cleaned-full armature" "$WORK_DIR/logs/04_prep_04a_bone_pruning.log" "${CMD_04[@]}"
    require_output_file "$OUT_04" "Prep 04a cleaned-full blend"

    CMD_05=(
        conda run --no-capture-output -n "$CONDA_ENV"
        python scripts/data_prep/prep_05_export_fitbox.py
        --input_dir "$WORK_DIR"
    )

    if [[ -n "$TARGET_FRAME_COUNT" ]]; then
        CMD_05+=(--target_frame_count "$TARGET_FRAME_COUNT")
    fi

    if [[ -n "$DEBUG_FRAME_MAX" ]]; then
        CMD_05+=(--debug_frame_max "$DEBUG_FRAME_MAX")
    fi

    if [[ "$INCLUDE_UNMODIFIED_TEMPLATES" -eq 1 ]]; then
        CMD_05+=(--include_unmodified_templates)
    fi

    run_step "Prep 05: Fitbox export" "$WORK_DIR/logs/05_prep_05_export_fitbox.log" "${CMD_05[@]}"
    require_output_file "$OUT_05_SUMMARY" "Prep 05 export summary"
    require_output_file "$OUT_05_VDB_SKEL" "Prep 05 fitbox_vdb skeleton"
    require_output_file "$OUT_05_VDB_ANIM" "Prep 05 fitbox_vdb animation"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        cat > "$WORK_DIR/final_dataset_data_prep_summary.txt" <<EOF
model: $MODEL_NAME
timestamp: $(date -Is)
work_dir: $WORK_DIR
source_blend: $SOURCE_BLEND
stage_input_blend: $STAGE_INPUT_BLEND
config: $DP_CONFIG
debug_frame_max: ${DEBUG_FRAME_MAX:-<unset>}
target_frame_count: ${TARGET_FRAME_COUNT:-<unset>}
nla_gap: $NLA_GAP
extra_bone_reduce_config: ${EXTRA_BONE_REDUCE_CONFIG:-<unset>}
copied_extra_bone_reduce_config: $COPIED_EXTRA_BONE_REDUCE_CONFIG
fitbox_vdb: $WORK_DIR/data_prep_05/fitbox_vdb
EOF
    fi

    echo
    echo "================================================================================"
    echo "Data Prep done for $MODEL_NAME"
    echo "work_dir   : $WORK_DIR"
    echo "fitbox_vdb : $WORK_DIR/data_prep_05/fitbox_vdb"
    echo "================================================================================"
}

echo "================================================================================"
echo "Dataset-side Data Prep runner"
echo "PROJECT_ROOT : $PROJECT_ROOT"
echo "DATASET_ROOT : $DATASET_ROOT"
echo "MODEL        : $MODEL"
echo "MODELS       : ${MODELS[*]}"
echo "DRY_RUN      : $DRY_RUN"
echo "================================================================================"

echo
echo "Preflight all requested models before running anything..."
for model in "${MODELS[@]}"; do
    preflight_model "$model"
done

echo
echo "Preflight complete."

for model in "${MODELS[@]}"; do
    run_model "$model"
done

echo
echo "================================================================================"
echo "All requested Data Prep runs completed."
echo "Models: ${MODELS[*]}"
echo "================================================================================"
