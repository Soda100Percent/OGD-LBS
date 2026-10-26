#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# Dataset-side SSDR runner for OGD-LBS.
#
# Supported models:
#   Vroid_18
#   Vroid_19
#   Vroid_20
#   Griffin_A2
#   Winter_Samurai
#   All
#
# Layouts:
#
#   Legacy VRoid layout:
#       /workspace/ogd_dataset/Vroid_18/data0
#       -> writes data1/data2/data3/data4 under /workspace/ogd_dataset/Vroid_18/
#
#   Data Prep handoff layout:
#       /workspace/ogd_dataset/<MODEL>/ssdr_work_dir/data0_json
#       -> writes data0/data1/data2/data3/data4 under ssdr_work_dir/
#
# This script intentionally contains dataset-side orchestration only.
# SSDR algorithms remain in scripts/ssdr/00-04 and src/ogd_lbs/ssdr/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET_ROOT="/workspace/ogd_dataset"
CONDA_ENV="lao-lbs"
ARCHIVE_ROOT="/workspace/ogd_backups/ssdr_work_dir_archives"

MODEL=""
DRY_RUN=0
ARCHIVE_EXISTING=0
OVERWRITE_OUTPUT=0

MAX_WORKERS="1"
QEM_MAX_WORKERS=""
QEM_CONFIG_OVERRIDE=""
VARIANT=""

MAX_FRAMES="210"
TARGET_FRAMES="200"

ITER="1"
PRINT_EVERY_N="1"
ORG_TYPE="3"
WEIGHT_SOLVER="legacy"
BT_COVARIANCE_MODE="corrected"

OPT_WEIGHT=0
INPUT_POSES=0
USE_WEIGHT_MASK=0
USE_WEIGHT_SMOOTHNESS=0
EVAL_MESH_HEALTH=0
EVAL_BONE_HEALTH=0
VALIDATE_OUTPUT=0
WEAK_VALIDATE=0
SAVE_INTERMEDIATE=0

LAMBDA_REG="0.1"
WEIGHTS_SMOOTH_STEP="1.0"
WEIGHT_SOFT_LOCK="0.0"
TRANS_AFFINE="10.0"
TRANS_AFFINE_NORM="4.0"
ROLL_AFFINITY="0.0"
ROLL_AFFINITY_NORM="2.0"
HEALTH_LEVEL="1"
HEALTH_EVERY="5"
HEALTH_WEIGHT_SCHEME="uniform"
BONE_HEALTH_EVERY="5"

RUN_STEP04=1
FRAME_REMOVAL_IDX_RANGES=""
FRAME_SAMPLING_CONFIG=""
HANDOFF_LEGACY_DATA0=0
LEGACY_SOURCE_DATA0=""
HANDOFF_SOURCE_DIR=""
HANDOFF_SOURCE_KIND=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ssdr/run_dataset_ssdr.sh <MODEL|All> [options]

Models:
  Vroid_18
  Vroid_19
  Vroid_20
  Griffin_A2
  Winter_Samurai
  All

Common options:
  --dataset_root PATH          Default: /workspace/ogd_dataset
  --conda_env NAME             Default: lao-lbs
  --archive_existing           Move generated stage/log/metric dirs to archive before running
  --archive_root PATH          Default: /workspace/ogd_backups/ssdr_work_dir_archives
  --overwrite_output           Pass --overwrite_output to SSDR Step 00-04
  --max_workers N              Default: 1
  --dry_run                    Print commands without executing them

Step 01 frame sampling:
  --max_frames N               Default: 210
  --target_frames N            Default: 200

Step 02:
  --variant NAME               Optional reduced skeleton variant for new vdb-root layout

Step 03 QEM:
  --qem_config PATH            Override model QEM config
  --qem_max_workers N          Override Step 03 workers only; default uses --max_workers

Step 04 optimization:
  --no_step04                  Run only Step 00-03
  --iter N                     Default: 1
  --print_every_n N            Default: 1
  --org_type N                 Default: 3
  --opt_weight
  --input_poses
  --use_weight_mask
  --weight_solver legacy|simplex
  --use_weight_smoothness
  --weights_smooth_step FLOAT  Default: 1.0
  --weight_soft_lock FLOAT     Default: 0.0
  --lambda_reg FLOAT           Default: 0.1
  --bt_covariance_mode legacy|corrected
  --trans_affine FLOAT         Default: 10.0
  --trans_affine_norm FLOAT    Default: 4.0
  --roll_affinity FLOAT        Default: 0.0
  --roll_affinity_norm FLOAT   Default: 2.0
  --eval_mesh_health
  --health_level N             Default: 1
  --health_every N             Default: 5
  --health_weight_scheme NAME  Default: uniform
  --eval_bone_health
  --bone_health_every N        Default: 5
  --validate_output
  --weak_validate
  --save_intermediate

Examples:
  bash scripts/ssdr/run_dataset_ssdr.sh Vroid_18 --dry_run

  bash scripts/ssdr/run_dataset_ssdr.sh Vroid_18 \
    --archive_existing \
    --overwrite_output \
    --iter 1 \
    --max_workers 1

  bash scripts/ssdr/run_dataset_ssdr.sh All --dry_run
EOF
}

log() {
  printf '[%s][run_dataset_ssdr] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

is_supported_model() {
  case "$1" in
    Vroid_18|Vroid_19|Vroid_20|Griffin_A2|Winter_Samurai) return 0 ;;
    *) return 1 ;;
  esac
}

parse_args() {
  if [ "$#" -lt 1 ]; then
    usage
    exit 2
  fi

  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
  esac

  MODEL="$1"
  shift

  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;

      --dataset_root)
        DATASET_ROOT="$2"; shift 2 ;;
      --conda_env)
        CONDA_ENV="$2"; shift 2 ;;
      --archive_existing)
        ARCHIVE_EXISTING=1; shift ;;
      --archive_root)
        ARCHIVE_ROOT="$2"; shift 2 ;;
      --overwrite_output)
        OVERWRITE_OUTPUT=1; shift ;;
      --max_workers)
        MAX_WORKERS="$2"; shift 2 ;;
      --dry_run)
        DRY_RUN=1; shift ;;

      --max_frames)
        MAX_FRAMES="$2"; shift 2 ;;
      --target_frames)
        TARGET_FRAMES="$2"; shift 2 ;;

      --variant)
        VARIANT="$2"; shift 2 ;;

      --qem_config)
        QEM_CONFIG_OVERRIDE="$2"; shift 2 ;;
      --qem_max_workers)
        QEM_MAX_WORKERS="$2"; shift 2 ;;

      --no_step04)
        RUN_STEP04=0; shift ;;
      --iter)
        ITER="$2"; shift 2 ;;
      --print_every_n)
        PRINT_EVERY_N="$2"; shift 2 ;;
      --org_type)
        ORG_TYPE="$2"; shift 2 ;;
      --opt_weight)
        OPT_WEIGHT=1; shift ;;
      --input_poses)
        INPUT_POSES=1; shift ;;
      --use_weight_mask)
        USE_WEIGHT_MASK=1; shift ;;
      --weight_solver)
        WEIGHT_SOLVER="$2"; shift 2 ;;
      --use_weight_smoothness)
        USE_WEIGHT_SMOOTHNESS=1; shift ;;
      --weights_smooth_step)
        WEIGHTS_SMOOTH_STEP="$2"; shift 2 ;;
      --weight_soft_lock)
        WEIGHT_SOFT_LOCK="$2"; shift 2 ;;
      --lambda_reg)
        LAMBDA_REG="$2"; shift 2 ;;
      --bt_covariance_mode)
        BT_COVARIANCE_MODE="$2"; shift 2 ;;
      --trans_affine)
        TRANS_AFFINE="$2"; shift 2 ;;
      --trans_affine_norm)
        TRANS_AFFINE_NORM="$2"; shift 2 ;;
      --roll_affinity)
        ROLL_AFFINITY="$2"; shift 2 ;;
      --roll_affinity_norm)
        ROLL_AFFINITY_NORM="$2"; shift 2 ;;
      --eval_mesh_health)
        EVAL_MESH_HEALTH=1; shift ;;
      --health_level)
        HEALTH_LEVEL="$2"; shift 2 ;;
      --health_every)
        HEALTH_EVERY="$2"; shift 2 ;;
      --health_weight_scheme)
        HEALTH_WEIGHT_SCHEME="$2"; shift 2 ;;
      --eval_bone_health)
        EVAL_BONE_HEALTH=1; shift ;;
      --bone_health_every)
        BONE_HEALTH_EVERY="$2"; shift 2 ;;
      --validate_output)
        VALIDATE_OUTPUT=1; shift ;;
      --weak_validate)
        WEAK_VALIDATE=1; shift ;;
      --save_intermediate)
        SAVE_INTERMEDIATE=1; shift ;;

      *)
        die "Unknown option: $1"
        ;;
    esac
  done

  if [ "$MODEL" != "All" ] && ! is_supported_model "$MODEL"; then
    die "Unsupported model: $MODEL"
  fi

  if [ -z "$QEM_MAX_WORKERS" ]; then
    QEM_MAX_WORKERS="$MAX_WORKERS"
  fi
}

model_list_for_target() {
  if [ "$MODEL" = "All" ]; then
    printf '%s\n' Vroid_18 Vroid_19 Vroid_20 Griffin_A2 Winter_Samurai
  else
    printf '%s\n' "$MODEL"
  fi
}

conda_python_prefix() {
  printf 'conda run --no-capture-output -n %s python\n' "$CONDA_ENV"
}

append_overwrite_arg() {
  local -n arr_ref="$1"
  if [ "$OVERWRITE_OUTPUT" -eq 1 ]; then
    arr_ref+=(--overwrite_output)
  fi
}

append_bool_arg() {
  local -n arr_ref="$1"
  local flag="$2"
  local enabled="$3"
  if [ "$enabled" -eq 1 ]; then
    arr_ref+=("$flag")
  fi
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_logged() {
  local log_path="$1"
  shift

  mkdir -p "$(dirname "$log_path")"

  echo
  log "Command log: $log_path"
  log "Command:"
  quote_cmd "$@" >&2

  if [ "$DRY_RUN" -eq 1 ]; then
    log "[dry-run] command not executed."
    return 0
  fi

  set +e
  "$@" 2>&1 | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e

  if [ "$status" -ne 0 ]; then
    die "Command failed with status $status. See log: $log_path"
  fi
}

resolve_work_layout() {
  local model="$1"
  local model_dir="$DATASET_ROOT/$model"
  local ssdr_dir="$model_dir/ssdr_work_dir"
  local root_data0="$model_dir/data0"
  local fitbox_vdb="$model_dir/data_prep_work_dir/data_prep_05/fitbox_vdb"

  MODEL_DIR="$model_dir"
  WORK_DIR="$ssdr_dir"
  RUN_STEP00=0
  INPUT_LAYOUT=""
  PRESERVE_DATA0=0
  HANDOFF_LEGACY_DATA0=0
  LEGACY_SOURCE_DATA0=""
  HANDOFF_SOURCE_DIR=""
  HANDOFF_SOURCE_KIND=""

  [ -d "$model_dir" ] || die "Missing model directory: $model_dir"

  # Priority 1: existing SSDR data0_json.
  #
  # This is the canonical SSDR source stage. If it already exists, use it
  # directly and run Step 00 data0_json -> data0.
  if [ -d "$ssdr_dir/data0_json" ]; then
    RUN_STEP00=1
    INPUT_LAYOUT="ssdr_work_dir_data0_json"
    PRESERVE_DATA0=0
    return 0
  fi

  # Priority 2: legacy root data0.
  #
  # Old VRoid datasets may have:
  #   <model_dir>/data0
  #
  # Treat it as source-only and hand it off to:
  #   <model_dir>/ssdr_work_dir/data0_json
  #
  # Then run Step 00 normally so all generated SSDR stages live under
  # ssdr_work_dir.
  if [ -d "$root_data0" ]; then
    RUN_STEP00=1
    INPUT_LAYOUT="legacy_model_root_data0_json_handoff"
    PRESERVE_DATA0=0
    HANDOFF_LEGACY_DATA0=1
    LEGACY_SOURCE_DATA0="$root_data0"
    HANDOFF_SOURCE_DIR="$root_data0"
    HANDOFF_SOURCE_KIND="model_root_data0"
    return 0
  fi

  # Priority 3: Data Prep Fitbox VDB output.
  #
  # New Data Prep outputs are expected at:
  #   <model_dir>/data_prep_work_dir/data_prep_05/fitbox_vdb
  #
  # Hand this directory off to:
  #   <model_dir>/ssdr_work_dir/data0_json
  #
  # so Step 00 can normalize json/pkl inputs into data0.
  if [ -d "$fitbox_vdb" ]; then
    RUN_STEP00=1
    INPUT_LAYOUT="data_prep_fitbox_vdb_data0_json_handoff"
    PRESERVE_DATA0=0
    HANDOFF_LEGACY_DATA0=1
    LEGACY_SOURCE_DATA0="$fitbox_vdb"
    HANDOFF_SOURCE_DIR="$fitbox_vdb"
    HANDOFF_SOURCE_KIND="data_prep_fitbox_vdb"
    return 0
  fi

  die "Could not detect SSDR input for $model. Expected one of:
  $ssdr_dir/data0_json
  $root_data0
  $fitbox_vdb"
}

resolve_qem_config() {
  local model="$1"
  local out=""

  if [ -n "$QEM_CONFIG_OVERRIDE" ]; then
    [ -f "$QEM_CONFIG_OVERRIDE" ] || die "Explicit --qem_config does not exist: $QEM_CONFIG_OVERRIDE"
    printf '%s\n' "$QEM_CONFIG_OVERRIDE"
    return 0
  fi

  case "$model" in
    Vroid_18|Vroid_19|Vroid_20)
      # Use the shared tuned VRoid QEM config as the default so its
      # non-specialized global parameters are the script default for all
      # legacy VRoid models. Use --qem_config to explicitly select a
      # model-specific override if needed.
      out="$REPO_ROOT/scripts/ssdr/configs/qem_config_vroid.json"
      ;;
    Griffin_A2)
      out="$REPO_ROOT/scripts/ssdr/configs/qem_config_griffin_a2.json"
      ;;
    Winter_Samurai)
      out="$REPO_ROOT/scripts/ssdr/configs/qem_config_winter_samurai.json"
      ;;
    *)
      out=""
      ;;
  esac

  if [ -n "$out" ] && [ -f "$out" ]; then
    printf '%s\n' "$out"
    return 0
  fi

  case "$model" in
    Vroid_18|Vroid_19|Vroid_20)
      die "Missing shared VRoid QEM config:
  $REPO_ROOT/scripts/ssdr/configs/qem_config_vroid.json
Use --qem_config to select an explicit model-specific override if needed."
      ;;
  esac

  # Empty means Step 03 will use work_dir default config if present,
  # otherwise built-in defaults.
  printf '%s\n' ""
}

resolve_frame_sampling_config_note() {
  local model="$1"

  FRAME_REMOVAL_IDX_RANGES=""
  FRAME_SAMPLING_CONFIG=""

  case "$model" in
    Vroid_18|Vroid_19|Vroid_20)
      local cfg="$REPO_ROOT/scripts/ssdr/configs/frame_sampling_config_vroid.json"
      if [ -f "$cfg" ]; then
        FRAME_SAMPLING_CONFIG="$cfg"
        FRAME_REMOVAL_IDX_RANGES="$(
          python -c 'import json, sys; from pathlib import Path; obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); print(str(obj.get("step01_params", {}).get("removal_idx_ranges_text", "")).strip())' "$cfg"
        )"

        if [ -n "$FRAME_REMOVAL_IDX_RANGES" ]; then
          log "VRoid frame sampling config detected: $cfg"
          log "Step 01 removal_idx_ranges will be applied:"
          log "  $FRAME_REMOVAL_IDX_RANGES"
        else
          log "WARNING: VRoid frame sampling config has empty removal_idx_ranges_text: $cfg"
        fi
      else
        log "NOTE: VRoid frame sampling config not found; Step 01 will use max/target frame sampling only."
      fi
      ;;
  esac
}

archive_existing_outputs() {
  local model="$1"
  local work_dir="$2"
  local preserve_data0="$3"

  if [ "$ARCHIVE_EXISTING" -ne 1 ]; then
    return 0
  fi

  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local archive_dir="$ARCHIVE_ROOT/${model}_${INPUT_LAYOUT}_${stamp}"

  log "Archiving generated SSDR outputs for $model"
  log "Archive dir: $archive_dir"

  if [ "$DRY_RUN" -eq 1 ]; then
    log "[dry-run] would create archive dir and move generated outputs."
    return 0
  fi

  mkdir -p "$archive_dir"

  local names=()
  if [ "$preserve_data0" -eq 1 ]; then
    names=(data1 data2 data3 data4 logs metrics run_dataset_ssdr_summary.json)
  else
    names=(data0 data1 data2 data3 data4 logs metrics run_dataset_ssdr_summary.json)
  fi

  local moved=0
  for name in "${names[@]}"; do
    if [ -e "$work_dir/$name" ]; then
      mv "$work_dir/$name" "$archive_dir/$name"
      moved=1
    fi
  done

  {
    echo "model=$model"
    echo "input_layout=$INPUT_LAYOUT"
    echo "work_dir=$work_dir"
    echo "preserve_data0=$preserve_data0"
    echo "created_at=$stamp"
  } > "$archive_dir/archive_manifest.txt"

  if [ "$moved" -eq 0 ]; then
    log "No generated outputs found to archive for $model."
  else
    log "Archived existing generated outputs to: $archive_dir"
  fi
}


handoff_legacy_data0_if_needed() {
  local model="$1"
  local src="$2"
  local dst="$3"

  if [ "$HANDOFF_LEGACY_DATA0" -ne 1 ]; then
    return 0
  fi

  [ -n "$src" ] || die "Internal error: handoff requested but source directory is empty."
  [ -d "$src" ] || die "Handoff source directory does not exist: $src"

  log "Handing off source directory for $model"
  log "  source kind      : ${HANDOFF_SOURCE_KIND:-unknown}"
  log "  source directory : $src"
  log "  target data0_json: $dst"

  if [ "$DRY_RUN" -eq 1 ]; then
    log "[dry-run] would recreate target data0_json from source directory."
    return 0
  fi

  mkdir -p "$(dirname "$dst")"

  if command -v rsync >/dev/null 2>&1; then
    mkdir -p "$dst"
    rsync -a --delete "$src/" "$dst/"
  else
    rm -rf "$dst"
    mkdir -p "$dst"
    cp -a "$src/." "$dst/"
  fi

  if [ ! -d "$dst" ]; then
    die "Handoff failed; target data0_json not found: $dst"
  fi

  log "data0_json handoff complete."
}

write_run_summary() {
  local model="$1"
  local work_dir="$2"
  local qem_config="$3"
  local status="$4"

  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi

  python - "$model" "$work_dir" "$INPUT_LAYOUT" "$qem_config" "$status" "$RUN_STEP00" "$RUN_STEP04" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

model, work_dir, input_layout, qem_config, status, run_step00, run_step04 = sys.argv[1:]
work = Path(work_dir)
out = work / "run_dataset_ssdr_summary.json"

obj = {
    "schema": "ogd_lbs_run_dataset_ssdr_summary_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": model,
    "work_dir": str(work),
    "input_layout": input_layout,
    "qem_config": qem_config or None,
    "status": status,
    "steps": {
        "00_convert_json_to_pickle": bool(int(run_step00)),
        "01_reduce_animation_frame": True,
        "02_reduce_bone_weight": True,
        "03_create_low_poly_shapes": True,
        "04_optimize_weight_and_bone_trans": bool(int(run_step04)),
    },
}

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    json.dump(obj, f, indent=4, ensure_ascii=False)
    f.write("\n")

print(f"Wrote summary: {out}", flush=True)
PY
}

run_one_model() {
  local model="$1"

  log "================================================================================"
  log "Running SSDR dataset pipeline for model: $model"
  log "================================================================================"

  resolve_work_layout "$model"

  local logs_dir="$WORK_DIR/logs"
  local metrics_dir="$WORK_DIR/metrics"
  local qem_config
  qem_config="$(resolve_qem_config "$model")"

  log "repo_root       : $REPO_ROOT"
  log "dataset_root    : $DATASET_ROOT"
  log "model_dir       : $MODEL_DIR"
  log "work_dir        : $WORK_DIR"
  log "input_layout    : $INPUT_LAYOUT"
  log "run_step00      : $RUN_STEP00"
  log "preserve_data0  : $PRESERVE_DATA0"
  log "handoff_data0_json: $HANDOFF_LEGACY_DATA0"
  if [ "$HANDOFF_LEGACY_DATA0" -eq 1 ]; then
    log "handoff_source_kind: $HANDOFF_SOURCE_KIND"
    log "handoff_source_dir : $HANDOFF_SOURCE_DIR"
  fi
  log "logs_dir        : $logs_dir"
  log "metrics_dir     : $metrics_dir"
  log "qem_config      : ${qem_config:-<Step03 default/builtin>}"
  log "conda_env       : $CONDA_ENV"
  log "dry_run         : $DRY_RUN"

  resolve_frame_sampling_config_note "$model"

  archive_existing_outputs "$model" "$WORK_DIR" "$PRESERVE_DATA0"

  handoff_legacy_data0_if_needed "$model" "$HANDOFF_SOURCE_DIR" "$WORK_DIR/data0_json"

  if [ "$DRY_RUN" -ne 1 ]; then
    mkdir -p "$logs_dir" "$metrics_dir"
  fi

  local -a py
  read -r -a py <<<"$(conda_python_prefix)"

  if [ "$RUN_STEP00" -eq 1 ]; then
    local -a cmd00=("${py[@]}" "$REPO_ROOT/scripts/ssdr/00_convert_json_to_pickle.py"
      --input_dir "$WORK_DIR"
      --metrics_dir "$metrics_dir"
      --max_workers "$MAX_WORKERS"
    )
    append_overwrite_arg cmd00

    run_logged "$logs_dir/00_convert_json_to_pickle.log" "${cmd00[@]}"
  else
    log "Skipping Step 00 because input layout already provides data0."
  fi

  local -a cmd01=("${py[@]}" "$REPO_ROOT/scripts/ssdr/01_reduce_animation_frame.py"
    --input_dir "$WORK_DIR"
    --metrics_dir "$metrics_dir"
    --max_frames "$MAX_FRAMES"
    --target_frames "$TARGET_FRAMES"
    --max_workers "$MAX_WORKERS"
  )
  append_overwrite_arg cmd01
  if [ -n "$FRAME_REMOVAL_IDX_RANGES" ]; then
    cmd01+=(--removal_idx_ranges "$FRAME_REMOVAL_IDX_RANGES")
  fi

  run_logged "$logs_dir/01_reduce_animation_frame.log" "${cmd01[@]}"

  local -a cmd02=("${py[@]}" "$REPO_ROOT/scripts/ssdr/02_reduce_bone_weight.py"
    --input_dir "$WORK_DIR"
    --metrics_dir "$metrics_dir"
    --max_workers "$MAX_WORKERS"
  )
  append_overwrite_arg cmd02
  if [ -n "$VARIANT" ]; then
    cmd02+=(--variant "$VARIANT")
  fi

  run_logged "$logs_dir/02_reduce_bone_weight.log" "${cmd02[@]}"

  local -a cmd03=("${py[@]}" "$REPO_ROOT/scripts/ssdr/03_create_low_poly_shapes.py"
    --input_dir "$WORK_DIR"
    --metrics_dir "$metrics_dir"
    --max_workers "$QEM_MAX_WORKERS"
  )
  append_overwrite_arg cmd03
  if [ -n "$qem_config" ]; then
    cmd03+=(--config "$qem_config")
  fi

  run_logged "$logs_dir/03_create_low_poly_shapes.log" "${cmd03[@]}"

  if [ "$RUN_STEP04" -eq 1 ]; then
    local -a cmd04=("${py[@]}" "$REPO_ROOT/scripts/ssdr/04_optimize_weight_and_bone_trans.py"
      --input_dir "$WORK_DIR"
      --metrics_dir "$metrics_dir"
      --print_every_n "$PRINT_EVERY_N"
      --iter "$ITER"
      --org_type "$ORG_TYPE"
      --weight_solver "$WEIGHT_SOLVER"
      --weights_smooth_step "$WEIGHTS_SMOOTH_STEP"
      --weight_soft_lock "$WEIGHT_SOFT_LOCK"
      --lambda_reg "$LAMBDA_REG"
      --bt_covariance_mode "$BT_COVARIANCE_MODE"
      --trans_affine "$TRANS_AFFINE"
      --trans_affine_norm "$TRANS_AFFINE_NORM"
      --roll_affinity "$ROLL_AFFINITY"
      --roll_affinity_norm "$ROLL_AFFINITY_NORM"
      --health_level "$HEALTH_LEVEL"
      --health_every "$HEALTH_EVERY"
      --health_weight_scheme "$HEALTH_WEIGHT_SCHEME"
      --bone_health_every "$BONE_HEALTH_EVERY"
      --max_workers "$MAX_WORKERS"
    )
    append_overwrite_arg cmd04
    append_bool_arg cmd04 --opt_weight "$OPT_WEIGHT"
    append_bool_arg cmd04 --input_poses "$INPUT_POSES"
    append_bool_arg cmd04 --use_weight_mask "$USE_WEIGHT_MASK"
    append_bool_arg cmd04 --use_weight_smoothness "$USE_WEIGHT_SMOOTHNESS"
    append_bool_arg cmd04 --eval_mesh_health "$EVAL_MESH_HEALTH"
    append_bool_arg cmd04 --eval_bone_health "$EVAL_BONE_HEALTH"
    append_bool_arg cmd04 --validate_output "$VALIDATE_OUTPUT"
    append_bool_arg cmd04 --weak_validate "$WEAK_VALIDATE"
    append_bool_arg cmd04 --save_intermediate "$SAVE_INTERMEDIATE"

    run_logged "$logs_dir/04_optimize_weight_and_bone_trans.log" "${cmd04[@]}"
  else
    log "Skipping Step 04 because --no_step04 was set."
  fi

  write_run_summary "$model" "$WORK_DIR" "$qem_config" "complete"

  log "Completed SSDR dataset pipeline for model: $model"
  log "work_dir: $WORK_DIR"
}

main() {
  parse_args "$@"

  cd "$REPO_ROOT"

  log "Starting run_dataset_ssdr.sh"
  log "target model : $MODEL"

  local model
  while IFS= read -r model; do
    [ -n "$model" ] || continue

    if ! is_supported_model "$model"; then
      die "Internal error: unsupported model in list: $model"
    fi

    run_one_model "$model"
  done < <(model_list_for_target)

  log "All requested SSDR dataset runs completed."
}

main "$@"
