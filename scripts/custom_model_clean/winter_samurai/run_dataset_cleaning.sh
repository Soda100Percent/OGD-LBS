#!/usr/bin/env bash
set -euo pipefail

# Winter Samurai formal dataset cleaning runner.
#
# Purpose:
#   /workspace/ogd_dataset/Winter_Samurai/{model,anim}
#     -> /workspace/ogd_dataset/Winter_Samurai/model_clean_work_dir/{01,02,03,04_weapon}
#     -> optional publication:
#        /workspace/ogd_dataset/Winter_Samurai/Winter_Samurai_Normalized.blend
#
# This is NOT the temporary regression runner. It is the formal dataset-side
# clean work_dir runner for ogd_dataset/Winter_Samurai.
#
# Recommended:
#   bash scripts/custom_model_clean/winter_samurai/run_dataset_cleaning.sh \
#     --archive_existing \
#     --publish copy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_DIR="$SCRIPT_DIR"
REPO="$(cd "$CLEAN_DIR/../../.." && pwd)"
cd "$REPO"

DATASET="${OGD_WS_DATASET_DIR:-/workspace/ogd_dataset/Winter_Samurai}"
WORK_DIR="${OGD_WS_MODEL_CLEAN_WORK_DIR:-}"
CONFIG="${OGD_WS_CONFIG:-$CLEAN_DIR/configs/winter_samurai_final.json}"
ENV_NAME="${OGD_WS_CONDA_ENV:-lao-lbs}"
PUBLIC_BLEND="${OGD_WS_PUBLIC_BLEND:-}"

ARCHIVE_EXISTING=0
OVERWRITE_WORK_DIR=0
OVERWRITE_PUBLICATION=0
DRY_RUN=0
PUBLISH_MODE="${OGD_WS_PUBLISH_MODE:-none}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/custom_model_clean/winter_samurai/run_dataset_cleaning.sh [options]

Options:
  --dataset_dir PATH             Default: /workspace/ogd_dataset/Winter_Samurai
  --work_dir PATH                Default: <dataset_dir>/model_clean_work_dir
  --config PATH                  Default: scripts/custom_model_clean/winter_samurai/configs/winter_samurai_final.json
  --conda_env NAME               Default: lao-lbs
  --public_blend PATH            Default: <dataset_dir>/Winter_Samurai_Normalized.blend

  --archive_existing             If work_dir exists, move it to model_clean_work_dir_previous_YYYYmmdd_HHMMSS
  --overwrite_work_dir           If work_dir exists, remove it. Mutually exclusive with --archive_existing.

  --publish none|copy|move       Default: none.
                                  Recommended: copy.
                                  move is supported but not recommended because it removes final blend from work_dir.
  --no_publish                   Same as --publish none.
  --overwrite_publication        If public blend exists, archive it before publishing.

  --dry_run                      Validate paths/config/assets and print resolution table, but do not run heavy cleaning.
  -h, --help                     Show this help.

Examples:
  # Preflight only:
  bash scripts/custom_model_clean/winter_samurai/run_dataset_cleaning.sh --dry_run

  # Formal clean and publish by copy:
  bash scripts/custom_model_clean/winter_samurai/run_dataset_cleaning.sh \
    --archive_existing \
    --publish copy

Environment:
  OGD_WS_DATASET_DIR
  OGD_WS_MODEL_CLEAN_WORK_DIR
  OGD_WS_CONFIG
  OGD_WS_CONDA_ENV
  OGD_WS_PUBLIC_BLEND
  OGD_WS_PUBLISH_MODE
  OGD_PROGRESS
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset_dir)
            DATASET="$2"
            shift 2
            ;;
        --work_dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --conda_env)
            ENV_NAME="$2"
            shift 2
            ;;
        --public_blend)
            PUBLIC_BLEND="$2"
            shift 2
            ;;
        --archive_existing)
            ARCHIVE_EXISTING=1
            shift
            ;;
        --overwrite_work_dir)
            OVERWRITE_WORK_DIR=1
            shift
            ;;
        --publish)
            PUBLISH_MODE="$2"
            shift 2
            ;;
        --no_publish)
            PUBLISH_MODE="none"
            shift
            ;;
        --overwrite_publication)
            OVERWRITE_PUBLICATION=1
            shift
            ;;
        --dry_run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$WORK_DIR" ]; then
    WORK_DIR="$DATASET/model_clean_work_dir"
fi

if [ -z "$PUBLIC_BLEND" ]; then
    PUBLIC_BLEND="$DATASET/Winter_Samurai_Normalized.blend"
fi

if [ "$ARCHIVE_EXISTING" -eq 1 ] && [ "$OVERWRITE_WORK_DIR" -eq 1 ]; then
    echo "ERROR: --archive_existing and --overwrite_work_dir are mutually exclusive." >&2
    exit 2
fi

case "$PUBLISH_MODE" in
    none|copy|move)
        ;;
    *)
        echo "ERROR: --publish must be one of: none, copy, move. Got: $PUBLISH_MODE" >&2
        exit 2
        ;;
esac

OGD_PROGRESS="${OGD_PROGRESS:-auto}"
export OGD_PROGRESS

# Make copied code snapshots import src/ogd_lbs reliably even when executed from dataset work_dir.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

STAGE="preflight"

command -v conda >/dev/null
test -f "$REPO/pyproject.toml"
test -f "$CONFIG"

SCRIPTS=(
  01_import_model_and_anims.py
  02_rebase_anims_to_skinning_armature.py
  03_rebase_to_rest_pose_base.py
  04_fix_weapon_mount_aim.py
)

for NAME in "${SCRIPTS[@]}"; do
    test -f "$CLEAN_DIR/$NAME"
done

# WorkDir-style preflight:
# - explicit path resolution
# - no fallback search
# - fail fast
# - clear resolution table before destructive/heavy work
conda run --no-capture-output -n "$ENV_NAME" python - \
  "$REPO" "$CLEAN_DIR" "$DATASET" "$WORK_DIR" "$CONFIG" "$PUBLIC_BLEND" "$PUBLISH_MODE" <<'PY'
import json
import sys
from pathlib import Path

repo, clean_dir, dataset, work_dir, config, public_blend = [
    Path(x).expanduser().resolve(strict=False) for x in sys.argv[1:7]
]
publish_mode_text = sys.argv[7]

sys.path.insert(0, str(repo / "src"))
from ogd_lbs.utils.work_dir import resolve_path_no_strict  # noqa: E402

repo = resolve_path_no_strict(repo)
clean_dir = resolve_path_no_strict(clean_dir)
dataset = resolve_path_no_strict(dataset)
work_dir = resolve_path_no_strict(work_dir)
config = resolve_path_no_strict(config)
public_blend = resolve_path_no_strict(public_blend)

def die(msg: str) -> None:
    raise RuntimeError(msg)

def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

records = [
    ("INPUT",  "repo_root",          repo,                         True,  "dir",  "OGD-LBS repository"),
    ("INPUT",  "clean_script_dir",   clean_dir,                    True,  "dir",  "custom_model_clean/winter_samurai"),
    ("INPUT",  "dataset_dir",        dataset,                      True,  "dir",  "formal ogd_dataset character root"),
    ("INPUT",  "model_fbx",          dataset / "model/Base_Mesh_Rig.fbx", True, "file", "raw commercial model FBX"),
    ("INPUT",  "rest_pose_base",     dataset / "model/rest_pose_base.blend", True, "file", "BetterFBX rest pose base"),
    ("INPUT",  "anim_dir",           dataset / "anim",             True,  "dir",  "animation FBX directory"),
    ("INPUT",  "weapon_config",      config,                       True,  "file", "Winter Samurai final weapon config"),
    ("OUTPUT", "work_dir",           work_dir,                     False, "dir",  "formal cleaning work_dir"),
    ("OUTPUT", "public_blend",       public_blend,                 False, "file", f"publish mode={publish_mode_text}"),
]

errors = []
for kind, name, path, must_exist, expected_type, note in records:
    if must_exist and not path.exists():
        errors.append(f"Missing required {kind.lower()} '{name}': {path}")
        continue
    if path.exists():
        if expected_type == "file" and not path.is_file():
            errors.append(f"Path type mismatch for '{name}': expected file: {path}")
        if expected_type == "dir" and not path.is_dir():
            errors.append(f"Path type mismatch for '{name}': expected dir: {path}")

if not (repo / "pyproject.toml").is_file():
    errors.append(f"repo_root does not contain pyproject.toml: {repo}")

if not is_relative_to(clean_dir, repo):
    errors.append(f"clean_script_dir must be inside repo: clean_dir={clean_dir}, repo={repo}")

if is_relative_to(work_dir, repo) or is_relative_to(repo, work_dir):
    errors.append(f"work_dir must not overlap repository: work_dir={work_dir}, repo={repo}")

if not is_relative_to(work_dir, dataset):
    errors.append(f"work_dir must be inside dataset_dir: work_dir={work_dir}, dataset={dataset}")

if work_dir == dataset:
    errors.append("work_dir must not equal dataset_dir.")

if public_blend.parent != dataset:
    errors.append(f"public_blend must be directly under dataset_dir: {public_blend}")

if public_blend == work_dir or is_relative_to(public_blend, work_dir):
    errors.append(f"public_blend must not be inside work_dir: {public_blend}")

cfg = {}
if config.is_file():
    cfg = json.loads(config.read_text(encoding="utf-8"))
    policies = cfg.get("action_policies")
    if not isinstance(policies, dict):
        errors.append("weapon_config.action_policies must be an object.")
    elif len(policies) != 18:
        errors.append(f"Expected 18 action policies, got {len(policies)}.")

    try:
        walk3_aim = cfg["action_policies"]["WS_Walk3"]["aim"]
        if walk3_aim != {
            "axis": "-Y",
            "target_bone": "Character1_LeftHand",
            "target_offset_fraction_of_length": [-0.5, 0.5, 0.0],
        }:
            errors.append(f"Unexpected WS_Walk3 aim config: {walk3_aim}")
        for name in ("WS_Run2", "WS_Run3"):
            if cfg["action_policies"][name]["mount"] != "back":
                errors.append(f"{name} must use back mount.")
        if cfg["mounts"]["back"]["reference_action"] != "WS_Run2":
            errors.append("Back mount reference_action must be WS_Run2.")
        if int(cfg["mounts"]["back"]["reference_frame"]) != 1:
            errors.append("Back mount reference_frame must be 1.")
    except Exception as exc:
        errors.append(f"Config invariant check failed: {exc}")

    try:
        expected = {
            "Anim_Winter_Samurai_" + name.removeprefix("WS_") + ".fbx"
            for name in cfg["action_policies"]
        }
        actual = {p.name for p in (dataset / "anim").glob("*.fbx") if p.is_file()}
        if actual != expected:
            errors.append(
                "Animation FBX set mismatch:\n"
                f"  missing={sorted(expected - actual)}\n"
                f"  extra={sorted(actual - expected)}"
            )
    except Exception as exc:
        errors.append(f"Animation set check failed: {exc}")

line = "=" * 96
subline = "-" * 96
print(line)
print("📂 [Winter Samurai Formal Dataset Clean] WorkDir Resolution Info")
print(f"   dataset_dir  : {dataset}")
print(f"   work_dir     : {work_dir}")
print(f"   public_blend : {public_blend}")
print(f"   publish_mode : {publish_mode_text}")
print(subline)
name_width = max(len(x[1]) for x in records)
kind_width = max(len(x[0]) for x in records)
for kind, name, path, must_exist, expected_type, note in records:
    print(
        f"   [{kind:<{kind_width}}] {name:<{name_width}} : {path} "
        f"(must_exist={must_exist} | type={expected_type} | exists={path.exists()})"
        f" | {note}"
    )
print(line)

if errors:
    raise RuntimeError("Preflight failed:\n" + "\n".join(f"  - {e}" for e in errors))

print("Formal dataset-clean preflight passed.")
PY

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "DRY RUN complete. No work_dir changes were made."
    exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -e "$WORK_DIR" ]; then
    if [ "$ARCHIVE_EXISTING" -eq 1 ]; then
        OLD="${WORK_DIR}_previous_${STAMP}"
        if [ -e "$OLD" ]; then
            echo "ERROR: archive target already exists: $OLD" >&2
            exit 1
        fi
        mv -- "$WORK_DIR" "$OLD"
        echo "Archived existing work_dir:"
        echo "  $WORK_DIR"
        echo "-> $OLD"
    elif [ "$OVERWRITE_WORK_DIR" -eq 1 ]; then
        case "$WORK_DIR" in
            "$DATASET"/model_clean_work_dir|"$DATASET"/model_clean_work_dir/)
                rm -rf -- "$WORK_DIR"
                echo "Removed existing work_dir: $WORK_DIR"
                ;;
            *)
                echo "ERROR: refusing to rm -rf unexpected work_dir: $WORK_DIR" >&2
                exit 1
                ;;
        esac
    else
        echo "ERROR: work_dir already exists: $WORK_DIR" >&2
        echo "Use --archive_existing or --overwrite_work_dir." >&2
        exit 1
    fi
fi

mkdir -p "$WORK_DIR"/{01,02,03,04_weapon,logs}

trap '
rc=$?
if [ "$rc" -ne 0 ]; then
    mkdir -p "$WORK_DIR"
    printf "FAILED stage=%s exit_code=%s\n" "$STAGE" "$rc" > "$WORK_DIR/FAILED.txt"
    echo
    echo "FAILED at stage: $STAGE"
    echo "All existing outputs retained: $WORK_DIR"
fi
' EXIT

cp "$CONFIG" "$WORK_DIR/config_used.json"

HASH_INPUTS=()
for NAME in "${SCRIPTS[@]}"; do
    HASH_INPUTS+=("$CLEAN_DIR/$NAME")
done
HASH_INPUTS+=("$WORK_DIR/config_used.json")
sha256sum "${HASH_INPUTS[@]}" > "$WORK_DIR/code_config_sha256.txt"

{
    git rev-parse HEAD || true
    git status --short || true
} > "$WORK_DIR/git_state.txt"

P=(conda run --no-capture-output -n "$ENV_NAME" python)

# Stage checker copied from the validated regression semantics, adapted to formal work_dir.
cat > "$WORK_DIR/check_stage.py" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stage = sys.argv[2]
cfg = json.loads((root / "config_used.json").read_text(encoding="utf-8"))
expected = set(cfg["action_policies"])

def load(name):
    return json.loads((root / name).read_text(encoding="utf-8"))

def check_actions(rows, key):
    assert len(rows) == len(expected), len(rows)
    assert {x[key] for x in rows} == expected
    assert all(x["passed"] for x in rows)

if stage == "01":
    r = load("01/import_report.json")
    assert r["status"] == "passed"
    assert r["config"]["ignore_leaf_bones"] is False
    assert r["model_import"]["used_kwargs"] == "full"
    imports = r["animation_imports"]
    assert len(imports) == 18
    assert all(x["status"] == "ok" and x["used_kwargs"] == "full" for x in imports)
    assert {
        "WS_" + Path(x["fbx_path"]).stem.removeprefix("Anim_Winter_Samurai_")
        for x in imports
    } == expected
    s = r["scene_final"]
    assert (s["armature_count"], s["mesh_count"], s["empty_count"]) == (19, 40, 0)
    assert all(a["bone_count"] == 110 for a in s["armatures"])

elif stage == "02":
    r = load("02/rebase_report.json")
    assert r["status"] == "passed"
    assert set(r["generated_actions"]) == expected
    check_actions(r["validation_after_reload"], "action_name")
    s = r["scene_after_reload"]
    assert (s["armature_count"], s["mesh_count"], s["empty_count"], s["action_count"]) == (1, 40, 0, 18)

elif stage == "03":
    r = load("03/rebase_report.json")
    assert r["status"] == "passed"
    assert r["counts"]["bones"] == 110
    assert r["counts"]["meshes"] == 40
    assert r["counts"]["actions"] == 18
    assert r["validation_before_save"]["passed"]
    assert r["validation_after_reload"]["passed"]
    check_actions(r["validation_after_reload"]["actions"], "action")
    assert all(r["input_files_unchanged"].values())
    independent = load("03/independent_audit.json")
    assert independent["passed"]

elif stage == "weapon":
    r = load("04_weapon/weapon_mount_fix_report.json")
    ext = load("04_weapon/aim_extension_report.json")
    aim = load("04_weapon/aim_audit_report.json")
    assert r["status"] == ext["status"] == aim["status"] == "passed"
    check_actions(r["validation_after_reload"], "action")
    specs = r["target_specs"]
    assert len(specs) == 18
    assert {x["action"] for x in specs} == expected
    for spec in specs:
        assert spec["mode"] == "mount"
        assert spec["mount"] == cfg["action_policies"][spec["action"]]["mount"]
    assert r["mounts"]["back"]["reference_action"] == "WS_Run2"
    assert r["mounts"]["back"]["reference_frame"] == 1
    assert r["detach_weapon_bone"]["rest_matrix_abs_max_after_detach"] <= 1e-5
    assert len(aim["actions"]) == 1
    assert aim["actions"][0]["action"] == "WS_Walk3"
    assert aim["actions"][0]["passed"]

    manifest = load("04_weapon/aim_audit_manifest.json")
    assert len(manifest["actions"]) == 1
    assert manifest["actions"][0]["target_offset_fraction_of_length"] == [-0.5, 0.5, 0.0]

    s = r["scene_after_reload"]
    assert (s["armature_count"], s["mesh_count"], s["empty_count"], s["action_count"]) == (1, 40, 0, 18)
    assert s["weapon_bone"]["parent"] is None
else:
    raise RuntimeError(stage)

print("PASS:", stage)
PY

STAGE="source_asset_hash_before"
"${P[@]}" - "$DATASET" "$WORK_DIR/source_assets_sha256.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset, output = map(Path, sys.argv[1:])
paths = [
    dataset / "model/Base_Mesh_Rig.fbx",
    dataset / "model/rest_pose_base.blend",
    *sorted((dataset / "anim").glob("*.fbx")),
]

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

output.write_text(
    json.dumps({str(p.resolve()): digest(p) for p in paths}, indent=2),
    encoding="utf-8",
)
PY

B01="$WORK_DIR/01/Winter_Samurai_clean01.blend"
B02="$WORK_DIR/02/Winter_Samurai_clean02.blend"
B03="$WORK_DIR/03/Winter_Samurai_clean03_rest_base_rebased.blend"
FINAL="$WORK_DIR/04_weapon/Winter_Samurai_Normalized.blend"

echo "============================================================"
echo "Winter Samurai formal dataset cleaning"
echo "Dataset      : $DATASET"
echo "Work dir     : $WORK_DIR"
echo "Final        : $FINAL"
echo "Public blend : $PUBLIC_BLEND"
echo "Publish mode : $PUBLISH_MODE"
echo "Progress     : $OGD_PROGRESS"
echo "============================================================"

STAGE="01_import"
"${P[@]}" "$CLEAN_DIR/01_import_model_and_anims.py" \
    --dataset_dir "$DATASET" \
    --no_ignore_leaf_bones \
    --output_dir "$WORK_DIR/01" \
    --output_blend "$B01" \
    2>&1 | tee "$WORK_DIR/01/run.log"
"${P[@]}" "$WORK_DIR/check_stage.py" "$WORK_DIR" 01

STAGE="02_rebase"
"${P[@]}" "$CLEAN_DIR/02_rebase_anims_to_skinning_armature.py" \
    --input_blend "$B01" \
    --import_report "$WORK_DIR/01/import_report.json" \
    --output_dir "$WORK_DIR/02" \
    --output_blend "$B02" \
    --position_tol 1e-4 \
    2>&1 | tee "$WORK_DIR/02/run.log"
"${P[@]}" "$WORK_DIR/check_stage.py" "$WORK_DIR" 02

STAGE="03_rest_base"
"${P[@]}" "$CLEAN_DIR/03_rebase_to_rest_pose_base.py" \
    --input_blend "$B02" \
    --dataset_dir "$DATASET" \
    --output_dir "$WORK_DIR/03" \
    --samples_per_frame 1 \
    --position_tol 1e-4 \
    --mesh_position_tol 1e-4 \
    2>&1 | tee "$WORK_DIR/03/run.log"

STAGE="03_independent_audit"
"${P[@]}" "$CLEAN_DIR/03_rebase_to_rest_pose_base.py" \
    --audit_saved \
    --output_dir "$WORK_DIR/03" \
    2>&1 | tee "$WORK_DIR/03/independent_audit.log"
"${P[@]}" "$WORK_DIR/check_stage.py" "$WORK_DIR" 03

STAGE="04_weapon_mount_and_aim"
"${P[@]}" "$CLEAN_DIR/04_fix_weapon_mount_aim.py" \
    --input_blend "$B03" \
    --output_dir "$WORK_DIR/04_weapon" \
    --output_blend "$FINAL" \
    --config "$WORK_DIR/config_used.json" \
    --position_tol 1e-4 \
    2>&1 | tee "$WORK_DIR/04_weapon/run.log"

STAGE="04_weapon_independent_audit"
"${P[@]}" "$CLEAN_DIR/04_fix_weapon_mount_aim.py" \
    --audit_saved \
    --output_dir "$WORK_DIR/04_weapon" \
    2>&1 | tee "$WORK_DIR/04_weapon/independent_aim_audit.log"
"${P[@]}" "$WORK_DIR/check_stage.py" "$WORK_DIR" weapon

STAGE="final_invariants"
cat > "$WORK_DIR/check_final_invariants.py" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

before_path, final_path, config_path, output_path = map(Path, sys.argv[1:])
cfg = json.loads(config_path.read_text(encoding="utf-8"))
expected_actions = set(cfg["action_policies"])

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def snapshot(path):
    assert bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False) == {"FINISHED"}
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    meshes = sorted([o for o in bpy.data.objects if o.type == "MESH"], key=lambda o: o.name)
    assert len(arms) == 1
    assert len(meshes) == 40
    assert not any(o.type == "EMPTY" for o in bpy.data.objects)
    assert set(bpy.data.actions.keys()) == expected_actions

    arm = arms[0]
    assert len(arm.data.bones) == 110
    weapon = arm.data.bones["Weapon"]

    mesh_hashes = {}
    for obj in meshes:
        mods = [m for m in obj.modifiers if m.type == "ARMATURE"]
        assert len(mods) == 1 and mods[0].object is arm, obj.name
        mesh_hashes[obj.name] = digest({
            "vertices": [list(v.co) for v in obj.data.vertices],
            "polygons": [list(p.vertices) for p in obj.data.polygons],
            "groups": [g.name for g in obj.vertex_groups],
            "weights": [[(g.group, float(g.weight)) for g in v.groups] for v in obj.data.vertices],
            "materials": [s.material.name if s.material else None for s in obj.material_slots],
            "material_indices": [p.material_index for p in obj.data.polygons],
            "uvs": {layer.name: [list(x.uv) for x in layer.data] for layer in obj.data.uv_layers},
            "parent": obj.parent.name if obj.parent else None,
            "parent_type": obj.parent_type,
            "parent_bone": obj.parent_bone,
            "matrix_basis": np.asarray(obj.matrix_basis).tolist(),
            "matrix_parent_inverse": np.asarray(obj.matrix_parent_inverse).tolist(),
            "modifier_target": mods[0].object.name,
        })

    prefix = arm.pose.bones["Weapon"].path_from_id() + "."
    curves_hashes = {}
    for action in sorted(bpy.data.actions, key=lambda a: a.name):
        slots = list(getattr(action, "slots", []))
        assert len(slots) <= 1
        curves = list(action.fcurves)
        assert len(curves) == 1100, (action.name, len(curves))
        payload = []
        for fc in sorted(curves, key=lambda f: (f.data_path, f.array_index)):
            if fc.data_path.startswith(prefix):
                continue
            payload.append({
                "path": fc.data_path,
                "index": fc.array_index,
                "extrapolation": fc.extrapolation,
                "keys": [
                    [list(k.co), list(k.handle_left), list(k.handle_right),
                     k.interpolation, k.handle_left_type, k.handle_right_type]
                    for k in fc.keyframe_points
                ],
            })
        curves_hashes[action.name] = digest(payload)

    return {
        "armature": arm.name,
        "mesh_hashes": mesh_hashes,
        "non_weapon_curve_hashes": curves_hashes,
        "rest": {b.name: np.asarray(b.matrix_local).tolist() for b in arm.data.bones},
        "parents": {b.name: b.parent.name if b.parent else None for b in arm.data.bones},
        "weapon_descendants": sorted(b.name for b in weapon.children_recursive),
        "roots": sorted(b.name for b in arm.data.bones if b.parent is None),
        "vertices": sum(len(m.data.vertices) for m in meshes),
    }

before = snapshot(before_path)
after = snapshot(final_path)

assert before["armature"] == after["armature"]
assert before["mesh_hashes"] == after["mesh_hashes"], "Mesh data/weights/binding changed"
assert before["non_weapon_curve_hashes"] == after["non_weapon_curve_hashes"], "Other curves changed"
assert after["parents"]["Weapon"] is None

maximum = 0.0
for name, matrix in before["rest"].items():
    error = float(np.max(np.abs(np.asarray(matrix) - np.asarray(after["rest"][name]))))
    maximum = max(maximum, error)
    assert error <= 1e-5, (name, error)
    if name != "Weapon":
        assert before["parents"][name] == after["parents"][name], name

result = {
    "status": "passed",
    "armatures": 1,
    "bones": 110,
    "meshes": 40,
    "vertices": after["vertices"],
    "actions": sorted(expected_actions),
    "empty_count": 0,
    "root_bones": after["roots"],
    "weapon_descendants": after["weapon_descendants"],
    "rest_matrix_abs_max_change": maximum,
    "base_meshes_weights_material_assignments_uvs_and_binding_unchanged": True,
    "non_weapon_animation_curves_unchanged": True,
    "scope_note": (
        "Unchanged descendant curves do not guarantee unchanged descendant world motion. "
        "Weapon motion is intentionally changed. No continuous-time equivalence claim."
    ),
}
output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
PY

"${P[@]}" "$WORK_DIR/check_final_invariants.py" \
    "$B03" "$FINAL" "$WORK_DIR/config_used.json" "$WORK_DIR/final_invariants.json" \
    2>&1 | tee "$WORK_DIR/final_invariants.log"

STAGE="summary"
"${P[@]}" - "$WORK_DIR" "$FINAL" "$DATASET" "$PUBLIC_BLEND" "$PUBLISH_MODE" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root, final, dataset, public_blend = map(Path, sys.argv[1:5])
publish_mode = sys.argv[5]

def read(name):
    return json.loads((root / name).read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

hashes = read("source_assets_sha256.json")
assert all(digest(path) == value for path, value in hashes.items()), "Source asset changed"
assert read("final_invariants.json")["status"] == "passed"

result = {
    "schema": "ogd_lbs_winter_samurai_formal_dataset_cleaning_v1",
    "status": "passed",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_dir": str(dataset.resolve()),
    "work_dir": str(root.resolve()),
    "final_blend": str(final.resolve()),
    "final_blend_sha256": digest(final),
    "public_blend": str(public_blend.resolve()),
    "publish_mode": publish_mode,
    "config_sha256": digest(root / "config_used.json"),
    "source_assets_unchanged": True,
    "visual_review": "pending_for_this_end_to_end_run",
    "dataset_publication": "pending" if publish_mode != "none" else "not_requested",
    "data_prep_ssdr_validation": "not_part_of_this_cleaning_run",
    "walk3_target_offset_fraction_of_length": [-0.5, 0.5, 0.0],
    "back_reference": "WS_Run2 / frame 1 / Character1_Spine1",
    "reports": {
        "step01": "01/import_report.json",
        "step02": "02/rebase_report.json",
        "step03": "03/rebase_report.json",
        "step03_independent": "03/independent_audit.json",
        "weapon": "04_weapon/weapon_mount_fix_report.json",
        "aim": "04_weapon/aim_audit_report.json",
        "invariants": "final_invariants.json",
    },
}
(root / "final_dataset_cleaning_summary.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(result, indent=2, ensure_ascii=False))
PY

if [ "$PUBLISH_MODE" != "none" ]; then
    STAGE="publish"

    if [ -e "$PUBLIC_BLEND" ]; then
        if [ "$OVERWRITE_PUBLICATION" -eq 1 ]; then
            PUB_OLD="${PUBLIC_BLEND%.blend}.previous_${STAMP}.blend"
            mv -- "$PUBLIC_BLEND" "$PUB_OLD"
            echo "Archived existing public blend:"
            echo "  $PUBLIC_BLEND"
            echo "-> $PUB_OLD"
        else
            echo "ERROR: public blend already exists: $PUBLIC_BLEND" >&2
            echo "Use --overwrite_publication to archive and replace it." >&2
            exit 1
        fi
    fi

    case "$PUBLISH_MODE" in
        copy)
            cp -p -- "$FINAL" "$PUBLIC_BLEND"
            ;;
        move)
            echo "WARNING: --publish move removes final blend from work_dir; this is not recommended."
            mv -- "$FINAL" "$PUBLIC_BLEND"
            ;;
    esac

    "${P[@]}" - "$WORK_DIR" "$FINAL" "$PUBLIC_BLEND" "$PUBLISH_MODE" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root, final_path, public_path = map(Path, sys.argv[1:4])
mode = sys.argv[4]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

report = {
    "schema": "ogd_lbs_winter_samurai_publication_v1",
    "status": "published",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "work_dir_final_blend": str(final_path),
    "work_dir_final_exists_after_publication": final_path.exists(),
    "public_blend": str(public_path.resolve()),
    "public_blend_sha256": digest(public_path),
}
(root / "publication_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(report, indent=2, ensure_ascii=False))
PY

    "${P[@]}" - "$WORK_DIR/final_dataset_cleaning_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
obj = json.loads(path.read_text(encoding="utf-8"))
obj["dataset_publication"] = "performed"
path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
PY
fi

STAGE="complete"

echo
echo "============================================================"
echo "WINTER SAMURAI FORMAL DATASET CLEANING PASSED"
echo "Work dir     : $WORK_DIR"
echo "Final blend  : $FINAL"
echo "Public blend : $PUBLIC_BLEND"
echo "Summary      : $WORK_DIR/final_dataset_cleaning_summary.json"
if [ "$PUBLISH_MODE" != "none" ]; then
    echo "Publication  : $WORK_DIR/publication_report.json"
fi
echo "Visual review of this end-to-end output remains required."
echo "============================================================"
