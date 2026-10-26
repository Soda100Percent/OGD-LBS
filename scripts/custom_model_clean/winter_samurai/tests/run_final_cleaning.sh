#!/usr/bin/env bash
set -euo pipefail

# Asset-specific, real-data regression. Not a lightweight unit test.
# Usage:
#   bash scripts/debug/winter_samurai_fbx_clean/tests/run_final_cleaning.sh
#
# Optional environment overrides:
#   OGD_WS_DATASET_DIR
#   OGD_WS_FINAL_TEST_ROOT
#   OGD_WS_CONDA_ENV
#
# Outputs are outside the repository. Existing current/ is archived.
# No dataset publication, git commit, or source asset modification is performed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$CLEAN_DIR/../../.." && pwd)"
cd "$REPO"

DATASET="${OGD_WS_DATASET_DIR:-/workspace/ogd_dataset/Winter_Samurai}"
ROOT="${OGD_WS_FINAL_TEST_ROOT:-/workspace/ogd_tmp_blends/winter_samurai_fbx_clean/final_regression_v1}"
ENV_NAME="${OGD_WS_CONDA_ENV:-lao-lbs}"
OGD_PROGRESS="${OGD_PROGRESS:-auto}"
export OGD_PROGRESS

CONFIG="$CLEAN_DIR/configs/winter_samurai_final.json"
RUN="$ROOT/current"
STAGE="preflight"

command -v conda >/dev/null
test -f "$REPO/pyproject.toml"
test -f "$CONFIG"
test -f "$DATASET/model/Base_Mesh_Rig.fbx"
test -f "$DATASET/model/rest_pose_base.blend"

SCRIPTS=(
  01_import_model_and_anims.py
  02_rebase_anims_to_skinning_armature.py
  03_rebase_to_rest_pose_base.py
  04_fix_weapon_mount_aim.py
)

for NAME in "${SCRIPTS[@]}"; do
    test -f "$CLEAN_DIR/$NAME"
done

# Resolve and validate paths and the complete set of expected assets
# before moving the previous experiment.
conda run --no-capture-output -n "$ENV_NAME" \
  python - "$DATASET" "$ROOT" "$REPO" "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

dataset, root, repo, config = [Path(x).expanduser().resolve() for x in sys.argv[1:]]
for protected in (dataset, repo):
    assert not root.is_relative_to(protected), (root, protected)
    assert not protected.is_relative_to(root), (root, protected)

cfg = json.loads(config.read_text(encoding="utf-8"))
assert len(cfg["action_policies"]) == 18
assert cfg["action_policies"]["WS_Walk3"]["aim"] == {
    "axis": "-Y",
    "target_bone": "Character1_LeftHand",
    "target_offset_fraction_of_length": [-0.5, 0.5, 0.0],
}
for name in ("WS_Run2", "WS_Run3"):
    assert cfg["action_policies"][name]["mount"] == "back"
assert cfg["mounts"]["back"]["reference_action"] == "WS_Run2"
assert cfg["mounts"]["back"]["reference_frame"] == 1

expected = {
    "Anim_Winter_Samurai_" + name.removeprefix("WS_") + ".fbx"
    for name in cfg["action_policies"]
}
actual = {p.name for p in (dataset / "anim").glob("*.fbx") if p.is_file()}
assert actual == expected, {
    "missing": sorted(expected - actual),
    "extra": sorted(actual - expected),
}
print("Asset/config preflight passed.")
PY

mkdir -p "$ROOT"
if [ -e "$RUN" ]; then
    i=1
    while :; do
        OLD="$(printf '%s/previous_%03d' "$ROOT" "$i")"
        [ -e "$OLD" ] || break
        i=$((i + 1))
    done
    mv -- "$RUN" "$OLD"
    echo "Archived previous experiment: $OLD"
fi

mkdir -p "$RUN"/{code,01,02,03,04_weapon}

trap '
rc=$?
if [ "$rc" -ne 0 ]; then
    printf "FAILED stage=%s exit_code=%s\n" "$STAGE" "$rc" > "$RUN/FAILED.txt"
    echo
    echo "FAILED at stage: $STAGE"
    echo "All existing outputs retained: $RUN"
fi
' EXIT

# Freeze exact code/config for this run.
for NAME in "${SCRIPTS[@]}"; do
    cp "$CLEAN_DIR/$NAME" "$RUN/code/$NAME"
done
cp "$CONFIG" "$RUN/config_used.json"
sha256sum "$RUN"/code/*.py "$RUN/config_used.json" > "$RUN/code_config_sha256.txt"

{
    git rev-parse HEAD
    git status --short
} > "$RUN/git_state.txt"

P=(conda run --no-capture-output -n "$ENV_NAME" python)

# Helper validates a stage immediately, including completeness checks
# where an upstream script's status alone is insufficient.
cat > "$RUN/check_stage.py" <<'PY'
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

# Record asset hashes before running.
"${P[@]}" - "$DATASET" "$RUN/source_assets_sha256.json" <<'PY'
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
output.write_text(json.dumps({str(p.resolve()): digest(p) for p in paths}, indent=2))
PY

B01="$RUN/01/Winter_Samurai_clean01.blend"
B02="$RUN/02/Winter_Samurai_clean02.blend"
B03="$RUN/03/Winter_Samurai_clean03_rest_base_rebased.blend"
FINAL="$RUN/04_weapon/Winter_Samurai_Normalized.blend"

echo "============================================================"
echo "Winter Samurai final cleaning regression v1"
echo "Dataset : $DATASET"
echo "Run     : $RUN"
echo "Final   : $FINAL"
echo "Progress: $OGD_PROGRESS"
echo "============================================================"

STAGE="01_import"
"${P[@]}" "$RUN/code/01_import_model_and_anims.py" \
    --dataset_dir "$DATASET" \
    --no_ignore_leaf_bones \
    --output_dir "$RUN/01" \
    --output_blend "$B01" \
    2>&1 | tee "$RUN/01/run.log"
"${P[@]}" "$RUN/check_stage.py" "$RUN" 01

STAGE="02_rebase"
"${P[@]}" "$RUN/code/02_rebase_anims_to_skinning_armature.py" \
    --input_blend "$B01" \
    --import_report "$RUN/01/import_report.json" \
    --output_dir "$RUN/02" \
    --output_blend "$B02" \
    --position_tol 1e-4 \
    2>&1 | tee "$RUN/02/run.log"
"${P[@]}" "$RUN/check_stage.py" "$RUN" 02

STAGE="03_rest_base"
"${P[@]}" "$RUN/code/03_rebase_to_rest_pose_base.py" \
    --input_blend "$B02" \
    --dataset_dir "$DATASET" \
    --output_dir "$RUN/03" \
    --samples_per_frame 1 \
    --position_tol 1e-4 \
    --mesh_position_tol 1e-4 \
    2>&1 | tee "$RUN/03/run.log"

STAGE="03_independent_audit"
"${P[@]}" "$RUN/code/03_rebase_to_rest_pose_base.py" \
    --audit_saved \
    --output_dir "$RUN/03" \
    2>&1 | tee "$RUN/03/independent_audit.log"
"${P[@]}" "$RUN/check_stage.py" "$RUN" 03

STAGE="04_weapon_mount_and_aim"
# IMPORTANT: Step 04 is the single aim-capable entrypoint in custom_model_clean.
"${P[@]}" "$RUN/code/04_fix_weapon_mount_aim.py" \
    --input_blend "$B03" \
    --output_dir "$RUN/04_weapon" \
    --output_blend "$FINAL" \
    --config "$RUN/config_used.json" \
    --position_tol 1e-4 \
    2>&1 | tee "$RUN/04_weapon/run.log"

STAGE="04_weapon_independent_audit"
"${P[@]}" "$RUN/code/04_fix_weapon_mount_aim.py" \
    --audit_saved \
    --output_dir "$RUN/04_weapon" \
    2>&1 | tee "$RUN/04_weapon/independent_aim_audit.log"
"${P[@]}" "$RUN/check_stage.py" "$RUN" weapon

# Compare Step 03 input with final output:
# base geometry, weights, material assignment, UVs and non-Weapon curves
# must be unchanged. Weapon descendants are explicitly reported.
STAGE="final_invariants"
cat > "$RUN/check_final_invariants.py" <<'PY'
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

"${P[@]}" "$RUN/check_final_invariants.py" \
    "$B03" "$FINAL" "$RUN/config_used.json" "$RUN/final_invariants.json" \
    2>&1 | tee "$RUN/final_invariants.log"

STAGE="summary"
"${P[@]}" - "$RUN" "$FINAL" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root, final = map(Path, sys.argv[1:])

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
    "schema": "ogd_lbs_winter_samurai_final_cleaning_regression_v1",
    "status": "passed",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "final_blend": str(final.resolve()),
    "final_blend_sha256": digest(final),
    "config_sha256": digest(root / "config_used.json"),
    "source_assets_unchanged": True,
    "visual_review": "pending_for_this_end_to_end_run",
    "dataset_publication": "not_performed",
    "data_prep_ssdr_validation": "not_part_of_this_test",
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
(root / "final_test_summary.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(result, indent=2, ensure_ascii=False))
PY

STAGE="complete"
echo
echo "============================================================"
echo "FINAL CLEANING REGRESSION PASSED"
echo "Final blend: $FINAL"
echo "Summary:     $RUN/final_test_summary.json"
echo "Visual review of this end-to-end output remains required."
echo "============================================================"
