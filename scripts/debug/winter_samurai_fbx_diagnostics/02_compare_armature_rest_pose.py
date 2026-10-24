#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Winter Samurai FBX diagnostics Step 02:
Compare rest poses between exactly two armatures in an imported diagnostic .blend.

This script is read-only:
    - opens an input .blend
    - does not retarget
    - does not modify animation
    - does not save the .blend

Identification rule:
    1. The scene must contain exactly two ARMATURE objects, otherwise fail fast.
    2. The skinning armature is auto-detected as the armature bound to meshes
       through Armature modifiers and/or mesh parenting.
    3. The animation armature is the other armature.
    4. Optional overrides are available:
         --skinning_armature_name
         --animation_armature_name

Comparisons:
    - bone name correspondence
    - hierarchy correspondence by same-name parent check
    - rest pose matrix difference for same-name bones
      both in armature-local and world space
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Vector


def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def log(msg: str) -> None:
    print(f"[winter-samurai-rest-compare] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[winter-samurai-rest-compare][WARN] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[winter-samurai-rest-compare][ERROR] {msg}", flush=True)
    raise RuntimeError(msg)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root() -> Path | None:
    this_file = Path(__file__).resolve(strict=False)
    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return None


def open_blend(path: Path) -> None:
    if not path.is_file():
        fail(f"Input blend does not exist: {path}")

    log(f"Opening blend: {path}")
    result = bpy.ops.wm.open_mainfile(filepath=str(path))

    if result != {"FINISHED"}:
        fail(f"Failed to open blend: {path}, result={result}")


def safe_float_list(values: Any) -> list[float]:
    try:
        return [float(x) for x in values]
    except Exception:
        return []


def vec_l2(a: Vector, b: Vector) -> float:
    return float((a - b).length)


def safe_normalized(v: Vector) -> Vector:
    out = Vector((float(v.x), float(v.y), float(v.z)))
    if out.length <= 1e-12:
        return Vector((0.0, 0.0, 0.0))
    out.normalize()
    return out


def axis_angle_deg(a: Matrix, b: Matrix, axis: int) -> float:
    va = safe_normalized(Vector((a[0][axis], a[1][axis], a[2][axis])))
    vb = safe_normalized(Vector((b[0][axis], b[1][axis], b[2][axis])))

    if va.length <= 1e-12 or vb.length <= 1e-12:
        return 0.0

    dot = max(-1.0, min(1.0, float(va.dot(vb))))
    return float(math.degrees(math.acos(dot)))


def matrix_no_scale(mat: Matrix) -> Matrix:
    loc, rot, _scale = mat.decompose()
    return Matrix.Translation(loc) @ rot.to_matrix().to_4x4()


def rotation_angle_deg(a: Matrix, b: Matrix) -> float:
    _la, qa, _sa = a.decompose()
    _lb, qb, _sb = b.decompose()

    try:
        delta = qa.rotation_difference(qb)
        return float(math.degrees(delta.angle))
    except Exception:
        return 0.0


def matrix_frobenius(a: Matrix, b: Matrix) -> float:
    total = 0.0
    for r in range(4):
        for c in range(4):
            d = float(a[r][c]) - float(b[r][c])
            total += d * d
    return float(math.sqrt(total))


def matrix_compare_stats(a: Matrix, b: Matrix) -> dict[str, float]:
    """
    Compare two matrices after decomposition.

    a:
        skinning armature rest matrix
    b:
        animation armature rest matrix
    """
    la, _ra, sa = a.decompose()
    lb, _rb, sb = b.decompose()

    a_ns = matrix_no_scale(a)
    b_ns = matrix_no_scale(b)

    scale_l2 = math.sqrt(
        (float(sa.x) - float(sb.x)) ** 2
        + (float(sa.y) - float(sb.y)) ** 2
        + (float(sa.z) - float(sb.z)) ** 2
    )

    return {
        "translation_l2": vec_l2(la, lb),
        "rotation_deg": rotation_angle_deg(a_ns, b_ns),
        "scale_l2": float(scale_l2),
        "frobenius_no_scale": matrix_frobenius(a_ns, b_ns),
        "x_axis_deg": axis_angle_deg(a_ns, b_ns, 0),
        "y_axis_deg": axis_angle_deg(a_ns, b_ns, 1),
        "z_axis_deg": axis_angle_deg(a_ns, b_ns, 2),
    }


def max_stats(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}

    keys = sorted(set().union(*(item.keys() for item in items)))
    return {
        f"{key}_max": float(max(float(item.get(key, 0.0)) for item in items))
        for key in keys
    }


def mean_stats(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}

    keys = sorted(set().union(*(item.keys() for item in items)))
    return {
        f"{key}_mean": float(sum(float(item.get(key, 0.0)) for item in items) / len(items))
        for key in keys
    }


def summarize_object_transform(obj: bpy.types.Object) -> dict[str, Any]:
    return {
        "name": obj.name,
        "type": obj.type,
        "location": safe_float_list(obj.location),
        "rotation_euler": safe_float_list(obj.rotation_euler),
        "scale": safe_float_list(obj.scale),
        "matrix_world": [[float(obj.matrix_world[r][c]) for c in range(4)] for r in range(4)],
    }


def get_armatures() -> list[bpy.types.Object]:
    return sorted(
        [obj for obj in bpy.data.objects if obj.type == "ARMATURE"],
        key=lambda obj: obj.name,
    )


def get_meshes() -> list[bpy.types.Object]:
    return sorted(
        [obj for obj in bpy.data.objects if obj.type == "MESH"],
        key=lambda obj: obj.name,
    )


def animation_info(armature: bpy.types.Object) -> dict[str, Any]:
    ad = armature.animation_data

    active_action_name = None
    nla_track_count = 0
    nla_strip_count = 0
    driver_count = 0

    if ad is not None:
        if ad.action is not None:
            active_action_name = ad.action.name

        nla_track_count = int(len(ad.nla_tracks))
        for track in ad.nla_tracks:
            nla_strip_count += int(len(track.strips))

        try:
            driver_count = int(len(ad.drivers))
        except Exception:
            driver_count = 0

    score = 0
    if active_action_name is not None:
        score += 10
    score += nla_strip_count * 5
    score += driver_count

    return {
        "has_animation_data": bool(ad is not None),
        "active_action": active_action_name,
        "nla_track_count": int(nla_track_count),
        "nla_strip_count": int(nla_strip_count),
        "driver_count": int(driver_count),
        "animation_score": int(score),
    }


def mesh_binding_info(armature: bpy.types.Object) -> dict[str, Any]:
    modifier_meshes: list[str] = []
    parented_meshes: list[str] = []
    any_armature_modifier_meshes: list[str] = []

    for mesh in get_meshes():
        if mesh.parent is armature:
            parented_meshes.append(mesh.name)

        for mod in mesh.modifiers:
            if mod.type != "ARMATURE":
                continue

            any_armature_modifier_meshes.append(mesh.name)

            if getattr(mod, "object", None) is armature:
                modifier_meshes.append(mesh.name)

    modifier_meshes = sorted(set(modifier_meshes))
    parented_meshes = sorted(set(parented_meshes))
    any_armature_modifier_meshes = sorted(set(any_armature_modifier_meshes))

    # Strongest signal is Armature modifier object == armature.
    # Parent relation is weaker but useful for imported diagnostic scenes.
    binding_score = len(modifier_meshes) * 10 + len(parented_meshes)

    return {
        "bound_meshes_by_armature_modifier": modifier_meshes,
        "bound_meshes_by_parent": parented_meshes,
        "all_meshes_with_any_armature_modifier_first_50": any_armature_modifier_meshes[:50],
        "armature_modifier_mesh_count": int(len(modifier_meshes)),
        "parented_mesh_count": int(len(parented_meshes)),
        "binding_score": int(binding_score),
    }


def armature_summary(armature: bpy.types.Object) -> dict[str, Any]:
    bones = list(armature.data.bones)
    bone_names = [bone.name for bone in bones]
    root_bones = [bone.name for bone in bones if bone.parent is None]

    return {
        "object": summarize_object_transform(armature),
        "data_name": armature.data.name,
        "bone_count": int(len(bone_names)),
        "root_bones": root_bones,
        "bone_names_first_100": bone_names[:100],
        "animation": animation_info(armature),
        "mesh_binding": mesh_binding_info(armature),
        "pose_position": str(getattr(armature.data, "pose_position", "")),
    }


@dataclass(frozen=True)
class SelectedArmatures:
    skinning: bpy.types.Object
    animation: bpy.types.Object
    reason: str


def select_armatures(
    *,
    skinning_armature_name: str | None,
    animation_armature_name: str | None,
) -> SelectedArmatures:
    armatures = get_armatures()

    if len(armatures) != 2:
        fail(
            "Expected exactly two ARMATURE objects in the diagnostic blend, "
            f"got {len(armatures)}: {[obj.name for obj in armatures]}"
        )

    if skinning_armature_name or animation_armature_name:
        if not skinning_armature_name or not animation_armature_name:
            fail(
                "If overriding armature selection, both --skinning_armature_name "
                "and --animation_armature_name must be provided."
            )

        skin = bpy.data.objects.get(skinning_armature_name)
        anim = bpy.data.objects.get(animation_armature_name)

        if skin is None or skin.type != "ARMATURE":
            fail(f"Invalid --skinning_armature_name: {skinning_armature_name}")
        if anim is None or anim.type != "ARMATURE":
            fail(f"Invalid --animation_armature_name: {animation_armature_name}")
        if skin is anim:
            fail("skinning and animation armature overrides refer to the same object.")

        return SelectedArmatures(
            skinning=skin,
            animation=anim,
            reason="explicit_override",
        )

    binding = {arm.name: mesh_binding_info(arm) for arm in armatures}
    anim_info = {arm.name: animation_info(arm) for arm in armatures}

    bound_candidates = [
        arm for arm in armatures
        if binding[arm.name]["binding_score"] > 0
    ]

    if len(bound_candidates) == 1:
        skin = bound_candidates[0]
        anim = armatures[0] if armatures[1] is skin else armatures[1]

        if anim_info[anim.name]["animation_score"] <= 0:
            warn(
                f"Auto-selected animation armature {anim.name!r}, but its direct "
                "animation_score is 0. This can still happen if FBX imported actions "
                "are stored on child/helper objects. Continuing because the other "
                "armature is the only mesh-bound armature."
            )

        return SelectedArmatures(
            skinning=skin,
            animation=anim,
            reason="unique_mesh_bound_armature",
        )

    # Fallback: choose no-animation + higher binding as skin if possible.
    no_anim_bound = [
        arm for arm in armatures
        if anim_info[arm.name]["animation_score"] == 0
        and binding[arm.name]["binding_score"] > 0
    ]

    if len(no_anim_bound) == 1:
        skin = no_anim_bound[0]
        anim = armatures[0] if armatures[1] is skin else armatures[1]
        return SelectedArmatures(
            skinning=skin,
            animation=anim,
            reason="unique_no_animation_mesh_bound_armature",
        )

    fail(
        "Could not auto-select skinning/animation armatures.\n"
        f"Armature summaries:\n"
        + json.dumps(
            [
                {
                    "name": arm.name,
                    "binding": binding[arm.name],
                    "animation": anim_info[arm.name],
                }
                for arm in armatures
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\nUse --skinning_armature_name and --animation_armature_name to override."
    )

    raise RuntimeError("unreachable")


def bone_parent_name(armature: bpy.types.Object, bone_name: str) -> str | None:
    bone = armature.data.bones.get(bone_name)
    if bone is None:
        return None
    return bone.parent.name if bone.parent is not None else None


def bone_local_rest_matrix(armature: bpy.types.Object, bone_name: str) -> Matrix:
    bone = armature.data.bones.get(bone_name)
    if bone is None:
        fail(f"Bone not found: armature={armature.name}, bone={bone_name}")
    return bone.matrix_local.copy()


def bone_world_rest_matrix(armature: bpy.types.Object, bone_name: str) -> Matrix:
    return armature.matrix_world @ bone_local_rest_matrix(armature, bone_name)


def compare_bones(
    *,
    skinning: bpy.types.Object,
    animation: bpy.types.Object,
    top_n: int,
) -> dict[str, Any]:
    skin_bones = [bone.name for bone in skinning.data.bones]
    anim_bones = [bone.name for bone in animation.data.bones]

    skin_set = set(skin_bones)
    anim_set = set(anim_bones)
    common = sorted(skin_set & anim_set)

    skin_only = sorted(skin_set - anim_set)
    anim_only = sorted(anim_set - skin_set)

    hierarchy_mismatches: list[dict[str, Any]] = []
    per_bone: list[dict[str, Any]] = []

    local_stats_items: list[dict[str, float]] = []
    world_stats_items: list[dict[str, float]] = []

    for bone_name in common:
        skin_parent = bone_parent_name(skinning, bone_name)
        anim_parent = bone_parent_name(animation, bone_name)

        hierarchy_match = skin_parent == anim_parent

        if not hierarchy_match:
            hierarchy_mismatches.append(
                {
                    "bone_name": bone_name,
                    "skinning_parent": skin_parent,
                    "animation_parent": anim_parent,
                }
            )

        skin_local = bone_local_rest_matrix(skinning, bone_name)
        anim_local = bone_local_rest_matrix(animation, bone_name)

        skin_world = bone_world_rest_matrix(skinning, bone_name)
        anim_world = bone_world_rest_matrix(animation, bone_name)

        local_stats = matrix_compare_stats(skin_local, anim_local)
        world_stats = matrix_compare_stats(skin_world, anim_world)

        local_stats_items.append(local_stats)
        world_stats_items.append(world_stats)

        per_bone.append(
            {
                "bone_name": bone_name,
                "skinning_parent": skin_parent,
                "animation_parent": anim_parent,
                "hierarchy_match": bool(hierarchy_match),
                "local": local_stats,
                "world": world_stats,
            }
        )

    def top_by(field_path: tuple[str, str]) -> list[dict[str, Any]]:
        scope, key = field_path
        return sorted(
            per_bone,
            key=lambda item: float(item[scope][key]),
            reverse=True,
        )[: int(top_n)]

    return {
        "bone_name_correspondence": {
            "skinning_bone_count": int(len(skin_bones)),
            "animation_bone_count": int(len(anim_bones)),
            "common_bone_count": int(len(common)),
            "exact_set_match": bool(skin_set == anim_set),
            "exact_order_match": bool(skin_bones == anim_bones),
            "skinning_only_count": int(len(skin_only)),
            "animation_only_count": int(len(anim_only)),
            "skinning_only_first_100": skin_only[:100],
            "animation_only_first_100": anim_only[:100],
            "skinning_bone_names_first_100": skin_bones[:100],
            "animation_bone_names_first_100": anim_bones[:100],
        },
        "hierarchy": {
            "checked_common_bones": int(len(common)),
            "mismatch_count": int(len(hierarchy_mismatches)),
            "all_match": bool(len(hierarchy_mismatches) == 0),
            "mismatches_first_100": hierarchy_mismatches[:100],
        },
        "rest_pose": {
            "checked_common_bones": int(len(common)),
            "local_summary": {
                **max_stats(local_stats_items),
                **mean_stats(local_stats_items),
            },
            "world_summary": {
                **max_stats(world_stats_items),
                **mean_stats(world_stats_items),
            },
            "top_local_rotation": top_by(("local", "rotation_deg")),
            "top_local_translation": top_by(("local", "translation_l2")),
            "top_world_rotation": top_by(("world", "rotation_deg")),
            "top_world_translation": top_by(("world", "translation_l2")),
            "top_world_frobenius": top_by(("world", "frobenius_no_scale")),
        },
        "per_bone": per_bone,
    }


def write_per_bone_csv(path: Path, per_bone: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "bone_name",
        "skinning_parent",
        "animation_parent",
        "hierarchy_match",
        "local_translation_l2",
        "local_rotation_deg",
        "local_scale_l2",
        "local_frobenius_no_scale",
        "local_x_axis_deg",
        "local_y_axis_deg",
        "local_z_axis_deg",
        "world_translation_l2",
        "world_rotation_deg",
        "world_scale_l2",
        "world_frobenius_no_scale",
        "world_x_axis_deg",
        "world_y_axis_deg",
        "world_z_axis_deg",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in per_bone:
            row = {
                "bone_name": item["bone_name"],
                "skinning_parent": item["skinning_parent"],
                "animation_parent": item["animation_parent"],
                "hierarchy_match": item["hierarchy_match"],
                "local_translation_l2": item["local"]["translation_l2"],
                "local_rotation_deg": item["local"]["rotation_deg"],
                "local_scale_l2": item["local"]["scale_l2"],
                "local_frobenius_no_scale": item["local"]["frobenius_no_scale"],
                "local_x_axis_deg": item["local"]["x_axis_deg"],
                "local_y_axis_deg": item["local"]["y_axis_deg"],
                "local_z_axis_deg": item["local"]["z_axis_deg"],
                "world_translation_l2": item["world"]["translation_l2"],
                "world_rotation_deg": item["world"]["rotation_deg"],
                "world_scale_l2": item["world"]["scale_l2"],
                "world_frobenius_no_scale": item["world"]["frobenius_no_scale"],
                "world_x_axis_deg": item["world"]["x_axis_deg"],
                "world_y_axis_deg": item["world"]["y_axis_deg"],
                "world_z_axis_deg": item["world"]["z_axis_deg"],
            }
            writer.writerow(row)


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare rest poses between skinning and animation armatures in a Winter Samurai diagnostic blend."
    )

    parser.add_argument("--input_blend", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--skinning_armature_name", type=str, default=None)
    parser.add_argument("--animation_armature_name", type=str, default=None)

    parser.add_argument("--top_n", type=int, default=20)

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero if bone names mismatch, hierarchy mismatches, or rest-pose "
            "differences exceed threshold."
        ),
    )
    parser.add_argument("--max_allowed_world_rotation_deg", type=float, default=1.0)
    parser.add_argument("--max_allowed_world_translation_l2", type=float, default=1e-4)

    args = parser.parse_args(script_argv())

    input_blend = Path(args.input_blend).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)

    report_path = output_dir / "rest_pose_compare_report.json"
    csv_path = output_dir / "rest_pose_compare_per_bone.csv"

    repo_root = resolve_repo_root()

    log("=" * 96)
    log("Winter Samurai armature rest-pose comparison")
    log(f"repo_root   : {repo_root}")
    log(f"input_blend : {input_blend}")
    log(f"output_dir  : {output_dir}")
    log(f"report_json : {report_path}")
    log(f"csv         : {csv_path}")
    log("=" * 96)

    open_blend(input_blend)

    armatures = get_armatures()

    if len(armatures) != 2:
        fail(
            "Expected exactly two ARMATURE objects in the diagnostic blend, "
            f"got {len(armatures)}: {[obj.name for obj in armatures]}"
        )

    selection = select_armatures(
        skinning_armature_name=args.skinning_armature_name,
        animation_armature_name=args.animation_armature_name,
    )

    skinning = selection.skinning
    animation = selection.animation

    log(f"Selected skinning armature : {skinning.name}")
    log(f"Selected animation armature: {animation.name}")
    log(f"Selection reason           : {selection.reason}")

    compare = compare_bones(
        skinning=skinning,
        animation=animation,
        top_n=int(args.top_n),
    )

    scene_summary = {
        "armature_count": int(len(armatures)),
        "armatures": [armature_summary(arm) for arm in armatures],
        "mesh_count": int(len(get_meshes())),
    }

    name_ok = bool(compare["bone_name_correspondence"]["exact_set_match"])
    hierarchy_ok = bool(compare["hierarchy"]["all_match"])

    world_summary = compare["rest_pose"]["world_summary"]
    world_rot_max = float(world_summary.get("rotation_deg_max", 0.0))
    world_trans_max = float(world_summary.get("translation_l2_max", 0.0))

    threshold_ok = (
        world_rot_max <= float(args.max_allowed_world_rotation_deg)
        and world_trans_max <= float(args.max_allowed_world_translation_l2)
    )

    status = {
        "name_correspondence_ok": name_ok,
        "hierarchy_ok": hierarchy_ok,
        "world_rest_pose_threshold_ok": threshold_ok,
        "world_rotation_deg_max": world_rot_max,
        "world_translation_l2_max": world_trans_max,
        "max_allowed_world_rotation_deg": float(args.max_allowed_world_rotation_deg),
        "max_allowed_world_translation_l2": float(args.max_allowed_world_translation_l2),
    }

    report = {
        "schema": "ogd_lbs_winter_samurai_rest_pose_compare_v1",
        "created_at_utc": utc_now_iso(),
        "repo_root": str(repo_root) if repo_root is not None else None,
        "input_blend": str(input_blend),
        "output_dir": str(output_dir),
        "report_json": str(report_path),
        "per_bone_csv": str(csv_path),
        "selected_armatures": {
            "skinning": skinning.name,
            "animation": animation.name,
            "reason": selection.reason,
        },
        "status": status,
        "scene": scene_summary,
        "comparison": compare,
    }

    dump_json(report_path, report)
    write_per_bone_csv(csv_path, compare["per_bone"])

    log("")
    log("=" * 96)
    log("Rest-pose comparison summary")
    log("=" * 96)
    log(f"skinning armature          : {skinning.name}")
    log(f"animation armature         : {animation.name}")
    log(f"name exact set match       : {name_ok}")
    log(f"name exact order match     : {compare['bone_name_correspondence']['exact_order_match']}")
    log(f"skinning only bones        : {compare['bone_name_correspondence']['skinning_only_count']}")
    log(f"animation only bones       : {compare['bone_name_correspondence']['animation_only_count']}")
    log(f"hierarchy all match        : {hierarchy_ok}")
    log(f"hierarchy mismatch count   : {compare['hierarchy']['mismatch_count']}")
    log(f"world rotation_deg_max     : {world_rot_max}")
    log(f"world translation_l2_max   : {world_trans_max}")
    log(f"report_json                : {report_path}")
    log(f"per_bone_csv               : {csv_path}")
    log("=" * 96)

    if args.strict:
        if not name_ok:
            fail("Strict mode failed: bone name sets do not match.")
        if not hierarchy_ok:
            fail("Strict mode failed: hierarchy mismatches detected.")
        if not threshold_ok:
            fail(
                "Strict mode failed: world rest-pose difference exceeds threshold. "
                f"rotation_max={world_rot_max}, translation_max={world_trans_max}"
            )


if __name__ == "__main__":
    main()
