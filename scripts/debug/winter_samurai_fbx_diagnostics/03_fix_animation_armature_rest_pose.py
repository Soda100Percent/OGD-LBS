#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Winter Samurai FBX diagnostics Step 03:
Fix animation armature rest pose by copying the skinning armature rest pose.

This is an experimental cleaning step.

Input:
    A diagnostic blend produced by 01_import_model_and_anims.py, expected to
    contain exactly two ARMATURE objects:
        - skinning armature: mesh-bound armature from model FBX
        - animation armature: armature from animation FBX

Default behavior:
    - fail fast unless exactly two armatures exist
    - auto-detect skinning armature as the unique mesh-bound armature
    - require identical bone name sets
    - set animation armature object matrix_world = skinning armature matrix_world
    - copy same-name bone hierarchy from skinning armature to animation armature
    - copy same-name edit-bone rest transforms from skinning to animation
    - save a new output blend
    - write before/after comparison reports

Important:
    This script does not retarget animation yet. It only changes the animation
    armature's rest pose / hierarchy to match the skinning armature.
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
    print(f"[winter-samurai-fix-rest] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[winter-samurai-fix-rest][WARN] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[winter-samurai-fix-rest][ERROR] {msg}", flush=True)
    raise RuntimeError(msg)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def open_blend(path: Path) -> None:
    if not path.is_file():
        fail(f"Input blend does not exist: {path}")

    log(f"Opening blend: {path}")
    result = bpy.ops.wm.open_mainfile(filepath=str(path))
    if result != {"FINISHED"}:
        fail(f"Failed to open blend: {path}, result={result}")


def ensure_object_mode() -> None:
    try:
        active = bpy.context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def update_view_layer() -> None:
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
    except Exception:
        pass


def safe_float_list(values: Any) -> list[float]:
    try:
        return [float(x) for x in values]
    except Exception:
        return []


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
    active_action = None
    nla_track_count = 0
    nla_strip_count = 0
    driver_count = 0

    if ad is not None:
        if ad.action is not None:
            active_action = ad.action.name

        nla_track_count = int(len(ad.nla_tracks))
        for track in ad.nla_tracks:
            nla_strip_count += int(len(track.strips))

        try:
            driver_count = int(len(ad.drivers))
        except Exception:
            driver_count = 0

    score = 0
    if active_action is not None:
        score += 10
    score += nla_strip_count * 5
    score += driver_count

    return {
        "has_animation_data": bool(ad is not None),
        "active_action": active_action,
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

    binding_score = len(modifier_meshes) * 10 + len(parented_meshes)

    return {
        "bound_meshes_by_armature_modifier": modifier_meshes,
        "bound_meshes_by_parent": parented_meshes,
        "all_meshes_with_any_armature_modifier_first_50": any_armature_modifier_meshes[:50],
        "armature_modifier_mesh_count": int(len(modifier_meshes)),
        "parented_mesh_count": int(len(parented_meshes)),
        "binding_score": int(binding_score),
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

        return SelectedArmatures(skinning=skin, animation=anim, reason="explicit_override")

    binding = {arm.name: mesh_binding_info(arm) for arm in armatures}

    bound_candidates = [
        arm for arm in armatures
        if binding[arm.name]["binding_score"] > 0
    ]

    if len(bound_candidates) == 1:
        skin = bound_candidates[0]
        anim = armatures[0] if armatures[1] is skin else armatures[1]
        return SelectedArmatures(
            skinning=skin,
            animation=anim,
            reason="unique_mesh_bound_armature",
        )

    fail(
        "Could not auto-select skinning/animation armatures.\n"
        + json.dumps(
            [
                {
                    "name": arm.name,
                    "binding": binding[arm.name],
                    "animation": animation_info(arm),
                }
                for arm in armatures
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\nUse --skinning_armature_name and --animation_armature_name to override."
    )

    raise RuntimeError("unreachable")


def bone_names(armature: bpy.types.Object) -> list[str]:
    return [bone.name for bone in armature.data.bones]


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


def vec_l2(a: Vector, b: Vector) -> float:
    return float((a - b).length)


def matrix_compare_stats(a: Matrix, b: Matrix) -> dict[str, float]:
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


def compare_rest_pose(
    *,
    skinning: bpy.types.Object,
    animation: bpy.types.Object,
) -> dict[str, Any]:
    skin_bones = bone_names(skinning)
    anim_bones = bone_names(animation)

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
        },
        "per_bone": per_bone,
    }


def validate_bone_name_sets(skinning: bpy.types.Object, animation: bpy.types.Object) -> None:
    skin_set = set(bone_names(skinning))
    anim_set = set(bone_names(animation))

    if skin_set != anim_set:
        fail(
            "Bone name sets differ; refusing to copy rest pose.\n"
            f"skinning_only={sorted(skin_set - anim_set)[:50]}\n"
            f"animation_only={sorted(anim_set - skin_set)[:50]}"
        )


def set_edit_bone_from_source_bone(
    *,
    edit_bone: bpy.types.EditBone,
    source_bone: bpy.types.Bone,
    min_length: float,
) -> None:
    mat = source_bone.matrix_local.copy()
    loc, rot, _scale = mat.decompose()

    length = max(float(source_bone.length), float(min_length))

    edit_bone.head = loc
    edit_bone.tail = loc + (rot @ Vector((0.0, length, 0.0)))

    try:
        edit_bone.align_roll(rot @ Vector((0.0, 0.0, 1.0)))
    except Exception as exc:
        warn(f"align_roll failed for bone {edit_bone.name!r}: {exc}")


def copy_rest_pose_to_animation_armature(
    *,
    skinning: bpy.types.Object,
    animation: bpy.types.Object,
    match_object_transform: bool,
    match_hierarchy: bool,
    copy_deform_flags: bool,
    min_bone_length: float,
) -> dict[str, Any]:
    """
    Copy skinning armature rest pose / hierarchy into animation armature.
    """
    validate_bone_name_sets(skinning, animation)

    ensure_object_mode()

    if match_object_transform:
        animation.matrix_world = skinning.matrix_world.copy()
        update_view_layer()

    skin_bones = {bone.name: bone for bone in skinning.data.bones}
    skin_bone_names = bone_names(skinning)

    bpy.ops.object.select_all(action="DESELECT")
    animation.select_set(True)
    bpy.context.view_layer.objects.active = animation

    bpy.ops.object.mode_set(mode="EDIT")

    ebones = animation.data.edit_bones

    # Disable connection first so parent changes and transform edits do not
    # immediately force heads to parent tails.
    for name in skin_bone_names:
        eb = ebones.get(name)
        if eb is not None:
            eb.use_connect = False

    if match_hierarchy:
        for name in skin_bone_names:
            eb = ebones.get(name)
            src = skin_bones[name]
            if eb is None:
                fail(f"Animation edit bone missing unexpectedly: {name}")

            if src.parent is None:
                eb.parent = None
            else:
                parent_eb = ebones.get(src.parent.name)
                if parent_eb is None:
                    fail(f"Animation edit bone missing source parent {src.parent.name!r} for bone {name!r}")
                eb.parent = parent_eb

    # Copy rest transforms after parent assignment.
    for name in skin_bone_names:
        eb = ebones.get(name)
        src = skin_bones[name]
        if eb is None:
            fail(f"Animation edit bone missing unexpectedly: {name}")

        set_edit_bone_from_source_bone(
            edit_bone=eb,
            source_bone=src,
            min_length=float(min_bone_length),
        )

        if copy_deform_flags:
            try:
                eb.use_deform = bool(src.use_deform)
            except Exception:
                pass

    # Restore use_connect to source state only after transforms are copied.
    if match_hierarchy:
        for name in skin_bone_names:
            eb = ebones.get(name)
            src = skin_bones[name]
            if eb is None:
                continue

            if src.parent is None:
                eb.use_connect = False
            else:
                try:
                    eb.use_connect = bool(src.use_connect)
                except Exception:
                    pass

    bpy.ops.object.mode_set(mode="OBJECT")
    update_view_layer()

    return {
        "match_object_transform": bool(match_object_transform),
        "match_hierarchy": bool(match_hierarchy),
        "copy_deform_flags": bool(copy_deform_flags),
        "min_bone_length": float(min_bone_length),
        "copied_bone_count": int(len(skin_bone_names)),
    }


def summarize_object(obj: bpy.types.Object) -> dict[str, Any]:
    return {
        "name": obj.name,
        "type": obj.type,
        "location": safe_float_list(obj.location),
        "rotation_euler": safe_float_list(obj.rotation_euler),
        "scale": safe_float_list(obj.scale),
        "matrix_world": [[float(obj.matrix_world[r][c]) for c in range(4)] for r in range(4)],
        "bone_count": int(len(obj.data.bones)) if obj.type == "ARMATURE" else None,
        "animation": animation_info(obj) if obj.type == "ARMATURE" else None,
        "mesh_binding": mesh_binding_info(obj) if obj.type == "ARMATURE" else None,
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
        "world_translation_l2",
        "world_rotation_deg",
        "world_scale_l2",
        "world_frobenius_no_scale",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in per_bone:
            writer.writerow(
                {
                    "bone_name": item["bone_name"],
                    "skinning_parent": item["skinning_parent"],
                    "animation_parent": item["animation_parent"],
                    "hierarchy_match": item["hierarchy_match"],
                    "local_translation_l2": item["local"]["translation_l2"],
                    "local_rotation_deg": item["local"]["rotation_deg"],
                    "local_scale_l2": item["local"]["scale_l2"],
                    "local_frobenius_no_scale": item["local"]["frobenius_no_scale"],
                    "world_translation_l2": item["world"]["translation_l2"],
                    "world_rotation_deg": item["world"]["rotation_deg"],
                    "world_scale_l2": item["world"]["scale_l2"],
                    "world_frobenius_no_scale": item["world"]["frobenius_no_scale"],
                }
            )


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text_note(
    *,
    output_blend: Path,
    report_path: Path,
    skinning: bpy.types.Object,
    animation: bpy.types.Object,
) -> None:
    text = bpy.data.texts.new("WINTER_SAMURAI_REST_POSE_FIX_NOTES")
    text.write("Winter Samurai animation armature rest-pose fix\n")
    text.write("=" * 80 + "\n")
    text.write(f"output_blend: {output_blend}\n")
    text.write(f"report_json : {report_path}\n")
    text.write(f"skinning armature : {skinning.name}\n")
    text.write(f"animation armature: {animation.name}\n")
    text.write("\n")
    text.write("This file was generated by copying the skinning armature rest pose\n")
    text.write("and hierarchy onto the animation armature. Animation retargeting has\n")
    text.write("not been performed in this step.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix Winter Samurai animation armature rest pose by copying skinning armature rest pose."
    )

    parser.add_argument("--input_blend", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_blend", type=str, default=None)

    parser.add_argument("--skinning_armature_name", type=str, default=None)
    parser.add_argument("--animation_armature_name", type=str, default=None)

    parser.add_argument(
        "--no_match_object_transform",
        dest="match_object_transform",
        action="store_false",
        help="Do not set animation armature object transform to skinning armature object transform.",
    )
    parser.set_defaults(match_object_transform=True)

    parser.add_argument(
        "--no_match_hierarchy",
        dest="match_hierarchy",
        action="store_false",
        help="Do not copy bone hierarchy / parent relationships.",
    )
    parser.set_defaults(match_hierarchy=True)

    parser.add_argument(
        "--no_copy_deform_flags",
        dest="copy_deform_flags",
        action="store_false",
    )
    parser.set_defaults(copy_deform_flags=True)

    parser.add_argument("--min_bone_length", type=float, default=1e-6)

    args = parser.parse_args(script_argv())

    input_blend = Path(args.input_blend).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_blend = (
        Path(args.output_blend).expanduser().resolve(strict=False)
        if args.output_blend
        else (output_dir / "Winter_Samurai_anim_rest_fixed.blend").resolve(strict=False)
    )

    report_path = output_dir / "rest_pose_fix_report.json"
    before_csv_path = output_dir / "rest_pose_fix_before_per_bone.csv"
    after_csv_path = output_dir / "rest_pose_fix_after_per_bone.csv"

    log("=" * 96)
    log("Winter Samurai animation armature rest-pose fix")
    log(f"input_blend : {input_blend}")
    log(f"output_dir  : {output_dir}")
    log(f"output_blend: {output_blend}")
    log(f"report_json : {report_path}")
    log("=" * 96)

    open_blend(input_blend)

    selected = select_armatures(
        skinning_armature_name=args.skinning_armature_name,
        animation_armature_name=args.animation_armature_name,
    )

    skinning = selected.skinning
    animation = selected.animation

    log(f"Selected skinning armature : {skinning.name}")
    log(f"Selected animation armature: {animation.name}")
    log(f"Selection reason           : {selected.reason}")

    before_compare = compare_rest_pose(skinning=skinning, animation=animation)

    name_ok = bool(before_compare["bone_name_correspondence"]["exact_set_match"])
    if not name_ok:
        fail("Bone name sets do not match. Refusing to modify animation armature.")

    fix_summary = copy_rest_pose_to_animation_armature(
        skinning=skinning,
        animation=animation,
        match_object_transform=bool(args.match_object_transform),
        match_hierarchy=bool(args.match_hierarchy),
        copy_deform_flags=bool(args.copy_deform_flags),
        min_bone_length=float(args.min_bone_length),
    )

    after_compare = compare_rest_pose(skinning=skinning, animation=animation)

    write_text_note(
        output_blend=output_blend,
        report_path=report_path,
        skinning=skinning,
        animation=animation,
    )

    output_blend.parent.mkdir(parents=True, exist_ok=True)
    save_result = bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    if save_result != {"FINISHED"}:
        fail(f"Failed to save output blend: {output_blend}, result={save_result}")

    write_per_bone_csv(before_csv_path, before_compare["per_bone"])
    write_per_bone_csv(after_csv_path, after_compare["per_bone"])

    before_world = before_compare["rest_pose"]["world_summary"]
    after_world = after_compare["rest_pose"]["world_summary"]

    report = {
        "schema": "ogd_lbs_winter_samurai_fix_animation_rest_pose_v1",
        "created_at_utc": utc_now_iso(),
        "input_blend": str(input_blend),
        "output_dir": str(output_dir),
        "output_blend": str(output_blend),
        "report_json": str(report_path),
        "before_csv": str(before_csv_path),
        "after_csv": str(after_csv_path),
        "selected_armatures": {
            "skinning": skinning.name,
            "animation": animation.name,
            "reason": selected.reason,
        },
        "config": {
            "match_object_transform": bool(args.match_object_transform),
            "match_hierarchy": bool(args.match_hierarchy),
            "copy_deform_flags": bool(args.copy_deform_flags),
            "min_bone_length": float(args.min_bone_length),
        },
        "fix_summary": fix_summary,
        "object_summaries_after": {
            "skinning": summarize_object(skinning),
            "animation": summarize_object(animation),
        },
        "before": before_compare,
        "after": after_compare,
    }

    dump_json(report_path, report)

    log("")
    log("=" * 96)
    log("Rest-pose fix summary")
    log("=" * 96)
    log(f"before hierarchy mismatch count : {before_compare['hierarchy']['mismatch_count']}")
    log(f"after hierarchy mismatch count  : {after_compare['hierarchy']['mismatch_count']}")
    log(f"before world rotation max       : {before_world.get('rotation_deg_max')}")
    log(f"after world rotation max        : {after_world.get('rotation_deg_max')}")
    log(f"before world translation max    : {before_world.get('translation_l2_max')}")
    log(f"after world translation max     : {after_world.get('translation_l2_max')}")
    log(f"output_blend                    : {output_blend}")
    log(f"report_json                     : {report_path}")
    log("=" * 96)


if __name__ == "__main__":
    main()
