#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
03a: Replace animation-armature rest data while preserving sampled evaluated
bone world matrices.

Compatibility version:
    Does NOT require Bone.convert_local_to_pose.

Run on the ORIGINAL Step 01 imported blend, not the previous Step 03 output.

Core idea:
    1. Cache evaluated world-space bone matrices before modifying the rig.
    2. Replace animation armature data with a copy of skinning armature data.
    3. Fix animation armature object transform to skinning armature transform.
    4. For each sampled frame and bone:
          desired_pose_matrix = inv(animation_object_world) @ cached_old_world_matrix
       then assign:
          pose_bone.matrix = desired_pose_matrix
       and keyframe location / rotation_quaternion / scale.
    5. Validate before save and after reopening the saved blend.

This script does not retarget the skinning mesh yet. It only makes the
animation armature have the skinning rest pose while preserving its sampled
world-space bone motion.
"""

import argparse
import csv
import json
import math
import sys
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix


TAG = "[ws-rest-rebase-03a]"


def log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{TAG}[WARN] {msg}", flush=True)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def np_mat(m: Matrix) -> np.ndarray:
    return np.asarray(m, dtype=np.float64).copy()


def bl_mat(a: np.ndarray) -> Matrix:
    return Matrix(np.asarray(a, dtype=np.float64).tolist())


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
        bpy.context.evaluated_depsgraph_get().update()
    except Exception:
        pass


def frame_set(t: float) -> None:
    base = math.floor(float(t))
    sub = float(t) - base
    bpy.context.scene.frame_set(base, subframe=sub)
    update_view_layer()


def open_blend(path: Path) -> None:
    require(path.is_file(), f"Input blend not found: {path}")
    log(f"Opening blend: {path}")
    result = bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    require(result == {"FINISHED"}, f"open_mainfile failed: {result}")
    ensure_object_mode()


def object_animation_info(obj: bpy.types.ID) -> dict[str, Any]:
    ad = obj.animation_data
    if ad is None:
        return {
            "present": False,
            "action": None,
            "nla_tracks": 0,
            "nla_strips": 0,
            "drivers": 0,
            "score": 0,
        }

    action = ad.action.name if ad.action else None
    nla_tracks = int(len(ad.nla_tracks))
    nla_strips = int(sum(len(track.strips) for track in ad.nla_tracks))
    try:
        drivers = int(len(ad.drivers))
    except Exception:
        drivers = 0

    score = 0
    if action:
        score += 10
    score += nla_strips * 5
    score += drivers

    return {
        "present": bool(action or nla_strips or drivers),
        "action": action,
        "nla_tracks": nla_tracks,
        "nla_strips": nla_strips,
        "drivers": drivers,
        "score": score,
    }


def mesh_binding_info(arm: bpy.types.Object) -> dict[str, Any]:
    modifier_meshes = []
    parented_meshes = []

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue

        if obj.parent is arm:
            parented_meshes.append(obj.name)

        for mod in obj.modifiers:
            if mod.type == "ARMATURE" and getattr(mod, "object", None) is arm:
                modifier_meshes.append(obj.name)

    modifier_meshes = sorted(set(modifier_meshes))
    parented_meshes = sorted(set(parented_meshes))

    score = len(modifier_meshes) * 10 + len(parented_meshes)

    return {
        "armature_modifier_meshes": modifier_meshes,
        "parented_meshes": parented_meshes,
        "armature_modifier_mesh_count": len(modifier_meshes),
        "parented_mesh_count": len(parented_meshes),
        "binding_score": score,
    }


def get_armatures() -> list[bpy.types.Object]:
    return sorted(
        [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"],
        key=lambda obj: obj.name,
    )


def get_meshes() -> list[bpy.types.Object]:
    return sorted(
        [obj for obj in bpy.context.scene.objects if obj.type == "MESH"],
        key=lambda obj: obj.name,
    )


def select_roles() -> tuple[bpy.types.Object, bpy.types.Object, list[str], list[dict[str, Any]]]:
    arms = get_armatures()

    require(
        len(arms) == 2,
        f"Expected exactly 2 armatures, got {len(arms)}: {[a.name for a in arms]}",
    )

    evidence = []
    for arm in arms:
        bind = mesh_binding_info(arm)
        obj_anim = object_animation_info(arm)
        data_anim = object_animation_info(arm.data)
        animated = obj_anim["present"] or data_anim["present"]

        evidence.append(
            {
                "name": arm.name,
                "mesh_binding": bind,
                "object_animation": obj_anim,
                "data_animation": data_anim,
                "animated": bool(animated),
            }
        )

    log("Role evidence:")
    log(json.dumps(evidence, indent=2, ensure_ascii=False))

    bound = [arm for arm in arms if mesh_binding_info(arm)["binding_score"] > 0]

    require(
        len(bound) == 1,
        "Expected exactly one mesh-bound armature. "
        f"Evidence={json.dumps(evidence, indent=2, ensure_ascii=False)}",
    )

    skin = bound[0]
    anim = arms[0] if arms[1] is skin else arms[1]

    skin_names = [b.name for b in skin.data.bones]
    anim_names = [b.name for b in anim.data.bones]

    require(
        set(skin_names) == set(anim_names),
        "Bone name sets differ; refusing to rebase. "
        f"skin_only={sorted(set(skin_names) - set(anim_names))[:50]}, "
        f"anim_only={sorted(set(anim_names) - set(skin_names))[:50]}",
    )

    require(len(skin_names) > 0, "Empty skeleton.")

    return skin, anim, sorted(skin_names), evidence


def parent_map(obj: bpy.types.Object, names: list[str]) -> dict[str, str | None]:
    out = {}
    for name in names:
        bone = obj.data.bones[name]
        out[name] = bone.parent.name if bone.parent else None
    return out


def rest_array(obj: bpy.types.Object, names: list[str]) -> np.ndarray:
    return np.stack([np_mat(obj.data.bones[name].matrix_local) for name in names], axis=0)


def assert_affine(a: np.ndarray, label: str) -> None:
    a = np.asarray(a, dtype=np.float64)
    require(np.isfinite(a).all(), f"{label}: NaN/Inf found.")
    require(
        np.allclose(a[..., 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0),
        f"{label}: invalid homogeneous matrix bottom row.",
    )
    s = np.linalg.svd(a[..., :3, :3], compute_uv=False)
    require(
        np.all(s[..., -1] > 1e-12),
        f"{label}: singular or near-singular transform.",
    )


def closest_rotation(linear: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(linear)
    r = u @ vt
    det = np.linalg.det(r)
    if np.any(det < 0):
        u = u.copy()
        if r.ndim == 2:
            u[:, -1] *= -1
        else:
            mask = det < 0
            u[mask, :, -1] *= -1
        r = u @ vt
    return r


def error_arrays(actual: np.ndarray, expected: np.ndarray) -> dict[str, np.ndarray]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite matrix in comparison.")

    trans = np.linalg.norm(actual[..., :3, 3] - expected[..., :3, 3], axis=-1)

    lin_rel = np.linalg.norm(
        actual[..., :3, :3] - expected[..., :3, :3],
        axis=(-2, -1),
    ) / np.maximum(np.linalg.norm(expected[..., :3, :3], axis=(-2, -1)), 1e-12)

    ra = closest_rotation(actual[..., :3, :3])
    rb = closest_rotation(expected[..., :3, :3])
    delta = np.swapaxes(ra, -1, -2) @ rb

    tr = np.trace(delta, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)

    skew = np.stack(
        [
            delta[..., 2, 1] - delta[..., 1, 2],
            delta[..., 0, 2] - delta[..., 2, 0],
            delta[..., 1, 0] - delta[..., 0, 1],
        ],
        axis=-1,
    )
    sin = 0.5 * np.linalg.norm(skew, axis=-1)
    rot = np.degrees(np.arctan2(sin, cos))

    mat_abs = np.max(np.abs(actual - expected), axis=(-2, -1))

    return {
        "translation_l2": trans,
        "rotation_deg": rot,
        "linear_relative": lin_rel,
        "matrix_abs": mat_abs,
    }


def summarize_errors(errors: dict[str, np.ndarray], times: np.ndarray, names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, values in errors.items():
        values = np.asarray(values)
        idx = int(np.argmax(values))
        unraveled = np.unravel_index(idx, values.shape)
        worst_frame = float(times[unraveled[0]]) if values.ndim >= 1 else None
        worst_bone = names[unraveled[1]] if values.ndim >= 2 and len(names) > unraveled[1] else None

        out[f"{key}_max"] = float(np.max(values))
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_worst"] = {
            "frame": worst_frame,
            "bone": worst_bone,
        }
    return out


def capture_world(
    object_names: list[str],
    bone_names: list[str],
    times: np.ndarray,
    label: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    poses = {
        obj_name: np.empty((len(times), len(bone_names), 4, 4), dtype=np.float64)
        for obj_name in object_names
    }
    objects = {
        obj_name: np.empty((len(times), 4, 4), dtype=np.float64)
        for obj_name in object_names
    }

    for fi, t in enumerate(times):
        frame_set(float(t))
        dg = bpy.context.evaluated_depsgraph_get()

        for obj_name in object_names:
            original = bpy.data.objects[obj_name]
            evaluated = original.evaluated_get(dg)
            obj_world = evaluated.matrix_world.copy()
            objects[obj_name][fi] = np_mat(obj_world)

            for bi, bone_name in enumerate(bone_names):
                poses[obj_name][fi, bi] = np_mat(
                    obj_world @ evaluated.pose.bones[bone_name].matrix
                )

        if fi == 0 or fi == len(times) - 1 or (fi + 1) % 25 == 0:
            log(f"{label}: sample {fi + 1}/{len(times)}, frame={t:g}")

    for obj_name in object_names:
        assert_affine(poses[obj_name], f"{label}/{obj_name}/bone_world")
        assert_affine(objects[obj_name], f"{label}/{obj_name}/object_world")

    return poses, objects


def preserve_actions(id_block: bpy.types.ID) -> list[str]:
    names = []
    ad = id_block.animation_data
    if ad is None:
        return names

    def keep(action):
        if action is not None:
            try:
                action.use_fake_user = True
                names.append(action.name)
            except Exception:
                pass

    keep(ad.action)

    for track in ad.nla_tracks:
        for strip in track.strips:
            keep(strip.action)

    return sorted(set(names))


def decompose_checked(matrix: Matrix, tol: float, label: str):
    loc, quat, scale = matrix.decompose()
    quat.normalize()

    rebuilt = (
        Matrix.Translation(loc)
        @ quat.to_matrix().to_4x4()
        @ Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0))
    )

    a = np_mat(matrix)
    b = np_mat(rebuilt)

    rel = float(
        np.linalg.norm(a[:3, :3] - b[:3, :3])
        / max(float(np.linalg.norm(a[:3, :3])), 1e-12)
    )

    require(
        math.isfinite(rel) and rel <= tol,
        f"{label}: local matrix not representable by location/quaternion/scale; "
        f"residual={rel:.9g}, tol={tol}",
    )

    return loc, quat, scale, rel


def prepare_target(
    skin: bpy.types.Object,
    anim: bpy.types.Object,
    reference_world: np.ndarray,
    args,
) -> dict[str, Any]:
    ensure_object_mode()

    changes: dict[str, Any] = {
        "old_armature_data": anim.data.name,
        "old_object_parent": anim.parent.name if anim.parent else None,
        "old_object_animation": object_animation_info(anim),
        "old_data_animation": object_animation_info(anim.data),
        "preserved_old_actions": sorted(set(preserve_actions(anim) + preserve_actions(anim.data))),
        "removed_object_constraints": [{"name": c.name, "type": c.type} for c in anim.constraints],
        "removed_pose_constraints": [
            {"bone": pb.name, "name": c.name, "type": c.type}
            for pb in anim.pose.bones
            for c in pb.constraints
        ],
    }

    old_data = anim.data
    old_data.use_fake_user = True

    anim.animation_data_clear()
    old_data.animation_data_clear()

    for c in list(anim.constraints):
        anim.constraints.remove(c)

    anim.parent = None
    anim.matrix_parent_inverse = Matrix.Identity(4)
    anim.delta_location = (0, 0, 0)
    anim.delta_rotation_euler = (0, 0, 0)
    anim.delta_rotation_quaternion = (1, 0, 0, 0)
    anim.delta_scale = (1, 1, 1)

    anim.rotation_mode = "QUATERNION"
    loc, quat, scale, obj_res = decompose_checked(
        bl_mat(reference_world),
        args.trs_tol,
        "destination_object_world",
    )
    anim.location = loc
    anim.rotation_quaternion = quat
    anim.scale = scale

    # Replace armature data with independent copy of skinning rest pose / hierarchy.
    anim.data = skin.data.copy()
    anim.data.name = old_data.name + "_03a_RebasedRest"
    anim.data.animation_data_clear()
    anim.data.pose_position = "POSE"

    update_view_layer()

    bpy.ops.object.select_all(action="DESELECT")
    anim.select_set(True)
    bpy.context.view_layer.objects.active = anim

    result = bpy.ops.object.mode_set(mode="EDIT")
    require(result == {"FINISHED"}, "Failed to enter EDIT mode for animation armature.")

    disconnected = []
    try:
        for eb in anim.data.edit_bones:
            if eb.use_connect:
                disconnected.append(eb.name)
            eb.use_connect = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    for pb in anim.pose.bones:
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        pb.rotation_mode = "QUATERNION"
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.scale = (1, 1, 1)

    update_view_layer()

    changes.update(
        {
            "new_armature_data": anim.data.name,
            "object_trs_residual": obj_res,
            "disconnected_bones": sorted(disconnected),
            "policy": "object fixed to skinning reference world; motion baked into pose bones",
        }
    )
    return changes


def parent_first_order(obj: bpy.types.Object) -> list[str]:
    roots = sorted([pb.name for pb in obj.pose.bones if pb.parent is None])
    q = deque(roots)
    order = []

    while q:
        name = q.popleft()
        order.append(name)
        children = sorted([child.name for child in obj.pose.bones[name].children])
        q.extend(children)

    require(
        len(order) == len(obj.pose.bones),
        "Parent-first order failed; hierarchy may be invalid.",
    )
    return order


def action_fcurves(action: bpy.types.Action):
    curves = []
    try:
        curves.extend(list(action.fcurves))
    except Exception:
        pass

    # Blender 4.4+ layered action fallback.
    if not curves:
        try:
            for layer in action.layers:
                for strip in layer.strips:
                    for bag in getattr(strip, "channelbags", []):
                        curves.extend(list(bag.fcurves))
        except Exception:
            pass

    return curves


def bake_world_preserved_action(
    anim: bpy.types.Object,
    bone_names: list[str],
    times: np.ndarray,
    old_anim_world: np.ndarray,
    target_object_world: np.ndarray,
    args,
) -> dict[str, Any]:
    order = parent_first_order(anim)
    name_to_idx = {name: i for i, name in enumerate(bone_names)}

    target_inv = bl_mat(target_object_world).inverted()

    worst_assignment_translation = 0.0
    worst_assignment_rotation = 0.0
    worst_assignment_linear = 0.0

    previous_quats: dict[str, Any] = {}

    for fi, t in enumerate(times):
        frame_set(float(t))

        desired_pose = {
            name: target_inv @ bl_mat(old_anim_world[fi, name_to_idx[name]])
            for name in bone_names
        }

        # Clear first, then assign parent-first.
        for pb in anim.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.scale = (1, 1, 1)

        update_view_layer()

        for name in order:
            pb = anim.pose.bones[name]
            pb.rotation_mode = "QUATERNION"

            pb.matrix = desired_pose[name]
            update_view_layer()

            # If Blender approximates because matrix is not representable,
            # catch it immediately.
            actual = np_mat(pb.matrix)[None, None]
            expected = np_mat(desired_pose[name])[None, None]
            e = error_arrays(actual, expected)

            worst_assignment_translation = max(worst_assignment_translation, float(e["translation_l2"][0, 0]))
            worst_assignment_rotation = max(worst_assignment_rotation, float(e["rotation_deg"][0, 0]))
            worst_assignment_linear = max(worst_assignment_linear, float(e["linear_relative"][0, 0]))

            require(
                float(e["translation_l2"][0, 0]) <= args.position_tol
                and float(e["rotation_deg"][0, 0]) <= args.rotation_tol_deg
                and float(e["linear_relative"][0, 0]) <= args.linear_tol,
                f"Pose assignment failed at frame={t:g}, bone={name}: "
                f"translation={float(e['translation_l2'][0,0])}, "
                f"rotation={float(e['rotation_deg'][0,0])}, "
                f"linear={float(e['linear_relative'][0,0])}",
            )

            # Quaternion sign continuity.
            q = pb.rotation_quaternion.copy()
            old_q = previous_quats.get(name)
            if old_q is not None and q.dot(old_q) < 0:
                q.negate()
                pb.rotation_quaternion = q
                update_view_layer()
            previous_quats[name] = q.copy()

            for channel in ("location", "rotation_quaternion", "scale"):
                ok = pb.keyframe_insert(data_path=channel, frame=float(t), group=name)
                require(ok, f"Failed to insert keyframe: frame={t}, bone={name}, channel={channel}")

        if fi == 0 or fi == len(times) - 1 or (fi + 1) % 25 == 0:
            log(f"Bake: sample {fi + 1}/{len(times)}, frame={t:g}")

    require(anim.animation_data is not None and anim.animation_data.action is not None, "No baked action created.")
    action = anim.animation_data.action
    action.name = args.action_name
    action.use_fake_user = True

    curves = action_fcurves(action)
    for curve in curves:
        try:
            curve.extrapolation = "CONSTANT"
            for key in curve.keyframe_points:
                key.interpolation = "LINEAR"
            curve.update()
        except Exception:
            pass

    return {
        "action_name": action.name,
        "sample_count": int(len(times)),
        "bone_count": int(len(bone_names)),
        "fcurve_count": int(len(curves)),
        "interpolation": "LINEAR",
        "extrapolation": "CONSTANT",
        "worst_assignment_translation_l2": worst_assignment_translation,
        "worst_assignment_rotation_deg": worst_assignment_rotation,
        "worst_assignment_linear_relative": worst_assignment_linear,
    }


def audit(
    skin_name: str,
    anim_name: str,
    bone_names: list[str],
    times: np.ndarray,
    baseline_world: dict[str, np.ndarray],
    reference_world: np.ndarray,
    reference_rest: np.ndarray,
    reference_parents: dict[str, str | None],
    args,
    label: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    skin = bpy.data.objects[skin_name]
    anim = bpy.data.objects[anim_name]

    poses, objects = capture_world([skin_name, anim_name], bone_names, times, label)

    anim_errors = error_arrays(poses[anim_name], baseline_world[anim_name])
    skin_errors = error_arrays(poses[skin_name], baseline_world[skin_name])

    object_expected = np.repeat(reference_world[None, None, :, :], len(times), axis=0)
    object_errors = error_arrays(objects[anim_name][:, None, :, :], object_expected)

    target_rest = rest_array(anim, bone_names)
    skin_rest = rest_array(skin, bone_names)

    rest_errors = error_arrays(
        (reference_world @ target_rest)[None, :, :, :],
        (reference_world @ reference_rest)[None, :, :, :],
    )

    current_parents = parent_map(anim, bone_names)
    hierarchy_mismatches = [
        name
        for name in bone_names
        if current_parents.get(name) != reference_parents.get(name)
    ]

    skin_rest_abs = float(np.max(np.abs(skin_rest - reference_rest)))
    dest_rest_abs = float(np.max(np.abs(target_rest - reference_rest)))

    result = {
        "animation_world": summarize_errors(anim_errors, times, bone_names),
        "skinning_world_unchanged": summarize_errors(skin_errors, times, bone_names),
        "destination_object_world": summarize_errors(object_errors, times, ["ARMATURE_OBJECT"]),
        "rest_world_at_reference": summarize_errors(rest_errors, np.asarray([args.reference_frame], dtype=float), bone_names),
        "hierarchy_mismatches": hierarchy_mismatches,
        "skinning_rest_matrix_abs_max": skin_rest_abs,
        "destination_rest_matrix_abs_max": dest_rest_abs,
    }

    def ok(scope: str) -> bool:
        s = result[scope]
        return (
            s["translation_l2_max"] <= args.position_tol
            and s["rotation_deg_max"] <= args.rotation_tol_deg
            and s["linear_relative_max"] <= args.linear_tol
        )

    result["passed"] = bool(
        ok("animation_world")
        and ok("skinning_world_unchanged")
        and ok("destination_object_world")
        and ok("rest_world_at_reference")
        and not hierarchy_mismatches
        and skin_rest_abs <= 1e-5
        and dest_rest_abs <= 1e-5
    )

    return result, anim_errors


def write_error_csv(path: Path, errors: dict[str, np.ndarray], times: np.ndarray, bone_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["bone"]
    for key in errors:
        fields.extend([f"{key}_max", f"{key}_frame"])

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for bi, name in enumerate(bone_names):
            row = {"bone": name}
            for key, values in errors.items():
                fi = int(np.argmax(values[:, bi]))
                row[f"{key}_max"] = float(values[fi, bi])
                row[f"{key}_frame"] = float(times[fi])
            writer.writerow(row)


def self_test() -> None:
    # Rotation shortest-angle sanity check.
    ident = np.eye(4)[None, None]
    rot250 = np_mat(Matrix.Rotation(math.radians(250), 4, "Y"))[None, None]
    e = error_arrays(rot250, ident)
    angle = float(e["rotation_deg"][0, 0])
    require(abs(angle - 110.0) < 1e-3, f"Shortest rotation angle self-test failed: {angle}")

    # Shear rejection sanity check.
    shear = Matrix.Identity(4)
    shear[0][1] = 0.3
    rejected = False
    try:
        decompose_checked(shear, 1e-5, "self-test shear")
    except RuntimeError:
        rejected = True
    require(rejected, "Shear rejection self-test failed.")

    log(f"Self-test passed. bpy={bpy.app.version_string}")


def execute(args) -> None:
    input_blend = Path(args.input_blend).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_blend = (
        Path(args.output_blend).expanduser().resolve(strict=False)
        if args.output_blend
        else (output_dir / "Winter_Samurai_anim_rest_rebased.blend").resolve(strict=False)
    )

    require(input_blend.is_file(), f"Missing input blend: {input_blend}")
    require(output_blend != input_blend, "Refusing to overwrite input blend.")
    require(output_blend.suffix.lower() == ".blend", "Output must be a .blend file.")

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "rebase_report.json"
    cache_path = output_dir / "source_world_cache.npz"
    csv_path = output_dir / "world_pose_error_per_bone.csv"

    for p in [output_blend, report_path, cache_path, csv_path]:
        require(not p.exists(), f"Output already exists; use a fresh output directory: {p}")

    report: dict[str, Any] = {
        "schema": "ogd_lbs_winter_samurai_rest_rebase_03a_v2_no_convert_local_to_pose",
        "created_at_utc": utc_now_iso(),
        "bpy_version": bpy.app.version_string,
        "input_blend": str(input_blend),
        "output_blend": str(output_blend),
        "output_dir": str(output_dir),
        "status": "running",
        "stage": "open",
    }

    try:
        open_blend(input_blend)

        skin, anim, bone_names, role_evidence = select_roles()
        skin_name = skin.name
        anim_name = anim.name

        scene = bpy.context.scene
        start = int(args.frame_start) if args.frame_start is not None else int(scene.frame_start)
        end = int(args.frame_end) if args.frame_end is not None else int(scene.frame_end)

        require(end >= start, f"Invalid frame range: {start}..{end}")
        require(args.samples_per_frame >= 1, "--samples_per_frame must be >= 1")

        sample_count = (end - start) * int(args.samples_per_frame) + 1
        require(sample_count <= 100000, f"Too many samples: {sample_count}")

        times = start + np.arange(sample_count, dtype=np.float64) / float(args.samples_per_frame)

        if args.reference_frame is None:
            args.reference_frame = start

        log("=" * 96)
        log("03a rest rebase begin")
        log(f"input_blend      : {input_blend}")
        log(f"output_blend     : {output_blend}")
        log(f"skinning armature: {skin_name}")
        log(f"animation armature: {anim_name}")
        log(f"frame range      : {start}..{end}")
        log(f"samples          : {sample_count}")
        log("=" * 96)

        reference_rest = rest_array(skin, bone_names)
        reference_parents = parent_map(skin, bone_names)
        original_anim_rest = rest_array(anim, bone_names)
        original_anim_parents = parent_map(anim, bone_names)

        report["selected_armatures"] = {"skinning": skin_name, "animation": anim_name}
        report["role_evidence"] = role_evidence
        report["sampling"] = {
            "frame_start": start,
            "frame_end": end,
            "samples_per_frame": int(args.samples_per_frame),
            "sample_count": int(sample_count),
            "reference_frame": int(args.reference_frame),
        }
        report["changed_parents"] = [
            {"bone": name, "old": original_anim_parents[name], "new": reference_parents[name]}
            for name in bone_names
            if original_anim_parents.get(name) != reference_parents.get(name)
        ]
        report["tolerances"] = {
            "position_tol": float(args.position_tol),
            "rotation_tol_deg": float(args.rotation_tol_deg),
            "linear_tol": float(args.linear_tol),
            "trs_tol": float(args.trs_tol),
        }

        report["stage"] = "capture_original"
        baseline_world, baseline_objects = capture_world(
            [skin_name, anim_name],
            bone_names,
            times,
            "Capture ORIGINAL",
        )

        frame_set(float(args.reference_frame))
        dg = bpy.context.evaluated_depsgraph_get()
        reference_world = np_mat(skin.evaluated_get(dg).matrix_world)
        assert_affine(reference_world, "reference_world")

        np.savez_compressed(
            cache_path,
            bone_names=np.asarray(bone_names),
            times=times,
            skinning_world=baseline_world[skin_name],
            animation_world=baseline_world[anim_name],
            skinning_object_world=baseline_objects[skin_name],
            animation_object_world=baseline_objects[anim_name],
            target_object_world=reference_world,
            target_rest=reference_rest,
            original_animation_rest=original_anim_rest,
        )

        log(f"Cached source world matrices: {cache_path}")

        report["stage"] = "prepare_target"
        report["changes"] = prepare_target(skin, anim, reference_world, args)

        report["stage"] = "bake"
        report["bake"] = bake_world_preserved_action(
            anim,
            bone_names,
            times,
            baseline_world[anim_name],
            reference_world,
            args,
        )

        scene.frame_start = start
        scene.frame_end = end

        report["stage"] = "validate_before_save"
        before_save, _errors = audit(
            skin_name,
            anim_name,
            bone_names,
            times,
            baseline_world,
            reference_world,
            reference_rest,
            reference_parents,
            args,
            "Validate BEFORE SAVE",
        )
        report["validation_before_save"] = before_save
        dump_json(report_path, report)
        require(before_save["passed"], f"Pre-save validation failed. See {report_path}")

        anim["ws_03a_rebased"] = True
        anim["ws_03a_source_blend"] = str(input_blend)
        anim["ws_03a_policy"] = "preserve sampled evaluated bone world matrices"

        frame_set(float(start))

        report["stage"] = "save"
        output_blend.parent.mkdir(parents=True, exist_ok=True)
        result = bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), check_existing=False)
        require(result == {"FINISHED"}, f"save_as_mainfile failed: {result}")

        report["stage"] = "validate_reloaded"
        open_blend(output_blend)

        after_reload, anim_errors = audit(
            skin_name,
            anim_name,
            bone_names,
            times,
            baseline_world,
            reference_world,
            reference_rest,
            reference_parents,
            args,
            "Validate AFTER RELOAD",
        )
        report["validation_after_reload"] = after_reload

        write_error_csv(csv_path, anim_errors, times, bone_names)
        require(after_reload["passed"], f"Reload validation failed. See {report_path}")

        report["status"] = "passed"
        report["stage"] = "complete"
        report["source_world_cache"] = str(cache_path)
        report["world_pose_error_csv"] = str(csv_path)
        report["scope"] = (
            "This validates sampled animation-armature bone world matrices, "
            "destination rest pose, hierarchy, object transform, save/reload stability, "
            "and skinning-armature unchangedness. It does not retarget the skinning mesh."
        )

        dump_json(report_path, report)

        log("=" * 96)
        log("03a rest rebase PASSED")
        log(f"output_blend : {output_blend}")
        log(f"report_json  : {report_path}")
        log(f"error_csv    : {csv_path}")
        log("=" * 96)

    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        dump_json(report_path, report)
        log(f"FAILED at stage={report.get('stage')}: {exc}")
        log(f"Failure report: {report_path}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Winter Samurai 03a rest-pose rebase without Bone.convert_local_to_pose.")

    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--input_blend", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_blend", type=str, default=None)

    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--reference_frame", type=int, default=None)
    parser.add_argument("--samples_per_frame", type=int, default=1)

    parser.add_argument("--position_tol", type=float, default=1e-5)
    parser.add_argument("--rotation_tol_deg", type=float, default=0.05)
    parser.add_argument("--linear_tol", type=float, default=1e-5)
    parser.add_argument("--trs_tol", type=float, default=1e-5)

    parser.add_argument("--action_name", type=str, default="WS_Attack1_03a_WorldPreserved")

    args = parser.parse_args(script_argv())

    self_test()

    if args.self_test:
        return

    if not args.input_blend or not args.output_dir:
        parser.error("--input_blend and --output_dir are required unless --self_test is used.")

    for key in ["position_tol", "rotation_tol_deg", "linear_tol", "trs_tol"]:
        value = float(getattr(args, key))
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{key} must be finite and positive.")

    execute(args)


if __name__ == "__main__":
    main()
