#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Winter Samurai FBX clean Step 04: detach Weapon root + virtual mount rebake.

Supports mount types:
    direct:
        offset = inverse(MountBone_world_ref) @ Weapon_world_ref

    mirror:
        mirrored_offset = Reflect @ source_offset @ Reflect
        Then:
            Weapon_world = MirrorMountBone_world @ mirrored_offset

The mirror mode is a cleaning-time computation only. It does not apply
negative scale to any object or bone in the output Blender file.

This step:
    - detaches Weapon from its rest parent if configured
    - makes Weapon a root bone
    - bakes selected Actions by virtual mount policies
    - optionally preserves unconfigured Actions by rebaking original Weapon world motion
    - validates before save and after reload
"""

import argparse
import csv
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector


TAG = "[ws-weapon-mount-04]"


def log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{TAG}[WARN] {msg}", flush=True)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root() -> Path | None:
    this_file = Path(__file__).resolve(strict=False)
    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return None


def json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Matrix):
        return [[float(obj[r][c]) for c in range(4)] for r in range(4)]
    if isinstance(obj, Vector):
        return [float(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, set):
        return sorted(json_safe(x) for x in obj)
    return obj


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(obj), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON config must be object: {path}")
    return obj


def np_mat(m: Matrix) -> np.ndarray:
    return np.asarray(m, dtype=np.float64).copy()


def bl_mat(a: np.ndarray) -> Matrix:
    return Matrix(np.asarray(a, dtype=np.float64).tolist())


def matrix_to_list(mat: Matrix) -> list[list[float]]:
    return [[float(mat[r][c]) for c in range(4)] for r in range(4)]


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
    update_view_layer()


def save_blend(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Saving blend: {path}")
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    require(result == {"FINISHED"}, f"save_as_mainfile failed: {result}")


def get_armature(name: str | None = None) -> bpy.types.Object:
    if name:
        obj = bpy.data.objects.get(name)
        require(obj is not None and obj.type == "ARMATURE", f"Armature not found: {name}")
        return obj

    arms = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    require(len(arms) == 1, f"Expected exactly one armature, got {[a.name for a in arms]}")
    return arms[0]


def get_action(name: str) -> bpy.types.Action:
    action = bpy.data.actions.get(name)
    require(action is not None, f"Action not found: {name}")
    return action


def get_pose_bone(arm: bpy.types.Object, bone_name: str) -> bpy.types.PoseBone:
    pb = arm.pose.bones.get(bone_name)
    require(pb is not None, f"Pose bone not found: {bone_name}")
    return pb


def set_armature_action(arm: bpy.types.Object, action: bpy.types.Action) -> None:
    arm.animation_data_create()
    arm.animation_data.action = action
    arm.data.pose_position = "POSE"
    update_view_layer()


def pose_bone_world_matrix(arm: bpy.types.Object, bone_name: str) -> Matrix:
    pb = get_pose_bone(arm, bone_name)
    return arm.matrix_world @ pb.matrix


def action_frame_range(action: bpy.types.Action) -> tuple[int, int]:
    a, b = action.frame_range
    return int(math.floor(float(a))), int(math.ceil(float(b)))


def resolve_action_frames(
    action: bpy.types.Action,
    *,
    frame_start: int | None,
    frame_end: int | None,
) -> list[int]:
    a, b = action_frame_range(action)
    start = int(frame_start) if frame_start is not None else a
    end = int(frame_end) if frame_end is not None else b
    require(end >= start, f"Invalid frame range for {action.name}: {start}..{end}")
    return list(range(start, end + 1))


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
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite matrix comparison.")

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


def summarize_error(errors: dict[str, np.ndarray], frames: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key, arr in errors.items():
        arr = np.asarray(arr)
        idx = int(np.argmax(arr))
        out[f"{key}_max"] = float(np.max(arr))
        out[f"{key}_mean"] = float(np.mean(arr))
        out[f"{key}_worst_frame"] = int(frames[idx]) if frames else None

    return out


def remove_bone_fcurves(action: bpy.types.Action, bone_name: str) -> dict[str, Any]:
    removed = []
    kept = 0

    prefix_dq = f'pose.bones["{bone_name}"].'
    prefix_sq = f"pose.bones['{bone_name}']."

    try:
        fcurves = list(action.fcurves)
    except Exception:
        fcurves = []

    for fc in fcurves:
        data_path = str(fc.data_path)
        if data_path.startswith(prefix_dq) or data_path.startswith(prefix_sq):
            removed.append(
                {
                    "data_path": data_path,
                    "array_index": int(fc.array_index),
                    "keyframe_count": int(len(fc.keyframe_points)),
                }
            )
            action.fcurves.remove(fc)
        else:
            kept += 1

    return {
        "action": action.name,
        "bone": bone_name,
        "removed_fcurve_count": int(len(removed)),
        "kept_fcurve_count": int(kept),
        "removed_first_50": removed[:50],
    }


def weapon_fcurves(action: bpy.types.Action, bone_name: str) -> list[Any]:
    prefix_dq = f'pose.bones["{bone_name}"].'
    prefix_sq = f"pose.bones['{bone_name}']."

    try:
        fcurves = list(action.fcurves)
    except Exception:
        return []

    out = []
    for fc in fcurves:
        data_path = str(fc.data_path)
        if data_path.startswith(prefix_dq) or data_path.startswith(prefix_sq):
            out.append(fc)
    return out


def set_weapon_fcurve_interpolation(action: bpy.types.Action, bone_name: str) -> dict[str, Any]:
    curves = weapon_fcurves(action, bone_name)

    for fc in curves:
        try:
            fc.extrapolation = "CONSTANT"
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            fc.update()
        except Exception:
            pass

    return {
        "action": action.name,
        "bone": bone_name,
        "weapon_fcurve_count": int(len(curves)),
        "interpolation": "LINEAR",
        "extrapolation": "CONSTANT",
    }


def all_action_names() -> list[str]:
    return sorted([action.name for action in bpy.data.actions])


def enabled_action_policies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = config.get("action_policies")
    require(isinstance(raw, dict) and raw, "Config must contain non-empty object field: action_policies")

    out: dict[str, dict[str, Any]] = {}
    for action_name, policy in raw.items():
        require(isinstance(policy, dict), f"action_policies[{action_name!r}] must be object")
        if bool(policy.get("enabled", True)):
            out[str(action_name)] = dict(policy)

    require(out, "No enabled action_policies found.")
    return out


def mirror_matrix_from_config(cfg: dict[str, Any]) -> Matrix:
    if "mirror_signs" in cfg:
        signs = list(cfg["mirror_signs"])
        require(len(signs) == 3, "mirror_signs must have exactly 3 numbers.")
        sx, sy, sz = [float(x) for x in signs]
        require(abs(sx) == 1.0 and abs(sy) == 1.0 and abs(sz) == 1.0, "mirror_signs values must be +/-1.")
        return Matrix.Diagonal((sx, sy, sz, 1.0))

    axes = cfg.get("mirror_axes", ["X"])
    if isinstance(axes, str):
        axes = [axes]
    axes = {str(a).upper() for a in axes}

    sx = -1.0 if "X" in axes else 1.0
    sy = -1.0 if "Y" in axes else 1.0
    sz = -1.0 if "Z" in axes else 1.0

    return Matrix.Diagonal((sx, sy, sz, 1.0))


def apply_optional_translation(offset: Matrix, cfg: dict[str, Any]) -> Matrix:
    if "translation_offset" not in cfg:
        return offset

    raw = cfg["translation_offset"]
    require(isinstance(raw, list) and len(raw) == 3, "translation_offset must be [x, y, z].")
    out = offset.copy()
    out.translation = out.translation + Vector((float(raw[0]), float(raw[1]), float(raw[2])))
    return out


def compute_direct_mount_offset(
    *,
    arm: bpy.types.Object,
    mount_name: str,
    mount_cfg: dict[str, Any],
    weapon_bone: str,
) -> dict[str, Any]:
    mount_bone = str(mount_cfg["mount_bone"])
    ref_action_name = str(mount_cfg["reference_action"])
    ref_frame = int(mount_cfg["reference_frame"])

    action = get_action(ref_action_name)
    set_armature_action(arm, action)
    frame_set(ref_frame)

    mount_world = pose_bone_world_matrix(arm, mount_bone)
    weapon_world = pose_bone_world_matrix(arm, weapon_bone)
    offset = mount_world.inverted() @ weapon_world

    return {
        "mount_name": str(mount_name),
        "type": "direct",
        "mount_bone": mount_bone,
        "reference_action": ref_action_name,
        "reference_frame": ref_frame,
        "mount_world_reference": matrix_to_list(mount_world),
        "weapon_world_reference": matrix_to_list(weapon_world),
        "offset_mount_to_weapon": matrix_to_list(offset),
        "offset_np": np_mat(offset),
        "description": str(mount_cfg.get("description", "")),
    }


def compute_mirror_mount_offset(
    *,
    mount_name: str,
    mount_cfg: dict[str, Any],
    source_mount_info: dict[str, Any],
) -> dict[str, Any]:
    mount_bone = str(mount_cfg["mount_bone"])
    source_mount = str(mount_cfg["source_mount"])

    source_offset = bl_mat(np.asarray(source_mount_info["offset_np"], dtype=np.float64))
    reflect = mirror_matrix_from_config(mount_cfg)

    mirrored_offset = reflect @ source_offset @ reflect
    mirrored_offset = apply_optional_translation(mirrored_offset, mount_cfg)

    return {
        "mount_name": str(mount_name),
        "type": "mirror",
        "mount_bone": mount_bone,
        "source_mount": source_mount,
        "source_mount_bone": source_mount_info.get("mount_bone"),
        "mirror_axes": mount_cfg.get("mirror_axes", ["X"]),
        "mirror_signs": [float(reflect[0][0]), float(reflect[1][1]), float(reflect[2][2])],
        "mirror_formula": "mirrored_offset = Reflect @ source_offset @ Reflect",
        "source_offset_mount_to_weapon": source_mount_info.get("offset_mount_to_weapon"),
        "offset_mount_to_weapon": matrix_to_list(mirrored_offset),
        "offset_np": np_mat(mirrored_offset),
        "description": str(mount_cfg.get("description", "")),
    }


def compute_mount_offsets_before_detach(
    *,
    arm: bpy.types.Object,
    config: dict[str, Any],
    weapon_bone: str,
) -> dict[str, dict[str, Any]]:
    mounts = config.get("mounts")
    require(isinstance(mounts, dict) and mounts, "Config must contain non-empty object field: mounts")

    out: dict[str, dict[str, Any]] = {}
    pending = {str(k): dict(v) for k, v in mounts.items()}

    while pending:
        progressed = False

        for mount_name in list(pending.keys()):
            mount_cfg = pending[mount_name]
            typ = str(mount_cfg.get("type", "direct"))

            if typ == "direct":
                info = compute_direct_mount_offset(
                    arm=arm,
                    mount_name=mount_name,
                    mount_cfg=mount_cfg,
                    weapon_bone=weapon_bone,
                )
                out[mount_name] = info
                pending.pop(mount_name)
                progressed = True

                log(
                    f"Computed direct mount offset BEFORE detach: {mount_name} "
                    f"ref_action={info['reference_action']} frame={info['reference_frame']} "
                    f"mount_bone={info['mount_bone']} weapon_bone={weapon_bone}"
                )

            elif typ == "mirror":
                source_mount = str(mount_cfg["source_mount"])
                if source_mount not in out:
                    continue

                info = compute_mirror_mount_offset(
                    mount_name=mount_name,
                    mount_cfg=mount_cfg,
                    source_mount_info=out[source_mount],
                )
                out[mount_name] = info
                pending.pop(mount_name)
                progressed = True

                log(
                    f"Computed mirror mount offset: {mount_name} "
                    f"source_mount={source_mount} "
                    f"mount_bone={info['mount_bone']} "
                    f"mirror_signs={info['mirror_signs']}"
                )

            else:
                raise RuntimeError(f"Unsupported mount type for {mount_name}: {typ}")

        require(
            progressed,
            "Could not resolve all mounts. Possible missing or cyclic source_mount references. "
            f"Pending={sorted(pending.keys())}, resolved={sorted(out.keys())}"
        )

    return out


def capture_weapon_world_for_frames(
    *,
    arm: bpy.types.Object,
    action: bpy.types.Action,
    weapon_bone: str,
    frames: list[int],
) -> np.ndarray:
    set_armature_action(arm, action)

    arr = np.empty((len(frames), 4, 4), dtype=np.float64)
    for i, frame in enumerate(frames):
        frame_set(frame)
        arr[i] = np_mat(pose_bone_world_matrix(arm, weapon_bone))

    return arr


def capture_original_weapon_worlds(
    *,
    arm: bpy.types.Object,
    weapon_bone: str,
    target_specs: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    for spec in target_specs:
        action = get_action(spec["action"])
        frames = list(spec["frames"])
        out[spec["action"]] = capture_weapon_world_for_frames(
            arm=arm,
            action=action,
            weapon_bone=weapon_bone,
            frames=frames,
        )
        log(f"Captured original Weapon world: action={spec['action']} frames={frames[0]}..{frames[-1]}")

    return out


def detach_weapon_bone_preserve_rest(
    *,
    arm: bpy.types.Object,
    weapon_bone: str,
) -> dict[str, Any]:
    ensure_object_mode()
    update_view_layer()

    bone = arm.data.bones.get(weapon_bone)
    require(bone is not None, f"Weapon bone not found in armature data: {weapon_bone}")

    before_parent = bone.parent.name if bone.parent is not None else None
    before_children = [child.name for child in bone.children]
    before_use_connect = bool(getattr(bone, "use_connect", False))
    before_matrix = bone.matrix_local.copy()
    before_length = float(bone.length)
    before_use_deform = bool(bone.use_deform)

    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm

    result = bpy.ops.object.mode_set(mode="EDIT")
    require(result == {"FINISHED"}, f"Failed to enter EDIT mode: {result}")

    try:
        eb = arm.data.edit_bones.get(weapon_bone)
        require(eb is not None, f"Edit bone not found: {weapon_bone}")

        loc, rot, _scale = before_matrix.decompose()
        length = max(before_length, 1e-6)

        eb.use_connect = False
        eb.parent = None

        eb.head = loc
        eb.tail = loc + (rot @ Vector((0.0, length, 0.0)))
        try:
            eb.align_roll(rot @ Vector((0.0, 0.0, 1.0)))
        except Exception as exc:
            warn(f"align_roll failed for {weapon_bone}: {exc}")

        eb.use_deform = bool(before_use_deform)

    finally:
        result = bpy.ops.object.mode_set(mode="OBJECT")
        require(result == {"FINISHED"}, f"Failed to return OBJECT mode: {result}")

    update_view_layer()

    after_bone = arm.data.bones.get(weapon_bone)
    require(after_bone is not None, f"Weapon bone disappeared after detach: {weapon_bone}")

    after_parent = after_bone.parent.name if after_bone.parent is not None else None
    after_children = [child.name for child in after_bone.children]
    after_matrix = after_bone.matrix_local.copy()

    rest_abs = float(np.max(np.abs(np_mat(after_matrix) - np_mat(before_matrix))))
    root_bones = sorted([bone.name for bone in arm.data.bones if bone.parent is None])

    return {
        "weapon_bone": weapon_bone,
        "before_parent": before_parent,
        "after_parent": after_parent,
        "before_children": before_children,
        "after_children": after_children,
        "before_use_connect": before_use_connect,
        "before_use_deform": before_use_deform,
        "after_use_deform": bool(after_bone.use_deform),
        "rest_matrix_abs_max_after_detach": rest_abs,
        "root_bones_after_detach": root_bones,
        "before_matrix_local": matrix_to_list(before_matrix),
        "after_matrix_local": matrix_to_list(after_matrix),
    }


def build_target_specs(
    *,
    config: dict[str, Any],
    preserve_unconfigured: bool,
) -> list[dict[str, Any]]:
    policies = enabled_action_policies(config)

    specs: list[dict[str, Any]] = []
    configured_names = set(policies.keys())

    for action_name, policy in policies.items():
        action = get_action(action_name)
        mode = str(policy.get("mode", "mount"))
        require(mode == "mount", f"Only policy mode='mount' is supported for configured actions, got {mode!r}")

        mount_name = str(policy["mount"])
        frame_start = policy.get("frame_start", None)
        frame_end = policy.get("frame_end", None)

        frames = resolve_action_frames(
            action,
            frame_start=int(frame_start) if frame_start is not None else None,
            frame_end=int(frame_end) if frame_end is not None else None,
        )

        specs.append(
            {
                "action": action_name,
                "mode": "mount",
                "mount": mount_name,
                "frames": frames,
                "policy": policy,
            }
        )

    if preserve_unconfigured:
        for action_name in all_action_names():
            if action_name in configured_names:
                continue

            action = get_action(action_name)
            frames = resolve_action_frames(action, frame_start=None, frame_end=None)

            specs.append(
                {
                    "action": action_name,
                    "mode": "preserve_original",
                    "mount": None,
                    "frames": frames,
                    "policy": {
                        "enabled": True,
                        "mode": "preserve_original",
                        "description": "Auto-preserved because Weapon bone is detached globally.",
                    },
                }
            )

    specs = sorted(specs, key=lambda x: (0 if x["mode"] == "mount" else 1, x["action"]))
    require(specs, "No action specs to bake.")
    return specs


def make_expected_world_for_spec(
    *,
    arm: bpy.types.Object,
    spec: dict[str, Any],
    mount_offsets: dict[str, dict[str, Any]],
    original_worlds: dict[str, np.ndarray],
) -> np.ndarray:
    action_name = str(spec["action"])
    mode = str(spec["mode"])
    frames = list(spec["frames"])

    if mode == "preserve_original":
        return np.asarray(original_worlds[action_name], dtype=np.float64).copy()

    require(mode == "mount", f"Unsupported bake mode: {mode}")

    mount_name = str(spec["mount"])
    require(mount_name in mount_offsets, f"Unknown mount {mount_name!r} for action {action_name}")

    mount_info = mount_offsets[mount_name]
    mount_bone = str(mount_info["mount_bone"])
    offset = bl_mat(np.asarray(mount_info["offset_np"], dtype=np.float64))

    action = get_action(action_name)
    set_armature_action(arm, action)

    expected = np.empty((len(frames), 4, 4), dtype=np.float64)

    for i, frame in enumerate(frames):
        frame_set(int(frame))
        mount_world = pose_bone_world_matrix(arm, mount_bone)
        expected[i] = np_mat(mount_world @ offset)

    return expected


def bake_weapon_to_expected_worlds(
    *,
    arm: bpy.types.Object,
    weapon_bone: str,
    spec: dict[str, Any],
    expected_world: np.ndarray,
    original_world: np.ndarray,
    position_tol: float,
    rotation_tol_deg: float,
    linear_tol: float,
) -> dict[str, Any]:
    action_name = str(spec["action"])
    action = get_action(action_name)
    frames = list(spec["frames"])

    require(expected_world.shape == (len(frames), 4, 4), f"Bad expected_world shape for {action_name}")
    require(original_world.shape == (len(frames), 4, 4), f"Bad original_world shape for {action_name}")

    set_armature_action(arm, action)

    remove_summary = remove_bone_fcurves(action, weapon_bone)

    pb = get_pose_bone(arm, weapon_bone)
    pb.rotation_mode = "QUATERNION"

    previous_quat = None

    for i, frame in enumerate(frames):
        frame_set(int(frame))

        target_world = bl_mat(expected_world[i])
        desired_pose = arm.matrix_world.inverted() @ target_world

        pb.rotation_mode = "QUATERNION"
        pb.matrix = desired_pose
        update_view_layer()

        q = pb.rotation_quaternion.copy()
        if previous_quat is not None and q.dot(previous_quat) < 0:
            q.negate()
            pb.rotation_quaternion = q
            update_view_layer()
        previous_quat = q.copy()

        for channel in ("location", "rotation_quaternion", "scale"):
            ok = pb.keyframe_insert(data_path=channel, frame=float(frame), group=weapon_bone)
            require(ok, f"Failed keyframe insert: action={action_name}, frame={frame}, channel={channel}")

    interp_summary = set_weapon_fcurve_interpolation(action, weapon_bone)

    actual_world = capture_weapon_world_for_frames(
        arm=arm,
        action=action,
        weapon_bone=weapon_bone,
        frames=frames,
    )

    errors = error_arrays(actual_world, expected_world)
    error_summary = summarize_error(errors, frames)

    original_delta = error_arrays(original_world, expected_world)
    original_delta_summary = summarize_error(original_delta, frames)

    passed = bool(
        error_summary["translation_l2_max"] <= position_tol
        and error_summary["rotation_deg_max"] <= rotation_tol_deg
        and error_summary["linear_relative_max"] <= linear_tol
    )

    return {
        "action": action_name,
        "mode": spec["mode"],
        "mount": spec.get("mount"),
        "weapon_bone": weapon_bone,
        "frame_start": int(frames[0]),
        "frame_end": int(frames[-1]),
        "frame_count": int(len(frames)),
        "policy": spec.get("policy", {}),
        "remove_old_weapon_fcurves": remove_summary,
        "weapon_fcurves_after_bake": interp_summary,
        "validation": {
            "passed": passed,
            "errors": error_summary,
            "tolerances": {
                "position_tol": float(position_tol),
                "rotation_tol_deg": float(rotation_tol_deg),
                "linear_tol": float(linear_tol),
            },
        },
        "original_to_fixed_delta": original_delta_summary,
        "_frames": frames,
        "_expected_world": expected_world,
        "_actual_world": actual_world,
        "_original_world": original_world,
    }


def validate_after_reload(
    *,
    arm_name: str,
    action_name: str,
    weapon_bone: str,
    frames: list[int],
    expected_world: np.ndarray,
    position_tol: float,
    rotation_tol_deg: float,
    linear_tol: float,
) -> dict[str, Any]:
    arm = get_armature(arm_name)
    action = get_action(action_name)

    actual_world = capture_weapon_world_for_frames(
        arm=arm,
        action=action,
        weapon_bone=weapon_bone,
        frames=frames,
    )

    errors = error_arrays(actual_world, expected_world)
    error_summary = summarize_error(errors, frames)

    passed = bool(
        error_summary["translation_l2_max"] <= position_tol
        and error_summary["rotation_deg_max"] <= rotation_tol_deg
        and error_summary["linear_relative_max"] <= linear_tol
    )

    return {
        "action": action_name,
        "passed": passed,
        "errors": error_summary,
    }


def write_frame_csv(path: Path, *, action_results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "action",
        "mode",
        "mount",
        "frame",
        "translation_l2",
        "rotation_deg",
        "linear_relative",
        "matrix_abs",
        "original_to_fixed_translation_l2",
        "original_to_fixed_rotation_deg",
        "original_to_fixed_linear_relative",
        "original_to_fixed_matrix_abs",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for result in action_results:
            action = result["action"]
            mode = result["mode"]
            mount = result.get("mount")
            frames = result["_frames"]
            expected = result["_expected_world"]
            actual = result["_actual_world"]
            original = result["_original_world"]

            errors = error_arrays(actual, expected)
            delta = error_arrays(original, expected)

            for i, frame in enumerate(frames):
                writer.writerow(
                    {
                        "action": action,
                        "mode": mode,
                        "mount": mount,
                        "frame": int(frame),
                        "translation_l2": float(errors["translation_l2"][i]),
                        "rotation_deg": float(errors["rotation_deg"][i]),
                        "linear_relative": float(errors["linear_relative"][i]),
                        "matrix_abs": float(errors["matrix_abs"][i]),
                        "original_to_fixed_translation_l2": float(delta["translation_l2"][i]),
                        "original_to_fixed_rotation_deg": float(delta["rotation_deg"][i]),
                        "original_to_fixed_linear_relative": float(delta["linear_relative"][i]),
                        "original_to_fixed_matrix_abs": float(delta["matrix_abs"][i]),
                    }
                )


def scene_summary(arm: bpy.types.Object | None = None, weapon_bone: str | None = None) -> dict[str, Any]:
    arms = sorted([obj.name for obj in bpy.data.objects if obj.type == "ARMATURE"])
    meshes = sorted([obj.name for obj in bpy.data.objects if obj.type == "MESH"])
    actions = sorted([a.name for a in bpy.data.actions])
    empties = sorted([obj.name for obj in bpy.data.objects if obj.type == "EMPTY"])

    out = {
        "armature_count": int(len(arms)),
        "mesh_count": int(len(meshes)),
        "action_count": int(len(actions)),
        "empty_count": int(len(empties)),
        "armatures": arms,
        "actions": actions,
        "meshes_first_100": meshes[:100],
        "empties_first_100": empties[:100],
    }

    if arm is not None:
        out["root_bones"] = sorted([bone.name for bone in arm.data.bones if bone.parent is None])

        if weapon_bone:
            bone = arm.data.bones.get(weapon_bone)
            out["weapon_bone"] = {
                "name": weapon_bone,
                "exists": bone is not None,
                "parent": bone.parent.name if bone is not None and bone.parent is not None else None,
                "children": [child.name for child in bone.children] if bone is not None else [],
                "use_deform": bool(bone.use_deform) if bone is not None else None,
            }

    return out


def strip_internal_fields(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def self_test() -> None:
    ident = np.eye(4)[None]
    rot250 = np_mat(Matrix.Rotation(math.radians(250), 4, "Y"))[None]
    e = error_arrays(rot250, ident)
    angle = float(e["rotation_deg"][0])
    require(abs(angle - 110.0) < 1e-3, f"Shortest rotation self-test failed: {angle}")

    reflect = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))
    rot = Matrix.Rotation(math.radians(30), 4, "Z")
    mirrored = reflect @ rot @ reflect
    det = np.linalg.det(np.asarray(mirrored, dtype=np.float64)[:3, :3])
    require(abs(det - 1.0) < 1e-6, f"Mirror sandwich produced improper rotation determinant: {det}")

    log(f"Self-test passed. bpy={bpy.app.version_string}")
    log(f"Repo root: {resolve_repo_root()}")


def execute(args: argparse.Namespace) -> None:
    input_blend = Path(args.input_blend).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_blend = (
        Path(args.output_blend).expanduser().resolve(strict=False)
        if args.output_blend
        else (output_dir / "Winter_Samurai_clean04_weapon_mount_fixed.blend").resolve(strict=False)
    )
    config_path = Path(args.config).expanduser().resolve(strict=False)

    report_path = output_dir / "weapon_mount_fix_report.json"
    csv_path = output_dir / "weapon_mount_fix_frame_errors.csv"

    require(input_blend.is_file(), f"Missing input_blend: {input_blend}")
    require(config_path.is_file(), f"Missing config: {config_path}")
    require(output_blend != input_blend, "Refusing to overwrite input blend.")

    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.overwrite):
        for p in [output_blend, report_path, csv_path]:
            if p.exists():
                p.unlink()
    else:
        for p in [output_blend, report_path, csv_path]:
            require(not p.exists(), f"Output already exists; use --overwrite or fresh output_dir: {p}")

    config = load_json(config_path)

    report: dict[str, Any] = {
        "schema": "ogd_lbs_winter_samurai_weapon_mount_fix_report_v3_mirror_support",
        "created_at_utc": utc_now_iso(),
        "repo_root": str(resolve_repo_root()),
        "bpy_version": bpy.app.version_string,
        "input_blend": str(input_blend),
        "output_dir": str(output_dir),
        "output_blend": str(output_blend),
        "config_path": str(config_path),
        "report_json": str(report_path),
        "frame_error_csv": str(csv_path),
        "status": "running",
        "stage": "open",
        "tolerances": {
            "position_tol": float(args.position_tol),
            "rotation_tol_deg": float(args.rotation_tol_deg),
            "linear_tol": float(args.linear_tol),
        },
    }

    try:
        open_blend(input_blend)

        arm = get_armature(args.armature_name)
        arm_name = arm.name

        weapon_bone = str(config.get("weapon_bone", "Weapon"))
        weapon_mesh = str(config.get("weapon_mesh", "Weapon"))

        require(arm.pose.bones.get(weapon_bone) is not None, f"Weapon bone missing: {weapon_bone}")

        mesh_obj = bpy.data.objects.get(weapon_mesh)
        require(mesh_obj is not None and mesh_obj.type == "MESH", f"Weapon mesh missing or not mesh: {weapon_mesh}")

        detach_cfg = config.get("detach_weapon_bone", {})
        require(isinstance(detach_cfg, dict), "detach_weapon_bone must be an object if present.")
        detach_enabled = bool(detach_cfg.get("enabled", True))
        preserve_unconfigured = bool(detach_cfg.get("preserve_unconfigured_actions", True))

        log("=" * 96)
        log("Winter Samurai Weapon Mount Fix: detach + direct/mirror mount rebake")
        log(f"input_blend : {input_blend}")
        log(f"output_blend: {output_blend}")
        log(f"config      : {config_path}")
        log(f"armature    : {arm_name}")
        log(f"weapon_bone : {weapon_bone}")
        log(f"weapon_mesh : {weapon_mesh}")
        log(f"detach      : {detach_enabled}")
        log(f"preserve_unconfigured_actions: {preserve_unconfigured}")
        log("=" * 96)

        report["stage"] = "scene_initial"
        report["scene_initial"] = scene_summary(arm, weapon_bone)

        report["stage"] = "compute_mount_offsets_before_detach"
        mount_offsets = compute_mount_offsets_before_detach(
            arm=arm,
            config=config,
            weapon_bone=weapon_bone,
        )
        report["mounts"] = {
            name: {k: v for k, v in item.items() if k != "offset_np"}
            for name, item in mount_offsets.items()
        }

        report["stage"] = "build_target_specs"
        target_specs = build_target_specs(
            config=config,
            preserve_unconfigured=preserve_unconfigured,
        )
        report["target_specs"] = [
            {
                "action": item["action"],
                "mode": item["mode"],
                "mount": item.get("mount"),
                "frame_start": int(item["frames"][0]),
                "frame_end": int(item["frames"][-1]),
                "frame_count": int(len(item["frames"])),
                "policy": item.get("policy", {}),
            }
            for item in target_specs
        ]

        report["stage"] = "capture_original_weapon_worlds"
        original_worlds = capture_original_weapon_worlds(
            arm=arm,
            weapon_bone=weapon_bone,
            target_specs=target_specs,
        )

        report["stage"] = "detach_weapon_bone"
        if detach_enabled:
            detach_report = detach_weapon_bone_preserve_rest(
                arm=arm,
                weapon_bone=weapon_bone,
            )
        else:
            warn("detach_weapon_bone.enabled=false; skipping detach.")
            detach_report = {
                "skipped": True,
                "reason": "detach_weapon_bone.enabled=false",
            }

        report["detach_weapon_bone"] = detach_report
        report["scene_after_detach"] = scene_summary(arm, weapon_bone)

        report["stage"] = "bake_actions"
        internal_results: list[dict[str, Any]] = []
        action_results: list[dict[str, Any]] = []

        for spec in target_specs:
            action_name = spec["action"]
            log(
                f"Baking Weapon: action={action_name} mode={spec['mode']} "
                f"mount={spec.get('mount')} frames={spec['frames'][0]}..{spec['frames'][-1]}"
            )

            expected_world = make_expected_world_for_spec(
                arm=arm,
                spec=spec,
                mount_offsets=mount_offsets,
                original_worlds=original_worlds,
            )

            result = bake_weapon_to_expected_worlds(
                arm=arm,
                weapon_bone=weapon_bone,
                spec=spec,
                expected_world=expected_world,
                original_world=original_worlds[action_name],
                position_tol=float(args.position_tol),
                rotation_tol_deg=float(args.rotation_tol_deg),
                linear_tol=float(args.linear_tol),
            )

            internal_results.append(result)
            action_results.append(strip_internal_fields(result))

        report["action_results_before_save"] = action_results
        write_frame_csv(csv_path, action_results=internal_results)

        failed = [
            item for item in action_results
            if not item.get("validation", {}).get("passed", False)
        ]
        require(not failed, f"Weapon validation before save failed: {failed}")

        report["stage"] = "save"
        report["scene_before_save"] = scene_summary(arm, weapon_bone)
        dump_json(report_path, report)
        save_blend(output_blend)

        report["stage"] = "validate_after_reload"
        open_blend(output_blend)

        reload_validations = []
        for result in internal_results:
            reload_validation = validate_after_reload(
                arm_name=arm_name,
                action_name=result["action"],
                weapon_bone=weapon_bone,
                frames=result["_frames"],
                expected_world=result["_expected_world"],
                position_tol=float(args.position_tol),
                rotation_tol_deg=float(args.rotation_tol_deg),
                linear_tol=float(args.linear_tol),
            )
            reload_validations.append(reload_validation)

        failed_reload = [item for item in reload_validations if not item.get("passed", False)]
        require(not failed_reload, f"Weapon validation after reload failed: {failed_reload}")

        arm_reloaded = get_armature(arm_name)

        report["status"] = "passed"
        report["stage"] = "complete"
        report["scene_after_reload"] = scene_summary(arm_reloaded, weapon_bone)
        report["validation_after_reload"] = reload_validations
        report["scope"] = (
            "This step detaches Weapon from its rest parent and bakes Weapon bone "
            "world-space animation. Configured actions are driven by direct or mirrored "
            "virtual mount points. Unconfigured actions are preserved by rebaking their "
            "original Weapon world motion."
        )

        dump_json(report_path, report)

        log("=" * 96)
        log("Weapon mount detach+rebake PASSED")
        log(f"output_blend : {output_blend}")
        log(f"report_json  : {report_path}")
        log(f"frame_csv    : {csv_path}")
        log(f"root_bones   : {report['scene_after_reload'].get('root_bones')}")
        log(f"weapon_parent: {report['scene_after_reload'].get('weapon_bone', {}).get('parent')}")
        for item in reload_validations:
            log(f"  after reload: {item['action']} passed={item['passed']} errors={item['errors']}")
        log("=" * 96)

    except Exception as exc:
        report["status"] = "failed"
        report["stage_failed"] = report.get("stage")
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        try:
            dump_json(report_path, report)
            log(f"Failure report written: {report_path}")
        except Exception:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detach Winter Samurai Weapon bone and rebake via direct/mirrored virtual mounts."
    )

    parser.add_argument("--self_test", action="store_true")

    parser.add_argument("--input_blend", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_blend", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--armature_name", type=str, default=None)

    parser.add_argument("--position_tol", type=float, default=1e-4)
    parser.add_argument("--rotation_tol_deg", type=float, default=0.05)
    parser.add_argument("--linear_tol", type=float, default=1e-5)

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(script_argv())

    self_test()

    if args.self_test:
        return

    if not args.input_blend or not args.output_dir or not args.config:
        parser.error("--input_blend, --output_dir, and --config are required unless --self_test is used.")

    for key in ["position_tol", "rotation_tol_deg", "linear_tol"]:
        value = float(getattr(args, key))
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{key} must be finite and positive.")

    execute(args)


if __name__ == "__main__":
    main()
