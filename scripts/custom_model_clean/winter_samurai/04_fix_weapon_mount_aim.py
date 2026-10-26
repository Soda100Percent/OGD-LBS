#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Winter Samurai custom clean Step 04.

Single-file version of the final weapon mount + aim cleaner.

This file embeds the previously validated generic weapon-mount core used by
the old debug scripts, then applies the aim extension. The custom_model_clean
directory intentionally exposes only this aim-capable Step 04 entrypoint.
"""

import types

_EMBEDDED_WEAPON_MOUNT_CORE_SOURCE = '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nfrom __future__ import annotations\n\n"""\nWinter Samurai FBX clean Step 04: detach Weapon root + virtual mount rebake.\n\nSupports mount types:\n    direct:\n        offset = inverse(MountBone_world_ref) @ Weapon_world_ref\n\n    mirror:\n        mirrored_offset = Reflect @ source_offset @ Reflect\n        Then:\n            Weapon_world = MirrorMountBone_world @ mirrored_offset\n\nThe mirror mode is a cleaning-time computation only. It does not apply\nnegative scale to any object or bone in the output Blender file.\n\nThis step:\n    - detaches Weapon from its rest parent if configured\n    - makes Weapon a root bone\n    - bakes selected Actions by virtual mount policies\n    - optionally preserves unconfigured Actions by rebaking original Weapon world motion\n    - validates before save and after reload\n"""\n\nimport argparse\nimport csv\nimport json\nimport math\nimport sys\nimport traceback\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Any\n\nimport bpy\nimport numpy as np\nfrom mathutils import Matrix, Vector\n\n\nTAG = "[ws-weapon-mount-04]"\n\n\ndef log(msg: str) -> None:\n    print(f"{TAG} {msg}", flush=True)\n\n\ndef warn(msg: str) -> None:\n    print(f"{TAG}[WARN] {msg}", flush=True)\n\n\ndef require(cond: bool, msg: str) -> None:\n    if not cond:\n        raise RuntimeError(msg)\n\n\ndef script_argv() -> list[str]:\n    if "--" in sys.argv:\n        return sys.argv[sys.argv.index("--") + 1:]\n    return sys.argv[1:]\n\n\ndef utc_now_iso() -> str:\n    return datetime.now(timezone.utc).isoformat()\n\n\ndef resolve_repo_root() -> Path | None:\n    this_file = Path(__file__).resolve(strict=False)\n    for parent in [this_file.parent, *this_file.parents]:\n        if (parent / "pyproject.toml").exists() and (parent / "src").exists():\n            return parent\n    return None\n\n\ndef json_safe(obj: Any) -> Any:\n    if isinstance(obj, np.ndarray):\n        return obj.tolist()\n    if isinstance(obj, np.generic):\n        return obj.item()\n    if isinstance(obj, Path):\n        return str(obj)\n    if isinstance(obj, Matrix):\n        return [[float(obj[r][c]) for c in range(4)] for r in range(4)]\n    if isinstance(obj, Vector):\n        return [float(x) for x in obj]\n    if isinstance(obj, dict):\n        return {str(k): json_safe(v) for k, v in obj.items()}\n    if isinstance(obj, (list, tuple)):\n        return [json_safe(x) for x in obj]\n    if isinstance(obj, set):\n        return sorted(json_safe(x) for x in obj)\n    return obj\n\n\ndef dump_json(path: Path, obj: dict[str, Any]) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(\n        json.dumps(json_safe(obj), indent=2, ensure_ascii=False, allow_nan=False),\n        encoding="utf-8",\n    )\n\n\ndef load_json(path: Path) -> dict[str, Any]:\n    with path.open("r", encoding="utf-8") as f:\n        obj = json.load(f)\n    require(isinstance(obj, dict), f"JSON config must be object: {path}")\n    return obj\n\n\ndef np_mat(m: Matrix) -> np.ndarray:\n    return np.asarray(m, dtype=np.float64).copy()\n\n\ndef bl_mat(a: np.ndarray) -> Matrix:\n    return Matrix(np.asarray(a, dtype=np.float64).tolist())\n\n\ndef matrix_to_list(mat: Matrix) -> list[list[float]]:\n    return [[float(mat[r][c]) for c in range(4)] for r in range(4)]\n\n\ndef ensure_object_mode() -> None:\n    try:\n        active = bpy.context.view_layer.objects.active\n        if active is not None and active.mode != "OBJECT":\n            bpy.ops.object.mode_set(mode="OBJECT")\n    except Exception:\n        pass\n\n\ndef update_view_layer() -> None:\n    try:\n        bpy.context.view_layer.update()\n    except Exception:\n        pass\n    try:\n        bpy.context.evaluated_depsgraph_get().update()\n    except Exception:\n        pass\n\n\ndef frame_set(t: float) -> None:\n    base = math.floor(float(t))\n    sub = float(t) - base\n    bpy.context.scene.frame_set(base, subframe=sub)\n    update_view_layer()\n\n\ndef open_blend(path: Path) -> None:\n    require(path.is_file(), f"Input blend not found: {path}")\n    log(f"Opening blend: {path}")\n    result = bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)\n    require(result == {"FINISHED"}, f"open_mainfile failed: {result}")\n    ensure_object_mode()\n    update_view_layer()\n\n\ndef save_blend(path: Path) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    log(f"Saving blend: {path}")\n    result = bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)\n    require(result == {"FINISHED"}, f"save_as_mainfile failed: {result}")\n\n\ndef get_armature(name: str | None = None) -> bpy.types.Object:\n    if name:\n        obj = bpy.data.objects.get(name)\n        require(obj is not None and obj.type == "ARMATURE", f"Armature not found: {name}")\n        return obj\n\n    arms = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]\n    require(len(arms) == 1, f"Expected exactly one armature, got {[a.name for a in arms]}")\n    return arms[0]\n\n\ndef get_action(name: str) -> bpy.types.Action:\n    action = bpy.data.actions.get(name)\n    require(action is not None, f"Action not found: {name}")\n    return action\n\n\ndef get_pose_bone(arm: bpy.types.Object, bone_name: str) -> bpy.types.PoseBone:\n    pb = arm.pose.bones.get(bone_name)\n    require(pb is not None, f"Pose bone not found: {bone_name}")\n    return pb\n\n\ndef set_armature_action(arm: bpy.types.Object, action: bpy.types.Action) -> None:\n    arm.animation_data_create()\n    arm.animation_data.action = action\n    arm.data.pose_position = "POSE"\n    update_view_layer()\n\n\ndef pose_bone_world_matrix(arm: bpy.types.Object, bone_name: str) -> Matrix:\n    pb = get_pose_bone(arm, bone_name)\n    return arm.matrix_world @ pb.matrix\n\n\ndef action_frame_range(action: bpy.types.Action) -> tuple[int, int]:\n    a, b = action.frame_range\n    return int(math.floor(float(a))), int(math.ceil(float(b)))\n\n\ndef resolve_action_frames(\n    action: bpy.types.Action,\n    *,\n    frame_start: int | None,\n    frame_end: int | None,\n) -> list[int]:\n    a, b = action_frame_range(action)\n    start = int(frame_start) if frame_start is not None else a\n    end = int(frame_end) if frame_end is not None else b\n    require(end >= start, f"Invalid frame range for {action.name}: {start}..{end}")\n    return list(range(start, end + 1))\n\n\ndef closest_rotation(linear: np.ndarray) -> np.ndarray:\n    u, _s, vt = np.linalg.svd(linear)\n    r = u @ vt\n    det = np.linalg.det(r)\n\n    if np.any(det < 0):\n        u = u.copy()\n        if r.ndim == 2:\n            u[:, -1] *= -1\n        else:\n            mask = det < 0\n            u[mask, :, -1] *= -1\n        r = u @ vt\n\n    return r\n\n\ndef error_arrays(actual: np.ndarray, expected: np.ndarray) -> dict[str, np.ndarray]:\n    actual = np.asarray(actual, dtype=np.float64)\n    expected = np.asarray(expected, dtype=np.float64)\n\n    require(actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}")\n    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite matrix comparison.")\n\n    trans = np.linalg.norm(actual[..., :3, 3] - expected[..., :3, 3], axis=-1)\n\n    lin_rel = np.linalg.norm(\n        actual[..., :3, :3] - expected[..., :3, :3],\n        axis=(-2, -1),\n    ) / np.maximum(np.linalg.norm(expected[..., :3, :3], axis=(-2, -1)), 1e-12)\n\n    ra = closest_rotation(actual[..., :3, :3])\n    rb = closest_rotation(expected[..., :3, :3])\n    delta = np.swapaxes(ra, -1, -2) @ rb\n\n    tr = np.trace(delta, axis1=-2, axis2=-1)\n    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)\n\n    skew = np.stack(\n        [\n            delta[..., 2, 1] - delta[..., 1, 2],\n            delta[..., 0, 2] - delta[..., 2, 0],\n            delta[..., 1, 0] - delta[..., 0, 1],\n        ],\n        axis=-1,\n    )\n    sin = 0.5 * np.linalg.norm(skew, axis=-1)\n    rot = np.degrees(np.arctan2(sin, cos))\n\n    mat_abs = np.max(np.abs(actual - expected), axis=(-2, -1))\n\n    return {\n        "translation_l2": trans,\n        "rotation_deg": rot,\n        "linear_relative": lin_rel,\n        "matrix_abs": mat_abs,\n    }\n\n\ndef summarize_error(errors: dict[str, np.ndarray], frames: list[int]) -> dict[str, Any]:\n    out: dict[str, Any] = {}\n\n    for key, arr in errors.items():\n        arr = np.asarray(arr)\n        idx = int(np.argmax(arr))\n        out[f"{key}_max"] = float(np.max(arr))\n        out[f"{key}_mean"] = float(np.mean(arr))\n        out[f"{key}_worst_frame"] = int(frames[idx]) if frames else None\n\n    return out\n\n\ndef remove_bone_fcurves(action: bpy.types.Action, bone_name: str) -> dict[str, Any]:\n    removed = []\n    kept = 0\n\n    prefix_dq = f\'pose.bones["{bone_name}"].\'\n    prefix_sq = f"pose.bones[\'{bone_name}\']."\n\n    try:\n        fcurves = list(action.fcurves)\n    except Exception:\n        fcurves = []\n\n    for fc in fcurves:\n        data_path = str(fc.data_path)\n        if data_path.startswith(prefix_dq) or data_path.startswith(prefix_sq):\n            removed.append(\n                {\n                    "data_path": data_path,\n                    "array_index": int(fc.array_index),\n                    "keyframe_count": int(len(fc.keyframe_points)),\n                }\n            )\n            action.fcurves.remove(fc)\n        else:\n            kept += 1\n\n    return {\n        "action": action.name,\n        "bone": bone_name,\n        "removed_fcurve_count": int(len(removed)),\n        "kept_fcurve_count": int(kept),\n        "removed_first_50": removed[:50],\n    }\n\n\ndef weapon_fcurves(action: bpy.types.Action, bone_name: str) -> list[Any]:\n    prefix_dq = f\'pose.bones["{bone_name}"].\'\n    prefix_sq = f"pose.bones[\'{bone_name}\']."\n\n    try:\n        fcurves = list(action.fcurves)\n    except Exception:\n        return []\n\n    out = []\n    for fc in fcurves:\n        data_path = str(fc.data_path)\n        if data_path.startswith(prefix_dq) or data_path.startswith(prefix_sq):\n            out.append(fc)\n    return out\n\n\ndef set_weapon_fcurve_interpolation(action: bpy.types.Action, bone_name: str) -> dict[str, Any]:\n    curves = weapon_fcurves(action, bone_name)\n\n    for fc in curves:\n        try:\n            fc.extrapolation = "CONSTANT"\n            for kp in fc.keyframe_points:\n                kp.interpolation = "LINEAR"\n            fc.update()\n        except Exception:\n            pass\n\n    return {\n        "action": action.name,\n        "bone": bone_name,\n        "weapon_fcurve_count": int(len(curves)),\n        "interpolation": "LINEAR",\n        "extrapolation": "CONSTANT",\n    }\n\n\ndef all_action_names() -> list[str]:\n    return sorted([action.name for action in bpy.data.actions])\n\n\ndef enabled_action_policies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:\n    raw = config.get("action_policies")\n    require(isinstance(raw, dict) and raw, "Config must contain non-empty object field: action_policies")\n\n    out: dict[str, dict[str, Any]] = {}\n    for action_name, policy in raw.items():\n        require(isinstance(policy, dict), f"action_policies[{action_name!r}] must be object")\n        if bool(policy.get("enabled", True)):\n            out[str(action_name)] = dict(policy)\n\n    require(out, "No enabled action_policies found.")\n    return out\n\n\ndef mirror_matrix_from_config(cfg: dict[str, Any]) -> Matrix:\n    if "mirror_signs" in cfg:\n        signs = list(cfg["mirror_signs"])\n        require(len(signs) == 3, "mirror_signs must have exactly 3 numbers.")\n        sx, sy, sz = [float(x) for x in signs]\n        require(abs(sx) == 1.0 and abs(sy) == 1.0 and abs(sz) == 1.0, "mirror_signs values must be +/-1.")\n        return Matrix.Diagonal((sx, sy, sz, 1.0))\n\n    axes = cfg.get("mirror_axes", ["X"])\n    if isinstance(axes, str):\n        axes = [axes]\n    axes = {str(a).upper() for a in axes}\n\n    sx = -1.0 if "X" in axes else 1.0\n    sy = -1.0 if "Y" in axes else 1.0\n    sz = -1.0 if "Z" in axes else 1.0\n\n    return Matrix.Diagonal((sx, sy, sz, 1.0))\n\n\ndef apply_optional_translation(offset: Matrix, cfg: dict[str, Any]) -> Matrix:\n    if "translation_offset" not in cfg:\n        return offset\n\n    raw = cfg["translation_offset"]\n    require(isinstance(raw, list) and len(raw) == 3, "translation_offset must be [x, y, z].")\n    out = offset.copy()\n    out.translation = out.translation + Vector((float(raw[0]), float(raw[1]), float(raw[2])))\n    return out\n\n\ndef compute_direct_mount_offset(\n    *,\n    arm: bpy.types.Object,\n    mount_name: str,\n    mount_cfg: dict[str, Any],\n    weapon_bone: str,\n) -> dict[str, Any]:\n    mount_bone = str(mount_cfg["mount_bone"])\n    ref_action_name = str(mount_cfg["reference_action"])\n    ref_frame = int(mount_cfg["reference_frame"])\n\n    action = get_action(ref_action_name)\n    set_armature_action(arm, action)\n    frame_set(ref_frame)\n\n    mount_world = pose_bone_world_matrix(arm, mount_bone)\n    weapon_world = pose_bone_world_matrix(arm, weapon_bone)\n    offset = mount_world.inverted() @ weapon_world\n\n    return {\n        "mount_name": str(mount_name),\n        "type": "direct",\n        "mount_bone": mount_bone,\n        "reference_action": ref_action_name,\n        "reference_frame": ref_frame,\n        "mount_world_reference": matrix_to_list(mount_world),\n        "weapon_world_reference": matrix_to_list(weapon_world),\n        "offset_mount_to_weapon": matrix_to_list(offset),\n        "offset_np": np_mat(offset),\n        "description": str(mount_cfg.get("description", "")),\n    }\n\n\ndef compute_mirror_mount_offset(\n    *,\n    mount_name: str,\n    mount_cfg: dict[str, Any],\n    source_mount_info: dict[str, Any],\n) -> dict[str, Any]:\n    mount_bone = str(mount_cfg["mount_bone"])\n    source_mount = str(mount_cfg["source_mount"])\n\n    source_offset = bl_mat(np.asarray(source_mount_info["offset_np"], dtype=np.float64))\n    reflect = mirror_matrix_from_config(mount_cfg)\n\n    mirrored_offset = reflect @ source_offset @ reflect\n    mirrored_offset = apply_optional_translation(mirrored_offset, mount_cfg)\n\n    return {\n        "mount_name": str(mount_name),\n        "type": "mirror",\n        "mount_bone": mount_bone,\n        "source_mount": source_mount,\n        "source_mount_bone": source_mount_info.get("mount_bone"),\n        "mirror_axes": mount_cfg.get("mirror_axes", ["X"]),\n        "mirror_signs": [float(reflect[0][0]), float(reflect[1][1]), float(reflect[2][2])],\n        "mirror_formula": "mirrored_offset = Reflect @ source_offset @ Reflect",\n        "source_offset_mount_to_weapon": source_mount_info.get("offset_mount_to_weapon"),\n        "offset_mount_to_weapon": matrix_to_list(mirrored_offset),\n        "offset_np": np_mat(mirrored_offset),\n        "description": str(mount_cfg.get("description", "")),\n    }\n\n\ndef compute_mount_offsets_before_detach(\n    *,\n    arm: bpy.types.Object,\n    config: dict[str, Any],\n    weapon_bone: str,\n) -> dict[str, dict[str, Any]]:\n    mounts = config.get("mounts")\n    require(isinstance(mounts, dict) and mounts, "Config must contain non-empty object field: mounts")\n\n    out: dict[str, dict[str, Any]] = {}\n    pending = {str(k): dict(v) for k, v in mounts.items()}\n\n    while pending:\n        progressed = False\n\n        for mount_name in list(pending.keys()):\n            mount_cfg = pending[mount_name]\n            typ = str(mount_cfg.get("type", "direct"))\n\n            if typ == "direct":\n                info = compute_direct_mount_offset(\n                    arm=arm,\n                    mount_name=mount_name,\n                    mount_cfg=mount_cfg,\n                    weapon_bone=weapon_bone,\n                )\n                out[mount_name] = info\n                pending.pop(mount_name)\n                progressed = True\n\n                log(\n                    f"Computed direct mount offset BEFORE detach: {mount_name} "\n                    f"ref_action={info[\'reference_action\']} frame={info[\'reference_frame\']} "\n                    f"mount_bone={info[\'mount_bone\']} weapon_bone={weapon_bone}"\n                )\n\n            elif typ == "mirror":\n                source_mount = str(mount_cfg["source_mount"])\n                if source_mount not in out:\n                    continue\n\n                info = compute_mirror_mount_offset(\n                    mount_name=mount_name,\n                    mount_cfg=mount_cfg,\n                    source_mount_info=out[source_mount],\n                )\n                out[mount_name] = info\n                pending.pop(mount_name)\n                progressed = True\n\n                log(\n                    f"Computed mirror mount offset: {mount_name} "\n                    f"source_mount={source_mount} "\n                    f"mount_bone={info[\'mount_bone\']} "\n                    f"mirror_signs={info[\'mirror_signs\']}"\n                )\n\n            else:\n                raise RuntimeError(f"Unsupported mount type for {mount_name}: {typ}")\n\n        require(\n            progressed,\n            "Could not resolve all mounts. Possible missing or cyclic source_mount references. "\n            f"Pending={sorted(pending.keys())}, resolved={sorted(out.keys())}"\n        )\n\n    return out\n\n\ndef capture_weapon_world_for_frames(\n    *,\n    arm: bpy.types.Object,\n    action: bpy.types.Action,\n    weapon_bone: str,\n    frames: list[int],\n) -> np.ndarray:\n    set_armature_action(arm, action)\n\n    arr = np.empty((len(frames), 4, 4), dtype=np.float64)\n    for i, frame in enumerate(frames):\n        frame_set(frame)\n        arr[i] = np_mat(pose_bone_world_matrix(arm, weapon_bone))\n\n    return arr\n\n\ndef capture_original_weapon_worlds(\n    *,\n    arm: bpy.types.Object,\n    weapon_bone: str,\n    target_specs: list[dict[str, Any]],\n) -> dict[str, np.ndarray]:\n    out: dict[str, np.ndarray] = {}\n\n    for spec in target_specs:\n        action = get_action(spec["action"])\n        frames = list(spec["frames"])\n        out[spec["action"]] = capture_weapon_world_for_frames(\n            arm=arm,\n            action=action,\n            weapon_bone=weapon_bone,\n            frames=frames,\n        )\n        log(f"Captured original Weapon world: action={spec[\'action\']} frames={frames[0]}..{frames[-1]}")\n\n    return out\n\n\ndef detach_weapon_bone_preserve_rest(\n    *,\n    arm: bpy.types.Object,\n    weapon_bone: str,\n) -> dict[str, Any]:\n    ensure_object_mode()\n    update_view_layer()\n\n    bone = arm.data.bones.get(weapon_bone)\n    require(bone is not None, f"Weapon bone not found in armature data: {weapon_bone}")\n\n    before_parent = bone.parent.name if bone.parent is not None else None\n    before_children = [child.name for child in bone.children]\n    before_use_connect = bool(getattr(bone, "use_connect", False))\n    before_matrix = bone.matrix_local.copy()\n    before_length = float(bone.length)\n    before_use_deform = bool(bone.use_deform)\n\n    bpy.ops.object.select_all(action="DESELECT")\n    arm.select_set(True)\n    bpy.context.view_layer.objects.active = arm\n\n    result = bpy.ops.object.mode_set(mode="EDIT")\n    require(result == {"FINISHED"}, f"Failed to enter EDIT mode: {result}")\n\n    try:\n        eb = arm.data.edit_bones.get(weapon_bone)\n        require(eb is not None, f"Edit bone not found: {weapon_bone}")\n\n        loc, rot, _scale = before_matrix.decompose()\n        length = max(before_length, 1e-6)\n\n        eb.use_connect = False\n        eb.parent = None\n\n        eb.head = loc\n        eb.tail = loc + (rot @ Vector((0.0, length, 0.0)))\n        try:\n            eb.align_roll(rot @ Vector((0.0, 0.0, 1.0)))\n        except Exception as exc:\n            warn(f"align_roll failed for {weapon_bone}: {exc}")\n\n        eb.use_deform = bool(before_use_deform)\n\n    finally:\n        result = bpy.ops.object.mode_set(mode="OBJECT")\n        require(result == {"FINISHED"}, f"Failed to return OBJECT mode: {result}")\n\n    update_view_layer()\n\n    after_bone = arm.data.bones.get(weapon_bone)\n    require(after_bone is not None, f"Weapon bone disappeared after detach: {weapon_bone}")\n\n    after_parent = after_bone.parent.name if after_bone.parent is not None else None\n    after_children = [child.name for child in after_bone.children]\n    after_matrix = after_bone.matrix_local.copy()\n\n    rest_abs = float(np.max(np.abs(np_mat(after_matrix) - np_mat(before_matrix))))\n    root_bones = sorted([bone.name for bone in arm.data.bones if bone.parent is None])\n\n    return {\n        "weapon_bone": weapon_bone,\n        "before_parent": before_parent,\n        "after_parent": after_parent,\n        "before_children": before_children,\n        "after_children": after_children,\n        "before_use_connect": before_use_connect,\n        "before_use_deform": before_use_deform,\n        "after_use_deform": bool(after_bone.use_deform),\n        "rest_matrix_abs_max_after_detach": rest_abs,\n        "root_bones_after_detach": root_bones,\n        "before_matrix_local": matrix_to_list(before_matrix),\n        "after_matrix_local": matrix_to_list(after_matrix),\n    }\n\n\ndef build_target_specs(\n    *,\n    config: dict[str, Any],\n    preserve_unconfigured: bool,\n) -> list[dict[str, Any]]:\n    policies = enabled_action_policies(config)\n\n    specs: list[dict[str, Any]] = []\n    configured_names = set(policies.keys())\n\n    for action_name, policy in policies.items():\n        action = get_action(action_name)\n        mode = str(policy.get("mode", "mount"))\n        require(mode == "mount", f"Only policy mode=\'mount\' is supported for configured actions, got {mode!r}")\n\n        mount_name = str(policy["mount"])\n        frame_start = policy.get("frame_start", None)\n        frame_end = policy.get("frame_end", None)\n\n        frames = resolve_action_frames(\n            action,\n            frame_start=int(frame_start) if frame_start is not None else None,\n            frame_end=int(frame_end) if frame_end is not None else None,\n        )\n\n        specs.append(\n            {\n                "action": action_name,\n                "mode": "mount",\n                "mount": mount_name,\n                "frames": frames,\n                "policy": policy,\n            }\n        )\n\n    if preserve_unconfigured:\n        for action_name in all_action_names():\n            if action_name in configured_names:\n                continue\n\n            action = get_action(action_name)\n            frames = resolve_action_frames(action, frame_start=None, frame_end=None)\n\n            specs.append(\n                {\n                    "action": action_name,\n                    "mode": "preserve_original",\n                    "mount": None,\n                    "frames": frames,\n                    "policy": {\n                        "enabled": True,\n                        "mode": "preserve_original",\n                        "description": "Auto-preserved because Weapon bone is detached globally.",\n                    },\n                }\n            )\n\n    specs = sorted(specs, key=lambda x: (0 if x["mode"] == "mount" else 1, x["action"]))\n    require(specs, "No action specs to bake.")\n    return specs\n\n\ndef make_expected_world_for_spec(\n    *,\n    arm: bpy.types.Object,\n    spec: dict[str, Any],\n    mount_offsets: dict[str, dict[str, Any]],\n    original_worlds: dict[str, np.ndarray],\n) -> np.ndarray:\n    action_name = str(spec["action"])\n    mode = str(spec["mode"])\n    frames = list(spec["frames"])\n\n    if mode == "preserve_original":\n        return np.asarray(original_worlds[action_name], dtype=np.float64).copy()\n\n    require(mode == "mount", f"Unsupported bake mode: {mode}")\n\n    mount_name = str(spec["mount"])\n    require(mount_name in mount_offsets, f"Unknown mount {mount_name!r} for action {action_name}")\n\n    mount_info = mount_offsets[mount_name]\n    mount_bone = str(mount_info["mount_bone"])\n    offset = bl_mat(np.asarray(mount_info["offset_np"], dtype=np.float64))\n\n    action = get_action(action_name)\n    set_armature_action(arm, action)\n\n    expected = np.empty((len(frames), 4, 4), dtype=np.float64)\n\n    for i, frame in enumerate(frames):\n        frame_set(int(frame))\n        mount_world = pose_bone_world_matrix(arm, mount_bone)\n        expected[i] = np_mat(mount_world @ offset)\n\n    return expected\n\n\ndef bake_weapon_to_expected_worlds(\n    *,\n    arm: bpy.types.Object,\n    weapon_bone: str,\n    spec: dict[str, Any],\n    expected_world: np.ndarray,\n    original_world: np.ndarray,\n    position_tol: float,\n    rotation_tol_deg: float,\n    linear_tol: float,\n) -> dict[str, Any]:\n    action_name = str(spec["action"])\n    action = get_action(action_name)\n    frames = list(spec["frames"])\n\n    require(expected_world.shape == (len(frames), 4, 4), f"Bad expected_world shape for {action_name}")\n    require(original_world.shape == (len(frames), 4, 4), f"Bad original_world shape for {action_name}")\n\n    set_armature_action(arm, action)\n\n    remove_summary = remove_bone_fcurves(action, weapon_bone)\n\n    pb = get_pose_bone(arm, weapon_bone)\n    pb.rotation_mode = "QUATERNION"\n\n    previous_quat = None\n\n    for i, frame in enumerate(frames):\n        frame_set(int(frame))\n\n        target_world = bl_mat(expected_world[i])\n        desired_pose = arm.matrix_world.inverted() @ target_world\n\n        pb.rotation_mode = "QUATERNION"\n        pb.matrix = desired_pose\n        update_view_layer()\n\n        q = pb.rotation_quaternion.copy()\n        if previous_quat is not None and q.dot(previous_quat) < 0:\n            q.negate()\n            pb.rotation_quaternion = q\n            update_view_layer()\n        previous_quat = q.copy()\n\n        for channel in ("location", "rotation_quaternion", "scale"):\n            ok = pb.keyframe_insert(data_path=channel, frame=float(frame), group=weapon_bone)\n            require(ok, f"Failed keyframe insert: action={action_name}, frame={frame}, channel={channel}")\n\n    interp_summary = set_weapon_fcurve_interpolation(action, weapon_bone)\n\n    actual_world = capture_weapon_world_for_frames(\n        arm=arm,\n        action=action,\n        weapon_bone=weapon_bone,\n        frames=frames,\n    )\n\n    errors = error_arrays(actual_world, expected_world)\n    error_summary = summarize_error(errors, frames)\n\n    original_delta = error_arrays(original_world, expected_world)\n    original_delta_summary = summarize_error(original_delta, frames)\n\n    passed = bool(\n        error_summary["translation_l2_max"] <= position_tol\n        and error_summary["rotation_deg_max"] <= rotation_tol_deg\n        and error_summary["linear_relative_max"] <= linear_tol\n    )\n\n    return {\n        "action": action_name,\n        "mode": spec["mode"],\n        "mount": spec.get("mount"),\n        "weapon_bone": weapon_bone,\n        "frame_start": int(frames[0]),\n        "frame_end": int(frames[-1]),\n        "frame_count": int(len(frames)),\n        "policy": spec.get("policy", {}),\n        "remove_old_weapon_fcurves": remove_summary,\n        "weapon_fcurves_after_bake": interp_summary,\n        "validation": {\n            "passed": passed,\n            "errors": error_summary,\n            "tolerances": {\n                "position_tol": float(position_tol),\n                "rotation_tol_deg": float(rotation_tol_deg),\n                "linear_tol": float(linear_tol),\n            },\n        },\n        "original_to_fixed_delta": original_delta_summary,\n        "_frames": frames,\n        "_expected_world": expected_world,\n        "_actual_world": actual_world,\n        "_original_world": original_world,\n    }\n\n\ndef validate_after_reload(\n    *,\n    arm_name: str,\n    action_name: str,\n    weapon_bone: str,\n    frames: list[int],\n    expected_world: np.ndarray,\n    position_tol: float,\n    rotation_tol_deg: float,\n    linear_tol: float,\n) -> dict[str, Any]:\n    arm = get_armature(arm_name)\n    action = get_action(action_name)\n\n    actual_world = capture_weapon_world_for_frames(\n        arm=arm,\n        action=action,\n        weapon_bone=weapon_bone,\n        frames=frames,\n    )\n\n    errors = error_arrays(actual_world, expected_world)\n    error_summary = summarize_error(errors, frames)\n\n    passed = bool(\n        error_summary["translation_l2_max"] <= position_tol\n        and error_summary["rotation_deg_max"] <= rotation_tol_deg\n        and error_summary["linear_relative_max"] <= linear_tol\n    )\n\n    return {\n        "action": action_name,\n        "passed": passed,\n        "errors": error_summary,\n    }\n\n\ndef write_frame_csv(path: Path, *, action_results: list[dict[str, Any]]) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n\n    fields = [\n        "action",\n        "mode",\n        "mount",\n        "frame",\n        "translation_l2",\n        "rotation_deg",\n        "linear_relative",\n        "matrix_abs",\n        "original_to_fixed_translation_l2",\n        "original_to_fixed_rotation_deg",\n        "original_to_fixed_linear_relative",\n        "original_to_fixed_matrix_abs",\n    ]\n\n    with path.open("w", encoding="utf-8", newline="") as f:\n        writer = csv.DictWriter(f, fieldnames=fields)\n        writer.writeheader()\n\n        for result in action_results:\n            action = result["action"]\n            mode = result["mode"]\n            mount = result.get("mount")\n            frames = result["_frames"]\n            expected = result["_expected_world"]\n            actual = result["_actual_world"]\n            original = result["_original_world"]\n\n            errors = error_arrays(actual, expected)\n            delta = error_arrays(original, expected)\n\n            for i, frame in enumerate(frames):\n                writer.writerow(\n                    {\n                        "action": action,\n                        "mode": mode,\n                        "mount": mount,\n                        "frame": int(frame),\n                        "translation_l2": float(errors["translation_l2"][i]),\n                        "rotation_deg": float(errors["rotation_deg"][i]),\n                        "linear_relative": float(errors["linear_relative"][i]),\n                        "matrix_abs": float(errors["matrix_abs"][i]),\n                        "original_to_fixed_translation_l2": float(delta["translation_l2"][i]),\n                        "original_to_fixed_rotation_deg": float(delta["rotation_deg"][i]),\n                        "original_to_fixed_linear_relative": float(delta["linear_relative"][i]),\n                        "original_to_fixed_matrix_abs": float(delta["matrix_abs"][i]),\n                    }\n                )\n\n\ndef scene_summary(arm: bpy.types.Object | None = None, weapon_bone: str | None = None) -> dict[str, Any]:\n    arms = sorted([obj.name for obj in bpy.data.objects if obj.type == "ARMATURE"])\n    meshes = sorted([obj.name for obj in bpy.data.objects if obj.type == "MESH"])\n    actions = sorted([a.name for a in bpy.data.actions])\n    empties = sorted([obj.name for obj in bpy.data.objects if obj.type == "EMPTY"])\n\n    out = {\n        "armature_count": int(len(arms)),\n        "mesh_count": int(len(meshes)),\n        "action_count": int(len(actions)),\n        "empty_count": int(len(empties)),\n        "armatures": arms,\n        "actions": actions,\n        "meshes_first_100": meshes[:100],\n        "empties_first_100": empties[:100],\n    }\n\n    if arm is not None:\n        out["root_bones"] = sorted([bone.name for bone in arm.data.bones if bone.parent is None])\n\n        if weapon_bone:\n            bone = arm.data.bones.get(weapon_bone)\n            out["weapon_bone"] = {\n                "name": weapon_bone,\n                "exists": bone is not None,\n                "parent": bone.parent.name if bone is not None and bone.parent is not None else None,\n                "children": [child.name for child in bone.children] if bone is not None else [],\n                "use_deform": bool(bone.use_deform) if bone is not None else None,\n            }\n\n    return out\n\n\ndef strip_internal_fields(result: dict[str, Any]) -> dict[str, Any]:\n    return {k: v for k, v in result.items() if not k.startswith("_")}\n\n\ndef self_test() -> None:\n    ident = np.eye(4)[None]\n    rot250 = np_mat(Matrix.Rotation(math.radians(250), 4, "Y"))[None]\n    e = error_arrays(rot250, ident)\n    angle = float(e["rotation_deg"][0])\n    require(abs(angle - 110.0) < 1e-3, f"Shortest rotation self-test failed: {angle}")\n\n    reflect = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))\n    rot = Matrix.Rotation(math.radians(30), 4, "Z")\n    mirrored = reflect @ rot @ reflect\n    det = np.linalg.det(np.asarray(mirrored, dtype=np.float64)[:3, :3])\n    require(abs(det - 1.0) < 1e-6, f"Mirror sandwich produced improper rotation determinant: {det}")\n\n    log(f"Self-test passed. bpy={bpy.app.version_string}")\n    log(f"Repo root: {resolve_repo_root()}")\n\n\ndef execute(args: argparse.Namespace) -> None:\n    input_blend = Path(args.input_blend).expanduser().resolve(strict=False)\n    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)\n    output_blend = (\n        Path(args.output_blend).expanduser().resolve(strict=False)\n        if args.output_blend\n        else (output_dir / "Winter_Samurai_clean04_weapon_mount_fixed.blend").resolve(strict=False)\n    )\n    config_path = Path(args.config).expanduser().resolve(strict=False)\n\n    report_path = output_dir / "weapon_mount_fix_report.json"\n    csv_path = output_dir / "weapon_mount_fix_frame_errors.csv"\n\n    require(input_blend.is_file(), f"Missing input_blend: {input_blend}")\n    require(config_path.is_file(), f"Missing config: {config_path}")\n    require(output_blend != input_blend, "Refusing to overwrite input blend.")\n\n    output_dir.mkdir(parents=True, exist_ok=True)\n\n    if bool(args.overwrite):\n        for p in [output_blend, report_path, csv_path]:\n            if p.exists():\n                p.unlink()\n    else:\n        for p in [output_blend, report_path, csv_path]:\n            require(not p.exists(), f"Output already exists; use --overwrite or fresh output_dir: {p}")\n\n    config = load_json(config_path)\n\n    report: dict[str, Any] = {\n        "schema": "ogd_lbs_winter_samurai_weapon_mount_fix_report_v3_mirror_support",\n        "created_at_utc": utc_now_iso(),\n        "repo_root": str(resolve_repo_root()),\n        "bpy_version": bpy.app.version_string,\n        "input_blend": str(input_blend),\n        "output_dir": str(output_dir),\n        "output_blend": str(output_blend),\n        "config_path": str(config_path),\n        "report_json": str(report_path),\n        "frame_error_csv": str(csv_path),\n        "status": "running",\n        "stage": "open",\n        "tolerances": {\n            "position_tol": float(args.position_tol),\n            "rotation_tol_deg": float(args.rotation_tol_deg),\n            "linear_tol": float(args.linear_tol),\n        },\n    }\n\n    try:\n        open_blend(input_blend)\n\n        arm = get_armature(args.armature_name)\n        arm_name = arm.name\n\n        weapon_bone = str(config.get("weapon_bone", "Weapon"))\n        weapon_mesh = str(config.get("weapon_mesh", "Weapon"))\n\n        require(arm.pose.bones.get(weapon_bone) is not None, f"Weapon bone missing: {weapon_bone}")\n\n        mesh_obj = bpy.data.objects.get(weapon_mesh)\n        require(mesh_obj is not None and mesh_obj.type == "MESH", f"Weapon mesh missing or not mesh: {weapon_mesh}")\n\n        detach_cfg = config.get("detach_weapon_bone", {})\n        require(isinstance(detach_cfg, dict), "detach_weapon_bone must be an object if present.")\n        detach_enabled = bool(detach_cfg.get("enabled", True))\n        preserve_unconfigured = bool(detach_cfg.get("preserve_unconfigured_actions", True))\n\n        log("=" * 96)\n        log("Winter Samurai Weapon Mount Fix: detach + direct/mirror mount rebake")\n        log(f"input_blend : {input_blend}")\n        log(f"output_blend: {output_blend}")\n        log(f"config      : {config_path}")\n        log(f"armature    : {arm_name}")\n        log(f"weapon_bone : {weapon_bone}")\n        log(f"weapon_mesh : {weapon_mesh}")\n        log(f"detach      : {detach_enabled}")\n        log(f"preserve_unconfigured_actions: {preserve_unconfigured}")\n        log("=" * 96)\n\n        report["stage"] = "scene_initial"\n        report["scene_initial"] = scene_summary(arm, weapon_bone)\n\n        report["stage"] = "compute_mount_offsets_before_detach"\n        mount_offsets = compute_mount_offsets_before_detach(\n            arm=arm,\n            config=config,\n            weapon_bone=weapon_bone,\n        )\n        report["mounts"] = {\n            name: {k: v for k, v in item.items() if k != "offset_np"}\n            for name, item in mount_offsets.items()\n        }\n\n        report["stage"] = "build_target_specs"\n        target_specs = build_target_specs(\n            config=config,\n            preserve_unconfigured=preserve_unconfigured,\n        )\n        report["target_specs"] = [\n            {\n                "action": item["action"],\n                "mode": item["mode"],\n                "mount": item.get("mount"),\n                "frame_start": int(item["frames"][0]),\n                "frame_end": int(item["frames"][-1]),\n                "frame_count": int(len(item["frames"])),\n                "policy": item.get("policy", {}),\n            }\n            for item in target_specs\n        ]\n\n        report["stage"] = "capture_original_weapon_worlds"\n        original_worlds = capture_original_weapon_worlds(\n            arm=arm,\n            weapon_bone=weapon_bone,\n            target_specs=target_specs,\n        )\n\n        report["stage"] = "detach_weapon_bone"\n        if detach_enabled:\n            detach_report = detach_weapon_bone_preserve_rest(\n                arm=arm,\n                weapon_bone=weapon_bone,\n            )\n        else:\n            warn("detach_weapon_bone.enabled=false; skipping detach.")\n            detach_report = {\n                "skipped": True,\n                "reason": "detach_weapon_bone.enabled=false",\n            }\n\n        report["detach_weapon_bone"] = detach_report\n        report["scene_after_detach"] = scene_summary(arm, weapon_bone)\n\n        report["stage"] = "bake_actions"\n        internal_results: list[dict[str, Any]] = []\n        action_results: list[dict[str, Any]] = []\n\n        for spec in target_specs:\n            action_name = spec["action"]\n            log(\n                f"Baking Weapon: action={action_name} mode={spec[\'mode\']} "\n                f"mount={spec.get(\'mount\')} frames={spec[\'frames\'][0]}..{spec[\'frames\'][-1]}"\n            )\n\n            expected_world = make_expected_world_for_spec(\n                arm=arm,\n                spec=spec,\n                mount_offsets=mount_offsets,\n                original_worlds=original_worlds,\n            )\n\n            result = bake_weapon_to_expected_worlds(\n                arm=arm,\n                weapon_bone=weapon_bone,\n                spec=spec,\n                expected_world=expected_world,\n                original_world=original_worlds[action_name],\n                position_tol=float(args.position_tol),\n                rotation_tol_deg=float(args.rotation_tol_deg),\n                linear_tol=float(args.linear_tol),\n            )\n\n            internal_results.append(result)\n            action_results.append(strip_internal_fields(result))\n\n        report["action_results_before_save"] = action_results\n        write_frame_csv(csv_path, action_results=internal_results)\n\n        failed = [\n            item for item in action_results\n            if not item.get("validation", {}).get("passed", False)\n        ]\n        require(not failed, f"Weapon validation before save failed: {failed}")\n\n        report["stage"] = "save"\n        report["scene_before_save"] = scene_summary(arm, weapon_bone)\n        dump_json(report_path, report)\n        save_blend(output_blend)\n\n        report["stage"] = "validate_after_reload"\n        open_blend(output_blend)\n\n        reload_validations = []\n        for result in internal_results:\n            reload_validation = validate_after_reload(\n                arm_name=arm_name,\n                action_name=result["action"],\n                weapon_bone=weapon_bone,\n                frames=result["_frames"],\n                expected_world=result["_expected_world"],\n                position_tol=float(args.position_tol),\n                rotation_tol_deg=float(args.rotation_tol_deg),\n                linear_tol=float(args.linear_tol),\n            )\n            reload_validations.append(reload_validation)\n\n        failed_reload = [item for item in reload_validations if not item.get("passed", False)]\n        require(not failed_reload, f"Weapon validation after reload failed: {failed_reload}")\n\n        arm_reloaded = get_armature(arm_name)\n\n        report["status"] = "passed"\n        report["stage"] = "complete"\n        report["scene_after_reload"] = scene_summary(arm_reloaded, weapon_bone)\n        report["validation_after_reload"] = reload_validations\n        report["scope"] = (\n            "This step detaches Weapon from its rest parent and bakes Weapon bone "\n            "world-space animation. Configured actions are driven by direct or mirrored "\n            "virtual mount points. Unconfigured actions are preserved by rebaking their "\n            "original Weapon world motion."\n        )\n\n        dump_json(report_path, report)\n\n        log("=" * 96)\n        log("Weapon mount detach+rebake PASSED")\n        log(f"output_blend : {output_blend}")\n        log(f"report_json  : {report_path}")\n        log(f"frame_csv    : {csv_path}")\n        log(f"root_bones   : {report[\'scene_after_reload\'].get(\'root_bones\')}")\n        log(f"weapon_parent: {report[\'scene_after_reload\'].get(\'weapon_bone\', {}).get(\'parent\')}")\n        for item in reload_validations:\n            log(f"  after reload: {item[\'action\']} passed={item[\'passed\']} errors={item[\'errors\']}")\n        log("=" * 96)\n\n    except Exception as exc:\n        report["status"] = "failed"\n        report["stage_failed"] = report.get("stage")\n        report["error"] = str(exc)\n        report["traceback"] = traceback.format_exc()\n        try:\n            dump_json(report_path, report)\n            log(f"Failure report written: {report_path}")\n        except Exception:\n            pass\n        raise\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser(\n        description="Detach Winter Samurai Weapon bone and rebake via direct/mirrored virtual mounts."\n    )\n\n    parser.add_argument("--self_test", action="store_true")\n\n    parser.add_argument("--input_blend", type=str, default=None)\n    parser.add_argument("--output_dir", type=str, default=None)\n    parser.add_argument("--output_blend", type=str, default=None)\n    parser.add_argument("--config", type=str, default=None)\n\n    parser.add_argument("--armature_name", type=str, default=None)\n\n    parser.add_argument("--position_tol", type=float, default=1e-4)\n    parser.add_argument("--rotation_tol_deg", type=float, default=0.05)\n    parser.add_argument("--linear_tol", type=float, default=1e-5)\n\n    parser.add_argument("--overwrite", action="store_true")\n\n    args = parser.parse_args(script_argv())\n\n    self_test()\n\n    if args.self_test:\n        return\n\n    if not args.input_blend or not args.output_dir or not args.config:\n        parser.error("--input_blend, --output_dir, and --config are required unless --self_test is used.")\n\n    for key in ["position_tol", "rotation_tol_deg", "linear_tol"]:\n        value = float(getattr(args, key))\n        if not math.isfinite(value) or value <= 0:\n            parser.error(f"--{key} must be finite and positive.")\n\n    execute(args)\n\n\nif __name__ == "__main__":\n    main()\n'


def _load_embedded_core():
    module = types.ModuleType("embedded_weapon_mount_core")

    # The embedded legacy core expects __file__ in resolve_repo_root().
    # A module created by types.ModuleType does not get __file__ automatically,
    # so define it explicitly before exec().
    from pathlib import Path as _Path
    module.__dict__["__file__"] = str(_Path(__file__).resolve(strict=False))
    module.__dict__["__package__"] = ""

    exec(_EMBEDDED_WEAPON_MOUNT_CORE_SOURCE, module.__dict__)
    required = (
        "execute",
        "make_expected_world_for_spec",
        "set_armature_action",
        "frame_set",
        "error_arrays",
        "update_view_layer",
        "open_blend",
    )
    for name in required:
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"Embedded weapon mount core missing callable: {name}")
    return module



"""
Weapon mount extension:
    preserve the direct right-hand mount's Weapon origin;
    point Weapon local -Y toward the evaluated left-hand midpoint + offset.

Reuses the existing sibling 04_fix_weapon_mount.py.
No mirror, negative scale, stretch or persistent constraint is created.

Offset:
    target_offset_fraction_of_length is expressed in target-bone local axes,
    multiplied by target rest bone length, then transformed into world space.

The original implementation handles detach, action rebake, save/reload.
This extension adds independent direction/origin/non-descendant-motion audits.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np

TAG = "[ws-weapon-aim]"


def log(text):
    print(f"{TAG} {text}", flush=True)


def require(ok, text):
    if not ok:
        raise RuntimeError(text)


def argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


def dump(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_core(path):
    return _load_embedded_core()


def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    length = float(np.linalg.norm(v))
    require(np.isfinite(v).all() and length > eps, "Degenerate direction.")
    return v / length


def aim_matrix(base, target):
    """Keep origin and positive axis scales; set local -Y toward target."""
    base = np.asarray(base, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    require(np.isfinite(base).all() and np.isfinite(target).all(),
            "Nonfinite aim input.")
    require(np.allclose(base[3], [0, 0, 0, 1], atol=1e-6, rtol=0),
            "Non-affine baseline.")

    linear = base[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    require(np.all(scales > 1e-12), "Singular baseline.")
    rotation = linear / scales[None, :]
    require(np.linalg.det(rotation) > 0, "Reflected baseline is unsupported.")
    require(np.max(np.abs(rotation.T @ rotation - np.eye(3))) < 1e-4,
            "Baseline contains shear; refusing silent approximation.")

    origin = base[:3, 3]
    direction = target - origin
    require(np.linalg.norm(direction) > 1e-9,
            "Aim target coincides with Weapon origin.")

    # Weapon -Y points toward the target.
    y = -unit(direction)

    # Preserve baseline Z as closely as possible to fix roll.
    z0 = rotation[:, 2]
    projected_z = z0 - np.dot(z0, y) * y

    if np.linalg.norm(projected_z) >= 1e-4:
        z = unit(projected_z)
        x = unit(np.cross(y, z))
        z = unit(np.cross(x, y))
        branch = "PROJECT_BASELINE_Z"
    else:
        # Explicitly reported fallback near the roll singularity.
        x0 = rotation[:, 0]
        x = unit(x0 - np.dot(x0, y) * y)
        z = unit(np.cross(x, y))
        x = unit(np.cross(y, z))
        branch = "FALLBACK_PROJECT_BASELINE_X"

    r = np.column_stack((x, y, z))
    require(abs(np.linalg.det(r) - 1.0) < 1e-8, "Improper aim rotation.")

    out = base.copy()
    out[:3, :3] = r * scales[None, :]
    return out, branch


def set_action(core, arm, action):
    core.set_armature_action(arm, action)
    slots = list(getattr(action, "slots", []))
    if slots:
        require(len(slots) == 1, "Aim test supports one-slot Actions.")
        arm.animation_data.action_slot = slots[0]
    core.update_view_layer()


def evaluated_state(arm, target_name, offset_fraction, watched_names):
    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = arm.evaluated_get(dg)
    a = np.asarray(evaluated.matrix_world, dtype=np.float64)
    pb = evaluated.pose.bones[target_name]

    # Midpoint of evaluated head/tail, in armature space.
    midpoint = (
        np.asarray(pb.head, dtype=np.float64)
        + np.asarray(pb.tail, dtype=np.float64)
    ) * 0.5

    length = float(arm.data.bones[target_name].length)
    local_offset = np.asarray(offset_fraction, dtype=np.float64) * length
    pose_linear = np.asarray(pb.matrix, dtype=np.float64)[:3, :3]

    target_armature = midpoint + pose_linear @ local_offset
    target_world = a[:3, :3] @ target_armature + a[:3, 3]

    watched = np.stack([
        a @ np.asarray(evaluated.pose.bones[n].matrix, dtype=np.float64)
        for n in watched_names
    ])
    return target_world, watched


def summarize_matrix_error(core, actual, expected):
    errors = core.error_arrays(actual, expected)
    return {key: float(np.max(value)) for key, value in errors.items()}


def within(errors, position, rotation, linear):
    return (
        errors["translation_l2"] <= position
        and errors["rotation_deg"] <= rotation
        and errors["linear_relative"] <= linear
    )


def independent_audit(core, out):
    manifest = json.loads((out / "aim_audit_manifest.json").read_text(encoding="utf-8"))
    core.open_blend(Path(manifest["output_blend"]))
    arm = bpy.data.objects[manifest["armature"]]
    weapon = manifest["weapon_bone"]
    require(arm.data.bones[weapon].parent is None, "Weapon was not detached.")

    result = {"status": "running", "actions": []}
    for entry in manifest["actions"]:
        with np.load(out / entry["cache"], allow_pickle=False) as z:
            times = z["times"]
            expected = z["expected_world"]
            baseline = z["baseline_world"]
            targets = z["target_world"]
            watched_expected = z["watched_world"]
            watched_names = z["watched_names"].tolist()

        set_action(core, arm, bpy.data.actions[entry["action"]])
        actual = []
        watched_actual = []
        current_targets = []

        for t in times:
            core.frame_set(float(t))
            target, watched = evaluated_state(
                arm, entry["target_bone"],
                entry["target_offset_fraction_of_length"], watched_names,
            )
            evaluated = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
            w = np.asarray(
                evaluated.matrix_world @ evaluated.pose.bones[weapon].matrix,
                dtype=np.float64,
            )
            actual.append(w)
            watched_actual.append(watched)
            current_targets.append(target)

        actual = np.stack(actual)
        watched_actual = np.stack(watched_actual)
        current_targets = np.stack(current_targets)

        pose_error = summarize_matrix_error(core, actual, expected)
        other_error = summarize_matrix_error(core, watched_actual, watched_expected)

        ray = current_targets - actual[:, :3, 3]
        ray = ray / np.linalg.norm(ray, axis=1)[:, None]
        minus_y = -actual[:, :3, 1]
        minus_y = minus_y / np.linalg.norm(minus_y, axis=1)[:, None]
        cross = np.linalg.norm(np.cross(minus_y, ray), axis=1)
        dot = np.clip(np.sum(minus_y * ray, axis=1), -1, 1)
        angles = np.degrees(np.arctan2(cross, dot))

        origin_error = float(np.max(np.linalg.norm(
            actual[:, :3, 3] - baseline[:, :3, 3], axis=1
        )))
        target_error = float(np.max(np.linalg.norm(current_targets - targets, axis=1)))
        det_min = float(np.min(np.linalg.det(actual[:, :3, :3])))

        tol = manifest["tolerances"]
        passed = bool(
            within(pose_error, tol["position"], tol["rotation_deg"], tol["linear"])
            and within(other_error, tol["position"], tol["rotation_deg"], tol["linear"])
            and origin_error <= tol["position"]
            and target_error <= tol["position"]
            and float(angles.max()) <= tol["aim_angle_deg"]
            and det_min > 0
        )

        item = {
            "action": entry["action"],
            "passed": passed,
            "sample_count": len(times),
            "aim_minus_y_angle_max_deg": float(angles.max()),
            "weapon_origin_vs_right_mount_max": origin_error,
            "left_target_change_max": target_error,
            "weapon_world_error": pose_error,
            "non_weapon_non_descendant_world_error": other_error,
            "excluded_weapon_descendants": entry["excluded_descendants"],
            "world_linear_det_min": det_min,
            "roll_fallback_count": entry["roll_fallback_count"],
        }
        result["actions"].append(item)
        log(str(item))

    require(result["actions"], "No aim actions were audited.")
    result["status"] = (
        "passed" if all(x["passed"] for x in result["actions"]) else "failed"
    )
    dump(out / "aim_audit_report.json", result)
    require(result["status"] == "passed", "Independent aim audit failed.")
    return result


def self_test():
    for target in ([1, 2, 3], [0, 0, 1], [0, 0, -1], [-2, 1, 0]):
        base = np.eye(4)
        base[:3, :3] *= 2
        result, _ = aim_matrix(base, target)
        require(np.allclose(-unit(result[:3, 1]), unit(target)),
                "-Y direction self-test failed.")
        require(np.array_equal(result[:3, 3], base[:3, 3]),
                "Origin preservation self-test failed.")
        require(np.allclose(np.linalg.norm(result[:3, :3], axis=0), [2, 2, 2]),
                "Scale preservation self-test failed.")
        require(np.linalg.det(result[:3, :3]) > 0, "Reflection self-test failed.")
    rejected = False
    try:
        aim_matrix(np.eye(4), [0, 0, 0])
    except RuntimeError:
        rejected = True
    require(rejected, "Coincident target must fail.")
    log(f"Math self-test passed; bpy={bpy.app.version_string}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--audit_saved", action="store_true")
    parser.add_argument("--input_blend")
    parser.add_argument("--output_dir")
    parser.add_argument("--output_blend")
    parser.add_argument("--config")
    parser.add_argument("--core_script")
    parser.add_argument("--armature_name")
    parser.add_argument("--position_tol", type=float, default=1e-4)
    parser.add_argument("--rotation_tol_deg", type=float, default=0.05)
    parser.add_argument("--linear_tol", type=float, default=1e-5)
    args = parser.parse_args(argv())

    if args.self_test:
        self_test()
        return

    require(args.output_dir, "--output_dir is required.")
    core_path = Path(args.core_script).resolve() if args.core_script else Path(__file__).resolve()
    core = load_core(core_path)
    out = Path(args.output_dir).expanduser().resolve()

    if args.audit_saved:
        independent_audit(core, out)
        return

    require(args.input_blend and args.config, "--input_blend and --config are required.")
    for value in (args.position_tol, args.rotation_tol_deg, args.linear_tol):
        require(math.isfinite(value) and value > 0, "Invalid tolerance.")

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    require(all(m.get("type", "direct") == "direct" for m in cfg["mounts"].values()),
            "This aim experiment accepts direct mounts only; mirror is disabled.")
    require(cfg.get("detach_weapon_bone", {}).get("enabled", False),
            "This experiment requires Weapon detach.")
    aims = {
        name: policy["aim"]
        for name, policy in cfg["action_policies"].items()
        if policy.get("enabled", True) and "aim" in policy
    }
    require(aims, "No enabled aim policy.")

    out.mkdir(parents=True, exist_ok=True)
    for name in ("aim_cache", "aim_audit_manifest.json", "aim_audit_report.json",
                 "aim_extension_report.json"):
        require(not (out / name).exists(), f"Output exists: {out/name}")
    (out / "aim_cache").mkdir()

    output_blend = (
        Path(args.output_blend).expanduser().resolve() if args.output_blend
        else out / "Winter_Samurai_weapon_aim.blend"
    )
    require(not output_blend.exists(), f"Output exists: {output_blend}")

    report = {
        "status": "running",
        "core_script": str(core_path),
        "core_sha256": sha(core_path),
        "output_blend": str(output_blend),
    }
    manifest = {
        "output_blend": str(output_blend),
        "weapon_bone": cfg["weapon_bone"],
        "actions": [],
        "tolerances": {
            "position": args.position_tol,
            "rotation_deg": args.rotation_tol_deg,
            "linear": args.linear_tol,
            "aim_angle_deg": 0.05,
        },
    }
    original_generator = core.make_expected_world_for_spec

    def extended_generator(*, arm, spec, mount_offsets, original_worlds):
        baseline = original_generator(
            arm=arm, spec=spec, mount_offsets=mount_offsets,
            original_worlds=original_worlds,
        )
        action_name = spec["action"]
        if action_name not in aims:
            return baseline

        settings = aims[action_name]
        require(settings.get("axis", "-Y") == "-Y", "This version supports only -Y aim.")
        target_name = settings["target_bone"]
        offset = np.asarray(
            settings.get("target_offset_fraction_of_length", [0, 0, 0]),
            dtype=float,
        )
        require(offset.shape == (3,) and np.isfinite(offset).all(), "Invalid target offset.")

        weapon = arm.data.bones[cfg["weapon_bone"]]
        excluded = sorted(b.name for b in weapon.children_recursive)
        require(target_name in arm.pose.bones, f"Missing target bone: {target_name}")
        require(target_name != weapon.name and target_name not in excluded,
                "Aim target must not depend on Weapon.")
        watched_names = sorted(
            b.name for b in arm.data.bones
            if b.name != weapon.name and b.name not in excluded
        )
        require(watched_names, "No independent bones to audit.")

        set_action(core, arm, bpy.data.actions[action_name])
        targets, watched, expected, branches = [], [], [], []

        for i, t in enumerate(spec["frames"]):
            core.frame_set(float(t))
            target, other = evaluated_state(arm, target_name, offset, watched_names)
            result, branch = aim_matrix(baseline[i], target)
            targets.append(target)
            watched.append(other)
            expected.append(result)
            branches.append(branch)

        expected = np.stack(expected)
        cache = f"aim_cache/action_{len(manifest['actions']):03d}.npz"
        np.savez_compressed(
            out / cache,
            times=np.asarray(spec["frames"], dtype=float),
            expected_world=expected,
            baseline_world=baseline,
            target_world=np.stack(targets),
            watched_world=np.stack(watched),
            watched_names=np.asarray(watched_names, dtype=str),
        )
        manifest["armature"] = arm.name
        manifest["actions"].append({
            "action": action_name,
            "cache": cache,
            "target_bone": target_name,
            "target_offset_fraction_of_length": offset.tolist(),
            "excluded_descendants": excluded,
            "roll_fallback_count": branches.count("FALLBACK_PROJECT_BASELINE_X"),
        })
        dump(out / "aim_audit_manifest.json", manifest)
        log(f"{action_name}: -Y aim; target={target_name}; offset_fraction={offset.tolist()}")
        return expected

    try:
        core.make_expected_world_for_spec = extended_generator
        core.execute(argparse.Namespace(
            input_blend=args.input_blend,
            output_dir=str(out),
            output_blend=str(output_blend),
            config=args.config,
            armature_name=args.armature_name,
            position_tol=args.position_tol,
            rotation_tol_deg=args.rotation_tol_deg,
            linear_tol=args.linear_tol,
            overwrite=False,
        ))
        require({x["action"] for x in manifest["actions"]} == set(aims),
                "Not all requested aim policies were executed.")
        independent_audit(core, out)
        report["status"] = "passed"
        report["scope"] = (
            "Sampled -Y aiming and Weapon-origin preservation; "
            "not a guarantee of mesh grip-point preservation or subframe contact."
        )
        dump(out / "aim_extension_report.json", report)
        log(f"COMPLETE: {output_blend}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        dump(out / "aim_extension_report.json", report)
        raise
    finally:
        core.make_expected_world_for_spec = original_generator


if __name__ == "__main__":
    main()
