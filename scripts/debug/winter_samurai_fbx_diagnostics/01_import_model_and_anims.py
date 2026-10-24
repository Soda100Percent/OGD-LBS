#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Winter Samurai FBX raw import diagnostics.

This script intentionally does ONLY the first debug step:

    model FBX + selected animation FBX(s) -> one inspection .blend

It does NOT:
    - retarget animation to the model armature
    - copy actions to the model armature
    - concatenate actions into NLA
    - delete animation FBX imported armatures
    - attempt to fix broken animation playback

Expected dataset layout:

    /workspace/ogd_dataset/Winter_Samurai/
    ├── model/
    │   └── Base_Mesh_Rig.fbx
    ├── anim/
    │   ├── Anim_Winter_Samurai_Attack1.fbx
    │   └── ...
    └── configs/

Typical use:

    python scripts/debug/winter_samurai_fbx_diagnostics/01_import_model_and_anims.py \
        --dataset_dir /workspace/ogd_dataset/Winter_Samurai \
        --anim_name Anim_Winter_Samurai_Attack1 \
        --output_dir /workspace/ogd_tmp_blends/ws_attack1 \
        --output_blend /workspace/ogd_tmp_blends/ws_attack1/Winter_Samurai_model_plus_Attack1.blend
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


BONE_FCURVE_RE = re.compile(
    r"""pose\.bones\[\"([^\"]+)\"\]|pose\.bones\[\'([^\']+)\'\]"""
)


def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[winter-samurai-fbx-import] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[winter-samurai-fbx-import][WARN] {msg}", flush=True)


def resolve_repo_root() -> Path | None:
    this_file = Path(__file__).resolve(strict=False)
    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return None


def resolve_maybe_relative(path_text: str | None, base_dir: Path) -> Path | None:
    if path_text is None or not str(path_text).strip():
        return None

    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)

    return (base_dir / raw).resolve(strict=False)


def safe_float_list(values: Any) -> list[float]:
    try:
        return [float(x) for x in values]
    except Exception:
        return []


def clear_scene() -> None:
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for collection in [
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.actions,
        bpy.data.cameras,
        bpy.data.lights,
    ]:
        for datablock in list(collection):
            if getattr(datablock, "users", 0) == 0:
                try:
                    collection.remove(datablock)
                except Exception:
                    pass


def ensure_fbx_importer_available() -> None:
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_fbx")
    except Exception:
        pass

    if not hasattr(bpy.ops, "import_scene") or not hasattr(bpy.ops.import_scene, "fbx"):
        raise RuntimeError(
            "Blender FBX importer bpy.ops.import_scene.fbx is not available. "
            "Run this script with a Blender/Python environment that includes the FBX importer."
        )


def discover_model_fbx(dataset_dir: Path, explicit_model_fbx: str | None) -> Path:
    explicit = resolve_maybe_relative(explicit_model_fbx, dataset_dir)
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Explicit model FBX not found: {explicit}")
        if explicit.suffix.lower() != ".fbx":
            raise RuntimeError(f"Explicit model path is not an FBX file: {explicit}")
        return explicit

    model_dir = dataset_dir / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    candidates = sorted([p for p in model_dir.glob("*.fbx") if p.is_file()], key=lambda p: p.name)

    if not candidates:
        raise FileNotFoundError(f"No model FBX found under: {model_dir}")

    if len(candidates) > 1:
        raise RuntimeError(
            "Expected exactly one model FBX. Please pass --model_fbx explicitly.\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )

    return candidates[0].resolve(strict=False)


def filter_anim_candidates_by_name(candidates: list[Path], anim_name: str) -> list[Path]:
    """
    Select animation FBX by stem or filename.

    Examples:
        --anim_name Anim_Winter_Samurai_Attack1
        --anim_name Anim_Winter_Samurai_Attack1.fbx

    Matching order:
        1. exact case-sensitive filename/stem
        2. exact case-insensitive filename/stem
    """
    selector = str(anim_name or "").strip()
    if not selector:
        return candidates

    selector_path = Path(selector)
    selector_name = selector_path.name
    selector_stem = selector_path.stem if selector_name.lower().endswith(".fbx") else selector

    exact = [
        p for p in candidates
        if p.name == selector_name or p.stem == selector_stem
    ]
    if exact:
        return exact

    selector_name_l = selector_name.lower()
    selector_stem_l = selector_stem.lower()

    ci = [
        p for p in candidates
        if p.name.lower() == selector_name_l or p.stem.lower() == selector_stem_l
    ]
    return ci


def discover_anim_fbxs(
    *,
    dataset_dir: Path,
    explicit_anim_dir: str | None,
    explicit_anim_fbx: str | None,
    anim_name: str | None,
    recursive: bool,
    start_anim_index: int,
    max_anims: int,
) -> list[Path]:
    """
    Discover animation FBXs.

    Priority:
        1. --anim_fbx explicit single file
        2. --anim_name exact stem/name match under anim_dir
        3. all discovered animation FBXs, optionally sliced by start/max
    """
    if explicit_anim_fbx and anim_name:
        raise RuntimeError("--anim_fbx and --anim_name are mutually exclusive.")

    explicit = resolve_maybe_relative(explicit_anim_fbx, dataset_dir)
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Explicit animation FBX not found: {explicit}")
        if explicit.suffix.lower() != ".fbx":
            raise RuntimeError(f"Explicit animation path is not an FBX file: {explicit}")
        return [explicit.resolve(strict=False)]

    anim_dir = resolve_maybe_relative(explicit_anim_dir, dataset_dir)
    if anim_dir is None:
        anim_dir = dataset_dir / "anim"

    if not anim_dir.is_dir():
        raise FileNotFoundError(f"Missing animation directory: {anim_dir}")

    if recursive:
        candidates = sorted(
            [p for p in anim_dir.rglob("*.fbx") if p.is_file()],
            key=lambda p: str(p.relative_to(anim_dir)),
        )
    else:
        candidates = sorted(
            [p for p in anim_dir.glob("*.fbx") if p.is_file()],
            key=lambda p: p.name,
        )

        if not candidates:
            recursive_candidates = sorted(
                [p for p in anim_dir.rglob("*.fbx") if p.is_file()],
                key=lambda p: str(p.relative_to(anim_dir)),
            )
            if recursive_candidates:
                warn(
                    f"No direct *.fbx files under {anim_dir}, but recursive FBX files exist. "
                    "Using recursive candidates. In the future, pass --recursive_anim explicitly."
                )
                candidates = recursive_candidates

    if not candidates:
        raise FileNotFoundError(f"No animation FBX files found under: {anim_dir}")

    if anim_name is not None and str(anim_name).strip():
        matched = filter_anim_candidates_by_name(candidates, str(anim_name))

        if not matched:
            available = [p.name for p in candidates]
            raise FileNotFoundError(
                f"No animation FBX matched --anim_name={anim_name!r} under {anim_dir}.\n"
                "Available animation FBXs:\n"
                + "\n".join(f"  - {name}" for name in available)
            )

        if len(matched) > 1:
            raise RuntimeError(
                f"Ambiguous --anim_name={anim_name!r}; matched multiple files:\n"
                + "\n".join(f"  - {p}" for p in matched)
            )

        selected = matched[0].resolve(strict=False)
        log(f"Selected animation by --anim_name={anim_name!r}: {selected}")
        return [selected]

    start = max(0, int(start_anim_index))
    selected = candidates[start:]

    if int(max_anims) > 0:
        selected = selected[: int(max_anims)]

    return [p.resolve(strict=False) for p in selected]


def get_object_names() -> set[str]:
    return {obj.name for obj in bpy.data.objects}


def get_action_names() -> set[str]:
    return {action.name for action in bpy.data.actions}


def import_fbx_file(
    *,
    fbx_path: Path,
    import_anim: bool,
    global_scale: float,
    axis_forward: str,
    axis_up: str,
    ignore_leaf_bones: bool,
    automatic_bone_orientation: bool,
    use_prepost_rot: bool,
) -> dict[str, Any]:
    before_objects = get_object_names()
    before_actions = get_action_names()

    kwargs: dict[str, Any] = {
        "filepath": str(fbx_path),
        "global_scale": float(global_scale),
        "axis_forward": str(axis_forward),
        "axis_up": str(axis_up),
        "use_anim": bool(import_anim),
        "ignore_leaf_bones": bool(ignore_leaf_bones),
        "automatic_bone_orientation": bool(automatic_bone_orientation),
        "use_prepost_rot": bool(use_prepost_rot),
    }

    log(
        "Importing FBX: "
        f"path={fbx_path}, import_anim={import_anim}, "
        f"global_scale={global_scale}, axis_forward={axis_forward}, axis_up={axis_up}"
    )

    t0 = time.perf_counter()
    used_kwargs = "full"

    try:
        result = bpy.ops.import_scene.fbx(**kwargs)
    except TypeError as exc:
        warn(f"FBX importer rejected full kwargs: {exc}")
        warn("Retrying with reduced FBX import kwargs.")
        used_kwargs = "reduced"

        reduced_kwargs = {
            "filepath": str(fbx_path),
            "global_scale": float(global_scale),
            "axis_forward": str(axis_forward),
            "axis_up": str(axis_up),
            "use_anim": bool(import_anim),
        }

        try:
            result = bpy.ops.import_scene.fbx(**reduced_kwargs)
        except TypeError as exc2:
            warn(f"FBX importer rejected reduced kwargs: {exc2}")
            warn("Retrying with filepath only.")
            used_kwargs = "filepath_only"
            result = bpy.ops.import_scene.fbx(filepath=str(fbx_path))

    elapsed = time.perf_counter() - t0

    if result != {"FINISHED"}:
        raise RuntimeError(f"FBX import failed for {fbx_path}: result={result}")

    after_objects = get_object_names()
    after_actions = get_action_names()

    new_objects = sorted(after_objects - before_objects)
    new_actions = sorted(after_actions - before_actions)

    log(
        f"Imported FBX done: {fbx_path.name} | "
        f"new_objects={len(new_objects)}, new_actions={len(new_actions)}, elapsed={elapsed:.3f}s"
    )

    return {
        "fbx_path": str(fbx_path),
        "import_anim": bool(import_anim),
        "status": "ok",
        "elapsed_seconds": float(elapsed),
        "used_kwargs": used_kwargs,
        "new_object_count": int(len(new_objects)),
        "new_action_count": int(len(new_actions)),
        "new_objects": new_objects,
        "new_actions": new_actions,
    }


def safe_get_object(name: str) -> bpy.types.Object | None:
    try:
        return bpy.data.objects.get(name)
    except ReferenceError:
        return None


def summarize_armature_by_name(name: str) -> dict[str, Any]:
    obj = safe_get_object(name)
    if obj is None:
        return {"name": name, "status": "missing_or_removed"}

    if obj.type != "ARMATURE":
        return {"name": name, "status": "not_armature", "type": obj.type}

    try:
        bone_names = [bone.name for bone in obj.data.bones]
        root_bones = [bone.name for bone in obj.data.bones if bone.parent is None]
        parent_names = [bone.parent.name if bone.parent is not None else None for bone in obj.data.bones]

        return {
            "name": obj.name,
            "status": "ok",
            "type": obj.type,
            "data_name": obj.data.name,
            "bone_count": int(len(bone_names)),
            "root_bones": root_bones,
            "bone_names_first_100": bone_names[:100],
            "parent_names_first_100": parent_names[:100],
            "location": safe_float_list(obj.location),
            "rotation_euler": safe_float_list(obj.rotation_euler),
            "scale": safe_float_list(obj.scale),
            "pose_position": str(getattr(obj.data, "pose_position", "")),
            "has_animation_data": bool(obj.animation_data is not None),
            "active_action": (
                obj.animation_data.action.name
                if obj.animation_data is not None and obj.animation_data.action is not None
                else None
            ),
            "nla_track_count": int(len(obj.animation_data.nla_tracks)) if obj.animation_data is not None else 0,
        }
    except ReferenceError:
        return {"name": name, "status": "removed_reference_error"}


def summarize_mesh_by_name(name: str) -> dict[str, Any]:
    obj = safe_get_object(name)
    if obj is None:
        return {"name": name, "status": "missing_or_removed"}

    if obj.type != "MESH":
        return {"name": name, "status": "not_mesh", "type": obj.type}

    try:
        armature_modifiers = []
        for mod in obj.modifiers:
            if mod.type != "ARMATURE":
                continue
            armature_modifiers.append(
                {
                    "name": mod.name,
                    "object": getattr(getattr(mod, "object", None), "name", None),
                    "use_vertex_groups": bool(getattr(mod, "use_vertex_groups", False)),
                    "use_bone_envelopes": bool(getattr(mod, "use_bone_envelopes", False)),
                }
            )

        return {
            "name": obj.name,
            "status": "ok",
            "type": obj.type,
            "data_name": obj.data.name,
            "vertices": int(len(obj.data.vertices)),
            "faces": int(len(obj.data.polygons)),
            "vertex_groups": int(len(obj.vertex_groups)),
            "material_slots": int(len(obj.material_slots)),
            "armature_modifiers": armature_modifiers,
            "location": safe_float_list(obj.location),
            "rotation_euler": safe_float_list(obj.rotation_euler),
            "scale": safe_float_list(obj.scale),
        }
    except ReferenceError:
        return {"name": name, "status": "removed_reference_error"}


def extract_bone_names_from_action(action: bpy.types.Action) -> set[str]:
    out: set[str] = set()

    try:
        fcurves = list(action.fcurves)
    except ReferenceError:
        return out

    for fcurve in fcurves:
        data_path = str(fcurve.data_path)
        match = BONE_FCURVE_RE.search(data_path)
        if not match:
            continue

        bone_name = match.group(1) or match.group(2)
        if bone_name:
            out.add(str(bone_name))

    return out


def summarize_action_by_name(name: str) -> dict[str, Any]:
    action = bpy.data.actions.get(name)
    if action is None:
        return {"name": name, "status": "missing"}

    try:
        frame_start, frame_end = action.frame_range
        animated_bones = sorted(extract_bone_names_from_action(action))

        keyframe_count = 0
        data_paths: set[str] = set()
        has_scale_fcurve = False
        has_object_transform_fcurve = False

        for fcurve in action.fcurves:
            data_path = str(fcurve.data_path)
            data_paths.add(data_path)
            keyframe_count += len(fcurve.keyframe_points)

            if data_path == "scale" or data_path.endswith(".scale"):
                has_scale_fcurve = True

            if data_path in {"location", "rotation_euler", "rotation_quaternion", "scale"}:
                has_object_transform_fcurve = True

        return {
            "name": action.name,
            "status": "ok",
            "frame_range": [float(frame_start), float(frame_end)],
            "fcurve_count": int(len(action.fcurves)),
            "keyframe_count": int(keyframe_count),
            "animated_bone_count": int(len(animated_bones)),
            "animated_bones_first_100": animated_bones[:100],
            "has_scale_fcurve": bool(has_scale_fcurve),
            "has_object_transform_fcurve": bool(has_object_transform_fcurve),
            "data_paths_first_100": sorted(data_paths)[:100],
            "use_fake_user": bool(getattr(action, "use_fake_user", False)),
        }
    except ReferenceError:
        return {"name": name, "status": "removed_reference_error"}


def summarize_import_item(import_item: dict[str, Any]) -> dict[str, Any]:
    object_names = list(import_item.get("new_objects", []))
    armatures = []
    meshes = []
    others = []

    for name in object_names:
        obj = safe_get_object(name)
        if obj is None:
            others.append({"name": name, "status": "missing_or_removed"})
            continue

        if obj.type == "ARMATURE":
            armatures.append(summarize_armature_by_name(name))
        elif obj.type == "MESH":
            meshes.append(summarize_mesh_by_name(name))
        else:
            others.append(
                {
                    "name": obj.name,
                    "status": "ok",
                    "type": obj.type,
                    "location": safe_float_list(obj.location),
                    "rotation_euler": safe_float_list(obj.rotation_euler),
                    "scale": safe_float_list(obj.scale),
                }
            )

    actions = [summarize_action_by_name(action_name) for action_name in import_item.get("new_actions", [])]

    out = dict(import_item)
    out["armatures"] = armatures
    out["meshes"] = meshes
    out["others"] = others
    out["actions"] = actions
    return out


def set_fake_user_for_actions() -> None:
    for action in bpy.data.actions:
        try:
            action.use_fake_user = True
        except Exception:
            pass


def setup_debug_camera_and_light() -> dict[str, Any]:
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        return {"status": "no_meshes"}

    points: list[Vector] = []
    for obj in meshes:
        try:
            points.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
        except Exception:
            pass

    if not points:
        return {"status": "no_bbox_points"}

    mins = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maxs = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))

    center = (mins + maxs) * 0.5
    dims = maxs - mins
    radius = max(float(dims.x), float(dims.y), float(dims.z), 1.0)

    cam_data = bpy.data.cameras.new("WS_Debug_Camera")
    cam_obj = bpy.data.objects.new("WS_Debug_Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = center + Vector((0.0, -2.8 * radius, 1.3 * radius))
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = 35.0
    cam_data.clip_end = max(10000.0, radius * 20.0)

    bpy.context.scene.camera = cam_obj

    light_data = bpy.data.lights.new("WS_Debug_Key_Light", "SUN")
    light_obj = bpy.data.objects.new("WS_Debug_Key_Light", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (0.8, 0.0, 0.6)
    light_data.energy = 2.5

    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100

    return {
        "status": "ok",
        "camera": cam_obj.name,
        "light": light_obj.name,
        "bbox_min": [float(mins.x), float(mins.y), float(mins.z)],
        "bbox_max": [float(maxs.x), float(maxs.y), float(maxs.z)],
        "bbox_radius": float(radius),
    }


def set_scene_frame_range_from_actions() -> dict[str, Any]:
    if not bpy.data.actions:
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = 1
        return {"status": "no_actions", "frame_start": 1, "frame_end": 1}

    start = None
    end = None

    for action in bpy.data.actions:
        try:
            a, b = action.frame_range
        except Exception:
            continue

        a_i = int(a)
        b_i = int(b)

        start = a_i if start is None else min(start, a_i)
        end = b_i if end is None else max(end, b_i)

    if start is None or end is None or end < start:
        start = 1
        end = 1

    bpy.context.scene.frame_start = int(start)
    bpy.context.scene.frame_end = int(end)

    return {"status": "ok", "frame_start": int(start), "frame_end": int(end)}


def write_text_block(
    *,
    output_blend: Path,
    report_path: Path,
    model_fbx: Path,
    anim_fbxs: list[Path],
) -> None:
    text = bpy.data.texts.new("WINTER_SAMURAI_RAW_FBX_IMPORT_NOTES")
    text.write("Winter Samurai raw FBX import diagnostics\n")
    text.write("=" * 80 + "\n")
    text.write(f"output_blend: {output_blend}\n")
    text.write(f"report_json : {report_path}\n")
    text.write(f"model_fbx   : {model_fbx}\n")
    text.write(f"anim_count  : {len(anim_fbxs)}\n")
    for anim_fbx in anim_fbxs:
        text.write(f"  anim_fbx  : {anim_fbx}\n")
    text.write("\n")
    text.write("Important:\n")
    text.write("  - This blend intentionally keeps imported model and animation FBXs as-is.\n")
    text.write("  - No retargeting was attempted.\n")
    text.write("  - No action was copied to the model armature.\n")
    text.write("  - Animation FBX armatures were kept for later probe analysis.\n")
    text.write("\n")
    text.write("Manual visual check:\n")
    text.write("  1. Open this blend in Blender GUI.\n")
    text.write("  2. Inspect imported model armature and mesh.\n")
    text.write("  3. Inspect imported animation armature/action.\n")
    text.write("  4. Use import_report.json/action_inventory.json for names and frame ranges.\n")


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def collect_scene_summary() -> dict[str, Any]:
    object_names = sorted([obj.name for obj in bpy.data.objects])
    armature_names = sorted([obj.name for obj in bpy.data.objects if obj.type == "ARMATURE"])
    mesh_names = sorted([obj.name for obj in bpy.data.objects if obj.type == "MESH"])
    action_names = sorted([action.name for action in bpy.data.actions])

    return {
        "object_count": int(len(object_names)),
        "armature_count": int(len(armature_names)),
        "mesh_count": int(len(mesh_names)),
        "action_count": int(len(action_names)),
        "objects_first_200": object_names[:200],
        "armatures": [summarize_armature_by_name(name) for name in armature_names],
        "meshes": [summarize_mesh_by_name(name) for name in mesh_names],
        "actions": [summarize_action_by_name(name) for name in action_names],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Winter Samurai raw FBX import diagnostics: import model and selected animation FBX(s) as-is."
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/workspace/ogd_dataset/Winter_Samurai",
        help="Dataset folder containing model/ and anim/.",
    )
    parser.add_argument(
        "--model_fbx",
        type=str,
        default=None,
        help="Optional model FBX path. Relative paths are resolved under dataset_dir. If omitted, discover model/*.fbx.",
    )
    parser.add_argument(
        "--anim_dir",
        type=str,
        default=None,
        help="Optional animation FBX directory. Relative paths are resolved under dataset_dir. Default: dataset_dir/anim.",
    )
    parser.add_argument(
        "--anim_fbx",
        type=str,
        default=None,
        help="Import exactly one animation FBX. Relative paths are resolved under dataset_dir.",
    )
    parser.add_argument(
        "--anim_name",
        type=str,
        default=None,
        help="Import exactly one animation FBX by filename stem or filename.",
    )
    parser.add_argument(
        "--recursive_anim",
        action="store_true",
        help="Search animation FBXs recursively under anim_dir.",
    )
    parser.add_argument(
        "--start_anim_index",
        type=int,
        default=0,
        help="Skip the first N animation FBXs. Useful for isolating bad files.",
    )
    parser.add_argument(
        "--max_anims",
        type=int,
        default=-1,
        help="Import at most N animation FBXs. <=0 means all selected animations.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output diagnostics directory.",
    )
    parser.add_argument(
        "--output_blend",
        type=str,
        default=None,
        help="Output .blend path. Default: <output_dir>/Winter_Samurai_imported.blend",
    )

    parser.add_argument("--global_scale", type=float, default=1.0)
    parser.add_argument("--axis_forward", type=str, default="-Z")
    parser.add_argument("--axis_up", type=str, default="Y")

    parser.add_argument("--no_ignore_leaf_bones", dest="ignore_leaf_bones", action="store_false")
    parser.set_defaults(ignore_leaf_bones=True)

    parser.add_argument("--automatic_bone_orientation", action="store_true")

    parser.add_argument("--no_use_prepost_rot", dest="use_prepost_rot", action="store_false")
    parser.set_defaults(use_prepost_rot=True)

    parser.add_argument(
        "--import_model_animation",
        action="store_true",
        help="Import animation stored in the model FBX. Default false.",
    )
    parser.add_argument(
        "--no_import_anims",
        dest="import_anims",
        action="store_false",
        help="Only import model FBX, skip anim/*.fbx.",
    )
    parser.set_defaults(import_anims=True)

    parser.add_argument("--no_clean_scene", dest="clean_scene", action="store_false")
    parser.set_defaults(clean_scene=True)

    parser.add_argument("--no_debug_camera", dest="setup_debug_camera", action="store_false")
    parser.set_defaults(setup_debug_camera=True)

    args = parser.parse_args(script_argv())

    t0 = time.perf_counter()

    repo_root = resolve_repo_root()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_blend = (
        Path(args.output_blend).expanduser().resolve(strict=False)
        if args.output_blend
        else (output_dir / "Winter_Samurai_imported.blend").resolve(strict=False)
    )

    report_path = output_dir / "import_report.json"
    action_inventory_path = output_dir / "action_inventory.json"

    model_fbx = discover_model_fbx(dataset_dir, args.model_fbx)

    anim_fbxs: list[Path] = []
    if bool(args.import_anims):
        anim_fbxs = discover_anim_fbxs(
            dataset_dir=dataset_dir,
            explicit_anim_dir=args.anim_dir,
            explicit_anim_fbx=args.anim_fbx,
            anim_name=args.anim_name,
            recursive=bool(args.recursive_anim),
            start_anim_index=int(args.start_anim_index),
            max_anims=int(args.max_anims),
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 96)
    log("Winter Samurai raw FBX import diagnostics")
    log(f"repo_root       : {repo_root}")
    log(f"dataset_dir     : {dataset_dir}")
    log(f"model_fbx       : {model_fbx}")
    log(f"anim_count      : {len(anim_fbxs)}")
    for anim_path in anim_fbxs:
        log(f"  anim_fbx      : {anim_path}")
    log(f"output_dir      : {output_dir}")
    log(f"output_blend    : {output_blend}")
    log(f"report_path     : {report_path}")
    log("=" * 96)

    ensure_fbx_importer_available()

    if bool(args.clean_scene):
        clear_scene()

    import_items: list[dict[str, Any]] = []

    model_import = import_fbx_file(
        fbx_path=model_fbx,
        import_anim=bool(args.import_model_animation),
        global_scale=float(args.global_scale),
        axis_forward=str(args.axis_forward),
        axis_up=str(args.axis_up),
        ignore_leaf_bones=bool(args.ignore_leaf_bones),
        automatic_bone_orientation=bool(args.automatic_bone_orientation),
        use_prepost_rot=bool(args.use_prepost_rot),
    )
    model_import["kind"] = "model"
    model_import["source_index"] = 0
    import_items.append(model_import)

    anim_imports: list[dict[str, Any]] = []

    for idx, anim_fbx in enumerate(anim_fbxs):
        try:
            item = import_fbx_file(
                fbx_path=anim_fbx,
                import_anim=True,
                global_scale=float(args.global_scale),
                axis_forward=str(args.axis_forward),
                axis_up=str(args.axis_up),
                ignore_leaf_bones=bool(args.ignore_leaf_bones),
                automatic_bone_orientation=bool(args.automatic_bone_orientation),
                use_prepost_rot=bool(args.use_prepost_rot),
            )
            item["kind"] = "animation"
            item["source_index"] = int(idx)
            item["selected_anim_index"] = int(idx + max(0, int(args.start_anim_index)))
            anim_imports.append(item)
            import_items.append(item)

        except Exception as exc:
            warn(f"Animation FBX import failed: {anim_fbx}: {exc}")
            item = {
                "kind": "animation",
                "source_index": int(idx),
                "selected_anim_index": int(idx + max(0, int(args.start_anim_index))),
                "fbx_path": str(anim_fbx),
                "import_anim": True,
                "status": "import_failed",
                "error": str(exc),
                "new_object_count": 0,
                "new_action_count": 0,
                "new_objects": [],
                "new_actions": [],
            }
            anim_imports.append(item)
            import_items.append(item)

    set_fake_user_for_actions()
    frame_range_summary = set_scene_frame_range_from_actions()

    camera_summary: dict[str, Any] = {"enabled": bool(args.setup_debug_camera), "status": "skipped"}
    if bool(args.setup_debug_camera):
        camera_summary = setup_debug_camera_and_light()
        camera_summary["enabled"] = True

    try:
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE":
                obj.data.pose_position = "POSE"
    except Exception:
        pass

    bpy.context.view_layer.update()

    write_text_block(
        output_blend=output_blend,
        report_path=report_path,
        model_fbx=model_fbx,
        anim_fbxs=anim_fbxs,
    )

    output_blend.parent.mkdir(parents=True, exist_ok=True)

    save_result = bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    if save_result != {"FINISHED"}:
        raise RuntimeError(f"Failed to save output blend: {output_blend}, result={save_result}")

    elapsed = time.perf_counter() - t0

    summarized_imports = [summarize_import_item(item) for item in import_items]
    scene_summary = collect_scene_summary()

    report = {
        "schema": "ogd_lbs_winter_samurai_raw_fbx_import_v3",
        "created_at_utc": utc_now_iso(),
        "elapsed_seconds": float(elapsed),
        "repo_root": str(repo_root) if repo_root is not None else None,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "output_blend": str(output_blend),
        "model_fbx": str(model_fbx),
        "anim_fbxs": [str(p) for p in anim_fbxs],
        "config": {
            "global_scale": float(args.global_scale),
            "axis_forward": str(args.axis_forward),
            "axis_up": str(args.axis_up),
            "ignore_leaf_bones": bool(args.ignore_leaf_bones),
            "automatic_bone_orientation": bool(args.automatic_bone_orientation),
            "use_prepost_rot": bool(args.use_prepost_rot),
            "import_model_animation": bool(args.import_model_animation),
            "import_anims": bool(args.import_anims),
            "anim_fbx": str(args.anim_fbx) if args.anim_fbx else None,
            "anim_name": str(args.anim_name) if args.anim_name else None,
            "start_anim_index": int(args.start_anim_index),
            "max_anims": int(args.max_anims),
            "clean_scene": bool(args.clean_scene),
        },
        "imports": summarized_imports,
        "model_import": summarize_import_item(model_import),
        "animation_imports": [summarize_import_item(item) for item in anim_imports],
        "scene": scene_summary,
        "frame_range": frame_range_summary,
        "debug_scene": camera_summary,
        "manual_visual_check": {
            "note": (
                "This blend keeps model and selected animation FBXs imported as-is. "
                "No retargeting, action copy, NLA concatenation, or cleanup was attempted."
            ),
            "open_blend": str(output_blend),
        },
    }

    action_inventory = {
        "schema": "ogd_lbs_winter_samurai_action_inventory_v3",
        "output_blend": str(output_blend),
        "action_count": int(len(bpy.data.actions)),
        "actions": [
            summarize_action_by_name(action.name)
            for action in sorted(bpy.data.actions, key=lambda a: a.name)
        ],
    }

    dump_json(report_path, report)
    dump_json(action_inventory_path, action_inventory)

    log("\n" + "=" * 96)
    log("Winter Samurai raw FBX import diagnostics complete.")
    log(f"output_blend        : {output_blend}")
    log(f"import_report       : {report_path}")
    log(f"action_inventory    : {action_inventory_path}")
    log(f"scene_armatures     : {scene_summary['armature_count']}")
    log(f"scene_meshes        : {scene_summary['mesh_count']}")
    log(f"scene_actions       : {scene_summary['action_count']}")
    log(f"elapsed_seconds     : {elapsed:.3f}")
    log("=" * 96)


if __name__ == "__main__":
    main()
