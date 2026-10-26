#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Winter Samurai custom clean Step 01.

Scope:
    - import model FBX
    - import selected animation FBXs
    - remove EMPTY objects while preserving surviving objects' current world matrices
    - save clean01 blend
    - write import_report.json and action_inventory.json

Generic operations are delegated to:
    ogd_lbs.blender.fbx_io
    ogd_lbs.blender.object_cleanup
"""

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Vector

# -----------------------------------------------------------------------------
# Standalone script bootstrap:
# Blender may not honor PYTHONPATH by default. Add <repo>/src before importing
# ogd_lbs so this script can run via:
#   blender -b --python this_script.py -- ...
# -----------------------------------------------------------------------------
def _ensure_repo_src_on_syspath() -> None:
    current = Path(__file__).expanduser().resolve(strict=False)
    for parent in [current.parent, *current.parents]:
        src = parent / "src"
        if (parent / "pyproject.toml").exists() and (src / "ogd_lbs").is_dir():
            src_str = str(src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return

_ensure_repo_src_on_syspath()
# -----------------------------------------------------------------------------

from ogd_lbs.blender.context import update_view_layer
from ogd_lbs.blender.object import ensure_object_mode
from ogd_lbs.blender.fbx_io import ensure_fbx_importer_available, import_fbx_with_fallback
from ogd_lbs.blender.object_cleanup import (
    get_object_animation_info,
    remove_empty_objects_preserve_children,
)
from ogd_lbs.blender.action_utils import iter_action_fcurves


TAG = "[ws-custom-clean-01]"
BONE_FCURVE_RE = re.compile(r"""pose\.bones\["([^"]+)"\]|pose\.bones\['([^']+)'\]""")


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


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def matrix_to_list(mat: Matrix) -> list[list[float]]:
    return [[float(mat[r][c]) for c in range(4)] for r in range(4)]


def safe_float_list(values: Any) -> list[float]:
    try:
        return [float(x) for x in values]
    except Exception:
        return []


def clear_scene() -> None:
    ensure_object_mode()

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

    update_view_layer()


def resolve_maybe_relative(path_text: str | None, base_dir: Path) -> Path | None:
    if path_text is None or not str(path_text).strip():
        return None

    raw = Path(str(path_text).strip()).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)

    return (base_dir / raw).resolve(strict=False)


def flatten_cli_items(items: list[str] | None) -> list[str]:
    out: list[str] = []

    for item in items or []:
        for chunk in str(item).split(","):
            text = chunk.strip()
            if text:
                out.append(text)

    return out


def dedupe_paths_keep_order(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        resolved = path.resolve(strict=False)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)

    return out


def discover_model_fbx(dataset_dir: Path, explicit_model_fbx: str | None) -> Path:
    explicit = resolve_maybe_relative(explicit_model_fbx, dataset_dir)

    if explicit is not None:
        require(explicit.is_file(), f"Explicit model FBX not found: {explicit}")
        require(explicit.suffix.lower() == ".fbx", f"Explicit model path is not FBX: {explicit}")
        return explicit

    model_dir = dataset_dir / "model"
    require(model_dir.is_dir(), f"Missing model directory: {model_dir}")

    preferred = model_dir / "Base_Mesh_Rig.fbx"
    if preferred.is_file():
        return preferred.resolve(strict=False)

    candidates = sorted([p for p in model_dir.glob("*.fbx") if p.is_file()], key=lambda p: p.name)
    require(candidates, f"No model FBX found under: {model_dir}")
    require(
        len(candidates) == 1,
        "Expected exactly one model FBX or Base_Mesh_Rig.fbx. "
        f"Candidates={[str(p) for p in candidates]}",
    )
    return candidates[0].resolve(strict=False)


def discover_rest_pose_base(dataset_dir: Path, require_file: bool) -> Path | None:
    path = (dataset_dir / "model" / "rest_pose_base.blend").resolve(strict=False)

    if path.is_file():
        return path

    if require_file:
        raise FileNotFoundError(f"Missing required BetterFBX rest pose base blend: {path}")

    warn(f"rest_pose_base.blend not found; continuing because requirement disabled: {path}")
    return None


def discover_all_anim_candidates(
    *,
    dataset_dir: Path,
    explicit_anim_dir: str | None,
    recursive: bool,
) -> tuple[Path, list[Path]]:
    anim_dir = resolve_maybe_relative(explicit_anim_dir, dataset_dir)
    if anim_dir is None:
        anim_dir = dataset_dir / "anim"

    require(anim_dir.is_dir(), f"Missing animation directory: {anim_dir}")

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
                warn(f"No direct *.fbx under {anim_dir}; using recursive candidates.")
                candidates = recursive_candidates

    require(candidates, f"No animation FBX files found under: {anim_dir}")

    return anim_dir.resolve(strict=False), [p.resolve(strict=False) for p in candidates]


def match_anim_candidate_by_name(candidates: list[Path], selector: str) -> Path:
    selector = str(selector).strip()
    require(selector, "Empty animation selector.")

    selector_path = Path(selector)
    selector_name = selector_path.name
    selector_stem = selector_path.stem if selector_name.lower().endswith(".fbx") else selector

    exact = [p for p in candidates if p.name == selector_name or p.stem == selector_stem]
    if len(exact) == 1:
        return exact[0].resolve(strict=False)

    require(len(exact) <= 1, f"Ambiguous --anim_name={selector!r}: {[str(p) for p in exact]}")

    selector_name_l = selector_name.lower()
    selector_stem_l = selector_stem.lower()

    ci = [
        p for p in candidates
        if p.name.lower() == selector_name_l or p.stem.lower() == selector_stem_l
    ]

    if len(ci) == 1:
        return ci[0].resolve(strict=False)

    require(len(ci) <= 1, f"Ambiguous case-insensitive --anim_name={selector!r}: {[str(p) for p in ci]}")

    raise FileNotFoundError(
        f"No animation FBX matched --anim_name={selector!r}.\n"
        "Available animation FBXs:\n"
        + "\n".join(f"  - {p.name}" for p in candidates)
    )


def discover_anim_fbxs(
    *,
    dataset_dir: Path,
    explicit_anim_dir: str | None,
    explicit_anim_fbx_items: list[str] | None,
    anim_name_items: list[str] | None,
    recursive: bool,
    start_anim_index: int,
    max_anims: int,
) -> dict[str, Any]:
    anim_dir, candidates = discover_all_anim_candidates(
        dataset_dir=dataset_dir,
        explicit_anim_dir=explicit_anim_dir,
        recursive=recursive,
    )

    explicit_anim_fbxs = flatten_cli_items(explicit_anim_fbx_items)
    anim_names = flatten_cli_items(anim_name_items)

    selected: list[Path] = []
    selection_mode = "auto_discover_all_or_slice"

    if explicit_anim_fbxs or anim_names:
        selection_mode = "explicit_anim_fbx_and_or_anim_name"

        for item in explicit_anim_fbxs:
            path = resolve_maybe_relative(item, dataset_dir)
            if path is None:
                continue
            require(path.is_file(), f"Explicit animation FBX not found: {path}")
            require(path.suffix.lower() == ".fbx", f"Explicit animation path is not FBX: {path}")
            selected.append(path.resolve(strict=False))

        for name in anim_names:
            selected.append(match_anim_candidate_by_name(candidates, name))

        selected = dedupe_paths_keep_order(selected)

    else:
        start = max(0, int(start_anim_index))
        selected = candidates[start:]
        if int(max_anims) > 0:
            selected = selected[: int(max_anims)]
        selected = dedupe_paths_keep_order(selected)

    require(selected, "No animation FBX selected.")

    return {
        "anim_dir": str(anim_dir),
        "candidate_count": int(len(candidates)),
        "candidates": [str(p) for p in candidates],
        "selection_mode": selection_mode,
        "selected_count": int(len(selected)),
        "selected": [str(p) for p in selected],
        "selected_paths": selected,
    }


def extract_bone_names_from_action(action: bpy.types.Action) -> set[str]:
    out: set[str] = set()

    for fcurve in iter_action_fcurves(action):
        match = BONE_FCURVE_RE.search(str(fcurve.data_path))
        if match:
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
        curves = iter_action_fcurves(action)
        animated_bones = sorted(extract_bone_names_from_action(action))

        keyframe_count = 0
        data_paths: set[str] = set()
        has_scale_fcurve = False
        has_object_transform_fcurve = False

        for fcurve in curves:
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
            "fcurve_count": int(len(curves)),
            "keyframe_count": int(keyframe_count),
            "animated_bone_count": int(len(animated_bones)),
            "animated_bones_first_100": animated_bones[:100],
            "has_scale_fcurve": bool(has_scale_fcurve),
            "has_object_transform_fcurve": bool(has_object_transform_fcurve),
            "data_paths_first_100": sorted(data_paths)[:100],
            "use_fake_user": bool(getattr(action, "use_fake_user", False)),
            "users": int(getattr(action, "users", 0)),
        }

    except ReferenceError:
        return {"name": name, "status": "removed_reference_error"}


def summarize_armature(obj: bpy.types.Object) -> dict[str, Any]:
    return {
        "name": obj.name,
        "data_name": obj.data.name,
        "bone_count": int(len(obj.data.bones)),
        "root_bones": [bone.name for bone in obj.data.bones if bone.parent is None],
        "bone_names_first_100": [bone.name for bone in obj.data.bones][:100],
        "matrix_world": matrix_to_list(obj.matrix_world),
        "object_animation": get_object_animation_info(obj),
        "data_animation": get_object_animation_info(obj.data),
    }


def summarize_mesh(obj: bpy.types.Object) -> dict[str, Any]:
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
        "data_name": obj.data.name,
        "vertices": int(len(obj.data.vertices)),
        "faces": int(len(obj.data.polygons)),
        "vertex_groups": int(len(obj.vertex_groups)),
        "parent": obj.parent.name if obj.parent is not None else None,
        "armature_modifiers": armature_modifiers,
        "location": safe_float_list(obj.location),
        "rotation_euler": safe_float_list(obj.rotation_euler),
        "scale": safe_float_list(obj.scale),
        "matrix_world": matrix_to_list(obj.matrix_world),
    }


def summarize_object_by_name(name: str) -> dict[str, Any]:
    obj = bpy.data.objects.get(name)
    if obj is None:
        return {"name": name, "status": "missing"}

    if obj.type == "ARMATURE":
        return {"name": name, "type": obj.type, **summarize_armature(obj)}

    if obj.type == "MESH":
        return {"name": name, "type": obj.type, **summarize_mesh(obj)}

    return {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent is not None else None,
        "children": [child.name for child in obj.children],
        "location": safe_float_list(obj.location),
        "rotation_euler": safe_float_list(obj.rotation_euler),
        "scale": safe_float_list(obj.scale),
        "matrix_world": matrix_to_list(obj.matrix_world),
        "object_animation": get_object_animation_info(obj),
    }


def summarize_import_item(item: dict[str, Any]) -> dict[str, Any]:
    object_names = list(item.get("new_objects", []))
    armatures = []
    meshes = []
    empties = []
    others = []

    for name in object_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            others.append({"name": name, "status": "missing_or_removed"})
            continue

        if obj.type == "ARMATURE":
            armatures.append(summarize_object_by_name(name))
        elif obj.type == "MESH":
            meshes.append(summarize_object_by_name(name))
        elif obj.type == "EMPTY":
            empties.append(summarize_object_by_name(name))
        else:
            others.append(summarize_object_by_name(name))

    out = dict(item)
    out["armatures"] = armatures
    out["meshes"] = meshes
    out["empties"] = empties
    out["others"] = others
    out["actions"] = [summarize_action_by_name(action_name) for action_name in item.get("new_actions", [])]
    return out


def set_fake_user_for_actions() -> None:
    for action in bpy.data.actions:
        try:
            action.use_fake_user = True
        except Exception:
            pass


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
        start, end = 1, 1

    bpy.context.scene.frame_start = int(start)
    bpy.context.scene.frame_end = int(end)
    bpy.context.scene.frame_set(int(start))
    update_view_layer()

    return {"status": "ok", "frame_start": int(start), "frame_end": int(end)}


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

    cam_data = bpy.data.cameras.new("WS_Clean01_Debug_Camera")
    cam_obj = bpy.data.objects.new("WS_Clean01_Debug_Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = center + Vector((0.0, -2.8 * radius, 1.3 * radius))
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = 35.0
    cam_data.clip_end = max(10000.0, radius * 20.0)
    bpy.context.scene.camera = cam_obj

    light_data = bpy.data.lights.new("WS_Clean01_Debug_Key_Light", "SUN")
    light_obj = bpy.data.objects.new("WS_Clean01_Debug_Key_Light", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (0.8, 0.0, 0.6)
    light_data.energy = 2.5

    return {
        "status": "ok",
        "camera": cam_obj.name,
        "light": light_obj.name,
        "bbox_min": [float(mins.x), float(mins.y), float(mins.z)],
        "bbox_max": [float(maxs.x), float(maxs.y), float(maxs.z)],
        "bbox_radius": float(radius),
    }


def collect_scene_summary() -> dict[str, Any]:
    objects = list(bpy.data.objects)
    armatures = sorted([o for o in objects if o.type == "ARMATURE"], key=lambda o: o.name)
    meshes = sorted([o for o in objects if o.type == "MESH"], key=lambda o: o.name)
    empties = sorted([o for o in objects if o.type == "EMPTY"], key=lambda o: o.name)
    actions = sorted(list(bpy.data.actions), key=lambda a: a.name)

    type_counts: dict[str, int] = {}
    for obj in objects:
        type_counts[obj.type] = type_counts.get(obj.type, 0) + 1

    return {
        "object_count": int(len(objects)),
        "object_type_counts": type_counts,
        "armature_count": int(len(armatures)),
        "mesh_count": int(len(meshes)),
        "empty_count": int(len(empties)),
        "action_count": int(len(actions)),
        "armature_names": [o.name for o in armatures],
        "mesh_names_first_200": [o.name for o in meshes[:200]],
        "empty_names_first_200": [o.name for o in empties[:200]],
        "action_names_first_200": [a.name for a in actions[:200]],
        "armatures": [summarize_armature(o) for o in armatures],
        "meshes_first_100": [summarize_mesh(o) for o in meshes[:100]],
        "actions": [summarize_action_by_name(a.name) for a in actions],
    }


def write_text_block(
    *,
    output_blend: Path,
    report_path: Path,
    model_fbx: Path,
    rest_pose_base: Path | None,
    anim_fbxs: list[Path],
    clean_empty_objects: bool,
) -> None:
    text = bpy.data.texts.new("WINTER_SAMURAI_CUSTOM_CLEAN01_IMPORT_NOTES")
    text.write("Winter Samurai custom clean Step 01\n")
    text.write("=" * 80 + "\n")
    text.write(f"output_blend      : {output_blend}\n")
    text.write(f"report_json       : {report_path}\n")
    text.write(f"model_fbx         : {model_fbx}\n")
    text.write(f"rest_pose_base    : {rest_pose_base}\n")
    text.write(f"anim_count        : {len(anim_fbxs)}\n")
    for anim in anim_fbxs:
        text.write(f"  anim_fbx        : {anim}\n")
    text.write("\n")
    text.write("Scope:\n")
    text.write("  - Import model FBX and selected animation FBXs.\n")
    text.write("  - Remove EMPTY objects if enabled.\n")
    text.write("  - Do not clean rest pose in Step 01.\n")
    text.write("  - rest_pose_base.blend is only checked and recorded here.\n")
    text.write(f"  - EMPTY cleanup enabled: {bool(clean_empty_objects)}\n")


def save_blend(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Saving blend: {path}")
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    require(result == {"FINISHED"}, f"save_as_mainfile failed: {result}")


def self_test() -> None:
    ensure_fbx_importer_available()
    log(f"Self-test passed. bpy={bpy.app.version_string}")
    log(f"Repo root: {resolve_repo_root()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Winter Samurai custom clean Step 01: FBX import and EMPTY cleanup."
    )

    parser.add_argument("--self_test", action="store_true")

    parser.add_argument("--dataset_dir", type=str, default="/workspace/ogd_dataset/Winter_Samurai")
    parser.add_argument("--model_fbx", type=str, default=None)
    parser.add_argument("--anim_dir", type=str, default=None)

    parser.add_argument("--anim_fbx", type=str, action="append", default=None)
    parser.add_argument("--anim_name", type=str, action="append", default=None)
    parser.add_argument("--recursive_anim", action="store_true")
    parser.add_argument("--start_anim_index", type=int, default=0)
    parser.add_argument("--max_anims", type=int, default=-1)
    parser.add_argument("--list_anims_only", action="store_true")

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_blend", type=str, default=None)

    parser.add_argument("--global_scale", type=float, default=1.0)
    parser.add_argument("--axis_forward", type=str, default="-Z")
    parser.add_argument("--axis_up", type=str, default="Y")

    parser.add_argument("--no_ignore_leaf_bones", dest="ignore_leaf_bones", action="store_false")
    parser.set_defaults(ignore_leaf_bones=True)

    parser.add_argument("--automatic_bone_orientation", action="store_true")

    parser.add_argument("--no_use_prepost_rot", dest="use_prepost_rot", action="store_false")
    parser.set_defaults(use_prepost_rot=True)

    parser.add_argument("--import_model_animation", action="store_true")
    parser.add_argument("--no_import_anims", dest="import_anims", action="store_false")
    parser.set_defaults(import_anims=True)

    parser.add_argument("--no_clean_scene", dest="clean_scene", action="store_false")
    parser.set_defaults(clean_scene=True)

    parser.add_argument("--no_clean_empty_objects", dest="clean_empty_objects", action="store_false")
    parser.set_defaults(clean_empty_objects=True)

    parser.add_argument("--fail_on_animated_empty", action="store_true")

    parser.add_argument("--no_debug_camera", dest="setup_debug_camera", action="store_false")
    parser.set_defaults(setup_debug_camera=True)

    parser.add_argument("--no_require_rest_pose_base", dest="require_rest_pose_base", action="store_false")
    parser.set_defaults(require_rest_pose_base=True)

    args = parser.parse_args(script_argv())

    if args.self_test:
        self_test()
        return

    repo_root = resolve_repo_root()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve(strict=False)

    model_fbx = discover_model_fbx(dataset_dir, args.model_fbx)
    rest_pose_base = discover_rest_pose_base(dataset_dir, bool(args.require_rest_pose_base))

    anim_selection: dict[str, Any] = {
        "anim_dir": None,
        "candidate_count": 0,
        "candidates": [],
        "selection_mode": "import_anims_disabled",
        "selected_count": 0,
        "selected": [],
        "selected_paths": [],
    }

    if bool(args.import_anims):
        anim_selection = discover_anim_fbxs(
            dataset_dir=dataset_dir,
            explicit_anim_dir=args.anim_dir,
            explicit_anim_fbx_items=args.anim_fbx,
            anim_name_items=args.anim_name,
            recursive=bool(args.recursive_anim),
            start_anim_index=int(args.start_anim_index),
            max_anims=int(args.max_anims),
        )

    anim_fbxs = list(anim_selection["selected_paths"])

    if args.list_anims_only:
        obj = {
            "schema": "ogd_lbs_winter_samurai_custom_clean01_anim_selection_v1",
            "repo_root": str(repo_root) if repo_root else None,
            "dataset_dir": str(dataset_dir),
            "model_fbx": str(model_fbx),
            "rest_pose_base": str(rest_pose_base) if rest_pose_base is not None else None,
            "rest_pose_base_exists": bool(rest_pose_base is not None),
            "anim_selection": {k: v for k, v in anim_selection.items() if k != "selected_paths"},
        }
        print(json.dumps(obj, indent=2, ensure_ascii=False), flush=True)
        return

    if args.output_dir is None or not str(args.output_dir).strip():
        parser.error("--output_dir is required unless --self_test or --list_anims_only is used.")

    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_blend = (
        Path(args.output_blend).expanduser().resolve(strict=False)
        if args.output_blend
        else (output_dir / "Winter_Samurai_clean01_imported.blend").resolve(strict=False)
    )

    report_path = output_dir / "import_report.json"
    action_inventory_path = output_dir / "action_inventory.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "ogd_lbs_winter_samurai_custom_clean01_import_v1",
        "created_at_utc": utc_now_iso(),
        "repo_root": str(repo_root) if repo_root else None,
        "dataset_dir": str(dataset_dir),
        "model_fbx": str(model_fbx),
        "rest_pose_base": str(rest_pose_base) if rest_pose_base is not None else None,
        "rest_pose_base_exists": bool(rest_pose_base is not None),
        "anim_fbxs": [str(p) for p in anim_fbxs],
        "output_dir": str(output_dir),
        "output_blend": str(output_blend),
        "report_path": str(report_path),
        "action_inventory_path": str(action_inventory_path),
        "status": "running",
        "stage": "init",
    }

    t0 = time.perf_counter()

    try:
        log("=" * 96)
        log("Winter Samurai custom clean Step 01: import + EMPTY cleanup")
        log(f"repo_root      : {repo_root}")
        log(f"dataset_dir    : {dataset_dir}")
        log(f"model_fbx      : {model_fbx}")
        log(f"rest_pose_base : {rest_pose_base}")
        log(f"anim_count     : {len(anim_fbxs)}")
        for anim_path in anim_fbxs:
            log(f"  anim_fbx     : {anim_path}")
        log(f"output_dir     : {output_dir}")
        log(f"output_blend   : {output_blend}")
        log("=" * 96)

        report["config"] = {
            "global_scale": float(args.global_scale),
            "axis_forward": str(args.axis_forward),
            "axis_up": str(args.axis_up),
            "ignore_leaf_bones": bool(args.ignore_leaf_bones),
            "automatic_bone_orientation": bool(args.automatic_bone_orientation),
            "use_prepost_rot": bool(args.use_prepost_rot),
            "import_model_animation": bool(args.import_model_animation),
            "import_anims": bool(args.import_anims),
            "anim_fbx": flatten_cli_items(args.anim_fbx),
            "anim_name": flatten_cli_items(args.anim_name),
            "recursive_anim": bool(args.recursive_anim),
            "start_anim_index": int(args.start_anim_index),
            "max_anims": int(args.max_anims),
            "clean_scene": bool(args.clean_scene),
            "clean_empty_objects": bool(args.clean_empty_objects),
            "fail_on_animated_empty": bool(args.fail_on_animated_empty),
            "setup_debug_camera": bool(args.setup_debug_camera),
            "require_rest_pose_base": bool(args.require_rest_pose_base),
            "rest_pose_scope": "rest_pose_base.blend is checked/recorded only. It is not imported or used in Step 01.",
        }
        report["anim_selection"] = {k: v for k, v in anim_selection.items() if k != "selected_paths"}

        report["stage"] = "ensure_fbx_importer"
        ensure_fbx_importer_available()

        report["stage"] = "clear_scene"
        if bool(args.clean_scene):
            clear_scene()

        import_items: list[dict[str, Any]] = []

        report["stage"] = "import_model"
        model_import = import_fbx_with_fallback(
            model_fbx,
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

        report["stage"] = "import_animations"
        for idx, anim_fbx in enumerate(anim_fbxs):
            try:
                item = import_fbx_with_fallback(
                    anim_fbx,
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
                    "traceback": traceback.format_exc(),
                    "new_object_count": 0,
                    "new_action_count": 0,
                    "new_objects": [],
                    "new_actions": [],
                }
                anim_imports.append(item)
                import_items.append(item)

        report["stage"] = "post_import_prepare"
        set_fake_user_for_actions()
        frame_range_summary = set_scene_frame_range_from_actions()
        scene_before_empty_cleanup = collect_scene_summary()

        if bool(args.clean_empty_objects):
            report["stage"] = "clean_empty_objects"
            empty_cleanup_summary = remove_empty_objects_preserve_children(
                fail_on_animated_empty=bool(args.fail_on_animated_empty)
            )
        else:
            empty_cleanup_summary = {
                "enabled": False,
                "policy": "disabled_by_cli",
                "removed_empty_count": 0,
                "remaining_empty_count": int(len([obj for obj in bpy.data.objects if obj.type == "EMPTY"])),
            }

        scene_after_empty_cleanup = collect_scene_summary()

        report["stage"] = "debug_scene"
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

        update_view_layer()
        set_fake_user_for_actions()

        write_text_block(
            output_blend=output_blend,
            report_path=report_path,
            model_fbx=model_fbx,
            rest_pose_base=rest_pose_base,
            anim_fbxs=anim_fbxs,
            clean_empty_objects=bool(args.clean_empty_objects),
        )

        report["stage"] = "save_blend"
        save_blend(output_blend)

        elapsed = time.perf_counter() - t0

        report["status"] = "passed"
        report["stage"] = "complete"
        report["elapsed_seconds"] = float(elapsed)
        report["imports"] = [summarize_import_item(item) for item in import_items]
        report["model_import"] = summarize_import_item(model_import)
        report["animation_imports"] = [summarize_import_item(item) for item in anim_imports]
        report["frame_range"] = frame_range_summary
        report["scene_before_empty_cleanup"] = scene_before_empty_cleanup
        report["empty_cleanup"] = empty_cleanup_summary
        report["scene_after_empty_cleanup"] = scene_after_empty_cleanup
        report["scene_final"] = collect_scene_summary()
        report["debug_scene"] = camera_summary
        report["scope"] = (
            "Step 01 imports model and selected animation FBXs, then removes EMPTY objects. "
            "It does not clean rest pose, rebase, retarget, transfer Actions, "
            "remove animation armatures, or remove pre-rebase Actions. "
            "rest_pose_base.blend is only checked and recorded."
        )

        action_inventory = {
            "schema": "ogd_lbs_winter_samurai_custom_clean01_action_inventory_v1",
            "output_blend": str(output_blend),
            "action_count": int(len(bpy.data.actions)),
            "actions": [summarize_action_by_name(action.name) for action in sorted(bpy.data.actions, key=lambda a: a.name)],
        }

        dump_json(report_path, report)
        dump_json(action_inventory_path, action_inventory)

        log("=" * 96)
        log("Winter Samurai custom clean Step 01 PASSED")
        log(f"output_blend     : {output_blend}")
        log(f"import_report    : {report_path}")
        log(f"action_inventory : {action_inventory_path}")
        log(f"armatures_final  : {report['scene_final']['armature_count']}")
        log(f"meshes_final     : {report['scene_final']['mesh_count']}")
        log(f"empties_final    : {report['scene_final']['empty_count']}")
        log(f"actions_final    : {report['scene_final']['action_count']}")
        log(f"elapsed_seconds  : {elapsed:.3f}")
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


if __name__ == "__main__":
    main()
