# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic FBX import helpers.

Callers must resolve dataset-specific paths themselves.
"""

import time
from pathlib import Path
from typing import Any

import bpy


def ensure_fbx_importer_available() -> None:
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_fbx")
    except Exception:
        pass

    if not (
        hasattr(bpy.ops, "import_scene")
        and hasattr(bpy.ops.import_scene, "fbx")
    ):
        raise RuntimeError("Blender FBX importer bpy.ops.import_scene.fbx is not available.")


def snapshot_object_names() -> set[str]:
    return {obj.name for obj in bpy.data.objects}


def snapshot_action_names() -> set[str]:
    return {action.name for action in bpy.data.actions}


def import_fbx_with_fallback(
    fbx_path: str | Path,
    *,
    import_anim: bool,
    global_scale: float = 1.0,
    axis_forward: str = "-Z",
    axis_up: str = "Y",
    ignore_leaf_bones: bool = True,
    automatic_bone_orientation: bool = False,
    use_prepost_rot: bool = True,
) -> dict[str, Any]:
    ensure_fbx_importer_available()

    path = Path(fbx_path).expanduser().resolve(strict=False)

    if not path.is_file():
        raise FileNotFoundError(f"FBX path not found: {path}")

    if path.suffix.lower() != ".fbx":
        raise RuntimeError(f"Expected .fbx file: {path}")

    before_objects = snapshot_object_names()
    before_actions = snapshot_action_names()

    full_kwargs: dict[str, Any] = {
        "filepath": str(path),
        "global_scale": float(global_scale),
        "axis_forward": str(axis_forward),
        "axis_up": str(axis_up),
        "use_anim": bool(import_anim),
        "ignore_leaf_bones": bool(ignore_leaf_bones),
        "automatic_bone_orientation": bool(automatic_bone_orientation),
        "use_prepost_rot": bool(use_prepost_rot),
    }

    reduced_kwargs: dict[str, Any] = {
        "filepath": str(path),
        "global_scale": float(global_scale),
        "axis_forward": str(axis_forward),
        "axis_up": str(axis_up),
        "use_anim": bool(import_anim),
    }

    used_kwargs = "full"
    t0 = time.perf_counter()

    try:
        result = bpy.ops.import_scene.fbx(**full_kwargs)
    except TypeError:
        used_kwargs = "reduced"
        try:
            result = bpy.ops.import_scene.fbx(**reduced_kwargs)
        except TypeError:
            used_kwargs = "filepath_only"
            result = bpy.ops.import_scene.fbx(filepath=str(path))

    elapsed = time.perf_counter() - t0

    if result != {"FINISHED"}:
        raise RuntimeError(f"FBX import failed for {path}: result={result}")

    new_objects = sorted(snapshot_object_names() - before_objects)
    new_actions = sorted(snapshot_action_names() - before_actions)

    return {
        "fbx_path": str(path),
        "import_anim": bool(import_anim),
        "status": "ok",
        "elapsed_seconds": float(elapsed),
        "used_kwargs": used_kwargs,
        "new_object_count": int(len(new_objects)),
        "new_action_count": int(len(new_actions)),
        "new_objects": new_objects,
        "new_actions": new_actions,
    }
