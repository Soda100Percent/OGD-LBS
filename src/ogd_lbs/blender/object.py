# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Iterable

import bpy


def safe_set_attr(obj: Any, attr_name: str, value: Any) -> bool:
    if obj is None or not hasattr(obj, attr_name):
        return False

    try:
        setattr(obj, attr_name, value)
        return True
    except Exception:
        return False


def ensure_object_mode() -> None:
    try:
        active = bpy.context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def deselect_all_objects() -> None:
    ensure_object_mode()
    try:
        bpy.ops.object.select_all(action="DESELECT")
    except Exception:
        pass


def set_active_object(obj: bpy.types.Object | None) -> None:
    if obj is None:
        return

    try:
        bpy.context.view_layer.objects.active = obj
    except Exception:
        pass


def select_only_object(obj: bpy.types.Object) -> None:
    ensure_object_mode()
    deselect_all_objects()
    try:
        obj.select_set(True)
    except Exception:
        pass
    set_active_object(obj)


def unhide_all_collections(layer_collection: Any) -> None:
    try:
        layer_collection.hide_viewport = False
    except Exception:
        pass

    try:
        layer_collection.exclude = False
    except Exception:
        pass

    for child in getattr(layer_collection, "children", []):
        unhide_all_collections(child)


def find_objects_by_type(
    object_type: str,
    *,
    scene: bpy.types.Scene | None = None,
) -> list[bpy.types.Object]:
    source = scene.objects if scene is not None else bpy.data.objects
    return [obj for obj in source if obj.type == object_type]


def find_armature(
    name: str | None = None,
    *,
    required: bool = True,
    scene: bpy.types.Scene | None = None,
    prefer_name: str | None = "Armature",
) -> bpy.types.Object | None:
    if name:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "ARMATURE":
            return obj

        if required:
            raise RuntimeError(f"Requested armature not found or not ARMATURE: {name}")

        return None

    if prefer_name:
        preferred = bpy.data.objects.get(prefer_name)
        if preferred is not None and preferred.type == "ARMATURE":
            return preferred

    armatures = find_objects_by_type("ARMATURE", scene=scene)

    if not armatures:
        if required:
            raise RuntimeError("No ARMATURE object found in the scene.")
        return None

    if len(armatures) > 1:
        print(
            f"⚠️ Multiple armatures found; using first: {[obj.name for obj in armatures]}",
            flush=True,
        )

    return armatures[0]


def find_mesh(
    name: str | None = None,
    *,
    required: bool = True,
    scene: bpy.types.Scene | None = None,
) -> bpy.types.Object | None:
    if name:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "MESH":
            return obj

        if required:
            raise RuntimeError(f"Requested mesh not found or not MESH: {name}")

        return None

    meshes = find_objects_by_type("MESH", scene=scene)

    if not meshes:
        if required:
            raise RuntimeError("No MESH object found in the scene.")
        return None

    return meshes[0]


def find_mesh_objects(
    *,
    scene: bpy.types.Scene | None = None,
) -> list[bpy.types.Object]:
    return find_objects_by_type("MESH", scene=scene)


def get_object_visibility_state(obj: bpy.types.Object) -> dict[str, Any]:
    try:
        hide_get_value = obj.hide_get()
    except Exception:
        hide_get_value = False

    return {
        "hide_get": bool(hide_get_value),
        "hide_viewport": bool(getattr(obj, "hide_viewport", False)),
        "hide_render": bool(getattr(obj, "hide_render", False)),
    }


def unhide_object_for_ops(obj: bpy.types.Object) -> dict[str, Any]:
    state = get_object_visibility_state(obj)

    try:
        obj.hide_set(False)
    except Exception:
        pass

    try:
        obj.hide_viewport = False
    except Exception:
        pass

    try:
        obj.hide_render = False
    except Exception:
        pass

    return state


def restore_object_visibility(
    obj: bpy.types.Object,
    state: dict[str, Any],
    *,
    restore_viewport_disable: bool = True,
    restore_render_disable: bool = True,
) -> None:
    try:
        obj.hide_set(bool(state.get("hide_get", False)))
    except Exception:
        pass

    if restore_viewport_disable:
        try:
            obj.hide_viewport = bool(state.get("hide_viewport", False))
        except Exception:
            pass

    if restore_render_disable:
        try:
            obj.hide_render = bool(state.get("hide_render", False))
        except Exception:
            pass


def restore_source_visibility(
    obj: bpy.types.Object,
    state: dict[str, Any],
) -> None:
    """
    Restore eye/render hiding but keep viewport evaluation enabled.

    This matches the data-prep behavior needed by Source_<VDB> meshes:
    they may be hidden from the user/render but should remain usable by
    depsgraph and Surface Deform.
    """
    try:
        obj.hide_set(bool(state.get("hide_get", True)))
    except Exception:
        pass

    try:
        obj.hide_viewport = False
    except Exception:
        pass

    try:
        obj.hide_render = bool(state.get("hide_render", True))
    except Exception:
        pass


def remove_modifier_if_exists(
    obj: bpy.types.Object,
    name: str,
) -> int:
    removed = 0

    for mod in list(obj.modifiers):
        if mod.name == name:
            obj.modifiers.remove(mod)
            removed += 1

    return removed


def remove_modifiers_by_type(
    obj: bpy.types.Object,
    modifier_type: str,
) -> int:
    removed = 0

    for mod in list(obj.modifiers):
        if mod.type == modifier_type:
            obj.modifiers.remove(mod)
            removed += 1

    return removed


def ensure_objects_unhidden_for_ops(
    objects: Iterable[bpy.types.Object],
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}

    for obj in objects:
        states[obj.name] = unhide_object_for_ops(obj)

    return states


def print_scene_inspect_summary(
    *,
    include_actions: bool = True,
) -> None:
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]

    print(f"\n{'=' * 80}", flush=True)
    print("🔍 Scene Inspect Summary", flush=True)
    print(f"📦 bpy version: {bpy.app.version_string}", flush=True)
    print(f"{'=' * 80}", flush=True)

    print(f"\n🦴 Armatures ({len(armatures)}):", flush=True)
    for arm in armatures:
        print(
            f"  - {arm.name} "
            f"(data={arm.data.name}, bones={len(arm.data.bones)}, "
            f"pose_position={getattr(arm.data, 'pose_position', 'unknown')})",
            flush=True,
        )

    print(f"\n📦 Meshes ({len(meshes)}):", flush=True)
    for mesh in meshes:
        parent = mesh.parent.name if mesh.parent else "None"
        mods = [f"{mod.name}:{mod.type}" for mod in mesh.modifiers]
        print(
            f"  - {mesh.name} "
            f"(data={mesh.data.name}, parent={parent}, "
            f"verts={len(mesh.data.vertices)}, faces={len(mesh.data.polygons)}, "
            f"groups={len(mesh.vertex_groups)}, mods={mods})",
            flush=True,
        )

    if include_actions:
        print(f"\n🎬 Actions ({len(bpy.data.actions)}):", flush=True)
        for action in bpy.data.actions:
            frame_start, frame_end = action.frame_range
            slots_info = ""
            if hasattr(action, "slots") and len(action.slots) > 0:
                slot_names = [
                    getattr(slot, "name_display", getattr(slot, "identifier", "Unknown"))
                    for slot in action.slots
                ]
                slots_info = f" | Slots: {slot_names}"
            print(
                f"  - {action.name} "
                f"(Frames: {int(frame_start)} to {int(frame_end)}){slots_info}",
                flush=True,
            )

    print(f"{'=' * 80}\n", flush=True)
