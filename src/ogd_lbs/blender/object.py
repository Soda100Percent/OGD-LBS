# filepath: /workspace/lao-lbs/src/lao_lbs/blender/object.py
from __future__ import annotations
from typing import Any
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