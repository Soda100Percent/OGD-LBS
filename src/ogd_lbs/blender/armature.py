# filepath: /workspace/lao-lbs/src/lao_lbs/blender/armature.py
from __future__ import annotations
from typing import Any
import bpy

# 修正导入路径
from ogd_lbs.utils.common import log, warn, fail
from ogd_lbs.blender.object import safe_set_attr, ensure_object_mode, deselect_all_objects
from ogd_lbs.blender.context import update_view_layer, force_ui_redraw, find_view3d_context, make_temp_override_context

def apply_custom_bone_color(
    bone_color: Any,
    normal_rgb: tuple[float, float, float],
    select_rgb: tuple[float, float, float],
    active_rgb: tuple[float, float, float],
) -> bool:
    if bone_color is None:
        return False
    applied = False
    try:
        bone_color.palette = "CUSTOM"
    except Exception as exc:
        warn(f"Failed to set bone_color.palette='CUSTOM': {exc}")

    if hasattr(bone_color, "custom"):
        custom = bone_color.custom
        try:
            if hasattr(custom, "normal"):
                custom.normal = normal_rgb
                applied = True
            if hasattr(custom, "select"):
                custom.select = select_rgb
                applied = True
            if hasattr(custom, "active"):
                custom.active = active_rgb
                applied = True
            if applied:
                return True
        except Exception as exc:
            warn(f"Failed to set bone_color.custom normal/select/active: {exc}")

    for attr_name, value in [
        ("custom_normal", normal_rgb),
        ("custom_select", select_rgb),
        ("custom_active", active_rgb),
    ]:
        if hasattr(bone_color, attr_name):
            try:
                setattr(bone_color, attr_name, value)
                applied = True
            except Exception as exc:
                warn(f"Failed to set bone_color.{attr_name}: {exc}")
    return applied

def setup_armature_style(
    armature_obj: bpy.types.Object,
    bone_style: str,
    normal_rgb: tuple[float, float, float],
    select_rgb: tuple[float, float, float],
    active_rgb: tuple[float, float, float],
) -> None:
    if armature_obj.type != "ARMATURE":
        fail(f"Expected ARMATURE, got {armature_obj.type}: {armature_obj.name}")
    arm_data = armature_obj.data

    try:
        arm_data.display_type = bone_style
        log(f"Armature display_type set to: {bone_style}")
    except Exception as exc:
        warn(f"Failed to set armature display_type={bone_style}: {exc}")

    try:
        armature_obj.show_in_front = True
    except Exception as exc:
        warn(f"Failed to set armature_obj.show_in_front=True: {exc}")

    if hasattr(arm_data, "show_bone_colors"):
        try:
            arm_data.show_bone_colors = True
        except Exception as exc:
            warn(f"Failed to set armature.data.show_bone_colors=True: {exc}")

    if hasattr(arm_data, "show_group_colors"):
        try:
            arm_data.show_group_colors = True
        except Exception:
            pass

    applied_count = 0
    failed_count = 0

    for bone in arm_data.bones:
        ok = apply_custom_bone_color(bone.color, normal_rgb, select_rgb, active_rgb)
        if ok:
            applied_count += 1
        else:
            failed_count += 1
            warn(f"Could not apply color to Armature Bone: {bone.name}")

    if armature_obj.pose is not None:
        for pose_bone in armature_obj.pose.bones:
            ok = apply_custom_bone_color(pose_bone.color, normal_rgb, select_rgb, active_rgb)
            if ok:
                applied_count += 1
            else:
                warn(f"Could not apply color to Pose Bone override: {pose_bone.name}")

    log(f"Bone color setup complete: applied={applied_count}, failed={failed_count}")
    update_view_layer()
    force_ui_redraw()

def get_armature_bone_collections(arm_data: Any) -> list[Any]:
    if arm_data is None:
        return []
    if hasattr(arm_data, "collections_all"):
        try:
            return list(arm_data.collections_all)
        except Exception:
            pass
    if hasattr(arm_data, "collections"):
        try:
            return list(arm_data.collections)
        except Exception:
            pass
    return []

def force_bone_collections_visible(armature_obj: bpy.types.Object) -> None:
    arm_data = armature_obj.data
    for bcoll in get_armature_bone_collections(arm_data):
        safe_set_attr(bcoll, "is_visible", True)
        safe_set_attr(bcoll, "is_solo_visible", False)
        safe_set_attr(bcoll, "is_solo", False)

def force_legacy_armature_layers_visible(armature_obj: bpy.types.Object) -> None:
    arm_data = armature_obj.data
    if not hasattr(arm_data, "layers"):
        return
    try:
        arm_data.layers = [True] * len(arm_data.layers)
    except Exception:
        pass

def force_armature_object_visible(armature_obj: bpy.types.Object) -> None:
    try:
        armature_obj.hide_set(False)
    except Exception:
        pass
    armature_obj.hide_viewport = False
    armature_obj.hide_render = False
    try:
        armature_obj.visible_camera = True
    except Exception:
        pass
    try:
        armature_obj.show_in_front = True
    except Exception:
        pass

def force_all_bones_unhidden_object_data(armature_obj: bpy.types.Object) -> None:
    arm_data = armature_obj.data
    for bone in getattr(arm_data, "bones", []):
        try:
            bone.hide = False
        except Exception:
            pass
        if hasattr(bone, "show_wire"):
            try:
                bone.show_wire = False
            except Exception:
                pass

def make_armature_active(armature_obj: bpy.types.Object) -> None:
    ensure_object_mode()
    deselect_all_objects()
    try:
        armature_obj.select_set(True)
    except Exception:
        pass
    try:
        bpy.context.view_layer.objects.active = armature_obj
    except Exception:
        pass
    update_view_layer()
    force_ui_redraw()

def reveal_pose_mode_bones(armature_obj: bpy.types.Object) -> None:
    make_armature_active(armature_obj)
    ctx = find_view3d_context()
    try:
        with bpy.context.temp_override(**make_temp_override_context(ctx)):
            bpy.ops.object.mode_set(mode="POSE")
            try:
                bpy.ops.pose.reveal(select=False)
            except TypeError:
                bpy.ops.pose.reveal()
            except Exception:
                pass
            try:
                bpy.ops.pose.select_all(action="DESELECT")
            except Exception:
                pass
    except Exception as exc:
        warn(f"Failed to reveal pose-mode bones: {exc}")
    update_view_layer()
    force_ui_redraw()

def reveal_edit_mode_bones(armature_obj: bpy.types.Object) -> None:
    make_armature_active(armature_obj)
    ctx = find_view3d_context()
    try:
        with bpy.context.temp_override(**make_temp_override_context(ctx)):
            bpy.ops.object.mode_set(mode="EDIT")
            try:
                bpy.ops.armature.reveal()
            except Exception:
                pass
            try:
                bpy.ops.armature.select_all(action="DESELECT")
            except Exception:
                pass
    except Exception as exc:
        warn(f"Failed to reveal edit-mode bones: {exc}")
    update_view_layer()
    force_ui_redraw()

def force_armature_viewport_ready(
    armature_obj: bpy.types.Object,
    *,
    reveal_bones: bool = True,
    final_mode: str = "POSE",
) -> None:
    force_armature_object_visible(armature_obj)
    force_bone_collections_visible(armature_obj)
    force_legacy_armature_layers_visible(armature_obj)
    force_all_bones_unhidden_object_data(armature_obj)

    if reveal_bones:
        reveal_pose_mode_bones(armature_obj)
        reveal_edit_mode_bones(armature_obj)

    make_armature_active(armature_obj)
    ctx = find_view3d_context()
    try:
        with bpy.context.temp_override(**make_temp_override_context(ctx)):
            if final_mode == "POSE":
                bpy.ops.object.mode_set(mode="POSE")
            elif final_mode == "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            else:
                warn(f"Unknown final armature mode '{final_mode}', fallback to POSE")
                bpy.ops.object.mode_set(mode="POSE")
    except Exception as exc:
        warn(f"Failed to set final armature viewport mode={final_mode}: {exc}")
    update_view_layer()
    force_ui_redraw()

def log_armature_visibility_debug(armature_obj: bpy.types.Object, *, prefix: str) -> None:
    arm_data = armature_obj.data
    hidden_bone_count = 0
    total_bone_count = 0
    for bone in getattr(arm_data, "bones", []):
        total_bone_count += 1
        try:
            if bone.hide:
                hidden_bone_count += 1
        except Exception:
            pass

    hidden_bcols = []
    for bcoll in get_armature_bone_collections(arm_data):
        try:
            if hasattr(bcoll, "is_visible") and not bcoll.is_visible:
                hidden_bcols.append(bcoll.name)
        except Exception:
            pass

    try:
        hide_get_value = armature_obj.hide_get()
    except Exception:
        hide_get_value = "unknown"

    try:
        visible_get_value = armature_obj.visible_get(view_layer=bpy.context.view_layer)
    except Exception:
        try:
            visible_get_value = armature_obj.visible_get()
        except Exception:
            visible_get_value = "unknown"

    log(
        f"{prefix}: armature='{armature_obj.name}', "
        f"mode={armature_obj.mode}, "
        f"hide_viewport={armature_obj.hide_viewport}, "
        f"hide_render={armature_obj.hide_render}, "
        f"hide_get={hide_get_value}, "
        f"visible_get={visible_get_value}, "
        f"show_in_front={getattr(armature_obj, 'show_in_front', 'unknown')}, "
        f"bones={total_bone_count}, "
        f"hidden_bones={hidden_bone_count}, "
        f"hidden_bone_collections={hidden_bcols[:10]}"
    )