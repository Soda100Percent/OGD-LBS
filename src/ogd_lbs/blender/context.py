# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

import bpy

from ogd_lbs.utils.common import fail


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


def update_depsgraph() -> None:
    """
    Backward-compatible alias for data-prep scripts.
    """
    update_view_layer()


def force_ui_redraw(iterations: int = 2) -> None:
    try:
        wm = bpy.context.window_manager
        for window in wm.windows:
            screen = window.screen
            if screen is None:
                continue
            for area in screen.areas:
                area.tag_redraw()
    except Exception:
        pass

    try:
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=iterations)
    except Exception:
        try:
            bpy.ops.wm.redraw_timer(type="DRAW", iterations=iterations)
        except Exception:
            pass


def find_view3d_context() -> dict[str, Any]:
    wm = bpy.context.window_manager
    window = bpy.context.window

    if window is None:
        windows = list(wm.windows)
        if windows:
            window = windows[0]

    if window is None:
        fail("No Blender window found. Are you running with -b/background?")

    screen = window.screen

    if screen is None:
        fail("No Blender screen found. Cannot create VIEW_3D context.")

    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue

        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if region is None:
            continue

        space = next((s for s in area.spaces if s.type == "VIEW_3D"), None)
        if space is None:
            continue

        region_data = getattr(space, "region_3d", None)

        return {
            "window": window,
            "screen": screen,
            "area": area,
            "region": region,
            "space_data": space,
            "region_data": region_data,
            "scene": bpy.context.scene,
            "view_layer": bpy.context.view_layer,
        }

    fail(
        "No VIEW_3D area found. Use Blender non-background UI mode, "
        "ideally with --factory-startup and xvfb-run."
    )
    return {}


def make_temp_override_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Remove 'screen' to avoid Blender 4.5+ temporary-screen override errors.
    """
    return {k: v for k, v in ctx.items() if k != "screen"}


def safe_overlay_set(overlay: Any, attr_name: str, value: Any) -> None:
    if hasattr(overlay, attr_name):
        try:
            setattr(overlay, attr_name, value)
        except Exception:
            pass


def align_scene_pose_frame(
    target_frame: int,
    *,
    armature: bpy.types.Object | None = None,
    set_all_armatures: bool = True,
) -> None:
    """
    Align scene and armature pose mode for data-prep operations.

    target_frame < 0:
        Set armature pose_position to REST. Do not force scene.frame_set.

    target_frame >= 0:
        Set armature pose_position to POSE and scene.frame_set(target_frame).

    If armature is provided and set_all_armatures is False, only that armature
    is changed. Otherwise all scene armatures are updated.
    """
    target_frame = int(target_frame)

    if armature is not None and not set_all_armatures:
        armatures = [armature]
    else:
        armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]

    pose_position = "POSE" if target_frame >= 0 else "REST"

    for arm in armatures:
        try:
            arm.data.pose_position = pose_position
        except Exception:
            pass

    if target_frame >= 0:
        bpy.context.scene.frame_set(target_frame)

    update_view_layer()
