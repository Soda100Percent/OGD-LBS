# filepath: /workspace/lao-lbs/src/lao_lbs/blender/viewport_hack.py
from __future__ import annotations

import argparse
from typing import Any

import bpy

# 修正为 lao_lbs 绝对导入
from ogd_lbs.utils.common import log, warn
from ogd_lbs.blender.object import safe_set_attr
from ogd_lbs.blender.context import (
    update_view_layer,
    force_ui_redraw,
    find_view3d_context,
    safe_overlay_set,
    make_temp_override_context,
)


def maximize_view3d_area_once(args: argparse.Namespace) -> None:
    if not args.maximize_view3d:
        return

    if getattr(args, "_view3d_maximized", False):
        return

    try:
        ctx = find_view3d_context()
        with bpy.context.temp_override(**make_temp_override_context(ctx)):
            bpy.ops.screen.screen_full_area(use_hide_panels=True)

        args._view3d_maximized = True
        log("VIEW_3D area maximized.")
        force_ui_redraw(iterations=3)

    except Exception as exc:
        warn(f"Failed to maximize VIEW_3D area, continue without maximize: {exc}")
        args._view3d_maximized = True


def configure_view3d_for_camera_render(
    args: argparse.Namespace,
    shading_type: str,
    *,
    pass_name: str,
    want_armature: bool,
    want_mesh: bool,
) -> dict[str, Any]:
    maximize_view3d_area_once(args)

    ctx = find_view3d_context()
    space = ctx["space_data"]
    region = ctx["region"]
    region_data = ctx.get("region_data")
    scene = bpy.context.scene

    try:
        if getattr(space, "local_view", None) is not None:
            with bpy.context.temp_override(**make_temp_override_context(ctx)):
                bpy.ops.view3d.localview(frame_selected=False)
            log(f"{pass_name}: exited local view")
    except Exception as exc:
        warn(f"{pass_name}: failed to exit local view: {exc}")

    try:
        if scene.camera is not None:
            safe_set_attr(space, "use_local_camera", False)
            space.camera = scene.camera
        else:
            safe_set_attr(space, "use_local_camera", False)
    except Exception as exc:
        warn(f"{pass_name}: failed to bind viewport camera: {exc}")

    try:
        if region_data is not None:
            region_data.view_perspective = "CAMERA"
            safe_set_attr(region_data, "view_camera_zoom", 0)
            safe_set_attr(region_data, "view_camera_offset", (0.0, 0.0))
    except Exception as exc:
        warn(f"{pass_name}: failed to set VIEW_3D region to CAMERA: {exc}")

    safe_set_attr(space, "clip_start", 0.001)
    safe_set_attr(space, "clip_end", 100000.0)

    try:
        space.shading.type = shading_type
    except Exception as exc:
        warn(f"{pass_name}: failed to set viewport shading to {shading_type}: {exc}")

    safe_set_attr(space.shading, "light", "STUDIO")
    safe_set_attr(space.shading, "show_xray", False)
    safe_set_attr(space.shading, "show_xray_wireframe", False)

    try:
        space.shading.background_type = "VIEWPORT"
    except Exception:
        pass

    try:
        space.shading.background_color = (0.0, 0.0, 0.0)
    except Exception:
        pass

    safe_set_attr(space, "show_object_viewport_armature", bool(want_armature))
    safe_set_attr(space, "show_object_viewport_mesh", bool(want_mesh))

    for attr_name in [
        "show_object_viewport_camera",
        "show_object_viewport_light",
        "show_object_viewport_empty",
        "show_object_viewport_lattice",
        "show_object_viewport_curve",
        "show_object_viewport_curves",
        "show_object_viewport_font",
        "show_object_viewport_speaker",
        "show_object_viewport_light_probe",
        "show_object_viewport_grease_pencil",
        "show_object_viewport_volume",
        "show_object_viewport_pointcloud",
        "show_object_viewport_meta",
        "show_object_viewport_surf",
    ]:
        safe_set_attr(space, attr_name, False)

    overlay = getattr(space, "overlay", None)

    if overlay is not None:
        safe_overlay_set(overlay, "show_overlays", True)
        safe_overlay_set(overlay, "show_bones", True)
        safe_overlay_set(overlay, "show_extras", True)

        safe_overlay_set(overlay, "show_floor", False)
        safe_overlay_set(overlay, "show_axis_x", False)
        safe_overlay_set(overlay, "show_axis_y", False)
        safe_overlay_set(overlay, "show_axis_z", False)
        safe_overlay_set(overlay, "show_cursor", False)
        safe_overlay_set(overlay, "show_object_origins", False)
        safe_overlay_set(overlay, "show_object_origins_all", False)
        safe_overlay_set(overlay, "show_relationship_lines", False)
        safe_overlay_set(overlay, "show_outline_selected", False)
        safe_overlay_set(overlay, "show_text", False)
        safe_overlay_set(overlay, "show_stats", False)
        safe_overlay_set(overlay, "show_annotation", False)
        safe_overlay_set(overlay, "show_face_orientation", False)

    update_view_layer()
    force_ui_redraw(iterations=3)

    if region_data is not None and region_data.view_perspective != "CAMERA":
        warn(
            f"{pass_name}: Viewport perspective is '{region_data.view_perspective}', "
            "forcing to 'CAMERA'."
        )
        region_data.view_perspective = "CAMERA"

    try:
        rd = ctx.get("region_data")
        log(
            f"{pass_name}: VIEW_3D configured, "
            f"region={region.width}x{region.height}, "
            f"perspective={getattr(rd, 'view_perspective', 'unknown')}, "
            f"use_local_camera={getattr(space, 'use_local_camera', 'unknown')}, "
            f"space.camera={getattr(getattr(space, 'camera', None), 'name', None)}, "
            f"scene.camera={getattr(scene.camera, 'name', None)}, "
            f"want_armature={want_armature}, want_mesh={want_mesh}"
        )
    except Exception:
        pass

    return ctx