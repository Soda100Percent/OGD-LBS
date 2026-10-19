# filepath: /workspace/lao-lbs/src/lao_lbs/blender/camera.py
from __future__ import annotations
from typing import Any
import bpy
import mathutils

# 修正导入路径
from ogd_lbs.utils.common import log
from ogd_lbs.blender.context import update_view_layer, force_ui_redraw

def setup_camera(
    cam_data: dict[str, Any],
    pullback: float = 0.0,
    res_preset: str = "ORIGINAL",
    custom_res: tuple[int, int] = (1920, 1080)
) -> bpy.types.Object:
    cam_name = str(cam_data.get("camera_name", "Camera"))

    existing = bpy.data.objects.get(cam_name)
    if existing is not None and existing.type == "CAMERA":
        bpy.data.objects.remove(existing, do_unlink=True)

    cam = bpy.data.cameras.new(cam_name)
    cam_obj = bpy.data.objects.new(cam.name, cam)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # 1. 基础位姿设置
    cam_obj.location = cam_data["location"]
    cam_obj.rotation_euler = cam_data["rotation_euler"]

    # 2. 相机后退逻辑 (Camera Pullback)
    if pullback != 0.0:
        # 获取相机的旋转矩阵
        rot_mat = cam_obj.rotation_euler.to_matrix()
        # 在 Blender 中，相机的局部 +Z 轴是指向相机后方（远离画面的方向）
        backward_vec = rot_mat @ mathutils.Vector((0.0, 0.0, 1.0))
        # 将相机沿后方移动指定距离
        cam_obj.location = cam_obj.location + backward_vec * pullback

    if "focal_length" in cam_data:
        cam.lens = cam_data["focal_length"]
    elif "lens" in cam_data:
        cam.lens = cam_data["lens"]

    if "sensor_width" in cam_data:
        cam.sensor_width = cam_data["sensor_width"]

    cam.shift_x = cam_data.get("shift_x", 0.0)
    cam.shift_y = cam_data.get("shift_y", 0.0)

    if "clip_start" in cam_data:
        cam.clip_start = cam_data["clip_start"]
    if "clip_end" in cam_data:
        cam.clip_end = cam_data["clip_end"]
    else:
        cam.clip_end = max(cam.clip_end, 10000.0)

    # 3. 分辨率调节逻辑 (Resolution Override)
    res_x = int(cam_data.get("resolution_x", 1920))
    res_y = int(cam_data.get("resolution_y", 1080))
    
    preset_upper = res_preset.upper()
    if preset_upper == "1080P":
        res_x, res_y = 1920, 1080
    elif preset_upper == "2K":
        res_x, res_y = 2560, 1440
    elif preset_upper == "4K":
        res_x, res_y = 3840, 2160
    elif preset_upper == "CUSTOM":
        res_x, res_y = custom_res[0], custom_res[1]

    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    bpy.context.scene.render.resolution_percentage = 100

    log(
        "Camera configured: "
        f"name={cam_obj.name}, "
        f"location={tuple(round(v, 6) for v in cam_obj.location)}, "
        f"rotation={tuple(round(v, 6) for v in cam_obj.rotation_euler)}, "
        f"pullback={pullback}, "
        f"lens={cam.lens}, "
        f"sensor_width={cam.sensor_width}, "
        f"shift=({cam.shift_x}, {cam.shift_y}), "
        f"resolution={res_x}x{res_y} ({preset_upper})"
    )

    update_view_layer()
    force_ui_redraw()
    return cam_obj