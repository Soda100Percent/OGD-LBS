import argparse
import sys
import math
from pathlib import Path
import bpy
import mathutils

from ogd_lbs.utils.common import log, fail, Timer
from ogd_lbs.blender.camera import setup_camera

def clear_scene():
    """清理 Blender 默认场景"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.armatures]:
        for item in block:
            if item.users == 0:
                block.remove(item)

def calculate_bounding_box():
    """计算场景中所有 Mesh 的全局包围盒"""
    min_c = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_c = mathutils.Vector((-float('inf'), -float('inf'), -float('inf')))
    
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        return mathutils.Vector((0,0,0)), mathutils.Vector((1,1,1))

    for obj in meshes:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                min_c[i] = min(min_c[i], world_corner[i])
                max_c[i] = max(max_c[i], world_corner[i])
                
    center = (min_c + max_c) / 2.0
    size = max_c - min_c
    return center, size

def calculate_camera_distance(frame_size, focal_length=50.0, sensor_size=36.0, margin=1.3):
    """基于三角函数计算完美的相机距离"""
    fov_half_tan = (sensor_size / 2.0) / focal_length
    distance = (frame_size / 2.0) / fov_half_tan
    return distance * margin

def setup_lighting():
    """设置基础三点光源"""
    light_data = bpy.data.lights.new(name="MainLight", type='SUN')
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new(name="MainLight", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    fill_data = bpy.data.lights.new(name="FillLight", type='SUN')
    fill_data.energy = 1.5
    fill_obj = bpy.data.objects.new(name="FillLight", object_data=fill_data)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(45), 0, math.radians(-45))

def main():
    parser = argparse.ArgumentParser(description="Headless Renderer for FBX/GLB")
    parser.add_argument("--asset", type=str, required=True, help="Path to input FBX or GLB")
    parser.add_argument("--out", type=str, required=True, help="Path to output PNG")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution")
    args = parser.parse_args()

    asset_path = Path(args.asset)
    out_path = Path(args.out)
    
    if not asset_path.exists():
        fail(f"Asset file not found: {asset_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Timer("Total Render Process"):
        log(f"Starting render for: {asset_path.name}")
        clear_scene()
        
        # 动态识别扩展名并导入
        ext = asset_path.suffix.lower()
        if ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=str(asset_path))
        elif ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=str(asset_path))
        else:
            fail(f"Unsupported file format: {ext}")
        
        # --- 核心修复逻辑 ---
        center, size = calculate_bounding_box()
        focal_length = 50.0
        
        # 1. 计算要框住 X(宽) 和 Z(高) 所需要的距离
        max_frame_dim = max(size.x, size.z)
        dist_from_front = calculate_camera_distance(max_frame_dim, focal_length=focal_length, margin=1.3)
        
        # 2. 找到物体最前端的 Y 坐标 (center.y - 深度的一半)
        front_y = center.y - (size.y / 2.0)
        
        # 3. 将相机从最前端再往后退
        camera_y = front_y - dist_from_front
        
        # 稍微抬高一点相机，俯视一点点，对于四足动物效果更好
        camera_z = center.z + (size.z * 0.2)
        
        cam_data = {
            "camera_name": "RenderCam",
            "location": (center.x, camera_y, camera_z),
            "rotation_euler": (math.radians(80), 0, 0), # 微微向下倾斜 10 度
            "focal_length": focal_length
        }
        setup_camera(cam_data, res_preset="CUSTOM", custom_res=(args.resolution, args.resolution))
        setup_lighting()
        
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 16
        scene.render.film_transparent = True
        scene.render.filepath = str(out_path)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        
        bpy.ops.render.render(write_still=True)
        log(f"Render saved to: {out_path}")

if __name__ == "__main__":
    main()
