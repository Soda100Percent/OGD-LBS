#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bpy
import sys
import argparse
import json
from pathlib import Path

def ensure_object_mode():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

def main():
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Prep 02b: Import Fused Meshes in Blender 4.x")
    parser.add_argument('--input_blend', type=str, required=True, help="Path to input .blend file")
    parser.add_argument('--manifest', type=str, required=True, help="Path to manifest.json from prep_02a")
    parser.add_argument('--output_blend', type=str, default=None, help="Path to save final .blend file")
    args = parser.parse_args(argv)

    input_path = Path(args.input_blend)
    manifest_path = Path(args.manifest)
    
    if args.output_blend:
        output_path = Path(args.output_blend)
    else:
        stem = input_path.stem.replace("_prep_01", "_prep_02")
        if "_prep_02" not in stem:
            stem += "_prep_02"
        output_path = input_path.parent / f"{stem}{input_path.suffix}"

    print(f"📂 Loading: {input_path}")
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    ensure_object_mode()

    # 从 manifest 读取生成 SDF 时的目标帧
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    target_frame = manifest.get("target_frame", -1)

    # 强制 Armature 对齐到目标姿态，方便 GUI 检查时完美重合
    print(f"\n🛑 Aligning scene to target_frame = {target_frame}...")
    for arm in [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']:
        if target_frame >= 0:
            arm.data.pose_position = 'POSE'
        else:
            arm.data.pose_position = 'REST'
            
    if target_frame >= 0:
        bpy.context.scene.frame_set(target_frame)
    bpy.context.view_layer.update()

    # 1. 原始网格大一统 (Original_Source_Mesh)
    original_meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if original_meshes:
        print("\n📦 Consolidating original meshes into 'Original_Source_Mesh'...")
        bpy.ops.object.select_all(action='DESELECT')
        for obj in original_meshes:
            obj.hide_set(False)
            obj.select_set(True)
            
        bpy.context.view_layer.objects.active = original_meshes[0]
        if len(original_meshes) > 1:
            bpy.ops.object.join()
            
        original_source = original_meshes[0]
        original_source.name = "Original_Source_Mesh"
        original_source.data.name = "Original_Source_Mesh"
        
        # 隐藏原始网格
        original_source.hide_set(True)
        original_source.hide_viewport = True
        original_source.hide_render = True

    # 2. 导入 5.x 生成的 OBJ
    print(f"\n📥 Importing fused meshes from manifest: {manifest_path}")
    imported_mesh_names = []
    for target_name, obj_file in manifest.get("exports", {}).items():
        if not Path(obj_file).exists():
            print(f"  ⚠️ Warning: {obj_file} not found!")
            continue
            
        print(f"  -> Importing {target_name}...")
        bpy.ops.wm.obj_import(filepath=obj_file)
        
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                obj.name = target_name
                obj.data.name = target_name
                imported_mesh_names.append(obj.name)

    # 3. 场景终极清理
    print("\n🧹 Cleaning up scene...")
    allowed_mesh_names = ["Original_Source_Mesh"] + imported_mesh_names
    meshes_to_delete = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.name not in allowed_mesh_names]
    
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes_to_delete:
        obj.hide_set(False)
        obj.select_set(True)
        
    if meshes_to_delete:
        bpy.ops.object.delete()

    # 4. 保存
    print(f"\n✅ Prep 02b Complete!")
    print(f"💾 Saving to: {output_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    print("🎉 Done!")

if __name__ == "__main__":
    main()
