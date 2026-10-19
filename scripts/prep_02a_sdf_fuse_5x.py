#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bpy
import sys
import argparse
import json
import math
from pathlib import Path

def ensure_object_mode():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

def create_sdf_grid_geonodes(obj, voxel_size=0.05, offset_distance=0.01):
    """使用 Blender 5.x 的 SDF Grid 节点构建精确偏移"""
    modifier = obj.modifiers.new(name="OGD_SDF_Offset", type='NODES')
    tree = bpy.data.node_groups.new(name="SDF_Remesh_Tree", type='GeometryNodeTree')
    modifier.node_group = tree
    
    tree.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    nodes = tree.nodes
    links = tree.links
    nodes.clear()
    
    node_in = nodes.new(type='NodeGroupInput')
    node_in.location = (-400, 0)
    
    # 1. Mesh to SDF Grid
    node_mesh_to_sdf = nodes.new(type='GeometryNodeMeshToSDFGrid')
    node_mesh_to_sdf.location = (-200, 0)
    node_mesh_to_sdf.inputs['Voxel Size'].default_value = voxel_size
    
    # 计算偏移距离需要多少层体素，并加上 4 层安全余量，转为整数
    band_width_voxels = int(math.ceil(abs(offset_distance) / voxel_size)) + 4
    node_mesh_to_sdf.inputs['Band Width'].default_value = band_width_voxels
    
    # 2. SDF Grid Offset
    node_sdf_offset = nodes.new(type='GeometryNodeSDFGridOffset')
    node_sdf_offset.location = (0, 0)
    # 正值向外膨胀(dilate)，负值向内收缩(erode)
    node_sdf_offset.inputs['Distance'].default_value = offset_distance
    
    # 3. Grid to Mesh
    node_grid_to_mesh = nodes.new(type='GeometryNodeGridToMesh')
    node_grid_to_mesh.location = (200, 0)
    
    node_out = nodes.new(type='NodeGroupOutput')
    node_out.location = (400, 0)
    
    # 连线
    links.new(node_in.outputs[0], node_mesh_to_sdf.inputs[0])
    links.new(node_mesh_to_sdf.outputs[0], node_sdf_offset.inputs[0])
    links.new(node_sdf_offset.outputs[0], node_grid_to_mesh.inputs[0])
    links.new(node_grid_to_mesh.outputs[0], node_out.inputs[0])
    
    return modifier

def main():
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Prep 02a: SDF Fusion in Blender 5.x (No Remesh)")
    parser.add_argument('--input_blend', type=str, required=True, help="Path to input .blend file")
    parser.add_argument('--config', type=str, default=None, help="[Optional] Path to JSON config file. If omitted, merges all meshes.")
    
    # 命令行参数，default=None 以便判断用户是否显式传参
    parser.add_argument('--voxel_size', type=float, default=None, help="Force override VDB voxel size")
    parser.add_argument('--offset', type=float, default=None, help="Force override SDF offset")
    parser.add_argument('--out_dir', type=str, default=None, help="Directory to save OBJ files")
    args = parser.parse_args(argv)

    input_path = Path(args.input_blend)
    if not input_path.exists():
        raise FileNotFoundError(f"Input blend file not found: {input_path}")

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent / f"{input_path.stem}_sdf_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading: {input_path}")
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    ensure_object_mode()

    # 强制 Armature 进入 REST Pose
    for arm in [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']:
        arm.data.pose_position = 'REST'

    # 解析合并规则
    if args.config and Path(args.config).exists():
        print(f"📄 Loading config from: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            merge_rules = json.load(f)
    else:
        print("ℹ️ No config provided. Defaulting to merge all meshes into 'main_mesh_VDB'.")
        all_mesh_names = [obj.name for obj in bpy.data.objects if obj.type == 'MESH']
        merge_rules = {
            "main_mesh_VDB": {
                "source_meshes": all_mesh_names
            }
        }

    manifest = {"source_blend": str(input_path.name), "exports": {}}

    print("\n🛠️ Starting SDF Grid Fusion (Blender 5.x)...")
    for target_name, config in merge_rules.items():
        source_names = config.get("source_meshes", [])
        
        # 优先级逻辑：命令行参数 > JSON配置 > 兜底默认值(0.05 / 0.01)
        voxel_size = args.voxel_size if args.voxel_size is not None else config.get("voxel_size", 0.05)
        offset = args.offset if args.offset is not None else config.get("offset", 0.01)

        valid_sources = [bpy.data.objects.get(n) for n in source_names if bpy.data.objects.get(n)]
        if not valid_sources:
            print(f"  ⏭️ Skip '{target_name}': No matching sources.")
            continue
            
        print(f"  🔄 Processing '{target_name}'...")
        
        bpy.ops.object.select_all(action='DESELECT')
        for obj in valid_sources:
            obj.hide_set(False)
            obj.select_set(True)
            
        bpy.ops.object.duplicate()
        active_obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = active_obj
        if len(bpy.context.selected_objects) > 1:
            bpy.ops.object.join()
            
        temp_mesh = active_obj
        temp_mesh.name = target_name
        
        print(f"     -> Applying SDF Grid (Voxel: {voxel_size}, Offset: {offset})")
        mod = create_sdf_grid_geonodes(temp_mesh, voxel_size, offset)
        bpy.ops.object.modifier_apply(modifier=mod.name)
        
        # 清理多余数据
        temp_mesh.modifiers.clear()
        temp_mesh.vertex_groups.clear()
            
        # 导出为 OBJ
        obj_path = out_dir / f"{target_name}.obj"
        print(f"     -> Exporting to {obj_path.name} (Vertices: {len(temp_mesh.data.vertices)})")
        
        ensure_object_mode()
        bpy.ops.object.select_all(action='DESELECT')
        temp_mesh.select_set(True)
        bpy.context.view_layer.objects.active = temp_mesh
        
        # Blender 4.0+ / 5.x 的新 OBJ 导出 API
        bpy.ops.wm.obj_export(filepath=str(obj_path), export_selected_objects=True)
        
        manifest["exports"][target_name] = str(obj_path.resolve())
        
        # 清理内存中的临时物体
        bpy.ops.object.delete()

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\n✅ Prep 02a Complete! Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()
