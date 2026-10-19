#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 02a: SDF Grid fusion in Blender 5.x.
Updated: Supports specifying a target animation frame for fusion to avoid REST pose webbing.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy

try:
    import numpy as np
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    np = None
    trimesh = None
    HAS_TRIMESH = False

def ensure_object_mode() -> None:
    try:
        active = bpy.context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]

def apply_target_pose(target_frame: int) -> None:
    """根据指定的帧设置姿态。如果为 -1，则使用 REST 姿态。"""
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            try:
                if target_frame >= 0:
                    obj.data.pose_position = "POSE"
                else:
                    obj.data.pose_position = "REST"
            except Exception:
                pass
                
    if target_frame >= 0:
        bpy.context.scene.frame_set(target_frame)
        print(f"🛑 Set scene to POSE mode at frame {target_frame}.", flush=True)
    else:
        print("🛑 Set scene to REST pose.", flush=True)

def print_scene_summary() -> None:
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    print(f"📦 bpy version: {bpy.app.version_string}", flush=True)
    print(f"🦴 Armatures: {[(a.name, len(a.data.bones)) for a in armatures]}", flush=True)
    print("📦 Meshes: " + str([(m.name, len(m.data.vertices), len(m.data.polygons), tuple(round(v, 6) for v in m.scale)) for m in meshes]), flush=True)

def create_sdf_grid_geonodes(obj, voxel_size, offset_distance, *, band_width_padding):
    modifier = obj.modifiers.new(name="OGD_SDF_Offset", type="NODES")
    tree = bpy.data.node_groups.new(name=f"SDF_Remesh_Tree_{obj.name}", type="GeometryNodeTree")
    modifier.node_group = tree
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    node_in = nodes.new(type="NodeGroupInput")
    node_in.location = (-500, 0)
    node_mesh_to_sdf = nodes.new(type="GeometryNodeMeshToSDFGrid")
    node_mesh_to_sdf.name = "Node_MeshToSDF"
    node_mesh_to_sdf.location = (-250, 0)
    node_mesh_to_sdf.inputs["Voxel Size"].default_value = float(voxel_size)
    
    band_width_voxels = int(math.ceil(abs(float(offset_distance)) / float(voxel_size))) + int(band_width_padding)
    node_mesh_to_sdf.inputs["Band Width"].default_value = band_width_voxels

    node_sdf_offset = nodes.new(type="GeometryNodeSDFGridOffset")
    node_sdf_offset.name = "Node_SDFOffset"
    node_sdf_offset.location = (0, 0)
    node_sdf_offset.inputs["Distance"].default_value = float(offset_distance)

    node_grid_to_mesh = nodes.new(type="GeometryNodeGridToMesh")
    node_grid_to_mesh.name = "Node_GridToMesh"
    node_grid_to_mesh.location = (250, 0)

    node_out = nodes.new(type="NodeGroupOutput")
    node_out.location = (500, 0)

    links.new(node_in.outputs["Geometry"], node_mesh_to_sdf.inputs["Mesh"])
    links.new(node_mesh_to_sdf.outputs["SDF Grid"], node_sdf_offset.inputs["Grid"])
    links.new(node_sdf_offset.outputs["Grid"], node_grid_to_mesh.inputs["Grid"])
    links.new(node_grid_to_mesh.outputs["Mesh"], node_out.inputs["Geometry"])
    return modifier

def update_sdf_nodes(modifier, *, voxel_size, offset_distance, band_width_padding):
    tree = modifier.node_group
    node_mesh_to_sdf = tree.nodes["Node_MeshToSDF"]
    node_sdf_offset = tree.nodes["Node_SDFOffset"]
    node_mesh_to_sdf.inputs["Voxel Size"].default_value = float(voxel_size)
    band_width_voxels = int(math.ceil(abs(float(offset_distance)) / float(voxel_size))) + int(band_width_padding)
    node_mesh_to_sdf.inputs["Band Width"].default_value = band_width_voxels
    node_sdf_offset.inputs["Distance"].default_value = float(offset_distance)

def duplicate_join_and_staticize_sources(source_objects, *, temp_name, apply_scale):
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in source_objects:
        obj.hide_set(False)
        obj.select_set(True)
    bpy.context.view_layer.objects.active = source_objects[0]
    bpy.ops.object.duplicate()
    duplicated = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = duplicated[0]
    if len(duplicated) > 1:
        bpy.ops.object.join()
    temp_obj = bpy.context.view_layer.objects.active
    temp_obj.name = temp_name
    temp_obj.data.name = temp_name
    
    # 这里会将当前帧的形变烘焙成静态网格
    bpy.ops.object.convert(target="MESH")
    temp_obj = bpy.context.object
    temp_obj.name = temp_name
    temp_obj.data.name = temp_name
    
    if apply_scale:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    try:
        temp_obj.vertex_groups.clear()
    except Exception:
        pass
    return temp_obj

def make_empty_topology_stats(reason):
    return {"ok": False, "reason": reason, "components_filtered": 0, "main_genus": None, "main_is_watertight": False, "is_triangle_soup_like": False}

def evaluate_topology_from_evaluated_mesh(obj, depsgraph, *, min_component_faces, min_component_face_ratio):
    if not HAS_TRIMESH:
        return make_empty_topology_stats("trimesh_not_available")
    eval_obj = obj.evaluated_get(depsgraph)
    try:
        eval_mesh = eval_obj.to_mesh()
        eval_mesh.calc_loop_triangles()
        verts = np.empty(len(eval_mesh.vertices) * 3, dtype=np.float64)
        eval_mesh.vertices.foreach_get("co", verts)
        faces = np.empty(len(eval_mesh.loop_triangles) * 3, dtype=np.int64)
        eval_mesh.loop_triangles.foreach_get("vertices", faces)
        mesh = trimesh.Trimesh(vertices=verts.reshape((-1, 3)), faces=faces.reshape((-1, 3)), process=False, validate=False)
        try: mesh.remove_unreferenced_vertices()
        except: pass
        
        components = mesh.split(only_watertight=False)
        face_threshold = max(int(min_component_faces), int(math.ceil(len(mesh.faces) * float(min_component_face_ratio))))
        components_filtered_list = [c for c in components if len(c.faces) >= face_threshold] if face_threshold > 0 else list(components)
        if not components_filtered_list and components:
            components_filtered_list = [max(components, key=lambda c: len(c.faces))]
            
        main = max(components_filtered_list, key=lambda c: len(c.faces)) if components_filtered_list else None
        main_genus = int(round(1.0 - (float(main.euler_number) / 2.0))) if main else None
        main_is_watertight = bool(main.is_watertight) if main else False
        is_triangle_soup_like = (len(components) / float(max(len(mesh.faces), 1))) > 0.25
        
        return {
            "ok": True, "reason": "ok", "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "components_raw": len(components), "components_filtered": len(components_filtered_list),
            "main_genus": main_genus, "main_is_watertight": main_is_watertight, "is_triangle_soup_like": is_triangle_soup_like
        }
    except Exception as exc:
        return make_empty_topology_stats(f"failed: {exc}")
    finally:
        try: eval_obj.to_mesh_clear()
        except: pass

def topology_is_success(stats, *, require_watertight):
    if not stats.get("ok", False) or stats.get("is_triangle_soup_like", False): return False
    if stats.get("components_filtered", 999) != 1 or stats.get("main_genus", None) != 0: return False
    if require_watertight and not stats.get("main_is_watertight", False): return False
    return True

def format_topology_line(stats):
    if not stats.get("ok", False): return f"FAILED: {stats.get('reason')}"
    return f"V={stats['vertices']}, F={stats['faces']}, C_raw={stats['components_raw']}, C_filtered={stats['components_filtered']}, g_main={stats['main_genus']}, watertight={stats['main_is_watertight']}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_blend", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--auto_offset", action="store_true")
    parser.add_argument("--max_offset_multiplier", type=float, default=8.0)
    parser.add_argument("--band_width_padding", type=int, default=8)
    parser.add_argument("--min_component_faces", type=int, default=10)
    parser.add_argument("--min_component_face_ratio", type=float, default=0.0)
    parser.add_argument("--require_watertight", action="store_true")
    parser.add_argument("--no_apply_scale", dest="apply_scale", action="store_false")
    parser.add_argument("--target_frame", type=int, default=-1, help="Animation frame for SDF fusion. -1 means REST pose.")
    parser.set_defaults(apply_scale=True)
    args = parser.parse_args(script_argv())

    input_path = Path(args.input_blend)
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent / f"{input_path.stem}_sdf_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading: {input_path}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    ensure_object_mode()
    
    # 应用目标姿态 (REST 或指定帧)
    apply_target_pose(args.target_frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()

    merge_rules = {}
    if args.config:
        with open(args.config, "r") as f: merge_rules = json.load(f)
    else:
        merge_rules = {"main_mesh_VDB": {"source_meshes": [o.name for o in bpy.data.objects if o.type=="MESH"]}}

    manifest = {
        "source_blend": str(input_path.resolve()),
        "bpy_version": bpy.app.version_string,
        "target_frame": args.target_frame,
        "exports": {}, "details": {}
    }

    for target_name, config in merge_rules.items():
        valid_sources = [bpy.data.objects.get(n) for n in config.get("source_meshes", []) if bpy.data.objects.get(n)]
        if not valid_sources: continue
        
        voxel_size = args.voxel_size if args.voxel_size else config.get("voxel_size", 0.05)
        use_auto = args.auto_offset or (args.offset is None and "offset" not in config)
        initial_offset = args.offset if args.offset is not None else config.get("offset", 0.0)

        temp_obj = duplicate_join_and_staticize_sources(valid_sources, temp_name=f"TEMP_{target_name}", apply_scale=args.apply_scale)
        modifier = create_sdf_grid_geonodes(temp_obj, voxel_size, initial_offset, band_width_padding=args.band_width_padding)
        
        selected_offset = initial_offset
        if use_auto and HAS_TRIMESH:
            multiplier, step, max_mult = 0.5, 0.5, args.max_offset_multiplier
            while multiplier <= max_mult + 1e-9:
                current_offset = voxel_size * multiplier
                update_sdf_nodes(modifier, voxel_size=voxel_size, offset_distance=current_offset, band_width_padding=args.band_width_padding)
                depsgraph.update()
                stats = evaluate_topology_from_evaluated_mesh(temp_obj, depsgraph, min_component_faces=args.min_component_faces, min_component_face_ratio=args.min_component_face_ratio)
                ok = topology_is_success(stats, require_watertight=args.require_watertight)
                print(f"     [Attempt] {multiplier:.1f}x offset={current_offset:.6g} | {format_topology_line(stats)} | {'✅' if ok else '❌'}", flush=True)
                selected_offset = current_offset
                if ok: break
                multiplier += step
        
        update_sdf_nodes(modifier, voxel_size=voxel_size, offset_distance=selected_offset, band_width_padding=args.band_width_padding)
        depsgraph.update()
        
        ensure_object_mode()
        bpy.ops.object.select_all(action="DESELECT")
        temp_obj.select_set(True)
        bpy.context.view_layer.objects.active = temp_obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        
        temp_obj.name = target_name
        temp_obj.data.name = target_name
        final_obj_path = out_dir / f"{target_name}.obj"
        bpy.ops.wm.obj_export(filepath=str(final_obj_path), export_selected_objects=True)
        bpy.ops.object.delete()
        
        manifest["exports"][target_name] = str(final_obj_path.resolve())
        manifest["details"][target_name] = {"voxel_size": voxel_size, "offset": selected_offset}

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f: json.dump(manifest, f, indent=4)
    print(f"\n✅ Prep 02a complete. Manifest saved to: {manifest_path}", flush=True)

if __name__ == "__main__":
    main()
