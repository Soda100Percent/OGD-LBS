#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bpy
import sys
import argparse
from pathlib import Path

def ensure_object_mode():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

def main():
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Prep 03: Surface Deform Binding")
    parser.add_argument('--input_blend', type=str, required=True, help="Path to input _prep_02.blend file")
    parser.add_argument('--output_blend', type=str, default=None, help="Path to save final _prep_03.blend file")
    parser.add_argument('--target_mesh', type=str, default="main_mesh_VDB", help="Name of the fused mesh")
    parser.add_argument('--source_mesh', type=str, default="Original_Source_Mesh", help="Name of the original weighted mesh")
    parser.add_argument('--max_falloff', type=float, default=16.0, help="Maximum falloff value to try for binding")
    parser.add_argument('--bind_frame', type=int, default=-1, help="Animation frame to bind on. Must match the frame used in prep_02a. -1 means REST pose.")
    args = parser.parse_args(argv)

    input_path = Path(args.input_blend)
    
    if args.output_blend:
        output_path = Path(args.output_blend)
    else:
        stem = input_path.stem.replace("_prep_02", "_prep_03")
        if "_prep_03" not in stem:
            stem += "_prep_03"
        output_path = input_path.parent / f"{stem}{input_path.suffix}"

    print(f"📂 Loading: {input_path}")
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    ensure_object_mode()

    armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
    if not armatures:
        raise RuntimeError("No Armature found in the scene!")
    armature = armatures[0]

    source_mesh = bpy.data.objects.get(args.source_mesh)
    target_mesh = bpy.data.objects.get(args.target_mesh)

    if not source_mesh or not target_mesh:
        raise RuntimeError("Source or Target mesh not found!")

    # 2. 强制进入指定的绑定姿态 (REST 或 指定帧)
    if args.bind_frame >= 0:
        print(f"\n🛑 Setting Armature to POSE mode at frame {args.bind_frame} for binding...")
        armature.data.pose_position = 'POSE'
        bpy.context.scene.frame_set(args.bind_frame)
    else:
        print("\n🛑 Setting Armature to REST pose for binding...")
        armature.data.pose_position = 'REST'
    
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()

    source_mesh_hide_state = source_mesh.hide_get()
    source_mesh.hide_set(False)
    target_mesh.hide_set(False)

    # 3. 源网格拓扑安全化 (防凹面报错)
    print("📐 Adding Triangulate modifier to Source Mesh...")
    for mod in source_mesh.modifiers:
        if mod.name == "OGD_Triangulate":
            source_mesh.modifiers.remove(mod)
            
    tri_mod = source_mesh.modifiers.new(name="OGD_Triangulate", type='TRIANGULATE')
    tri_mod.keep_custom_normals = True
    tri_mod.quad_method = 'BEAUTY'
    tri_mod.ngon_method = 'BEAUTY'

    # 4. 目标网格添加 Surface Deform
    print("🔗 Setting up Surface Deform modifier on Target Mesh...")
    for mod in target_mesh.modifiers:
        if mod.name == "OGD_SurfaceDeform":
            target_mesh.modifiers.remove(mod)

    surf_mod = target_mesh.modifiers.new(name="OGD_SurfaceDeform", type='SURFACE_DEFORM')
    surf_mod.target = source_mesh

    # 5. 自适应 Falloff 绑定
    bpy.ops.object.select_all(action='DESELECT')
    target_mesh.select_set(True)
    bpy.context.view_layer.objects.active = target_mesh

    is_bound = False
    current_falloff = 4.0
    
    print("\n⚙️ Starting auto-tuning bind process...")
    while current_falloff <= args.max_falloff:
        surf_mod.falloff = current_falloff
        print(f"  -> Attempting bind with falloff = {current_falloff:.1f}...")
        
        try:
            with bpy.context.temp_override(active_object=target_mesh, selected_editable_objects=[target_mesh]):
                bpy.ops.object.surfacedeform_bind(modifier=surf_mod.name)
        except Exception as e:
            print(f"     [Warning] Bind operator threw an exception: {e}")

        if surf_mod.is_bound:
            print(f"  ✅ Successfully bound with falloff = {current_falloff:.1f}!")
            is_bound = True
            break
        else:
            print(f"  ❌ Failed. Increasing falloff...")
            current_falloff += 2.0

    if not is_bound:
        raise RuntimeError(f"Failed to bind Surface Deform up to falloff {args.max_falloff}.")

    # 6. 恢复状态 (绑定完成后，确保模型处于 POSE 模式以便播放动画)
    print("\n▶️ Ensuring Armature is in POSE mode for playback...")
    armature.data.pose_position = 'POSE'
    source_mesh.hide_set(source_mesh_hide_state)
    bpy.context.view_layer.update()

    # 7. 保存
    print(f"\n✅ Prep 03 Complete!")
    print(f"💾 Saving to: {output_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    print("🎉 Done!")

if __name__ == "__main__":
    main()
