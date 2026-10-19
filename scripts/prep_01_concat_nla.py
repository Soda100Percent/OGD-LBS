#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bpy
import sys
import argparse
from pathlib import Path

def print_inspect_info():
    """打印模型内部信息 (Inspect)"""
    print(f"\n{'='*60}")
    print(f"🔍 Inspecting Blend File...")
    print(f"📦 Python bpy version: {bpy.app.version_string}")
    print(f"{'='*60}")

    armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']

    print(f"\n🦴 Armatures ({len(armatures)}):")
    for arm in armatures:
        print(f"  - {arm.name} (Bones: {len(arm.data.bones)})")

    print(f"\n📦 Meshes ({len(meshes)}):")
    for mesh in meshes:
        parent = mesh.parent.name if mesh.parent else 'None'
        print(f"  - {mesh.name} (Parent: {parent})")

    print(f"\n🎬 Actions ({len(bpy.data.actions)}):")
    for action in bpy.data.actions:
        frame_start, frame_end = action.frame_range
        # 检查 Blender 4.4+ 的 slots 属性，安全获取 slot 名称
        slots_info = ""
        if hasattr(action, "slots") and len(action.slots) > 0:
            slot_names = [getattr(s, 'name_display', getattr(s, 'identifier', 'Unknown')) for s in action.slots]
            slots_info = f" | Slots: {slot_names}"
        print(f"  - {action.name} (Frames: {int(frame_start)} to {int(frame_end)}){slots_info}")
    
    print(f"{'='*60}\n")
    return armatures

def main():
    # 兼容直接 python 运行 和 blender -b -P 运行
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Concat all actions into a single NLA track.")
    parser.add_argument('--input_blend', type=str, required=True, help="Path to the raw purchased .blend file")
    parser.add_argument('--output_blend', type=str, default=None, help="Path to save the concatenated .blend file (Optional)")
    parser.add_argument('--gap', type=int, default=10, help="Empty frames between actions (default: 10)")
    parser.add_argument('--exclude', type=str, default="", help="Comma-separated action names to exclude")
    args = parser.parse_args(argv)

    input_path = Path(args.input_blend)

    if not input_path.exists():
        raise FileNotFoundError(f"Input blend file not found: {input_path}")

    if args.output_blend:
        output_path = Path(args.output_blend)
    else:
        stem = input_path.stem
        if "_prep_" not in stem:
            stem += "_prep_01"  # 恢复为标准的 _prep_01 后缀
        output_path = input_path.parent / f"{stem}{input_path.suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading: {input_path}")
    bpy.ops.wm.open_mainfile(filepath=str(input_path))

    armatures = print_inspect_info()
    if not armatures:
        raise ValueError("No ARMATURE found in the scene!")
    
    arm_obj = armatures[0]
    print(f"🎯 Target Armature selected: {arm_obj.name}")

    # 强制切换到 POSE 模式，防止卡在 Rest Position
    arm_obj.data.pose_position = 'POSE'

    if not arm_obj.animation_data:
        arm_obj.animation_data_create()

    print("🧹 Clearing existing NLA tracks and active actions...")
    arm_obj.animation_data.action = None
    for track in reversed(arm_obj.animation_data.nla_tracks):
        arm_obj.animation_data.nla_tracks.remove(track)

    track = arm_obj.animation_data.nla_tracks.new()
    track.name = "OGD_Concat_Track"
    track.is_solo = False
    track.mute = False

    exclude_list = [name.strip() for name in args.exclude.split(",")] if args.exclude else []
    
    current_frame = 1
    added_count = 0

    print("🔗 Starting NLA concatenation...")
    for action in bpy.data.actions:
        if action.name in exclude_list:
            continue
            
        strip = track.strips.new(name=action.name, start=current_frame, action=action)
        
        # 核心修复：Blender 4.4+ Action Slot 绑定
        if hasattr(action, "slots") and len(action.slots) > 0:
            slot = action.slots[0]
            if hasattr(strip, "action_slot"):
                try:
                    strip.action_slot = slot
                except Exception as e:
                    print(f"    [Warning] Failed to bind action_slot: {e}")
            elif hasattr(strip, "action_slot_handle"):
                try:
                    strip.action_slot_handle = slot.handle
                except Exception as e:
                    print(f"    [Warning] Failed to bind action_slot_handle: {e}")

        # 显式设置 NLA 参数，防止隐式默认值导致不播放
        strip.blend_type = 'REPLACE'
        strip.extrapolation = 'NOTHING'
        strip.use_auto_blend = False
        strip.mute = False
        
        added_count += 1
        start, end = action.frame_range
        action_length = int(end - start)
        
        print(f"  ➕ Added: {action.name} (Start: {current_frame}, Length: {action_length})")
        current_frame = int(strip.frame_end) + args.gap

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = current_frame - args.gap
    
    print(f"\n✅ Concatenation complete! Added {added_count} actions.")
    print(f"⏱️ New scene frame range: 1 to {bpy.context.scene.frame_end}")

    print(f"💾 Saving to: {output_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    print("🎉 Done!")

if __name__ == "__main__":
    main()
