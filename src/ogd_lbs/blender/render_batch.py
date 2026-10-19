# filepath: /workspace/lao-lbs/src/lao_lbs/blender/render_batch.py
from __future__ import annotations
import argparse
import shutil
import traceback
from pathlib import Path
import bpy

# 修正为 lao_lbs 绝对导入
from ogd_lbs.utils.common import log, warn, fail, load_json, dump_json, resolve_under_root, safe_relpath, hex_to_rgb
from ogd_lbs.utils import pipeline
from ogd_lbs.blender.object import unhide_all_collections, ensure_object_mode
from ogd_lbs.blender.context import update_view_layer, force_ui_redraw
from ogd_lbs.blender.camera import setup_camera
from ogd_lbs.blender.armature import setup_armature_style, force_armature_viewport_ready, log_armature_visibility_debug

# 内部模块导入
from ogd_lbs.blender.render_passes import (
    setup_render_output, ensure_default_light, find_primary_armature, 
    find_mesh_objects, render_one_named_frame
)

def open_blend_keep_current_ui(blend_path: Path) -> None:
    log(f"Loading blend file: {blend_path}")
    try:
        result = bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
    except TypeError:
        warn("bpy.ops.wm.open_mainfile(load_ui=False) not supported; fallback to default open_mainfile")
        result = bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    if result != {"FINISHED"}:
        fail(f"Failed to open blend file: {blend_path}, result={result}")

    try:
        if hasattr(bpy.context.view_layer, "layer_collection"):
            unhide_all_collections(bpy.context.view_layer.layer_collection)
    except Exception:
        pass
    update_view_layer()
    force_ui_redraw()

def exec_render(blend_path: Path, out_dir: Path, args: argparse.Namespace) -> None:
    open_blend_keep_current_ui(blend_path)
    scene = bpy.context.scene

    setup_render_output(scene, args)
    if args.char_render_method == "FINAL":
        ensure_default_light(scene)

    cam_data = load_json(args.camera_json)
    
    setup_camera(
        cam_data,
        pullback=args.camera_pullback,
        res_preset=args.resolution_preset,
        custom_res=(args.custom_res_x, args.custom_res_y)
    )

    armature_obj = find_primary_armature(scene)
    mesh_objects = find_mesh_objects(scene)
    log(f"Found armature: {armature_obj.name}")
    log(f"Found mesh objects: {[obj.name for obj in mesh_objects]}")

    normal_rgb = hex_to_rgb(args.bone_color)
    select_rgb = hex_to_rgb(args.bone_select_color) if args.bone_select_color else normal_rgb
    active_rgb = hex_to_rgb(args.bone_active_color) if args.bone_active_color else normal_rgb

    setup_armature_style(
        armature_obj=armature_obj, bone_style=args.bone_style,
        normal_rgb=normal_rgb, select_rgb=select_rgb, active_rgb=active_rgb,
    )

    force_armature_viewport_ready(
        armature_obj, reveal_bones=bool(args.bone_force_reveal), final_mode=args.bone_view_mode,
    )
    log_armature_visibility_debug(armature_obj, prefix="Initial armature visibility after setup")
    ensure_object_mode()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.render_rest:
        log("Rendering Rest Pose")
        armature_obj.data.pose_position = "REST"
        scene.frame_set(scene.frame_start)
        update_view_layer()
        force_ui_redraw()
        render_one_named_frame(out_dir=out_dir, frame_name="rest", armature_obj=armature_obj, mesh_objects=mesh_objects, args=args)

    if args.render_animation:
        log("Rendering Animation Frames")
        armature_obj.data.pose_position = "POSE"
        update_view_layer()
        force_ui_redraw()

        target_frame_id = getattr(args, "frame_id", -1)
        if target_frame_id >= 0:
            log(f"Target frame specified: {target_frame_id}. Skipping animation loop.")
            scene.frame_set(target_frame_id)
            update_view_layer()
            force_ui_redraw()
            frame_name = f"frame_{target_frame_id:04d}"
            render_one_named_frame(out_dir=out_dir, frame_name=frame_name, armature_obj=armature_obj, mesh_objects=mesh_objects, args=args)
        else:
            frame_start = int(scene.frame_start)
            frame_end = int(scene.frame_end)
            if args.frame_start is not None:
                frame_start = max(frame_start, int(args.frame_start))
            if args.frame_end is not None:
                frame_end = min(frame_end, int(args.frame_end))

            if frame_end < frame_start:
                warn(f"Empty frame range: {frame_start}..{frame_end}")
                return

            rendered_frame_count = 0
            for frame in range(frame_start, frame_end + 1):
                scene.frame_set(frame)
                update_view_layer()
                force_ui_redraw()
                frame_name = f"frame_{frame:04d}"
                render_one_named_frame(out_dir=out_dir, frame_name=frame_name, armature_obj=armature_obj, mesh_objects=mesh_objects, args=args)
                rendered_frame_count += 1
                if args.frame_process_max > 0 and rendered_frame_count >= args.frame_process_max:
                    log(f"frame_process_max reached: {args.frame_process_max}")
                    break

def process_leaf_render(source_dir_abs: Path, target_dir_abs: Path, args: argparse.Namespace) -> None:
    done_path = target_dir_abs / "done"
    if done_path.exists():
        if args.overwrite:
            warn(f"Overwrite enabled; removing old done mark: {done_path}")
            if done_path.is_dir():
                shutil.rmtree(done_path)
            else:
                done_path.unlink()
        else:
            log(f"Pass already done: {target_dir_abs}")
            return

    config_path = source_dir_abs / "config_local_iter.json"
    if not config_path.exists():
        warn(f"Leaf config_local_iter.json not found, skip: {config_path}")
        return

    config_obj = load_json(config_path)
    blend_rel_path = config_obj.get("step2_out_blend_file_rel2oup")
    if not blend_rel_path:
        fail(f"Missing 'step2_out_blend_file_rel2oup' in config: {config_path}")

    blend_abs_path = resolve_under_root(args.input_root, blend_rel_path)
    if not blend_abs_path.exists():
        fail(f"Blend file not found: {blend_abs_path}")

    target_dir_abs.mkdir(parents=True, exist_ok=True)
    log("-" * 60)
    log(f"Processing Render for Iteration: {target_dir_abs.name}")
    log(f"Source leaf: {source_dir_abs}")
    log(f"Target leaf: {target_dir_abs}")
    log(f"Blend: {blend_abs_path}")

    exec_render(blend_path=blend_abs_path, out_dir=target_dir_abs, args=args)

    target_config_obj = config_obj.copy()
    target_config_obj.update({
        "step3_out_render_dir_rel2oup": safe_relpath(target_dir_abs, args.output_root),
        "step3_camera_json_abs": str(args.camera_json),
        "step3_camera_pullback": args.camera_pullback,
        "step3_resolution_preset": args.resolution_preset,
        "step3_custom_res_x": args.custom_res_x,
        "step3_custom_res_y": args.custom_res_y,
        "step3_image_format": getattr(args, "image_format", "PNG"),
        "step3_render_mode": getattr(args, "render_mode", "SEPARATE"),
        "step3_frame_id": getattr(args, "frame_id", -1),
        "step3_bone_style": args.bone_style,
        "step3_bone_color": args.bone_color,
        "step3_bone_select_color": args.bone_select_color,
        "step3_bone_active_color": args.bone_active_color,
        "step3_render_engine": args.render_engine,
        "step3_char_render_method": args.char_render_method,
        "step3_char_viewport_shading": args.char_viewport_shading,
        "step3_bone_viewport_shading": args.viewport_shading,
        "step3_opengl_view_context_legacy": bool(args.opengl_view_context),
        "step3_char_opengl_view_context": bool(args.char_opengl_view_context),
        "step3_bone_opengl_view_context": bool(args.bone_opengl_view_context),
        "step3_bone_view_mode": args.bone_view_mode,
        "step3_bone_force_reveal": bool(args.bone_force_reveal),
        "step3_maximize_view3d": bool(args.maximize_view3d),
    })

    dump_json(target_dir_abs / "config_local_iter.json", target_config_obj)
    done_path.touch()
    args.process_n += 1
    log(f"Done: {target_dir_abs}\n")

def should_process_iter_dir(iter_dir: Path, args: argparse.Namespace, iter_max: int) -> bool:
    iter_id, _ = pipeline.parse_iter_name(iter_dir, iter_max)
    
    if getattr(args, "only_iter_max", False):
        if iter_dir.name == "opt_out":
            return True
        if iter_max >= 0 and iter_id == iter_max:
            return True
        return False

    if args.iter_id < 0:
        return True
    if iter_dir.name == "opt_out":
        return bool(args.include_opt_out)
    return iter_id == args.iter_id

def process_directory(
    source_dir_abs: Path,
    target_dir_abs: Path,
    depth: int,
    iter_max: int,
    args: argparse.Namespace,
) -> None:
    if 0 < args.process_max <= args.process_n:
        return
    if not source_dir_abs.exists():
        warn(f"Source dir does not exist, skip: {source_dir_abs}")
        return

    target_dir_abs.mkdir(parents=True, exist_ok=True)

    if depth < args.config_depth - 1:
        for item in sorted(source_dir_abs.iterdir(), key=lambda p: p.name):
            if not item.is_dir():
                continue
            process_directory(source_dir_abs=item, target_dir_abs=target_dir_abs / item.name, depth=depth + 1, iter_max=iter_max, args=args)
            if 0 < args.process_max <= args.process_n:
                return

    elif depth == args.config_depth - 1:
        sub_dirs = [p for p in source_dir_abs.iterdir() if p.is_dir()]
        if not sub_dirs:
            warn(f"No Step2 out_* subdirectories under: {source_dir_abs}")
            return
            
        newest_source_sub_dir = sorted(sub_dirs, key=lambda p: p.name)[-1]
        
        execution_out_dir = pipeline.resolve_execution_dir(target_dir_abs, getattr(args, "new_out_dir", False))
        execution_out_dir.mkdir(parents=True, exist_ok=True)
        log(f"Step 3 execution dir: {execution_out_dir}")

        process_directory(source_dir_abs=newest_source_sub_dir, target_dir_abs=execution_out_dir, depth=depth + 1, iter_max=iter_max, args=args)

    elif depth == args.config_depth:
        config_path = source_dir_abs / "config_local.json"
        if not config_path.exists():
            warn(f"Config at config_depth not found, skip: {config_path}")
            return
            
        ssdr_config = load_json(config_path)
        dump_json(target_dir_abs / "config_local.json", ssdr_config)

        iter_max_next = int(ssdr_config.get("iter", ssdr_config.get("iter_max", -1)))

        iter_dirs = [p for p in source_dir_abs.iterdir() if p.is_dir()]
        if not iter_dirs:
            warn(f"No iteration directories under: {source_dir_abs}")
            return
            
        def sort_key(p: Path):
            if p.name == "opt_out": return 999999999
            return pipeline.parse_iter_name(p, -1)[0]
            
        sorted_iter_dirs = sorted(iter_dirs, key=sort_key)
        
        dirs_to_process = [p for p in sorted_iter_dirs if should_process_iter_dir(p, args, iter_max_next)]

        if not dirs_to_process:
            warn(f"No matching iteration dirs under {source_dir_abs}; iter_id={args.iter_id}, only_iter_max={getattr(args, 'only_iter_max', False)}")
            return

        for iter_dir in dirs_to_process:
            process_directory(source_dir_abs=iter_dir, target_dir_abs=target_dir_abs / iter_dir.name, depth=depth + 1, iter_max=iter_max_next, args=args)
            if 0 < args.process_max <= args.process_n:
                return

    elif depth == args.config_depth + 1:
        try:
            process_leaf_render(source_dir_abs=source_dir_abs, target_dir_abs=target_dir_abs, args=args)
        except Exception as exc:
            if args.continue_on_error:
                warn(f"Failed leaf render but continue_on_error enabled: {source_dir_abs}")
                warn(str(exc))
                traceback.print_exc()
            else:
                raise