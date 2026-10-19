# filepath: /workspace/lao-lbs/src/lao_lbs/blender/render_passes.py
from __future__ import annotations

import argparse
from pathlib import Path

import bpy

# 修正为 lao_lbs 绝对导入
from ogd_lbs.utils.common import log, warn, fail, Timer
from ogd_lbs.blender.object import ensure_object_mode, deselect_all_objects, unhide_all_collections
from ogd_lbs.blender.context import update_view_layer, force_ui_redraw, make_temp_override_context
from ogd_lbs.blender.armature import force_armature_viewport_ready, log_armature_visibility_debug
from ogd_lbs.blender.image_io import (
    get_image_format,
    get_image_extension,
    configure_scene_for_png_capture,
    configure_scene_for_final_format,
    make_temp_capture_path,
    finalize_captured_png,
    remove_stale_alternative_outputs,
    cleanup_temp_files_for_final,
    assert_final_output_exists,
    require_cwebp,
)

# 内部模块导入
from ogd_lbs.blender.viewport_hack import configure_view3d_for_camera_render


# -----------------------------------------------------------------------------
# Render engine / output setup
# -----------------------------------------------------------------------------

def set_render_engine(scene: bpy.types.Scene, requested_engine: str = "AUTO") -> str:
    requested_engine = requested_engine.upper()

    if requested_engine != "AUTO":
        candidates = [requested_engine]
    else:
        if bpy.app.version >= (4, 2, 0):
            candidates = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"]
        else:
            candidates = ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"]

    last_error = None

    for engine in candidates:
        try:
            scene.render.engine = engine
            log(f"Render engine set to: {scene.render.engine}")
            return scene.render.engine
        except Exception as exc:
            last_error = exc
            warn(f"Render engine not available or failed: {engine} ({exc})")

    fail(f"Could not set any render engine from candidates={candidates}. Last error={last_error}")
    return ""


def setup_render_output(scene: bpy.types.Scene, args: argparse.Namespace) -> None:
    set_render_engine(scene, args.render_engine)

    scene.render.film_transparent = True
    scene.render.use_file_extension = False
    scene.render.resolution_percentage = 100

    final_fmt = get_image_format(args)

    if final_fmt == "WEBP":
        require_cwebp()

    configure_scene_for_final_format(scene, final_fmt)

    if final_fmt == "WEBP":
        log(
            "Final image format requested: WEBP. "
            "Each pass will capture temporary PNG first, then encode lossless WEBP via cwebp."
        )
    else:
        log("Final image format requested: PNG.")

    try:
        scene.eevee.taa_render_samples = args.eevee_samples
    except Exception:
        pass

    try:
        scene.eevee.taa_samples = args.eevee_samples
    except Exception:
        pass


def ensure_default_light(scene: bpy.types.Scene) -> None:
    lights = [obj for obj in scene.objects if obj.type == "LIGHT"]

    if lights:
        return

    light_data = bpy.data.lights.new(name="DefaultSun", type="SUN")
    light_data.energy = 2.0

    light_obj = bpy.data.objects.new(name="DefaultSun", object_data=light_data)
    scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = (0.785, 0.0, 0.785)

    log("No lights found in scene. Added default Sun light for FINAL render.")


# -----------------------------------------------------------------------------
# Scene discovery
# -----------------------------------------------------------------------------

def find_primary_armature(scene: bpy.types.Scene) -> bpy.types.Object:
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]

    if not armatures:
        fail("No armature found in the scene")

    if len(armatures) > 1:
        warn(f"Multiple armatures found; using first one: {[obj.name for obj in armatures]}")

    return armatures[0]


def find_mesh_objects(scene: bpy.types.Scene) -> list[bpy.types.Object]:
    return [obj for obj in scene.objects if obj.type == "MESH"]


# -----------------------------------------------------------------------------
# Pass visibility
# -----------------------------------------------------------------------------

def set_character_pass_visibility(
    armature_obj: bpy.types.Object,
    mesh_objects: list[bpy.types.Object],
) -> None:
    ensure_object_mode()

    try:
        armature_obj.hide_set(True)
    except Exception:
        pass

    armature_obj.hide_viewport = True
    armature_obj.hide_render = True

    for obj in mesh_objects:
        try:
            obj.hide_set(False)
        except Exception:
            pass

        obj.hide_viewport = False
        obj.hide_render = False

    deselect_all_objects()
    update_view_layer()
    force_ui_redraw()


def set_bone_pass_visibility(
    armature_obj: bpy.types.Object,
    mesh_objects: list[bpy.types.Object],
    args: argparse.Namespace,
) -> None:
    ensure_object_mode()

    try:
        if hasattr(bpy.context.view_layer, "layer_collection"):
            unhide_all_collections(bpy.context.view_layer.layer_collection)
    except Exception:
        pass

    for obj in mesh_objects:
        try:
            obj.hide_set(True)
        except Exception:
            pass

        obj.hide_viewport = True
        obj.hide_render = True

    force_armature_viewport_ready(
        armature_obj,
        reveal_bones=bool(args.bone_force_reveal),
        final_mode=args.bone_view_mode,
    )

    log_armature_visibility_debug(
        armature_obj,
        prefix="Bone pass visibility after force reset",
    )

    update_view_layer()
    force_ui_redraw()


def set_fused_pass_visibility(
    armature_obj: bpy.types.Object,
    mesh_objects: list[bpy.types.Object],
    args: argparse.Namespace,
) -> None:
    ensure_object_mode()

    try:
        if hasattr(bpy.context.view_layer, "layer_collection"):
            unhide_all_collections(bpy.context.view_layer.layer_collection)
    except Exception:
        pass

    for obj in mesh_objects:
        try:
            obj.hide_set(False)
        except Exception:
            pass
        obj.hide_viewport = False
        obj.hide_render = False

    force_armature_viewport_ready(
        armature_obj,
        reveal_bones=bool(args.bone_force_reveal),
        final_mode=args.bone_view_mode,
    )

    try:
        armature_obj.show_in_front = True
    except Exception:
        pass

    log_armature_visibility_debug(
        armature_obj,
        prefix="Fused pass visibility after force reset",
    )

    update_view_layer()
    force_ui_redraw()


# -----------------------------------------------------------------------------
# Render passes
# -----------------------------------------------------------------------------

def prepare_final_path(final_path: Path) -> None:
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    remove_stale_alternative_outputs(final_path)
    cleanup_temp_files_for_final(final_path)

    if final_path.exists():
        try:
            warn(f"Removing existing final output before render: {final_path}")
            final_path.unlink()
        except Exception as exc:
            warn(f"Failed to remove existing final output: {final_path}: {exc}")


def render_regular_pass(
    filepath: Path,
    args: argparse.Namespace,
    *,
    pass_name: str = "REGULAR",
) -> None:
    filepath = Path(filepath)
    prepare_final_path(filepath)

    scene = bpy.context.scene
    final_fmt = get_image_format(args)
    tmp_png = make_temp_capture_path(filepath, pass_name=pass_name)

    try:
        if tmp_png.exists():
            tmp_png.unlink()
    except Exception:
        pass

    configure_scene_for_png_capture(scene)
    scene.render.filepath = str(tmp_png)

    log(f"{pass_name}: regular render capture PNG path: {tmp_png}")
    log(f"{pass_name}: final output path: {filepath}, final_format={final_fmt}")

    with Timer(f"[T2] Regular Render ({pass_name})"):
        result = bpy.ops.render.render(write_still=True)

    if result != {"FINISHED"}:
        warn(f"{pass_name}: Regular render returned non-FINISHED result: {result}")

    if not tmp_png.exists():
        fail(f"{pass_name}: Regular render did not create capture PNG: {tmp_png}")

    finalize_captured_png(
        capture_png_path=tmp_png,
        final_path=filepath,
        image_format=final_fmt,
        remove_input=True,
    )

    assert_final_output_exists(filepath, final_fmt)

    log(f"{pass_name}: regular render final saved: {filepath}")


def render_opengl_pass(
    filepath: Path,
    args: argparse.Namespace,
    shading_type: str,
    *,
    pass_name: str,
    want_armature: bool,
    want_mesh: bool,
    view_context: bool,
) -> None:
    if bpy.app.background:
        fail(
            "Cannot render with bpy.ops.render.opengl() because bpy.app.background is True. "
            "Run with xvfb-run blender --python, without -b/--background."
        )

    filepath = Path(filepath)
    prepare_final_path(filepath)

    scene = bpy.context.scene
    final_fmt = get_image_format(args)
    tmp_png = make_temp_capture_path(filepath, pass_name=pass_name)

    try:
        if tmp_png.exists():
            tmp_png.unlink()
    except Exception:
        pass

    configure_scene_for_png_capture(scene)
    scene.render.filepath = str(tmp_png)

    log(f"{pass_name}: OpenGL capture PNG path: {tmp_png}")
    log(f"{pass_name}: final output path: {filepath}, final_format={final_fmt}")

    ctx = configure_view3d_for_camera_render(
        args,
        shading_type=shading_type,
        pass_name=pass_name,
        want_armature=want_armature,
        want_mesh=want_mesh,
    )

    force_ui_redraw(iterations=3)

    try:
        with bpy.context.temp_override(**make_temp_override_context(ctx)):
            with Timer(f"[T1] OpenGL Render ({pass_name})"):
                result = bpy.ops.render.opengl(
                    animation=False,
                    sequencer=False,
                    write_still=True,
                    view_context=bool(view_context),
                )

        if result != {"FINISHED"}:
            warn(f"{pass_name}: OpenGL render returned non-FINISHED result: {result}")

    except Exception as exc:
        warn(
            f"{pass_name}: OpenGL viewport render failed. "
            "Make sure you are running with xvfb-run and without -b/--background. "
            f"Original error: {exc}"
        )
        raise

    if not tmp_png.exists():
        fail(
            f"{pass_name}: OpenGL render did not create expected capture PNG: {tmp_png}. "
            "Check scene.render.filepath and Blender output settings."
        )

    log(
        f"{pass_name}: OpenGL render capture saved: {tmp_png}, "
        f"shading={shading_type}, view_context={view_context}, "
        f"want_armature={want_armature}, want_mesh={want_mesh}"
    )

    finalize_captured_png(
        capture_png_path=tmp_png,
        final_path=filepath,
        image_format=final_fmt,
        remove_input=True,
    )

    assert_final_output_exists(filepath, final_fmt)

    log(f"{pass_name}: final image saved and validated: {filepath}")


def render_one_named_frame(
    out_dir: Path,
    frame_name: str,
    armature_obj: bpy.types.Object,
    mesh_objects: list[bpy.types.Object],
    args: argparse.Namespace,
) -> None:
    with Timer(f"[T3] Render Frame Total: {frame_name}"):
        ext = get_image_extension(args)
        render_mode = getattr(args, "render_mode", "SEPARATE").upper()

        if render_mode == "FUSED":
            fused_path = out_dir / f"{frame_name}_fused{ext}"
            log(f"[{frame_name}] Target fused path: {fused_path}")
            
            set_fused_pass_visibility(armature_obj, mesh_objects, args)
            
            render_opengl_pass(
                fused_path,
                args,
                shading_type=args.char_viewport_shading,
                pass_name=f"{frame_name}_FUSED",
                want_armature=True,
                want_mesh=True,
                view_context=True,
            )
            log(f"Rendered fused frame: {frame_name}")
        else:
            char_path = out_dir / f"{frame_name}_char{ext}"
            bone_path = out_dir / f"{frame_name}_bone{ext}"

            log(f"[{frame_name}] Target char path: {char_path}")
            log(f"[{frame_name}] Target bone path: {bone_path}")

            # Character layer.
            set_character_pass_visibility(armature_obj, mesh_objects)

            if args.char_render_method == "OPENGL":
                render_opengl_pass(
                    char_path,
                    args,
                    shading_type=args.char_viewport_shading,
                    pass_name=f"{frame_name}_CHAR",
                    want_armature=False,
                    want_mesh=True,
                    view_context=bool(args.char_opengl_view_context),
                )
            else:
                render_regular_pass(
                    char_path,
                    args,
                    pass_name=f"{frame_name}_CHAR_FINAL",
                )

            # Bone layer.
            set_bone_pass_visibility(armature_obj, mesh_objects, args)

            render_opengl_pass(
                bone_path,
                args,
                shading_type=args.viewport_shading,
                pass_name=f"{frame_name}_BONE",
                want_armature=True,
                want_mesh=False,
                view_context=bool(args.bone_opengl_view_context),
            )

            log(f"Rendered layered frame: {frame_name}")