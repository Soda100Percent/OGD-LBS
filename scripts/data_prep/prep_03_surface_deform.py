#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 03: Manifest-driven Surface Deform binding + LBS weight transfer.

Default drive mode:
    BOTH

When --drive_mode BOTH:
    - bind / transfer once
    - save Surface-driven blend:
        <work_dir>/<base_stem>_prep_03_surface_drive_mode.blend
    - switch modifiers to LBS mode
    - save LBS-driven blend:
        <work_dir>/<base_stem>_prep_03_lbs_drive_mode.blend

When --drive_mode SURFACE or LBS:
    - save only one blend
    - --output_blend may override the single output path

Path behavior:
    required:
        --input_blend

    optional:
        --manifest
            If omitted:
                <work_dir>/<base_stem>_prep_01_sdf_exports/manifest.json

        --output_blend
            Valid only for single-output modes SURFACE / LBS.

        --output_blend_surface
        --output_blend_lbs
            Valid only for --drive_mode BOTH.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def ensure_repo_src_in_syspath() -> None:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src_dir = parent / "src"
            src_s = str(src_dir)
            if src_s not in sys.path:
                sys.path.insert(0, src_s)
            return


ensure_repo_src_in_syspath()

import bpy  # noqa: E402

from ogd_lbs.utils.common import script_argv  # noqa: E402
from ogd_lbs.blender.object import ensure_object_mode, find_armature  # noqa: E402
from ogd_lbs.blender.context import align_scene_pose_frame, update_view_layer  # noqa: E402
from ogd_lbs.blender.surface_deform import (  # noqa: E402
    process_surface_deform_pair,
    apply_drive_mode_to_target,
    print_modifier_summary,
)

from paths import (  # noqa: E402
    make_data_prep_context,
    prep_02_manifest_rel,
    prep_03_output_blend_rel,
    prep_03_surface_output_blend_rel,
    prep_03_lbs_output_blend_rel,
    validate_drive_mode,
)
from manifest import (  # noqa: E402
    load_manifest,
    require_vdb_mapping,
    get_source_objects_mapping,
)


def find_required_mesh(name: str) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)

    if obj is None:
        raise RuntimeError(f"Required mesh object not found: {name}")

    if obj.type != "MESH":
        raise RuntimeError(f"Required object is not MESH: {name}, type={obj.type}")

    return obj


def save_blend(path: Path) -> None:
    print(f"\n💾 Saving to: {path}", flush=True)
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path))
    if result != {"FINISHED"}:
        raise RuntimeError(f"Failed to save blend file: {path}, result={result}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prep 03: manifest-driven Surface Deform + LBS transfer."
    )

    parser.add_argument("--input_blend", type=str, required=True)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Manifest from Prep 02a/02b. Default: <work_dir>/<base_stem>_prep_01_sdf_exports/manifest.json",
    )

    parser.add_argument(
        "--output_blend",
        type=str,
        default=None,
        help="Single-output override. Valid only when --drive_mode is SURFACE or LBS.",
    )
    parser.add_argument(
        "--output_blend_surface",
        type=str,
        default=None,
        help="Surface output override. Valid only when --drive_mode BOTH.",
    )
    parser.add_argument(
        "--output_blend_lbs",
        type=str,
        default=None,
        help="LBS output override. Valid only when --drive_mode BOTH.",
    )

    parser.add_argument(
        "--bind_frame",
        type=int,
        default=None,
        help="Override bind frame. If omitted, use manifest target_frame.",
    )
    parser.add_argument(
        "--drive_mode",
        choices=["SURFACE", "LBS", "BOTH"],
        default="BOTH",
    )

    parser.add_argument("--max_falloff", type=float, default=16.0)
    parser.add_argument("--falloff_start", type=float, default=4.0)
    parser.add_argument("--falloff_step", type=float, default=2.0)
    parser.add_argument("--weight_eps", type=float, default=1e-8)

    args = parser.parse_args(script_argv())

    drive_mode = validate_drive_mode(args.drive_mode)

    if drive_mode == "BOTH" and args.output_blend is not None:
        raise RuntimeError(
            "--output_blend is only valid for single-output modes SURFACE or LBS. "
            "Use --output_blend_surface and/or --output_blend_lbs with --drive_mode BOTH."
        )

    if drive_mode != "BOTH" and (args.output_blend_surface is not None or args.output_blend_lbs is not None):
        raise RuntimeError(
            "--output_blend_surface and --output_blend_lbs are only valid with --drive_mode BOTH."
        )

    ctx = make_data_prep_context(args.input_blend)

    input_path = ctx.resolve_input(
        "input_blend",
        args.input_blend,
        None,
        expected_type="file",
    )
    manifest_path = ctx.resolve_input(
        "manifest",
        args.manifest,
        prep_02_manifest_rel(ctx),
        expected_type="file",
    )

    output_surface_path: Path | None = None
    output_lbs_path: Path | None = None
    output_single_path: Path | None = None

    if drive_mode == "BOTH":
        output_surface_path = ctx.resolve_output(
            "output_blend_surface",
            args.output_blend_surface,
            prep_03_surface_output_blend_rel(ctx),
            expected_type="file",
        )
        output_lbs_path = ctx.resolve_output(
            "output_blend_lbs",
            args.output_blend_lbs,
            prep_03_lbs_output_blend_rel(ctx),
            expected_type="file",
        )
    else:
        output_single_path = ctx.resolve_output(
            "output_blend",
            args.output_blend,
            prep_03_output_blend_rel(ctx, drive_mode=drive_mode),
            expected_type="file",
        )

    ctx.print_and_validate("Prep 03: Surface Deform + LBS Drive Modes")

    manifest_obj = load_manifest(manifest_path)
    vdb_mapping = require_vdb_mapping(manifest_obj)
    source_objects = get_source_objects_mapping(manifest_obj, vdb_mapping)

    print(f"📂 Loading: {input_path}", flush=True)
    result = bpy.ops.wm.open_mainfile(filepath=str(input_path))
    if result != {"FINISHED"}:
        raise RuntimeError(f"Failed to open blend file: {input_path}, result={result}")

    ensure_object_mode()

    armature = find_armature(required=True, prefer_name="Armature")
    if armature is None:
        raise RuntimeError("No ARMATURE found in the scene.")

    bind_frame = int(args.bind_frame) if args.bind_frame is not None else int(manifest_obj.get("target_frame", -1))

    print(f"\n🛑 Aligning scene to bind_frame={bind_frame}...", flush=True)
    align_scene_pose_frame(bind_frame, armature=armature, set_all_armatures=True)

    print("\n🎯 Core armature:", flush=True)
    print(f"   {armature.name} (bones={len(armature.data.bones)})", flush=True)

    print("\n🔎 Resolving VDB pairs from manifest:", flush=True)

    target_meshes: list[bpy.types.Object] = []

    for vdb_name in vdb_mapping.keys():
        source_name = source_objects.get(vdb_name, f"Source_{vdb_name}")

        source_mesh = find_required_mesh(source_name)
        target_mesh = find_required_mesh(vdb_name)

        print(f"   {source_mesh.name} -> {target_mesh.name}", flush=True)

        target_meshes.append(target_mesh)

    initial_single_drive = "SURFACE" if drive_mode == "BOTH" else drive_mode

    for vdb_name in vdb_mapping.keys():
        source_name = source_objects.get(vdb_name, f"Source_{vdb_name}")

        source_mesh = find_required_mesh(source_name)
        target_mesh = find_required_mesh(vdb_name)

        process_surface_deform_pair(
            armature=armature,
            source_mesh=source_mesh,
            target_mesh=target_mesh,
            drive_mode=initial_single_drive,
            max_falloff=float(args.max_falloff),
            falloff_start=float(args.falloff_start),
            falloff_step=float(args.falloff_step),
            weight_eps=float(args.weight_eps),
        )

    print("\n▶️ Setting Armature to POSE mode for playback...", flush=True)
    armature.data.pose_position = "POSE"
    update_view_layer()

    if drive_mode == "BOTH":
        if output_surface_path is None or output_lbs_path is None:
            raise RuntimeError("Internal error: BOTH outputs were not resolved.")

        print("\n🎛️ Applying SURFACE drive mode to all VDB targets...", flush=True)
        for target_mesh in target_meshes:
            apply_drive_mode_to_target(target_mesh, drive_mode="SURFACE")
            print_modifier_summary(target_mesh)

        save_blend(output_surface_path)

        print("\n🎛️ Applying LBS drive mode to all VDB targets...", flush=True)
        for target_mesh in target_meshes:
            apply_drive_mode_to_target(target_mesh, drive_mode="LBS")
            print_modifier_summary(target_mesh)

        save_blend(output_lbs_path)

        print("\n🎉 Prep 03 done. BOTH drive-mode outputs saved.", flush=True)
        print(f"   surface: {output_surface_path}", flush=True)
        print(f"   lbs:     {output_lbs_path}", flush=True)

    else:
        if output_single_path is None:
            raise RuntimeError("Internal error: single output was not resolved.")

        print(f"\n🎛️ Applying {drive_mode} drive mode to all VDB targets...", flush=True)
        for target_mesh in target_meshes:
            apply_drive_mode_to_target(target_mesh, drive_mode=drive_mode)
            print_modifier_summary(target_mesh)

        save_blend(output_single_path)

        print(f"\n🎉 Prep 03 done. {drive_mode} drive-mode output saved.", flush=True)
        print(f"   output: {output_single_path}", flush=True)


if __name__ == "__main__":
    main()
