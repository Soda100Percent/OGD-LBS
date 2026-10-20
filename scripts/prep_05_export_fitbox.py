#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 05: Manifest-driven dual Fitbox export.

Exports:

    <out_dir>/fitbox_original/
        skeleton_hierarchy_data.pkl
        animation_data.pkl
        Source_<VDB>.pkl
        <unmerged>.pkl

    <out_dir>/fitbox_vdb/
        skeleton_hierarchy_data.pkl
        animation_data.pkl
        skeleton_hierarchy_data_bone_reduced.pkl
        animation_data_bone_reduced.pkl
        <VDB>.pkl
        <unmerged>.pkl

Debug support:
    --debug_frame_max X
        Export only the first X frames from the resolved frame range.

    --frame_start N
        Override start frame.

    --frame_end N
        Override end frame, inclusive.

Notes:
    - Mesh weight matrix columns are based on Armature_Original.
    - No weight repair / normalization is performed here.
    - Baked animation vertices are taken from evaluated depsgraph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy

from ogd_lbs.fitbox.blender_io import (
    export_skeleton,
    export_animation,
    export_mesh,
    get_nla_frame_range,
)


def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def load_config_mapping(config_path: Path | None) -> dict[str, list[str]]:
    if config_path is None:
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    return {str(k): list(v.get("source_meshes", [])) for k, v in obj.items()}


def ensure_manifest_mapping(manifest: dict[str, Any], config_path: Path | None) -> dict[str, list[str]]:
    mapping = manifest.get("vdb_mapping", None)

    if isinstance(mapping, dict) and mapping:
        return {str(k): list(v) for k, v in mapping.items()}

    config_mapping = load_config_mapping(config_path)

    if config_mapping:
        print("⚠️ manifest has no vdb_mapping; using --config source_meshes as fallback.", flush=True)
        return config_mapping

    raise RuntimeError(
        "manifest.json does not contain non-empty 'vdb_mapping'. "
        "Please rerun fixed prep_02a or pass --config."
    )


def find_armature(name: str, *, required: bool = True) -> bpy.types.Object | None:
    obj = bpy.data.objects.get(name)

    if obj is not None and obj.type == "ARMATURE":
        return obj

    if required:
        raise RuntimeError(f"Required armature not found: {name}")

    return None


def find_mesh(name: str, *, required: bool = True) -> bpy.types.Object | None:
    obj = bpy.data.objects.get(name)

    if obj is not None and obj.type == "MESH":
        return obj

    if required:
        raise RuntimeError(f"Required mesh not found: {name}")

    return None


def unique_keep_order(names: list[str]) -> list[str]:
    out = []
    seen = set()

    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)

    return out


def resolve_export_frame_range(
    *,
    armature: bpy.types.Object,
    frame_start_arg: int | None,
    frame_end_arg: int | None,
    debug_frame_max: int,
) -> tuple[int, int]:
    full_start, full_end = get_nla_frame_range(armature)

    start = int(frame_start_arg) if frame_start_arg is not None else int(full_start)

    if frame_end_arg is not None:
        end = int(frame_end_arg) + 1
    else:
        end = int(full_end)

    if start < full_start:
        print(
            f"⚠️ Requested frame_start={start} is before full NLA start={full_start}; using requested value anyway.",
            flush=True,
        )

    if end > full_end:
        print(
            f"⚠️ Requested end_exclusive={end} is after full NLA end_exclusive={full_end}; using requested value anyway.",
            flush=True,
        )

    if debug_frame_max > 0:
        old_end = end
        end = min(end, start + int(debug_frame_max))
        print(
            f"🐞 Debug frame limit enabled: debug_frame_max={debug_frame_max}, "
            f"frame range clipped from {start}..{old_end - 1} to {start}..{end - 1}",
            flush=True,
        )

    if end <= start:
        raise RuntimeError(
            f"Invalid resolved export frame range: start={start}, end_exclusive={end}. "
            f"Full NLA range was {full_start}..{full_end - 1}."
        )

    print(
        f"🎞️ Export frame range resolved: {start}..{end - 1} "
        f"(count={end - start}); full NLA range={full_start}..{full_end - 1}",
        flush=True,
    )

    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep 05: dual Fitbox export.")

    parser.add_argument("--input_blend", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Fallback config if manifest lacks vdb_mapping.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--format", choices=["pkl", "json"], default="pkl")

    parser.add_argument(
        "--debug_frame_max",
        type=int,
        default=-1,
        help="Debug mode: export only the first X frames from the resolved frame range. -1 disables.",
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        default=None,
        help="Optional explicit start frame.",
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=None,
        help="Optional explicit end frame, inclusive.",
    )
    parser.add_argument(
        "--animation_progress_every",
        type=int,
        default=100,
        help="Print animation export progress every N frames.",
    )
    parser.add_argument(
        "--mesh_progress_every",
        type=int,
        default=25,
        help="Print mesh baking progress every N frames.",
    )

    args = parser.parse_args(script_argv())

    input_path = Path(args.input_blend)
    manifest_path = Path(args.manifest)
    config_path = Path(args.config) if args.config else None
    out_root = Path(args.out_dir)

    print(f"📂 Loading: {input_path}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(input_path))

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    vdb_mapping = ensure_manifest_mapping(manifest, config_path)
    vdb_names = list(vdb_mapping.keys())
    source_objects = manifest.get("source_objects", {})

    source_names = []
    for vdb_name in vdb_names:
        source_names.append(source_objects.get(vdb_name, f"Source_{vdb_name}"))

    all_mesh_names = [obj.name for obj in bpy.data.objects if obj.type == "MESH"]

    protected = set(vdb_names) | set(source_names)
    manifest_unmerged = manifest.get("unmerged_meshes", None)

    if isinstance(manifest_unmerged, list):
        unmerged_names = [name for name in manifest_unmerged if name in all_mesh_names]
    else:
        unmerged_names = [name for name in all_mesh_names if name not in protected]

    source_names = unique_keep_order([name for name in source_names if name in all_mesh_names])
    vdb_names = unique_keep_order([name for name in vdb_names if name in all_mesh_names])
    unmerged_names = unique_keep_order(unmerged_names)

    print("\n📋 Export classification:", flush=True)
    print(f"   Source meshes:   {source_names}", flush=True)
    print(f"   VDB meshes:      {vdb_names}", flush=True)
    print(f"   Unmerged meshes: {unmerged_names}", flush=True)

    arm_orig = find_armature("Armature_Original", required=True)
    arm_clean = find_armature("Armature_Cleaned", required=True)

    frame_range = resolve_export_frame_range(
        armature=arm_orig,
        frame_start_arg=args.frame_start,
        frame_end_arg=args.frame_end,
        debug_frame_max=int(args.debug_frame_max),
    )

    out_original = out_root / "fitbox_original"
    out_vdb = out_root / "fitbox_vdb"
    out_original.mkdir(parents=True, exist_ok=True)
    out_vdb.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    ext = fmt

    print("\n🚀 Exporting fitbox_original...", flush=True)
    export_skeleton(
        arm_orig,
        out_original / f"skeleton_hierarchy_data.{ext}",
        fmt=fmt,
    )
    export_animation(
        arm_orig,
        out_original / f"animation_data.{ext}",
        fmt=fmt,
        frame_range=frame_range,
        progress_every=int(args.animation_progress_every),
    )

    for mesh_name in source_names + unmerged_names:
        mesh = find_mesh(mesh_name, required=True)
        export_mesh(
            mesh,
            arm_orig,
            out_original / f"{mesh_name}.{ext}",
            fmt=fmt,
            frame_range=frame_range,
            progress_every=int(args.mesh_progress_every),
        )

    print("\n🚀 Exporting fitbox_vdb...", flush=True)
    export_skeleton(
        arm_orig,
        out_vdb / f"skeleton_hierarchy_data.{ext}",
        fmt=fmt,
    )
    export_animation(
        arm_orig,
        out_vdb / f"animation_data.{ext}",
        fmt=fmt,
        frame_range=frame_range,
        progress_every=int(args.animation_progress_every),
    )

    export_skeleton(
        arm_clean,
        out_vdb / f"skeleton_hierarchy_data_bone_reduced.{ext}",
        fmt=fmt,
    )
    export_animation(
        arm_clean,
        out_vdb / f"animation_data_bone_reduced.{ext}",
        fmt=fmt,
        frame_range=frame_range,
        progress_every=int(args.animation_progress_every),
    )

    for mesh_name in vdb_names + unmerged_names:
        mesh = find_mesh(mesh_name, required=True)
        export_mesh(
            mesh,
            arm_orig,
            out_vdb / f"{mesh_name}.{ext}",
            fmt=fmt,
            frame_range=frame_range,
            progress_every=int(args.mesh_progress_every),
        )

    print("\n🎉 Prep 05 dual Fitbox export done.", flush=True)
    print(f"   original: {out_original}", flush=True)
    print(f"   vdb:      {out_vdb}", flush=True)
    print(f"   frame_range: {frame_range[0]}..{frame_range[1] - 1}", flush=True)


if __name__ == "__main__":
    main()
