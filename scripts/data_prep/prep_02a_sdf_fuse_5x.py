#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 02a: SDF Grid fusion in Blender 5.x.

Path behavior:
    required:
        --input_blend

    optional path inputs / outputs:
        --config
            If omitted:
                <work_dir>/prep_02_sdf_merge_config.json

        --out_dir
            If omitted:
                <work_dir>/<base_stem>_prep_01_sdf_exports/

Configuration override rule:
    1. Load config JSON.
    2. Validate group entries.
    3. Apply command-line config override arguments.
    4. Final resolved group config is used for SDF fusion.

Config override arguments have highest priority:
    --voxel_size
    --offset
    --auto_offset
    --no_auto_offset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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
from ogd_lbs.blender.object import ensure_object_mode, print_scene_inspect_summary  # noqa: E402
from ogd_lbs.blender.context import align_scene_pose_frame, update_view_layer  # noqa: E402
from ogd_lbs.blender.sdf_grid import (  # noqa: E402
    HAS_TRIMESH,
    create_sdf_grid_geonodes,
    update_sdf_nodes,
    duplicate_join_and_staticize_sources,
    evaluate_topology_from_evaluated_mesh,
    topology_is_success,
    format_topology_line,
    export_selected_obj,
)

from paths import (  # noqa: E402
    make_data_prep_context,
    prep_02a_config_rel,
    prep_02a_out_dir_rel,
)


def load_merge_rules(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict) or not obj:
        raise RuntimeError(f"Prep 02a config must be a non-empty JSON object: {config_path}")

    rules: dict[str, dict[str, Any]] = {}

    for target_name, cfg in obj.items():
        if not isinstance(target_name, str) or not target_name:
            raise RuntimeError(f"Invalid VDB target name in config: {target_name!r}")

        if not isinstance(cfg, dict):
            raise RuntimeError(f"Config for VDB target {target_name!r} must be an object.")

        rules[target_name] = dict(cfg)

    return rules


def resolve_group_config(
    *,
    target_name: str,
    raw_config: dict[str, Any],
    voxel_size_override: float | None,
    offset_override: float | None,
    auto_offset_override: bool | None,
) -> dict[str, Any]:
    source_meshes = raw_config.get("source_meshes", None)

    if not isinstance(source_meshes, list) or not source_meshes:
        raise RuntimeError(
            f"Config target {target_name!r} must contain non-empty list field 'source_meshes'."
        )

    source_meshes = [str(item) for item in source_meshes]

    if voxel_size_override is not None:
        voxel_size = float(voxel_size_override)
        voxel_size_source = "CLI_OVERRIDE"
    else:
        if "voxel_size" not in raw_config:
            raise RuntimeError(
                f"Config target {target_name!r} has no 'voxel_size', and --voxel_size was not provided."
            )
        voxel_size = float(raw_config["voxel_size"])
        voxel_size_source = "CONFIG"

    if voxel_size <= 0.0:
        raise RuntimeError(f"Resolved voxel_size must be > 0 for {target_name!r}, got {voxel_size}")

    if offset_override is not None:
        initial_offset = float(offset_override)
        offset_source = "CLI_OVERRIDE"
    else:
        initial_offset = float(raw_config.get("offset", 0.0))
        offset_source = "CONFIG" if "offset" in raw_config else "IMPLICIT_DEFAULT_0"

    if auto_offset_override is not None:
        auto_offset = bool(auto_offset_override)
        auto_offset_source = "CLI_OVERRIDE"
    elif "auto_offset" in raw_config:
        auto_offset = bool(raw_config["auto_offset"])
        auto_offset_source = "CONFIG"
    else:
        # Backward-compatible behavior:
        # if neither config offset nor CLI offset is provided, auto-search offset.
        auto_offset = ("offset" not in raw_config and offset_override is None)
        auto_offset_source = "IMPLICIT_FROM_OFFSET_ABSENCE"

    return {
        "source_meshes": source_meshes,
        "voxel_size": voxel_size,
        "offset": initial_offset,
        "auto_offset": auto_offset,
        "override_sources": {
            "voxel_size": voxel_size_source,
            "offset": offset_source,
            "auto_offset": auto_offset_source,
        },
    }


def validate_source_meshes_exist(
    *,
    target_name: str,
    source_names: list[str],
) -> list[bpy.types.Object]:
    objs: list[bpy.types.Object] = []

    missing: list[str] = []
    wrong_type: list[str] = []

    for name in source_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
            continue
        if obj.type != "MESH":
            wrong_type.append(f"{name}:{obj.type}")
            continue
        objs.append(obj)

    if missing or wrong_type:
        raise RuntimeError(
            f"Invalid source meshes for VDB target {target_name!r}. "
            f"missing={missing}, wrong_type={wrong_type}"
        )

    return objs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep 02a: Blender 5.x SDF Grid fusion.")

    required = parser.add_argument_group("required arguments")
    required.add_argument("--input_blend", type=str, required=True)

    path_args = parser.add_argument_group("path arguments")
    path_args.add_argument(
        "--config",
        type=str,
        default=None,
        help="SDF merge config. Default: <work_dir>/prep_02_sdf_merge_config.json",
    )
    path_args.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <work_dir>/<base_stem>_prep_01_sdf_exports/",
    )

    run_args = parser.add_argument_group("runtime arguments")
    run_args.add_argument(
        "--target_frame",
        type=int,
        default=-1,
        help="Animation frame for SDF fusion. -1 means REST pose.",
    )
    run_args.add_argument("--max_offset_multiplier", type=float, default=8.0)
    run_args.add_argument("--band_width_padding", type=int, default=8)
    run_args.add_argument("--min_component_faces", type=int, default=10)
    run_args.add_argument("--min_component_face_ratio", type=float, default=0.0)
    run_args.add_argument("--require_watertight", action="store_true")
    run_args.add_argument("--no_apply_scale", dest="apply_scale", action="store_false")
    parser.set_defaults(apply_scale=True)

    override_args = parser.add_argument_group("config override arguments")
    override_args.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Override every config group's voxel_size.",
    )
    override_args.add_argument(
        "--offset",
        type=float,
        default=None,
        help="Override every config group's initial/final offset.",
    )
    auto_group = override_args.add_mutually_exclusive_group()
    auto_group.add_argument(
        "--auto_offset",
        dest="auto_offset_override",
        action="store_true",
        default=None,
        help="Override config and enable auto offset search for every group.",
    )
    auto_group.add_argument(
        "--no_auto_offset",
        dest="auto_offset_override",
        action="store_false",
        help="Override config and disable auto offset search for every group.",
    )

    args = parser.parse_args(script_argv())

    ctx = make_data_prep_context(args.input_blend)

    input_path = ctx.resolve_input(
        "input_blend",
        args.input_blend,
        None,
        expected_type="file",
    )
    config_path = ctx.resolve_input(
        "config",
        args.config,
        prep_02a_config_rel(ctx),
        expected_type="file",
    )
    out_dir = ctx.resolve_output(
        "out_dir",
        args.out_dir,
        prep_02a_out_dir_rel(ctx),
        expected_type="dir",
    )

    ctx.print_and_validate("Prep 02a: SDF Grid Fusion")

    print(f"📂 Loading: {input_path}", flush=True)
    result = bpy.ops.wm.open_mainfile(filepath=str(input_path))
    if result != {"FINISHED"}:
        raise RuntimeError(f"Failed to open blend file: {input_path}, result={result}")

    ensure_object_mode()

    print_scene_inspect_summary(include_actions=False)

    print(f"\n🛑 Applying target_frame={int(args.target_frame)}...", flush=True)
    align_scene_pose_frame(int(args.target_frame), set_all_armatures=True)

    print(f"\n📋 Loading SDF config: {config_path}", flush=True)
    merge_rules = load_merge_rules(config_path)

    resolved_rules: dict[str, dict[str, Any]] = {}

    print("\n🧩 Resolving config groups with CLI override priority...", flush=True)
    for target_name, raw_cfg in merge_rules.items():
        resolved = resolve_group_config(
            target_name=target_name,
            raw_config=raw_cfg,
            voxel_size_override=args.voxel_size,
            offset_override=args.offset,
            auto_offset_override=args.auto_offset_override,
        )
        resolved_rules[target_name] = resolved

        print(f"   {target_name}:", flush=True)
        print(f"      source_meshes : {resolved['source_meshes']}", flush=True)
        print(
            f"      voxel_size    : {resolved['voxel_size']} "
            f"({resolved['override_sources']['voxel_size']})",
            flush=True,
        )
        print(
            f"      offset        : {resolved['offset']} "
            f"({resolved['override_sources']['offset']})",
            flush=True,
        )
        print(
            f"      auto_offset   : {resolved['auto_offset']} "
            f"({resolved['override_sources']['auto_offset']})",
            flush=True,
        )

    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()

    manifest: dict[str, Any] = {
        "source_blend": str(input_path.resolve()),
        "config": str(config_path.resolve()),
        "bpy_version": bpy.app.version_string,
        "target_frame": int(args.target_frame),
        "exports": {},
        "details": {},
        "vdb_mapping": {},
    }

    for target_name, cfg in resolved_rules.items():
        source_names = list(cfg["source_meshes"])
        source_objs = validate_source_meshes_exist(
            target_name=target_name,
            source_names=source_names,
        )

        voxel_size = float(cfg["voxel_size"])
        initial_offset = float(cfg["offset"])
        use_auto = bool(cfg["auto_offset"])

        manifest["vdb_mapping"][target_name] = [obj.name for obj in source_objs]

        print("\n" + "=" * 80, flush=True)
        print(f"🧪 Processing VDB target: {target_name}", flush=True)
        print(f"   source_meshes: {[obj.name for obj in source_objs]}", flush=True)
        print(f"   voxel_size:    {voxel_size}", flush=True)
        print(f"   initial_offset:{initial_offset}", flush=True)
        print(f"   auto_offset:   {use_auto}", flush=True)

        temp_obj = duplicate_join_and_staticize_sources(
            source_objs,
            temp_name=f"TEMP_{target_name}",
            apply_scale=bool(args.apply_scale),
        )

        modifier = create_sdf_grid_geonodes(
            temp_obj,
            voxel_size,
            initial_offset,
            band_width_padding=int(args.band_width_padding),
        )

        selected_offset = initial_offset
        selected_stats = None

        if use_auto:
            if not HAS_TRIMESH:
                raise RuntimeError(
                    "auto_offset was requested, but trimesh/numpy is not available in this environment."
                )

            multiplier = 0.5
            step = 0.5
            max_mult = float(args.max_offset_multiplier)

            while multiplier <= max_mult + 1e-9:
                current_offset = voxel_size * multiplier

                update_sdf_nodes(
                    modifier,
                    voxel_size=voxel_size,
                    offset_distance=current_offset,
                    band_width_padding=int(args.band_width_padding),
                )
                depsgraph.update()

                stats = evaluate_topology_from_evaluated_mesh(
                    temp_obj,
                    depsgraph,
                    min_component_faces=int(args.min_component_faces),
                    min_component_face_ratio=float(args.min_component_face_ratio),
                )

                ok = topology_is_success(
                    stats,
                    require_watertight=bool(args.require_watertight),
                )

                print(
                    f"     [Attempt] {multiplier:.1f}x "
                    f"offset={current_offset:.6g} | "
                    f"{format_topology_line(stats)} | {'✅' if ok else '❌'}",
                    flush=True,
                )

                selected_offset = current_offset
                selected_stats = stats

                if ok:
                    break

                multiplier += step

        update_sdf_nodes(
            modifier,
            voxel_size=voxel_size,
            offset_distance=selected_offset,
            band_width_padding=int(args.band_width_padding),
        )
        depsgraph.update()

        ensure_object_mode()
        bpy.ops.object.select_all(action="DESELECT")
        temp_obj.select_set(True)
        bpy.context.view_layer.objects.active = temp_obj

        print(f"⚙️ Applying SDF modifier for {target_name}...", flush=True)
        apply_result = bpy.ops.object.modifier_apply(modifier=modifier.name)
        if apply_result != {"FINISHED"}:
            raise RuntimeError(f"modifier_apply failed for {target_name}: {apply_result}")

        temp_obj.name = target_name
        temp_obj.data.name = target_name

        final_obj_path = out_dir / f"{target_name}.obj"

        print(f"📤 Exporting OBJ: {final_obj_path}", flush=True)
        export_selected_obj(temp_obj, final_obj_path)

        bpy.ops.object.delete()

        manifest["exports"][target_name] = str(final_obj_path.resolve())
        manifest["details"][target_name] = {
            "source_meshes": [obj.name for obj in source_objs],
            "voxel_size": voxel_size,
            "offset": selected_offset,
            "auto_offset": use_auto,
            "override_sources": cfg["override_sources"],
            "topology_stats": selected_stats,
        }

        update_view_layer()

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(f"\n✅ Prep 02a complete. Manifest saved to: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
