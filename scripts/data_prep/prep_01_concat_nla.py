#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 01: Concatenate all Actions into one NLA track.

Staged data-prep behavior:

    --input_dir <data_prep_work_dir>
        reads : <data_prep_work_dir>/data_prep_00/<raw_asset>.blend
        writes: <data_prep_work_dir>/data_prep_01/<asset>_prep_01.blend

    --input_dir <data_prep_work_dir>/data_prep_00
        reads : <data_prep_work_dir>/data_prep_00/<raw_asset>.blend
        writes: <data_prep_work_dir>/data_prep_01/<asset>_prep_01.blend

    --input_dir <custom_input_tree> --output_dir <custom_output_tree>
        reads : <custom_input_tree>/<raw_asset>.blend
        writes: <custom_output_tree>/<asset>_prep_01.blend

Raw .blend discovery:
    - Search direct children of input_tree.
    - Exclude known staged files such as *_prep_01.blend.
    - Exactly one raw .blend must exist; otherwise fail-fast.
"""

from __future__ import annotations

import argparse
from pathlib import Path


import _env  # noqa: F401

import bpy  # noqa: E402

from ogd_lbs.utils.common import script_argv  # noqa: E402
from ogd_lbs.blender.object import (  # noqa: E402
    ensure_object_mode,
    find_armature,
    print_scene_inspect_summary,
)
from ogd_lbs.blender.animation import concat_all_actions_to_nla  # noqa: E402
from ogd_lbs.blender.blend_io import open_blend_file, save_blend_file  # noqa: E402

from paths import (  # noqa: E402
    resolve_data_prep_stage_dirs,
    validate_data_prep_stage_resolution,
    print_data_prep_resolution_info,
    discover_raw_input_blend,
    extract_asset_base_stem,
    prep_01_output_blend_name,
)


def parse_exclude_names(text: str) -> list[str]:
    if not text:
        return []

    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prep 01: concatenate all actions into a single NLA track."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Data-prep work_dir or data_prep_00 input tree. "
            "The script discovers exactly one raw .blend inside data_prep_00."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional explicit output tree. If omitted, writes to "
            "<work_dir>/data_prep_01."
        ),
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=10,
        help="Empty frames between actions. Default: 10.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated action names to exclude.",
    )

    args = parser.parse_args(script_argv())

    resolution = resolve_data_prep_stage_dirs(
        step_id="01",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="Prep 01: NLA Concatenation",
    )

    print_data_prep_resolution_info(resolution)

    try:
        validate_data_prep_stage_resolution(resolution)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    input_path = discover_raw_input_blend(resolution.input_tree)
    base_stem = extract_asset_base_stem(input_path)
    output_path = resolution.output_tree / prep_01_output_blend_name(base_stem)

    if output_path.resolve(strict=False) == input_path.resolve(strict=False):
        raise RuntimeError(
            f"Output blend would overwrite input blend: {output_path}"
        )

    resolution.output_tree.mkdir(parents=True, exist_ok=True)

    print("\n📋 Prep 01 resolved files:", flush=True)
    print(f"   input_blend : {input_path}", flush=True)
    print(f"   base_stem   : {base_stem}", flush=True)
    print(f"   output_blend: {output_path}", flush=True)

    open_blend_file(input_path, role="Prep 01 input blend")

    ensure_object_mode()

    print_scene_inspect_summary(include_actions=True)

    armature = find_armature(required=True, prefer_name="Armature")
    if armature is None:
        raise RuntimeError("No ARMATURE found in the scene.")

    print(f"🎯 Target Armature selected: {armature.name}", flush=True)

    exclude_names = parse_exclude_names(args.exclude)

    concat_result = concat_all_actions_to_nla(
        armature,
        gap=int(args.gap),
        exclude_names=exclude_names,
        track_name="OGD_Concat_Track",
        start_frame=1,
        fail_if_empty=True,
        verbose=True,
    )

    print("\n📋 Prep 01 result summary:", flush=True)
    print(f"   track_name        : {concat_result.track_name}", flush=True)
    print(f"   added_count       : {concat_result.added_count}", flush=True)
    print(
        f"   scene_frame_range : "
        f"{concat_result.scene_frame_start}..{concat_result.scene_frame_end}",
        flush=True,
    )

    save_blend_file(output_path, role="Prep 01 output blend")

    print("🎉 Prep 01 done.", flush=True)


if __name__ == "__main__":
    main()
