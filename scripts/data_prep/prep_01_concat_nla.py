#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 01: Concatenate all Actions into one NLA track.

Default path behavior:
    required:
        --input_blend

    optional:
        --output_blend

If --output_blend is omitted, the output is placed next to --input_blend:

    <work_dir>/<base_stem>_prep_01.blend

Examples:

    python scripts/data_prep/prep_01_concat_nla.py \\
        --input_blend /workspace/data/dataset/Griffin_A2/Blender/Griffin_A2.blend

    python scripts/data_prep/prep_01_concat_nla.py \\
        --input_blend /workspace/data/dataset/Griffin_A2/Blender/Griffin_A2.blend \\
        --output_blend /tmp/Griffin_A2_prep_01_test.blend \\
        --gap 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def ensure_repo_src_in_syspath() -> None:
    """
    Make src/ importable before importing ogd_lbs.

    This keeps the script robust in both:
        - editable install environments
        - standalone Blender/bpy environments
    """
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
from ogd_lbs.blender.object import (  # noqa: E402
    ensure_object_mode,
    find_armature,
    print_scene_inspect_summary,
)
from ogd_lbs.blender.animation import concat_all_actions_to_nla  # noqa: E402

from paths import make_data_prep_context, prep_01_output_blend_rel  # noqa: E402


def parse_exclude_names(text: str) -> list[str]:
    if not text:
        return []

    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prep 01: concatenate all actions into a single NLA track."
    )

    parser.add_argument(
        "--input_blend",
        type=str,
        required=True,
        help="Path to the raw purchased .blend file.",
    )
    parser.add_argument(
        "--output_blend",
        type=str,
        default=None,
        help="Optional output .blend path. Default: <work_dir>/<base_stem>_prep_01.blend.",
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

    ctx = make_data_prep_context(args.input_blend)

    input_path = ctx.resolve_input(
        "input_blend",
        args.input_blend,
        None,
        expected_type="file",
    )
    output_path = ctx.resolve_output(
        "output_blend",
        args.output_blend,
        prep_01_output_blend_rel(ctx),
        expected_type="file",
    )

    ctx.print_and_validate("Prep 01: NLA Concatenation")

    print(f"📂 Loading: {input_path}", flush=True)
    result = bpy.ops.wm.open_mainfile(filepath=str(input_path))
    if result != {"FINISHED"}:
        raise RuntimeError(f"Failed to open blend file: {input_path}, result={result}")

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

    print(f"\n💾 Saving to: {output_path}", flush=True)
    save_result = bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    if save_result != {"FINISHED"}:
        raise RuntimeError(f"Failed to save blend file: {output_path}, result={save_result}")

    print("🎉 Prep 01 done.", flush=True)


if __name__ == "__main__":
    main()
