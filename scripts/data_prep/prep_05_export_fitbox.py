#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 05: Manifest-driven Fitbox export with SSDR-style frame sampling.

Output layout:

    <out_dir>/
    ├── export_summary.json
    ├── fitbox_original/
    │   ├── skeleton_hierarchy_data.pkl
    │   ├── animation_data.pkl
    │   └── Source_<VDB>.pkl
    ├── fitbox_vdb/
    │   ├── skeleton_hierarchy_data.pkl
    │   ├── animation_data.pkl
    │   ├── <VDB>.pkl
    │   └── <unmerged>.pkl
    └── bone_reduced_versions/
        ├── cleaned_full/
        │   ├── skeleton_hierarchy_data.pkl
        │   └── animation_data.pkl
        ├── bone_reduced_001/
        │   ├── skeleton_hierarchy_data.pkl
        │   └── animation_data.pkl
        └── ...

Important:
    - fitbox_original and fitbox_vdb use Armature_Original.
    - bone_reduced_versions/cleaned_full uses Armature_Cleaned_Full.
    - manual variants are discovered from <work_dir>/bone_reduced_versions/*.blend.
    - generated templates with unchanged size/mtime_ns are skipped by default.
    - rest_pose is stored separately; animation_poses[0] is NOT rest_pose.
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
from ogd_lbs.fitbox.blender_io import (  # noqa: E402
    export_skeleton,
    export_animation,
    export_mesh,
    get_nla_frame_range,
)
from ogd_lbs.fitbox.sampling import resolve_sampled_frames  # noqa: E402
from ogd_lbs.utils.work_dir import parse_idx_ranges  # noqa: E402

from paths import (  # noqa: E402
    make_data_prep_context,
    prep_02_manifest_rel,
    prep_05_out_dir_rel,
    prep_04b_manual_versions_dir_rel,
)
from manifest import (  # noqa: E402
    load_manifest,
    require_vdb_mapping,
    get_source_objects_mapping,
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


def resolve_candidate_frame_range(
    *,
    armature: bpy.types.Object,
    frame_start_arg: int | None,
    frame_end_arg: int | None,
) -> tuple[int, int]:
    full_start, full_end_exclusive = get_nla_frame_range(armature)

    start = int(frame_start_arg) if frame_start_arg is not None else int(full_start)

    if frame_end_arg is not None:
        end_exclusive = int(frame_end_arg) + 1
    else:
        end_exclusive = int(full_end_exclusive)

    if end_exclusive <= start:
        raise RuntimeError(
            f"Invalid export frame range: start={start}, end_exclusive={end_exclusive}. "
            f"Full NLA range was {full_start}..{full_end_exclusive - 1}."
        )

    print(
        f"🎞️ Candidate frame range: {start}..{end_exclusive - 1} "
        f"(count={end_exclusive - start}); full NLA range={full_start}..{full_end_exclusive - 1}",
        flush=True,
    )

    return start, end_exclusive


def load_manual_versions_manifest(manual_dir: Path) -> dict[str, Any]:
    manifest_path = manual_dir / "manifest.json"

    if not manifest_path.exists():
        return {}

    with manifest_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(f"Manual versions manifest must be an object: {manifest_path}")

    return obj


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime": float(stat.st_mtime),
    }


def template_is_unmodified(
    *,
    blend_path: Path,
    template_entry: dict[str, Any],
) -> bool:
    initial = template_entry.get("initial_file_metadata", {})
    if not isinstance(initial, dict):
        return False

    current = file_metadata(blend_path)

    return (
        int(initial.get("size", -1)) == int(current["size"])
        and int(initial.get("mtime_ns", -1)) == int(current["mtime_ns"])
    )


def discover_manual_variant_blends(
    *,
    manual_dir: Path,
    include_unmodified_templates: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns:
        exported_variants, skipped_variants
    """
    if not manual_dir.exists():
        return [], []

    manual_manifest = load_manual_versions_manifest(manual_dir)
    templates = manual_manifest.get("templates", {})
    if not isinstance(templates, dict):
        templates = {}

    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for blend_path in sorted(manual_dir.glob("*.blend"), key=lambda p: p.name):
        entry = templates.get(blend_path.name)
        is_template = isinstance(entry, dict)

        if is_template:
            skip_if_unmodified = bool(entry.get("skip_if_unmodified", False))
            unchanged = template_is_unmodified(blend_path=blend_path, template_entry=entry)

            if skip_if_unmodified and unchanged and not include_unmodified_templates:
                skipped.append(
                    {
                        "variant_name": blend_path.stem,
                        "blend_path": str(blend_path),
                        "reason": "generated_template_unmodified",
                        "armature_name": entry.get("armature_name"),
                        "current_file_metadata": file_metadata(blend_path),
                    }
                )
                continue

            armature_name = entry.get("armature_name")
        else:
            armature_name = None

        exported.append(
            {
                "variant_name": blend_path.stem,
                "blend_path": str(blend_path),
                "armature_name": armature_name,
                "is_recorded_template": bool(is_template),
                "current_file_metadata": file_metadata(blend_path),
            }
        )

    return exported, skipped


def find_manual_variant_armature(
    *,
    variant_name: str,
    preferred_name: str | None,
) -> bpy.types.Object:
    if preferred_name:
        obj = bpy.data.objects.get(str(preferred_name))
        if obj is not None and obj.type == "ARMATURE":
            return obj
        raise RuntimeError(
            f"Manual variant {variant_name!r} preferred armature not found: {preferred_name}"
        )

    expected_suffix = variant_name.replace("bone_reduced_", "")
    candidates_by_prefix = [
        obj for obj in bpy.data.objects
        if obj.type == "ARMATURE" and obj.name.startswith("Armature_Bone_Reduced")
    ]

    if len(candidates_by_prefix) == 1:
        return candidates_by_prefix[0]

    non_original = [
        obj for obj in bpy.data.objects
        if obj.type == "ARMATURE"
        and obj.name not in {"Armature_Original", "Armature_Cleaned_Full"}
    ]

    if len(non_original) == 1:
        return non_original[0]

    raise RuntimeError(
        f"Could not uniquely resolve manual reduced armature for variant={variant_name!r}. "
        f"Candidates={[(obj.name, len(obj.data.bones)) for obj in non_original]}"
    )


def export_armature_only_fitbox(
    *,
    armature: bpy.types.Object,
    out_dir: Path,
    fmt: str,
    frames: list[int],
    animation_progress_every: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    export_skeleton(
        armature,
        out_dir / f"skeleton_hierarchy_data.{fmt}",
        fmt=fmt,
    )
    export_animation(
        armature,
        out_dir / f"animation_data.{fmt}",
        fmt=fmt,
        frames=frames,
        progress_every=animation_progress_every,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep 05: Fitbox export with frame sampling and bone variants.")

    parser.add_argument("--input_blend", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--manual_versions_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--format", choices=["pkl", "json"], default="pkl")

    parser.add_argument("--debug_frame_max", type=int, default=-1)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)

    parser.add_argument("--removal_rate", type=float, default=0.9)
    parser.add_argument("--removal_bias", type=float, default=1.0)
    parser.add_argument(
        "--removal_idx_ranges",
        type=str,
        default="",
        help='Comma-separated 0-based candidate-frame index ranges, e.g. "20:60,260:340".',
    )

    parser.add_argument("--animation_progress_every", type=int, default=100)
    parser.add_argument("--mesh_progress_every", type=int, default=25)

    parser.add_argument(
        "--include_unmodified_templates",
        action="store_true",
        help="Export generated manual templates even if manifest says they are unmodified.",
    )

    args = parser.parse_args(script_argv())

    ctx = make_data_prep_context(args.input_blend)

    input_path = ctx.resolve_input("input_blend", args.input_blend, None, expected_type="file")
    manifest_path = ctx.resolve_input("manifest", args.manifest, prep_02_manifest_rel(ctx), expected_type="file")
    out_root = ctx.resolve_output(
        "out_dir",
        args.out_dir,
        prep_05_out_dir_rel(ctx, debug_frame_max=int(args.debug_frame_max)),
        expected_type="dir",
    )

    manual_dir = ctx.add_info_path(
        "manual_versions_dir",
        Path(args.manual_versions_dir) if args.manual_versions_dir else ctx.work_dir / prep_04b_manual_versions_dir_rel(ctx),
        expected_type="dir",
        note="optional input directory; missing directory means no manual variants",
    )

    ctx.print_and_validate("Prep 05: Fitbox Export")

    fmt = args.format
    ext = fmt

    print(f"📂 Loading main Prep 04a blend: {input_path}", flush=True)
    result = bpy.ops.wm.open_mainfile(filepath=str(input_path))
    if result != {"FINISHED"}:
        raise RuntimeError(f"Failed to open blend file: {input_path}, result={result}")

    manifest_obj = load_manifest(manifest_path)
    vdb_mapping = require_vdb_mapping(manifest_obj)
    source_objects = get_source_objects_mapping(manifest_obj, vdb_mapping)

    all_mesh_names = [obj.name for obj in bpy.data.objects if obj.type == "MESH"]

    vdb_names = unique_keep_order([name for name in vdb_mapping.keys() if name in all_mesh_names])

    source_names = []
    for vdb_name in vdb_mapping.keys():
        source_names.append(source_objects.get(vdb_name, f"Source_{vdb_name}"))
    source_names = unique_keep_order(source_names)

    manifest_unmerged = manifest_obj.get("unmerged_meshes", [])
    if manifest_unmerged is None:
        manifest_unmerged = []
    if not isinstance(manifest_unmerged, list):
        raise RuntimeError("manifest field 'unmerged_meshes' must be a list if present.")
    unmerged_names = unique_keep_order([str(name) for name in manifest_unmerged if str(name) in all_mesh_names])

    missing_sources = [name for name in source_names if name not in all_mesh_names]
    missing_vdb = [name for name in vdb_mapping.keys() if name not in all_mesh_names]

    if missing_sources:
        raise RuntimeError(f"Source meshes referenced by manifest are missing in main blend: {missing_sources}")
    if missing_vdb:
        raise RuntimeError(f"VDB meshes referenced by manifest are missing in main blend: {missing_vdb}")

    print("\n📋 Export classification:", flush=True)
    print(f"   Source meshes:   {source_names}", flush=True)
    print(f"   VDB meshes:      {vdb_names}", flush=True)
    print(f"   Unmerged meshes: {unmerged_names}", flush=True)

    arm_orig = find_armature("Armature_Original", required=True)
    arm_cleaned_full = find_armature("Armature_Cleaned_Full", required=True)

    assert arm_orig is not None
    assert arm_cleaned_full is not None

    candidate_start, candidate_end = resolve_candidate_frame_range(
        armature=arm_orig,
        frame_start_arg=args.frame_start,
        frame_end_arg=args.frame_end,
    )

    removal_idx_ranges = parse_idx_ranges(args.removal_idx_ranges)

    sampling = resolve_sampled_frames(
        frame_start=candidate_start,
        frame_end_exclusive=candidate_end,
        removal_rate=float(args.removal_rate),
        removal_bias=float(args.removal_bias),
        removal_idx_ranges=removal_idx_ranges,
        debug_frame_max=int(args.debug_frame_max),
    )

    frames = sampling.retained_frames

    print("\n🎞️ Final sampled frames:", flush=True)
    print(f"   count: {len(frames)}", flush=True)
    print(f"   first: {frames[0]}", flush=True)
    print(f"   last : {frames[-1]}", flush=True)
    print("   rest_pose note: rest_pose is separate; animation_poses[0] is not rest_pose.", flush=True)

    out_original = out_root / "fitbox_original"
    out_vdb = out_root / "fitbox_vdb"
    out_reduced_root = out_root / "bone_reduced_versions"

    out_original.mkdir(parents=True, exist_ok=True)
    out_vdb.mkdir(parents=True, exist_ok=True)
    out_reduced_root.mkdir(parents=True, exist_ok=True)

    print("\n🚀 Exporting fitbox_original...", flush=True)
    export_skeleton(arm_orig, out_original / f"skeleton_hierarchy_data.{ext}", fmt=fmt)
    export_animation(
        arm_orig,
        out_original / f"animation_data.{ext}",
        fmt=fmt,
        frames=frames,
        progress_every=int(args.animation_progress_every),
    )

    for mesh_name in source_names + unmerged_names:
        mesh = find_mesh(mesh_name, required=True)
        assert mesh is not None
        export_mesh(
            mesh,
            arm_orig,
            out_original / f"{mesh_name}.{ext}",
            fmt=fmt,
            frames=frames,
            progress_every=int(args.mesh_progress_every),
        )

    print("\n🚀 Exporting fitbox_vdb...", flush=True)
    export_skeleton(arm_orig, out_vdb / f"skeleton_hierarchy_data.{ext}", fmt=fmt)
    export_animation(
        arm_orig,
        out_vdb / f"animation_data.{ext}",
        fmt=fmt,
        frames=frames,
        progress_every=int(args.animation_progress_every),
    )

    for mesh_name in vdb_names + unmerged_names:
        mesh = find_mesh(mesh_name, required=True)
        assert mesh is not None
        export_mesh(
            mesh,
            arm_orig,
            out_vdb / f"{mesh_name}.{ext}",
            fmt=fmt,
            frames=frames,
            progress_every=int(args.mesh_progress_every),
        )

    print("\n🚀 Exporting bone_reduced_versions/cleaned_full...", flush=True)
    cleaned_full_dir = out_reduced_root / "cleaned_full"
    export_armature_only_fitbox(
        armature=arm_cleaned_full,
        out_dir=cleaned_full_dir,
        fmt=fmt,
        frames=frames,
        animation_progress_every=int(args.animation_progress_every),
    )

    exported_manual_variants, skipped_manual_variants = discover_manual_variant_blends(
        manual_dir=manual_dir,
        include_unmodified_templates=bool(args.include_unmodified_templates),
    )

    manual_export_summaries: list[dict[str, Any]] = []

    for variant in exported_manual_variants:
        variant_name = str(variant["variant_name"])
        variant_blend = Path(str(variant["blend_path"]))
        preferred_armature_name = variant.get("armature_name")

        print("\n" + "=" * 80, flush=True)
        print(f"🚀 Exporting manual bone-reduced variant: {variant_name}", flush=True)
        print(f"   blend: {variant_blend}", flush=True)

        result = bpy.ops.wm.open_mainfile(filepath=str(variant_blend))
        if result != {"FINISHED"}:
            raise RuntimeError(f"Failed to open manual variant blend: {variant_blend}, result={result}")

        reduced_armature = find_manual_variant_armature(
            variant_name=variant_name,
            preferred_name=str(preferred_armature_name) if preferred_armature_name else None,
        )

        variant_out_dir = out_reduced_root / variant_name
        export_armature_only_fitbox(
            armature=reduced_armature,
            out_dir=variant_out_dir,
            fmt=fmt,
            frames=frames,
            animation_progress_every=int(args.animation_progress_every),
        )

        manual_export_summaries.append(
            {
                "variant_name": variant_name,
                "blend_path": str(variant_blend),
                "output_dir": str(variant_out_dir),
                "armature_name": reduced_armature.name,
                "bone_count": len(reduced_armature.data.bones),
                "is_recorded_template": bool(variant.get("is_recorded_template", False)),
            }
        )

    summary = {
        "schema": "ogd_lbs_prep05_fitbox_export_v2",
        "input_blend": str(input_path),
        "manifest": str(manifest_path),
        "out_dir": str(out_root),
        "format": fmt,
        "frame_sampling": sampling.stats,
        "mesh_classification": {
            "source_meshes": source_names,
            "vdb_meshes": vdb_names,
            "unmerged_meshes": unmerged_names,
        },
        "armatures": {
            "original": {
                "name": "Armature_Original",
                "bones": len(arm_orig.data.bones),
            },
            "cleaned_full": {
                "name": "Armature_Cleaned_Full",
                "bones": len(arm_cleaned_full.data.bones),
                "output_dir": str(cleaned_full_dir),
            },
            "manual_variants_exported": manual_export_summaries,
            "manual_variants_skipped": skipped_manual_variants,
        },
        "outputs": {
            "fitbox_original": str(out_original),
            "fitbox_vdb": str(out_vdb),
            "bone_reduced_versions": str(out_reduced_root),
        },
        "rest_pose_semantics": (
            "Fitbox animation_data stores rest_pose separately. "
            "animation_poses[0] is the first sampled Blender scene frame, "
            "not an implicit rest pose."
        ),
    }

    summary_path = out_root / "export_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n🎉 Prep 05 Fitbox export done.", flush=True)
    print(f"   original:       {out_original}", flush=True)
    print(f"   vdb:            {out_vdb}", flush=True)
    print(f"   reduced root:   {out_reduced_root}", flush=True)
    print(f"   summary:        {summary_path}", flush=True)
    print(f"   sampled frames: {len(frames)}", flush=True)
    print(f"   manual exported:{len(manual_export_summaries)}", flush=True)
    print(f"   manual skipped: {len(skipped_manual_variants)}", flush=True)


if __name__ == "__main__":
    main()
