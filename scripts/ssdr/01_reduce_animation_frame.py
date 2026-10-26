#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSDR Step 01: Dynamic animation frame reduction.

Default pipeline semantics:
    --input_dir <work_dir>
        reads : <work_dir>/data0
        writes: <work_dir>/data1

    --input_dir <work_dir>/data0
        reads : <work_dir>/data0
        writes: <work_dir>/data1

Explicit automation mode:
    --input_dir <custom_input_tree> --output_dir <custom_output_tree>
        reads : <custom_input_tree>
        writes: <custom_output_tree>

Processing:
    - Recursively scan *.pkl.
    - Determine a single global frame sampling plan from the first time-series
      Fitbox object found under the input tree.
    - If frame_count <= max_frames, keep all frames.
    - If frame_count > max_frames, uniformly sample target_frames frames.
    - Apply the same sampling plan to:
        animation_data*.pkl:
            animation_poses
            scene_frames if present

        mesh *.pkl:
            baked_animation_vertices
            baked_animation_vertex_normals
            scene_frames if present

    - Copy all non-time-series pkl files unchanged.
    - Preserve relative directory structure.
    - rest_pose is never modified.

This step assumes Step 00 has already normalized JSON/PKL into data0.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from _env import ensure_repo_paths_in_syspath
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from ._env import ensure_repo_paths_in_syspath

ensure_repo_paths_in_syspath(__file__)

try:
    from _stage_io import prepare_output_dir, write_stage_summary
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from ._stage_io import prepare_output_dir, write_stage_summary



import numpy as np  # noqa: E402

from ogd_lbs.utils.work_dir import parse_idx_ranges  # noqa: E402

from ogd_lbs.utils.common import (  # noqa: E402
    log,
    warn,
    fail,
    script_argv,
    Timer,
    load_data,
    dump_pkl,
    record_metric,
    dump_metrics,
)

from ogd_lbs.ssdr.frame_sampling import (  # noqa: E402
    DynamicFrameSamplePlan,
    make_dynamic_frame_sample_plan,
    infer_animation_frame_count,
    apply_frame_sample_plan_to_animation,
    apply_frame_sample_plan_to_mesh,
)

from paths import (  # noqa: E402
    resolve_ssdr_stage_dirs,
    validate_ssdr_stage_resolution,
    print_ssdr_resolution_info,
)


def discover_pkl_files(input_tree: Path) -> list[Path]:
    return sorted(
        [p for p in input_tree.rglob("*.pkl") if p.is_file()],
        key=lambda p: str(p.relative_to(input_tree)),
    )


def classify_pkl_object(obj: Any) -> str:
    """
    Returns:
        animation
        mesh
        copy
    """
    if not isinstance(obj, dict):
        return "copy"

    if "animation_poses" in obj:
        return "animation"

    if "baked_animation_vertices" in obj or "baked_animation_vertex_normals" in obj:
        return "mesh"

    return "copy"


def infer_frame_count_from_mesh_obj(mesh_obj: dict[str, Any]) -> int:
    if "baked_animation_vertices" not in mesh_obj:
        raise KeyError("mesh object is missing baked_animation_vertices")

    arr = np.asarray(mesh_obj["baked_animation_vertices"])

    if arr.ndim < 1:
        raise ValueError(f"baked_animation_vertices must have at least 1 dimension, got {arr.shape}")

    return int(arr.shape[0])


def find_reference_frame_count(
    *,
    input_tree: Path,
    pkl_files: list[Path],
) -> tuple[int, Path, str]:
    """
    Find a reference time-series file and return:
        frame_count, path, kind

    Prefer animation files over mesh files for clarity.
    """
    first_mesh_candidate: tuple[int, Path, str] | None = None

    for pkl_path in pkl_files:
        obj = load_data(pkl_path)
        kind = classify_pkl_object(obj)

        if kind == "animation":
            frame_count = infer_animation_frame_count(obj)
            return frame_count, pkl_path, kind

        if kind == "mesh" and first_mesh_candidate is None:
            if not isinstance(obj, dict):
                continue
            frame_count = infer_frame_count_from_mesh_obj(obj)
            first_mesh_candidate = (frame_count, pkl_path, kind)

    if first_mesh_candidate is not None:
        return first_mesh_candidate

    fail(
        f"No time-series Fitbox PKL found under {input_tree}. "
        "Expected at least one animation_data*.pkl with animation_poses, "
        "or one mesh pkl with baked_animation_vertices."
    )

    raise RuntimeError("unreachable")



def make_frame_sample_plan_with_optional_removal_ranges(
    frame_count: int,
    *,
    max_frames: int,
    target_frames: int,
    removal_idx_ranges_text: str | None,
):
    """
    Build the Step 01 frame sampling plan.

    Default behavior is exactly the existing dynamic frame sampling plan.

    When removal_idx_ranges_text is non-empty, explicit candidate-frame index
    ranges are removed first, then the remaining frames are uniformly sampled
    down to target_frames only if the remaining count still exceeds max_frames.

    Range convention:
        - 0-based candidate-frame indices;
        - left-closed, right-open: start:end;
        - same convention as ogd_lbs.utils.work_dir.parse_idx_ranges().
    """
    text = str(removal_idx_ranges_text or "").strip()

    if not text:
        return make_dynamic_frame_sample_plan(
            int(frame_count),
            max_frames=int(max_frames),
            target_frames=int(target_frames),
        )

    frame_count_i = int(frame_count)
    max_frames_i = int(max_frames)
    target_frames_i = int(target_frames)

    if frame_count_i <= 0:
        raise RuntimeError(f"frame_count must be positive, got {frame_count_i}")

    if max_frames_i <= 0:
        raise RuntimeError(f"max_frames must be positive, got {max_frames_i}")

    if target_frames_i <= 0:
        raise RuntimeError(f"target_frames must be positive, got {target_frames_i}")

    ranges = parse_idx_ranges(text)
    removal_mask = np.zeros(frame_count_i, dtype=bool)

    for start, end in sorted(ranges):
        if int(end) > frame_count_i:
            raise RuntimeError(
                f"removal_idx_ranges item {start}:{end} exceeds frame_count={frame_count_i}"
            )
        removal_mask[int(start):int(end)] = True

    remaining_indices = np.arange(frame_count_i, dtype=np.int64)[~removal_mask]

    if remaining_indices.size <= 0:
        raise RuntimeError(
            f"removal_idx_ranges removed all frames: {text!r}, frame_count={frame_count_i}"
        )

    explicit_removed_count = int(np.count_nonzero(removal_mask))

    if int(remaining_indices.size) > max_frames_i:
        sample_count = min(target_frames_i, int(remaining_indices.size))

        if sample_count == 1:
            selected_sub = np.asarray([0], dtype=np.int64)
        else:
            raw = np.rint(
                np.linspace(0, int(remaining_indices.size) - 1, num=sample_count)
            ).astype(np.int64)

            selected_list: list[int] = []
            used: set[int] = set()

            for raw_idx in raw.tolist():
                idx = max(0, min(int(raw_idx), int(remaining_indices.size) - 1))
                if idx not in used:
                    selected_list.append(idx)
                    used.add(idx)

            if len(selected_list) < sample_count:
                for idx in range(int(remaining_indices.size)):
                    if idx in used:
                        continue
                    selected_list.append(idx)
                    used.add(idx)
                    if len(selected_list) >= sample_count:
                        break

            selected_sub = np.asarray(sorted(selected_list[:sample_count]), dtype=np.int64)

        selected_indices = remaining_indices[selected_sub]
        target_removed_count = int(remaining_indices.size) - int(selected_indices.size)
        sampling_mode = "removal_idx_ranges_plus_target_frames"
    else:
        selected_indices = remaining_indices
        target_removed_count = 0
        sampling_mode = "removal_idx_ranges_only"

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    selected_mask = np.zeros(frame_count_i, dtype=bool)
    selected_mask[selected_indices] = True

    final_removal_mask = ~selected_mask
    reduced = bool(
        selected_indices.shape[0] != frame_count_i
        or not np.array_equal(selected_indices, np.arange(frame_count_i, dtype=np.int64))
    )

    stats = {
        "sampling_mode": sampling_mode,
        "original_frame_count": int(frame_count_i),
        "max_frames": int(max_frames_i),
        "target_frames": int(target_frames_i),
        "target_frames_enabled": bool(int(remaining_indices.size) > max_frames_i),
        "reduced": bool(reduced),
        "retained_count": int(selected_indices.shape[0]),
        "selected_indices": [int(x) for x in selected_indices.tolist()],
        "removal_mask": [bool(x) for x in final_removal_mask.tolist()],
        "removal_idx_ranges": [list(x) for x in sorted(ranges)],
        "removal_idx_ranges_text": text,
        "explicit_removed_count": int(explicit_removed_count),
        "target_removed_count": int(target_removed_count),
        "removed_count": int(np.count_nonzero(final_removal_mask)),
        "remaining_after_explicit_count": int(remaining_indices.size),
        "note": (
            "removal_idx_ranges are 0-based candidate-frame indices, "
            "left-closed and right-open. Explicit ranges are removed before "
            "optional target-frame sampling."
        ),
    }

    return SimpleNamespace(
        original_frame_count=int(frame_count_i),
        selected_indices=selected_indices,
        removal_mask=final_removal_mask,
        reduced=bool(reduced),
        stats=stats,
    )

def process_single_pkl_file(
    pkl_path: Path,
    input_tree: Path,
    output_tree: Path,
    plan: DynamicFrameSamplePlan,
) -> dict[str, Any]:
    rel = pkl_path.relative_to(input_tree)
    out_path = output_tree / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obj = load_data(pkl_path)
    kind = classify_pkl_object(obj)

    if kind == "animation":
        if not isinstance(obj, dict):
            raise RuntimeError(f"Internal classification error for animation file: {pkl_path}")

        out_obj = apply_frame_sample_plan_to_animation(obj, plan, copy_obj=True)
        dump_pkl(out_path, out_obj)

        before_count = int(plan.original_frame_count)
        after_count = int(np.asarray(out_obj["animation_poses"]).shape[0])
        operation = "sample_animation"

    elif kind == "mesh":
        if not isinstance(obj, dict):
            raise RuntimeError(f"Internal classification error for mesh file: {pkl_path}")

        out_obj = apply_frame_sample_plan_to_mesh(obj, plan, copy_obj=True)
        dump_pkl(out_path, out_obj)

        before_count = int(plan.original_frame_count)
        after_count = int(np.asarray(out_obj["baked_animation_vertices"]).shape[0])
        operation = "sample_mesh"

    else:
        shutil.copy2(pkl_path, out_path)
        before_count = -1
        after_count = -1
        operation = "copy_pkl"

    stat_in = pkl_path.stat()
    stat_out = out_path.stat()

    return {
        "operation": operation,
        "input": str(pkl_path),
        "relative_path": str(rel),
        "output": str(out_path),
        "input_bytes": int(stat_in.st_size),
        "output_bytes": int(stat_out.st_size),
        "frame_count_before": before_count,
        "frame_count_after": after_count,
    }


def reduce_animation_tree(
    *,
    input_tree: Path,
    output_tree: Path,
    max_frames: int,
    target_frames: int,
    removal_idx_ranges: str | None = None,
    max_workers: int | None = None,
    overwrite_output: bool,
    allow_merge: bool,
    allow_no_files: bool,
    write_summary_file: bool,
) -> dict[str, Any]:
    with Timer("time_01_file_scanning"):
        pkl_files = discover_pkl_files(input_tree)

    if not pkl_files:
        msg = f"No .pkl files found under input tree: {input_tree}"

        if allow_no_files:
            warn(msg)
            prepare_output_dir(
                output_tree,
                overwrite_output=overwrite_output,
                allow_merge=allow_merge,
            )
            summary = {
                "schema": "ogd_lbs_ssdr_step01_dynamic_frame_sampling_v1",
                "input_tree": str(input_tree),
                "output_tree": str(output_tree),
                "pkl_file_count": 0,
                "processed_count": 0,
                "allow_no_files": True,
                "sampling": None,
                "files": [],
            }
            if write_summary_file:
                write_stage_summary(output_tree / "step01_frame_sampling_summary.json", summary)
            return summary

        fail(msg)

    with Timer("time_01_make_sampling_plan"):
        frame_count, ref_path, ref_kind = find_reference_frame_count(
            input_tree=input_tree,
            pkl_files=pkl_files,
        )
        plan = make_frame_sample_plan_with_optional_removal_ranges(
            frame_count,
            max_frames=int(max_frames),
            target_frames=int(target_frames),
            removal_idx_ranges_text=removal_idx_ranges,
        )

    prepare_output_dir(
        output_tree,
        overwrite_output=overwrite_output,
        allow_merge=allow_merge,
    )

    log(f"Found {len(pkl_files)} PKL files under: {input_tree}")
    log(f"Reference time-series file: {ref_path.relative_to(input_tree)} ({ref_kind})")
    log(f"Original frame count: {plan.original_frame_count}")
    log(f"Reduced: {plan.reduced}")
    log(f"Retained frame count: {len(plan.selected_indices)}")
    log(f"Output tree: {output_tree}")

    processed: list[dict[str, Any]] = []

    with Timer("time_01_process_pkl_files"):
        if max_workers == 1:
            for pkl_path in pkl_files:
                result = process_single_pkl_file(pkl_path, input_tree, output_tree, plan)
                processed.append(result)
                log(f"Processed ({result['operation']}): {result['relative_path']}")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_single_pkl_file, pkl_path, input_tree, output_tree, plan)
                    for pkl_path in pkl_files
                ]

                for future in as_completed(futures):
                    result = future.result()
                    processed.append(result)
                    log(f"Processed ({result['operation']}): {result['relative_path']}")

    processed = sorted(processed, key=lambda item: item["relative_path"])

    sampled_animation_count = sum(1 for item in processed if item["operation"] == "sample_animation")
    sampled_mesh_count = sum(1 for item in processed if item["operation"] == "sample_mesh")
    copied_pkl_count = sum(1 for item in processed if item["operation"] == "copy_pkl")

    input_bytes = int(sum(item["input_bytes"] for item in processed))
    output_bytes = int(sum(item["output_bytes"] for item in processed))

    summary = {
        "schema": "ogd_lbs_ssdr_step01_dynamic_frame_sampling_v1",
        "input_tree": str(input_tree),
        "output_tree": str(output_tree),
        "reference_time_series_file": str(ref_path),
        "reference_time_series_relative_path": str(ref_path.relative_to(input_tree)),
        "reference_time_series_kind": ref_kind,
        "pkl_file_count": int(len(pkl_files)),
        "processed_count": int(len(processed)),
        "sampled_animation_count": int(sampled_animation_count),
        "sampled_mesh_count": int(sampled_mesh_count),
        "copied_pkl_count": int(copied_pkl_count),
        "input_bytes_total": input_bytes,
        "output_bytes_total": output_bytes,
        "sampling": plan.stats,
        "files": processed,
    }

    if write_summary_file:
        summary_path = output_tree / "step01_frame_sampling_summary.json"
        write_stage_summary(summary_path, summary)
        log(f"Step 01 summary written: {summary_path}")

    record_metric("01_pkl_file_count", int(len(pkl_files)))
    record_metric("01_processed_count", int(len(processed)))
    record_metric("01_sampled_animation_count", int(sampled_animation_count))
    record_metric("01_sampled_mesh_count", int(sampled_mesh_count))
    record_metric("01_copied_pkl_count", int(copied_pkl_count))
    record_metric("01_original_frame_count", int(plan.original_frame_count))
    record_metric("01_final_retained_count", int(len(plan.selected_indices)))
    record_metric("01_reduced", bool(plan.reduced))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSDR Step 01: dynamically reduce animation frames from data0 to data1."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Primary input. Normally work_dir or work_dir/data0. "
            "With explicit --output_dir, this may also be a custom input tree."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Explicit output tree. If omitted, Step 01 writes to <work_dir>/data1.",
    )
    parser.add_argument(
        "-m",
        "--metrics_dir",
        type=str,
        default=None,
        help="Optional directory to save metrics_01.json.",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=210,
        help="If frame count <= max_frames, keep all frames. Default: 210.",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=200,
        help="If frame count > max_frames, uniformly sample to this count. Default: 200.",
    )

    parser.add_argument(
        "--removal_idx_ranges",
        type=str,
        default="",
        help=(
            "Optional explicit candidate-frame index ranges to remove before "
            "dynamic target-frame sampling, e.g. '20:60,260:340'. "
            "Ranges are 0-based and left-closed/right-open."
        ),
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum parallel worker processes. Use --max_workers 1 for sequential debugging.",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Remove existing output directory before Step 01.",
    )
    parser.add_argument(
        "--allow_merge",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--allow_no_files",
        action="store_true",
        help="Do not fail if no .pkl files are found.",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Do not write step01_frame_sampling_summary.json.",
    )
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        help="Enable Timer profiling logs via common.Timer.",
    )

    args = parser.parse_args(script_argv())

    resolution = resolve_ssdr_stage_dirs(
        step_id="01",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="SSDR Step 01: data0 -> data1 Frame Sampling",
    )

    print_ssdr_resolution_info(resolution)

    try:
        validate_ssdr_stage_resolution(resolution)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    log("=" * 80)
    log("SSDR Step 01: data0 -> data1 Frame Sampling")
    log(f"work_dir          : {resolution.work_dir}")
    log(f"input_tree        : {resolution.input_tree}")
    log(f"output_tree       : {resolution.output_tree}")
    log(f"input_source      : {resolution.input_source}")
    log(f"output_source     : {resolution.output_source}")
    log(f"max_frames        : {int(args.max_frames)}")
    log(f"target_frames     : {int(args.target_frames)}")
    log(f"removal_idx_ranges: {args.removal_idx_ranges}")
    log(f"max_workers       : {args.max_workers}")
    log(f"overwrite_output  : {bool(args.overwrite_output)}")
    log(f"allow_merge       : {bool(args.allow_merge)}")
    log("=" * 80)

    with Timer("time_01_total"):
        summary = reduce_animation_tree(
            input_tree=resolution.input_tree,
            output_tree=resolution.output_tree,
            max_frames=int(args.max_frames),
            target_frames=int(args.target_frames),
            removal_idx_ranges=args.removal_idx_ranges,
            max_workers=args.max_workers,
            overwrite_output=bool(args.overwrite_output),
            allow_merge=bool(args.allow_merge),
            allow_no_files=bool(args.allow_no_files),
            write_summary_file=not bool(args.no_summary),
        )

    if args.metrics_dir:
        dump_metrics(Path(args.metrics_dir) / "metrics_01.json")

    sampling = summary.get("sampling") or {}

    log("SSDR Step 01 complete.")
    log(f"processed_count          : {summary['processed_count']}")
    log(f"sampled_animation_count  : {summary['sampled_animation_count']}")
    log(f"sampled_mesh_count       : {summary['sampled_mesh_count']}")
    log(f"copied_pkl_count         : {summary['copied_pkl_count']}")
    log(f"reduced                  : {sampling.get('reduced')}")
    log(f"final_retained_count     : {sampling.get('retained_count')}")
    log(f"output_tree              : {resolution.output_tree}")


if __name__ == "__main__":
    main()
