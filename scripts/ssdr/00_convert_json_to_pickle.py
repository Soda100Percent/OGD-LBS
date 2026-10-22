#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSDR Step 00: Normalize source Fitbox tree into data0.

Default pipeline semantics:
    --input_dir <work_dir>
        reads : <work_dir>/data0_json
        writes: <work_dir>/data0

    --input_dir <work_dir>/data0_json
        reads : <work_dir>/data0_json
        writes: <work_dir>/data0

Explicit automation mode:
    --input_dir <custom_input_tree> --output_dir <custom_output_tree>
        reads : <custom_input_tree>
        writes: <custom_output_tree>

Processing:
    - Recursively convert *.json -> *.pkl.
    - Recursively copy *.pkl -> *.pkl.
    - Preserve relative directory structure.
    - Ignore other file types.

This step does not inspect or interpret Fitbox internal structure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def ensure_repo_paths_in_syspath() -> None:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src_dir = parent / "src"
            ssdr_script_dir = parent / "scripts" / "ssdr"

            for item in [src_dir, ssdr_script_dir]:
                item_s = str(item)
                if item_s not in sys.path:
                    sys.path.insert(0, item_s)
            return

    raise RuntimeError("Could not locate repository root containing pyproject.toml and src/.")


ensure_repo_paths_in_syspath()

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

from paths import (  # noqa: E402
    resolve_ssdr_stage_dirs,
    validate_ssdr_stage_resolution,
    print_ssdr_resolution_info,
)


def path_has_any_children(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def discover_source_files(input_tree: Path) -> list[Path]:
    """
    Discover source files for Step 00.

    Supported:
        .json -> converted to .pkl
        .pkl  -> copied as .pkl
    """
    files: list[Path] = []

    for suffix in ("*.json", "*.pkl"):
        files.extend([p for p in input_tree.rglob(suffix) if p.is_file()])

    return sorted(
        files,
        key=lambda p: str(p.relative_to(input_tree)),
    )


def output_path_for_source_file(
    *,
    src_path: Path,
    input_tree: Path,
    output_tree: Path,
) -> Path:
    rel = src_path.relative_to(input_tree)

    if src_path.suffix.lower() == ".json":
        return output_tree / rel.with_suffix(".pkl")

    if src_path.suffix.lower() == ".pkl":
        return output_tree / rel

    raise ValueError(f"Unsupported source file for Step 00: {src_path}")


def process_single_source_file(
    src_path: Path,
    input_tree: Path,
    output_tree: Path,
) -> dict[str, Any]:
    rel = src_path.relative_to(input_tree)
    out_path = output_path_for_source_file(
        src_path=src_path,
        input_tree=input_tree,
        output_tree=output_tree,
    )

    suffix = src_path.suffix.lower()

    if suffix == ".json":
        obj = load_data(src_path)
        dump_pkl(out_path, obj)
        operation = "json_to_pkl"

    elif suffix == ".pkl":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, out_path)
        operation = "copy_pkl"

    else:
        raise RuntimeError(f"Unsupported source suffix: {src_path}")

    stat_in = src_path.stat()
    stat_out = out_path.stat()

    return {
        "operation": operation,
        "input": str(src_path),
        "relative_path": str(rel),
        "output": str(out_path),
        "input_bytes": int(stat_in.st_size),
        "output_bytes": int(stat_out.st_size),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)


def prepare_output_dir(
    output_tree: Path,
    *,
    overwrite_output: bool,
    allow_merge: bool,
) -> None:
    if overwrite_output and allow_merge:
        fail("--overwrite_output and --allow_merge are mutually exclusive.")

    if output_tree.exists():
        if not output_tree.is_dir():
            fail(f"Output path exists but is not a directory: {output_tree}")

        if overwrite_output:
            warn(f"Removing existing output directory because --overwrite_output is set: {output_tree}")
            shutil.rmtree(output_tree)
            output_tree.mkdir(parents=True, exist_ok=True)
            return

        if path_has_any_children(output_tree) and not allow_merge:
            fail(
                f"Output directory already exists and is non-empty: {output_tree}\n"
                "Use --overwrite_output to clear it, or --allow_merge to write into it."
            )

        output_tree.mkdir(parents=True, exist_ok=True)
        return

    output_tree.mkdir(parents=True, exist_ok=True)


def convert_or_copy_tree_to_data0(
    *,
    input_tree: Path,
    output_tree: Path,
    max_workers: int | None,
    overwrite_output: bool,
    allow_merge: bool,
    allow_no_files: bool,
    write_summary_file: bool,
) -> dict[str, Any]:
    with Timer("time_00_file_scanning"):
        source_files = discover_source_files(input_tree)

    if not source_files:
        msg = f"No .json or .pkl files found under input tree: {input_tree}"

        if allow_no_files:
            warn(msg)
            prepare_output_dir(
                output_tree,
                overwrite_output=overwrite_output,
                allow_merge=allow_merge,
            )
            summary = {
                "schema": "ogd_lbs_ssdr_step00_to_data0_v2",
                "input_tree": str(input_tree),
                "output_tree": str(output_tree),
                "source_file_count": 0,
                "converted_json_count": 0,
                "copied_pkl_count": 0,
                "processed_count": 0,
                "allow_no_files": True,
                "files": [],
            }
            if write_summary_file:
                write_summary(output_tree / "step00_convert_summary.json", summary)
            return summary

        fail(msg)

    prepare_output_dir(
        output_tree,
        overwrite_output=overwrite_output,
        allow_merge=allow_merge,
    )

    log(f"Found {len(source_files)} source files under: {input_tree}")
    log(f"Output tree: {output_tree}")

    processed: list[dict[str, Any]] = []

    with Timer("time_00_convert_or_copy"):
        if max_workers == 1:
            for src_path in source_files:
                result = process_single_source_file(src_path, input_tree, output_tree)
                processed.append(result)
                log(f"Processed ({result['operation']}): {result['relative_path']}")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_single_source_file, src_path, input_tree, output_tree)
                    for src_path in source_files
                ]

                for future in as_completed(futures):
                    result = future.result()
                    processed.append(result)
                    log(f"Processed ({result['operation']}): {result['relative_path']}")

    processed = sorted(processed, key=lambda item: item["relative_path"])

    converted_json_count = sum(1 for item in processed if item["operation"] == "json_to_pkl")
    copied_pkl_count = sum(1 for item in processed if item["operation"] == "copy_pkl")

    input_bytes = int(sum(item["input_bytes"] for item in processed))
    output_bytes = int(sum(item["output_bytes"] for item in processed))

    summary = {
        "schema": "ogd_lbs_ssdr_step00_to_data0_v2",
        "input_tree": str(input_tree),
        "output_tree": str(output_tree),
        "source_file_count": int(len(source_files)),
        "processed_count": int(len(processed)),
        "converted_json_count": int(converted_json_count),
        "copied_pkl_count": int(copied_pkl_count),
        "input_bytes_total": input_bytes,
        "output_bytes_total": output_bytes,
        "output_over_input_size_ratio": (
            float(output_bytes) / float(input_bytes)
            if input_bytes > 0
            else None
        ),
        "files": processed,
    }

    if write_summary_file:
        summary_path = output_tree / "step00_convert_summary.json"
        write_summary(summary_path, summary)
        log(f"Step 00 summary written: {summary_path}")

    record_metric("00_source_file_count", int(len(source_files)))
    record_metric("00_processed_count", int(len(processed)))
    record_metric("00_converted_json_count", int(converted_json_count))
    record_metric("00_copied_pkl_count", int(copied_pkl_count))
    record_metric("00_input_bytes_total", input_bytes)
    record_metric("00_output_bytes_total", output_bytes)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSDR Step 00: convert/copy source Fitbox tree into data0."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Primary input. Normally work_dir or work_dir/data0_json. "
            "With explicit --output_dir, this may also be a custom input tree."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Explicit output tree. If omitted, Step 00 writes to <work_dir>/data0."
        ),
    )
    parser.add_argument(
        "-m",
        "--metrics_dir",
        type=str,
        default=None,
        help="Optional directory to save metrics_00.json.",
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
        help="Remove existing output directory before Step 00.",
    )
    parser.add_argument(
        "--allow_merge",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--allow_no_files",
        action="store_true",
        help="Do not fail if no .json or .pkl files are found.",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Do not write step00_convert_summary.json.",
    )
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        help="Enable Timer profiling logs via common.Timer.",
    )

    args = parser.parse_args(script_argv())

    resolution = resolve_ssdr_stage_dirs(
        step_id="00",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="SSDR Step 00: Source Fitbox -> data0",
    )

    print_ssdr_resolution_info(resolution)

    try:
        validate_ssdr_stage_resolution(resolution)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    log("=" * 80)
    log("SSDR Step 00: Source Fitbox -> data0")
    log(f"work_dir          : {resolution.work_dir}")
    log(f"input_tree        : {resolution.input_tree}")
    log(f"output_tree       : {resolution.output_tree}")
    log(f"input_source      : {resolution.input_source}")
    log(f"output_source     : {resolution.output_source}")
    log(f"max_workers       : {args.max_workers}")
    log(f"overwrite_output  : {bool(args.overwrite_output)}")
    log(f"allow_merge       : {bool(args.allow_merge)}")
    log("=" * 80)

    with Timer("time_00_total"):
        summary = convert_or_copy_tree_to_data0(
            input_tree=resolution.input_tree,
            output_tree=resolution.output_tree,
            max_workers=args.max_workers,
            overwrite_output=bool(args.overwrite_output),
            allow_merge=bool(args.allow_merge),
            allow_no_files=bool(args.allow_no_files),
            write_summary_file=not bool(args.no_summary),
        )

    if args.metrics_dir:
        dump_metrics(Path(args.metrics_dir) / "metrics_00.json")

    log("SSDR Step 00 complete.")
    log(f"processed_count       : {summary['processed_count']}")
    log(f"converted_json_count  : {summary['converted_json_count']}")
    log(f"copied_pkl_count      : {summary['copied_pkl_count']}")
    log(f"output_tree           : {resolution.output_tree}")


if __name__ == "__main__":
    main()
