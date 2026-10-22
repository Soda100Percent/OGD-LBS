#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSDR pipeline path conventions.

Unlike data_prep, SSDR is a staged data directory pipeline:

    data0_json -> data0 -> data1 -> data2 -> ...

Default behavior:
    - First resolve work_dir.
    - Each step reads one hard-coded input directory under work_dir.
    - Each step writes one hard-coded output directory under work_dir.
    - Explicit input/output directory usage is still supported for automation.

Step defaults:
    Step 00:
        input : <work_dir>/data0_json
        output: <work_dir>/data0

    Step 01:
        input : <work_dir>/data0
        output: <work_dir>/data1

    Step 02:
        input : <work_dir>/data1
        output: <work_dir>/data2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PathSource = Literal[
    "EXPLICIT_STAGE_DIR",
    "EXPLICIT_OVERRIDE",
    "WORK_DIR_DEFAULT",
]


SSDR_STAGE_INPUT_DIR_BY_STEP: dict[str, str] = {
    "00": "data0_json",
    "01": "data0",
    "02": "data1",
}

SSDR_STAGE_OUTPUT_DIR_BY_STEP: dict[str, str] = {
    "00": "data0",
    "01": "data1",
    "02": "data2",
}

SSDR_KNOWN_STAGE_DIR_NAMES: tuple[str, ...] = (
    "data0_json",
    "data0",
    "data1",
    "data2",
    "data3",
    "data4",
    "data5",
)


@dataclass(frozen=True)
class SSDRPathResolution:
    step_id: str
    step_title: str

    input_arg: Path
    output_arg: Path | None

    work_dir: Path
    input_tree: Path
    output_tree: Path

    input_source: PathSource
    output_source: PathSource

    input_stage_dir_name: str
    output_stage_dir_name: str

    mode: str


def resolve_path_no_strict(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def get_stage_input_dir_name(step_id: str) -> str:
    step = str(step_id)
    if step not in SSDR_STAGE_INPUT_DIR_BY_STEP:
        raise ValueError(
            f"Unsupported SSDR step_id={step_id!r}. "
            f"Known steps={sorted(SSDR_STAGE_INPUT_DIR_BY_STEP.keys())}"
        )
    return SSDR_STAGE_INPUT_DIR_BY_STEP[step]


def get_stage_output_dir_name(step_id: str) -> str:
    step = str(step_id)
    if step not in SSDR_STAGE_OUTPUT_DIR_BY_STEP:
        raise ValueError(
            f"Unsupported SSDR step_id={step_id!r}. "
            f"Known steps={sorted(SSDR_STAGE_OUTPUT_DIR_BY_STEP.keys())}"
        )
    return SSDR_STAGE_OUTPUT_DIR_BY_STEP[step]


def resolve_ssdr_stage_dirs(
    *,
    step_id: str,
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    step_title: str = "",
) -> SSDRPathResolution:
    """
    Resolve SSDR step input/output directories.

    Priority:
        1. If input_dir itself is the expected stage input dir
           e.g. Step 00 gets /path/to/work_dir/data0_json:
               work_dir = input_dir.parent
               input_tree = input_dir

        2. Else if input_dir contains the expected stage input dir
           e.g. Step 00 gets /path/to/work_dir:
               work_dir = input_dir
               input_tree = input_dir/data0_json

        3. Else if output_dir is explicit:
               input_tree = input_dir
               work_dir = input_dir.parent
           This supports automation with custom explicit input/output trees.

        4. Else:
               input_dir is treated as work_dir and the hard-coded default
               input_tree is input_dir/<stage_input_dir>. Validation will fail
               if that directory does not exist.

    Output:
        - If output_dir is explicit, use it.
        - Otherwise use <work_dir>/<stage_output_dir>.
    """
    step = str(step_id)
    in_stage = get_stage_input_dir_name(step)
    out_stage = get_stage_output_dir_name(step)

    raw_input = resolve_path_no_strict(input_dir)
    raw_output = resolve_path_no_strict(output_dir) if output_dir is not None and str(output_dir).strip() else None

    if raw_input.name == in_stage:
        work_dir = raw_input.parent
        input_tree = raw_input
        input_source: PathSource = "EXPLICIT_STAGE_DIR"
        mode = "input_arg_is_expected_stage_dir"

    elif (raw_input / in_stage).exists() and (raw_input / in_stage).is_dir():
        work_dir = raw_input
        input_tree = raw_input / in_stage
        input_source = "WORK_DIR_DEFAULT"
        mode = "input_arg_is_work_dir_with_stage_dir"

    elif raw_output is not None:
        work_dir = raw_input.parent
        input_tree = raw_input
        input_source = "EXPLICIT_OVERRIDE"
        mode = "explicit_input_tree_and_output_tree"

    else:
        work_dir = raw_input
        input_tree = raw_input / in_stage
        input_source = "WORK_DIR_DEFAULT"
        mode = "input_arg_treated_as_work_dir_missing_or_unvalidated_stage_dir"

    if raw_output is not None:
        output_tree = raw_output
        output_source: PathSource = "EXPLICIT_OVERRIDE"
    else:
        output_tree = work_dir / out_stage
        output_source = "WORK_DIR_DEFAULT"

    return SSDRPathResolution(
        step_id=step,
        step_title=step_title or f"SSDR Step {step}",
        input_arg=raw_input,
        output_arg=raw_output,
        work_dir=work_dir.resolve(strict=False),
        input_tree=input_tree.resolve(strict=False),
        output_tree=output_tree.resolve(strict=False),
        input_source=input_source,
        output_source=output_source,
        input_stage_dir_name=in_stage,
        output_stage_dir_name=out_stage,
        mode=mode,
    )


def path_is_relative_to(child: Path, parent: Path) -> bool:
    child = child.resolve(strict=False)
    parent = parent.resolve(strict=False)

    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_ssdr_stage_resolution(
    resolution: SSDRPathResolution,
    *,
    require_input_exists: bool = True,
    forbid_output_inside_input: bool = True,
) -> None:
    errors: list[str] = []

    if require_input_exists:
        if not resolution.input_tree.exists():
            errors.append(
                f"Missing input tree: {resolution.input_tree} "
                f"(source={resolution.input_source}, expected stage dir={resolution.input_stage_dir_name})"
            )
        elif not resolution.input_tree.is_dir():
            errors.append(
                f"Input tree is not a directory: {resolution.input_tree}"
            )

    if resolution.output_tree == resolution.input_tree:
        errors.append(
            f"Output tree would equal input tree: {resolution.output_tree}"
        )

    if forbid_output_inside_input and path_is_relative_to(resolution.output_tree, resolution.input_tree):
        errors.append(
            f"Output tree must not be inside input tree: output={resolution.output_tree}, input={resolution.input_tree}"
        )

    if errors:
        raise FileNotFoundError(
            "SSDR path resolution validation failed:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )


def print_ssdr_resolution_info(resolution: SSDRPathResolution) -> None:
    line = "=" * 96
    subline = "-" * 96

    print(line, flush=True)
    print(f"📂 [{resolution.step_title}] SSDR WorkDir Resolution Info", flush=True)
    print(f"   step_id         : {resolution.step_id}", flush=True)
    print(f"   mode            : {resolution.mode}", flush=True)
    print(f"   work_dir        : {resolution.work_dir}", flush=True)
    print(subline, flush=True)
    print(f"   [INPUT_ARG ] input_dir   : {resolution.input_arg}", flush=True)
    print(f"   [INPUT    ] input_tree  : {resolution.input_tree} ({resolution.input_source} | exists={resolution.input_tree.exists()} | type=dir)", flush=True)
    if resolution.output_arg is not None:
        print(f"   [OUTPUT_ARG] output_dir  : {resolution.output_arg}", flush=True)
    print(f"   [OUTPUT   ] output_tree : {resolution.output_tree} ({resolution.output_source} | exists={resolution.output_tree.exists()} | type=dir)", flush=True)
    print(subline, flush=True)
    print(f"   default input stage dir  : {resolution.input_stage_dir_name}", flush=True)
    print(f"   default output stage dir : {resolution.output_stage_dir_name}", flush=True)
    print(line, flush=True)
