# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Data-prep-specific path conventions for CLI scripts.

Unified staged directory convention:

    <experiment_dir>/
    ├── data_prep_work_dir/
    │   ├── prep_02_sdf_merge_config.json
    │   ├── data_prep_00/   # raw input .blend
    │   ├── data_prep_01/   # Prep 01 output
    │   ├── data_prep_02/   # Prep 02 output
    │   ├── data_prep_03/   # Prep 03 output
    │   ├── data_prep_04/   # Prep 04a output
    │   └── data_prep_05/   # Prep 05 Fitbox output
    └── ssdr_work_dir/
        ├── data0_json/
        ├── data0/
        ├── data1/
        └── data2/

Data-prep stage mapping:

    Prep 01:
        input : data_prep_00
        output: data_prep_01

    Prep 02a / 02b:
        input : data_prep_01
        output: data_prep_02

    Prep 03:
        input : data_prep_02
        output: data_prep_03

    Prep 04a:
        input : data_prep_03
        output: data_prep_04

    Prep 05:
        input : data_prep_04
        output: data_prep_05
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ogd_lbs.utils.work_dir import WorkDirContext, extract_base_stem


DriveMode = Literal["SURFACE", "LBS", "BOTH"]

DataPrepPathSource = Literal[
    "EXPLICIT_STAGE_DIR",
    "EXPLICIT_OVERRIDE",
    "WORK_DIR_DEFAULT",
]


DATA_PREP_SUFFIXES: tuple[str, ...] = (
    "_prep_04a_cleaned_full",
    "_prep_04a_bone_pruning",
    "_prep_03_surface_drive_mode",
    "_prep_03_lbs_drive_mode",
    "_prep_03_LBS_baseline",
    "_prep_03",
    "_prep_02",
    "_prep_01",
)


DATA_PREP_STAGE_INPUT_DIR_BY_STEP: dict[str, str] = {
    "01": "data_prep_00",
    "02": "data_prep_01",
    "02a": "data_prep_01",
    "02b": "data_prep_01",
    "03": "data_prep_02",
    "04a": "data_prep_03",
    "05": "data_prep_04",
}


DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP: dict[str, str] = {
    "01": "data_prep_01",
    "02": "data_prep_02",
    "02a": "data_prep_02",
    "02b": "data_prep_02",
    "03": "data_prep_03",
    "04a": "data_prep_04",
    "05": "data_prep_05",
}


DATA_PREP_KNOWN_STAGE_DIR_NAMES: tuple[str, ...] = (
    "data_prep_00",
    "data_prep_01",
    "data_prep_02",
    "data_prep_03",
    "data_prep_04",
    "data_prep_05",
)


@dataclass(frozen=True)
class DataPrepPathResolution:
    step_id: str
    step_title: str

    input_arg: Path
    output_arg: Path | None

    work_dir: Path
    input_tree: Path
    output_tree: Path

    input_source: DataPrepPathSource
    output_source: DataPrepPathSource

    input_stage_dir_name: str
    output_stage_dir_name: str

    mode: str


def resolve_path_no_strict(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def path_is_relative_to(child: Path, parent: Path) -> bool:
    child = child.resolve(strict=False)
    parent = parent.resolve(strict=False)

    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def extract_asset_base_stem(path_or_name: str | Path) -> str:
    return extract_base_stem(path_or_name, DATA_PREP_SUFFIXES)


def is_stage_blend_name(path_or_name: str | Path) -> bool:
    stem = Path(path_or_name).stem
    return extract_asset_base_stem(path_or_name) != stem


def make_data_prep_context(input_blend: str | Path) -> WorkDirContext:
    return WorkDirContext(
        input_blend,
        strip_suffixes=DATA_PREP_SUFFIXES,
        primary_expected_type="file",
    )


def make_data_prep_work_dir_context(work_dir: str | Path) -> WorkDirContext:
    return WorkDirContext(
        work_dir,
        strip_suffixes=DATA_PREP_SUFFIXES,
        primary_expected_type="dir",
    )


def get_data_prep_stage_input_dir_name(step_id: str) -> str:
    step = str(step_id)
    if step not in DATA_PREP_STAGE_INPUT_DIR_BY_STEP:
        raise ValueError(
            f"Unsupported data_prep step_id={step_id!r}. "
            f"Known input-stage steps={sorted(DATA_PREP_STAGE_INPUT_DIR_BY_STEP.keys())}"
        )
    return DATA_PREP_STAGE_INPUT_DIR_BY_STEP[step]


def get_data_prep_stage_output_dir_name(step_id: str) -> str:
    step = str(step_id)
    if step not in DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP:
        raise ValueError(
            f"Unsupported data_prep step_id={step_id!r}. "
            f"Known output-stage steps={sorted(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP.keys())}"
        )
    return DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP[step]


def resolve_data_prep_stage_dirs(
    *,
    step_id: str,
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    step_title: str = "",
) -> DataPrepPathResolution:
    """
    Resolve data_prep stage input/output directories.

    Mirrors SSDR behavior:

        --input_dir <work_dir>
            reads  <work_dir>/<stage_input_dir>
            writes <work_dir>/<stage_output_dir>

        --input_dir <work_dir>/<stage_input_dir>
            reads  that stage dir
            writes <work_dir>/<stage_output_dir>

        --input_dir <custom_input_tree> --output_dir <custom_output_tree>
            explicit custom mode
    """
    step = str(step_id)
    in_stage = get_data_prep_stage_input_dir_name(step)
    out_stage = get_data_prep_stage_output_dir_name(step)

    raw_input = resolve_path_no_strict(input_dir)
    raw_output = (
        resolve_path_no_strict(output_dir)
        if output_dir is not None and str(output_dir).strip()
        else None
    )

    if raw_input.name == in_stage:
        work_dir = raw_input.parent
        input_tree = raw_input
        input_source: DataPrepPathSource = "EXPLICIT_STAGE_DIR"
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
        output_source: DataPrepPathSource = "EXPLICIT_OVERRIDE"
    else:
        output_tree = work_dir / out_stage
        output_source = "WORK_DIR_DEFAULT"

    return DataPrepPathResolution(
        step_id=step,
        step_title=step_title or f"Data Prep Step {step}",
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


def validate_data_prep_stage_resolution(
    resolution: DataPrepPathResolution,
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
            errors.append(f"Input tree is not a directory: {resolution.input_tree}")

    if resolution.output_tree == resolution.input_tree:
        errors.append(f"Output tree would equal input tree: {resolution.output_tree}")

    if forbid_output_inside_input and path_is_relative_to(resolution.output_tree, resolution.input_tree):
        errors.append(
            f"Output tree must not be inside input tree: "
            f"output={resolution.output_tree}, input={resolution.input_tree}"
        )

    if errors:
        raise FileNotFoundError(
            "Data-prep path resolution validation failed:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )


def print_data_prep_resolution_info(resolution: DataPrepPathResolution) -> None:
    line = "=" * 96
    subline = "-" * 96

    print(line, flush=True)
    print(f"📂 [{resolution.step_title}] Data Prep WorkDir Resolution Info", flush=True)
    print(f"   step_id         : {resolution.step_id}", flush=True)
    print(f"   mode            : {resolution.mode}", flush=True)
    print(f"   work_dir        : {resolution.work_dir}", flush=True)
    print(subline, flush=True)
    print(f"   [INPUT_ARG ] input_dir   : {resolution.input_arg}", flush=True)
    print(
        f"   [INPUT    ] input_tree  : {resolution.input_tree} "
        f"({resolution.input_source} | exists={resolution.input_tree.exists()} | type=dir)",
        flush=True,
    )
    if resolution.output_arg is not None:
        print(f"   [OUTPUT_ARG] output_dir  : {resolution.output_arg}", flush=True)
    print(
        f"   [OUTPUT   ] output_tree : {resolution.output_tree} "
        f"({resolution.output_source} | exists={resolution.output_tree.exists()} | type=dir)",
        flush=True,
    )
    print(subline, flush=True)
    print(f"   default input stage dir  : {resolution.input_stage_dir_name}", flush=True)
    print(f"   default output stage dir : {resolution.output_stage_dir_name}", flush=True)
    print(line, flush=True)


def discover_unique_file(
    root: str | Path,
    *,
    pattern: str,
    role: str,
) -> Path:
    root = Path(root).expanduser().resolve(strict=False)

    if not root.exists():
        raise FileNotFoundError(f"Cannot discover {role}; root does not exist: {root}")

    if not root.is_dir():
        raise FileNotFoundError(f"Cannot discover {role}; root is not a directory: {root}")

    candidates = sorted([p for p in root.glob(pattern) if p.is_file()], key=lambda p: p.name)

    if not candidates:
        raise FileNotFoundError(
            f"Could not discover {role} under {root} with pattern {pattern!r}."
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous {role} under {root} with pattern {pattern!r}. "
            f"Candidates={[str(p) for p in candidates]}"
        )

    return candidates[0].resolve(strict=False)


def discover_unique_blend(
    root: str | Path,
    *,
    pattern: str,
    role: str,
) -> Path:
    if not pattern.endswith(".blend"):
        raise ValueError(f"discover_unique_blend pattern should end with .blend, got {pattern!r}")

    return discover_unique_file(root, pattern=pattern, role=role)


def discover_raw_input_blend(input_tree: str | Path) -> Path:
    """
    Discover the raw input .blend for Prep 01.

    Rules:
        - Search only direct children of input_tree.
        - Include *.blend.
        - Exclude known staged blend suffixes, e.g. *_prep_01.blend.
        - If zero or multiple raw blend files are found, fail-fast.
    """
    root = Path(input_tree).expanduser().resolve(strict=False)

    if not root.exists():
        raise FileNotFoundError(f"Prep 01 input tree does not exist: {root}")

    if not root.is_dir():
        raise FileNotFoundError(f"Prep 01 input tree is not a directory: {root}")

    candidates = []
    for path in sorted(root.glob("*.blend"), key=lambda p: p.name):
        if not path.is_file():
            continue
        if is_stage_blend_name(path):
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No raw .blend file found under Prep 01 input tree: {root}. "
            "Expected exactly one non-staged .blend file."
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple raw .blend files found under Prep 01 input tree: {root}. "
            f"Fail-fast to avoid selecting the wrong asset. "
            f"Candidates={[str(p) for p in candidates]}"
        )

    return candidates[0].resolve(strict=False)


def prep_01_output_blend_name(base_stem: str) -> str:
    return f"{base_stem}_prep_01.blend"


def prep_02b_output_blend_name(base_stem: str) -> str:
    return f"{base_stem}_prep_02.blend"


def prep_03_surface_output_blend_name(base_stem: str) -> str:
    return f"{base_stem}_prep_03_surface_drive_mode.blend"


def prep_03_lbs_output_blend_name(base_stem: str) -> str:
    return f"{base_stem}_prep_03_lbs_drive_mode.blend"


def prep_04a_output_blend_name(base_stem: str) -> str:
    return f"{base_stem}_prep_04a_cleaned_full.blend"


def prep_01_output_blend_rel(ctx: WorkDirContext) -> Path:
    return Path(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["01"]) / prep_01_output_blend_name(ctx.base_stem)


def prep_02a_config_rel(_ctx: WorkDirContext) -> Path:
    return Path("prep_02_sdf_merge_config.json")


def prep_02a_out_dir_rel(_ctx: WorkDirContext) -> Path:
    return Path(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["02a"]) / "sdf_exports"


def prep_02_manifest_rel(ctx: WorkDirContext) -> Path:
    return prep_02a_out_dir_rel(ctx) / "manifest.json"


def prep_02b_output_blend_rel(ctx: WorkDirContext) -> Path:
    return Path(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["02b"]) / prep_02b_output_blend_name(ctx.base_stem)


def prep_03_surface_output_blend_rel(ctx: WorkDirContext) -> Path:
    return Path(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["03"]) / prep_03_surface_output_blend_name(ctx.base_stem)


def prep_03_lbs_output_blend_rel(ctx: WorkDirContext) -> Path:
    return Path(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["03"]) / prep_03_lbs_output_blend_name(ctx.base_stem)


def prep_03_output_blend_rel(
    ctx: WorkDirContext,
    *,
    drive_mode: str = "SURFACE",
) -> Path:
    drive = validate_drive_mode(drive_mode)

    if drive == "SURFACE":
        return prep_03_surface_output_blend_rel(ctx)

    if drive == "LBS":
        return prep_03_lbs_output_blend_rel(ctx)

    raise ValueError(
        "drive_mode=BOTH produces two outputs. Use "
        "prep_03_surface_output_blend_rel() and prep_03_lbs_output_blend_rel()."
    )


def prep_04a_output_blend_rel(ctx: WorkDirContext) -> Path:
    return Path(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["04a"]) / prep_04a_output_blend_name(ctx.base_stem)


def prep_04b_manual_versions_dir_rel(_ctx: WorkDirContext) -> Path:
    return Path(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["04a"]) / "bone_reduced_versions"


def prep_05_out_dir_rel(
    _ctx: WorkDirContext,
    *,
    debug_frame_max: int = -1,
) -> Path:
    _ = int(debug_frame_max)
    return Path(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["05"])


def prep_05_fitbox_original_rel(ctx: WorkDirContext, *, debug_frame_max: int = -1) -> Path:
    return prep_05_out_dir_rel(ctx, debug_frame_max=debug_frame_max) / "fitbox_original"


def prep_05_fitbox_vdb_rel(ctx: WorkDirContext, *, debug_frame_max: int = -1) -> Path:
    return prep_05_out_dir_rel(ctx, debug_frame_max=debug_frame_max) / "fitbox_vdb"


def validate_drive_mode(drive_mode: str) -> DriveMode:
    drive = str(drive_mode).upper()

    if drive not in {"SURFACE", "LBS", "BOTH"}:
        raise ValueError(
            f"Unsupported drive_mode={drive_mode!r}. Expected SURFACE, LBS, or BOTH."
        )

    return drive  # type: ignore[return-value]
