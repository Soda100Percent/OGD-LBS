# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Data-prep-specific path conventions for CLI scripts.

This module intentionally lives under scripts/data_prep/, not src/ogd_lbs/,
because these names are pipeline entrypoint conventions rather than reusable
library algorithms.

Generic path resolution stays in:
    src/ogd_lbs/utils/work_dir.py
"""

from pathlib import Path
from typing import Literal

from ogd_lbs.utils.work_dir import WorkDirContext, extract_base_stem


DriveMode = Literal["SURFACE", "LBS", "BOTH"]


DATA_PREP_SUFFIXES: tuple[str, ...] = (
    "_prep_04a_cleaned_full",
    "_prep_04a_bone_pruning",       # legacy suffix, kept for compatibility
    "_prep_03_surface_drive_mode",
    "_prep_03_lbs_drive_mode",
    "_prep_03",
    "_prep_02",
    "_prep_01",
)


def extract_asset_base_stem(path_or_name: str | Path) -> str:
    return extract_base_stem(path_or_name, DATA_PREP_SUFFIXES)


def make_data_prep_context(input_blend: str | Path) -> WorkDirContext:
    return WorkDirContext(
        input_blend,
        strip_suffixes=DATA_PREP_SUFFIXES,
        primary_expected_type="file",
    )


def prep_01_output_blend_rel(ctx: WorkDirContext) -> Path:
    return Path(f"{ctx.base_stem}_prep_01.blend")


def prep_02a_config_rel(_ctx: WorkDirContext) -> Path:
    return Path("prep_02_sdf_merge_config.json")


def prep_02a_out_dir_rel(ctx: WorkDirContext) -> Path:
    return Path(f"{ctx.base_stem}_prep_01_sdf_exports")


def prep_02_manifest_rel(ctx: WorkDirContext) -> Path:
    return prep_02a_out_dir_rel(ctx) / "manifest.json"


def prep_02b_output_blend_rel(ctx: WorkDirContext) -> Path:
    return Path(f"{ctx.base_stem}_prep_02.blend")


def prep_03_surface_output_blend_rel(ctx: WorkDirContext) -> Path:
    return Path(f"{ctx.base_stem}_prep_03_surface_drive_mode.blend")


def prep_03_lbs_output_blend_rel(ctx: WorkDirContext) -> Path:
    return Path(f"{ctx.base_stem}_prep_03_lbs_drive_mode.blend")


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
    return Path(f"{ctx.base_stem}_prep_04a_cleaned_full.blend")


def prep_04b_manual_versions_dir_rel(_ctx: WorkDirContext) -> Path:
    return Path("bone_reduced_versions")


def prep_05_out_dir_rel(
    _ctx: WorkDirContext,
    *,
    debug_frame_max: int = -1,
) -> Path:
    if int(debug_frame_max) > 0:
        return Path(f"fitbox_exports_debug_{int(debug_frame_max)}f")

    return Path("fitbox_exports")


def validate_drive_mode(drive_mode: str) -> DriveMode:
    drive = str(drive_mode).upper()

    if drive not in {"SURFACE", "LBS", "BOTH"}:
        raise ValueError(
            f"Unsupported drive_mode={drive_mode!r}. Expected SURFACE, LBS, or BOTH."
        )

    return drive  # type: ignore[return-value]
