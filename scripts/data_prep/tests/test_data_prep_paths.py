#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test data-prep path conventions and generic WorkDirContext behavior.

This test is pure Python:
    - does not import bpy
    - does not open Blender files
    - does not depend on external datasets
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


def ensure_repo_paths_in_syspath() -> None:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src_dir = parent / "src"
            data_prep_dir = parent / "scripts" / "data_prep"

            for item in [src_dir, data_prep_dir]:
                item_s = str(item)
                if item_s not in sys.path:
                    sys.path.insert(0, item_s)

            return

    raise RuntimeError("Could not locate repository root containing pyproject.toml and src/.")


ensure_repo_paths_in_syspath()

from paths import (  # noqa: E402
    DATA_PREP_STAGE_INPUT_DIR_BY_STEP,
    DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP,
    discover_raw_input_blend,
    discover_unique_blend,
    extract_asset_base_stem,
    is_stage_blend_name,
    make_data_prep_context,
    prep_01_output_blend_name,
    prep_01_output_blend_rel,
    prep_02a_config_rel,
    prep_02a_out_dir_rel,
    prep_02_manifest_rel,
    prep_02b_output_blend_rel,
    prep_03_lbs_output_blend_rel,
    prep_03_output_blend_rel,
    prep_03_surface_output_blend_rel,
    prep_04a_output_blend_rel,
    prep_04b_manual_versions_dir_rel,
    prep_05_fitbox_original_rel,
    prep_05_fitbox_vdb_rel,
    prep_05_out_dir_rel,
    resolve_data_prep_stage_dirs,
    validate_data_prep_stage_resolution,
    validate_drive_mode,
)
from ogd_lbs.utils.work_dir import parse_idx_ranges  # noqa: E402


class TestDataPrepPathConventions(unittest.TestCase):
    def test_extract_asset_base_stem(self) -> None:
        cases = {
            "Griffin_A2.blend": "Griffin_A2",
            "Griffin_A2_prep_01.blend": "Griffin_A2",
            "Griffin_A2_prep_02.blend": "Griffin_A2",
            "Griffin_A2_prep_03.blend": "Griffin_A2",
            "Griffin_A2_prep_03_surface_drive_mode.blend": "Griffin_A2",
            "Griffin_A2_prep_03_lbs_drive_mode.blend": "Griffin_A2",
            "Griffin_A2_prep_04a_cleaned_full.blend": "Griffin_A2",
            "Griffin_A2_prep_04a_bone_pruning.blend": "Griffin_A2",
            "My_prep_Asset_prep_03.blend": "My_prep_Asset",
            "Asset_with_prep_word.blend": "Asset_with_prep_word",
        }

        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(extract_asset_base_stem(name), expected)

    def test_is_stage_blend_name(self) -> None:
        self.assertFalse(is_stage_blend_name("Griffin_A2.blend"))
        self.assertFalse(is_stage_blend_name("Asset_with_prep_word.blend"))
        self.assertTrue(is_stage_blend_name("Griffin_A2_prep_01.blend"))
        self.assertTrue(is_stage_blend_name("Griffin_A2_prep_03_lbs_drive_mode.blend"))

    def test_stage_directory_constants_shifted_for_raw_input(self) -> None:
        self.assertEqual(DATA_PREP_STAGE_INPUT_DIR_BY_STEP["01"], "data_prep_00")
        self.assertEqual(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["01"], "data_prep_01")

        self.assertEqual(DATA_PREP_STAGE_INPUT_DIR_BY_STEP["02a"], "data_prep_01")
        self.assertEqual(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["02a"], "data_prep_02")

        self.assertEqual(DATA_PREP_STAGE_INPUT_DIR_BY_STEP["03"], "data_prep_02")
        self.assertEqual(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["03"], "data_prep_03")

        self.assertEqual(DATA_PREP_STAGE_INPUT_DIR_BY_STEP["04a"], "data_prep_03")
        self.assertEqual(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["04a"], "data_prep_04")

        self.assertEqual(DATA_PREP_STAGE_INPUT_DIR_BY_STEP["05"], "data_prep_04")
        self.assertEqual(DATA_PREP_STAGE_OUTPUT_DIR_BY_STEP["05"], "data_prep_05")

    def test_default_path_conventions_shifted_stage_dirs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            root = Path(td)
            raw = root / "data_prep_00" / "Griffin_A2.blend"
            raw.parent.mkdir()
            raw.touch()

            ctx = make_data_prep_context(raw)

            self.assertEqual(ctx.work_dir, raw.parent.resolve())
            self.assertEqual(ctx.base_stem, "Griffin_A2")

            self.assertEqual(
                prep_01_output_blend_rel(ctx),
                Path("data_prep_01/Griffin_A2_prep_01.blend"),
            )
            self.assertEqual(prep_01_output_blend_name("Griffin_A2"), "Griffin_A2_prep_01.blend")

            self.assertEqual(prep_02a_config_rel(ctx), Path("prep_02_sdf_merge_config.json"))
            self.assertEqual(prep_02a_out_dir_rel(ctx), Path("data_prep_02/sdf_exports"))
            self.assertEqual(prep_02_manifest_rel(ctx), Path("data_prep_02/sdf_exports/manifest.json"))
            self.assertEqual(prep_02b_output_blend_rel(ctx), Path("data_prep_02/Griffin_A2_prep_02.blend"))

            self.assertEqual(
                prep_03_surface_output_blend_rel(ctx),
                Path("data_prep_03/Griffin_A2_prep_03_surface_drive_mode.blend"),
            )
            self.assertEqual(
                prep_03_lbs_output_blend_rel(ctx),
                Path("data_prep_03/Griffin_A2_prep_03_lbs_drive_mode.blend"),
            )
            self.assertEqual(
                prep_03_output_blend_rel(ctx, drive_mode="SURFACE"),
                Path("data_prep_03/Griffin_A2_prep_03_surface_drive_mode.blend"),
            )
            self.assertEqual(
                prep_03_output_blend_rel(ctx, drive_mode="LBS"),
                Path("data_prep_03/Griffin_A2_prep_03_lbs_drive_mode.blend"),
            )
            with self.assertRaises(ValueError):
                prep_03_output_blend_rel(ctx, drive_mode="BOTH")

            self.assertEqual(
                prep_04a_output_blend_rel(ctx),
                Path("data_prep_04/Griffin_A2_prep_04a_cleaned_full.blend"),
            )
            self.assertEqual(prep_04b_manual_versions_dir_rel(ctx), Path("data_prep_04/bone_reduced_versions"))

            self.assertEqual(prep_05_out_dir_rel(ctx, debug_frame_max=-1), Path("data_prep_05"))
            self.assertEqual(prep_05_out_dir_rel(ctx, debug_frame_max=5), Path("data_prep_05"))
            self.assertEqual(prep_05_fitbox_original_rel(ctx), Path("data_prep_05/fitbox_original"))
            self.assertEqual(prep_05_fitbox_vdb_rel(ctx), Path("data_prep_05/fitbox_vdb"))

    def test_validate_drive_mode(self) -> None:
        self.assertEqual(validate_drive_mode("SURFACE"), "SURFACE")
        self.assertEqual(validate_drive_mode("surface"), "SURFACE")
        self.assertEqual(validate_drive_mode("LBS"), "LBS")
        self.assertEqual(validate_drive_mode("lbs"), "LBS")
        self.assertEqual(validate_drive_mode("BOTH"), "BOTH")
        self.assertEqual(validate_drive_mode("both"), "BOTH")

        with self.assertRaises(ValueError):
            validate_drive_mode("BAD_MODE")

    def test_data_prep_stage_resolver_step01_input_arg_as_work_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            work_dir = Path(td)
            (work_dir / "data_prep_00").mkdir()

            res = resolve_data_prep_stage_dirs(
                step_id="01",
                input_dir=work_dir,
                output_dir=None,
                step_title="test",
            )

            self.assertEqual(res.work_dir, work_dir.resolve(strict=False))
            self.assertEqual(res.input_tree, (work_dir / "data_prep_00").resolve(strict=False))
            self.assertEqual(res.output_tree, (work_dir / "data_prep_01").resolve(strict=False))
            self.assertEqual(res.input_source, "WORK_DIR_DEFAULT")
            self.assertEqual(res.output_source, "WORK_DIR_DEFAULT")

            validate_data_prep_stage_resolution(res)

    def test_data_prep_stage_resolver_step01_input_arg_as_stage_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            work_dir = Path(td)
            stage = work_dir / "data_prep_00"
            stage.mkdir()

            res = resolve_data_prep_stage_dirs(
                step_id="01",
                input_dir=stage,
                output_dir=None,
                step_title="test",
            )

            self.assertEqual(res.work_dir, work_dir.resolve(strict=False))
            self.assertEqual(res.input_tree, stage.resolve(strict=False))
            self.assertEqual(res.output_tree, (work_dir / "data_prep_01").resolve(strict=False))
            self.assertEqual(res.input_source, "EXPLICIT_STAGE_DIR")
            self.assertEqual(res.output_source, "WORK_DIR_DEFAULT")

            validate_data_prep_stage_resolution(res)

    def test_data_prep_stage_resolver_step03(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            work_dir = Path(td)
            (work_dir / "data_prep_02").mkdir()

            res = resolve_data_prep_stage_dirs(
                step_id="03",
                input_dir=work_dir,
                output_dir=None,
                step_title="test",
            )

            self.assertEqual(res.input_tree, (work_dir / "data_prep_02").resolve(strict=False))
            self.assertEqual(res.output_tree, (work_dir / "data_prep_03").resolve(strict=False))
            validate_data_prep_stage_resolution(res)

    def test_data_prep_stage_resolver_explicit_custom_input_and_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            root = Path(td)
            custom_input = root / "custom_in"
            custom_output = root / "custom_out"
            custom_input.mkdir()

            res = resolve_data_prep_stage_dirs(
                step_id="01",
                input_dir=custom_input,
                output_dir=custom_output,
                step_title="test",
            )

            self.assertEqual(res.work_dir, root.resolve(strict=False))
            self.assertEqual(res.input_tree, custom_input.resolve(strict=False))
            self.assertEqual(res.output_tree, custom_output.resolve(strict=False))
            self.assertEqual(res.input_source, "EXPLICIT_OVERRIDE")
            self.assertEqual(res.output_source, "EXPLICIT_OVERRIDE")

            validate_data_prep_stage_resolution(res)

    def test_data_prep_missing_stage_dir_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            work_dir = Path(td)

            res = resolve_data_prep_stage_dirs(
                step_id="01",
                input_dir=work_dir,
                output_dir=None,
                step_title="test",
            )

            with self.assertRaises(FileNotFoundError):
                validate_data_prep_stage_resolution(res)

    def test_discover_raw_input_blend(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            stage = Path(td) / "data_prep_00"
            stage.mkdir()

            raw = stage / "Griffin_A2.blend"
            raw.touch()

            # Staged files should be ignored.
            (stage / "Griffin_A2_prep_01.blend").touch()

            found = discover_raw_input_blend(stage)
            self.assertEqual(found, raw.resolve(strict=False))

    def test_discover_raw_input_blend_missing_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            stage = Path(td) / "data_prep_00"
            stage.mkdir()

            (stage / "Griffin_A2_prep_01.blend").touch()

            with self.assertRaises(FileNotFoundError):
                discover_raw_input_blend(stage)

    def test_discover_raw_input_blend_multiple_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            stage = Path(td) / "data_prep_00"
            stage.mkdir()

            (stage / "A.blend").touch()
            (stage / "B.blend").touch()

            with self.assertRaises(RuntimeError):
                discover_raw_input_blend(stage)

    def test_discover_unique_blend(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            stage = Path(td) / "data_prep_02"
            stage.mkdir()

            target = stage / "Griffin_A2_prep_02.blend"
            target.touch()

            found = discover_unique_blend(
                stage,
                pattern="*_prep_02.blend",
                role="Prep 03 input blend",
            )
            self.assertEqual(found, target.resolve(strict=False))

    def test_discover_unique_blend_missing_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            stage = Path(td) / "data_prep_02"
            stage.mkdir()

            with self.assertRaises(FileNotFoundError):
                discover_unique_blend(
                    stage,
                    pattern="*_prep_02.blend",
                    role="Prep 03 input blend",
                )

    def test_discover_unique_blend_ambiguous_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            stage = Path(td) / "data_prep_02"
            stage.mkdir()

            (stage / "A_prep_02.blend").touch()
            (stage / "B_prep_02.blend").touch()

            with self.assertRaises(RuntimeError):
                discover_unique_blend(
                    stage,
                    pattern="*_prep_02.blend",
                    role="Prep 03 input blend",
                )

    def test_parse_idx_ranges(self) -> None:
        self.assertEqual(parse_idx_ranges(""), set())
        self.assertEqual(parse_idx_ranges("20:60,260:340"), {(20, 60), (260, 340)})
        self.assertEqual(parse_idx_ranges(" 1:3 , 5:8 "), {(1, 3), (5, 8)})

        with self.assertRaises(ValueError):
            parse_idx_ranges("20")

        with self.assertRaises(ValueError):
            parse_idx_ranges("60:20")

        with self.assertRaises(ValueError):
            parse_idx_ranges("-1:5")


if __name__ == "__main__":
    unittest.main()
