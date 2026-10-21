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
    extract_asset_base_stem,
    make_data_prep_context,
    prep_01_output_blend_rel,
    prep_02a_config_rel,
    prep_02a_out_dir_rel,
    prep_02_manifest_rel,
    prep_02b_output_blend_rel,
    prep_03_surface_output_blend_rel,
    prep_03_lbs_output_blend_rel,
    prep_03_output_blend_rel,
    prep_04a_output_blend_rel,
    prep_04b_manual_versions_dir_rel,
    prep_05_out_dir_rel,
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

    def test_default_path_conventions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            root = Path(td)
            blend = root / "Griffin_A2_prep_02.blend"
            blend.touch()

            ctx = make_data_prep_context(blend)

            self.assertEqual(ctx.work_dir, root.resolve())
            self.assertEqual(ctx.base_stem, "Griffin_A2")

            self.assertEqual(prep_01_output_blend_rel(ctx), Path("Griffin_A2_prep_01.blend"))
            self.assertEqual(prep_02a_config_rel(ctx), Path("prep_02_sdf_merge_config.json"))
            self.assertEqual(prep_02a_out_dir_rel(ctx), Path("Griffin_A2_prep_01_sdf_exports"))
            self.assertEqual(
                prep_02_manifest_rel(ctx),
                Path("Griffin_A2_prep_01_sdf_exports/manifest.json"),
            )
            self.assertEqual(prep_02b_output_blend_rel(ctx), Path("Griffin_A2_prep_02.blend"))

            self.assertEqual(
                prep_03_surface_output_blend_rel(ctx),
                Path("Griffin_A2_prep_03_surface_drive_mode.blend"),
            )
            self.assertEqual(
                prep_03_lbs_output_blend_rel(ctx),
                Path("Griffin_A2_prep_03_lbs_drive_mode.blend"),
            )
            self.assertEqual(
                prep_03_output_blend_rel(ctx, drive_mode="SURFACE"),
                Path("Griffin_A2_prep_03_surface_drive_mode.blend"),
            )
            self.assertEqual(
                prep_03_output_blend_rel(ctx, drive_mode="LBS"),
                Path("Griffin_A2_prep_03_lbs_drive_mode.blend"),
            )
            with self.assertRaises(ValueError):
                prep_03_output_blend_rel(ctx, drive_mode="BOTH")

            self.assertEqual(
                prep_04a_output_blend_rel(ctx),
                Path("Griffin_A2_prep_04a_cleaned_full.blend"),
            )
            self.assertEqual(
                prep_04b_manual_versions_dir_rel(ctx),
                Path("bone_reduced_versions"),
            )

            self.assertEqual(prep_05_out_dir_rel(ctx, debug_frame_max=-1), Path("fitbox_exports"))
            self.assertEqual(prep_05_out_dir_rel(ctx, debug_frame_max=5), Path("fitbox_exports_debug_5f"))

    def test_validate_drive_mode(self) -> None:
        self.assertEqual(validate_drive_mode("SURFACE"), "SURFACE")
        self.assertEqual(validate_drive_mode("surface"), "SURFACE")
        self.assertEqual(validate_drive_mode("LBS"), "LBS")
        self.assertEqual(validate_drive_mode("lbs"), "LBS")
        self.assertEqual(validate_drive_mode("BOTH"), "BOTH")
        self.assertEqual(validate_drive_mode("both"), "BOTH")

        with self.assertRaises(ValueError):
            validate_drive_mode("BAD_MODE")

    def test_work_dir_resolution_valid_case(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            root = Path(td)
            blend = root / "Griffin_A2_prep_02.blend"
            blend.touch()

            manifest_dir = root / "Griffin_A2_prep_01_sdf_exports"
            manifest_dir.mkdir()
            manifest = manifest_dir / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")

            ctx = make_data_prep_context(blend)

            input_blend = ctx.resolve_input(
                "input_blend",
                str(blend),
                None,
                expected_type="file",
            )
            manifest_path = ctx.resolve_input(
                "manifest",
                None,
                prep_02_manifest_rel(ctx),
                expected_type="file",
            )
            output_path = ctx.resolve_output(
                "output_blend",
                None,
                prep_03_surface_output_blend_rel(ctx),
                expected_type="file",
            )

            ctx.print_and_validate("TestDataPrepPathConventions.valid_case")

            self.assertEqual(input_blend, blend.resolve())
            self.assertEqual(manifest_path, manifest.resolve())
            self.assertEqual(
                output_path,
                (root / "Griffin_A2_prep_03_surface_drive_mode.blend").resolve(),
            )

    def test_missing_default_input_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            root = Path(td)
            blend = root / "Griffin_A2_prep_02.blend"
            blend.touch()

            ctx = make_data_prep_context(blend)
            ctx.resolve_input(
                "missing_manifest",
                None,
                Path("Griffin_A2_prep_01_sdf_exports/missing_manifest.json"),
                expected_type="file",
            )

            with self.assertRaises(FileNotFoundError):
                ctx.print_and_validate("TestDataPrepPathConventions.missing_default_input")

    def test_output_overwriting_input_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ogd_lbs_data_prep_paths_") as td:
            root = Path(td)
            blend = root / "Griffin_A2_prep_02.blend"
            blend.touch()

            ctx = make_data_prep_context(blend)
            ctx.resolve_input("input_blend", str(blend), None, expected_type="file")
            ctx.resolve_output("output_blend", str(blend), None, expected_type="file")

            with self.assertRaises(FileNotFoundError):
                ctx.print_and_validate("TestDataPrepPathConventions.output_overwrite_input")

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
