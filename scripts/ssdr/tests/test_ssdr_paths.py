#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def ensure_repo_src_in_syspath() -> Path:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src_dir = parent / "src"
            src_s = str(src_dir)
            if src_s not in sys.path:
                sys.path.insert(0, src_s)
            return parent

    raise RuntimeError("Could not locate repository root.")


REPO_ROOT = ensure_repo_src_in_syspath()


def load_ssdr_paths_module():
    path = REPO_ROOT / "scripts" / "ssdr" / "paths.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_ssdr_paths", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)

    # Important for modules containing @dataclass:
    # dataclasses may inspect sys.modules[cls.__module__] while processing
    # postponed annotations. importlib.exec_module() does not automatically
    # insert module_from_spec() results into sys.modules.
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)
    return module


class TestSSDRPaths(unittest.TestCase):
    def test_step00_input_arg_as_work_dir(self) -> None:
        mod = load_ssdr_paths_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_paths_") as td:
            work_dir = Path(td)
            (work_dir / "data0_json").mkdir()

            res = mod.resolve_ssdr_stage_dirs(
                step_id="00",
                input_dir=work_dir,
                output_dir=None,
                step_title="test",
            )

            self.assertEqual(res.work_dir, work_dir.resolve(strict=False))
            self.assertEqual(res.input_tree, (work_dir / "data0_json").resolve(strict=False))
            self.assertEqual(res.output_tree, (work_dir / "data0").resolve(strict=False))
            self.assertEqual(res.input_source, "WORK_DIR_DEFAULT")
            self.assertEqual(res.output_source, "WORK_DIR_DEFAULT")

            mod.validate_ssdr_stage_resolution(res)

    def test_step00_input_arg_as_data0_json_dir(self) -> None:
        mod = load_ssdr_paths_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_paths_") as td:
            work_dir = Path(td)
            data0_json = work_dir / "data0_json"
            data0_json.mkdir()

            res = mod.resolve_ssdr_stage_dirs(
                step_id="00",
                input_dir=data0_json,
                output_dir=None,
                step_title="test",
            )

            self.assertEqual(res.work_dir, work_dir.resolve(strict=False))
            self.assertEqual(res.input_tree, data0_json.resolve(strict=False))
            self.assertEqual(res.output_tree, (work_dir / "data0").resolve(strict=False))
            self.assertEqual(res.input_source, "EXPLICIT_STAGE_DIR")

            mod.validate_ssdr_stage_resolution(res)

    def test_step00_work_dir_with_explicit_output_dir(self) -> None:
        mod = load_ssdr_paths_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_paths_") as td:
            root = Path(td)
            work_dir = root / "asset"
            custom_out = root / "custom_data0"
            (work_dir / "data0_json").mkdir(parents=True)

            res = mod.resolve_ssdr_stage_dirs(
                step_id="00",
                input_dir=work_dir,
                output_dir=custom_out,
                step_title="test",
            )

            self.assertEqual(res.work_dir, work_dir.resolve(strict=False))
            self.assertEqual(res.input_tree, (work_dir / "data0_json").resolve(strict=False))
            self.assertEqual(res.output_tree, custom_out.resolve(strict=False))
            self.assertEqual(res.input_source, "WORK_DIR_DEFAULT")
            self.assertEqual(res.output_source, "EXPLICIT_OVERRIDE")

            mod.validate_ssdr_stage_resolution(res)

    def test_explicit_custom_input_and_output_tree(self) -> None:
        mod = load_ssdr_paths_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_paths_") as td:
            root = Path(td)
            custom_input = root / "custom_fitbox"
            custom_output = root / "custom_output"
            custom_input.mkdir()

            res = mod.resolve_ssdr_stage_dirs(
                step_id="00",
                input_dir=custom_input,
                output_dir=custom_output,
                step_title="test",
            )

            self.assertEqual(res.work_dir, root.resolve(strict=False))
            self.assertEqual(res.input_tree, custom_input.resolve(strict=False))
            self.assertEqual(res.output_tree, custom_output.resolve(strict=False))
            self.assertEqual(res.input_source, "EXPLICIT_OVERRIDE")
            self.assertEqual(res.output_source, "EXPLICIT_OVERRIDE")

            mod.validate_ssdr_stage_resolution(res)

    def test_missing_default_stage_dir_fails_validation(self) -> None:
        mod = load_ssdr_paths_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_paths_") as td:
            work_dir = Path(td)

            res = mod.resolve_ssdr_stage_dirs(
                step_id="00",
                input_dir=work_dir,
                output_dir=None,
                step_title="test",
            )

            with self.assertRaises(FileNotFoundError):
                mod.validate_ssdr_stage_resolution(res)


if __name__ == "__main__":
    unittest.main()
