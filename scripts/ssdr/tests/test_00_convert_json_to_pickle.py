#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path


def ensure_repo_src_in_syspath() -> Path:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src_dir = parent / "src"
            ssdr_dir = parent / "scripts" / "ssdr"
            for item in [src_dir, ssdr_dir]:
                item_s = str(item)
                if item_s not in sys.path:
                    sys.path.insert(0, item_s)
            return parent

    raise RuntimeError("Could not locate repository root.")


REPO_ROOT = ensure_repo_src_in_syspath()

from ogd_lbs.utils.common import load_pkl  # noqa: E402


def load_step00_module():
    path = REPO_ROOT / "scripts" / "ssdr" / "00_convert_json_to_pickle.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_ssdr_step00", path)
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


class TestSSDRStep00ConvertJsonToPickle(unittest.TestCase):
    def test_json_to_pkl_and_pkl_copy_preserve_tree(self) -> None:
        mod = load_step00_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step00_") as td:
            root = Path(td)
            input_tree = root / "data0_json"
            output_tree = root / "data0"

            (input_tree / "fitbox_vdb").mkdir(parents=True)
            (input_tree / "bone_reduced_versions" / "bone_reduced_001").mkdir(parents=True)

            with (input_tree / "fitbox_vdb" / "animation_data.json").open("w", encoding="utf-8") as f:
                json.dump({"bone_names": ["root"], "animation_poses": [[1, 2, 3]]}, f)

            with (input_tree / "fitbox_vdb" / "mesh_A.pkl").open("wb") as f:
                pickle.dump({"name": "mesh_A", "vertices": [[0, 0, 0]]}, f)

            with (
                input_tree
                / "bone_reduced_versions"
                / "bone_reduced_001"
                / "skeleton_hierarchy_data.json"
            ).open("w", encoding="utf-8") as f:
                json.dump({"bone_names": ["root"], "parent_indices": [None]}, f)

            summary = mod.convert_or_copy_tree_to_data0(
                input_tree=input_tree,
                output_tree=output_tree,
                max_workers=1,
                overwrite_output=False,
                allow_merge=False,
                allow_no_files=False,
                write_summary_file=True,
            )

            self.assertEqual(summary["processed_count"], 3)
            self.assertEqual(summary["converted_json_count"], 2)
            self.assertEqual(summary["copied_pkl_count"], 1)

            anim_pkl = output_tree / "fitbox_vdb" / "animation_data.pkl"
            mesh_pkl = output_tree / "fitbox_vdb" / "mesh_A.pkl"
            skel_pkl = (
                output_tree
                / "bone_reduced_versions"
                / "bone_reduced_001"
                / "skeleton_hierarchy_data.pkl"
            )

            self.assertTrue(anim_pkl.is_file())
            self.assertTrue(mesh_pkl.is_file())
            self.assertTrue(skel_pkl.is_file())
            self.assertTrue((output_tree / "step00_convert_summary.json").is_file())

            self.assertEqual(load_pkl(anim_pkl)["bone_names"], ["root"])
            self.assertEqual(load_pkl(mesh_pkl)["name"], "mesh_A")
            self.assertEqual(load_pkl(skel_pkl)["parent_indices"], [None])

    def test_existing_non_empty_output_fails_without_overwrite_or_merge(self) -> None:
        mod = load_step00_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step00_") as td:
            root = Path(td)
            input_tree = root / "data0_json"
            output_tree = root / "data0"
            input_tree.mkdir()
            output_tree.mkdir()

            with (input_tree / "a.json").open("w", encoding="utf-8") as f:
                json.dump({"x": 1}, f)

            (output_tree / "stale.pkl").write_bytes(b"stale")

            with self.assertRaises(RuntimeError):
                mod.convert_or_copy_tree_to_data0(
                    input_tree=input_tree,
                    output_tree=output_tree,
                    max_workers=1,
                    overwrite_output=False,
                    allow_merge=False,
                    allow_no_files=False,
                    write_summary_file=False,
                )

    def test_overwrite_output_allows_rerun(self) -> None:
        mod = load_step00_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step00_") as td:
            root = Path(td)
            input_tree = root / "data0_json"
            output_tree = root / "data0"
            input_tree.mkdir()
            output_tree.mkdir()

            with (input_tree / "a.json").open("w", encoding="utf-8") as f:
                json.dump({"x": 1}, f)

            (output_tree / "stale.pkl").write_bytes(b"stale")

            summary = mod.convert_or_copy_tree_to_data0(
                input_tree=input_tree,
                output_tree=output_tree,
                max_workers=1,
                overwrite_output=True,
                allow_merge=False,
                allow_no_files=False,
                write_summary_file=False,
            )

            self.assertEqual(summary["processed_count"], 1)
            self.assertFalse((output_tree / "stale.pkl").exists())
            self.assertEqual(load_pkl(output_tree / "a.pkl")["x"], 1)

    def test_discover_ignores_non_json_non_pkl(self) -> None:
        mod = load_step00_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step00_") as td:
            root = Path(td)
            input_tree = root / "data0_json"
            input_tree.mkdir()

            (input_tree / "ignored.txt").write_text("hello", encoding="utf-8")
            with (input_tree / "a.json").open("w", encoding="utf-8") as f:
                json.dump({"x": 1}, f)

            files = mod.discover_source_files(input_tree)
            self.assertEqual([p.name for p in files], ["a.json"])


if __name__ == "__main__":
    unittest.main()
