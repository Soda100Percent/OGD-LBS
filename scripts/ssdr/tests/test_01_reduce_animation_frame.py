#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


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

from ogd_lbs.utils.common import load_pkl, dump_pkl  # noqa: E402


def load_step01_module():
    path = REPO_ROOT / "scripts" / "ssdr" / "01_reduce_animation_frame.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_ssdr_step01", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")

    module = importlib.util.module_from_spec(spec)

    # Keep consistent with other tests and safe for dataclass-containing imports.
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)
    return module


def make_animation(frame_count: int, bone_count: int = 2, *, start_scene_frame: int = 100) -> dict:
    return {
        "bone_names": [f"bone_{i}" for i in range(bone_count)],
        "rest_pose": np.zeros((bone_count, 4, 4), dtype=np.float64),
        "animation_poses": np.arange(frame_count * bone_count * 16, dtype=np.float64).reshape(
            frame_count,
            bone_count,
            4,
            4,
        ),
        "scene_frames": np.arange(start_scene_frame, start_scene_frame + frame_count, dtype=np.int64),
    }


def make_mesh(frame_count: int, vertex_count: int = 4, bone_count: int = 2, *, start_scene_frame: int = 100) -> dict:
    return {
        "name": "mesh_A",
        "vertices": np.zeros((vertex_count, 3), dtype=np.float64),
        "faces": [[0, 1, 2]],
        "bone_names": [f"bone_{i}" for i in range(bone_count)],
        "vertex_bone_weights": np.ones((vertex_count, bone_count), dtype=np.float64) / float(bone_count),
        "baked_animation_vertices": np.arange(frame_count * vertex_count * 3, dtype=np.float64).reshape(
            frame_count,
            vertex_count,
            3,
        ),
        "baked_animation_vertex_normals": np.ones((frame_count, vertex_count, 3), dtype=np.float64),
        "scene_frames": np.arange(start_scene_frame, start_scene_frame + frame_count, dtype=np.int64),
    }


class TestSSDRStep01ReduceAnimationFrame(unittest.TestCase):
    def test_reduction_samples_all_animation_and_mesh_files(self) -> None:
        mod = load_step01_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step01_") as td:
            root = Path(td)
            input_tree = root / "data0"
            output_tree = root / "data1"

            (input_tree / "fitbox_vdb").mkdir(parents=True)
            (input_tree / "bone_reduced_versions" / "bone_reduced_001").mkdir(parents=True)

            dump_pkl(input_tree / "fitbox_vdb" / "animation_data.pkl", make_animation(12, 2, start_scene_frame=100))
            dump_pkl(input_tree / "fitbox_vdb" / "mesh_A.pkl", make_mesh(12, 4, 2, start_scene_frame=100))
            dump_pkl(
                input_tree / "bone_reduced_versions" / "bone_reduced_001" / "animation_data.pkl",
                make_animation(12, 1, start_scene_frame=100),
            )
            dump_pkl(
                input_tree / "fitbox_vdb" / "skeleton_hierarchy_data.pkl",
                {"bone_names": ["bone_0", "bone_1"], "parent_indices": [None, 0]},
            )

            summary = mod.reduce_animation_tree(
                input_tree=input_tree,
                output_tree=output_tree,
                max_frames=10,
                target_frames=5,
                max_workers=1,
                overwrite_output=False,
                allow_merge=False,
                allow_no_files=False,
                write_summary_file=True,
            )

            self.assertEqual(summary["processed_count"], 4)
            self.assertEqual(summary["sampled_animation_count"], 2)
            self.assertEqual(summary["sampled_mesh_count"], 1)
            self.assertEqual(summary["copied_pkl_count"], 1)
            self.assertTrue(summary["sampling"]["reduced"])
            self.assertEqual(summary["sampling"]["retained_count"], 5)

            idx = np.array(summary["sampling"]["selected_indices"], dtype=np.int64)
            expected_scene_frames = np.arange(100, 112, dtype=np.int64)[idx]

            out_anim = load_pkl(output_tree / "fitbox_vdb" / "animation_data.pkl")
            out_mesh = load_pkl(output_tree / "fitbox_vdb" / "mesh_A.pkl")
            out_variant_anim = load_pkl(
                output_tree / "bone_reduced_versions" / "bone_reduced_001" / "animation_data.pkl"
            )
            out_skel = load_pkl(output_tree / "fitbox_vdb" / "skeleton_hierarchy_data.pkl")

            self.assertEqual(out_anim["animation_poses"].shape[0], 5)
            self.assertEqual(out_mesh["baked_animation_vertices"].shape[0], 5)
            self.assertEqual(out_mesh["baked_animation_vertex_normals"].shape[0], 5)
            self.assertEqual(out_variant_anim["animation_poses"].shape[0], 5)
            self.assertTrue(np.array_equal(out_anim["scene_frames"], expected_scene_frames))
            self.assertTrue(np.array_equal(out_mesh["scene_frames"], expected_scene_frames))
            self.assertEqual(out_skel["bone_names"], ["bone_0", "bone_1"])
            self.assertTrue((output_tree / "step01_frame_sampling_summary.json").is_file())

    def test_no_reduction_when_under_threshold(self) -> None:
        mod = load_step01_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step01_") as td:
            root = Path(td)
            input_tree = root / "data0"
            output_tree = root / "data1"
            input_tree.mkdir()

            dump_pkl(input_tree / "animation_data.pkl", make_animation(12, 2, start_scene_frame=1))
            dump_pkl(input_tree / "mesh_A.pkl", make_mesh(12, 3, 2, start_scene_frame=1))

            summary = mod.reduce_animation_tree(
                input_tree=input_tree,
                output_tree=output_tree,
                max_frames=20,
                target_frames=10,
                max_workers=1,
                overwrite_output=False,
                allow_merge=False,
                allow_no_files=False,
                write_summary_file=False,
            )

            self.assertFalse(summary["sampling"]["reduced"])
            self.assertEqual(summary["sampling"]["retained_count"], 12)

            out_anim = load_pkl(output_tree / "animation_data.pkl")
            out_mesh = load_pkl(output_tree / "mesh_A.pkl")

            self.assertEqual(out_anim["animation_poses"].shape[0], 12)
            self.assertEqual(out_mesh["baked_animation_vertices"].shape[0], 12)
            self.assertTrue(np.array_equal(out_anim["scene_frames"], np.arange(1, 13)))

    def test_existing_non_empty_output_fails_without_overwrite_or_merge(self) -> None:
        mod = load_step01_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step01_") as td:
            root = Path(td)
            input_tree = root / "data0"
            output_tree = root / "data1"
            input_tree.mkdir()
            output_tree.mkdir()

            dump_pkl(input_tree / "animation_data.pkl", make_animation(12))
            (output_tree / "stale.pkl").write_bytes(b"stale")

            with self.assertRaises(RuntimeError):
                mod.reduce_animation_tree(
                    input_tree=input_tree,
                    output_tree=output_tree,
                    max_frames=10,
                    target_frames=5,
                    max_workers=1,
                    overwrite_output=False,
                    allow_merge=False,
                    allow_no_files=False,
                    write_summary_file=False,
                )

    def test_overwrite_output_allows_rerun(self) -> None:
        mod = load_step01_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step01_") as td:
            root = Path(td)
            input_tree = root / "data0"
            output_tree = root / "data1"
            input_tree.mkdir()
            output_tree.mkdir()

            dump_pkl(input_tree / "animation_data.pkl", make_animation(12))
            (output_tree / "stale.pkl").write_bytes(b"stale")

            summary = mod.reduce_animation_tree(
                input_tree=input_tree,
                output_tree=output_tree,
                max_frames=10,
                target_frames=5,
                max_workers=1,
                overwrite_output=True,
                allow_merge=False,
                allow_no_files=False,
                write_summary_file=False,
            )

            self.assertEqual(summary["processed_count"], 1)
            self.assertFalse((output_tree / "stale.pkl").exists())
            self.assertTrue((output_tree / "animation_data.pkl").is_file())


if __name__ == "__main__":
    unittest.main()
