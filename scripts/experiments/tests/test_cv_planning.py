#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def find_repo_root() -> Path:
    this_file = Path(__file__).resolve(strict=False)
    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Could not locate repository root.")


REPO_ROOT = find_repo_root()


def ensure_repo_paths() -> None:
    for item in [
        REPO_ROOT / "src",
        REPO_ROOT / "scripts" / "experiments",
        REPO_ROOT / "scripts" / "ssdr",
    ]:
        item_s = str(item)
        if item_s not in sys.path:
            sys.path.insert(0, item_s)


ensure_repo_paths()

from ogd_lbs.utils.common import dump_pkl, load_pkl  # noqa: E402
from ogd_lbs.fitbox.schema import (  # noqa: E402
    KEY_ANIMATION_POSES,
    KEY_BAKED_NORMALS,
    KEY_BAKED_VERTICES,
    KEY_BONE_NAMES,
    KEY_FACES,
    KEY_REST_POSE,
    KEY_SCENE_FRAMES,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTEX_NORMALS,
    KEY_VERTICES,
)


def load_cv_utils():
    import cv_utils
    return cv_utils


def load_builder():
    path = REPO_ROOT / "scripts" / "experiments" / "build_test_data3_from_train_topology.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_build_test_data3_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner():
    path = REPO_ROOT / "scripts" / "experiments" / "run_4fold_cv.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_run_4fold_cv_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_skel() -> dict:
    return {
        KEY_BONE_NAMES: ["root"],
        "parent_indices": [None],
    }


def make_anim(frame_count: int) -> dict:
    return {
        KEY_BONE_NAMES: ["root"],
        KEY_REST_POSE: np.eye(4, dtype=np.float64)[None, :, :],
        KEY_ANIMATION_POSES: np.tile(np.eye(4, dtype=np.float64), (frame_count, 1, 1, 1)),
        KEY_SCENE_FRAMES: np.arange(100, 100 + frame_count, dtype=np.int64),
    }


class TestCVPlanning(unittest.TestCase):
    def test_special_frames_never_enter_test(self) -> None:
        cv_utils = load_cv_utils()

        for fold in range(4):
            split = cv_utils.make_kfold_frame_split(
                12,
                kfold=4,
                fold_index=fold,
                special_train_ranges=[[0, 2]],
            )
            self.assertTrue({0, 1}.issubset(set(split.train_indices.tolist())))
            self.assertFalse({0, 1} & set(split.test_indices.tolist()))
            self.assertFalse(set(split.train_indices.tolist()) & set(split.test_indices.tolist()))
            self.assertGreater(split.test_indices.shape[0], 0)

    def test_build_test_data3_reuses_train_topology_and_test_animation(self) -> None:
        builder = load_builder()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_cv_build_test_") as td:
            root = Path(td)
            train_data3 = root / "train_data3"
            test_data2 = root / "test_data2"
            out = root / "data3_test"
            train_data3.mkdir()
            test_data2.mkdir()

            dump_pkl(train_data3 / "skeleton_hierarchy_data.pkl", make_skel())
            dump_pkl(test_data2 / "skeleton_hierarchy_data.pkl", make_skel())
            dump_pkl(train_data3 / "animation_data.pkl", make_anim(3))
            dump_pkl(test_data2 / "animation_data.pkl", make_anim(2))

            train_mesh = {
                "name": "mesh",
                KEY_VERTICES: np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
                KEY_VERTEX_NORMALS: np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float64),
                KEY_FACES: np.array([[0, 1, 1]], dtype=np.int64),
                KEY_BAKED_VERTICES: np.zeros((3, 2, 3), dtype=np.float64),
                KEY_BAKED_NORMALS: np.ones((3, 2, 3), dtype=np.float64),
                KEY_BONE_NAMES: ["root"],
                KEY_VERTEX_BONE_WEIGHTS: np.ones((2, 1), dtype=np.float64),
                "qem_vertex_map": np.array([0, 2], dtype=np.int64),
                "qem_face_map": np.array([0], dtype=np.int64),
                "qem_lineage": {"schema": "ogd_lbs_qem_lineage_v1"},
            }

            full_vertices = np.array(
                [
                    [[0, 0, 0], [10, 0, 0], [2, 0, 0]],
                    [[0, 1, 0], [10, 1, 0], [2, 1, 0]],
                ],
                dtype=np.float64,
            )

            test_mesh = {
                "name": "mesh",
                KEY_VERTICES: np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64),
                KEY_VERTEX_NORMALS: np.ones((3, 3), dtype=np.float64),
                KEY_FACES: np.array([[0, 1, 2]], dtype=np.int64),
                KEY_BAKED_VERTICES: full_vertices,
                KEY_BAKED_NORMALS: np.ones((2, 3, 3), dtype=np.float64),
                KEY_BONE_NAMES: ["root"],
                KEY_VERTEX_BONE_WEIGHTS: np.ones((3, 1), dtype=np.float64),
                KEY_SCENE_FRAMES: np.array([101, 105], dtype=np.int64),
            }

            dump_pkl(train_data3 / "mesh.pkl", train_mesh)
            dump_pkl(test_data2 / "mesh.pkl", test_mesh)

            summary = builder.build_test_data3_tree(
                train_data3=train_data3,
                test_data2=test_data2,
                output_dir=out,
                overwrite_output=True,
                validate_output=False,
                weak_validate=False,
            )

            self.assertEqual(summary["counts"]["frames"], 2)
            out_mesh = load_pkl(out / "mesh.pkl")
            self.assertTrue(np.array_equal(out_mesh[KEY_BAKED_VERTICES], full_vertices[:, [0, 2], :]))
            self.assertTrue(np.array_equal(out_mesh[KEY_VERTICES], train_mesh[KEY_VERTICES]))
            self.assertTrue(np.array_equal(out_mesh[KEY_VERTEX_BONE_WEIGHTS], train_mesh[KEY_VERTEX_BONE_WEIGHTS]))
            self.assertTrue(np.array_equal(out_mesh[KEY_SCENE_FRAMES], np.array([101, 105], dtype=np.int64)))

    def test_test_step04_command_forces_freeze_weights(self) -> None:
        runner = load_runner()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_cv_cmd_") as td:
            root = Path(td)
            cfg = {
                "conda_env": "dummy",
                "ssdr_defaults": {},
            }
            cmd = runner.build_step04_cmd(
                cfg,
                input_data3=root / "data3_test",
                output_data4=root / "data4_test",
                metrics_dir=root,
                step04_cfg={
                    "iter": 70,
                    "org_type": 3,
                    "opt_weight": False,
                    "input_poses": True,
                    "use_weight_mask": True,
                    "weight_solver": "simplex",
                    "s_max": 4,
                    "tau": "adaptive",
                },
                force_freeze_weights=True,
                no_conda=True,
            )

            self.assertIn("--freeze_weights", cmd)
            self.assertNotIn("--opt_weight", cmd)
            self.assertIn("--input_poses", cmd)
            self.assertIn("--use_weight_mask", cmd)
            self.assertIn("--mask_mode", cmd)
            self.assertIn("adaptive", cmd)


class TestCVDatasetSpecialTrainConfig(unittest.TestCase):
    def test_dataset_special_train_fraction_resolves_first_10_percent(self) -> None:
        cv_utils = load_cv_utils()

        ranges = cv_utils.resolve_special_train_ranges_for_dataset(
            frame_count=200,
            dataset={
                "model": "Vroid_18",
                "special_train_fraction": 0.10,
            },
            cfg={
                "special_train_ranges": [],
            },
        )

        self.assertEqual(ranges, [[0, 20]])

        split = cv_utils.make_kfold_frame_split(
            200,
            kfold=4,
            fold_index=0,
            special_train_ranges=ranges,
        )

        self.assertEqual(split.stats["special_count"], 20)
        self.assertEqual(split.stats["pool_count"], 180)
        self.assertEqual(split.stats["train_count"], 155)
        self.assertEqual(split.stats["test_count"], 45)

        core = set(range(20))
        self.assertTrue(core.issubset(set(split.train_indices.tolist())))
        self.assertFalse(core & set(split.test_indices.tolist()))

    def test_dataset_empty_ranges_override_global_core_block(self) -> None:
        cv_utils = load_cv_utils()

        ranges = cv_utils.resolve_special_train_ranges_for_dataset(
            frame_count=200,
            dataset={
                "model": "Griffin_A2",
                "special_train_ranges": [],
            },
            cfg={
                "special_train_ranges": [[0, 20]],
            },
        )

        self.assertEqual(ranges, [])

        split = cv_utils.make_kfold_frame_split(
            200,
            kfold=4,
            fold_index=0,
            special_train_ranges=ranges,
        )

        self.assertEqual(split.stats["special_count"], 0)
        self.assertEqual(split.stats["pool_count"], 200)
        self.assertEqual(split.stats["train_count"], 150)
        self.assertEqual(split.stats["test_count"], 50)

    def test_dataset_explicit_ranges_have_priority_over_fraction(self) -> None:
        cv_utils = load_cv_utils()

        ranges = cv_utils.resolve_special_train_ranges_for_dataset(
            frame_count=200,
            dataset={
                "model": "SomeModel",
                "special_train_ranges": [[5, 15]],
                "special_train_fraction": 0.10,
            },
            cfg={
                "special_train_ranges": [],
            },
        )

        self.assertEqual(ranges, [[5, 15]])


if __name__ == "__main__":
    unittest.main()
