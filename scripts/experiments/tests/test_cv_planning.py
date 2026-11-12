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
from ogd_lbs.utils.common import dump_pkl, load_json  # noqa: E402


def load_step04():
    path = REPO_ROOT / "scripts" / "ssdr" / "04_optimize_weight_and_bone_trans.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_step04_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_augment():
    path = REPO_ROOT / "scripts" / "experiments" / "augment_stage_metrics.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_augment_for_tests", path)
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


class TestInlineCVAndAugmentation(unittest.TestCase):
    def test_inline_kfold_split_with_and_without_core_fraction(self) -> None:
        step04 = load_step04()

        # 1. Vroid style: first 10% (20 of 200 frames) always in train, 180 pool -> 45 test per fold
        for fold in range(4):
            split_vroid = step04.make_inline_kfold_split(
                200,
                kfold=4,
                fold_index=fold,
                special_train_fraction=0.1,
            )
            self.assertEqual(split_vroid.stats["special_count"], 20)
            self.assertEqual(split_vroid.stats["pool_count"], 180)
            self.assertEqual(split_vroid.stats["train_count"], 155)
            self.assertEqual(split_vroid.stats["test_count"], 45)
            core = set(range(20))
            self.assertTrue(core.issubset(set(split_vroid.train_indices.tolist())))
            self.assertFalse(core & set(split_vroid.test_indices.tolist()))
            self.assertFalse(set(split_vroid.train_indices.tolist()) & set(split_vroid.test_indices.tolist()))

        # 2. Griffin / Winter style: special_train_fraction=0.0 -> 150 train, 50 test per fold
        for fold in range(4):
            split_std = step04.make_inline_kfold_split(
                200,
                kfold=4,
                fold_index=fold,
                special_train_fraction=0.0,
            )
            self.assertEqual(split_std.stats["special_count"], 0)
            self.assertEqual(split_std.stats["pool_count"], 200)
            self.assertEqual(split_std.stats["train_count"], 150)
            self.assertEqual(split_std.stats["test_count"], 50)

    def test_step04_inline_kfold_trains_and_evaluates_test_split(self) -> None:
        step04 = load_step04()
        augment = load_augment()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_step04_inline_cv_") as td:
            root = Path(td)
            data3 = root / "data3"
            data4 = root / "data4"
            data3.mkdir()

            frame_count = 20
            dump_pkl(data3 / "skeleton_hierarchy_data.pkl", make_skel())
            dump_pkl(data3 / "animation_data.pkl", make_anim(frame_count))

            rest_v = np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            )
            baked_v = np.tile(rest_v[None, :, :], (frame_count, 1, 1))
            for f_idx in range(frame_count):
                baked_v[f_idx, :, 0] += 0.01 * float(f_idx)

            mesh_obj = {
                "name": "body",
                KEY_VERTICES: rest_v,
                KEY_VERTEX_NORMALS: np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (3, 1)),
                KEY_FACES: np.array([[0, 1, 2]], dtype=np.int64),
                KEY_BAKED_VERTICES: baked_v,
                KEY_BAKED_NORMALS: np.tile(np.array([[[0.0, 0.0, 1.0]]], dtype=np.float64), (frame_count, 3, 1)),
                KEY_BONE_NAMES: ["root"],
                KEY_VERTEX_BONE_WEIGHTS: np.ones((3, 1), dtype=np.float64),
                KEY_SCENE_FRAMES: np.arange(1, 1 + frame_count, dtype=np.int64),
            }
            dump_pkl(data3 / "body.pkl", mesh_obj)

            summary = step04.optimize_weight_and_bone_trans_tree(
                input_tree=data3,
                output_tree=data4,
                baked_schedule_path=None,
                print_every_n=1,
                max_iter=2,
                org_type=3,
                opt_weight=True,
                input_poses=True,
                use_weight_mask=True,
                s_max=4,
                mask_mode="adaptive",
                mask_threshold=-1.0,
                freeze_weights=False,
                weight_solver="simplex",
                use_weight_smoothness=False,
                weights_smooth_step=1.0,
                weight_soft_lock=0.0,
                lambda_reg=0.1,
                bt_covariance_mode="corrected",
                trans_affine=0.0,
                trans_affine_norm=4.0,
                roll_affinity=0.0,
                roll_affinity_norm=2.0,
                eval_mesh_health=True,
                health_level=1,
                health_every=1,
                health_weight_scheme="uniform",
                eval_bone_health=True,
                bone_health_every=1,
                max_workers=1,
                enable_profiling=False,
                validate_output=True,
                weak_validate=False,
                save_intermediate=False,
                overwrite_output=True,
                allow_merge=False,
                write_summary_file=True,
                kfold=4,
                fold_index=1,
                special_train_fraction=0.1,
            )

            self.assertEqual(summary["counts"]["input_frames_total"], 20)
            self.assertEqual(summary["counts"]["frames"], 15)
            self.assertFalse(summary["final_metrics"]["04_Freeze_Weights"])
            self.assertIn("test_final_metrics", summary)
            self.assertTrue(summary["test_final_metrics"]["04_Freeze_Weights"])

            test_summary = load_json(data4 / "step04_test_summary.json")
            self.assertEqual(test_summary["counts"]["frames"], 5)
            self.assertTrue(test_summary["config"]["freeze_weights"])
            self.assertEqual(test_summary["split"]["special_count"], 2)

            with np.load(data4 / "opt_out" / "metrics_history_per_frame.npz") as tr_npz:
                self.assertEqual(tr_npz["l2_err"].shape, (2, 15))
                self.assertEqual(tr_npz["hausdorff"].shape, (1, 15))
                self.assertEqual(tr_npz["arap_ratio"].shape, (1, 15))

            with np.load(data4 / "opt_out_test" / "metrics_history_per_frame.npz") as te_npz:
                self.assertEqual(te_npz["l2_err"].shape, (2, 5))
                self.assertEqual(te_npz["hausdorff"].shape, (1, 5))
                self.assertEqual(te_npz["arap_ratio"].shape, (1, 5))

            # Also verify augment_stage_metrics on both data3 (computed) and data4 (reused summary)
            norm_d3 = augment.compute_normalized_stage_metrics(data3, stage_name="ssdr_stage03")
            self.assertTrue(augment.is_valid_normalized_metrics_obj(norm_d3))
            self.assertEqual(norm_d3["complexity"]["num_vertices"], 3)
            self.assertEqual(norm_d3["complexity"]["num_triangles"], 1)
            self.assertEqual(norm_d3["complexity"]["num_bones_total"], 1)

            norm_d4 = augment.compute_normalized_stage_metrics(data4, stage_name="ssdr_stage04")
            self.assertTrue(augment.is_valid_normalized_metrics_obj(norm_d4))
            self.assertIsNotNone(norm_d4["test_metrics"])


if __name__ == "__main__":
    unittest.main()
