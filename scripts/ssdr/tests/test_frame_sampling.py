#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


def ensure_repo_src_in_syspath() -> None:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src_dir = parent / "src"
            src_s = str(src_dir)
            if src_s not in sys.path:
                sys.path.insert(0, src_s)
            return

    raise RuntimeError("Could not locate repository root.")


ensure_repo_src_in_syspath()

from ogd_lbs.ssdr.frame_sampling import (  # noqa: E402
    make_dynamic_frame_sample_plan,
    apply_frame_sample_plan_to_animation,
    apply_frame_sample_plan_to_mesh,
)


class TestSSDRFrameSampling(unittest.TestCase):
    def test_no_reduction_when_under_threshold(self) -> None:
        plan = make_dynamic_frame_sample_plan(
            210,
            max_frames=210,
            target_frames=200,
        )

        self.assertFalse(plan.reduced)
        self.assertEqual(plan.selected_indices.shape[0], 210)
        self.assertEqual(int(plan.selected_indices[0]), 0)
        self.assertEqual(int(plan.selected_indices[-1]), 209)
        self.assertEqual(int(np.count_nonzero(plan.removal_mask)), 0)

    def test_reduction_to_target_count(self) -> None:
        plan = make_dynamic_frame_sample_plan(
            8437,
            max_frames=210,
            target_frames=200,
        )

        self.assertTrue(plan.reduced)
        self.assertEqual(plan.selected_indices.shape[0], 200)
        self.assertEqual(int(plan.selected_indices[0]), 0)
        self.assertEqual(int(plan.selected_indices[-1]), 8436)
        self.assertEqual(int(np.count_nonzero(plan.removal_mask)), 8437 - 200)

    def test_apply_to_animation(self) -> None:
        plan = make_dynamic_frame_sample_plan(
            12,
            max_frames=10,
            target_frames=5,
        )

        anim = {
            "bone_names": ["a", "b"],
            "rest_pose": np.zeros((2, 4, 4)),
            "animation_poses": np.zeros((12, 2, 4, 4)),
            "scene_frames": np.arange(100, 112),
        }

        out = apply_frame_sample_plan_to_animation(anim, plan)

        self.assertEqual(out["animation_poses"].shape, (5, 2, 4, 4))
        self.assertEqual(out["scene_frames"].shape, (5,))
        self.assertTrue(np.array_equal(out["scene_frames"], np.arange(100, 112)[plan.selected_indices]))
        self.assertEqual(out["rest_pose"].shape, (2, 4, 4))

    def test_apply_to_mesh(self) -> None:
        plan = make_dynamic_frame_sample_plan(
            12,
            max_frames=10,
            target_frames=5,
        )

        mesh = {
            "name": "mesh",
            "vertices": np.zeros((3, 3)),
            "faces": [[0, 1, 2]],
            "baked_animation_vertices": np.zeros((12, 3, 3)),
            "baked_animation_vertex_normals": np.zeros((12, 3, 3)),
            "scene_frames": np.arange(1, 13),
        }

        out = apply_frame_sample_plan_to_mesh(mesh, plan)

        self.assertEqual(out["baked_animation_vertices"].shape, (5, 3, 3))
        self.assertEqual(out["baked_animation_vertex_normals"].shape, (5, 3, 3))
        self.assertEqual(out["scene_frames"].shape, (5,))


if __name__ == "__main__":
    unittest.main()
