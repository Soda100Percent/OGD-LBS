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

from ogd_lbs.ssdr.weight_reduction import (  # noqa: E402
    SkeletonContext,
    align_skeleton_hierarchy_to_bone_names,
    reduce_mesh_weights,
)


class TestSSDRWeightReduction(unittest.TestCase):
    def test_align_skeleton_hierarchy_to_animation_bone_order(self) -> None:
        skel = {
            "bone_names": ["root", "a", "b"],
            "parent_indices": [None, 0, 1],
        }
        aligned = align_skeleton_hierarchy_to_bone_names(
            skel,
            ["a", "root", "b"],
        )

        self.assertEqual(aligned["bone_names"], ["a", "root", "b"])
        # a's parent root -> index 1 in target order.
        # root has no parent.
        # b's parent a -> index 0 in target order.
        self.assertEqual(aligned["parent_indices"], [1, None, 0])

    def test_reduce_mesh_weights_to_reduced_skeleton(self) -> None:
        full_skel = {
            "bone_names": ["root", "a", "b"],
            "parent_indices": [None, 0, 1],
        }
        reduced_skel = {
            "bone_names": ["root", "b"],
            "parent_indices": [None, 0],
        }

        ctx = SkeletonContext(full_skel, reduced_skel)

        mesh = {
            "name": "toy_mesh",
            "vertices": np.zeros((4, 3), dtype=np.float64),
            "faces": [[0, 1, 2]],
            "bone_names": ["root", "a", "b"],
            "vertex_bone_weights": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.2, 0.5, 0.3],
                ],
                dtype=np.float64,
            ),
        }

        reduced, stats = reduce_mesh_weights(mesh, ctx)

        self.assertEqual(reduced["bone_names"], ["root", "b"])
        self.assertEqual(reduced["vertex_bone_weights"].shape, (4, 2))
        self.assertEqual(stats.full_bone_count, 3)
        self.assertEqual(stats.reduced_bone_count, 2)
        self.assertEqual(stats.invalid_affinity_count, 0)

        row_sums = reduced["vertex_bone_weights"].sum(axis=1)
        self.assertTrue(np.allclose(row_sums, np.ones_like(row_sums), atol=1e-8))

    def test_mesh_bone_order_can_differ_from_full_skeleton(self) -> None:
        full_skel = {
            "bone_names": ["root", "a", "b"],
            "parent_indices": [None, 0, 1],
        }
        reduced_skel = {
            "bone_names": ["root", "b"],
            "parent_indices": [None, 0],
        }

        ctx = SkeletonContext(full_skel, reduced_skel)

        mesh = {
            "name": "toy_mesh_reordered",
            "bone_names": ["b", "root", "a"],
            "vertex_bone_weights": np.array(
                [
                    [0.0, 1.0, 0.0],  # root
                    [0.0, 0.0, 1.0],  # a
                    [1.0, 0.0, 0.0],  # b
                ],
                dtype=np.float64,
            ),
        }

        reduced, _stats = reduce_mesh_weights(mesh, ctx)
        self.assertEqual(reduced["bone_names"], ["root", "b"])
        self.assertEqual(reduced["vertex_bone_weights"].shape, (3, 2))

        row_sums = reduced["vertex_bone_weights"].sum(axis=1)
        self.assertTrue(np.allclose(row_sums, np.ones_like(row_sums), atol=1e-8))


if __name__ == "__main__":
    unittest.main()
