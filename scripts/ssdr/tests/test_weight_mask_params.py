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

from ogd_lbs.fitbox.schema import KEY_VERTEX_BONE_WEIGHTS_MASK  # noqa: E402
from ogd_lbs.ssdr.weight_mask import build_weight_mask  # noqa: E402


class TestSSDRWeightMaskParams(unittest.TestCase):
    def test_disabled_mask_returns_all_ones(self) -> None:
        vbw = np.array([[0.9, 0.1], [0.0, 1.0]], dtype=np.float64)
        mask = build_weight_mask(
            {},
            vbw,
            False,
            "mesh",
            mask_mode="fixed",
            mask_threshold=0.5,
        )
        self.assertTrue(np.array_equal(mask, np.ones_like(vbw)))

    def test_adaptive_all_ones_generates_from_nonzero_weights(self) -> None:
        vbw = np.array([[0.9, 0.1, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        mesh = {KEY_VERTEX_BONE_WEIGHTS_MASK: np.ones_like(vbw)}
        mask = build_weight_mask(mesh, vbw, True, "mesh", mask_mode="adaptive")
        expected = (vbw > 0).astype(np.float64)
        self.assertTrue(np.array_equal(mask, expected))

    def test_fixed_threshold_tau(self) -> None:
        vbw = np.array([[0.9, 0.1, 0.0], [0.02, 0.0, 0.98]], dtype=np.float64)
        mask = build_weight_mask(
            {},
            vbw,
            True,
            "mesh",
            mask_mode="fixed",
            mask_threshold=0.05,
        )
        expected = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.assertTrue(np.array_equal(mask, expected))

    def test_fixed_threshold_zero_row_falls_back_to_nonzero_weights(self) -> None:
        vbw = np.array([[0.001, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        mask = build_weight_mask(
            {},
            vbw,
            True,
            "mesh",
            mask_mode="fixed",
            mask_threshold=0.01,
        )
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.assertTrue(np.array_equal(mask, expected))

    def test_fixed_threshold_requires_nonnegative_tau(self) -> None:
        vbw = np.array([[1.0, 0.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            build_weight_mask(
                {},
                vbw,
                True,
                "mesh",
                mask_mode="fixed",
                mask_threshold=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
