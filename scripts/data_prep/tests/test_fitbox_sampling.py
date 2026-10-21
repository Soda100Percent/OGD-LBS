#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import unittest
from pathlib import Path


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

from ogd_lbs.fitbox.sampling import calc_idxs_by_rate, resolve_sampled_frames  # noqa: E402


class TestFitboxSampling(unittest.TestCase):
    def test_calc_idxs_by_rate_old_semantics(self) -> None:
        self.assertEqual(calc_idxs_by_rate(10, 0.0, 1.0), [])
        self.assertEqual(calc_idxs_by_rate(10, 0.5, 0.0), [0, 2, 4, 6, 8])
        self.assertEqual(calc_idxs_by_rate(10, 0.5, 1.0), [1, 3, 5, 7, 9])

    def test_resolve_sampled_frames_no_removal(self) -> None:
        result = resolve_sampled_frames(
            frame_start=1,
            frame_end_exclusive=6,
            removal_rate=0.0,
            removal_bias=1.0,
            removal_idx_ranges=set(),
            debug_frame_max=-1,
        )
        self.assertEqual(result.retained_frames, [1, 2, 3, 4, 5])

    def test_resolve_sampled_frames_rate_and_debug(self) -> None:
        result = resolve_sampled_frames(
            frame_start=1,
            frame_end_exclusive=11,
            removal_rate=0.5,
            removal_bias=1.0,
            removal_idx_ranges=set(),
            debug_frame_max=3,
        )
        self.assertEqual(result.retained_frames_before_debug, [1, 3, 5, 7, 9])
        self.assertEqual(result.retained_frames, [1, 3, 5])

    def test_explicit_ranges(self) -> None:
        result = resolve_sampled_frames(
            frame_start=100,
            frame_end_exclusive=106,
            removal_rate=0.0,
            removal_bias=1.0,
            removal_idx_ranges={(1, 3)},
            debug_frame_max=-1,
        )
        self.assertEqual(result.retained_frames, [100, 103, 104, 105])

    def test_invalid_rate(self) -> None:
        with self.assertRaises(ValueError):
            resolve_sampled_frames(
                frame_start=1,
                frame_end_exclusive=10,
                removal_rate=1.0,
            )


if __name__ == "__main__":
    unittest.main()
