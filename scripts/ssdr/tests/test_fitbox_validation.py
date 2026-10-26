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

from ogd_lbs.fitbox.schema import KEY_FACES, KEY_VERTICES  # noqa: E402
from ogd_lbs.fitbox.validation import validate_fitbox_mesh  # noqa: E402


def make_mesh(faces) -> dict:
    return {
        KEY_VERTICES: np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        KEY_FACES: faces,
    }


class TestFitboxValidation(unittest.TestCase):
    def test_triangle_faces_pass(self) -> None:
        mesh = make_mesh(
            np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                ],
                dtype=np.int64,
            )
        )

        validate_fitbox_mesh(mesh, name="tri_mesh")

    def test_quad_faces_fail(self) -> None:
        mesh = make_mesh(
            np.array(
                [
                    [0, 1, 2, 3],
                ],
                dtype=np.int64,
            )
        )

        with self.assertRaises(RuntimeError) as ctx:
            validate_fitbox_mesh(mesh, name="quad_mesh")

        self.assertIn("faces must have shape (F, 3)", str(ctx.exception))

    def test_empty_triangle_faces_fail(self) -> None:
        mesh = make_mesh(np.zeros((0, 3), dtype=np.int64))

        with self.assertRaises(RuntimeError) as ctx:
            validate_fitbox_mesh(mesh, name="empty_faces_mesh")

        self.assertIn("faces must have shape (F, 3)", str(ctx.exception))

    def test_ragged_faces_fail(self) -> None:
        mesh = make_mesh(
            [
                [0, 1, 2],
                [0, 2, 3, 1],
            ]
        )

        with self.assertRaises(RuntimeError):
            validate_fitbox_mesh(mesh, name="ragged_faces_mesh")


if __name__ == "__main__":
    unittest.main()
