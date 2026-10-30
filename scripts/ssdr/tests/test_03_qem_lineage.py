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
    for item in [REPO_ROOT / "src", REPO_ROOT / "scripts" / "ssdr"]:
        item_s = str(item)
        if item_s not in sys.path:
            sys.path.insert(0, item_s)


ensure_repo_paths()

from ogd_lbs.utils.common import dump_pkl, load_pkl  # noqa: E402
from ogd_lbs.fitbox.schema import (  # noqa: E402
    KEY_BAKED_NORMALS,
    KEY_BAKED_VERTICES,
    KEY_BONE_NAMES,
    KEY_FACES,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTEX_NORMALS,
    KEY_VERTICES,
)


def load_step03_module():
    path = REPO_ROOT / "scripts" / "ssdr" / "03_create_low_poly_shapes.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_ssdr_step03_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_octahedron_mesh() -> dict:
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int64,
    )

    normals = vertices / np.maximum(np.linalg.norm(vertices, axis=1, keepdims=True), 1e-10)
    baked_vertices = np.stack([vertices.copy(), vertices.copy() * 1.01], axis=0)
    baked_normals = np.stack([normals.copy(), normals.copy()], axis=0)

    return {
        "name": "toy_octa",
        KEY_VERTICES: vertices,
        KEY_VERTEX_NORMALS: normals,
        KEY_FACES: faces,
        KEY_BAKED_VERTICES: baked_vertices,
        KEY_BAKED_NORMALS: baked_normals,
        KEY_BONE_NAMES: ["root"],
        KEY_VERTEX_BONE_WEIGHTS: np.ones((vertices.shape[0], 1), dtype=np.float64),
    }


class TestStep03QEMLineage(unittest.TestCase):
    def test_process_single_mesh_persists_qem_lineage(self) -> None:
        mod = load_step03_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_qem_lineage_") as td:
            root = Path(td)
            input_mesh = root / "toy_octa.pkl"
            out_dir = root / "out"
            out_dir.mkdir()

            dump_pkl(input_mesh, make_octahedron_mesh())

            cfg = dict(mod.BUILTIN_DEFAULTS)
            cfg.update(
                {
                    "simp_rate_base": 0.5,
                    "simp_rate_multiplier": 1.0,
                    "min_target_faces": 1,
                    "maximum_error": 9999999.0,
                    "qem_frames": "bind",
                    "qem_sample_count": 1,
                    "preserve_topology": False,
                    "preserve_boundary_mode": "none",
                    "preserve_normal": False,
                    "quality_check": False,
                    "hard_quality_check": False,
                    "validate_output": True,
                    "weak_validate": False,
                    "cleanup_mesh": True,
                }
            )

            summary = mod.process_single_mesh(
                mesh_path=input_mesh,
                output_tree=out_dir,
                global_cfg=cfg,
                mesh_overrides={},
                cli_specific_rates={},
                cli_mesh_rates={},
                n_poses=2,
                n_bones=1,
            )

            self.assertEqual(summary["name"], "toy_octa")
            self.assertTrue(summary["has_qem_vertex_map"])
            self.assertGreater(summary["qem_vertex_map_length"], 0)
            self.assertIn("has_qem_face_map", summary)
            self.assertIn("qem_face_map_length", summary)

            out_mesh = load_pkl(out_dir / "toy_octa.pkl")

            self.assertIn("qem_vertex_map", out_mesh)
            self.assertIn("qem_face_map", out_mesh)
            self.assertIn("qem_lineage", out_mesh)

            qem_vertex_map = np.asarray(out_mesh["qem_vertex_map"], dtype=np.int64)
            self.assertEqual(qem_vertex_map.ndim, 1)
            self.assertEqual(qem_vertex_map.shape[0], np.asarray(out_mesh[KEY_VERTICES]).shape[0])

            qem_face_map = out_mesh["qem_face_map"]
            if qem_face_map is not None:
                qem_face_map_arr = np.asarray(qem_face_map, dtype=np.int64)
                self.assertEqual(qem_face_map_arr.ndim, 1)
                self.assertEqual(qem_face_map_arr.shape[0], np.asarray(out_mesh[KEY_FACES]).shape[0])

            lineage = out_mesh["qem_lineage"]
            self.assertEqual(lineage["schema"], "ogd_lbs_qem_lineage_v1")
            self.assertEqual(lineage["mesh_name"], "toy_octa")
            self.assertIn("vertex_map_semantics", lineage)
            self.assertIn("face_map_semantics", lineage)


if __name__ == "__main__":
    unittest.main()
