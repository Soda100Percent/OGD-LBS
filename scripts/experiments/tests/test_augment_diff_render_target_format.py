# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

from augment_diff_render_target_format import (  # noqa: E402
    build_target_format_from_diff_render_stage,
    is_valid_existing_target_format_dir,
)
from ogd_lbs.utils.common import dump_pkl, load_data  # noqa: E402


class TestAugmentDiffRenderTargetFormat(unittest.TestCase):
    def test_build_target_format_bakes_lbs_and_body_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data5 = root / "data5"
            opt_out = data5 / "opt_out"
            tex_dir = opt_out / "textures"
            mask_dir = opt_out / "masks"
            tex_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            dump_pkl(
                opt_out / "skeleton_hierarchy_data.pkl",
                {"bone_names": ["root"], "parent_indices": [None]},
            )
            eye = np.eye(4, dtype=np.float32)
            dump_pkl(
                opt_out / "animation_data.pkl",
                {
                    "bone_names": ["root"],
                    "rest_pose": eye[None, ...],
                    "animation_poses": np.repeat(eye[None, None, ...], 5, axis=0),
                },
            )
            dump_pkl(
                opt_out / "body_merged.pkl",
                {
                    "name": "body_merged",
                    "bone_names": ["root"],
                    "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                    "vertex_normals": np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
                    "vertex_bone_weights": np.ones((3, 1), dtype=np.float32),
                    "faces": np.array([[0, 1, 2]], dtype=np.int32),
                    "face_uvs": np.zeros((1, 3, 2), dtype=np.float32),
                },
            )

            # Solid opaque white texture + half-zero body mask
            tex_rgba = np.full((8, 8, 4), 255, dtype=np.uint8)
            Image.fromarray(tex_rgba, mode="RGBA").save(tex_dir / "body_merged_texture_kd.png")
            mask_l = np.full((8, 8), 255, dtype=np.uint8)
            mask_l[:4, :] = 0
            Image.fromarray(mask_l, mode="L").save(mask_dir / "body_merged_body_mask.png")

            out_dir = data5 / "target_format"
            summary = build_target_format_from_diff_render_stage(data5, out_dir)
            self.assertEqual(summary["mesh_count"], 1)
            self.assertTrue(is_valid_existing_target_format_dir(out_dir))
            self.assertTrue((out_dir / "body.pkl").is_file())
            self.assertTrue((out_dir / "body.png").is_file())
            self.assertTrue((out_dir / "body_mtl1.png").is_file())

            mesh_out = load_data(out_dir / "body.pkl")
            self.assertEqual(mesh_out["baked_animation_vertices"].shape, (5, 3, 3))
            self.assertEqual(mesh_out["baked_animation_vertex_normals"].shape, (5, 3, 3))
            self.assertEqual(mesh_out["material_names"], ["body"])

            with Image.open(out_dir / "body.png") as im_out:
                arr_out = np.asarray(im_out)
                self.assertEqual(int(arr_out[0, 0, 3]), 0)
                self.assertEqual(int(arr_out[6, 6, 3]), 255)


if __name__ == "__main__":
    unittest.main()
