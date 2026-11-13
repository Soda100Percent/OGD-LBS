#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
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
for p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ogd_lbs.utils.common import dump_pkl  # noqa: E402


def load_render_mod():
    path = REPO_ROOT / "scripts" / "experiments" / "augment_stage_renders.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_augment_stage_renders_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_png_ihdr(path: Path) -> tuple[int, int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    w, h, _bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    return int(w), int(h), int(color_type)


class TestAugmentStageRendersBase(unittest.TestCase):
    def test_render_identity_and_subdir_dedup_keys(self) -> None:
        mod = load_render_mod()

        id1 = mod.build_render_identity(camera="front", frame=0, render_type="clay_512x512")
        id2 = mod.build_render_identity(camera="front", frame=0, render_type="clay_512x512")
        self.assertEqual(id1["render_key"], id2["render_key"])
        self.assertEqual(id1["output_subdir"], id2["output_subdir"])
        self.assertTrue(id1["output_subdir"].startswith("renders/clay_512x512/f0000_cam-front_"))

        id_f5 = mod.build_render_identity(camera="front", frame=5, render_type="clay_512x512")
        self.assertNotEqual(id1["render_key"], id_f5["render_key"])

        id_cam = mod.build_render_identity(camera="iso45", frame=0, render_type="clay_512x512")
        self.assertNotEqual(id1["render_key"], id_cam["render_key"])

        id_wire = mod.build_render_identity(camera="front", frame=0, render_type="wireframe_512x512")
        self.assertNotEqual(id1["render_key"], id_wire["render_key"])

    def test_headless_workbench_renders_optical_guardrails_and_fast_exit(self) -> None:
        mod = load_render_mod()
        script_path = REPO_ROOT / "scripts" / "experiments" / "augment_stage_renders.py"

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_render_qa_") as td:
            root = Path(td)
            data4 = root / "data4"
            data4.mkdir(parents=True)

            skel_obj = {"bone_names": ["Root", "Child"], "parent_indices": [None, 0]}
            rbt = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 2, axis=0)
            rbt[0, :3, 3] = [100.0, 200.0, 50.0]
            rbt[1, :3, 3] = [100.0, 200.0, 52.0]
            bt = np.repeat(rbt[None, :, :, :], 3, axis=0)
            bt[1, 1, 0, 3] += 0.3
            anim_obj = {"bone_names": ["Root", "Child"], "rest_pose": rbt, "animation_poses": bt}

            # Intentionally place mesh far from origin [100, 200, 50] to verify native C++ framing centers it
            verts = np.array(
                [
                    [99.0, 199.5, 50.0],
                    [101.0, 199.5, 50.0],
                    [100.0, 201.0, 50.0],
                    [100.0, 200.0, 52.5],
                ],
                dtype=np.float64,
            )
            faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
            baked = np.repeat(verts[None, :, :], 3, axis=0)
            vbw = np.array([[0.8, 0.2], [0.7, 0.3], [0.5, 0.5], [0.1, 0.9]], dtype=np.float64)

            dump_pkl(data4 / "skeleton_hierarchy_data.pkl", skel_obj)
            dump_pkl(data4 / "animation_data.pkl", anim_obj)
            dump_pkl(
                data4 / "body.pkl",
                {
                    "name": "body",
                    "vertices": verts,
                    "faces": faces,
                    "baked_animation_vertices": baked,
                    "vertex_bone_weights": vbw,
                    "bone_names": ["Root", "Child"],
                },
            )

            for rtype in ("clay_256x256", "wireframe_256x256", "bone_256x256", "l2_heatmap_256x256", "gt_clay_256x256"):
                out_sub = root / "out" / rtype
                cmd = [
                    sys.executable,
                    str(script_path),
                    "--input_dir",
                    str(data4),
                    "--output_dir",
                    str(out_sub),
                    "--stage",
                    "ssdr_stage04",
                    "--camera",
                    "iso45",
                    "--frame",
                    "1",
                    "--render_type",
                    rtype,
                ]
                proc1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
                lines1 = [ln for ln in proc1.stdout.splitlines() if ln.strip().startswith("{")]
                rep1 = json.loads(lines1[-1])
                self.assertEqual(rep1["status"], "AUGMENTED")
                self.assertGreater(rep1["alpha_coverage_pct"], 0.2)

                png_path = out_sub / "render.png"
                meta_path = out_sub / "render_meta.json"
                w, h, color_type = _read_png_ihdr(png_path)
                self.assertEqual((w, h), (256, 256))
                self.assertEqual(color_type, 6)

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                qa = meta["optical_qa"]
                self.assertTrue(qa["passed"])
                self.assertEqual(qa["border_clipped_px"], 0)
                self.assertLess(qa["bbox_center_offset"], 0.22)
                if rtype.startswith("l2_heatmap_"):
                    self.assertTrue((out_sub / "colorbar.png").is_file())
                    self.assertIn("heatmap_range", meta["camera_framing"])

                mtime_before = png_path.stat().st_mtime_ns
                proc2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
                lines2 = [ln for ln in proc2.stdout.splitlines() if ln.strip().startswith("{")]
                rep2 = json.loads(lines2[-1])
                self.assertEqual(rep2["status"], "ALREADY_AUGMENTED")
                self.assertEqual(png_path.stat().st_mtime_ns, mtime_before)

            # Verify Optical Guardrail rejects a blank/transparent image
            import bpy
            blank_png = root / "blank.png"
            blank_img = bpy.data.images.new("BlankTest", width=128, height=128, alpha=True)
            blank_img.pixels[:] = [0.0] * (128 * 128 * 4)
            blank_img.filepath_raw = str(blank_png)
            blank_img.file_format = "PNG"
            blank_img.save()
            bpy.data.images.remove(blank_img)

            bad_qa = mod.evaluate_optical_guardrails(blank_png, render_mode="clay")
            self.assertFalse(bad_qa["passed"])
            self.assertTrue(any("alpha_coverage_too_low" in r for r in bad_qa["reasons"]))


    def test_contact_sheet_generation_and_reverse_lookup(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "ogd_lbs_render_qa_inspector_test",
            REPO_ROOT / "scripts" / "experiments" / "render_qa_inspector.py",
        )
        assert spec is not None and spec.loader is not None
        insp = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = insp
        spec.loader.exec_module(insp)

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_sheet_test_") as td:
            root = Path(td)
            pool_dir = root / "pool"
            r1 = pool_dir / "results" / "r000001"
            sub = r1 / "data4" / "renders" / "clay_256x256" / "f0000_cam-front_12345678"
            sub.mkdir(parents=True)

            (r1 / "case_parameters.json").write_text(
                json.dumps({
                    "stage": "ssdr_stage04",
                    "pool_dimensions": {
                        "data_prep_handoff.model_name": "Griffin_A2",
                        "ssdr_stage02.variant": "bone_reduced_conservative",
                        "ssdr_stage02": "v1",
                        "ssdr_stage03": "ue_v1",
                    },
                }),
                encoding="utf-8",
            )

            img = np.zeros((128, 128, 4), dtype=np.float32)
            yy, xx = np.indices((128, 128))
            mask = (xx >= 32) & (xx < 96) & (yy >= 24) & (yy < 104)
            img[mask, 0] = 0.4 + 0.4 * ((xx[mask] - 32) / 64.0)
            img[mask, 1] = 0.5
            img[mask, 2] = 0.7
            img[mask, 3] = 1.0
            import matplotlib.pyplot as plt
            plt.imsave(str(sub / "render.png"), img)

            (sub / "render_meta.json").write_text(
                json.dumps({
                    "schema": "ogd_lbs_stage_render_meta_v1",
                    "render_key": "deadbeef12345678",
                    "output_subdir": "renders/clay_256x256/f0000_cam-front_12345678",
                    "stage": "ssdr_stage04",
                    "render_type": "clay_256x256",
                    "render_mode": "clay",
                    "frame_index_effective": 0,
                    "frame_token": "f0000",
                    "camera_id": "front_12345678",
                    "optical_qa": {
                        "passed": True,
                        "alpha_coverage_pct": 31.25,
                        "border_clipped_px": 0,
                        "bbox_center_offset": 0.0,
                    },
                }),
                encoding="utf-8",
            )

            items = insp.discover_pool_renders(pool_dir, qa_mode="fast-nn")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["thumb_code"], "#0001")
            self.assertEqual(items[0]["overall_status"], "PASS")
            self.assertTrue(items[0]["fast_nn_qa"]["passed"])

            sheet_dir = root / "sheets"
            idx_obj = insp.build_contact_sheets(items, sheet_dir, cols=2, rows_per_page=2)
            self.assertEqual(idx_obj["total_thumbnails"], 1)
            self.assertTrue((sheet_dir / "Render_QA.html").is_file())

            looked_up = insp.lookup_thumbnail_ids(sheet_dir / "contact_sheet_index.json", [1])
            self.assertEqual(len(looked_up), 1)
            self.assertEqual(looked_up[0]["data_dir"], "results/r000001")
            self.assertEqual(looked_up[0]["model"], "Griffin_A2")
            self.assertEqual(looked_up[0]["s03_version"], "ue_v1")


if __name__ == "__main__":
    unittest.main()
