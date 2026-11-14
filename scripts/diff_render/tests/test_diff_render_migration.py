# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts" / "diff_render") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "diff_render"))

from ogd_lbs.diff_render.config import DiffRenderConfig  # noqa: E402
from ogd_lbs.diff_render.mesh_loader import BSDF_TO_ID, MeshLoader  # noqa: E402
from ogd_lbs.fitbox.schema import ANIMATION_STEM, SKELETON_STEM  # noqa: E402
from ogd_lbs.utils.common import clear_metrics, record_metric  # noqa: E402
from ogd_lbs.uv_mapping import generate_fitbox_uvs  # noqa: E402


class TestDiffRenderMigration(unittest.TestCase):
    def test_schema_and_metrics_helpers(self) -> None:
        self.assertEqual(ANIMATION_STEM, "animation_data")
        self.assertEqual(SKELETON_STEM, "skeleton_hierarchy_data")
        clear_metrics()
        record_metric("test_k", 1.25)
        clear_metrics()

    def test_auto_uv_xatlas(self) -> None:
        mesh_obj = {
            "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "faces": [[0, 1, 2], [0, 1, 3]],
        }
        out = generate_fitbox_uvs(mesh_obj, method="xatlas", overwrite=True)
        uvs = np.asarray(out["face_uvs"])
        self.assertEqual(uvs.shape, (2, 3, 2))

    def test_mesh_loader_excludes_non_mesh_metadata_jsons(self) -> None:
        import torch

        loader = MeshLoader(torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            t_dir = Path(tmp)
            # Write non-mesh metadata files that exist in PipelinePool stage dirs
            for meta_name in (
                "step04_optimize_summary.json",
                "normalized_metrics_v1.json",
                "case_parameters.json",
                "fold_plan.json",
                "body_bone_alpha.json",
                "target_export_summary.json",
                "animation_data.json",
                "skeleton_hierarchy_data.json",
            ):
                (t_dir / meta_name).write_text("{}", encoding="utf-8")

            # Write one valid mesh json
            (t_dir / "body_merged.json").write_text(
                json.dumps({"name": "body_merged", "vertices": [[0, 0, 0]], "faces": [[0, 0, 0]]}),
                encoding="utf-8",
            )
            infos = loader.scan_mesh_names(t_dir, ignore_suffixes=["_merged"])
            self.assertEqual(len(infos), 1)
            self.assertEqual(infos[0]["clean_stem"], "body")

    def test_stage_runner_cli_help(self) -> None:
        script = REPO_ROOT / "scripts" / "diff_render" / "03_run_diff_render_stage.py"
        res = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--pipeline_mode", res.stdout)
        self.assertIn("diffuse_2sided_back_deepen", BSDF_TO_ID)
        cfg = DiffRenderConfig()
        self.assertEqual(cfg.train_res, 512)


if __name__ == "__main__":
    unittest.main()
