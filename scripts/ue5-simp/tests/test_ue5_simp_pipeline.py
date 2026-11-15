# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

UE5_SIMP_DIR = Path(__file__).resolve().parents[1]


class TestUE5SimpPipelineScripts(unittest.TestCase):
    def test_all_scripts_compile_and_show_help(self) -> None:
        scripts = [
            "00_prepare_textured_inputs.py",
            "01_reduce_animation_frame.py",
            "02_ue_reduce_bones_client.py",
            "03_ue_qem_low_poly_client.py",
            "03_ue_remesh_low_poly_client.py",
            "04_bake_and_evaluate.py",
        ]
        for s_name in scripts:
            s_path = UE5_SIMP_DIR / s_name
            self.assertTrue(s_path.is_file(), f"Missing script: {s_path}")
            proc = subprocess.run(
                [sys.executable, str(s_path), "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"{s_name} --help failed:\n{proc.stderr}")
            self.assertIn("--input_dir", proc.stdout)

    def test_default_config_valid_json(self) -> None:
        cfg_path = UE5_SIMP_DIR / "configs" / "default_ue5_simp.json"
        self.assertTrue(cfg_path.is_file())
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertIn("models", cfg)
        for m in ("Griffin_A2", "Winter_Samurai", "Vroid_18", "Vroid_19", "Vroid_20"):
            self.assertIn(m, cfg["models"])
            self.assertTrue(cfg["models"][m]["target_faces_map"])


if __name__ == "__main__":
    unittest.main()
