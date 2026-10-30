#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def find_repo_root() -> Path:
    cur = Path(__file__).resolve(strict=False)
    for parent in [cur.parent, *cur.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("repo root not found")


REPO_ROOT = find_repo_root()

for item in [REPO_ROOT / "src", REPO_ROOT / "scripts" / "experiments"]:
    s = str(item)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_module(name: str, rel: str):
    path = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class TestArtifactCachePlanner(unittest.TestCase):
    def test_plan_marks_data3_cache_hit_and_skips_final_hit(self) -> None:
        rp = load_module("ogd_lbs_result_pool_i3_tests", "scripts/experiments/result_pool.py")
        art = load_module("ogd_lbs_result_pool_artifacts_i3_tests", "scripts/experiments/result_pool_artifacts.py")
        planner = load_module("ogd_lbs_artifact_cache_planner_tests", "scripts/experiments/plan_ofat_with_artifact_cache.py")

        with tempfile.TemporaryDirectory(prefix="ogd_i3_plan_") as td:
            root = Path(td)
            pool = root / "pool"
            rp.init_pool(pool)
            art.ensure_artifact_profiles(pool)

            # Create a final ssdr_data4 artifact for s_max=2.
            idx_path = pool / "index.json"
            index = json.loads(idx_path.read_text(encoding="utf-8"))

            a1 = "a000001"
            a1dir = pool / "artifacts" / a1
            a1dir.mkdir(parents=True)
            dump_json(
                a1dir / "raw_dimensions.json",
                {
                    "dimensions": {
                        "artifact_type": "ssdr_data4",
                        "model": "M",
                        "param": "s_max",
                        "value": 2,
                        "s_max": 2,
                        "tau": "adaptive",
                        "alpha": 0.5,
                        "beta": 0.3,
                        "gamma": 0.2,
                        "omega_b": 1000.0,
                    }
                },
            )
            index["artifacts"][a1] = {
                "artifact_id": a1,
                "artifact_type": "ssdr_data4",
                "status": "succeeded",
                "artifact_dir": f"artifacts/{a1}",
                "model": "M",
                "param": "s_max",
                "value": 2,
            }

            # Create a data3 artifact that can be reused by s_max=4.
            a2 = "a000002"
            a2dir = pool / "artifacts" / a2
            (a2dir / "payload" / "data3").mkdir(parents=True)
            (a2dir / "payload" / "data3" / "dummy.txt").write_text("data3\n", encoding="utf-8")
            dump_json(
                a2dir / "raw_dimensions.json",
                {
                    "dimensions": {
                        "artifact_type": "ssdr_data3",
                        "model": "M",
                        "bone_variant": "default",
                        "alpha": 0.5,
                        "beta": 0.3,
                        "gamma": 0.2,
                        "omega_b": 1000.0,
                    }
                },
            )
            index["artifacts"][a2] = {
                "artifact_id": a2,
                "artifact_type": "ssdr_data3",
                "status": "succeeded",
                "artifact_dir": f"artifacts/{a2}",
                "model": "M",
                "param": "s_max",
                "value": 4,
                "payload_path": "payload/data3",
            }

            idx_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

            cfg = {
                "schema": "ogd_lbs_ofat_sweep_config_v1",
                "output_root": str(root / "out"),
                "datasets": [
                    {
                        "model": "M",
                        "ssdr_work_dir": "/tmp/M/ssdr_work_dir",
                    }
                ],
                "defaults": {
                    "alpha": 0.5,
                    "beta": 0.3,
                    "gamma": 0.2,
                    "s_max": 4,
                    "tau": "adaptive",
                    "omega_b": 1000.0,
                },
                "abc_coupling": {"mode": "keep_sum_one"},
                "sweeps": {"s_max": [2, 4]},
                "delta_sweeps_by_model": {},
                "step04_defaults": {
                    "iter": 70,
                    "org_type": 3,
                    "opt_weight": True,
                    "input_poses": True,
                    "use_weight_mask": True,
                    "weight_solver": "simplex",
                    "bt_covariance_mode": "corrected",
                    "use_weight_smoothness": False,
                    "weights_smooth_step": 1.0,
                    "weight_soft_lock": 0.0,
                    "lambda_reg": 0.1,
                    "trans_affine": 0.0,
                    "trans_affine_norm": 4.0,
                    "roll_affinity": 0.0,
                    "roll_affinity_norm": 2.0,
                    "s_max": 4,
                },
                "ssdr_defaults": {
                    "max_frames": 210,
                    "target_frames": 200,
                },
            }
            cfg_path = root / "config.json"
            dump_json(cfg_path, cfg)

            plan_dir = root / "plan"
            manifest_path = plan_dir / "case_manifest.json"

            report = planner.plan(
                pool=pool,
                config_path=cfg_path,
                output_manifest=manifest_path,
                output_plan_dir=plan_dir,
                output_root=str(root / "filtered_out"),
                only_models=None,
                only_params=None,
                no_conda=True,
            )

            self.assertEqual(report["candidate_count"], 2)
            self.assertEqual(report["final_hit_count"], 1)
            self.assertEqual(report["miss_count"], 1)
            self.assertEqual(report["data3_cache_hit_count"], 1)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["case_count"], 1)
            case = manifest["cases"][0]
            self.assertEqual(case["value"], 4)
            self.assertIsNotNone(case["cache_hits"]["ssdr_data3"])
            self.assertIn("run_cached_ofat_case.py", " ".join(case["command"]))
            self.assertIn("--no_conda", case["command"])

            case_record = json.loads(Path(case["case_record_json"]).read_text(encoding="utf-8"))
            self.assertEqual(case_record["value"], 4)
            self.assertIsNotNone(case_record["cache_hits"]["ssdr_data3"])


if __name__ == "__main__":
    unittest.main()
