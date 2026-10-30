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


def load_module():
    path = REPO_ROOT / "scripts" / "experiments" / "result_pool.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_result_pool_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class TestResultPool(unittest.TestCase):
    def test_adaptive_tau_equivalence(self) -> None:
        rp = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_pool_") as td:
            pool = Path(td) / "pool"
            rp.init_pool(pool)

            dims, profile, _index = rp.require_pool(pool)

            raw_a = {
                "model": "M",
                "param": "tau",
                "value": "adaptive",
                "tau": "adaptive",
                "step04_mask_mode": "adaptive",
                "step04_mask_threshold": -1.0,
            }
            raw_b = {
                "model": "M",
                "param": "tau",
                "value": "adaptive",
                "tau": "adaptive",
                "step04_mask_mode": "adaptive",
                "step04_mask_threshold": 0.123,
            }

            _ra, ca, _ta, ha = rp.canonicalize_key(raw_a, dimensions_obj=dims, profile_obj=profile, context={})
            _rb, cb, _tb, hb = rp.canonicalize_key(raw_b, dimensions_obj=dims, profile_obj=profile, context={})

            self.assertEqual(ha, hb)
            self.assertIsNone(ca["dimensions"]["step04_mask_threshold"])
            self.assertIsNone(cb["dimensions"]["step04_mask_threshold"])

    def test_ingest_and_plan_ofat_filters_existing_case(self) -> None:
        rp = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_pool_plan_") as td:
            root = Path(td)
            pool = root / "pool"
            run_root = root / "ofat" / "M" / "sweep_s_max" / "s_max_2"
            run_root.mkdir(parents=True)

            dump_json(
                run_root / "case_config.json",
                {
                    "model": "M",
                    "param": "s_max",
                    "value": 2,
                    "case_id": "s_max_2",
                    "effective_params": {
                        "alpha": 0.5,
                        "beta": 0.3,
                        "gamma": 0.2,
                        "s_max": 2,
                        "tau": "adaptive",
                        "omega_b": 1000.0,
                    },
                    "dataset": {
                        "model": "M",
                        "ssdr_work_dir": "/tmp/M/ssdr_work_dir",
                    },
                },
            )
            dump_json(run_root / "metrics_04.json", {"04_L2_Err": 1.0})
            (run_root / "done").write_text("done\n", encoding="utf-8")
            (run_root / "data4").mkdir()
            dump_json(
                run_root / "data4" / "step04_optimize_summary.json",
                {
                    "config": {
                        "iter": 70,
                        "org_type": 3,
                        "opt_weight": True,
                        "input_poses": True,
                        "use_weight_mask": True,
                        "mask_mode": "adaptive",
                        "mask_threshold": -1.0,
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
                        "s_max": 2,
                    }
                },
            )

            rp.init_pool(pool)
            report = rp.ingest_ofat(pool=pool, input_root=root / "ofat")
            self.assertEqual(report["ingested"], 1)

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
                "ssdr_defaults": {},
            }
            cfg_path = root / "config.json"
            dump_json(cfg_path, cfg)

            plan_dir = root / "plan"
            manifest_path = plan_dir / "case_manifest.json"

            plan_report = rp.plan_ofat(
                pool=pool,
                config_path=cfg_path,
                output_manifest=manifest_path,
                output_plan_dir=plan_dir,
                output_root=str(root / "filtered_out"),
                only_models=None,
                only_params=None,
                no_conda=True,
            )

            self.assertEqual(plan_report["candidate_count"], 2)
            self.assertEqual(plan_report["hit_count"], 1)
            self.assertEqual(plan_report["miss_count"], 1)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["case_count"], 1)
            self.assertEqual(manifest["cases"][0]["value"], 4)
            self.assertIn("--no_conda", manifest["cases"][0]["command"])

    def test_add_dimension_updates_index(self) -> None:
        rp = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_pool_dim_") as td:
            pool = Path(td) / "pool"
            rp.init_pool(pool)

            # Empty pool should still accept dimension addition.
            report = rp.add_dimension(pool, name="new_dimension", default=0)
            self.assertEqual(report["updated_cases"], 0)

            dims, _profile, _index = rp.require_pool(pool)
            self.assertIn("new_dimension", dims["dimensions"])
            self.assertEqual(dims["defaults"]["new_dimension"], 0)


if __name__ == "__main__":
    unittest.main()
