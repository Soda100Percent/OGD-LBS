#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def find_repo_root() -> Path:
    this_file = Path(__file__).resolve(strict=False)
    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Could not locate repository root.")


REPO_ROOT = find_repo_root()


def ensure_repo_paths() -> None:
    for item in [
        REPO_ROOT / "src",
        REPO_ROOT / "scripts" / "experiments",
        REPO_ROOT / "scripts" / "ssdr",
        REPO_ROOT / "scripts" / "data_prep",
    ]:
        item_s = str(item)
        if item_s not in sys.path:
            sys.path.insert(0, item_s)


def load_module():
    ensure_repo_paths()
    path = REPO_ROOT / "scripts" / "experiments" / "run_param_sweep.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_run_param_sweep_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestParamSweepPlanning(unittest.TestCase):
    def test_alpha_coupling_keeps_sum_one(self) -> None:
        mod = load_module()
        eff = mod.effective_params(
            "alpha",
            0.7,
            {
                "defaults": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
                "abc_coupling": {"mode": "keep_sum_one"},
            },
        )

        self.assertAlmostEqual(eff["alpha"], 0.7)
        self.assertAlmostEqual(eff["alpha"] + eff["beta"] + eff["gamma"], 1.0)
        self.assertAlmostEqual(eff["beta"] / eff["gamma"], 0.3 / 0.2)

    def test_iter_cases_filters_delta_by_model(self) -> None:
        mod = load_module()

        cfg = {
            "datasets": [
                {"model": "A", "ssdr_work_dir": "/tmp/A"},
                {"model": "B", "ssdr_work_dir": "/tmp/B"},
            ],
            "defaults": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
            "abc_coupling": {"mode": "keep_sum_one"},
            "sweeps": {"s_max": [2, 4]},
            "delta_sweeps_by_model": {"B": [1.0, 2.0]},
        }

        cases = mod.iter_cases(
            cfg,
            datasets=cfg["datasets"],
            params=["s_max", "delta"],
        )

        self.assertEqual(len(cases), 6)
        delta_cases = [c for c in cases if c["param"] == "delta"]
        self.assertEqual(len(delta_cases), 2)
        self.assertTrue(all(c["model"] == "B" for c in delta_cases))

    def test_build_step04_tau_fixed_command(self) -> None:
        mod = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ofat_cmd_") as td:
            root = Path(td)
            cfg = {
                "conda_env": "dummy",
                "step04_defaults": {
                    "iter": 70,
                    "print_every_n": 5,
                    "org_type": 3,
                    "weight_solver": "simplex",
                    "bt_covariance_mode": "corrected",
                    "trans_affine": 0.0,
                    "trans_affine_norm": 4.0,
                    "roll_affinity": 0.0,
                    "roll_affinity_norm": 2.0,
                    "health_level": 1,
                    "health_every": 5,
                    "health_weight_scheme": "uniform",
                    "bone_health_every": 5,
                    "opt_weight": True,
                    "input_poses": True,
                    "use_weight_mask": True,
                    "validate_output": True,
                },
                "defaults": {"s_max": 4},
                "ssdr_defaults": {},
            }

            cmd = mod.build_step04_cmd(
                cfg,
                input_data3=root / "data3",
                output_data4=root / "data4",
                metrics_dir=root,
                eff={"tau": 0.01, "s_max": 6},
                no_conda=True,
            )

            joined = " ".join(cmd)
            self.assertIn("--mask_mode fixed", joined)
            self.assertIn("--tau 0.01", joined)
            self.assertIn("--s_max 6", joined)
            self.assertIn("--opt_weight", cmd)
            self.assertIn("--input_poses", cmd)
            self.assertIn("--use_weight_mask", cmd)

    def test_build_step04_tau_adaptive_command(self) -> None:
        mod = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ofat_cmd_") as td:
            root = Path(td)
            cfg = {
                "conda_env": "dummy",
                "step04_defaults": {},
                "defaults": {"s_max": 4},
                "ssdr_defaults": {},
            }

            cmd = mod.build_step04_cmd(
                cfg,
                input_data3=root / "data3",
                output_data4=root / "data4",
                metrics_dir=root,
                eff={"tau": "adaptive", "s_max": 4},
                no_conda=True,
            )

            joined = " ".join(cmd)
            self.assertIn("--mask_mode adaptive", joined)
            self.assertNotIn("--tau", cmd)


if __name__ == "__main__":
    unittest.main()
