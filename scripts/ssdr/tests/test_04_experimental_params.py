#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
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
    for item in [REPO_ROOT / "src", REPO_ROOT / "scripts" / "ssdr"]:
        item_s = str(item)
        if item_s not in sys.path:
            sys.path.insert(0, item_s)


def make_env() -> dict[str, str]:
    env = dict(os.environ)
    paths = [
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "scripts" / "ssdr"),
    ]
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(paths + ([old] if old else []))
    return env


def load_step04_module():
    ensure_repo_paths()
    path = REPO_ROOT / "scripts" / "ssdr" / "04_optimize_weight_and_bone_trans.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_ssdr_step04_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestStep04ExperimentalParams(unittest.TestCase):
    def test_help_exposes_experiment_params(self) -> None:
        script = REPO_ROOT / "scripts" / "ssdr" / "04_optimize_weight_and_bone_trans.py"

        res = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(REPO_ROOT),
            env=make_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertEqual(res.returncode, 0, msg=res.stdout)
        self.assertIn("--s_max", res.stdout)
        self.assertIn("--mask_mode", res.stdout)
        self.assertIn("--mask_threshold", res.stdout)
        self.assertIn("--tau", res.stdout)
        self.assertIn("--freeze_weights", res.stdout)

    def test_optimize_function_signature_contains_new_params(self) -> None:
        mod = load_step04_module()
        sig = inspect.signature(mod.optimize_weight_and_bone_trans_tree)

        for name in ["s_max", "mask_mode", "mask_threshold", "freeze_weights"]:
            self.assertIn(name, sig.parameters)

    def test_org_type_3_can_force_weight_updates_without_freeze(self) -> None:
        ensure_repo_paths()
        from ogd_lbs.ssdr.schedulers import get_legacy_schedule

        # This documents why StepA added --freeze_weights.
        # For org_type=3, even when base_opt_weight=False, later iterations
        # turn to_opt_weight=True.
        to_opt_early, _mode_early, _jnt_early = get_legacy_schedule(
            it=0,
            max_iter=70,
            org_type=3,
            base_opt_weight=False,
        )
        to_opt_late, _mode_late, _jnt_late = get_legacy_schedule(
            it=35,
            max_iter=70,
            org_type=3,
            base_opt_weight=False,
        )

        self.assertFalse(to_opt_early)
        self.assertTrue(to_opt_late)

    def test_freeze_weights_override_logic_present(self) -> None:
        script = REPO_ROOT / "scripts" / "ssdr" / "04_optimize_weight_and_bone_trans.py"
        text = script.read_text(encoding="utf-8")

        self.assertIn("if bool(freeze_weights):", text)
        self.assertIn("to_opt_weight = False", text)
        self.assertIn("+freeze_weights", text)


if __name__ == "__main__":
    unittest.main()
