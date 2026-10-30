#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

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


def make_env() -> dict[str, str]:
    env = dict(os.environ)
    paths = [
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "scripts" / "ssdr"),
    ]
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(paths + ([old] if old else []))
    return env


class TestStep02SymbolAliases(unittest.TestCase):
    def test_help_exposes_alpha_beta_gamma_aliases(self) -> None:
        script = REPO_ROOT / "scripts" / "ssdr" / "02_reduce_bone_weight.py"

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
        self.assertIn("--par_weight", res.stdout)
        self.assertIn("--alpha", res.stdout)
        self.assertIn("--bone_weight", res.stdout)
        self.assertIn("--beta", res.stdout)
        self.assertIn("--child_weight", res.stdout)
        self.assertIn("--gamma", res.stdout)


if __name__ == "__main__":
    unittest.main()
