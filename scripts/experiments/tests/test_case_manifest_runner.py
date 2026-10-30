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
    path = REPO_ROOT / "scripts" / "experiments" / "run_case_manifest.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_run_case_manifest_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestCaseManifestRunner(unittest.TestCase):
    def test_shard_dry_run(self) -> None:
        mod = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_case_manifest_") as td:
            root = Path(td)
            manifest = {
                "schema": "ogd_lbs_case_manifest_v1",
                "mode": "ofat",
                "cases": [
                    {"model": "M", "param": "s_max", "value": 1, "case_hash": "h1", "command": ["python", "-c", "print(1)"]},
                    {"model": "M", "param": "s_max", "value": 2, "case_hash": "h2", "command": ["python", "-c", "print(2)"]},
                    {"model": "M", "param": "s_max", "value": 3, "case_hash": "h3", "command": ["python", "-c", "print(3)"]},
                ],
            }
            p = root / "manifest.json"
            p.write_text(json.dumps(manifest), encoding="utf-8")

            old_argv = sys.argv[:]
            try:
                sys.argv = [
                    "run_case_manifest.py",
                    "--manifest",
                    str(p),
                    "--worker_id",
                    "1",
                    "--num_workers",
                    "2",
                    "--dry_run",
                    "--log_dir",
                    str(root / "logs"),
                ]
                mod.main()
            finally:
                sys.argv = old_argv

            report_path = root / "logs" / "run_report_worker_1.json"
            self.assertTrue(report_path.is_file())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["total_cases"], 3)
            self.assertEqual(report["assigned_cases"], 1)
            self.assertEqual(report["failed_count"], 0)


if __name__ == "__main__":
    unittest.main()
