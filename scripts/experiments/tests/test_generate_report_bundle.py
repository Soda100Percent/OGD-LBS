#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
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
    ]:
        item_s = str(item)
        if item_s not in sys.path:
            sys.path.insert(0, item_s)


ensure_repo_paths()

from experiment_io import dump_json  # noqa: E402


def load_module():
    path = REPO_ROOT / "scripts" / "experiments" / "generate_report_bundle.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_generate_report_bundle_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestGenerateReportBundle(unittest.TestCase):
    def test_generate_alpha_beta_bundle_with_zip(self) -> None:
        mod = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_report_bundle_") as td:
            root = Path(td)
            artifacts = root / "artifacts"

            for idx, param in enumerate(["alpha", "beta"], start=1):
                case_root = artifacts / f"a{idx:06d}"
                (case_root / "data4").mkdir(parents=True)

                dump_json(
                    case_root / "case_config.json",
                    {
                        "model": "M",
                        "param": param,
                        "value": 0.5 if param == "alpha" else 0.3,
                        "case_id": f"{param}_case",
                        "stage": "ssdr_step04",
                        "effective_params": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
                    },
                )
                dump_json(
                    case_root / "metrics_04.json",
                    {
                        "04_L2_Err": float(idx),
                        "04_Org_Loss": float(idx) / 100.0,
                    },
                )
                dump_json(
                    case_root / "data4" / "step04_optimize_summary.json",
                    {
                        "final_metrics": {"04_L2_Err": float(idx)},
                        "counts": {"frames": 3},
                        "config": {"s_max": 4},
                    },
                )

            spec_path = root / "bundle_spec.json"
            dump_json(
                spec_path,
                {
                    "schema": "ogd_lbs_report_bundle_spec_v1",
                    "title": "Alpha Beta Bundle",
                    "input_root": str(artifacts),
                    "mode": "sweep",
                    "group_by": ["param", "value"],
                    "reports": [
                        {
                            "id": "alpha_data4",
                            "title": "Alpha Report",
                            "where": "param=alpha",
                            "output_html": "Alpha.html",
                        },
                        {
                            "id": "beta_data4",
                            "title": "Beta Report",
                            "where": "param=beta",
                            "output_html": "Beta.html",
                        },
                    ],
                    "zip": {
                        "enabled": True,
                        "name": "bundle.zip",
                    },
                },
            )

            out = root / "bundle"
            manifest = mod.generate_report_bundle(
                spec_path=spec_path,
                output_dir=out,
                overwrite=False,
            )

            self.assertEqual(manifest["report_count"], 2)
            self.assertTrue((out / "index.html").is_file())
            self.assertTrue((out / "bundle_manifest.json").is_file())

            alpha = out / "reports" / "alpha_data4"
            beta = out / "reports" / "beta_data4"

            self.assertTrue((alpha / "Alpha.html").is_file())
            self.assertTrue((beta / "Beta.html").is_file())
            self.assertTrue((alpha / "report_manifest.json").is_file())
            self.assertTrue((beta / "report_manifest.json").is_file())

            alpha_rows = read_csv_rows(alpha / "aggregated_metrics.csv")
            beta_rows = read_csv_rows(beta / "aggregated_metrics.csv")

            self.assertEqual(len(alpha_rows), 1)
            self.assertEqual(len(beta_rows), 1)
            self.assertEqual(alpha_rows[0]["param"], "alpha")
            self.assertEqual(beta_rows[0]["param"], "beta")

            index_html = (out / "index.html").read_text(encoding="utf-8")
            self.assertIn("Alpha Report", index_html)
            self.assertIn("Beta Report", index_html)
            self.assertIn("Download offline report ZIP", index_html)

            zip_path = out / "downloads" / "bundle.zip"
            self.assertTrue(zip_path.is_file())

            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())

            self.assertIn("index.html", names)
            self.assertIn("bundle_manifest.json", names)
            self.assertIn("reports/alpha_data4/Alpha.html", names)
            self.assertIn("reports/beta_data4/Beta.html", names)
            self.assertIn("reports/alpha_data4/aggregated_metrics.csv", names)
            self.assertIn("reports/beta_data4/aggregated_metrics.csv", names)


if __name__ == "__main__":
    unittest.main()
