#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
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
    ]:
        item_s = str(item)
        if item_s not in sys.path:
            sys.path.insert(0, item_s)


ensure_repo_paths()

from experiment_io import dump_json  # noqa: E402


def load_module():
    path = REPO_ROOT / "scripts" / "experiments" / "generate_report.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_generate_report_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestGenerateReport(unittest.TestCase):
    def test_generate_sweep_report(self) -> None:
        mod = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_report_sweep_") as td:
            root = Path(td)
            aggregate_dir = root / "reports"
            aggregate_dir.mkdir()

            dump_json(
                aggregate_dir / "aggregate_manifest.json",
                {
                    "schema": "ogd_lbs_aggregate_metrics_manifest_v1",
                    "mode": "sweep",
                    "row_count": 2,
                },
            )

            write_csv(
                aggregate_dir / "aggregated_metrics.csv",
                [
                    {
                        "mode": "sweep",
                        "model": "Vroid_18",
                        "param": "s_max",
                        "value": "2",
                        "case_id": "s_max_2",
                        "stage": "ssdr_step04",
                        "eff_s_max": "2",
                        "04_L2_Err": "1.2",
                        "04_Hausdorff_Mean": "2.3",
                        "04_Org_Loss": "0.01",
                    },
                    {
                        "mode": "sweep",
                        "model": "Vroid_18",
                        "param": "s_max",
                        "value": "4",
                        "case_id": "s_max_4",
                        "stage": "ssdr_step04",
                        "eff_s_max": "4",
                        "04_L2_Err": "1.0",
                        "04_Hausdorff_Mean": "2.0",
                        "04_Org_Loss": "0.02",
                    },
                ],
            )

            write_csv(
                aggregate_dir / "summary_by_group.csv",
                [
                    {
                        "param": "s_max",
                        "value": "2",
                        "n_runs": "1",
                        "04_L2_Err_mean": "1.2",
                        "04_L2_Err_std": "0.0",
                        "04_L2_Err_n": "1",
                    }
                ],
            )

            out = aggregate_dir / "OGD_LBS_Report.html"

            result = mod.generate_report(
                aggregate_dir=aggregate_dir,
                output_html=out,
                mode="sweep",
                title="Smoke Report",
            )

            self.assertEqual(result, out.resolve(strict=False))
            self.assertTrue(out.is_file())

            html = out.read_text(encoding="utf-8")
            self.assertIn("Smoke Report", html)
            self.assertIn("window.DASHBOARD_DATA", html)
            self.assertIn("s_max_2", html)
            self.assertIn("04_L2_Err", html)
            self.assertIn("chart-pareto", html)
            self.assertIn("exportChart", html)

    def test_generate_cv_report_with_ttest(self) -> None:
        mod = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_report_cv_") as td:
            root = Path(td)
            aggregate_dir = root / "reports"
            aggregate_dir.mkdir()

            dump_json(
                aggregate_dir / "aggregate_manifest.json",
                {"mode": "cv", "row_count": 2},
            )

            write_csv(
                aggregate_dir / "aggregated_metrics.csv",
                [
                    {
                        "mode": "cv",
                        "model": "Vroid_18",
                        "fold_index": "0",
                        "split": "train",
                        "split_train_count": "150",
                        "split_test_count": "50",
                        "summary_config_freeze_weights": "False",
                        "04_L2_Err": "0.8",
                        "04_Hausdorff_Mean": "1.8",
                    },
                    {
                        "mode": "cv",
                        "model": "Vroid_18",
                        "fold_index": "0",
                        "split": "test",
                        "split_train_count": "150",
                        "split_test_count": "50",
                        "summary_config_freeze_weights": "True",
                        "04_L2_Err": "1.1",
                        "04_Hausdorff_Mean": "2.1",
                    },
                ],
            )

            write_csv(
                aggregate_dir / "summary_by_group.csv",
                [
                    {
                        "split": "train",
                        "n_runs": "1",
                        "04_L2_Err_mean": "0.8",
                        "04_L2_Err_std": "0.0",
                        "04_L2_Err_n": "1",
                    },
                    {
                        "split": "test",
                        "n_runs": "1",
                        "04_L2_Err_mean": "1.1",
                        "04_L2_Err_std": "0.0",
                        "04_L2_Err_n": "1",
                    },
                ],
            )

            write_csv(
                aggregate_dir / "ttest_results.csv",
                [
                    {
                        "test": "paired",
                        "split": "test",
                        "metric": "04_L2_Err",
                        "p_value": "0.05",
                    }
                ],
            )

            out = aggregate_dir / "CV_Report.html"

            mod.generate_report(
                aggregate_dir=aggregate_dir,
                output_html=out,
                mode="cv",
                title="CV Report",
            )

            html = out.read_text(encoding="utf-8")
            self.assertIn("CV Report", html)
            self.assertIn("fold 0 | train", html)
            self.assertIn("fold 0 | test", html)
            self.assertIn("t-test", html.lower())
            self.assertIn("freeze_weights", html)

    def test_generate_generic_report_without_optional_files(self) -> None:
        mod = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_report_generic_") as td:
            root = Path(td)
            aggregate_dir = root / "reports"
            aggregate_dir.mkdir()

            write_csv(
                aggregate_dir / "aggregated_metrics.csv",
                [
                    {
                        "mode": "generic",
                        "run_name": "runA",
                        "04_L2_Err": "1.23",
                    }
                ],
            )

            out = aggregate_dir / "Generic_Report.html"
            mod.generate_report(
                aggregate_dir=aggregate_dir,
                output_html=out,
                mode="generic",
                title="Generic Report",
            )

            html = out.read_text(encoding="utf-8")
            self.assertIn("Generic Report", html)
            self.assertIn("runA", html)
            self.assertIn("No summary_by_group.csv found", html)
            self.assertIn("No t-test results found", html)


if __name__ == "__main__":
    unittest.main()
