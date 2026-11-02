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
                        "04_Max_Roll": "10.0",
                        "04_Max_Swing": "20.0",
                        "history_l2_err_final_mean": "1.0",
                        "history_l2_err_final_min": "0.7",
                        "history_l2_err_final_max": "1.4",
                        "history_l2_err_final_std": "0.11",
                        "history_org_loss_final_mean": "0.01",
                        "history_org_loss_final_min": "0.0",
                        "history_org_loss_final_max": "0.03",
                        "history_org_loss_final_std": "0.001",
                        "history_bone_roll_max_final_mean": "10.0",
                        "history_bone_roll_max_final_min": "8.0",
                        "history_bone_roll_max_final_max": "1000000000000.0",
                        "history_bone_roll_max_final_std": "1.5",
                        "history_bone_swing_max_final_mean": "20.0",
                        "history_bone_swing_max_final_min": "15.0",
                        "history_bone_swing_max_final_max": "25.0",
                        "history_bone_swing_max_final_std": "2.5",
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
                        "04_Max_Roll": "11.0",
                        "04_Max_Swing": "21.0",
                        "history_l2_err_final_mean": "1.2",
                        "history_l2_err_final_min": "0.8",
                        "history_l2_err_final_max": "1.5",
                        "history_l2_err_final_std": "0.12",
                        "history_org_loss_final_mean": "0.02",
                        "history_org_loss_final_min": "0.0",
                        "history_org_loss_final_max": "0.04",
                        "history_org_loss_final_std": "0.002",
                        "history_bone_roll_max_final_mean": "11.0",
                        "history_bone_roll_max_final_min": "9.0",
                        "history_bone_roll_max_final_max": "14.0",
                        "history_bone_roll_max_final_std": "1.6",
                        "history_bone_swing_max_final_mean": "21.0",
                        "history_bone_swing_max_final_min": "16.0",
                        "history_bone_swing_max_final_max": "26.0",
                        "history_bone_swing_max_final_std": "2.6",
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

            # R2a: sweep mean±std and model-grouped raw metric chart payloads.
            self.assertNotIn("sweep-mean-std-container", html)
            self.assertIn("model-grouped-raw-container", html)
            self.assertNotIn("Sweep Mean ± Std Charts", html)
            self.assertIn("modelGroupedRaw", html)
            self.assertIn("defaultSweepValues", html)
            self.assertIn("★ default", html)

            # R2f: per-frame std error bars for model-grouped raw charts.
            self.assertIn("perFrameStd", html)
            self.assertIn("perFrameStdField", html)
            self.assertIn("hasPerFrameStd", html)
            self.assertIn("history_l2_err_final_std", html)
            self.assertIn("history_org_loss_final_std", html)
            self.assertIn("history_bone_roll_max_final_std", html)
            self.assertIn("history_bone_swing_max_final_std", html)
            self.assertIn("per-frame std", html)

            # R2g: chart whiskers show min-max fluctuation ranges;
            # strict std remains available in the table and tooltip.
            self.assertIn("Per-frame Fluctuation / Std Table", html)
            self.assertIn("per-frame min-max range", html)
            self.assertIn("frameMin", html)
            self.assertIn("frameMax", html)
            self.assertIn("frameStd", html)
            self.assertIn("drawLow", html)
            self.assertIn("drawHigh", html)
            self.assertIn("hasFrameFluctuation", html)
            self.assertIn("positiveMetric", html)
            self.assertIn("yAxisMax", html)
            self.assertIn("VD_frame_std", html)
            self.assertIn("SO_frame_std", html)
            self.assertIn("Roll_frame_std", html)
            self.assertIn("Swing_frame_std", html)
            # R2h: model-wise metric scaling metadata and chart annotations.
            self.assertIn("modelScaleValues", html)
            self.assertIn("scaledMetrics", html)
            self.assertIn("modelScale", html)
            self.assertIn("rawMetricValue", html)
            self.assertIn("metricScaled", html)
            self.assertIn("scaleAnnotation", html)
            self.assertIn("scaleAnnotationVisible", html)
            self.assertIn("chart-scale-note", html)
            self.assertIn("displayed = raw × modelScale", html)
            # R2i: pseudo-log nonlinear axis metadata and notes.
            self.assertIn("axisMode", html)
            self.assertIn("pseudo_log", html)
            self.assertIn("axisLinearThreshold", html)
            self.assertIn("axisTransform", html)
            self.assertIn("axisOriginalMax", html)
            self.assertIn("axisDynamicRatio", html)
            self.assertIn("pseudoLogRatioThreshold", html)
            self.assertIn("axisFormula", html)
            self.assertIn("chart-axis-note", html)
            self.assertIn("Y-axis uses pseudo-log scale", html)

            import json as _json
            import re as _re
            m = _re.search(r"window\.DASHBOARD_DATA = (.*?);\n", html, flags=_re.S)
            self.assertIsNotNone(m)
            data = _json.loads(m.group(1))
            charts = data["charts"]["modelGroupedRaw"]
            roll_chart = next(c for c in charts if c["metric"] == "04_Max_Roll")
            self.assertEqual(roll_chart["axisMode"], "pseudo_log")
            self.assertGreater(float(roll_chart["axisOriginalMax"]), 1e10)
            self.assertFalse(bool(roll_chart.get("axisClipped", False)))

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
