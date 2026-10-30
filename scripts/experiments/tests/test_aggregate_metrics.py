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
    path = REPO_ROOT / "scripts" / "experiments" / "aggregate_metrics.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_aggregate_metrics_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestAggregateMetrics(unittest.TestCase):
    def test_sweep_aggregation_and_paired_ttest(self) -> None:
        mod = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_agg_sweep_") as td:
            root = Path(td)
            out = root / "reports"

            for model, base_err, test_err in [
                ("A", 1.0, 0.8),
                ("B", 2.0, 1.7),
            ]:
                for case_id, value, err in [
                    ("s_max_4", 4, base_err),
                    ("s_max_2", 2, test_err),
                ]:
                    case_root = root / model / "sweep_s_max" / case_id
                    (case_root / "data4").mkdir(parents=True)

                    dump_json(
                        case_root / "case_config.json",
                        {
                            "model": model,
                            "param": "s_max",
                            "value": value,
                            "case_id": case_id,
                            "stage": "ssdr_step04",
                            "effective_params": {"s_max": value, "tau": "adaptive"},
                        },
                    )
                    dump_json(
                        case_root / "metrics_04.json",
                        {
                            "04_L2_Err": err,
                            "04_Hausdorff_Max": err * 10.0,
                            "04_S_Max": value,
                        },
                    )
                    dump_json(
                        case_root / "data4" / "step04_optimize_summary.json",
                        {
                            "final_metrics": {
                                "04_L2_Err": err,
                                "04_Hausdorff_Max": err * 10.0,
                            },
                            "counts": {"frames": 10},
                            "config": {"s_max": value},
                        },
                    )

            manifest = mod.aggregate(
                input_root=root,
                output_dir=out,
                mode="sweep",
                metrics_requested=["04_L2_Err", "04_Hausdorff_Max"],
                group_by=["param", "value"],
                ttest_kind="paired",
                baseline_query_text="param=s_max,value=4",
                pair_by=["model"],
                ttest_group_by=["param", "value"],
            )

            self.assertEqual(manifest["row_count"], 4)
            self.assertTrue((out / "aggregated_metrics.csv").is_file())
            self.assertTrue((out / "summary_by_group.csv").is_file())
            self.assertTrue((out / "summary_table.md").is_file())
            self.assertTrue((out / "summary_table.tex").is_file())
            self.assertTrue((out / "ttest_results.csv").is_file())

            summary_rows = read_csv_rows(out / "summary_by_group.csv")
            self.assertEqual(len(summary_rows), 2)

            ttest_rows = read_csv_rows(out / "ttest_results.csv")
            self.assertTrue(any(r["metric"] == "04_L2_Err" for r in ttest_rows))

    def test_cv_aggregation(self) -> None:
        mod = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_agg_cv_") as td:
            root = Path(td)
            out = root / "reports"

            for fold_idx in [0, 1]:
                fold_root = root / "ModelA" / f"fold_{fold_idx:02d}"
                dump_json(
                    fold_root / "fold_plan.json",
                    {
                        "model": "ModelA",
                        "fold_index": fold_idx,
                        "split": {
                            "train_count": 75,
                            "test_count": 25,
                            "special_count": 0,
                        },
                    },
                )

                for split, err in [("train", 1.0 + fold_idx), ("test", 2.0 + fold_idx)]:
                    run_root = fold_root / split
                    out_dir = run_root / ("data4_test" if split == "test" else "data4")
                    out_dir.mkdir(parents=True)

                    dump_json(
                        run_root / "metrics_04.json",
                        {
                            "04_L2_Err": err,
                            "04_Hausdorff_Max": err * 5.0,
                        },
                    )
                    dump_json(
                        out_dir / "step04_optimize_summary.json",
                        {
                            "final_metrics": {"04_L2_Err": err},
                            "counts": {"frames": 25 if split == "test" else 75},
                            "config": {"freeze_weights": split == "test"},
                        },
                    )

            manifest = mod.aggregate(
                input_root=root,
                output_dir=out,
                mode="cv",
                metrics_requested=["04_L2_Err", "04_Hausdorff_Max"],
                group_by=["split"],
                ttest_kind="paired",
                baseline_query_text="split=train",
                pair_by=["model", "fold_index"],
                ttest_group_by=["split"],
            )

            self.assertEqual(manifest["row_count"], 4)
            self.assertTrue((out / "aggregated_metrics.json").is_file())
            self.assertTrue((out / "ttest_results.md").is_file())

            summary_rows = read_csv_rows(out / "summary_by_group.csv")
            splits = {row["split"] for row in summary_rows}
            self.assertEqual(splits, {"train", "test"})

    def test_generic_aggregation(self) -> None:
        mod = load_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_agg_generic_") as td:
            root = Path(td)
            out = root / "reports"
            run = root / "some" / "run"
            run.mkdir(parents=True)

            dump_json(run / "metrics_04.json", {"04_L2_Err": 1.23})

            manifest = mod.aggregate(
                input_root=root,
                output_dir=out,
                mode="generic",
                metrics_requested=["04_L2_Err"],
                group_by=[],
                ttest_kind="none",
                baseline_query_text="",
                pair_by=None,
                ttest_group_by=None,
            )

            self.assertEqual(manifest["row_count"], 1)
            rows = read_csv_rows(out / "aggregated_metrics.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["04_L2_Err"], "1.23")


if __name__ == "__main__":
    unittest.main()
