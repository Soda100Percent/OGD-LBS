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


class TestResultPoolSubsetMaterialization(unittest.TestCase):
    def test_materialize_subset_rewrites_manifest_and_case_records(self) -> None:
        rp = load_module(
            "ogd_lbs_result_pool_for_subset_tests",
            "scripts/experiments/result_pool.py",
        )
        art = load_module(
            "ogd_lbs_result_pool_artifacts_for_subset_tests",
            "scripts/experiments/result_pool_artifacts.py",
        )
        subset = load_module(
            "ogd_lbs_result_pool_subset_for_tests",
            "scripts/experiments/materialize_result_pool_subset.py",
        )

        with tempfile.TemporaryDirectory(prefix="ogd_subset_") as td:
            root = Path(td)
            source_pool = root / "source_pool"
            rp.init_pool(source_pool)
            art.ensure_artifact_profiles(source_pool)

            idx_path = source_pool / "index.json"
            index = json.loads(idx_path.read_text(encoding="utf-8"))

            artifact_id = "a000001"
            adir = source_pool / "artifacts" / artifact_id
            (adir / "payload" / "data3").mkdir(parents=True)
            (adir / "payload" / "data3" / "dummy_data3.txt").write_text("data3\n", encoding="utf-8")

            raw_data3 = {
                "artifact_type": "ssdr_data3",
                "model": "M",
                "input_source_kind": "fitbox_vdb",
                "input_dataset_version": "v1",
                "frame_sampling_policy": "target_200",
                "max_frames": 210,
                "target_frames": 200,
                "frame_sampling_config_name": "default",
                "frame_removal_idx_ranges": "",
                "step00_code_version": "step00_v1",
                "step01_code_version": "step01_v1",
                "bone_variant": "default",
                "alpha": 0.5,
                "beta": 0.3,
                "gamma": 0.2,
                "step02_affinity_eps": 1e-5,
                "step02_code_version": "step02_v1",
                "omega_b": 1000.0,
                "qem_config_name": "default",
                "qem_config_fingerprint": None,
                "step03_code_version": "step03_v1",
            }

            dump_json(adir / "raw_dimensions.json", {"dimensions": raw_data3})
            dump_json(
                adir / "artifact_manifest.json",
                {
                    "artifact_id": artifact_id,
                    "artifact_type": "ssdr_data3",
                    "status": "succeeded",
                    "payload_path": "payload/data3",
                },
            )
            (adir / "done").write_text("done\n", encoding="utf-8")

            index["artifacts"][artifact_id] = {
                "artifact_id": artifact_id,
                "artifact_type": "ssdr_data3",
                "status": "succeeded",
                "artifact_dir": f"artifacts/{artifact_id}",
                "model": "M",
                "param": "alpha",
                "value": 0.5,
                "payload_path": "payload/data3",
            }
            idx_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

            controller_plan = root / "controller_plan"
            case_records = controller_plan / "case_records"
            single_cfgs = controller_plan / "single_case_configs"
            case_records.mkdir(parents=True)
            single_cfgs.mkdir(parents=True)

            single_cfg_path = single_cfgs / "case_00000_M_alpha_0p5.json"
            dump_json(
                single_cfg_path,
                {
                    "schema": "ogd_lbs_ofat_sweep_config_v1",
                    "output_root": "/workspace/ogd_experiments/subset_test_runs",
                    "datasets": [{"model": "M", "ssdr_work_dir": "/tmp/M/ssdr_work_dir"}],
                    "sweeps": {"alpha": [0.5]},
                    "delta_sweeps_by_model": {},
                    "defaults": {
                        "alpha": 0.5,
                        "beta": 0.3,
                        "gamma": 0.2,
                        "s_max": 4,
                        "tau": "adaptive",
                        "omega_b": 1000.0,
                    },
                    "abc_coupling": {"mode": "keep_sum_one"},
                    "step04_defaults": {},
                    "ssdr_defaults": {"max_frames": 210, "target_frames": 200},
                },
            )

            case_record_path = case_records / "case_00000_M_alpha_0p5.json"
            source_hit = {
                "artifact_id": artifact_id,
                "artifact_dir": str(adir),
                "record": index["artifacts"][artifact_id],
            }

            case_record = {
                "source_case_index": 0,
                "model": "M",
                "param": "alpha",
                "value": 0.5,
                "case_id": "alpha_0p5",
                "target_stage": "ssdr_data4",
                "single_case_config": str(single_cfg_path),
                "case_record_json": str(case_record_path),
                "case_config_payload": {
                    "model": "M",
                    "param": "alpha",
                    "value": 0.5,
                    "case_id": "alpha_0p5",
                    "effective_params": {
                        "alpha": 0.5,
                        "beta": 0.3,
                        "gamma": 0.2,
                        "s_max": 4,
                        "tau": "adaptive",
                        "omega_b": 1000.0,
                    },
                    "dataset": {"model": "M", "ssdr_work_dir": "/tmp/M/ssdr_work_dir"},
                },
                "cache_hits": {
                    "ssdr_data3": source_hit,
                    "ssdr_data2": None,
                    "ssdr_data1": None,
                },
                "command": [
                    sys.executable,
                    "scripts/experiments/run_cached_ofat_case.py",
                    "--case_json",
                    str(case_record_path),
                    "--target_stage",
                    "ssdr_data4",
                    "--no_conda",
                ],
            }
            dump_json(case_record_path, case_record)

            manifest_path = controller_plan / "case_manifest.json"
            manifest = {
                "schema": "ogd_lbs_case_manifest_v2",
                "mode": "ofat_artifact_cache",
                "target_stage": "ssdr_data4",
                "case_count": 1,
                "cases": [case_record],
            }
            dump_json(manifest_path, manifest)

            output_pool = root / "subset_pool"
            output_manifest = root / "input" / "case_manifest.json"
            report_path = root / "input" / "plan" / "subset_report.json"

            report = subset.materialize_subset(
                source_pool=source_pool,
                case_manifest=manifest_path,
                output_pool=output_pool,
                runtime_pool="/workspace/ogd_run_pool",
                runtime_plan_dir="/oss/input/plan",
                output_manifest=output_manifest,
                report_path=report_path,
                overwrite=False,
            )

            self.assertEqual(report["artifact_count"], 1)
            self.assertEqual(report["by_artifact_type"], {"ssdr_data3": 1})
            self.assertTrue((output_pool / "index.json").is_file())
            self.assertTrue((output_pool / "artifacts" / artifact_id / "payload" / "data3" / "dummy_data3.txt").is_file())

            subset_index = json.loads((output_pool / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(subset_index["artifacts"].keys()), [artifact_id])

            rewritten_manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
            self.assertEqual(rewritten_manifest["case_count"], 1)

            rewritten_case = rewritten_manifest["cases"][0]
            self.assertEqual(
                rewritten_case["case_record_json"],
                "/oss/input/plan/case_records/case_00000_M_alpha_0p5.json",
            )
            self.assertEqual(
                rewritten_case["single_case_config"],
                "/oss/input/plan/single_case_configs/case_00000_M_alpha_0p5.json",
            )
            self.assertEqual(
                rewritten_case["cache_hits"]["ssdr_data3"]["artifact_dir"],
                f"/workspace/ogd_run_pool/artifacts/{artifact_id}",
            )
            self.assertEqual(
                rewritten_case["command"][
                    rewritten_case["command"].index("--case_json") + 1
                ],
                "/oss/input/plan/case_records/case_00000_M_alpha_0p5.json",
            )

            local_rewritten_record = root / "input" / "plan" / "case_records" / "case_00000_M_alpha_0p5.json"
            self.assertTrue(local_rewritten_record.is_file())

            rec = json.loads(local_rewritten_record.read_text(encoding="utf-8"))
            self.assertEqual(
                rec["cache_hits"]["ssdr_data3"]["artifact_dir"],
                f"/workspace/ogd_run_pool/artifacts/{artifact_id}",
            )
            self.assertEqual(
                rec["command"][rec["command"].index("--case_json") + 1],
                "/oss/input/plan/case_records/case_00000_M_alpha_0p5.json",
            )

            local_single_cfg = root / "input" / "plan" / "single_case_configs" / "case_00000_M_alpha_0p5.json"
            self.assertTrue(local_single_cfg.is_file())

            hits = rp.lookup_artifacts(
                pool=output_pool,
                raw_dims=raw_data3,
                profile_name="ssdr_data3",
            )
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["artifact_id"], artifact_id)


    def test_default_cache_stage_selection_uses_direct_upstream(self) -> None:
        subset = load_module(
            "ogd_lbs_result_pool_subset_stage_selection_tests",
            "scripts/experiments/materialize_result_pool_subset.py",
        )

        self.assertEqual(
            subset.default_cache_stages_for_manifest({"target_stage": "ssdr_data2"}),
            ["ssdr_data1"],
        )
        self.assertEqual(
            subset.default_cache_stages_for_manifest({"target_stage": "ssdr_data3"}),
            ["ssdr_data2"],
        )
        self.assertEqual(
            subset.default_cache_stages_for_manifest({"target_stage": "ssdr_data4"}),
            ["ssdr_data3"],
        )

        manifest = {
            "target_stage": "ssdr_data4",
            "cases": [
                {
                    "cache_hits": {
                        "ssdr_data1": {"artifact_id": "a1"},
                        "ssdr_data2": {"artifact_id": "a2"},
                        "ssdr_data3": {"artifact_id": "a3"},
                    }
                }
            ],
        }

        self.assertEqual(subset.collect_artifact_ids(manifest), ["a3"])
        self.assertEqual(
            subset.collect_artifact_ids(manifest, include_all_cache_hits=True),
            ["a1", "a2", "a3"],
        )
        self.assertEqual(
            subset.collect_artifact_ids(manifest, cache_stages=["ssdr_data2"]),
            ["a2"],
        )


if __name__ == "__main__":
    unittest.main()
