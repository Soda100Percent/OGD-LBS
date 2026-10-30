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


class TestResultPoolArtifactsData1(unittest.TestCase):
    def test_ensure_profiles_contains_data1(self) -> None:
        rp = load_module(
            "ogd_lbs_result_pool_for_data1_profile_tests",
            "scripts/experiments/result_pool.py",
        )
        art = load_module(
            "ogd_lbs_result_pool_artifacts_for_data1_profile_tests",
            "scripts/experiments/result_pool_artifacts.py",
        )

        with tempfile.TemporaryDirectory(prefix="ogd_data1_profile_") as td:
            pool = Path(td) / "pool"
            rp.init_pool(pool)
            art.ensure_artifact_profiles(pool)

            dims, _eq, _idx = rp.require_pool(pool)
            self.assertIn("ssdr_data1", dims["profiles"])
            self.assertIn("ssdr_data2", dims["profiles"])
            self.assertIn("ssdr_data3", dims["profiles"])

            self.assertIn("input_source_kind", dims["profiles"]["ssdr_data1"]["dimensions"])
            self.assertIn("frame_sampling_config_name", dims["profiles"]["ssdr_data1"]["dimensions"])

    def test_ingest_and_lookup_ssdr_data1(self) -> None:
        rp = load_module(
            "ogd_lbs_result_pool_for_data1_ingest_tests",
            "scripts/experiments/result_pool.py",
        )
        art = load_module(
            "ogd_lbs_result_pool_artifacts_for_data1_ingest_tests",
            "scripts/experiments/result_pool_artifacts.py",
        )

        with tempfile.TemporaryDirectory(prefix="ogd_data1_ingest_") as td:
            root = Path(td)
            pool = root / "pool"
            input_root = root / "cache_round1_data1"
            run_root = input_root / "Vroid_18" / "ssdr_data1"

            (run_root / "data1").mkdir(parents=True)
            (run_root / "data1" / "dummy_data1.txt").write_text("data1\n", encoding="utf-8")

            dump_json(
                run_root / "case_config.json",
                {
                    "schema": "ogd_lbs_data1_cache_case_v1",
                    "artifact_type": "ssdr_data1",
                    "model": "Vroid_18",
                    "input_source_kind": "vroid_data0",
                    "input_dataset_version": "v1",
                    "frame_sampling_policy": "target_200",
                    "max_frames": 210,
                    "target_frames": 200,
                    "frame_sampling_config_name": "frame_sampling_config_vroid.json",
                    "frame_removal_idx_ranges": "20:60,260:340",
                    "step00_code_version": "step00_v1",
                    "step01_code_version": "step01_v1",
                },
            )
            (run_root / "done").write_text("done\n", encoding="utf-8")

            rp.init_pool(pool)
            art.ensure_artifact_profiles(pool)

            report = art.ingest_artifacts(
                pool=pool,
                input_root=input_root,
                stages=["ssdr_data1"],
            )

            self.assertEqual(report["ingested"], 1)
            self.assertEqual(report["skipped_missing_payload"], 0)

            stats = rp.stats(pool)
            self.assertEqual(stats["artifact_count"], 1)
            self.assertEqual(stats["by_artifact_type"]["ssdr_data1"], 1)

            key = root / "data1_key.json"
            dump_json(
                key,
                {
                    "dimensions": {
                        "artifact_type": "ssdr_data1",
                        "model": "Vroid_18",
                        "input_source_kind": "vroid_data0",
                        "frame_sampling_config_name": "frame_sampling_config_vroid.json",
                        "frame_removal_idx_ranges": "20:60,260:340",
                    }
                },
            )

            lookup = art.lookup_artifact(pool=pool, stage="ssdr_data1", key_json=key)
            self.assertEqual(lookup["hit_count"], 1)

            hit_dir = Path(lookup["hits"][0]["artifact_dir"])
            self.assertTrue((hit_dir / "payload" / "data1" / "dummy_data1.txt").is_file())

    def test_data1_dimension_upgrade_uses_defaults_without_rewrite(self) -> None:
        rp = load_module(
            "ogd_lbs_result_pool_for_data1_dim_tests",
            "scripts/experiments/result_pool.py",
        )
        art = load_module(
            "ogd_lbs_result_pool_artifacts_for_data1_dim_tests",
            "scripts/experiments/result_pool_artifacts.py",
        )

        with tempfile.TemporaryDirectory(prefix="ogd_data1_dim_") as td:
            root = Path(td)
            pool = root / "pool"
            rp.init_pool(pool)
            art.ensure_artifact_profiles(pool)

            index_path = pool / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))

            artifact_id = "a000001"
            adir = pool / "artifacts" / artifact_id
            adir.mkdir(parents=True)

            old_dims = {
                "artifact_type": "ssdr_data1",
                "model": "M",
                "input_source_kind": "fitbox_vdb",
                "frame_sampling_policy": "target_200",
            }
            dump_json(adir / "raw_dimensions.json", {"dimensions": old_dims})
            dump_json(
                adir / "artifact_manifest.json",
                {
                    "artifact_id": artifact_id,
                    "artifact_type": "ssdr_data1",
                    "status": "succeeded",
                },
            )

            index["artifacts"][artifact_id] = {
                "artifact_id": artifact_id,
                "artifact_type": "ssdr_data1",
                "status": "succeeded",
                "artifact_dir": f"artifacts/{artifact_id}",
                "dimension_fingerprint": "old",
                "model": "M",
            }
            index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

            rp.add_dimension(
                pool,
                profile_name="ssdr_data1",
                name="data1_new_knob",
                default=0,
            )

            key_default = root / "key_default.json"
            dump_json(
                key_default,
                {
                    "dimensions": {
                        "artifact_type": "ssdr_data1",
                        "model": "M",
                        "input_source_kind": "fitbox_vdb",
                        "data1_new_knob": 0,
                    }
                },
            )
            lookup_default = art.lookup_artifact(pool=pool, stage="ssdr_data1", key_json=key_default)
            self.assertEqual(lookup_default["hit_count"], 1)

            key_non_default = root / "key_non_default.json"
            dump_json(
                key_non_default,
                {
                    "dimensions": {
                        "artifact_type": "ssdr_data1",
                        "model": "M",
                        "input_source_kind": "fitbox_vdb",
                        "data1_new_knob": 1,
                    }
                },
            )
            lookup_non_default = art.lookup_artifact(pool=pool, stage="ssdr_data1", key_json=key_non_default)
            self.assertEqual(lookup_non_default["hit_count"], 0)


if __name__ == "__main__":
    unittest.main()
