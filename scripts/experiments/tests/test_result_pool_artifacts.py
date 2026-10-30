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


class TestResultPoolArtifacts(unittest.TestCase):
    def test_ensure_profiles_and_ingest_data2_data3(self) -> None:
        rp = load_module("ogd_lbs_result_pool_for_artifact_tests", "scripts/experiments/result_pool.py")
        art = load_module("ogd_lbs_result_pool_artifacts_for_tests", "scripts/experiments/result_pool_artifacts.py")

        with tempfile.TemporaryDirectory(prefix="ogd_artifact_pool_") as td:
            root = Path(td)
            pool = root / "pool"
            input_root = root / "ofat"
            run_root = input_root / "M" / "sweep_alpha" / "alpha_0p5"

            (run_root / "data2").mkdir(parents=True)
            (run_root / "data2" / "dummy_data2.txt").write_text("data2\n", encoding="utf-8")

            (run_root / "data3").mkdir(parents=True)
            (run_root / "data3" / "dummy_data3.txt").write_text("data3\n", encoding="utf-8")

            dump_json(
                run_root / "case_config.json",
                {
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
                    "dataset": {
                        "model": "M",
                        "ssdr_work_dir": "/tmp/M/ssdr_work_dir",
                        "variant": "bone_reduced_conservative",
                        "qem_config": None,
                    },
                },
            )
            (run_root / "done").write_text("done\n", encoding="utf-8")

            rp.init_pool(pool)
            art.ensure_artifact_profiles(pool)

            dims, _eq, _index = rp.require_pool(pool)
            self.assertIn("ssdr_data2", dims["profiles"])
            self.assertIn("ssdr_data3", dims["profiles"])

            report = art.ingest_artifacts(
                pool=pool,
                input_root=input_root,
                stages=["ssdr_data2", "ssdr_data3"],
            )

            self.assertEqual(report["ingested"], 2)
            self.assertEqual(report["skipped_missing_payload"], 0)

            stats = rp.stats(pool)
            self.assertEqual(stats["artifact_count"], 2)
            self.assertEqual(stats["by_artifact_type"]["ssdr_data2"], 1)
            self.assertEqual(stats["by_artifact_type"]["ssdr_data3"], 1)

            key_data2 = root / "key_data2.json"
            dump_json(
                key_data2,
                {
                    "dimensions": {
                        "artifact_type": "ssdr_data2",
                        "model": "M",
                        "bone_variant": "bone_reduced_conservative",
                        "alpha": 0.5,
                        "beta": 0.3,
                        "gamma": 0.2,
                    }
                },
            )

            lookup2 = art.lookup_artifact(pool=pool, stage="ssdr_data2", key_json=key_data2)
            self.assertEqual(lookup2["hit_count"], 1)

            key_data3 = root / "key_data3.json"
            dump_json(
                key_data3,
                {
                    "dimensions": {
                        "artifact_type": "ssdr_data3",
                        "model": "M",
                        "bone_variant": "bone_reduced_conservative",
                        "alpha": 0.5,
                        "beta": 0.3,
                        "gamma": 0.2,
                        "omega_b": 1000.0,
                    }
                },
            )

            lookup3 = art.lookup_artifact(pool=pool, stage="ssdr_data3", key_json=key_data3)
            self.assertEqual(lookup3["hit_count"], 1)

            # Payloads are copied into artifact directories.
            hit_dir = Path(lookup3["hits"][0]["artifact_dir"])
            self.assertTrue((hit_dir / "payload" / "data3" / "dummy_data3.txt").is_file())

    def test_stage_profile_dimension_upgrade_uses_defaults_without_rewrite(self) -> None:
        rp = load_module("ogd_lbs_result_pool_for_artifact_dim_tests", "scripts/experiments/result_pool.py")
        art = load_module("ogd_lbs_result_pool_artifacts_for_dim_tests", "scripts/experiments/result_pool_artifacts.py")

        with tempfile.TemporaryDirectory(prefix="ogd_artifact_dim_") as td:
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
                "artifact_type": "ssdr_data2",
                "model": "M",
                "bone_variant": "default",
                "alpha": 0.5,
                "beta": 0.3,
                "gamma": 0.2,
            }
            dump_json(adir / "raw_dimensions.json", {"dimensions": old_dims})
            dump_json(adir / "artifact_manifest.json", {"artifact_id": artifact_id, "artifact_type": "ssdr_data2", "status": "succeeded"})

            index["artifacts"][artifact_id] = {
                "artifact_id": artifact_id,
                "artifact_type": "ssdr_data2",
                "status": "succeeded",
                "artifact_dir": f"artifacts/{artifact_id}",
                "dimension_fingerprint": "old",
                "model": "M",
                "param": "alpha",
                "value": 0.5,
            }
            index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

            rp.add_dimension(pool, profile_name="ssdr_data2", name="step02_new_knob", default=0)

            key_default = root / "key_default.json"
            dump_json(
                key_default,
                {
                    "dimensions": {
                        "artifact_type": "ssdr_data2",
                        "model": "M",
                        "bone_variant": "default",
                        "alpha": 0.5,
                        "beta": 0.3,
                        "gamma": 0.2,
                        "step02_new_knob": 0,
                    }
                },
            )
            lookup_default = art.lookup_artifact(pool=pool, stage="ssdr_data2", key_json=key_default)
            self.assertEqual(lookup_default["hit_count"], 1)

            key_non_default = root / "key_non_default.json"
            dump_json(
                key_non_default,
                {
                    "dimensions": {
                        "artifact_type": "ssdr_data2",
                        "model": "M",
                        "bone_variant": "default",
                        "alpha": 0.5,
                        "beta": 0.3,
                        "gamma": 0.2,
                        "step02_new_knob": 1,
                    }
                },
            )
            lookup_non_default = art.lookup_artifact(pool=pool, stage="ssdr_data2", key_json=key_non_default)
            self.assertEqual(lookup_non_default["hit_count"], 0)


if __name__ == "__main__":
    unittest.main()
