#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Plan OFAT cases with local result/artifact pool de-duplication.

This planner outputs an explicit case manifest. It uses:
  - ssdr_data4 flat-dimension lookup to skip already completed final results
  - ssdr_data3 artifact lookup to mark cases that can start from cached data3
  - ssdr_data2 artifact lookup to mark cases that can start from cached data2

It does not execute anything.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


def ensure_repo_paths() -> Path:
    cur = Path(__file__).resolve(strict=False)
    for parent in [cur.parent, *cur.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            repo = parent
            break
    else:
        repo = Path.cwd()

    for item in [
        repo / "src",
        repo / "scripts" / "experiments",
        repo / "scripts" / "ssdr",
        repo / "scripts" / "data_prep",
    ]:
        s = str(item)
        if s not in sys.path:
            sys.path.insert(0, s)

    return repo


REPO_ROOT = ensure_repo_paths()

import result_pool as rp  # noqa: E402
import result_pool_artifacts as rpa  # noqa: E402
import run_param_sweep as rps  # noqa: E402


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rp.json_safe(obj), indent=4, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def make_case_config_dict(cfg: dict[str, Any], case: dict[str, Any], output_root: str | None) -> dict[str, Any]:
    single = copy.deepcopy(cfg)
    single["datasets"] = [copy.deepcopy(case["dataset"])]

    if output_root:
        single["output_root"] = output_root

    param = str(case["param"])
    value = case["value"]
    model = str(case["model"])

    single["sweeps"] = {}
    single["delta_sweeps_by_model"] = {}

    if param == "delta":
        single["delta_sweeps_by_model"] = {model: [value]}
    else:
        single["sweeps"] = {param: [value]}

    return single


def make_case_config_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ogd_lbs_ofat_case_config_v1",
        "model": case.get("model"),
        "param": case.get("param"),
        "value": case.get("value"),
        "case_id": case.get("case_id"),
        "stage": case.get("stage"),
        "effective_params": case.get("effective_params", {}),
        "dataset": case.get("dataset", {}),
    }


def combined_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(cfg.get("defaults") or {})
    ssdr_defaults = dict(cfg.get("ssdr_defaults") or {})

    defaults.setdefault("max_frames", ssdr_defaults.get("max_frames", 210))
    defaults.setdefault("target_frames", ssdr_defaults.get("target_frames", 200))
    defaults.setdefault("frame_sampling_policy", "target_200")
    defaults.setdefault("step02_affinity_eps", 1e-5)
    defaults.setdefault("step01_code_version", "step01_v1")
    defaults.setdefault("step02_code_version", "step02_v1")
    defaults.setdefault("step03_code_version", "step03_v1")
    defaults.setdefault("step04_code_version", "step04_v1")
    defaults.setdefault("qem_config_fingerprint", None)

    return defaults


def lookup_first(pool: Path, raw_dims: dict[str, Any], profile_name: str) -> dict[str, Any] | None:
    hits = rp.lookup_artifacts(pool=pool, raw_dims=raw_dims, profile_name=profile_name)
    if hits:
        return hits[0]
    return None



def normalize_target_stage(value: str) -> str:
    text = str(value or "ssdr_data4").strip()
    aliases = {
        "data2": "ssdr_data2",
        "ssdr_data2": "ssdr_data2",
        "stage02": "ssdr_data2",
        "stage02_data2": "ssdr_data2",

        "data3": "ssdr_data3",
        "ssdr_data3": "ssdr_data3",
        "stage03": "ssdr_data3",
        "stage03_data3": "ssdr_data3",

        "data4": "ssdr_data4",
        "ssdr_data4": "ssdr_data4",
        "stage04": "ssdr_data4",
        "stage04_data4": "ssdr_data4",
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported target_stage={value!r}; "
            "expected ssdr_data2, ssdr_data3, or ssdr_data4."
        )
    return aliases[text]

def plan(
    *,
    pool: Path,
    config_path: Path,
    output_manifest: Path,
    output_plan_dir: Path,
    output_root: str | None,
    only_models: list[str] | None,
    only_params: list[str] | None,
    no_conda: bool,
    target_stage: str = "ssdr_data4",
    require_input_cache: bool = False,
) -> dict[str, Any]:
    rp.init_pool(pool, force=False)
    rpa.ensure_artifact_profiles(pool)

    target_stage = normalize_target_stage(target_stage)

    cfg_raw = rps.load_json(config_path)
    cfg = rps.normalize_config(cfg_raw, output_root_override=output_root)

    datasets = rps.selected_datasets(cfg, only_models)
    params = rps.selected_params(cfg, only_params)
    cases = rps.iter_cases(cfg, datasets=datasets, params=params)

    output_plan_dir.mkdir(parents=True, exist_ok=True)
    single_cfg_dir = output_plan_dir / "single_case_configs"
    case_record_dir = output_plan_dir / "case_records"
    logs_dir = output_plan_dir / "logs"

    single_cfg_dir.mkdir(parents=True, exist_ok=True)
    case_record_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    defaults = combined_defaults(cfg)

    target_hits: list[dict[str, Any]] = []
    final_hits: list[dict[str, Any]] = []
    planned_cases: list[dict[str, Any]] = []
    input_cache_misses: list[dict[str, Any]] = []

    for source_idx, case in enumerate(cases):
        case_cfg_payload = make_case_config_payload(case)

        raw_data4 = rp.build_raw_dims_from_rps_case(case, cfg)
        raw_data3 = rpa.raw_stage_dims_from_case_config(
            case_cfg_payload,
            artifact_type="ssdr_data3",
            config_defaults=defaults,
        )
        raw_data2 = rpa.raw_stage_dims_from_case_config(
            case_cfg_payload,
            artifact_type="ssdr_data2",
            config_defaults=defaults,
        )
        raw_data1 = rpa.raw_stage_dims_from_case_config(
            case_cfg_payload,
            artifact_type="ssdr_data1",
            config_defaults=defaults,
        )

        raw_by_stage = {
            "ssdr_data1": raw_data1,
            "ssdr_data2": raw_data2,
            "ssdr_data3": raw_data3,
            "ssdr_data4": raw_data4,
        }

        raw_target = raw_by_stage[target_stage]
        target_hit = lookup_first(pool, raw_target, target_stage)

        base_record = {
            "source_case_index": int(source_idx),
            "model": case.get("model"),
            "param": case.get("param"),
            "value": case.get("value"),
            "case_id": case.get("case_id"),
            "stage": case.get("stage"),
            "target_stage": target_stage,
            "case_config_payload": case_cfg_payload,
            "raw_dimensions": {
                "ssdr_data4": raw_data4,
                "ssdr_data3": raw_data3,
                "ssdr_data2": raw_data2,
                "ssdr_data1": raw_data1,
            },
        }

        if target_hit is not None:
            rec = dict(base_record)
            rec["pool_hit"] = True
            rec["pool_hit_stage"] = target_stage
            rec["pool_artifact"] = target_hit
            target_hits.append(rec)
            if target_stage == "ssdr_data4":
                final_hits.append(rec)
            continue

        data3_hit = lookup_first(pool, raw_data3, "ssdr_data3")
        data2_hit = lookup_first(pool, raw_data2, "ssdr_data2")
        data1_hit = lookup_first(pool, raw_data1, "ssdr_data1")

        if target_stage == "ssdr_data2" and data1_hit is None:
            miss = dict(base_record)
            miss["pool_hit"] = False
            miss["missing_required_input_cache"] = "ssdr_data1"
            input_cache_misses.append(miss)
            if require_input_cache:
                continue

        safe = f"case_{len(planned_cases):05d}_{case['model']}_{case['param']}_{rps.slugify(case['value'])}"
        single_cfg_path = single_cfg_dir / f"{safe}.json"

        single_cfg = make_case_config_dict(cfg, case, output_root)
        rps.dump_json(single_cfg_path, single_cfg)

        rec = dict(base_record)
        rec["pool_hit"] = False
        rec["manifest_case_index"] = len(planned_cases)
        rec["single_case_config"] = str(single_cfg_path)
        rec["log_path"] = str(logs_dir / f"{safe}.log")
        rec["cache_hits"] = {
            "ssdr_data3": data3_hit,
            "ssdr_data2": data2_hit,
            "ssdr_data1": data1_hit,
        }

        case_record_path = case_record_dir / f"{safe}.json"
        rec["case_record_json"] = str(case_record_path)

        cmd = [
            sys.executable,
            "scripts/experiments/run_cached_ofat_case.py",
            "--case_json",
            str(case_record_path),
            "--target_stage",
            target_stage,
        ]

        if no_conda:
            cmd.append("--no_conda")

        if require_input_cache:
            cmd.append("--require_input_cache")

        rec["command"] = cmd

        dump_json(case_record_path, rec)
        planned_cases.append(rec)

    manifest = {
        "schema": "ogd_lbs_case_manifest_v2",
        "mode": "ofat_artifact_cache",
        "source_config": str(config_path),
        "output_root": output_root or cfg.get("output_root"),
        "target_stage": target_stage,
        "pool": {
            "path": str(pool),
            "candidate_count": len(cases),
            "target_stage": target_stage,
            "target_hit_count": len(target_hits),
            "final_hit_count": len(final_hits),
            "miss_count": len(planned_cases),
            "planned_count": len(planned_cases),
            "match_mode": "flat_dimensions_linear_scan",
            "require_input_cache": bool(require_input_cache),
            "input_cache_miss_count": len(input_cache_misses),
        },
        "case_count": len(planned_cases),
        "cases": planned_cases,
        "input_cache_misses": input_cache_misses,
    }

    dump_json(output_manifest, manifest)

    report = {
        "schema": "ogd_lbs_artifact_cache_plan_report_v2",
        "source_config": str(config_path),
        "pool": str(pool),
        "target_stage": target_stage,
        "candidate_count": len(cases),
        "target_hit_count": len(target_hits),
        "final_hit_count": len(final_hits),
        "miss_count": len(planned_cases),
        "planned_count": len(planned_cases),
        "data3_cache_hit_count": sum(1 for c in planned_cases if c.get("cache_hits", {}).get("ssdr_data3")),
        "data2_cache_hit_count": sum(1 for c in planned_cases if c.get("cache_hits", {}).get("ssdr_data2")),
        "data1_cache_hit_count": sum(1 for c in planned_cases if c.get("cache_hits", {}).get("ssdr_data1")),
        "data1_cache_miss_count": sum(1 for c in planned_cases if not c.get("cache_hits", {}).get("ssdr_data1")),
        "require_input_cache": bool(require_input_cache),
        "input_cache_misses": input_cache_misses,
        "target_hits": target_hits,
        "final_hits": final_hits,
        "manifest": str(output_manifest),
        "plan_dir": str(output_plan_dir),
    }

    dump_json(output_plan_dir / "artifact_cache_plan_report.json", report)

    if require_input_cache and input_cache_misses:
        raise RuntimeError(
            f"Missing required input cache for {len(input_cache_misses)} cases; "
            f"see {output_plan_dir / 'artifact_cache_plan_report.json'}"
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan OFAT cases with result/artifact cache lookup.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--output_plan_dir", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--only_models", nargs="*", default=None)
    parser.add_argument("--only_params", nargs="*", default=None)
    parser.add_argument("--no_conda", action="store_true")
    parser.add_argument("--target_stage", default="ssdr_data4", choices=["ssdr_data2", "ssdr_data3", "ssdr_data4"])
    parser.add_argument("--require_input_cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = plan(
        pool=Path(args.pool).expanduser().resolve(strict=False),
        config_path=Path(args.config).expanduser().resolve(strict=False),
        output_manifest=Path(args.output_manifest).expanduser().resolve(strict=False),
        output_plan_dir=Path(args.output_plan_dir).expanduser().resolve(strict=False),
        output_root=args.output_root,
        only_models=args.only_models,
        only_params=args.only_params,
        no_conda=bool(args.no_conda),
        target_stage=str(args.target_stage),
        require_input_cache=bool(args.require_input_cache),
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
