#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Materialize a run-specific result-pool subset from a central artifact-cache plan.

Purpose
-------
For large result pools, it is wasteful to rsync the whole pool to every worker.
This tool reads a case manifest produced by plan_ofat_with_artifact_cache.py,
extracts only the cache artifacts referenced by planned cases, creates a
read-only subset pool, and rewrites the manifest/case-record paths for worker
runtime.

Typical data4 workflow
----------------------
1. Controller runs planner against the full source pool:
      plan_ofat_with_artifact_cache.py --target_stage ssdr_data4 ...

2. Controller materializes subset:
      materialize_result_pool_subset.py
        --source_pool /workspace/ogd_experiment_pool
        --case_manifest local_runs/<RUN_ID>/controller_plan/case_manifest.json
        --output_pool local_runs/<RUN_ID>/pool_subset
        --runtime_pool /workspace/ogd_run_pool
        --runtime_plan_dir /oss/input/plan
        --output_manifest local_runs/<RUN_ID>/input/case_manifest.json
        --report local_runs/<RUN_ID>/input/plan/subset_report.json

3. Wrapper syncs local_runs/<RUN_ID>/pool_subset to:
      /workspace/ogd_run_pools/<RUN_ID>

4. Docker mounts:
      /workspace/ogd_run_pools/<RUN_ID>:/workspace/ogd_run_pool:ro

5. Worker only runs run_case_manifest.py on /oss/input/case_manifest.json.
"""

import argparse
import copy
import json
import shutil
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


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rp.json_safe(obj), indent=4, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return p


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size

    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            continue
    return total


def local_plan_dir_for_output_manifest(
    *,
    output_manifest: Path,
    runtime_plan_dir: str,
) -> Path:
    """
    Map worker runtime plan dir to local filesystem under output_manifest.parent.

    Common case:
      output_manifest      = local_runs/<RUN_ID>/input/case_manifest.json
      runtime_plan_dir     = /oss/input/plan
      local plan dir       = local_runs/<RUN_ID>/input/plan
    """
    runtime = Path(runtime_plan_dir)

    if runtime.is_absolute():
        parts = runtime.parts
        if len(parts) >= 3 and parts[1] == "oss" and parts[2] == "input":
            rel_parts = parts[3:]
            return output_manifest.parent.joinpath(*rel_parts) if rel_parts else output_manifest.parent
        return output_manifest.parent / runtime.name

    return output_manifest.parent / runtime


def artifact_id_from_hit(hit: Any) -> str | None:
    if not isinstance(hit, dict):
        return None

    aid = hit.get("artifact_id")
    if aid:
        return str(aid)

    artifact_dir = hit.get("artifact_dir")
    if artifact_dir:
        name = Path(str(artifact_dir)).name
        if name:
            return name

    rec = hit.get("record")
    if isinstance(rec, dict):
        aid = rec.get("artifact_id")
        if aid:
            return str(aid)

        rec_dir = rec.get("artifact_dir")
        if rec_dir:
            name = Path(str(rec_dir)).name
            if name:
                return name

    return None


def iter_cache_hits_from_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []

    cache_hits = case.get("cache_hits")
    if isinstance(cache_hits, dict):
        for value in cache_hits.values():
            if isinstance(value, dict):
                hits.append(value)

    # Some manifests may use pool_artifact/pool_artifacts for hits.
    pool_artifact = case.get("pool_artifact")
    if isinstance(pool_artifact, dict):
        hits.append(pool_artifact)

    pool_artifacts = case.get("pool_artifacts")
    if isinstance(pool_artifacts, list):
        for value in pool_artifacts:
            if isinstance(value, dict):
                hits.append(value)

    return hits


DEFAULT_REQUIRED_CACHE_STAGES_BY_TARGET = {
    "ssdr_data2": ["ssdr_data1"],
    "ssdr_data3": ["ssdr_data2"],
    "ssdr_data4": ["ssdr_data3"],
}


def infer_target_stage(manifest: dict[str, Any]) -> str:
    value = manifest.get("target_stage")
    if value:
        return str(value)

    pool_obj = manifest.get("pool")
    if isinstance(pool_obj, dict) and pool_obj.get("target_stage"):
        return str(pool_obj["target_stage"])

    for case in manifest.get("cases", []) or []:
        if isinstance(case, dict) and case.get("target_stage"):
            return str(case["target_stage"])

    # Backward-compatible default: final OFAT target.
    return "ssdr_data4"


def default_cache_stages_for_manifest(manifest: dict[str, Any]) -> list[str]:
    target_stage = infer_target_stage(manifest)
    return list(DEFAULT_REQUIRED_CACHE_STAGES_BY_TARGET.get(target_stage, []))


def normalize_cache_stage_filter(
    manifest: dict[str, Any],
    *,
    cache_stages: list[str] | None = None,
    include_all_cache_hits: bool = False,
) -> list[str] | None:
    """
    Return stages to collect from case.cache_hits.

    None means collect all cache hits.

    Default behavior intentionally collects only the direct upstream cache
    required to execute the target stage:
      ssdr_data4 <- ssdr_data3
      ssdr_data3 <- ssdr_data2
      ssdr_data2 <- ssdr_data1

    The planner may include older upstream hits for diagnostics, but they are
    not required at runtime once a nearer upstream artifact is available.
    """
    if include_all_cache_hits:
        return None

    if cache_stages:
        return [str(x) for x in cache_stages]

    stages = default_cache_stages_for_manifest(manifest)
    return stages if stages else None


def collect_artifact_ids(
    manifest: dict[str, Any],
    *,
    cache_stages: list[str] | None = None,
    include_all_cache_hits: bool = False,
) -> list[str]:
    ids: set[str] = set()
    stage_filter = normalize_cache_stage_filter(
        manifest,
        cache_stages=cache_stages,
        include_all_cache_hits=include_all_cache_hits,
    )
    allowed = set(stage_filter) if stage_filter is not None else None

    for case in manifest.get("cases", []) or []:
        if not isinstance(case, dict):
            continue

        cache_hits = case.get("cache_hits")
        if isinstance(cache_hits, dict):
            for stage_name, hit in cache_hits.items():
                if allowed is not None and str(stage_name) not in allowed:
                    continue
                if isinstance(hit, dict):
                    aid = artifact_id_from_hit(hit)
                    if aid:
                        ids.add(aid)

        # pool_artifact/pool_artifacts represent direct target hits/final hits.
        # They are normally not present for planned miss cases, but preserve
        # support for explicit include_all_cache_hits/debug use.
        if include_all_cache_hits:
            for hit in iter_cache_hits_from_case(case):
                aid = artifact_id_from_hit(hit)
                if aid:
                    ids.add(aid)

    return sorted(ids)


def rewrite_hit_for_runtime(hit: Any, *, runtime_pool: str) -> Any:
    if not isinstance(hit, dict):
        return hit

    out = copy.deepcopy(hit)
    aid = artifact_id_from_hit(out)

    if aid:
        out["artifact_id"] = aid
        out["artifact_dir"] = str(Path(runtime_pool) / "artifacts" / aid)

    return out


def rewrite_cache_hits_for_runtime(obj: Any, *, runtime_pool: str) -> Any:
    """
    Rewrite artifact hit dictionaries recursively enough for the known manifest
    and case-record structures, without changing arbitrary metadata.
    """
    if isinstance(obj, list):
        return [rewrite_cache_hits_for_runtime(x, runtime_pool=runtime_pool) for x in obj]

    if not isinstance(obj, dict):
        return obj

    out: dict[str, Any] = {}

    for key, value in obj.items():
        if key == "cache_hits" and isinstance(value, dict):
            out[key] = {
                str(k): rewrite_hit_for_runtime(v, runtime_pool=runtime_pool)
                for k, v in value.items()
            }
        elif key in {"pool_artifact"} and isinstance(value, dict):
            out[key] = rewrite_hit_for_runtime(value, runtime_pool=runtime_pool)
        elif key in {"pool_artifacts"} and isinstance(value, list):
            out[key] = [
                rewrite_hit_for_runtime(v, runtime_pool=runtime_pool) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            out[key] = rewrite_cache_hits_for_runtime(value, runtime_pool=runtime_pool)

    return out


def update_command_case_json(command: Any, runtime_case_record: str) -> Any:
    if not isinstance(command, list):
        return command

    out = [str(x) for x in command]
    for i, item in enumerate(out):
        if item == "--case_json" and i + 1 < len(out):
            out[i + 1] = runtime_case_record
            return out
    return out


def copy_file_if_exists(src: Path, dst: Path, *, role: str) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing {role}: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_pool_metadata(
    *,
    source_pool: Path,
    output_pool: Path,
    artifact_ids: list[str],
) -> tuple[dict[str, Any], dict[str, int], int]:
    rp.require_pool(source_pool)
    source_paths = rp.get_pool_paths(source_pool)
    output_paths = rp.get_pool_paths(output_pool)

    if output_pool.exists():
        raise FileExistsError(f"output_pool already exists; use --overwrite to replace: {output_pool}")

    output_paths["artifacts"].mkdir(parents=True, exist_ok=True)

    # Copy stable metadata.
    for key in ["manifest", "dimensions", "profile"]:
        if source_paths[key].is_file():
            shutil.copy2(source_paths[key], output_paths[key])

    source_index = rp.load_json(source_paths["index"])
    source_artifacts = source_index.get("artifacts", {}) or {}

    missing: list[str] = []
    subset_artifacts: dict[str, Any] = {}
    by_type: dict[str, int] = {}
    total_bytes = 0

    for aid in artifact_ids:
        rec = source_artifacts.get(aid)
        if not isinstance(rec, dict):
            missing.append(aid)
            continue

        src_dir = rp.artifact_dir(source_pool, aid)
        dst_dir = rp.artifact_dir(output_pool, aid)

        if not src_dir.is_dir():
            missing.append(aid)
            continue

        shutil.copytree(src_dir, dst_dir)

        subset_artifacts[aid] = copy.deepcopy(rec)
        atype = str(rec.get("artifact_type", "<unknown>"))
        by_type[atype] = by_type.get(atype, 0) + 1
        total_bytes += path_size_bytes(dst_dir)

    if missing:
        shutil.rmtree(output_pool, ignore_errors=True)
        raise FileNotFoundError(f"Missing artifacts in source pool: {missing}")

    subset_index = {
        "schema": source_index.get("schema", "ogd_lbs_result_pool_index_v2"),
        "updated_at_utc": rp.utc_now_iso(),
        "next_artifact_id": source_index.get("next_artifact_id", 1),
        "artifacts": subset_artifacts,
        "optional_fast_indexes": {},
        "subset_from_pool": str(source_pool),
        "subset_artifact_count": len(subset_artifacts),
    }
    rp.dump_json(output_paths["index"], subset_index)

    return subset_index, dict(sorted(by_type.items())), total_bytes


def rewrite_manifest_and_records(
    *,
    manifest: dict[str, Any],
    source_manifest_path: Path,
    output_manifest: Path,
    runtime_pool: str,
    runtime_plan_dir: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local_plan_dir = local_plan_dir_for_output_manifest(
        output_manifest=output_manifest,
        runtime_plan_dir=runtime_plan_dir,
    )
    local_case_record_dir = local_plan_dir / "case_records"
    local_single_cfg_dir = local_plan_dir / "single_case_configs"

    local_case_record_dir.mkdir(parents=True, exist_ok=True)
    local_single_cfg_dir.mkdir(parents=True, exist_ok=True)

    rewritten_manifest = copy.deepcopy(manifest)
    rewritten_manifest["subset_runtime"] = {
        "runtime_pool": runtime_pool,
        "runtime_plan_dir": runtime_plan_dir,
        "source_manifest": str(source_manifest_path),
    }

    records_report: list[dict[str, Any]] = []
    new_cases: list[dict[str, Any]] = []

    for idx, case in enumerate(manifest.get("cases", []) or []):
        if not isinstance(case, dict):
            continue

        source_record_path = Path(str(case.get("case_record_json", ""))).expanduser()
        if not source_record_path.is_file():
            raise FileNotFoundError(f"case_record_json not found for case {idx}: {source_record_path}")

        source_single_cfg_path = Path(str(case.get("single_case_config", ""))).expanduser()
        if not source_single_cfg_path.is_file():
            raise FileNotFoundError(f"single_case_config not found for case {idx}: {source_single_cfg_path}")

        case_record_name = source_record_path.name
        single_cfg_name = source_single_cfg_path.name

        local_record_path = local_case_record_dir / case_record_name
        local_single_cfg_path = local_single_cfg_dir / single_cfg_name

        runtime_record_path = str(Path(runtime_plan_dir) / "case_records" / case_record_name)
        runtime_single_cfg_path = str(Path(runtime_plan_dir) / "single_case_configs" / single_cfg_name)

        # Copy single case config to local runtime plan dir.
        copy_file_if_exists(source_single_cfg_path, local_single_cfg_path, role="single_case_config")

        # Rewrite case record.
        record = rp.load_json(source_record_path)
        record = rewrite_cache_hits_for_runtime(record, runtime_pool=runtime_pool)
        record["case_record_json"] = runtime_record_path
        record["single_case_config"] = runtime_single_cfg_path
        record["command"] = update_command_case_json(record.get("command"), runtime_record_path)
        record["subset_runtime"] = {
            "runtime_pool": runtime_pool,
            "runtime_plan_dir": runtime_plan_dir,
            "source_case_record_json": str(source_record_path),
            "source_single_case_config": str(source_single_cfg_path),
        }
        dump_json(local_record_path, record)

        # Rewrite case entry in manifest.
        new_case = rewrite_cache_hits_for_runtime(case, runtime_pool=runtime_pool)
        new_case["case_record_json"] = runtime_record_path
        new_case["single_case_config"] = runtime_single_cfg_path
        new_case["command"] = update_command_case_json(new_case.get("command"), runtime_record_path)
        new_case["subset_runtime"] = {
            "runtime_pool": runtime_pool,
            "runtime_plan_dir": runtime_plan_dir,
            "source_case_record_json": str(source_record_path),
            "source_single_case_config": str(source_single_cfg_path),
        }
        new_cases.append(new_case)

        records_report.append(
            {
                "source_case_record": str(source_record_path),
                "local_case_record": str(local_record_path),
                "runtime_case_record": runtime_record_path,
                "source_single_case_config": str(source_single_cfg_path),
                "local_single_case_config": str(local_single_cfg_path),
                "runtime_single_case_config": runtime_single_cfg_path,
            }
        )

    rewritten_manifest["cases"] = new_cases
    rewritten_manifest["case_count"] = len(new_cases)

    dump_json(output_manifest, rewritten_manifest)
    return rewritten_manifest, records_report


def materialize_subset(
    *,
    source_pool: Path,
    case_manifest: Path,
    output_pool: Path,
    runtime_pool: str,
    runtime_plan_dir: str,
    output_manifest: Path,
    report_path: Path,
    overwrite: bool,
    cache_stages: list[str] | None = None,
    include_all_cache_hits: bool = False,
) -> dict[str, Any]:
    source_pool = Path(source_pool).expanduser().resolve(strict=False)
    case_manifest = Path(case_manifest).expanduser().resolve(strict=False)
    output_pool = Path(output_pool).expanduser().resolve(strict=False)
    output_manifest = Path(output_manifest).expanduser().resolve(strict=False)
    report_path = Path(report_path).expanduser().resolve(strict=False)

    if overwrite and output_pool.exists():
        shutil.rmtree(output_pool)

    if output_manifest.exists() and overwrite:
        output_manifest.unlink()

    if report_path.exists() and overwrite:
        report_path.unlink()

    manifest = rp.load_json(case_manifest)
    if not isinstance(manifest, dict):
        raise ValueError(f"case_manifest must contain a JSON object: {case_manifest}")

    selected_cache_stages = normalize_cache_stage_filter(
        manifest,
        cache_stages=cache_stages,
        include_all_cache_hits=include_all_cache_hits,
    )
    artifact_ids = collect_artifact_ids(
        manifest,
        cache_stages=cache_stages,
        include_all_cache_hits=include_all_cache_hits,
    )

    subset_index, by_type, total_bytes = copy_pool_metadata(
        source_pool=source_pool,
        output_pool=output_pool,
        artifact_ids=artifact_ids,
    )

    rewritten_manifest, records_report = rewrite_manifest_and_records(
        manifest=manifest,
        source_manifest_path=case_manifest,
        output_manifest=output_manifest,
        runtime_pool=runtime_pool,
        runtime_plan_dir=runtime_plan_dir,
    )

    report = {
        "schema": "ogd_lbs_result_pool_subset_report_v1",
        "source_pool": str(source_pool),
        "source_manifest": str(case_manifest),
        "output_pool": str(output_pool),
        "runtime_pool": str(runtime_pool),
        "runtime_plan_dir": str(runtime_plan_dir),
        "output_manifest": str(output_manifest),
        "report": str(report_path),
        "case_count": int(len(rewritten_manifest.get("cases", []) or [])),
        "artifact_count": int(len(artifact_ids)),
        "artifact_ids": artifact_ids,
        "selected_cache_stages": selected_cache_stages,
        "include_all_cache_hits": bool(include_all_cache_hits),
        "by_artifact_type": by_type,
        "total_subset_size_bytes": int(total_bytes),
        "subset_index_artifact_count": int(len((subset_index.get("artifacts") or {}))),
        "records": records_report,
    }

    dump_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a run-specific OGD-LBS result-pool subset from a planned manifest."
    )
    parser.add_argument("--source_pool", required=True)
    parser.add_argument("--case_manifest", required=True)
    parser.add_argument("--output_pool", required=True)
    parser.add_argument("--runtime_pool", required=True)
    parser.add_argument("--runtime_plan_dir", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--cache_stages",
        nargs="*",
        default=None,
        help="Cache hit stages to include. Default selects direct upstream stage for target_stage.",
    )
    parser.add_argument(
        "--include_all_cache_hits",
        action="store_true",
        help="Include all cache_hits from planned cases. Mainly for diagnostics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = materialize_subset(
        source_pool=Path(args.source_pool),
        case_manifest=Path(args.case_manifest),
        output_pool=Path(args.output_pool),
        runtime_pool=str(args.runtime_pool),
        runtime_plan_dir=str(args.runtime_plan_dir),
        output_manifest=Path(args.output_manifest),
        report_path=Path(args.report),
        overwrite=bool(args.overwrite),
        cache_stages=args.cache_stages,
        include_all_cache_hits=bool(args.include_all_cache_hits),
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
