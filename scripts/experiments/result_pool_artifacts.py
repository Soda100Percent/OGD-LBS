#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Stage artifact cache utilities for OGD-LBS local result pool.

Supported artifact types:
    - ssdr_data1
    - ssdr_data2
    - ssdr_data3

Design:
    - flat canonical dimensions are the primary match mechanism
    - missing dimensions are filled from current profile defaults
    - equivalence rules are applied to both query and record
    - lookup uses linear scan
    - dimension_fingerprint is diagnostic only, not the identity
"""

import argparse
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
        item_s = str(item)
        if item_s not in sys.path:
            sys.path.insert(0, item_s)

    return repo


REPO_ROOT = ensure_repo_paths()

import result_pool as rp  # noqa: E402


SSDR_DATA1_DIMENSIONS = [
    "artifact_type",
    "model",

    "input_source_kind",
    "input_dataset_version",

    "frame_sampling_policy",
    "max_frames",
    "target_frames",
    "frame_sampling_config_name",
    "frame_removal_idx_ranges",

    "step00_code_version",
    "step01_code_version",
]

SSDR_DATA1_DEFAULTS = {
    "artifact_type": "ssdr_data1",

    # Default to the commercial/DataPrep handoff source so older/minimal
    # data2/data3 lookup keys that omit input_source_kind still match
    # non-VRoid artifacts. VRoid cache-prep records should explicitly set
    # input_source_kind="vroid_data0".
    "input_source_kind": "fitbox_vdb",
    "input_dataset_version": "v1",

    "frame_sampling_policy": "target_200",
    "max_frames": 210,
    "target_frames": 200,
    "frame_sampling_config_name": "default",
    "frame_removal_idx_ranges": "",

    "step00_code_version": "step00_v1",
    "step01_code_version": "step01_v1",
}

SSDR_DATA2_DIMENSIONS = [
    *SSDR_DATA1_DIMENSIONS,

    "bone_variant",

    "alpha",
    "beta",
    "gamma",
    "step02_affinity_eps",

    "step02_code_version",
]

SSDR_DATA2_DEFAULTS = {
    **SSDR_DATA1_DEFAULTS,
    "artifact_type": "ssdr_data2",

    "bone_variant": "default",

    "alpha": 0.5,
    "beta": 0.3,
    "gamma": 0.2,
    "step02_affinity_eps": 1e-5,

    "step02_code_version": "step02_v1",
}

SSDR_DATA3_DIMENSIONS = [
    *SSDR_DATA2_DIMENSIONS,

    "omega_b",
    "qem_config_name",
    "qem_config_fingerprint",

    "step03_code_version",
]

SSDR_DATA3_DEFAULTS = {
    **SSDR_DATA2_DEFAULTS,
    "artifact_type": "ssdr_data3",

    "omega_b": 1000.0,
    "qem_config_name": "default",
    "qem_config_fingerprint": None,

    "step03_code_version": "step03_v1",
}

STAGE_TO_PAYLOAD_DIR = {
    "ssdr_data1": "data1",
    "ssdr_data2": "data2",
    "ssdr_data3": "data3",
}


def normalize_stage_name(stage: str) -> str:
    text = str(stage).strip()
    aliases = {
        "data1": "ssdr_data1",
        "ssdr_data1": "ssdr_data1",
        "stage01": "ssdr_data1",
        "stage01_data1": "ssdr_data1",

        "data2": "ssdr_data2",
        "ssdr_data2": "ssdr_data2",
        "stage02": "ssdr_data2",
        "stage02_data2": "ssdr_data2",

        "data3": "ssdr_data3",
        "ssdr_data3": "ssdr_data3",
        "stage03": "ssdr_data3",
        "stage03_data3": "ssdr_data3",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported artifact stage: {stage!r}")
    return aliases[text]


def supported_stages() -> list[str]:
    return ["ssdr_data1", "ssdr_data2", "ssdr_data3"]


def ensure_artifact_profiles(pool: Path) -> None:
    """
    Ensure dimensions.json has ssdr_data1, ssdr_data2, and ssdr_data3 profiles.

    This is intentionally non-destructive:
      - existing profiles are preserved
      - missing profiles are added
      - optional_fast_indexes are invalidated only if dimensions change
    """
    rp.init_pool(pool, force=False)
    paths = rp.get_pool_paths(pool)
    dimensions_obj, _eq_profile, index = rp.require_pool(pool)

    profiles = dimensions_obj.setdefault("profiles", {})
    changed = False

    wanted = {
        "ssdr_data1": {
            "profile_version": 1,
            "dimensions": list(SSDR_DATA1_DIMENSIONS),
            "defaults": dict(SSDR_DATA1_DEFAULTS),
        },
        "ssdr_data2": {
            "profile_version": 1,
            "dimensions": list(SSDR_DATA2_DIMENSIONS),
            "defaults": dict(SSDR_DATA2_DEFAULTS),
        },
        "ssdr_data3": {
            "profile_version": 1,
            "dimensions": list(SSDR_DATA3_DIMENSIONS),
            "defaults": dict(SSDR_DATA3_DEFAULTS),
        },
    }

    for name, profile in wanted.items():
        if name not in profiles:
            profiles[name] = profile
            changed = True

    if changed:
        dimensions_obj["profiles"] = profiles
        rp.dump_json(paths["dimensions"], dimensions_obj)

        index["optional_fast_indexes"] = {}
        rp.save_index(pool, index)


def case_dim_value(
    case_cfg: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """
    Read artifact dimensions from several possible locations.

    Preferred future shape:
        case_config["artifact_dimensions"][key]

    Also supports top-level keys for simple cache-prep jobs.
    """
    for container_key in ["artifact_dimensions", "dimensions", "stage_dimensions"]:
        container = case_cfg.get(container_key)
        if isinstance(container, dict) and key in container:
            return container[key]

    if key in case_cfg:
        return case_cfg[key]

    return default


def infer_input_source_kind(case_cfg: dict[str, Any]) -> str:
    value = case_dim_value(case_cfg, "input_source_kind", None)
    if value is not None:
        return str(value)

    model = str(case_cfg.get("model", ""))
    if model.startswith("Vroid_"):
        return "vroid_data0"

    return "fitbox_vdb"


def infer_frame_sampling_config_name(case_cfg: dict[str, Any]) -> str:
    value = case_dim_value(case_cfg, "frame_sampling_config_name", None)
    if value is not None:
        return str(value)

    model = str(case_cfg.get("model", ""))
    if model.startswith("Vroid_"):
        return "frame_sampling_config_vroid.json"

    return "default"



def _format_range_pairs_as_text(value: Any) -> str:
    """
    Convert JSON-style [[start, end], ...] ranges to the canonical
    left-closed/right-open text representation used by cache dimensions:
        "20:60,260:340"
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if not isinstance(value, (list, tuple)):
        return ""

    parts: list[str] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return ""
        start = int(item[0])
        end = int(item[1])
        parts.append(f"{start}:{end}")

    return ",".join(parts)


def load_vroid_frame_removal_idx_ranges_text() -> str:
    """
    Read the shared VRoid Step01 frame-removal contract from repo config.

    This keeps result-pool lookup dimensions aligned with the R1 data1 cache
    preparation job and with run_dataset_ssdr.sh.
    """
    cfg = REPO_ROOT / "scripts" / "ssdr" / "configs" / "frame_sampling_config_vroid.json"
    if not cfg.is_file():
        return ""

    try:
        obj = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return ""

    step01 = obj.get("step01_params", {})
    if not isinstance(step01, dict):
        return ""

    text = str(step01.get("removal_idx_ranges_text", "") or "").strip()
    if text:
        return text

    return _format_range_pairs_as_text(step01.get("removal_idx_ranges"))


def infer_frame_removal_idx_ranges(
    case_cfg: dict[str, Any],
    *,
    config_defaults: dict[str, Any] | None = None,
) -> str:
    """
    Infer the Step01 frame-removal dimension for staged artifacts.

    Priority:
      1. explicit case artifact_dimensions/dimensions/stage_dimensions/top-level value
      2. config_defaults["frame_removal_idx_ranges"]
      3. shared VRoid config for Vroid_* / frame_sampling_config_vroid.json
      4. empty string
    """
    config_defaults = dict(config_defaults or {})

    explicit = case_dim_value(case_cfg, "frame_removal_idx_ranges", None)
    if explicit is not None:
        return str(explicit).strip()

    default_value = config_defaults.get("frame_removal_idx_ranges", None)
    if default_value is not None:
        return str(default_value).strip()

    model = str(case_cfg.get("model", ""))
    fs_config = infer_frame_sampling_config_name(case_cfg)

    if model.startswith("Vroid_") or fs_config == "frame_sampling_config_vroid.json":
        return load_vroid_frame_removal_idx_ranges_text()

    return ""

def infer_bone_variant(dataset: dict[str, Any] | None) -> str:
    dataset = dataset or {}
    return str(dataset.get("variant") or "default")


def infer_qem_config_name(dataset: dict[str, Any] | None) -> str:
    dataset = dataset or {}
    qem = dataset.get("qem_config")
    if qem is None or str(qem).strip() == "":
        return "default"
    return Path(str(qem)).name


def raw_stage_dims_from_case_config(
    case_cfg: dict[str, Any],
    *,
    artifact_type: str,
    step04_config: dict[str, Any] | None = None,
    config_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_type = normalize_stage_name(artifact_type)
    _step04_config = dict(step04_config or {})
    config_defaults = dict(config_defaults or {})
    dataset = dict(case_cfg.get("dataset") or {})
    eff = dict(case_cfg.get("effective_params") or {})

    def first(*values: Any) -> Any:
        for v in values:
            if v is not None:
                return v
        return None

    raw: dict[str, Any] = {
        "artifact_type": artifact_type,
        "model": case_cfg.get("model"),

        "input_source_kind": case_dim_value(
            case_cfg,
            "input_source_kind",
            infer_input_source_kind(case_cfg),
        ),
        "input_dataset_version": case_dim_value(
            case_cfg,
            "input_dataset_version",
            config_defaults.get("input_dataset_version"),
        ),

        "frame_sampling_policy": case_dim_value(
            case_cfg,
            "frame_sampling_policy",
            config_defaults.get("frame_sampling_policy"),
        ),
        "max_frames": case_dim_value(
            case_cfg,
            "max_frames",
            config_defaults.get("max_frames"),
        ),
        "target_frames": case_dim_value(
            case_cfg,
            "target_frames",
            config_defaults.get("target_frames"),
        ),
        "frame_sampling_config_name": case_dim_value(
            case_cfg,
            "frame_sampling_config_name",
            infer_frame_sampling_config_name(case_cfg),
        ),
        "frame_removal_idx_ranges": infer_frame_removal_idx_ranges(
            case_cfg,
            config_defaults=config_defaults,
        ),

        "step00_code_version": case_dim_value(
            case_cfg,
            "step00_code_version",
            config_defaults.get("step00_code_version"),
        ),
        "step01_code_version": case_dim_value(
            case_cfg,
            "step01_code_version",
            config_defaults.get("step01_code_version"),
        ),
    }

    if artifact_type == "ssdr_data1":
        return raw

    raw.update(
        {
            "bone_variant": case_dim_value(
                case_cfg,
                "bone_variant",
                infer_bone_variant(dataset),
            ),

            "alpha": first(eff.get("alpha"), case_dim_value(case_cfg, "alpha", None), config_defaults.get("alpha")),
            "beta": first(eff.get("beta"), case_dim_value(case_cfg, "beta", None), config_defaults.get("beta")),
            "gamma": first(eff.get("gamma"), case_dim_value(case_cfg, "gamma", None), config_defaults.get("gamma")),
            "step02_affinity_eps": case_dim_value(
                case_cfg,
                "step02_affinity_eps",
                config_defaults.get("step02_affinity_eps"),
            ),

            "step02_code_version": case_dim_value(
                case_cfg,
                "step02_code_version",
                config_defaults.get("step02_code_version"),
            ),
        }
    )

    if artifact_type == "ssdr_data2":
        return raw

    if artifact_type == "ssdr_data3":
        raw.update(
            {
                "omega_b": first(eff.get("omega_b"), case_dim_value(case_cfg, "omega_b", None), config_defaults.get("omega_b")),
                "qem_config_name": case_dim_value(
                    case_cfg,
                    "qem_config_name",
                    infer_qem_config_name(dataset),
                ),
                "qem_config_fingerprint": case_dim_value(
                    case_cfg,
                    "qem_config_fingerprint",
                    config_defaults.get("qem_config_fingerprint"),
                ),
                "step03_code_version": case_dim_value(
                    case_cfg,
                    "step03_code_version",
                    config_defaults.get("step03_code_version"),
                ),
            }
        )
        return raw

    raise ValueError(f"Unsupported artifact type: {artifact_type}")


def stage_payload_path(run_root: Path, artifact_type: str) -> Path:
    artifact_type = normalize_stage_name(artifact_type)
    return run_root / STAGE_TO_PAYLOAD_DIR[artifact_type]


def iter_case_run_roots(input_root: Path) -> list[Path]:
    roots: list[Path] = []
    for p in sorted(input_root.rglob("case_config.json")):
        if p.parent.is_dir():
            roots.append(p.parent)
    return roots


def load_step04_config_if_any(run_root: Path) -> dict[str, Any]:
    try:
        return rp.load_summary_config(run_root)
    except Exception:
        return {}


def copy_payload(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def ingest_artifacts(
    *,
    pool: Path,
    input_root: Path,
    stages: list[str],
    overwrite: bool = False,
    allow_no_done: bool = False,
    copy_payloads: bool = True,
) -> dict[str, Any]:
    rp.init_pool(pool, force=False)
    ensure_artifact_profiles(pool)
    dimensions_obj, eq_profile, index = rp.require_pool(pool)

    normalized_stages = [normalize_stage_name(s) for s in stages]

    run_roots = iter_case_run_roots(input_root)

    records: list[dict[str, Any]] = []
    ingested = 0
    skipped_existing = 0
    skipped_missing_payload = 0
    skipped_no_done = 0

    for run_root in run_roots:
        done_path = run_root / "done"
        if not done_path.exists() and not allow_no_done:
            skipped_no_done += 1
            continue

        case_cfg_path = run_root / "case_config.json"
        if not case_cfg_path.is_file():
            continue

        try:
            case_cfg = rp.load_json(case_cfg_path)
        except Exception as exc:
            records.append(
                {
                    "run_root": str(run_root),
                    "status": "skip_bad_case_config",
                    "error": repr(exc),
                }
            )
            continue

        step04_config = load_step04_config_if_any(run_root)

        for stage in normalized_stages:
            payload_src = stage_payload_path(run_root, stage)

            if not payload_src.exists():
                skipped_missing_payload += 1
                records.append(
                    {
                        "run_root": str(run_root),
                        "stage": stage,
                        "status": "skip_missing_payload",
                        "payload": str(payload_src),
                    }
                )
                continue

            raw_dims = raw_stage_dims_from_case_config(
                case_cfg,
                artifact_type=stage,
                step04_config=step04_config,
                config_defaults={},
            )

            normalized_dims, traces, fingerprint = rp.normalize_dimensions(
                raw_dims,
                dimensions_obj=dimensions_obj,
                equivalence_profile_obj=eq_profile,
                profile_name=stage,
                context={
                    "source": "ingest-artifacts",
                    "stage": stage,
                    "run_root": str(run_root),
                },
            )

            existing = rp.lookup_artifacts(pool=pool, raw_dims=raw_dims, profile_name=stage)

            if existing and not overwrite:
                skipped_existing += 1
                records.append(
                    {
                        "artifact_id": existing[0]["artifact_id"],
                        "dimension_fingerprint": fingerprint,
                        "run_root": str(run_root),
                        "stage": stage,
                        "status": "skipped_existing",
                    }
                )
                continue

            if existing and overwrite:
                artifact_id = existing[0]["artifact_id"]
            else:
                artifact_id = rp.allocate_artifact_id(index)

            adir = rp.artifact_dir(pool, artifact_id)

            if adir.exists() and overwrite:
                shutil.rmtree(adir)
            adir.mkdir(parents=True, exist_ok=True)

            rp.dump_json(adir / "raw_dimensions.json", {
                "schema": "ogd_lbs_result_pool_raw_dimensions_v1",
                "dimensions": raw_dims,
            })
            rp.dump_json(adir / "canonical_dimensions.json", {
                "schema": "ogd_lbs_result_pool_canonical_dimensions_v1",
                "profile": stage,
                "dimensions": normalized_dims,
                "dimension_fingerprint": fingerprint,
            })
            rp.dump_json(adir / "override_trace.json", traces)
            rp.dump_json(adir / "source_info.json", {
                "source_run_root": str(run_root),
                "source_input_root": str(input_root),
                "source_case_config": str(case_cfg_path),
                "ingested_at_utc": rp.utc_now_iso(),
            })

            shutil.copy2(case_cfg_path, adir / "case_config.json")
            if done_path.exists():
                shutil.copy2(done_path, adir / "done")

            if copy_payloads:
                payload_dst = adir / "payload" / STAGE_TO_PAYLOAD_DIR[stage]
                copy_payload(payload_src, payload_dst)
                payload_rel = str(payload_dst.relative_to(adir))
            else:
                payload_rel = None

            rp.dump_json(adir / "artifact_manifest.json", {
                "schema": "ogd_lbs_result_pool_artifact_manifest_v1",
                "artifact_id": artifact_id,
                "artifact_type": stage,
                "status": "succeeded",
                "dimension_fingerprint": fingerprint,
                "created_at_utc": rp.utc_now_iso(),
                "source_run_root": str(run_root),
                "payload_path": payload_rel,
                "payload_source": str(payload_src),
            })

            index.setdefault("artifacts", {})[artifact_id] = {
                "artifact_id": artifact_id,
                "artifact_type": stage,
                "status": "succeeded",
                "artifact_dir": str(adir.relative_to(pool)),
                "created_at_utc": rp.utc_now_iso(),
                "updated_at_utc": rp.utc_now_iso(),
                "dimension_fingerprint": fingerprint,
                "model": normalized_dims.get("model"),
                "param": case_cfg.get("param"),
                "value": case_cfg.get("value"),
                "payload_path": payload_rel,
            }

            ingested += 1
            records.append(
                {
                    "artifact_id": artifact_id,
                    "dimension_fingerprint": fingerprint,
                    "run_root": str(run_root),
                    "stage": stage,
                    "status": "ingested",
                }
            )

    index["optional_fast_indexes"] = {}
    rp.save_index(pool, index)

    return {
        "schema": "ogd_lbs_result_pool_artifact_ingest_report_v1",
        "pool": str(pool),
        "input_root": str(input_root),
        "stages": normalized_stages,
        "run_root_count": len(run_roots),
        "ingested": ingested,
        "skipped_existing": skipped_existing,
        "skipped_missing_payload": skipped_missing_payload,
        "skipped_no_done": skipped_no_done,
        "records": records,
    }


def lookup_artifact(
    *,
    pool: Path,
    stage: str,
    key_json: Path,
) -> dict[str, Any]:
    rp.init_pool(pool, force=False)
    ensure_artifact_profiles(pool)

    stage = normalize_stage_name(stage)

    obj = rp.load_json(key_json)
    if not isinstance(obj, dict):
        raise ValueError(f"key_json must contain a JSON object: {key_json}")

    if "dimensions" in obj and isinstance(obj["dimensions"], dict):
        raw_dims = dict(obj["dimensions"])
    else:
        raw_dims = dict(obj)

    raw_dims["artifact_type"] = stage

    hits = rp.lookup_artifacts(pool=pool, raw_dims=raw_dims, profile_name=stage)

    return {
        "schema": "ogd_lbs_result_pool_artifact_lookup_report_v1",
        "pool": str(pool),
        "stage": stage,
        "key_json": str(key_json),
        "hit_count": len(hits),
        "hits": hits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage artifact cache helpers for OGD-LBS result pool.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ensure-profiles")
    p.add_argument("--pool", required=True)

    p = sub.add_parser("ingest-artifacts")
    p.add_argument("--pool", required=True)
    p.add_argument("--input_root", required=True)
    p.add_argument("--stages", nargs="+", default=["ssdr_data2", "ssdr_data3"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow_no_done", action="store_true")
    p.add_argument("--no_payload", action="store_true")
    p.add_argument("--report", default=None)

    p = sub.add_parser("lookup-artifact")
    p.add_argument("--pool", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--key_json", required=True)

    p = sub.add_parser("supported-stages")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "supported-stages":
        print(json.dumps({"supported_stages": supported_stages()}, indent=2, ensure_ascii=False))
        return

    if args.cmd == "ensure-profiles":
        pool = Path(args.pool).expanduser().resolve(strict=False)
        rp.init_pool(pool, force=False)
        ensure_artifact_profiles(pool)
        print(f"Ensured artifact profiles in pool: {pool}")
        return

    if args.cmd == "ingest-artifacts":
        pool = Path(args.pool).expanduser().resolve(strict=False)
        input_root = Path(args.input_root).expanduser().resolve(strict=False)

        report = ingest_artifacts(
            pool=pool,
            input_root=input_root,
            stages=list(args.stages),
            overwrite=bool(args.overwrite),
            allow_no_done=bool(args.allow_no_done),
            copy_payloads=not bool(args.no_payload),
        )

        if args.report:
            rp.dump_json(args.report, report)

        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.cmd == "lookup-artifact":
        pool = Path(args.pool).expanduser().resolve(strict=False)
        key_json = Path(args.key_json).expanduser().resolve(strict=False)

        report = lookup_artifact(
            pool=pool,
            stage=str(args.stage),
            key_json=key_json,
        )

        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    raise RuntimeError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
