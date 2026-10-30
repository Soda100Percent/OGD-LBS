#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import importlib
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


SSDR_DATA4_DIMENSIONS = [
    "artifact_type",
    "experiment_family",
    "model",
    "param",
    "value",

    "bone_variant",
    "frame_sampling_policy",
    "max_frames",
    "target_frames",

    "alpha",
    "beta",
    "gamma",
    "step02_affinity_eps",

    "omega_b",
    "qem_config_name",
    "qem_config_fingerprint",

    "s_max",
    "tau",
    "step04_iter",
    "step04_org_type",
    "step04_opt_weight",
    "step04_input_poses",
    "step04_use_weight_mask",
    "step04_mask_mode",
    "step04_mask_threshold",
    "step04_weight_solver",
    "step04_bt_covariance_mode",
    "step04_use_weight_smoothness",
    "step04_weights_smooth_step",
    "step04_weight_soft_lock",
    "step04_lambda_reg",
    "step04_trans_affine",
    "step04_trans_affine_norm",
    "step04_roll_affinity",
    "step04_roll_affinity_norm",

    "step01_code_version",
    "step02_code_version",
    "step03_code_version",
    "step04_code_version",
]

SSDR_DATA4_DEFAULTS = {
    "artifact_type": "ssdr_data4",
    "experiment_family": "ofat",

    "bone_variant": "default",
    "frame_sampling_policy": "target_200",
    "max_frames": 210,
    "target_frames": 200,

    "alpha": 0.5,
    "beta": 0.3,
    "gamma": 0.2,
    "step02_affinity_eps": 1e-5,

    "omega_b": 1000.0,
    "qem_config_name": "default",
    "qem_config_fingerprint": None,

    "s_max": 4,
    "tau": "adaptive",
    "step04_iter": 70,
    "step04_org_type": 3,
    "step04_opt_weight": True,
    "step04_input_poses": True,
    "step04_use_weight_mask": True,
    "step04_mask_mode": "adaptive",
    "step04_mask_threshold": -1.0,
    "step04_weight_solver": "simplex",
    "step04_bt_covariance_mode": "corrected",
    "step04_use_weight_smoothness": False,
    "step04_weights_smooth_step": 1.0,
    "step04_weight_soft_lock": 0.0,
    "step04_lambda_reg": 0.1,
    "step04_trans_affine": 0.0,
    "step04_trans_affine_norm": 4.0,
    "step04_roll_affinity": 0.0,
    "step04_roll_affinity_norm": 2.0,

    "step01_code_version": "step01_v1",
    "step02_code_version": "step02_v1",
    "step03_code_version": "step03_v1",
    "step04_code_version": "step04_v1",
}

DEFAULT_PROFILES = {
    "ssdr_data4": {
        "profile_version": 2,
        "dimensions": list(SSDR_DATA4_DIMENSIONS),
        "defaults": dict(SSDR_DATA4_DEFAULTS),
    }
}

DEFAULT_EQUIVALENCE_ITEMS = [
    {
        "name": "normalize_disabled_weight_mask",
        "module": "result_pool_rules.ssdr_ofat_v1",
        "function": "normalize_disabled_weight_mask",
        "version": "1",
    },
    {
        "name": "normalize_adaptive_tau",
        "module": "result_pool_rules.ssdr_ofat_v1",
        "function": "normalize_adaptive_tau",
        "version": "1",
    },
    {
        "name": "normalize_disabled_weight_smoothness",
        "module": "result_pool_rules.ssdr_ofat_v1",
        "function": "normalize_disabled_weight_smoothness",
        "version": "1",
    },
    {
        "name": "normalize_zero_trans_affine",
        "module": "result_pool_rules.ssdr_ofat_v1",
        "function": "normalize_zero_trans_affine",
        "version": "1",
    },
    {
        "name": "normalize_zero_roll_affinity",
        "module": "result_pool_rules.ssdr_ofat_v1",
        "function": "normalize_zero_roll_affinity",
        "version": "1",
    },
]


def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=4, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    return p


def json_safe(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(
        json_safe(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def get_pool_paths(pool: Path) -> dict[str, Path]:
    return {
        "pool": pool,
        "manifest": pool / "pool_manifest.json",
        "dimensions": pool / "dimensions.json",
        "profile": pool / "equivalence_profile.json",
        "index": pool / "index.json",
        "artifacts": pool / "artifacts",
    }


def profile_fingerprint(profile: dict[str, Any]) -> str:
    return sha256_obj(
        {
            "profile_version": profile.get("profile_version"),
            "dimensions": profile.get("dimensions", []),
            "defaults": profile.get("defaults", {}),
        }
    )


def init_pool(pool: Path, *, force: bool = False) -> None:
    paths = get_pool_paths(pool)

    if pool.exists() and force:
        shutil.rmtree(pool)

    paths["artifacts"].mkdir(parents=True, exist_ok=True)

    if not paths["manifest"].exists():
        dump_json(
            paths["manifest"],
            {
                "schema": "ogd_lbs_result_pool_manifest_v2",
                "created_at_utc": utc_now_iso(),
                "description": "OGD-LBS local experiment result/artifact pool using flat dimension matching.",
            },
        )

    if not paths["dimensions"].exists():
        dump_json(
            paths["dimensions"],
            {
                "schema": "ogd_lbs_result_pool_dimensions_v2",
                "profiles": copy.deepcopy(DEFAULT_PROFILES),
            },
        )
    else:
        maybe_migrate_dimensions_to_v2(paths["dimensions"])

    if not paths["profile"].exists():
        dump_json(
            paths["profile"],
            {
                "schema": "ogd_lbs_result_pool_equivalence_profile_v1",
                "profile_name": "ssdr_ofat_v1",
                "items": list(DEFAULT_EQUIVALENCE_ITEMS),
            },
        )

    if not paths["index"].exists():
        dump_json(
            paths["index"],
            {
                "schema": "ogd_lbs_result_pool_index_v2",
                "updated_at_utc": utc_now_iso(),
                "next_artifact_id": 1,
                "artifacts": {},
                "optional_fast_indexes": {},
            },
        )
    else:
        maybe_migrate_index_to_v2(paths["index"])


def maybe_migrate_dimensions_to_v2(dimensions_path: Path) -> None:
    obj = load_json(dimensions_path)
    if not isinstance(obj, dict):
        return

    if obj.get("schema") == "ogd_lbs_result_pool_dimensions_v2" and "profiles" in obj:
        return

    # Best-effort conversion from v1 single dimension list to v2 ssdr_data4 profile.
    dims = obj.get("dimensions", list(SSDR_DATA4_DIMENSIONS))
    defaults = dict(SSDR_DATA4_DEFAULTS)
    defaults.update(obj.get("defaults", {}))

    converted = {
        "schema": "ogd_lbs_result_pool_dimensions_v2",
        "profiles": {
            "ssdr_data4": {
                "profile_version": 2,
                "dimensions": list(dims),
                "defaults": defaults,
            }
        },
        "migrated_from": obj.get("schema", "unknown"),
        "migrated_at_utc": utc_now_iso(),
    }
    dump_json(dimensions_path, converted)


def maybe_migrate_index_to_v2(index_path: Path) -> None:
    obj = load_json(index_path)
    if not isinstance(obj, dict):
        return

    if obj.get("schema") == "ogd_lbs_result_pool_index_v2" and "artifacts" in obj:
        return

    # Keep existing content but move old cases under artifacts if present.
    artifacts = {}
    for i, (old_key, rec) in enumerate(sorted((obj.get("cases") or {}).items()), start=1):
        artifact_id = f"a{i:06d}"
        artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "legacy_key": old_key,
            "artifact_type": "ssdr_data4",
            "status": rec.get("status", "succeeded"),
            "artifact_dir": rec.get("case_dir", ""),
            "created_at_utc": rec.get("updated_at_utc", utc_now_iso()),
            "dimension_fingerprint": old_key,
            "migrated_from_v1": True,
        }

    converted = {
        "schema": "ogd_lbs_result_pool_index_v2",
        "updated_at_utc": utc_now_iso(),
        "next_artifact_id": len(artifacts) + 1,
        "artifacts": artifacts,
        "optional_fast_indexes": {},
        "migrated_from": obj.get("schema", "unknown"),
    }
    dump_json(index_path, converted)


def require_pool(pool: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = get_pool_paths(pool)
    if not paths["dimensions"].is_file() or not paths["profile"].is_file() or not paths["index"].is_file():
        raise FileNotFoundError(
            f"Result pool is not initialized: {pool}\n"
            f"Run first: python scripts/experiments/result_pool.py init --pool {pool}"
        )

    maybe_migrate_dimensions_to_v2(paths["dimensions"])
    maybe_migrate_index_to_v2(paths["index"])

    dims = load_json(paths["dimensions"])
    profile = load_json(paths["profile"])
    index = load_json(paths["index"])
    return dims, profile, index


def save_index(pool: Path, index: dict[str, Any]) -> None:
    index["updated_at_utc"] = utc_now_iso()
    dump_json(get_pool_paths(pool)["index"], index)


def get_dimension_profile(dimensions_obj: dict[str, Any], profile_name: str = "ssdr_data4") -> dict[str, Any]:
    profiles = dimensions_obj.get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"Dimension profile not found: {profile_name}")
    return profiles[profile_name]


def normalize_dimensions(
    raw_dims: dict[str, Any],
    *,
    dimensions_obj: dict[str, Any],
    equivalence_profile_obj: dict[str, Any],
    profile_name: str = "ssdr_data4",
    context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    profile = get_dimension_profile(dimensions_obj, profile_name)
    dim_names = list(profile.get("dimensions", []))
    defaults = dict(profile.get("defaults", {}))

    out: dict[str, Any] = {}
    for key in dim_names:
        if key in raw_dims and raw_dims[key] is not None:
            out[key] = raw_dims[key]
        else:
            out[key] = defaults.get(key)

    traces: list[dict[str, Any]] = []
    ctx = dict(context or {})
    ctx.setdefault("dimension_profile", profile_name)

    for item in equivalence_profile_obj.get("items", []):
        module_name = str(item["module"])
        func_name = str(item["function"])

        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        new_out, trace = func(out, context=ctx)
        trace = dict(trace)
        trace.setdefault("name", item.get("name"))
        trace.setdefault("version", item.get("version"))
        trace.setdefault("module", module_name)
        trace.setdefault("function", func_name)

        # Keep only profile dimensions even if a rule adds temporary keys.
        out = {k: new_out.get(k) for k in dim_names}
        traces.append(trace)

    fingerprint = sha256_obj(
        {
            "schema": "ogd_lbs_dimension_fingerprint_v1",
            "profile": profile_name,
            "normalized_dimensions": out,
        }
    )
    return out, traces, fingerprint


def dimensions_equal(
    query_raw: dict[str, Any],
    record_raw: dict[str, Any],
    *,
    dimensions_obj: dict[str, Any],
    equivalence_profile_obj: dict[str, Any],
    profile_name: str = "ssdr_data4",
) -> bool:
    q, _qt, _qh = normalize_dimensions(
        query_raw,
        dimensions_obj=dimensions_obj,
        equivalence_profile_obj=equivalence_profile_obj,
        profile_name=profile_name,
        context={"source": "query"},
    )
    r, _rt, _rh = normalize_dimensions(
        record_raw,
        dimensions_obj=dimensions_obj,
        equivalence_profile_obj=equivalence_profile_obj,
        profile_name=profile_name,
        context={"source": "record"},
    )
    return q == r


def allocate_artifact_id(index: dict[str, Any]) -> str:
    n = int(index.get("next_artifact_id", 1))
    index["next_artifact_id"] = n + 1
    return f"a{n:06d}"


def artifact_dir(pool: Path, artifact_id: str) -> Path:
    return get_pool_paths(pool)["artifacts"] / artifact_id


def load_artifact_raw_dimensions(pool: Path, artifact_id: str, rec: dict[str, Any]) -> dict[str, Any] | None:
    adir = artifact_dir(pool, artifact_id)

    candidates = [
        adir / "raw_dimensions.json",
        adir / "canonical_dimensions.json",
        adir / "raw_key.json",
        adir / "canonical_key.json",
    ]

    for p in candidates:
        if not p.is_file():
            continue
        try:
            obj = load_json(p)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        if "dimensions" in obj and isinstance(obj["dimensions"], dict):
            return dict(obj["dimensions"])

        if "canonical_dimensions" in obj and isinstance(obj["canonical_dimensions"], dict):
            return dict(obj["canonical_dimensions"])

        if "raw_dimensions" in obj and isinstance(obj["raw_dimensions"], dict):
            return dict(obj["raw_dimensions"])

    # v1 migrated artifact may have missing files. No match by default.
    return None


def lookup_artifacts(
    *,
    pool: Path,
    raw_dims: dict[str, Any],
    profile_name: str = "ssdr_data4",
) -> list[dict[str, Any]]:
    init_pool(pool, force=False)
    dimensions_obj, eq_profile, index = require_pool(pool)

    matches: list[dict[str, Any]] = []

    for artifact_id, rec in sorted(index.get("artifacts", {}).items()):
        if rec.get("status") != "succeeded":
            continue

        artifact_type = rec.get("artifact_type", "ssdr_data4")
        if artifact_type != profile_name:
            continue

        record_raw = load_artifact_raw_dimensions(pool, artifact_id, rec)
        if record_raw is None:
            continue

        if dimensions_equal(
            raw_dims,
            record_raw,
            dimensions_obj=dimensions_obj,
            equivalence_profile_obj=eq_profile,
            profile_name=profile_name,
        ):
            matches.append(
                {
                    "artifact_id": artifact_id,
                    "record": rec,
                    "artifact_dir": str(artifact_dir(pool, artifact_id)),
                }
            )

    return matches


def find_step04_summary(run_root: Path) -> Path | None:
    candidates = [
        run_root / "data4" / "step04_optimize_summary.json",
        run_root / "data4_test" / "step04_optimize_summary.json",
    ]
    candidates.extend(sorted(run_root.rglob("step04_optimize_summary.json")))

    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            return p
    return None


def load_summary_config(run_root: Path) -> dict[str, Any]:
    p = find_step04_summary(run_root)
    if not p:
        return {}
    obj = load_json(p)
    if not isinstance(obj, dict):
        return {}
    cfg = obj.get("config", {})
    return cfg if isinstance(cfg, dict) else {}


def infer_qem_config_name(dataset: dict[str, Any] | None) -> str:
    dataset = dataset or {}
    qem = dataset.get("qem_config")
    if qem is None or str(qem).strip() == "":
        return "default"
    return Path(str(qem)).name


def raw_dims_from_case_config(
    case_cfg: dict[str, Any],
    *,
    step04_config: dict[str, Any] | None = None,
    config_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step04_config = dict(step04_config or {})
    config_defaults = dict(config_defaults or {})
    dataset = dict(case_cfg.get("dataset") or {})
    eff = dict(case_cfg.get("effective_params") or {})

    def first(*values: Any) -> Any:
        for v in values:
            if v is not None:
                return v
        return None

    tau = first(eff.get("tau"), config_defaults.get("tau"))

    raw: dict[str, Any] = {
        "artifact_type": "ssdr_data4",
        "experiment_family": "ofat",
        "model": case_cfg.get("model"),
        "param": case_cfg.get("param"),
        "value": case_cfg.get("value"),

        "bone_variant": dataset.get("variant") or "default",
        "frame_sampling_policy": "target_200",
        "max_frames": config_defaults.get("max_frames"),
        "target_frames": config_defaults.get("target_frames"),

        "alpha": first(eff.get("alpha"), config_defaults.get("alpha")),
        "beta": first(eff.get("beta"), config_defaults.get("beta")),
        "gamma": first(eff.get("gamma"), config_defaults.get("gamma")),
        "step02_affinity_eps": config_defaults.get("step02_affinity_eps"),

        "omega_b": first(eff.get("omega_b"), config_defaults.get("omega_b")),
        "qem_config_name": infer_qem_config_name(dataset),
        "qem_config_fingerprint": None,

        "s_max": first(eff.get("s_max"), step04_config.get("s_max"), config_defaults.get("s_max")),
        "tau": tau,

        "step04_iter": step04_config.get("iter"),
        "step04_org_type": step04_config.get("org_type"),
        "step04_opt_weight": step04_config.get("opt_weight"),
        "step04_input_poses": step04_config.get("input_poses"),
        "step04_use_weight_mask": step04_config.get("use_weight_mask"),
        "step04_mask_mode": step04_config.get("mask_mode"),
        "step04_mask_threshold": step04_config.get("mask_threshold"),
        "step04_weight_solver": step04_config.get("weight_solver"),
        "step04_bt_covariance_mode": step04_config.get("bt_covariance_mode"),
        "step04_use_weight_smoothness": step04_config.get("use_weight_smoothness"),
        "step04_weights_smooth_step": step04_config.get("weights_smooth_step"),
        "step04_weight_soft_lock": step04_config.get("weight_soft_lock"),
        "step04_lambda_reg": step04_config.get("lambda_reg"),
        "step04_trans_affine": step04_config.get("trans_affine"),
        "step04_trans_affine_norm": step04_config.get("trans_affine_norm"),
        "step04_roll_affinity": step04_config.get("roll_affinity"),
        "step04_roll_affinity_norm": step04_config.get("roll_affinity_norm"),
    }

    if raw.get("step04_mask_mode") is None:
        if str(tau).strip().lower() == "adaptive":
            raw["step04_mask_mode"] = "adaptive"
        elif tau is not None:
            raw["step04_mask_mode"] = "fixed"

    if raw.get("step04_mask_threshold") is None:
        if str(tau).strip().lower() == "adaptive":
            raw["step04_mask_threshold"] = -1.0
        elif tau is not None:
            try:
                raw["step04_mask_threshold"] = float(tau)
            except Exception:
                pass

    return raw


def build_raw_dims_from_rps_case(case: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    case_cfg = {
        "schema": case.get("schema", "ogd_lbs_ofat_case_v1"),
        "model": case.get("model"),
        "param": case.get("param"),
        "value": case.get("value"),
        "stage": case.get("stage"),
        "case_id": case.get("case_id"),
        "effective_params": case.get("effective_params", {}),
        "dataset": case.get("dataset", {}),
    }

    step04_config = dict(cfg.get("step04_defaults") or {})
    defaults = dict(cfg.get("defaults") or {})

    ssdr_defaults = dict(cfg.get("ssdr_defaults") or {})
    defaults.setdefault("max_frames", ssdr_defaults.get("max_frames", 210))
    defaults.setdefault("target_frames", ssdr_defaults.get("target_frames", 200))
    defaults.setdefault("step02_affinity_eps", 1e-5)

    return raw_dims_from_case_config(
        case_cfg,
        step04_config=step04_config,
        config_defaults=defaults,
    )


def iter_ofat_run_roots(input_root: Path) -> list[Path]:
    roots: list[Path] = []
    for p in sorted(input_root.rglob("case_config.json")):
        run_root = p.parent
        if (run_root / "metrics_04.json").is_file():
            roots.append(run_root)
    return roots


def ingest_ofat(
    *,
    pool: Path,
    input_root: Path,
    overwrite: bool = False,
    allow_no_done: bool = False,
) -> dict[str, Any]:
    init_pool(pool, force=False)
    dimensions_obj, eq_profile, index = require_pool(pool)

    run_roots = iter_ofat_run_roots(input_root)

    ingested = 0
    skipped_existing = 0
    skipped_no_done = 0
    records: list[dict[str, Any]] = []

    for run_root in run_roots:
        done_path = run_root / "done"
        if not done_path.exists() and not allow_no_done:
            skipped_no_done += 1
            continue

        case_cfg_path = run_root / "case_config.json"
        metrics_path = run_root / "metrics_04.json"

        case_cfg = load_json(case_cfg_path)
        step04_config = load_summary_config(run_root)
        raw_dims = raw_dims_from_case_config(case_cfg, step04_config=step04_config)

        normalized_dims, traces, fingerprint = normalize_dimensions(
            raw_dims,
            dimensions_obj=dimensions_obj,
            equivalence_profile_obj=eq_profile,
            profile_name="ssdr_data4",
            context={
                "source": "ingest-ofat",
                "run_root": str(run_root),
            },
        )

        existing = lookup_artifacts(pool=pool, raw_dims=raw_dims, profile_name="ssdr_data4")

        if existing and not overwrite:
            skipped_existing += 1
            records.append(
                {
                    "artifact_id": existing[0]["artifact_id"],
                    "dimension_fingerprint": fingerprint,
                    "run_root": str(run_root),
                    "status": "skipped_existing",
                }
            )
            continue

        if existing and overwrite:
            artifact_id = existing[0]["artifact_id"]
        else:
            artifact_id = allocate_artifact_id(index)

        adir = artifact_dir(pool, artifact_id)
        if adir.exists() and overwrite:
            shutil.rmtree(adir)
        adir.mkdir(parents=True, exist_ok=True)

        dump_json(adir / "raw_dimensions.json", {
            "schema": "ogd_lbs_result_pool_raw_dimensions_v1",
            "dimensions": raw_dims,
        })
        dump_json(adir / "canonical_dimensions.json", {
            "schema": "ogd_lbs_result_pool_canonical_dimensions_v1",
            "profile": "ssdr_data4",
            "dimensions": normalized_dims,
            "dimension_fingerprint": fingerprint,
        })
        dump_json(adir / "override_trace.json", traces)
        dump_json(adir / "source_info.json", {
            "source_run_root": str(run_root),
            "source_input_root": str(input_root),
            "ingested_at_utc": utc_now_iso(),
        })

        shutil.copy2(case_cfg_path, adir / "case_config.json")
        shutil.copy2(metrics_path, adir / "metrics_04.json")

        summary_path = find_step04_summary(run_root)
        if summary_path:
            shutil.copy2(summary_path, adir / "step04_optimize_summary.json")

        if done_path.exists():
            shutil.copy2(done_path, adir / "done")

        dump_json(adir / "artifact_manifest.json", {
            "schema": "ogd_lbs_result_pool_artifact_manifest_v1",
            "artifact_id": artifact_id,
            "artifact_type": "ssdr_data4",
            "status": "succeeded",
            "dimension_fingerprint": fingerprint,
            "created_at_utc": utc_now_iso(),
            "source_run_root": str(run_root),
        })

        index.setdefault("artifacts", {})[artifact_id] = {
            "artifact_id": artifact_id,
            "artifact_type": "ssdr_data4",
            "status": "succeeded",
            "artifact_dir": str(adir.relative_to(pool)),
            "created_at_utc": utc_now_iso(),
            "updated_at_utc": utc_now_iso(),
            "dimension_fingerprint": fingerprint,
            "model": normalized_dims.get("model"),
            "param": normalized_dims.get("param"),
            "value": normalized_dims.get("value"),
        }

        ingested += 1
        records.append(
            {
                "artifact_id": artifact_id,
                "dimension_fingerprint": fingerprint,
                "run_root": str(run_root),
                "status": "ingested",
            }
        )

    # Any ingest invalidates optional indexes because pool contents changed.
    index["optional_fast_indexes"] = {}
    save_index(pool, index)

    return {
        "schema": "ogd_lbs_result_pool_ingest_report_v2",
        "pool": str(pool),
        "input_root": str(input_root),
        "run_root_count": len(run_roots),
        "ingested": ingested,
        "skipped_existing": skipped_existing,
        "skipped_no_done": skipped_no_done,
        "records": records,
    }


def make_single_case_config(cfg: dict[str, Any], case: dict[str, Any], output_root: str | None) -> dict[str, Any]:
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


def plan_ofat(
    *,
    pool: Path,
    config_path: Path,
    output_manifest: Path,
    output_plan_dir: Path,
    output_root: str | None,
    only_models: list[str] | None,
    only_params: list[str] | None,
    no_conda: bool,
) -> dict[str, Any]:
    import run_param_sweep as rps  # type: ignore

    init_pool(pool, force=False)
    dimensions_obj, eq_profile, _index = require_pool(pool)

    cfg_raw = rps.load_json(config_path)
    cfg = rps.normalize_config(cfg_raw, output_root_override=output_root)

    datasets = rps.selected_datasets(cfg, only_models)
    params = rps.selected_params(cfg, only_params)
    cases = rps.iter_cases(cfg, datasets=datasets, params=params)

    output_plan_dir.mkdir(parents=True, exist_ok=True)
    single_cfg_dir = output_plan_dir / "single_case_configs"
    single_cfg_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_plan_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    hit_records: list[dict[str, Any]] = []
    miss_cases: list[dict[str, Any]] = []

    for idx, case in enumerate(cases):
        raw_dims = build_raw_dims_from_rps_case(case, cfg)
        normalized_dims, traces, fingerprint = normalize_dimensions(
            raw_dims,
            dimensions_obj=dimensions_obj,
            equivalence_profile_obj=eq_profile,
            profile_name="ssdr_data4",
            context={
                "source": "plan-ofat",
                "config": str(config_path),
                "case_index": idx,
            },
        )

        record_base = {
            "source_case_index": idx,
            "model": case.get("model"),
            "param": case.get("param"),
            "value": case.get("value"),
            "case_id": case.get("case_id"),
            "dimension_fingerprint": fingerprint,
            "raw_dimensions": raw_dims,
            "canonical_dimensions": normalized_dims,
            "override_trace": traces,
        }

        hits = lookup_artifacts(pool=pool, raw_dims=raw_dims, profile_name="ssdr_data4")

        if hits:
            rec = dict(record_base)
            rec["pool_hit"] = True
            rec["pool_artifacts"] = hits
            hit_records.append(rec)
            continue

        safe = f"case_{len(miss_cases):05d}_{case['model']}_{case['param']}_{rps.slugify(case['value'])}"
        single_cfg_path = single_cfg_dir / f"{safe}.json"
        single_cfg = make_single_case_config(cfg, case, output_root)
        dump_json(single_cfg_path, single_cfg)

        cmd = [
            sys.executable,
            "scripts/experiments/run_param_sweep.py",
            "--config",
            str(single_cfg_path),
            "--only_models",
            str(case["model"]),
            "--only_params",
            str(case["param"]),
        ]
        if no_conda:
            cmd.append("--no_conda")

        rec = dict(record_base)
        rec.update(
            {
                "pool_hit": False,
                "manifest_case_index": len(miss_cases),
                "single_case_config": str(single_cfg_path),
                "command": cmd,
                "log_path": str(logs_dir / f"{safe}.log"),
            }
        )
        miss_cases.append(rec)

    manifest = {
        "schema": "ogd_lbs_case_manifest_v1",
        "mode": "ofat",
        "source_config": str(config_path),
        "output_root": output_root or cfg.get("output_root"),
        "pool": {
            "path": str(pool),
            "candidate_count": len(cases),
            "hit_count": len(hit_records),
            "miss_count": len(miss_cases),
            "match_mode": "flat_dimensions_linear_scan",
        },
        "case_count": len(miss_cases),
        "cases": miss_cases,
    }
    dump_json(output_manifest, manifest)

    report = {
        "schema": "ogd_lbs_result_pool_filter_report_v2",
        "source_config": str(config_path),
        "pool": str(pool),
        "candidate_count": len(cases),
        "hit_count": len(hit_records),
        "miss_count": len(miss_cases),
        "match_mode": "flat_dimensions_linear_scan",
        "hits": hit_records,
        "manifest": str(output_manifest),
        "plan_dir": str(output_plan_dir),
    }
    dump_json(output_plan_dir / "filter_report.json", report)

    return report


def add_dimension(pool: Path, *, profile_name: str, name: str, default: Any) -> dict[str, Any]:
    init_pool(pool, force=False)
    dimensions_obj, _profile, index = require_pool(pool)
    paths = get_pool_paths(pool)

    profiles = dimensions_obj.setdefault("profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"Unknown dimension profile: {profile_name}")

    prof = profiles[profile_name]
    dims = list(prof.get("dimensions", []))
    defaults = dict(prof.get("defaults", {}))

    changed = False

    if name not in dims:
        dims.append(name)
        changed = True

    if defaults.get(name) != default:
        defaults[name] = default
        changed = True

    prof["dimensions"] = dims
    prof["defaults"] = defaults
    prof["profile_version"] = int(prof.get("profile_version", 1)) + (1 if changed else 0)

    dimensions_obj["profiles"] = profiles
    dump_json(paths["dimensions"], dimensions_obj)

    # Dimensions changed; optional fast indexes are no longer trusted.
    index["optional_fast_indexes"] = {}
    save_index(pool, index)

    return {
        "schema": "ogd_lbs_result_pool_add_dimension_report_v2",
        "pool": str(pool),
        "profile": profile_name,
        "dimension": name,
        "default": default,
        "changed": changed,
        "note": "Existing artifact files are not rewritten; missing dimensions are filled from defaults at query time.",
    }


def stats(pool: Path) -> dict[str, Any]:
    init_pool(pool, force=False)
    dimensions_obj, eq_profile, index = require_pool(pool)

    artifacts = index.get("artifacts", {})
    by_model: dict[str, int] = {}
    by_param: dict[str, int] = {}
    by_type: dict[str, int] = {}

    for artifact_id, rec in artifacts.items():
        atype = str(rec.get("artifact_type", "<unknown>"))
        model = str(rec.get("model", "<unknown>"))
        param = str(rec.get("param", "<unknown>"))

        by_type[atype] = by_type.get(atype, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
        by_param[param] = by_param.get(param, 0) + 1

    return {
        "schema": "ogd_lbs_result_pool_stats_v2",
        "pool": str(pool),
        "artifact_count": len(artifacts),
        "by_artifact_type": dict(sorted(by_type.items())),
        "by_model": dict(sorted(by_model.items())),
        "by_param": dict(sorted(by_param.items())),
        "profiles": {
            name: {
                "profile_version": prof.get("profile_version"),
                "dimension_count": len(prof.get("dimensions", [])),
                "profile_fingerprint": profile_fingerprint(prof),
            }
            for name, prof in (dimensions_obj.get("profiles") or {}).items()
        },
        "equivalence_profile": eq_profile.get("profile_name", "unknown"),
        "match_mode": "flat_dimensions_linear_scan",
    }


def parse_value(text: str) -> Any:
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except Exception:
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OGD-LBS local experiment result pool.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--pool", required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("ingest-ofat")
    p.add_argument("--pool", required=True)
    p.add_argument("--input_root", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow_no_done", action="store_true")
    p.add_argument("--report", default=None)

    p = sub.add_parser("plan-ofat")
    p.add_argument("--pool", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output_manifest", required=True)
    p.add_argument("--output_plan_dir", required=True)
    p.add_argument("--output_root", default=None)
    p.add_argument("--only_models", nargs="*", default=None)
    p.add_argument("--only_params", nargs="*", default=None)
    p.add_argument("--no_conda", action="store_true")

    p = sub.add_parser("stats")
    p.add_argument("--pool", required=True)

    p = sub.add_parser("add-dimension")
    p.add_argument("--pool", required=True)
    p.add_argument("--profile", default="ssdr_data4")
    p.add_argument("--name", required=True)
    p.add_argument("--default", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = Path(args.pool).expanduser().resolve(strict=False)

    if args.cmd == "init":
        init_pool(pool, force=bool(args.force))
        print(f"Initialized result pool: {pool}")

    elif args.cmd == "ingest-ofat":
        report = ingest_ofat(
            pool=pool,
            input_root=Path(args.input_root).expanduser().resolve(strict=False),
            overwrite=bool(args.overwrite),
            allow_no_done=bool(args.allow_no_done),
        )
        if args.report:
            dump_json(args.report, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))

    elif args.cmd == "plan-ofat":
        report = plan_ofat(
            pool=pool,
            config_path=Path(args.config).expanduser().resolve(strict=False),
            output_manifest=Path(args.output_manifest).expanduser().resolve(strict=False),
            output_plan_dir=Path(args.output_plan_dir).expanduser().resolve(strict=False),
            output_root=args.output_root,
            only_models=args.only_models,
            only_params=args.only_params,
            no_conda=bool(args.no_conda),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))

    elif args.cmd == "stats":
        print(json.dumps(stats(pool), indent=2, ensure_ascii=False))

    elif args.cmd == "add-dimension":
        report = add_dimension(
            pool=pool,
            profile_name=str(args.profile),
            name=str(args.name),
            default=parse_value(str(args.default)),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))

    else:
        raise RuntimeError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
