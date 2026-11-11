#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate OGD-LBS OFAT reports (delta, gamma, s_max, tau, omega_b) from pd-rew-v2 PipelinePool."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
OGD_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_PD_REW_ROOT = Path("/workspace/pd-rew-v2")
DEFAULT_POOL = Path("/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1")
DEFAULT_BINDING = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_repo_binding.yaml"
DEFAULT_TABLE = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_experiment_table.yaml"
DEFAULT_ALIASES = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_parameter_aliases.yaml"
DEFAULT_REPORT_ROOT = Path("/workspace/ogd_reports")

PARAM_CHOICE_PREFIXES = {
    "delta": ("delta-",),
    "gamma": ("weight-",),
    "s_max": ("s_max-",),
    "tau": ("tau-",),
    "omega_b": ("boundary_weight-",),
}

PARAM_TITLES = {
    "delta": "OGD-LBS VDB Offset (δ) Sweep — pd-rew-v2 Result Pool",
    "gamma": "OGD-LBS Weight Allocation (α/β/γ) Sweep — pd-rew-v2 Result Pool",
    "s_max": "OGD-LBS Max Bones per Vertex (s_max) Sweep — pd-rew-v2 Result Pool",
    "tau": "OGD-LBS Weight Pruning Threshold (τ) Sweep — pd-rew-v2 Result Pool",
    "omega_b": "OGD-LBS QEM Boundary Penalty (ω_b) Sweep — pd-rew-v2 Result Pool",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def strict_json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json_object(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(obj, dict), f"JSON root must be object: {path}")
    return obj


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_pool_data_dir(pool: Path, value: Any) -> Path:
    require(isinstance(value, str) and value, f"Invalid pool data_dir: {value!r}")
    rel = PurePosixPath(value)
    require(
        not rel.is_absolute()
        and ".." not in rel.parts
        and str(rel) == value
        and len(rel.parts) == 2
        and rel.parts[0] == "results"
        and rel.parts[1].startswith("r"),
        f"Unsafe or unexpected pool data_dir: {value!r}",
    )
    return pool / rel


def materialize_pool_file(*, datalad_api: Any, pool: Path, path: Path, required: bool) -> bool:
    if not os.path.lexists(path):
        if required:
            raise RuntimeError(f"Required pool report file is not tracked: {path}")
        return False
    datalad_api.get(path=str(path), dataset=str(pool), result_renderer="disabled", on_failure="stop")
    if not path.is_file():
        if required:
            raise RuntimeError(f"Required pool report file is unavailable: {path}")
        return False
    return True


def _row_matches_param_family(row: dict[str, Any], param_family: str) -> bool:
    choices = row.get("sweep_choices") or {}
    w_choice = str(choices.get("weights") or "")
    prefixes = PARAM_CHOICE_PREFIXES[param_family]
    return any(w_choice.startswith(pfx) for pfx in prefixes)


def case_from_expanded_row(
    row: dict[str, Any],
    *,
    param_family: str,
    stage04_data_dir: str,
) -> dict[str, Any]:
    flat = row.get("flat_parameters") or {}
    choices = row.get("sweep_choices") or {}

    model = flat.get("data_prep_handoff.model_name") or flat.get("ssdr_handoff.model_name")
    alpha = flat.get("ssdr_stage02.par_weight", flat.get("stage02.par_weight", 0.5))
    beta = flat.get("ssdr_stage02.bone_weight", flat.get("stage02.bone_weight", 0.3))
    gamma = flat.get("ssdr_stage02.child_weight", flat.get("stage02.child_weight", 0.2))
    omega_b = flat.get("ssdr_stage03.boundary_weight", flat.get("stage03.boundary_weight", 1000.0))
    s_max = flat.get("ssdr_stage04.s_max", flat.get("stage04.s_max", 4))
    mask_mode = flat.get("ssdr_stage04.mask_mode", flat.get("stage04.mask_mode", "adaptive"))
    mask_threshold = flat.get("ssdr_stage04.mask_threshold", flat.get("stage04.mask_threshold", -1.0))
    tau = "adaptive" if str(mask_mode).lower() == "adaptive" else mask_threshold
    delta = flat.get("data_prep_stage02.offset")

    value_by_family = {
        "delta": delta,
        "gamma": gamma,
        "s_max": s_max,
        "tau": tau,
        "omega_b": omega_b,
    }
    swept_value = value_by_family[param_family]
    require(swept_value is not None, f"{row.get('row_id')}: missing value for {param_family}")

    return {
        "schema": "ogd_lbs_ofat_case_v1",
        "model": model,
        "param": param_family,
        "value": swept_value,
        "stage": "data_prep_prep02" if param_family == "delta" else f"ssdr_{param_family}",
        "case_id": row.get("row_id"),
        "effective_params": {
            "delta": delta,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "omega_b": omega_b,
            "s_max": s_max,
            "tau": tau,
        },
        "dataset": {"model": model},
        "pd_rew": {
            "row_id": row.get("row_id"),
            "row_index": row.get("row_index"),
            "sweep_choices": choices,
            "sweep_values": row.get("sweep_values") or {},
            "stage04_data_dir": stage04_data_dir,
        },
    }


def generate_param_report(
    *,
    param_family: str,
    pd_rew_root: Path = DEFAULT_PD_REW_ROOT,
    pool: Path = DEFAULT_POOL,
    binding_path: Path = DEFAULT_BINDING,
    table_path: Path = DEFAULT_TABLE,
    aliases_path: Path = DEFAULT_ALIASES,
    report_root: Path = DEFAULT_REPORT_ROOT,
    output_dir: Path | None = None,
    title: str | None = None,
    require_history: bool = True,
) -> dict[str, Any]:
    src = pd_rew_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import datalad.api as dl
    import aggregate_metrics
    import generate_report
    from add_pdrew_report_downloads import build_downloads
    from pd_rew_v2.experiment import expand_experiment_table, query_cache_status
    from pd_rew_v2.pool import PipelinePool

    pipe = PipelinePool(pool)
    pipe.verify(check_annex_hashes=False)

    expanded = expand_experiment_table(binding_path, table_path)
    final_stage = expanded["stages"][-1]["name"]
    cache_status = query_cache_status(
        binding_path,
        expanded,
        pipe,
        start_stage=final_stage,
        end_stage=final_stage,
    )
    st_sum = cache_status["stage_summaries_by_name"][final_stage]
    require(
        st_sum["missing_rows"] == 0 and st_sum["query_error_rows"] == 0,
        f"{final_stage} cache is incomplete: {st_sum}",
    )

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        output_dir = report_root / f"pdrew_{param_family}_report_{timestamp}"
    else:
        output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compat_root = output_dir / "compat_input"
    aggregate_dir = output_dir / "aggregate"
    html_filename = f"{param_family.capitalize()}_Report.html"
    report_html = output_dir / html_filename
    compat_root.mkdir()
    aggregate_dir.mkdir()

    cache_rows = {r["row_id"]: r for r in cache_status["rows"]}
    selected_rows = [r for r in expanded["rows"] if _row_matches_param_family(r, param_family)]
    require(bool(selected_rows), f"No rows matched parameter family {param_family!r}")

    source_records: list[dict[str, Any]] = []
    for row in selected_rows:
        row_id = row["row_id"]
        st_info = cache_rows[row_id]["stage_statuses"][final_stage]
        require(st_info["status"] == "HIT", f"{row_id}: {final_stage} not HIT")
        data_dir = st_info["data_dir"]
        data4_dir = safe_pool_data_dir(pool, data_dir) / "data4"

        summary_src = data4_dir / "step04_optimize_summary.json"
        history_src = data4_dir / "opt_out/metrics_history_per_frame.npz"
        materialize_pool_file(datalad_api=dl, pool=pool, path=summary_src, required=True)
        history_present = materialize_pool_file(
            datalad_api=dl, pool=pool, path=history_src, required=require_history
        )

        summary = load_json_object(summary_src)
        final_metrics = summary.get("final_metrics") or {}
        require(bool(final_metrics), f"{row_id}: empty final_metrics in {summary_src}")

        case_cfg = case_from_expanded_row(row, param_family=param_family, stage04_data_dir=data_dir)
        model = str(case_cfg["model"])
        val = case_cfg["value"]
        val_slug = str(val).replace(".", "p").replace("-", "m")
        case_root = compat_root / model / f"sweep_{param_family}" / f"{param_family}_{val_slug}"
        case_root.mkdir(parents=True, exist_ok=True)

        strict_json_write(case_root / "case_config.json", case_cfg)
        strict_json_write(case_root / "metrics_04.json", final_metrics)

        summary_dst = case_root / "data4/step04_optimize_summary.json"
        summary_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_src, summary_dst)
        if history_present:
            history_dst = case_root / "data4/opt_out/metrics_history_per_frame.npz"
            history_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(history_src, history_dst)

        source_records.append({
            "row_id": row_id,
            "model": model,
            "param": param_family,
            "value": val,
            "stage04_data_dir": data_dir,
            "summary_sha256": sha256_file(summary_src),
            "history_sha256": sha256_file(history_src) if history_present else None,
            "compatibility_case_root": str(case_root),
        })
        print(f"[materialized] {row_id} model={model} {param_family}={val} -> {data_dir}", flush=True)

    strict_json_write(
        output_dir / "pdrew_source_manifest.json",
        {
            "schema": "ogd_lbs_pdrew_report_source_v2",
            "param_family": param_family,
            "pool": str(pool),
            "binding": str(binding_path),
            "table": str(table_path),
            "row_count": len(source_records),
            "records": source_records,
        },
    )

    agg_manifest = aggregate_metrics.aggregate(
        input_root=compat_root,
        output_dir=aggregate_dir,
        mode="sweep",
        metrics_requested=None,
        group_by=["model", "param", "value"] if param_family == "delta" else ["param", "value"],
        ttest_kind="none",
        baseline_query_text="",
        pair_by=None,
        ttest_group_by=None,
        where_text=f"param={param_family}",
    )

    report_title = title or PARAM_TITLES.get(param_family, f"OGD-LBS {param_family} Sweep Report")
    generate_report.generate_report(
        aggregate_dir=aggregate_dir,
        output_html=report_html,
        mode="sweep",
        title=report_title,
        default_sweep_values=None,
        model_scale_values=None,
        scaled_metrics=None,
        axis_mode=generate_report.DEFAULT_AXIS_MODE,
        pseudo_log_ratio_threshold=generate_report.DEFAULT_PSEUDO_LOG_RATIO_THRESHOLD,
    )

    strict_json_write(
        output_dir / "report_manifest.json",
        {
            "schema": "ogd_lbs_pdrew_report_v2",
            "status": "GENERATED",
            "param_family": param_family,
            "pool": str(pool),
            "row_count": len(source_records),
            "aggregate_dir": str(aggregate_dir),
            "report_html": str(report_html),
        },
    )
    shutil.rmtree(compat_root, ignore_errors=True)

    dl_receipt = build_downloads(output_dir, html_filename=html_filename, title=report_title)
    return {
        "param_family": param_family,
        "row_count": len(source_records),
        "output_dir": str(output_dir),
        "report_html": str(report_html),
        "index_html": dl_receipt["index_html"],
        "zip_path": dl_receipt["zip_path"],
        "aggregate_manifest": agg_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OGD-LBS OFAT report from pd-rew-v2 pool.")
    parser.add_argument(
        "-p", "--param",
        choices=["delta", "gamma", "s_max", "tau", "omega_b"],
        default="delta",
        help="Parameter family to generate report for (default: delta).",
    )
    parser.add_argument("--pd-rew-root", type=Path, default=DEFAULT_PD_REW_ROOT)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--require-history", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    res = generate_param_report(
        param_family=args.param,
        pd_rew_root=args.pd_rew_root.expanduser().resolve(),
        pool=args.pool.expanduser().resolve(),
        binding_path=args.binding.expanduser().resolve(),
        table_path=args.table.expanduser().resolve(),
        aliases_path=args.aliases.expanduser().resolve(),
        report_root=args.report_root.expanduser().resolve(strict=False),
        output_dir=args.output_dir,
        title=args.title,
        require_history=bool(args.require_history),
    )
    print("=" * 88)
    print(f"PD-REW-V2 REPORT GENERATED ({res['param_family'].upper()})")
    print("ROWS        :", res["row_count"])
    print("REPORT HTML :", res["report_html"])
    print("INDEX HTML  :", res["index_html"])
    print("DOWNLOAD ZIP:", res["zip_path"])
    print("=" * 88)


if __name__ == "__main__":
    main()
