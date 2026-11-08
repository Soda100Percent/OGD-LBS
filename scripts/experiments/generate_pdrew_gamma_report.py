#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate an OGD-LBS gamma report from a pd-rew pipeline result pool.

Temporary compatibility adapter:

    pd-rew experiment table / pipeline pool
        -> legacy-compatible sweep input tree
        -> aggregate_metrics.py
        -> generate_report.py

The source pool index and payloads are not modified. DataLad may materialize
annexed report files into the existing pool working tree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any


HERE = Path(__file__).resolve().parent
OGD_ROOT = HERE.parents[1]

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


DEFAULT_PD_REW_ROOT = Path("/workspace/pd-rew")
DEFAULT_POOL = Path("/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1")
DEFAULT_BINDING = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_repo_binding.yaml"
DEFAULT_TABLE = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_experiment_table.yaml"
DEFAULT_ALIASES = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_parameter_aliases.yaml"
DEFAULT_REPORT_ROOT = Path("/workspace/ogd_reports")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def strict_json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    path.write_text(data, encoding="utf-8")


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


def add_pd_rew_source(pd_rew_root: Path) -> None:
    src = pd_rew_root / "src"
    require(src.is_dir(), f"pd-rew src directory missing: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def materialize_pool_file(
    *,
    datalad_api: Any,
    pool: Path,
    path: Path,
    required: bool,
) -> bool:
    """Materialize one tracked file without exporting the complete data4 tree."""
    if not os.path.lexists(path):
        if required:
            raise RuntimeError(f"Required pool report file is not tracked: {path}")
        return False

    datalad_api.get(
        path=str(path),
        dataset=str(pool),
        result_renderer="disabled",
        on_failure="stop",
    )

    if not path.is_file():
        if required:
            raise RuntimeError(f"Required pool report file is unavailable: {path}")
        return False

    return True


def gamma_case_from_expanded_row(
    row: dict[str, Any],
    *,
    stage04_data_dir: str,
) -> dict[str, Any]:
    flat = row.get("flat_parameters") or {}
    choices = row.get("sweep_choices") or {}

    model = flat.get("ssdr_handoff.model_name")
    alpha = flat.get("stage02.par_weight")
    beta = flat.get("stage02.bone_weight")
    gamma = flat.get("stage02.child_weight")

    require(isinstance(model, str) and model, f"{row.get('row_id')}: missing model")
    for name, value in [
        ("alpha", alpha),
        ("beta", beta),
        ("gamma", gamma),
    ]:
        require(
            type(value) in (int, float),
            f"{row.get('row_id')}: missing/non-numeric {name}: {value!r}",
        )

    mask_mode = flat.get("stage04.mask_mode")
    mask_threshold = flat.get("stage04.mask_threshold")
    tau = "adaptive" if str(mask_mode).lower() == "adaptive" else mask_threshold

    return {
        "schema": "ogd_lbs_ofat_case_v1",
        "model": model,
        "param": "gamma",
        "value": gamma,
        "stage": "ssdr_step04",
        "case_id": row.get("row_id"),
        "effective_params": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "omega_b": flat.get("stage03.boundary_weight", 1000.0),
            "s_max": flat.get("stage04.s_max", 4),
            "tau": tau,
        },
        "dataset": {
            "model": model,
        },
        "pd_rew": {
            "row_id": row.get("row_id"),
            "row_index": row.get("row_index"),
            "sweep_choices": choices,
            "sweep_values": row.get("sweep_values") or {},
            "stage04_data_dir": stage04_data_dir,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an OGD-LBS gamma report from a pd-rew result pool."
    )
    parser.add_argument("--pd-rew-root", type=Path, default=DEFAULT_PD_REW_ROOT)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", default="OGD-LBS Gamma Sweep — pd-rew Result Pool")
    parser.add_argument(
        "--require-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require data4/opt_out/metrics_history_per_frame.npz. "
            "Use --no-require-history to generate without fluctuation whiskers."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pd_rew_root = args.pd_rew_root.expanduser().resolve(strict=True)
    pool = args.pool.expanduser().resolve(strict=True)
    binding_path = args.binding.expanduser().resolve(strict=True)
    table_path = args.table.expanduser().resolve(strict=True)
    aliases_path = args.aliases.expanduser().resolve(strict=True)
    report_root = args.report_root.expanduser().resolve(strict=False)

    add_pd_rew_source(pd_rew_root)

    import datalad.api as dl
    import pd_rew
    import aggregate_metrics
    import generate_report

    from pd_rew.experiment_plan import (
        format_experiment_table_cache_status,
        query_experiment_table_cache_status,
    )
    from pd_rew.experiment_table import expand_experiment_table
    from pd_rew.parameter_aliases import validate_parameter_aliases
    from pd_rew.result_pool import verify_pool

    expected_pd_rew_init = pd_rew_root / "src/pd_rew/__init__.py"
    require(
        Path(pd_rew.__file__).resolve() == expected_pd_rew_init,
        f"Wrong pd_rew installation: {pd_rew.__file__}",
    )

    require(pool.is_dir(), f"Result pool missing: {pool}")
    require(report_root != Path("/"), "Refusing to use filesystem root as report root")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir is None:
        output_dir = report_root / f"pdrew_gamma_report_{timestamp}"
    else:
        output_dir = args.output_dir.expanduser().resolve(strict=False)

    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    require(
        output_dir != pool
        and not output_dir.is_relative_to(pool)
        and not pool.is_relative_to(output_dir),
        "Report output and source pool must not overlap",
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    compatibility_root = output_dir / "compat_input"
    aggregate_dir = output_dir / "aggregate"
    report_html = output_dir / "Gamma_Report.html"
    compatibility_root.mkdir()
    aggregate_dir.mkdir()

    print("PD_REW_ROOT :", pd_rew_root, flush=True)
    print("OGD_ROOT    :", OGD_ROOT, flush=True)
    print("POOL        :", pool, flush=True)
    print("BINDING     :", binding_path, flush=True)
    print("TABLE       :", table_path, flush=True)
    print("OUTPUT      :", output_dir, flush=True)

    verify_pool(pool, check_annex_hashes=False)

    alias_validation = validate_parameter_aliases(binding_path, aliases_path)
    expanded = expand_experiment_table(binding_path, table_path)

    cache_status = query_experiment_table_cache_status(
        binding_path,
        table_path,
        pool,
        start_stage="stage04",
        end_stage="stage04",
    )

    print()
    print(
        format_experiment_table_cache_status(
            cache_status,
            max_rows=25,
            include_choices=True,
        ),
        flush=True,
    )

    stage04_summary = cache_status["stage_summaries_by_name"]["stage04"]
    require(
        stage04_summary["query_error_rows"] == 0,
        "Stage04 cache query contains errors",
    )
    require(
        stage04_summary["parameters_incomplete_rows"] == 0,
        "Stage04 cache query contains incomplete parameter rows",
    )
    require(
        stage04_summary["hit_rows"] == cache_status["row_count"],
        (
            "Stage04 cache is incomplete: "
            f"HIT={stage04_summary['hit_rows']} "
            f"ROWS={cache_status['row_count']}"
        ),
    )
    require(
        cache_status["row_count"] == expanded["row_count"] == 25,
        (
            "This temporary gamma adapter expects exactly 25 rows: "
            f"cache={cache_status['row_count']} "
            f"expanded={expanded['row_count']}"
        ),
    )

    cache_rows = {
        row["row_id"]: row
        for row in cache_status["rows"]
    }

    source_records: list[dict[str, Any]] = []

    for row in expanded["rows"]:
        row_id = row["row_id"]
        cache_row = cache_rows.get(row_id)
        require(cache_row is not None, f"Cache row missing: {row_id}")

        stage_status = cache_row["stage_statuses"]["stage04"]
        require(
            stage_status.get("status") == "HIT",
            f"{row_id}: Stage04 is not HIT",
        )

        data_dir = stage_status["data_dir"]
        result_dir = safe_pool_data_dir(pool, data_dir)
        data4_dir = result_dir / "data4"

        summary_src = data4_dir / "step04_optimize_summary.json"
        history_src = data4_dir / "opt_out/metrics_history_per_frame.npz"

        materialize_pool_file(
            datalad_api=dl,
            pool=pool,
            path=summary_src,
            required=True,
        )
        history_present = materialize_pool_file(
            datalad_api=dl,
            pool=pool,
            path=history_src,
            required=bool(args.require_history),
        )

        summary = load_json_object(summary_src)
        final_metrics = summary.get("final_metrics")
        require(
            isinstance(final_metrics, dict) and final_metrics,
            f"{row_id}: summary.final_metrics missing or empty: {summary_src}",
        )
        require(
            any(
                key in final_metrics
                for key in [
                    "04_L2_Err",
                    "04_Org_Loss",
                    "04_Hausdorff_Mean",
                    "04_Hausdorff_Max",
                ]
            ),
            f"{row_id}: no reportable Stage04 metrics in {summary_src}",
        )

        case_config = gamma_case_from_expanded_row(
            row,
            stage04_data_dir=data_dir,
        )

        model = str(case_config["model"])
        gamma = case_config["value"]
        gamma_slug = str(gamma).replace(".", "p").replace("-", "m")
        case_root = (
            compatibility_root
            / model
            / "sweep_gamma"
            / f"gamma_{gamma_slug}"
        )
        require(not case_root.exists(), f"Duplicate compatibility case: {case_root}")

        case_root.mkdir(parents=True)
        strict_json_write(case_root / "case_config.json", case_config)

        # aggregate_metrics.py requires a sibling metrics_04.json. The Stage04
        # final_metrics object is the authoritative source for this adapter.
        strict_json_write(case_root / "metrics_04.json", final_metrics)

        summary_dst = case_root / "data4/step04_optimize_summary.json"
        summary_dst.parent.mkdir(parents=True)
        shutil.copy2(summary_src, summary_dst)

        history_dst = case_root / "data4/opt_out/metrics_history_per_frame.npz"
        if history_present:
            history_dst.parent.mkdir(parents=True)
            shutil.copy2(history_src, history_dst)

        source_records.append(
            {
                "row_id": row_id,
                "model": model,
                "gamma": gamma,
                "stage04_data_dir": data_dir,
                "summary_source": str(summary_src),
                "summary_sha256": sha256_file(summary_src),
                "history_source": str(history_src) if history_present else None,
                "history_sha256": (
                    sha256_file(history_src) if history_present else None
                ),
                "compatibility_case_root": str(case_root),
            }
        )

        print(
            f"[materialized] {row_id} model={model} "
            f"gamma={gamma} source={data_dir}",
            flush=True,
        )

    source_manifest = {
        "schema": "ogd_lbs_pdrew_gamma_report_source_v1",
        "pool": str(pool),
        "binding": str(binding_path),
        "table": str(table_path),
        "aliases": str(aliases_path),
        "row_count": len(source_records),
        "primary_parameter": "stage02.child_weight",
        "display_parameter": "gamma",
        "history_required": bool(args.require_history),
        "alias_validation": alias_validation,
        "records": source_records,
    }
    strict_json_write(output_dir / "pdrew_source_manifest.json", source_manifest)

    aggregate_manifest = aggregate_metrics.aggregate(
        input_root=compatibility_root,
        output_dir=aggregate_dir,
        mode="sweep",
        metrics_requested=None,
        group_by=["param", "value"],
        ttest_kind="none",
        baseline_query_text="",
        pair_by=None,
        ttest_group_by=None,
        where_text="param=gamma",
    )

    aggregate_manifest["pd_rew_source"] = {
        "pool": str(pool),
        "binding": str(binding_path),
        "table": str(table_path),
        "aliases": str(aliases_path),
        "source_manifest": str(output_dir / "pdrew_source_manifest.json"),
        "stage04_hit_rows": stage04_summary["hit_rows"],
    }
    strict_json_write(
        aggregate_dir / "aggregate_manifest.json",
        aggregate_manifest,
    )

    generate_report.generate_report(
        aggregate_dir=aggregate_dir,
        output_html=report_html,
        mode="sweep",
        title=str(args.title),
        default_sweep_values=None,
        model_scale_values=None,
        scaled_metrics=None,
        axis_mode=generate_report.DEFAULT_AXIS_MODE,
        pseudo_log_ratio_threshold=(
            generate_report.DEFAULT_PSEUDO_LOG_RATIO_THRESHOLD
        ),
    )

    require(report_html.is_file(), f"Report was not generated: {report_html}")

    final_manifest = {
        "schema": "ogd_lbs_pdrew_gamma_report_v1",
        "status": "GENERATED",
        "pool": str(pool),
        "binding": str(binding_path),
        "table": str(table_path),
        "row_count": len(source_records),
        "aggregate_dir": str(aggregate_dir),
        "report_html": str(report_html),
        "compatibility_input": str(compatibility_root),
        "source_manifest": str(output_dir / "pdrew_source_manifest.json"),
        "pool_mutation_performed": False,
        "result_publication_performed": False,
    }
    strict_json_write(output_dir / "report_manifest.json", final_manifest)

    # Add served download center and an integrity-checked report ZIP.
    from add_pdrew_report_downloads import build_downloads
    download_receipt = build_downloads(output_dir)

    print()
    print("=" * 96)
    print("PD-REW GAMMA REPORT GENERATED")
    print("ROWS        :", len(source_records))
    print("REPORT      :", report_html)
    print("AGGREGATE   :", aggregate_dir)
    print("MANIFEST    :", output_dir / "report_manifest.json")
    print("INDEX       :", download_receipt["index_html"])
    print("DOWNLOAD ZIP:", download_receipt["zip_path"])
    print("=" * 96)


if __name__ == "__main__":
    main()
