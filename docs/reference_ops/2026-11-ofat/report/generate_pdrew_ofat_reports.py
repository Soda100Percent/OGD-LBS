#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_PDREW_ROOT = Path("/workspace/pd-rew")
DEFAULT_OGD_ROOT = Path("/workspace/OGD-LBS")
DEFAULT_POOL = Path(
    "/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1"
)
DEFAULT_BINDING = (
    DEFAULT_PDREW_ROOT / "configs/ogd_lbs_repo_binding.yaml"
)
DEFAULT_TABLE = (
    DEFAULT_PDREW_ROOT / "configs/ogd_lbs_experiment_table.yaml"
)
DEFAULT_ALIASES = (
    DEFAULT_PDREW_ROOT / "configs/ogd_lbs_parameter_aliases.yaml"
)
DEFAULT_REPORT_ROOT = Path("/workspace/ogd_reports")

EXPECTED_POOL_HEAD = (
    "110f1834862e1a2e1892d3092532cabfc0bb9d16"
)
EXPECTED_SOURCE_COMMIT = (
    "1c76292ca3ff2c260aa758e505d3651d5a89ee8c"
)
EXPECTED_IMAGE_UUID = "b1abbc45c40247f2a6bc4f4fa8abcaa8"

FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "gamma": {
        "profile_prefix": "weight-",
        "param": "gamma",
        "value_ref": "stage02.child_weight",
        "report_filename": "Gamma_Report.html",
        "title": "OGD-LBS Gamma Sweep — pd-rew Result Pool",
        "expected_values": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "s_max": {
        "profile_prefix": "s_max-",
        "param": "s_max",
        "value_ref": "stage04.s_max",
        "report_filename": "SMax_Report.html",
        "title": "OGD-LBS s_max Sweep — pd-rew Result Pool",
        "expected_values": [1, 2, 4, 6, 8],
    },
    "tau": {
        "profile_prefix": "tau-",
        "param": "tau",
        "value_ref": None,
        "report_filename": "Tau_Report.html",
        "title": "OGD-LBS Mask Threshold τ Sweep — pd-rew Result Pool",
        "expected_values": [
            "adaptive",
            0.0001,
            0.001,
            0.01,
            0.05,
        ],
    },
    "omega_b": {
        "profile_prefix": "boundary_weight-",
        "param": "omega_b",
        "value_ref": "stage03.boundary_weight",
        "report_filename": "Boundary_Weight_Report.html",
        "title": "OGD-LBS QEM Boundary Weight ω_b Sweep — pd-rew Result Pool",
        "expected_values": [100.0, 500.0, 1000.0, 2000.0, 5000.0],
    },
}

EXPECTED_MODELS = {
    "Griffin_A2",
    "Winter_Samurai",
    "Vroid_18",
    "Vroid_19",
    "Vroid_20",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def strict_json_bytes(obj: Any) -> bytes:
    return (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(name)

    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, strict_json_bytes(obj))


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def load_json_object(path: Path, *, field: str) -> dict[str, Any]:
    require(
        path.is_file(),
        f"{field} is missing or unavailable: {path}",
    )

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"{field} is not valid JSON: {path}"
        ) from exc

    require(
        isinstance(value, dict),
        f"{field} root must be an object: {path}",
    )
    return value


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"file is unavailable: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(
        proc.returncode == 0,
        f"git failed in {repo}: {args!r}\n"
        f"{proc.stderr[-3000:]}",
    )
    return proc.stdout.strip()


def family_for_profile(profile: Any) -> str:
    text = str(profile)

    matches = [
        family
        for family, spec in FAMILY_SPECS.items()
        if text.startswith(spec["profile_prefix"])
    ]
    require(
        len(matches) == 1,
        f"unsupported or ambiguous OFAT profile: {profile!r}",
    )
    return matches[0]


def value_for_family(
    family: str,
    flat: dict[str, Any],
) -> Any:
    if family == "tau":
        mode = str(flat["stage04.mask_mode"])
        threshold = float(flat["stage04.mask_threshold"])
        if mode == "adaptive":
            require(
                abs(threshold - (-1.0)) <= 1e-12,
                "adaptive tau row does not use -1.0 sentinel",
            )
            return "adaptive"

        require(mode == "fixed", f"unsupported mask mode: {mode!r}")
        require(threshold >= 0.0, "fixed tau is negative")
        return threshold

    ref = FAMILY_SPECS[family]["value_ref"]
    require(isinstance(ref, str), f"{family}: value_ref is invalid")
    return flat[ref]


def canonical_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        require(math.isfinite(value), "non-finite sweep value")
        return format(value, ".15g")
    raise RuntimeError(f"unsupported sweep value: {value!r}")


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    return abs(float(left) - float(right)) <= 1e-12


def safe_pool_data_dir(pool: Path, value: Any) -> Path:
    require(
        isinstance(value, str) and value,
        f"invalid Pool data_dir: {value!r}",
    )

    rel = PurePosixPath(value)
    require(
        not rel.is_absolute()
        and ".." not in rel.parts
        and str(rel) == value
        and len(rel.parts) == 2
        and rel.parts[0] == "results"
        and re.fullmatch(r"r[0-9]{6}", rel.parts[1]) is not None,
        f"unsafe Pool data_dir: {value!r}",
    )
    return pool / rel


def materialize_pool_file(
    *,
    datalad_api: Any,
    pool: Path,
    path: Path,
) -> None:
    require(
        os.path.lexists(path),
        f"required report file is not tracked: {path}",
    )

    datalad_api.get(
        path=str(path),
        dataset=str(pool),
        result_renderer="disabled",
        on_failure="stop",
    )

    require(
        path.is_file(),
        f"required report file could not be materialized: {path}",
    )


def report_sort_key(value: Any) -> tuple[int, float | str]:
    """Put the τ adaptive baseline before numeric values."""
    text = str(value)

    if text == "adaptive":
        return (-1, 0.0)

    try:
        return (0, float(text))
    except Exception:
        return (1, text)


def case_config_for_row(
    row: dict[str, Any],
    *,
    family: str,
    value: Any,
    data_dir: str,
) -> dict[str, Any]:
    flat = row.get("flat_parameters") or {}
    choices = row.get("sweep_choices") or {}

    model = flat.get("ssdr_handoff.model_name")
    require(isinstance(model, str) and model, "row lacks model")

    alpha = float(flat["stage02.par_weight"])
    beta = float(flat["stage02.bone_weight"])
    gamma = float(flat["stage02.child_weight"])
    omega_b = float(flat["stage03.boundary_weight"])
    s_max = int(flat["stage04.s_max"])
    mask_mode = str(flat["stage04.mask_mode"])
    mask_threshold = float(flat["stage04.mask_threshold"])

    tau = (
        "adaptive"
        if mask_mode == "adaptive"
        else mask_threshold
    )

    return {
        "schema": "ogd_lbs_ofat_case_v1",
        "model": model,
        "param": FAMILY_SPECS[family]["param"],
        "value": value,
        "stage": "ssdr_step04",
        "case_id": row["row_id"],
        "effective_params": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "omega_b": omega_b,
            "s_max": s_max,
            "tau": tau,
        },
        "dataset": {"model": model},
        "pd_rew": {
            "family": family,
            "row_id": row["row_id"],
            "row_index": row["row_index"],
            "profile": choices.get("weights"),
            "sweep_choices": choices,
            "sweep_values": row.get("sweep_values") or {},
            "stage04_data_dir": data_dir,
        },
    }


def validate_family_records(
    records: list[dict[str, Any]],
) -> None:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)

    require(
        set(by_family) == set(FAMILY_SPECS),
        f"report families differ: {sorted(by_family)!r}",
    )

    for family, spec in FAMILY_SPECS.items():
        rows = by_family[family]
        require(
            len(rows) == 25,
            f"{family}: expected 25 rows, observed {len(rows)}",
        )

        models = {row["model"] for row in rows}
        require(
            models == EXPECTED_MODELS,
            f"{family}: model set differs: {models!r}",
        )

        expected_values = spec["expected_values"]
        observed_values = []
        for row in rows:
            if not any(
                values_equal(row["value"], existing)
                for existing in observed_values
            ):
                observed_values.append(row["value"])

        require(
            len(observed_values) == len(expected_values)
            and all(
                any(values_equal(wanted, got) for got in observed_values)
                for wanted in expected_values
            ),
            f"{family}: value set differs: {observed_values!r}",
        )

        combinations = {
            (row["model"], canonical_value(row["value"]))
            for row in rows
        }
        require(
            len(combinations) == 25,
            f"{family}: duplicate model/value combinations",
        )

        if family != "gamma":
            for row in rows:
                require(
                    abs(row["alpha"] - 0.5) <= 1e-12
                    and abs(row["beta"] - 0.3) <= 1e-12
                    and abs(row["gamma"] - 0.2) <= 1e-12,
                    f"{family}: non-Gamma baseline differs",
                )


def patch_report_toolbar(
    report_html: Path,
    *,
    family: str,
    family_zip_name: str,
) -> None:
    source = report_html.read_text(encoding="utf-8")
    require("<body>" in source, f"{report_html} lacks <body>")

    aggregate_rel = f"families/{family}/aggregate"
    toolbar = f"""
<style>
.pdrew-report-nav {{
  position: sticky; top: 0; z-index: 950;
  display: flex; flex-wrap: wrap; justify-content: center;
  gap: 8px; margin: 0 auto 20px; padding: 10px;
  background: rgba(255,255,255,.97);
  border: 1px solid #d0d5dd; border-radius: 8px;
}}
.pdrew-report-nav a {{
  padding: 7px 11px; border-radius: 5px;
  background: #1C328A; color: #fff;
  text-decoration: none; font-weight: 600;
}}
.pdrew-report-nav a.primary {{ background: #0f766e; }}
</style>
<nav class="pdrew-report-nav" aria-label="Report downloads">
  <a href="index.html">Report Center</a>
  <a class="primary" href="downloads/{html.escape(family_zip_name, quote=True)}" download>Family ZIP</a>
  <a href="{html.escape(aggregate_rel, quote=True)}/aggregated_metrics.csv" download>CSV</a>
  <a href="{html.escape(aggregate_rel, quote=True)}/aggregated_metrics.json" download>JSON</a>
  <a href="{html.escape(aggregate_rel, quote=True)}/summary_table.md" download>Markdown</a>
  <a href="{html.escape(aggregate_rel, quote=True)}/summary_table.tex" download>LaTeX</a>
</nav>
"""
    atomic_write_text(
        report_html,
        source.replace("<body>", "<body>\n" + toolbar, 1),
    )


def regular_file_record(
    root: Path,
    relative: str,
) -> dict[str, Any]:
    path = root / relative
    require(
        path.is_file() and not path.is_symlink(),
        f"bundle input missing or unsafe: {path}",
    )
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_zip(
    *,
    root: Path,
    relative_files: list[str],
    destination: Path,
) -> dict[str, Any]:
    require(
        not destination.exists() and not destination.is_symlink(),
        f"ZIP already exists: {destination}",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)

    relative_files = list(dict.fromkeys(relative_files))

    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for relative in relative_files:
                path = root / relative
                require(
                    path.is_file() and not path.is_symlink(),
                    f"unsafe ZIP member: {path}",
                )
                archive.write(path, arcname=relative)

        with zipfile.ZipFile(temporary, "r") as archive:
            bad_member = archive.testzip()
            require(
                bad_member is None,
                f"ZIP integrity failure: {bad_member}",
            )

        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "path": str(destination),
        "relative_path": destination.relative_to(root).as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "member_count": len(relative_files),
        "members": relative_files,
    }


def build_index(
    *,
    report_records: list[dict[str, Any]],
    complete_zip_name: str,
    provenance: dict[str, Any],
) -> str:
    rows = []

    for record in report_records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['display_name'])}</td>"
            f"<td>{record['row_count']}</td>"
            f"<td><a href=\"{html.escape(record['report_html'], quote=True)}\">Open report</a></td>"
            f"<td><a href=\"{html.escape(record['zip_relpath'], quote=True)}\" download>ZIP</a></td>"
            f"<td><a href=\"{html.escape(record['aggregate_csv'], quote=True)}\" download>CSV</a></td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OGD-LBS pd-rew OFAT Report Center</title>
<style>
body {{
  font-family: Segoe UI,Tahoma,sans-serif;
  max-width: 1050px; margin: 40px auto; padding: 0 20px;
  color: #263238; background: #f4f6f8;
}}
main {{
  background: #fff; padding: 28px; border-radius: 10px;
  box-shadow: 0 4px 14px rgba(0,0,0,.08);
}}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th,td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
th {{ background: #f1f5f9; }}
a {{ color: #1C328A; font-weight: 600; }}
.primary {{
  display: inline-block; color: #fff; background: #0f766e;
  padding: 10px 15px; border-radius: 6px; text-decoration: none;
}}
.note {{ color: #667085; }}
code {{ background: #f1f5f9; padding: 2px 5px; }}
</style>
</head>
<body>
<main>
<h1>OGD-LBS pd-rew OFAT Report Center</h1>
<p>
<a class="primary" href="downloads/{html.escape(complete_zip_name, quote=True)}" download>
Download complete report ZIP
</a>
</p>
<table>
<thead>
<tr><th>Experiment family</th><th>Rows</th><th>Interactive report</th><th>ZIP</th><th>CSV</th></tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<h2>Provenance</h2>
<ul>
<li>Pool HEAD: <code>{html.escape(str(provenance['pool_head']))}</code></li>
<li>OGD-LBS algorithm commit: <code>{html.escape(str(provenance['algorithm_source_commit']))}</code></li>
<li>Image UUID: <code>{html.escape(str(provenance['image_uuid']))}</code></li>
<li>Experiment rows: 100</li>
<li>Unique Stage04 results: 90</li>
</ul>
<p class="note">
Distance charts use the established model scale rules. Chart whiskers are
per-frame min-max ranges, not mean ± standard deviation.
</p>
<p class="note">
ECharts and jsPDF are loaded from a CDN. CSV, JSON and ZIP downloads do not
depend on those CDN resources. This HTTP report service has no authentication.
</p>
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Gamma, s_max, tau and omega_b reports from "
            "the completed 100-row pd-rew experiment table."
        )
    )
    parser.add_argument(
        "--pd-rew-root",
        type=Path,
        default=DEFAULT_PDREW_ROOT,
    )
    parser.add_argument(
        "--ogd-root",
        type=Path,
        default=DEFAULT_OGD_ROOT,
    )
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; it must not already exist.",
    )
    parser.add_argument(
        "--expected-pool-head",
        default=EXPECTED_POOL_HEAD,
    )
    parser.add_argument(
        "--algorithm-source-commit",
        default=EXPECTED_SOURCE_COMMIT,
    )
    parser.add_argument(
        "--image-uuid",
        default=EXPECTED_IMAGE_UUID,
    )
    parser.add_argument(
        "--require-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdrew_root = args.pd_rew_root.expanduser().resolve(strict=True)
    ogd_root = args.ogd_root.expanduser().resolve(strict=True)
    pool = args.pool.expanduser().resolve(strict=True)
    binding = args.binding.expanduser().resolve(strict=True)
    table = args.table.expanduser().resolve(strict=True)
    aliases = args.aliases.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)

    require(
        not output_dir.exists() and not output_dir.is_symlink(),
        f"output directory already exists: {output_dir}",
    )
    require(
        output_dir != pool
        and not output_dir.is_relative_to(pool)
        and not pool.is_relative_to(output_dir),
        "report output and Pool must not overlap",
    )

    experiments_dir = ogd_root / "scripts/experiments"
    require(experiments_dir.is_dir(), "OGD experiments directory missing")

    sys.path.insert(0, str(pdrew_root / "src"))
    sys.path.insert(0, str(experiments_dir))

    import datalad.api as dl
    import pd_rew
    import aggregate_metrics
    import generate_report

    from pd_rew.experiment_plan import (
        query_experiment_table_cache_status,
    )
    from pd_rew.experiment_table import expand_experiment_table
    from pd_rew.parameter_aliases import validate_parameter_aliases
    from pd_rew.result_pool import verify_pool

    require(
        Path(pd_rew.__file__).resolve()
        == pdrew_root / "src/pd_rew/__init__.py",
        "wrong pd_rew package imported",
    )

    # The unused broken mean/std helper is deliberately not called.
    # Override only the value sorter so τ=adaptive is rendered first.
    generate_report.sweep_value_sort_key = report_sort_key

    verify_pool(pool, check_annex_hashes=False)
    pool_head_before = git_output(pool, "rev-parse", "--verify", "HEAD")
    pool_status_before = git_output(
        pool,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    require(
        pool_head_before == args.expected_pool_head,
        f"Pool HEAD changed: {pool_head_before} "
        f"!= {args.expected_pool_head}",
    )
    require(not pool_status_before, "Pool is dirty before report generation")

    alias_validation = validate_parameter_aliases(binding, aliases)
    expanded = expand_experiment_table(binding, table)
    cache = query_experiment_table_cache_status(
        binding,
        table,
        pool,
        start_stage="stage04",
        end_stage="stage04",
    )

    require(
        expanded.get("row_count") == 100
        and cache.get("row_count") == 100,
        "report generator requires exactly 100 experiment rows",
    )

    summary = cache["stage_summaries_by_name"]["stage04"]
    require(
        summary.get("hit_rows") == 100
        and summary.get("missing_rows") == 0
        and summary.get("parameters_incomplete_rows") == 0
        and summary.get("query_error_rows") == 0,
        f"Stage04 cache is incomplete: {summary!r}",
    )

    cache_by_row = {
        row["row_id"]: row
        for row in cache["rows"]
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    compat_root = output_dir / "compat_input"
    families_root = output_dir / "families"
    downloads_root = output_dir / "downloads"
    compat_root.mkdir()
    families_root.mkdir()
    downloads_root.mkdir()

    report_code_files = [
        experiments_dir / "aggregate_metrics.py",
        experiments_dir / "generate_report.py",
        experiments_dir / "report_templates/ogd_report_styles.css",
        experiments_dir / "report_templates/ogd_report_dashboard.js",
    ]
    report_code_hashes = {
        str(path): sha256_file(path)
        for path in report_code_files
    }

    materialized: dict[str, dict[str, Any]] = {}
    source_records: list[dict[str, Any]] = []

    for row in expanded["rows"]:
        row_id = row["row_id"]
        choices = row.get("sweep_choices") or {}
        profile = choices.get("weights")
        family = family_for_profile(profile)
        flat = row.get("flat_parameters") or {}
        value = value_for_family(family, flat)
        model = str(flat["ssdr_handoff.model_name"])

        status_row = cache_by_row[row_id]["stage_statuses"]["stage04"]
        require(
            status_row.get("status") == "HIT",
            f"{row_id}: Stage04 is not HIT",
        )

        data_dir = status_row["data_dir"]
        result_dir = safe_pool_data_dir(pool, data_dir)
        summary_src = (
            result_dir / "data4/step04_optimize_summary.json"
        )
        history_src = (
            result_dir
            / "data4/opt_out/metrics_history_per_frame.npz"
        )

        if data_dir not in materialized:
            materialize_pool_file(
                datalad_api=dl,
                pool=pool,
                path=summary_src,
            )
            materialize_pool_file(
                datalad_api=dl,
                pool=pool,
                path=history_src,
            )

            summary_obj = load_json_object(
                summary_src,
                field=f"{data_dir} Stage04 summary",
            )
            final_metrics = summary_obj.get("final_metrics")
            require(
                isinstance(final_metrics, dict) and final_metrics,
                f"{data_dir}: final_metrics missing",
            )

            config = summary_obj.get("config")
            require(
                isinstance(config, dict),
                f"{data_dir}: Stage04 config missing",
            )

            materialized[data_dir] = {
                "summary": summary_obj,
                "summary_sha256": sha256_file(summary_src),
                "history_sha256": sha256_file(history_src),
            }

        source = materialized[data_dir]
        summary_obj = source["summary"]
        config = summary_obj["config"]

        require(
            int(config.get("s_max")) == int(flat["stage04.s_max"]),
            f"{row_id}: summary s_max differs from table",
        )
        require(
            str(config.get("mask_mode"))
            == str(flat["stage04.mask_mode"]),
            f"{row_id}: summary mask_mode differs from table",
        )
        require(
            abs(
                float(config.get("mask_threshold"))
                - float(flat["stage04.mask_threshold"])
            )
            <= 1e-9,
            f"{row_id}: summary mask_threshold differs from table",
        )

        case_config = case_config_for_row(
            row,
            family=family,
            value=value,
            data_dir=data_dir,
        )

        profile_slug = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(profile),
        )
        case_root = (
            compat_root
            / family
            / model
            / f"sweep_{FAMILY_SPECS[family]['param']}"
            / profile_slug
        )
        require(
            not case_root.exists(),
            f"duplicate compatibility case: {case_root}",
        )
        case_root.mkdir(parents=True)

        atomic_write_json(
            case_root / "case_config.json",
            case_config,
        )
        atomic_write_json(
            case_root / "metrics_04.json",
            summary_obj["final_metrics"],
        )

        summary_dst = (
            case_root / "data4/step04_optimize_summary.json"
        )
        history_dst = (
            case_root
            / "data4/opt_out/metrics_history_per_frame.npz"
        )
        summary_dst.parent.mkdir(parents=True)
        history_dst.parent.mkdir(parents=True)

        shutil.copy2(summary_src, summary_dst)
        shutil.copy2(history_src, history_dst)

        source_records.append({
            "family": family,
            "profile": profile,
            "row_id": row_id,
            "model": model,
            "value": value,
            "alpha": float(flat["stage02.par_weight"]),
            "beta": float(flat["stage02.bone_weight"]),
            "gamma": float(flat["stage02.child_weight"]),
            "omega_b": float(flat["stage03.boundary_weight"]),
            "s_max": int(flat["stage04.s_max"]),
            "mask_mode": str(flat["stage04.mask_mode"]),
            "mask_threshold": float(
                flat["stage04.mask_threshold"]
            ),
            "stage04_data_dir": data_dir,
            "summary_source": str(summary_src),
            "summary_sha256": source["summary_sha256"],
            "history_source": str(history_src),
            "history_sha256": source["history_sha256"],
            "compatibility_case_root": str(case_root),
        })

        print(
            f"MATERIALIZED {row_id} family={family} "
            f"model={model} value={value} source={data_dir}",
            flush=True,
        )

    validate_family_records(source_records)

    data_dir_counts = Counter(
        record["stage04_data_dir"]
        for record in source_records
    )
    require(
        len(data_dir_counts) == 90,
        f"expected 90 unique Stage04 results, got {len(data_dir_counts)}",
    )
    require(
        sorted(data_dir_counts.values()).count(3) == 5
        and all(count in {1, 3} for count in data_dir_counts.values()),
        "unexpected cross-family baseline reuse pattern",
    )

    source_manifest = {
        "schema": "ogd_lbs_pdrew_ofat_report_source_v1",
        "status": "MATERIALIZED",
        "generated_at_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "pool": str(pool),
        "pool_head": pool_head_before,
        "binding": str(binding),
        "table": str(table),
        "aliases": str(aliases),
        "algorithm_source_commit": args.algorithm_source_commit,
        "image_uuid": args.image_uuid,
        "experiment_row_count": 100,
        "unique_stage04_result_count": 90,
        "history_required": bool(args.require_history),
        "alias_validation": alias_validation,
        "report_code_hashes": report_code_hashes,
        "records": source_records,
    }
    atomic_write_json(
        output_dir / "pdrew_source_manifest.json",
        source_manifest,
    )

    records_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        records_by_family[record["family"]].append(record)

    report_records: list[dict[str, Any]] = []
    family_zip_specs: list[tuple[str, list[str], Path]] = []

    default_sweep_values = {
        "alpha": "0.5",
        "beta": "0.3",
        "gamma": "0.2",
        "s_max": "4",
        "tau": "adaptive",
        "omega_b": "1000.0",
    }

    timestamp = dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    for family, spec in FAMILY_SPECS.items():
        family_root = families_root / family
        aggregate_dir = family_root / "aggregate"
        family_root.mkdir()
        aggregate_dir.mkdir()

        family_source_manifest = {
            "schema": "ogd_lbs_pdrew_ofat_family_source_v1",
            "family": family,
            "param": spec["param"],
            "row_count": len(records_by_family[family]),
            "records": records_by_family[family],
        }
        atomic_write_json(
            family_root / "source_manifest.json",
            family_source_manifest,
        )

        aggregate_manifest = aggregate_metrics.aggregate(
            input_root=compat_root / family,
            output_dir=aggregate_dir,
            mode="sweep",
            metrics_requested=None,
            group_by=["param", "value"],
            ttest_kind="none",
            baseline_query_text="",
            pair_by=None,
            ttest_group_by=None,
            where_text=f"param={spec['param']}",
        )
        require(
            aggregate_manifest.get("row_count") == 25,
            f"{family}: aggregate did not contain 25 rows",
        )

        aggregate_manifest["pd_rew_source"] = {
            "family": family,
            "pool": str(pool),
            "pool_head": pool_head_before,
            "source_manifest": str(
                family_root / "source_manifest.json"
            ),
        }
        atomic_write_json(
            aggregate_dir / "aggregate_manifest.json",
            aggregate_manifest,
        )

        report_html = output_dir / spec["report_filename"]
        generate_report.generate_report(
            aggregate_dir=aggregate_dir,
            output_html=report_html,
            mode="sweep",
            title=spec["title"],
            default_sweep_values=default_sweep_values,
            model_scale_values=None,
            scaled_metrics=None,
            axis_mode=generate_report.DEFAULT_AXIS_MODE,
            pseudo_log_ratio_threshold=(
                generate_report.DEFAULT_PSEUDO_LOG_RATIO_THRESHOLD
            ),
        )
        require(
            report_html.is_file(),
            f"{family}: report HTML was not generated",
        )

        family_zip_name = (
            f"ogd_lbs_pdrew_{family}_{timestamp}.zip"
        )
        patch_report_toolbar(
            report_html,
            family=family,
            family_zip_name=family_zip_name,
        )

        aggregate_rel = f"families/{family}/aggregate"
        family_files = [
            spec["report_filename"],
            f"families/{family}/source_manifest.json",
            f"{aggregate_rel}/aggregate_manifest.json",
            f"{aggregate_rel}/aggregated_metrics.csv",
            f"{aggregate_rel}/aggregated_metrics.json",
            f"{aggregate_rel}/summary_by_group.csv",
            f"{aggregate_rel}/summary_table.md",
            f"{aggregate_rel}/summary_table.tex",
        ]
        family_zip_path = downloads_root / family_zip_name
        family_zip_specs.append(
            (family, family_files, family_zip_path)
        )

        report_records.append({
            "family": family,
            "display_name": spec["title"],
            "param": spec["param"],
            "row_count": 25,
            "report_html": spec["report_filename"],
            "aggregate_csv": (
                f"{aggregate_rel}/aggregated_metrics.csv"
            ),
            "zip_name": family_zip_name,
            "zip_relpath": f"downloads/{family_zip_name}",
        })

    report_manifest = {
        "schema": "ogd_lbs_pdrew_ofat_report_v1",
        "status": "GENERATED",
        "generated_at_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "pool": str(pool),
        "pool_head": pool_head_before,
        "binding": str(binding),
        "table": str(table),
        "algorithm_source_commit": args.algorithm_source_commit,
        "image_uuid": args.image_uuid,
        "experiment_row_count": 100,
        "unique_stage04_result_count": 90,
        "families": report_records,
        "model_scale_rules": {
            "Vroid_": 1.0,
            "Winter_Samurai": 10.0,
            "Griffin_A2": 0.3,
        },
        "scaled_metrics": [
            "04_L2_Err",
            "04_Hausdorff_Mean",
            "04_Hausdorff_Max",
        ],
        "whisker_semantics": "per-frame min-max",
        "per_frame_std_ddof": 0,
        "pool_mutation_performed": False,
        "result_publication_performed": False,
    }
    atomic_write_json(
        output_dir / "report_manifest.json",
        report_manifest,
    )

    family_zip_receipts = []
    for family, files, destination in family_zip_specs:
        receipt = write_zip(
            root=output_dir,
            relative_files=files,
            destination=destination,
        )
        receipt["family"] = family
        family_zip_receipts.append(receipt)

    # Update report records with verified family ZIP identities.
    zip_by_family = {
        item["family"]: item
        for item in family_zip_receipts
    }
    for record in report_records:
        record["zip_sha256"] = zip_by_family[
            record["family"]
        ]["sha256"]
        record["zip_size_bytes"] = zip_by_family[
            record["family"]
        ]["size_bytes"]

    atomic_write_json(
        output_dir / "report_manifest.json",
        report_manifest,
    )

    complete_zip_name = (
        f"ogd_lbs_pdrew_ofat_100_{timestamp}.zip"
    )

    index_html = build_index(
        report_records=report_records,
        complete_zip_name=complete_zip_name,
        provenance={
            "pool_head": pool_head_before,
            "algorithm_source_commit": (
                args.algorithm_source_commit
            ),
            "image_uuid": args.image_uuid,
        },
    )
    atomic_write_text(output_dir / "index.html", index_html)

    complete_files = [
        "index.html",
        "report_manifest.json",
        "pdrew_source_manifest.json",
    ]

    for _family, files, _destination in family_zip_specs:
        complete_files.extend(files)

    complete_files = list(dict.fromkeys(complete_files))
    bundle_file_records = [
        regular_file_record(output_dir, relative)
        for relative in complete_files
    ]

    download_manifest = {
        "schema": "ogd_lbs_pdrew_ofat_download_bundle_v1",
        "created_at_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "report_directory": str(output_dir),
        "complete_zip_name": complete_zip_name,
        "files": bundle_file_records,
        "family_zips": family_zip_receipts,
        "excluded": [
            "compat_input/",
            "complete Stage04 model payloads",
            "result-pool internals",
        ],
        "chart_runtime": {
            "echarts": "CDN",
            "jspdf": "CDN",
            "fully_offline_charts": False,
        },
    }
    atomic_write_json(
        output_dir / "download_bundle_manifest.json",
        download_manifest,
    )

    complete_files.append("download_bundle_manifest.json")
    complete_zip = write_zip(
        root=output_dir,
        relative_files=complete_files,
        destination=downloads_root / complete_zip_name,
    )

    download_receipt = {
        "schema": "ogd_lbs_pdrew_ofat_download_receipt_v1",
        "status": "CREATED",
        "report_directory": str(output_dir),
        "index_html": str(output_dir / "index.html"),
        "family_zips": family_zip_receipts,
        "complete_zip": complete_zip,
    }
    atomic_write_json(
        output_dir / "download_receipt.json",
        download_receipt,
    )

    verify_pool(pool, check_annex_hashes=False)
    pool_head_after = git_output(pool, "rev-parse", "--verify", "HEAD")
    pool_status_after = git_output(
        pool,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    require(
        pool_head_after == pool_head_before,
        "Pool HEAD changed during report generation",
    )
    require(
        pool_status_after == pool_status_before,
        "Pool Git status changed during report generation",
    )

    final_result = {
        "schema": "ogd_lbs_pdrew_ofat_report_generation_result_v1",
        "status": "COMPLETED",
        "report_directory": str(output_dir),
        "index_html": str(output_dir / "index.html"),
        "pool_head": pool_head_after,
        "algorithm_source_commit": args.algorithm_source_commit,
        "image_uuid": args.image_uuid,
        "row_count": 100,
        "unique_stage04_result_count": 90,
        "family_reports": report_records,
        "family_zip_receipts": family_zip_receipts,
        "complete_zip": complete_zip,
        "pool_head_unchanged": True,
        "pool_status_unchanged": True,
        "http_service_started": False,
    }
    atomic_write_json(
        output_dir / "generation_result.json",
        final_result,
    )

    print()
    print("=" * 96)
    print("PD-REW OFAT REPORTS GENERATED")
    print("=" * 96)
    print(f"REPORT_DIRECTORY={output_dir}")
    print(f"INDEX_HTML={output_dir / 'index.html'}")
    for record in report_records:
        print(
            f"{record['family'].upper()}_REPORT="
            f"{output_dir / record['report_html']}"
        )
    print(
        "COMPLETE_ZIP="
        f"{complete_zip['path']}"
    )
    print(
        "COMPLETE_ZIP_SHA256="
        f"{complete_zip['sha256']}"
    )
    print(f"POOL_HEAD={pool_head_after}")
    print("POOL_HEAD_UNCHANGED=YES")
    print("POOL_STATUS_UNCHANGED=YES")
    print("HTTP_SERVICE_STARTED=NO")
    print("OFAT_REPORT_GENERATION_COMPLETED")


if __name__ == "__main__":
    main()
