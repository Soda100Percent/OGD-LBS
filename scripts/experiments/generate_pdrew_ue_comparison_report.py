#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generate the OGD-LBS Equal-Bone-Budget & UE5 Ablation Interactive Report
with separate sections for:
  - Step 02 (Bone Reduction)
  - Step 03 (Bone + QEM)
  - Step 04 (Bone + QEM + SSDR)
directly from in-pool normalized_metrics_v1.json files.
"""

import argparse
import csv
import datetime as dt
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from generate_pdrew_gamma_report import (  # noqa: E402
    DEFAULT_BINDING,
    DEFAULT_PD_REW_ROOT,
    DEFAULT_POOL,
    DEFAULT_TABLE,
    materialize_pool_file,
    safe_pool_data_dir,
    start_daemon_http_server,
    strict_json_write,
)

MODEL_SCALES = {
    "Griffin_A2": 0.3,
    "Winter_Samurai": 10.0,
    "Vroid_18": 1.0,
    "Vroid_19": 1.0,
    "Vroid_20": 1.0,
}

METRICS_META = [
    {"key": "l2_err", "short": "VD (L2 Err) ↓", "label": "VD / L2 Reconstruction Error ↓", "fstat_key": "l2_err", "scaled": True},
    {"key": "hausdorff_mean", "short": "HD Mean ↓", "label": "Hausdorff Mean ↓", "fstat_key": "hausdorff", "scaled": True},
    {"key": "hausdorff_max", "short": "HD Max ↓", "label": "Hausdorff Max ↓", "fstat_key": "hausdorff", "scaled": True},
    {"key": "org_loss", "short": "SO (Org Loss) ↓", "label": "SO / Bone Org Loss ↓", "fstat_key": "org_loss", "scaled": False},
    {"key": "arap_ratio", "short": "ARAP Ratio ↓", "label": "Mesh ARAP Ratio ↓", "fstat_key": "arap_ratio", "scaled": False},
    {"key": "edge_excess", "short": "Edge Excess ↓", "label": "Edge Log-Ratio p99 Excess ↓", "fstat_key": None, "scaled": False},
    {"key": "normal_flip", "short": "Normal Flip ↓", "label": "Normal Flip Ratio ↓", "fstat_key": None, "scaled": False},
    {"key": "max_roll", "short": "Max Roll (°) ↓", "label": "Bone Max Roll (°) ↓", "fstat_key": "max_roll", "scaled": False},
    {"key": "max_swing", "short": "Max Swing (°) ↓", "label": "Bone Max Swing (°) ↓", "fstat_key": "max_swing", "scaled": False},
]

QUOTA_BENCHMARKS = [
    {"model": "Griffin_A2", "variant": "bone_reduced_conservative", "quota_short": "Conservative", "quota_display": "Conservative Quota"},
    {"model": "Griffin_A2", "variant": "bone_reduced_moderate", "quota_short": "Moderate", "quota_display": "Moderate Quota"},
    {"model": "Winter_Samurai", "variant": "bone_reduced_conservative", "quota_short": "Conservative", "quota_display": "Conservative Quota"},
    {"model": "Winter_Samurai", "variant": "bone_reduced_moderate", "quota_short": "Moderate", "quota_display": "Moderate Quota"},
    {"model": "Vroid_18", "variant": "", "quota_short": "Default", "quota_display": "Default Quota"},
    {"model": "Vroid_19", "variant": "", "quota_display": "Default Quota", "quota_short": "Default"},
    {"model": "Vroid_20", "variant": "", "quota_display": "Default Quota", "quota_short": "Default"},
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _load_in_pool_normalized_metrics(
    *,
    dl: Any,
    pool: Path,
    data_dir: str,
    expected_subdir: str,
    metrics_name: str = "normalized_metrics_v1.json",
) -> dict[str, Any]:
    r_root = safe_pool_data_dir(pool, data_dir)
    candidates = [
        r_root / expected_subdir / metrics_name,
        r_root / "data0_json" / metrics_name,
        r_root / "data_prep_00" / metrics_name,
        r_root / metrics_name,
    ]
    for cand in candidates:
        if os.path.lexists(cand):
            materialize_pool_file(datalad_api=dl, pool=pool, path=cand, required=True)
            return json.loads(cand.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"normalized_metrics_v1.json not found in {r_root} (expected_subdir={expected_subdir})")


def _decorate_norm_obj(
    norm_obj: dict[str, Any],
    *,
    model: str,
    data_dir: str,
    method_key: str,
    method_label: str,
    method_short: str,
) -> dict[str, Any]:
    scale = float(MODEL_SCALES.get(model, 1.0))
    raw_m = norm_obj.get("metrics") or {}
    raw_f = norm_obj.get("frame_stats") or {}

    m_raw: dict[str, float | None] = {}
    m_scaled: dict[str, float | None] = {}
    f_raw: dict[str, Any] = {}
    f_scaled: dict[str, Any] = {}

    for m in METRICS_META:
        mk = m["key"]
        fk = m["fstat_key"]
        factor = scale if m["scaled"] else 1.0
        val = raw_m.get(mk)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            m_raw[mk] = float(val)
            m_scaled[mk] = float(val) * factor
        else:
            m_raw[mk] = None
            m_scaled[mk] = None

        if fk and fk in raw_f and isinstance(raw_f[fk], dict):
            f_raw[fk] = dict(raw_f[fk])
            f_scaled[fk] = {k: float(v) * factor for k, v in raw_f[fk].items() if isinstance(v, (int, float))}

    return {
        "model": model,
        "model_scale": scale,
        "data_dir": data_dir,
        "method_key": method_key,
        "method_label": method_label,
        "method_short": method_short,
        "complexity": dict(norm_obj.get("complexity") or {}),
        "metrics_raw": m_raw,
        "metrics_scaled": m_scaled,
        "frame_stats_raw": f_raw,
        "frame_stats_scaled": f_scaled,
    }


def _build_latex_table(benchmarks: list[dict[str, Any]], *, scale_mode: str) -> str:
    m_field = "metrics_scaled" if scale_mode == "scaled" else "metrics_raw"
    lines = [
        f"% Auto-generated Equal-Bone-Budget & UE5 Comparison Table ({scale_mode})",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Equal-bone-budget comparison across Step 02 (Bone Reduction), Step 03 (Bone + QEM), and Step 04 (Bone + QEM + SSDR). "
        "Within each (Model, Quota) block, total bone count $B$ is strictly identical.}",
        "\\label{tab:ue5_equal_bone_budget}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccccccc}",
        "\\hline",
        "Model & Quota ($B$) & Method (Step 04: Bone + QEM + SSDR) & Verts ($V$) & Tris ($F$) & Max $K$ & Loose & VD ($L_2$) $\\downarrow$ & HD Mean $\\downarrow$ & HD Max $\\downarrow$ & SO (Org) $\\downarrow$ & ARAP $\\downarrow$ \\\\",
        "\\hline",
    ]

    for bm in benchmarks:
        m_tex = bm["model"].replace("_", "\\_")
        q_tex = f"{bm['quota_short']} ({bm['bones_count']})"
        s04_keys = ["s04_v1_v1", "s04_ue_v1", "s04_v1_ue", "s04_ue_ue"]
        items = [bm["step04"][k] for k in s04_keys if k in bm["step04"]]

        for idx, it in enumerate(items):
            c = it["complexity"]
            mt = it[m_field]
            m_lbl = (
                "\\textbf{Ours Bone + Ours QEM + SSDR}"
                if it["method_key"] == "s04_v1_v1"
                else it["method_short"].replace("_", "\\_")
            )
            row_cells = [
                m_tex if idx == 0 else "",
                q_tex if idx == 0 else "",
                m_lbl,
                f"{int(c['num_vertices']):,}",
                f"{int(c['num_triangles']):,}",
                str(int(c["max_bones_per_vertex"])),
                f"{int(c['num_loose_parts'])}/{int(c['num_loose_parts_welded'])}",
                f"{mt['l2_err']:.4g}" if mt["l2_err"] is not None else "-",
                f"{mt['hausdorff_mean']:.4g}" if mt["hausdorff_mean"] is not None else "-",
                f"{mt['hausdorff_max']:.4g}" if mt["hausdorff_max"] is not None else "-",
                f"{mt['org_loss']:.4g}" if mt["org_loss"] is not None else "-",
                f"{mt['arap_ratio']:.4g}" if mt["arap_ratio"] is not None else "-",
            ]
            lines.append(" & ".join(row_cells) + " \\\\")
        lines.append("\\hline")

    lines.extend([
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def generate_ue_comparison_dashboard(
    *,
    pd_rew_root: Path = DEFAULT_PD_REW_ROOT,
    pool: Path = DEFAULT_POOL,
    binding_path: Path = DEFAULT_BINDING,
    table_path: Path = DEFAULT_TABLE,
    output_dir: Path = Path("/workspace/ogd_reports/pdrew_ue_comparison_report_latest"),
    title: str = "OGD-LBS vs. UE5 Equal-Bone-Budget Report: Step 02 (Bone Reduction) / Step 03 (Bone + QEM) / Step 04 (Bone + QEM + SSDR)",
) -> dict[str, Any]:
    src = pd_rew_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import datalad.api as dl
    from pd_rew_v2.experiment import expand_experiment_table, query_cache_status
    from pd_rew_v2.pool import PipelinePool

    pipe = PipelinePool(pool)
    pipe.verify(check_annex_hashes=False)
    expanded = expand_experiment_table(binding_path, table_path)
    cache_rep = query_cache_status(binding_path, expanded, pipe)
    cache_by_row = {r["row_id"]: r["stage_statuses"] for r in cache_rep["rows"]}

    target_choices = {
        "s_max-4",
        "ue-s02_ue-s03_v1",
        "ue-s02_ue-s03_ue",
        "ue-s02_v1-s03_ue",
        "mod-s02_v1-s03_v1",
        "mod-s02_ue-s03_v1",
        "mod-s02_ue-s03_ue",
        "mod-s02_v1-s03_ue",
    }

    row_lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    handoff_dir_by_model: dict[str, str] = {}
    stage01_dir_by_model: dict[str, str] = {}

    for r in expanded["rows"]:
        w_choice = str((r.get("sweep_choices") or {}).get("weights") or "")
        if w_choice not in target_choices:
            continue
        flat = r.get("flat_parameters") or {}
        model = str(flat.get("data_prep_handoff.model_name"))
        variant = str(flat.get("ssdr_stage02.variant") or "")
        s02_ver = str(flat.get("ssdr_stage02.version") or "v1")
        s03_ver = str(flat.get("ssdr_stage03.version") or "v1")
        st_map = cache_by_row[r["row_id"]]

        if w_choice == "s_max-4":
            handoff_dir_by_model[model] = str(st_map["data_prep_handoff"]["data_dir"])
            stage01_dir_by_model[model] = str(st_map["ssdr_stage01"]["data_dir"])

        row_lookup[(model, variant, s02_ver, s03_ver)] = {
            "row": r,
            "stage_statuses": st_map,
        }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_dir = output_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    complexity_audit_rows: list[dict[str, Any]] = []
    step02_csv_rows: list[dict[str, Any]] = []
    step03_csv_rows: list[dict[str, Any]] = []
    step04_csv_rows: list[dict[str, Any]] = []
    benchmarks_payload: list[dict[str, Any]] = []

    seen_input_models: set[str] = set()

    for bm_spec in QUOTA_BENCHMARKS:
        model = bm_spec["model"]
        variant = bm_spec["variant"]

        # 1. Original Input rows (Raw .blend & Data1 Full Skeleton) once per model
        if model not in seen_input_models:
            seen_input_models.add(model)
            h_dd = handoff_dir_by_model[model]
            s01_dd = stage01_dir_by_model[model]

            if model in {"Griffin_A2", "Winter_Samurai"}:
                h_norm = _load_in_pool_normalized_metrics(
                    dl=dl, pool=pool, data_dir=h_dd, expected_subdir="data_prep_00"
                )
                complexity_audit_rows.append({
                    "model": model,
                    "tier_label": "0A. Raw Input (.blend)",
                    "method_label": "Raw Asset (Pre-VDB)",
                    "is_input": True,
                    "is_ours": False,
                    "is_ours_full": False,
                    "is_hybrid": False,
                    "data_dir": h_dd,
                    "complexity": h_norm["complexity"],
                })

            s01_norm = _load_in_pool_normalized_metrics(
                dl=dl, pool=pool, data_dir=s01_dd, expected_subdir="data1"
            )
            complexity_audit_rows.append({
                "model": model,
                "tier_label": "0B. Data1 Full Skeleton" if model in {"Griffin_A2", "Winter_Samurai"} else "0. Original Input (data1 Full)",
                "method_label": "Full Skeleton (Pre-Step02)",
                "is_input": True,
                "is_ours": False,
                "is_ours_full": False,
                "is_hybrid": False,
                "data_dir": s01_dd,
                "complexity": s01_norm["complexity"],
            })

        rec_v1_v1 = row_lookup[(model, variant, "v1", "v1")]
        rec_ue_v1 = row_lookup[(model, variant, "ue_v1", "v1")]
        rec_v1_ue = row_lookup[(model, variant, "v1", "ue_v1")]
        rec_ue_ue = row_lookup[(model, variant, "ue_v1", "ue_v1")]

        # 2. Step 02 (Bone Reduction)
        s02_v1_dd = str(rec_v1_v1["stage_statuses"]["ssdr_stage02"]["data_dir"])
        s02_ue_dd = str(rec_ue_v1["stage_statuses"]["ssdr_stage02"]["data_dir"])

        s02_v1_norm = _load_in_pool_normalized_metrics(dl=dl, pool=pool, data_dir=s02_v1_dd, expected_subdir="data2")
        s02_ue_norm = _load_in_pool_normalized_metrics(dl=dl, pool=pool, data_dir=s02_ue_dd, expected_subdir="data2")

        bones_count = int(s02_v1_norm["complexity"]["num_bones_total"])
        tier_full_label = f"{bm_spec['quota_short']} (B={bones_count})"

        s02_v1_dec = _decorate_norm_obj(
            s02_v1_norm,
            model=model,
            data_dir=s02_v1_dd,
            method_key="s02_v1",
            method_label="Ours Bone Reduction ★",
            method_short="Ours Bone Reduction ★",
        )
        s02_ue_dec = _decorate_norm_obj(
            s02_ue_norm,
            model=model,
            data_dir=s02_ue_dd,
            method_key="s02_ue",
            method_label="UE5 Bone Reduction",
            method_short="UE5 Bone Reduction",
        )

        for dec_s02, is_ours in [(s02_v1_dec, True), (s02_ue_dec, False)]:
            complexity_audit_rows.append({
                "model": model,
                "tier_label": f"Step 02 (Bone Reduction) — {tier_full_label}",
                "method_label": dec_s02["method_label"],
                "is_input": False,
                "is_ours": is_ours,
                "is_ours_full": False,
                "is_hybrid": False,
                "data_dir": dec_s02["data_dir"],
                "complexity": dec_s02["complexity"],
            })
            step02_csv_rows.append({
                "model": model,
                "quota_tier": bm_spec["quota_short"],
                "bones_total": bones_count,
                "method": dec_s02["method_label"],
                "data_dir": dec_s02["data_dir"],
                **dec_s02["complexity"],
                **{f"raw_{k}": v for k, v in dec_s02["metrics_raw"].items()},
                **{f"scaled_{k}": v for k, v in dec_s02["metrics_scaled"].items()},
            })

        # 3. Step 03 (Bone + QEM) — 4 branches
        s03_branches_spec = [
            ("s03_v1_v1", rec_v1_v1, "Ours Bone + Ours QEM ★", "Ours Bone + Ours QEM ★", True, False),
            ("s03_ue_v1", rec_ue_v1, "UE5 Bone + Ours QEM", "UE5 Bone + Ours QEM", False, True),
            ("s03_v1_ue", rec_v1_ue, "Ours Bone + UE5 QEM", "Ours Bone + UE5 QEM", False, True),
            ("s03_ue_ue", rec_ue_ue, "UE5 Bone + UE5 QEM", "UE5 Bone + UE5 QEM", False, False),
        ]
        step03_map: dict[str, Any] = {}
        for b_key, b_rec, m_lbl, m_short, is_ours_f, is_hyb in s03_branches_spec:
            dd = str(b_rec["stage_statuses"]["ssdr_stage03"]["data_dir"])
            norm_obj = _load_in_pool_normalized_metrics(dl=dl, pool=pool, data_dir=dd, expected_subdir="data3")
            dec = _decorate_norm_obj(
                norm_obj,
                model=model,
                data_dir=dd,
                method_key=b_key,
                method_label=m_lbl,
                method_short=m_short,
            )
            step03_map[b_key] = dec
            complexity_audit_rows.append({
                "model": model,
                "tier_label": f"Step 03 (Bone + QEM) — {tier_full_label}",
                "method_label": m_lbl,
                "is_input": False,
                "is_ours": is_ours_f,
                "is_ours_full": False,
                "is_hybrid": is_hyb,
                "data_dir": dd,
                "complexity": dec["complexity"],
            })
            step03_csv_rows.append({
                "model": model,
                "quota_tier": bm_spec["quota_short"],
                "bones_total": bones_count,
                "branch_key": b_key,
                "method": m_lbl,
                "data_dir": dd,
                **dec["complexity"],
                **{f"raw_{k}": v for k, v in dec["metrics_raw"].items()},
                **{f"scaled_{k}": v for k, v in dec["metrics_scaled"].items()},
            })

        # 4. Step 04 (Bone + QEM + SSDR) — 4 branches
        s04_branches_spec = [
            ("s04_v1_v1", rec_v1_v1, "Ours Bone + Ours QEM + SSDR ★", "Ours Bone + Ours QEM + SSDR ★", True, False),
            ("s04_ue_v1", rec_ue_v1, "UE5 Bone + Ours QEM + SSDR", "UE5 Bone + Ours QEM + SSDR", False, True),
            ("s04_v1_ue", rec_v1_ue, "Ours Bone + UE5 QEM + SSDR", "Ours Bone + UE5 QEM + SSDR", False, True),
            ("s04_ue_ue", rec_ue_ue, "UE5 Bone + UE5 QEM + SSDR", "UE5 Bone + UE5 QEM + SSDR", False, False),
        ]
        step04_map: dict[str, Any] = {}
        for b_key, b_rec, m_lbl, m_short, is_ours_f, is_hyb in s04_branches_spec:
            dd = str(b_rec["stage_statuses"]["ssdr_stage04"]["data_dir"])
            norm_obj = _load_in_pool_normalized_metrics(dl=dl, pool=pool, data_dir=dd, expected_subdir="data4")
            dec = _decorate_norm_obj(
                norm_obj,
                model=model,
                data_dir=dd,
                method_key=b_key,
                method_label=m_lbl,
                method_short=m_short,
            )
            step04_map[b_key] = dec
            complexity_audit_rows.append({
                "model": model,
                "tier_label": f"Step 04 (Bone + QEM + SSDR) — {tier_full_label}",
                "method_label": m_lbl,
                "is_input": False,
                "is_ours": is_ours_f,
                "is_ours_full": is_ours_f,
                "is_hybrid": is_hyb,
                "data_dir": dd,
                "complexity": dec["complexity"],
            })
            step04_csv_rows.append({
                "model": model,
                "quota_tier": bm_spec["quota_short"],
                "bones_total": bones_count,
                "branch_key": b_key,
                "method": m_lbl,
                "data_dir": dd,
                **dec["complexity"],
                **{f"raw_{k}": v for k, v in dec["metrics_raw"].items()},
                **{f"scaled_{k}": v for k, v in dec["metrics_scaled"].items()},
            })

        benchmarks_payload.append({
            "model": model,
            "variant": variant,
            "quota_short": bm_spec["quota_short"],
            "quota_display": bm_spec["quota_display"],
            "bones_count": bones_count,
            "step02": {
                "s02_v1": s02_v1_dec,
                "s02_ue": s02_ue_dec,
            },
            "step03": step03_map,
            "step04": step04_map,
        })

    _write_csv(agg_dir / "complexity_audit_table.csv", [
        {"model": r["model"], "tier": r["tier_label"], "method": r["method_label"], "data_dir": r["data_dir"], **r["complexity"]}
        for r in complexity_audit_rows
    ])
    _write_csv(agg_dir / "step02_comparison_summary.csv", step02_csv_rows)
    _write_csv(agg_dir / "step03_comparison_summary.csv", step03_csv_rows)
    _write_csv(agg_dir / "step04_comparison_summary.csv", step04_csv_rows)

    latex_scaled = _build_latex_table(benchmarks_payload, scale_mode="scaled")
    latex_raw = _build_latex_table(benchmarks_payload, scale_mode="raw")
    (agg_dir / "paper_table_equal_bone_budget_scaled.tex").write_text(latex_scaled, encoding="utf-8")
    (agg_dir / "paper_table_equal_bone_budget_raw.tex").write_text(latex_raw, encoding="utf-8")

    payload = {
        "meta": {
            "title": title,
            "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pool": str(pool),
            "benchmark_count": len(benchmarks_payload),
            "model_scales": MODEL_SCALES,
        },
        "metricsMeta": METRICS_META,
        "complexityAuditRows": complexity_audit_rows,
        "benchmarks": benchmarks_payload,
        "latex": {
            "scaled": latex_scaled,
            "raw": latex_raw,
        },
    }

    strict_json_write(output_dir / "ue_comparison_dashboard_data.json", payload)

    tpl_path = HERE / "report_templates" / "ogd_ue_comparison_template.html"
    tpl_text = tpl_path.read_text(encoding="utf-8")
    html_text = tpl_text.replace("__TITLE_HTML__", html.escape(title)).replace(
        "__DATA_JSON__", json.dumps(payload, ensure_ascii=False, allow_nan=False)
    )
    report_html_path = output_dir / "UE_Comparison_Report.html"
    report_html_path.write_text(html_text, encoding="utf-8")

    zip_name = "ogd_lbs_ue5_equal_bone_budget_report_bundle.zip"
    zip_path = output_dir / "downloads" / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #f4f6f9; color: #0f172a; }}
main {{ background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.06); }}
.btn {{ display: inline-block; padding: 11px 18px; border-radius: 7px; text-decoration: none; font-weight: 600; margin-right: 10px; color: #fff; }}
.btn-primary {{ background: #1C328A; }}
.btn-teal {{ background: #0f766e; }}
li {{ margin: 10px 0; }}
a {{ color: #1C328A; font-weight: 600; }}
</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<p>All complexity, loose-parts, Step 02 (Bone Reduction), Step 03 (Bone + QEM), and Step 04 (Bone + QEM + SSDR) metrics are sourced from in-pool <code>normalized_metrics_v1.json</code> files.</p>
<p style="margin: 22px 0;">
  <a class="btn btn-primary" href="UE_Comparison_Report.html">📊 Open Interactive Equal-Bone-Budget Dashboard</a>
  <a class="btn btn-teal" href="downloads/{html.escape(zip_name)}" download>📦 Download Full Report ZIP</a>
</p>
<h2>Direct Data &amp; Paper Table Exports</h2>
<ul>
  <li><a href="aggregate/paper_table_equal_bone_budget_scaled.tex" download>📄 Paper LaTeX Table (Model-Scaled)</a></li>
  <li><a href="aggregate/paper_table_equal_bone_budget_raw.tex" download>📄 Paper LaTeX Table (Raw Unscaled)</a></li>
  <li><a href="aggregate/complexity_audit_table.csv" download>📐 Input &amp; Multi-Stage Complexity / Loose-Parts Audit (CSV)</a></li>
  <li><a href="aggregate/step02_comparison_summary.csv" download>📊 Step 02 (Bone Reduction) High-Poly Summary (CSV)</a></li>
  <li><a href="aggregate/step03_comparison_summary.csv" download>📊 Step 03 (Bone + QEM) Low-Poly Summary (CSV)</a></li>
  <li><a href="aggregate/step04_comparison_summary.csv" download>📊 Step 04 (Bone + QEM + SSDR) Final Optimized Summary (CSV)</a></li>
  <li><a href="ue_comparison_dashboard_data.json" download>🗂️ Complete Dashboard JSON Payload</a></li>
</ul>
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    bundle_files = [
        "UE_Comparison_Report.html",
        "index.html",
        "ue_comparison_dashboard_data.json",
        "aggregate/complexity_audit_table.csv",
        "aggregate/step02_comparison_summary.csv",
        "aggregate/step03_comparison_summary.csv",
        "aggregate/step04_comparison_summary.csv",
        "aggregate/paper_table_equal_bone_budget_scaled.tex",
        "aggregate/paper_table_equal_bone_budget_raw.tex",
    ]
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in bundle_files:
            zf.write(output_dir / rel, arcname=rel)

    return {
        "output_dir": str(output_dir),
        "report_html": str(report_html_path),
        "index_html": str(output_dir / "index.html"),
        "zip_path": str(zip_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OGD-LBS vs UE5 Equal-Bone-Budget Report.")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/ogd_reports/pdrew_ue_comparison_report_latest"))
    parser.add_argument("--serve-daemon", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    res = generate_ue_comparison_dashboard(output_dir=args.output_dir)
    print("=" * 92, flush=True)
    print("EQUAL-BONE-BUDGET & UE5 ABLATION REPORT GENERATED (SEPARATE STEP 02 / 03 / 04)", flush=True)
    print(f"REPORT HTML   : {res['report_html']}", flush=True)
    print(f"INDEX HTML    : {res['index_html']}", flush=True)
    print(f"DOWNLOAD ZIP  : {res['zip_path']}", flush=True)
    print("=" * 92, flush=True)

    if args.serve_daemon:
        srv = start_daemon_http_server(
            Path(res["output_dir"]),
            "UE_Comparison_Report.html",
            port=int(args.port),
        )
        print(f"🌐 HTTP Server Active (PID={srv['pid']}, Container Port={srv['port']})", flush=True)
        print(f"👉 Host LAN URL (Port 8081->8080): http://10.241.1.178:8081/UE_Comparison_Report.html", flush=True)
        print(f"👉 Container Direct URL          : {srv['report_url']}", flush=True)
        print(f"👉 Download Center               : http://10.241.1.178:8081/index.html", flush=True)


if __name__ == "__main__":
    main()
