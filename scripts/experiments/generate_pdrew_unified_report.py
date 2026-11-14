#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Unified Multi-Page OGD-LBS Experiment Report Suite Generator.

Generates all 9 interconnected pages into a single served directory
(default: /workspace/ogd_reports/pdrew_ue_comparison_report_latest):
  1. index.html                 — Suite Hub & Download Center
  2. UE_Comparison_Report.html  — UE5 Equal-Bone-Budget (Step 02 / 03 / 04)
  3. CV_Report.html             — 4-Fold Cross-Validation & Paired t-Test
  4. Delta_Report.html          — OFAT Sweep: VDB Offset (δ)
  5. Gamma_Report.html          — OFAT Sweep: Weight Allocation (α/β/γ)
  6. S_max_Report.html          — OFAT Sweep: Sparsity Limit (s_max)
  7. Tau_Report.html            — OFAT Sweep: Pruning Threshold (τ)
  8. Omega_b_Report.html        — OFAT Sweep: QEM Boundary Penalty (ω_b)
  9. Render_QA.html             — Numbered Thumbnail Optical & Neural QA Inspector
"""

import argparse
import datetime as dt
import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import report_core  # noqa: E402
report_core.ensure_pdrew_python_env(__file__)

from generate_pdrew_gamma_report import (  # noqa: E402
    DEFAULT_BINDING,
    DEFAULT_PD_REW_ROOT,
    DEFAULT_POOL,
    DEFAULT_TABLE,
    PARAM_CHOICE_PREFIXES,
    PARAM_TITLES,
    materialize_pool_file,
    safe_pool_data_dir,
    start_daemon_http_server,
    strict_json_write,
)
from generate_pdrew_ue_comparison_report import (  # noqa: E402
    METRICS_META,
    MODEL_SCALES,
    _decorate_norm_obj,
    _load_in_pool_normalized_metrics,
    generate_ue_comparison_dashboard,
    render_ue_html_from_cached_payload,
)
from generate_pdrew_cv_report import (  # noqa: E402
    generate_cv_dashboard,
    render_cv_dashboard_html,
)
import subprocess

SWEEP_SPECS = [
    {
        "family": "delta",
        "filename": "Delta_Report.html",
        "param_display": "δ (VDB Offset)",
        "default_by_model": {"Griffin_A2": "15.0", "Winter_Samurai": "1.5"},
    },
    {
        "family": "gamma",
        "filename": "Gamma_Report.html",
        "param_display": "γ (Child Bone Weight)",
        "default_by_model": {"*": "0.2"},
    },
    {
        "family": "s_max",
        "filename": "S_max_Report.html",
        "param_display": "s_max (Max Bones/Vertex)",
        "default_by_model": {"*": "4"},
    },
    {
        "family": "tau",
        "filename": "Tau_Report.html",
        "param_display": "τ (Mask Pruning Threshold)",
        "default_by_model": {"*": "adaptive"},
    },
    {
        "family": "omega_b",
        "filename": "Omega_b_Report.html",
        "param_display": "ω_b (QEM Boundary Penalty)",
        "default_by_model": {"*": "1000.0"},
    },
]


def _extract_swept_value(flat_params: dict[str, Any], family: str) -> tuple[Any, str]:
    if family == "delta":
        v = flat_params.get("data_prep_stage02.offset")
        return v, str(v)
    if family == "gamma":
        a = flat_params.get("ssdr_stage02.par_weight", 0.5)
        b = flat_params.get("ssdr_stage02.bone_weight", 0.3)
        g = flat_params.get("ssdr_stage02.child_weight", 0.2)
        return float(g), f"γ={g} (α={a},β={b})"
    if family == "s_max":
        v = int(flat_params.get("ssdr_stage04.s_max", 4))
        return v, str(v)
    if family == "tau":
        mode = str(flat_params.get("ssdr_stage04.mask_mode", "adaptive"))
        thr = flat_params.get("ssdr_stage04.mask_threshold", -1.0)
        if mode.lower() == "adaptive":
            return "adaptive", "adaptive"
        return float(thr), str(thr)
    if family == "omega_b":
        v = float(flat_params.get("ssdr_stage03.boundary_weight", 1000.0))
        return v, str(v)
    return "", ""


def _sort_key_val(v: Any) -> tuple[int, float | str]:
    if isinstance(v, (int, float)):
        return (0, float(v))
    try:
        return (0, float(str(v)))
    except Exception:
        return (1, str(v))


def build_all_ofat_sweep_pages(
    *,
    dl: Any,
    pool: Path,
    expanded: dict[str, Any],
    cache_by_row: dict[str, Any],
    output_dir: Path,
    renders_by_dd: dict[str, Any],
    summary_by_dd: dict[str, Any],
    model_baseline_dd: dict[str, str],
    ui_only: bool = False,
) -> list[dict[str, Any]]:
    shared_assets = report_core.load_shared_frontend_assets(output_dir)
    tpl_path = HERE / "report_templates" / "ogd_sweep_dashboard_template.html"
    tpl_text = tpl_path.read_text(encoding="utf-8")
    bundled_dummy: list[str] = []
    results = []

    for spec in SWEEP_SPECS:
        family = spec["family"]
        html_name = spec["filename"]
        json_path = output_dir / f"sweep_{family}_dashboard_data.json"
        title = PARAM_TITLES.get(family, f"OGD-LBS {family} Sweep Report")

        if ui_only and json_path.is_file():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        else:
            prefixes = PARAM_CHOICE_PREFIXES[family]
            matched_rows = [
                r for r in expanded["rows"]
                if any(str((r.get("sweep_choices") or {}).get("weights") or "").startswith(p) for p in prefixes)
            ]
            # For gamma, tau, omega_b, also include the default s_max-4 row so the baseline ★ point is included!
            if family in {"gamma", "omega_b"}:
                default_rows = [
                    r for r in expanded["rows"]
                    if str((r.get("sweep_choices") or {}).get("weights") or "") == "s_max-4"
                ]
                matched_rows = default_rows + matched_rows

            sweep_rows_out = []
            models_seen = set()

            for r in matched_rows:
                rid = r["row_id"]
                st_map = cache_by_row[rid]
                s04_info = st_map["ssdr_stage04"]
                if s04_info["status"] != "HIT" or not s04_info.get("data_dir"):
                    continue
                dd = str(s04_info["data_dir"])
                flat = r.get("flat_parameters") or {}
                model = str(flat.get("data_prep_handoff.model_name"))
                models_seen.add(model)

                raw_val, disp_val = _extract_swept_value(flat, family)
                def_map = spec["default_by_model"]
                exp_def = def_map.get(model, def_map.get("*", ""))
                is_def = (str(raw_val) == str(exp_def)) or (
                    str((r.get("sweep_choices") or {}).get("weights") or "") == "s_max-4"
                )

                norm_obj = _load_in_pool_normalized_metrics(dl=dl, pool=pool, data_dir=dd, expected_subdir="data4")
                dec = _decorate_norm_obj(
                    norm_obj,
                    model=model,
                    stage_label=f"OFAT Sweep ({spec['param_display']})",
                    quota_label=disp_val,
                    data_dir=dd,
                    method_key=f"{family}_{disp_val}",
                    method_label=f"{spec['param_display']} = {disp_val}" + (" ★" if is_def else ""),
                    method_short=disp_val,
                )

                if dd not in renders_by_dd:
                    renders_by_dd[dd] = report_core.collect_entry_renders(
                        dl=dl,
                        pool=pool,
                        data_dir=dd,
                        expected_subdir="data4",
                        output_dir=output_dir,
                        bundled_rel_files=bundled_dummy,
                        materialize_fn=materialize_pool_file,
                        safe_dir_fn=safe_pool_data_dir,
                    )
                summary_by_dd[dd] = dec

                # If this specific OFAT case doesn't have its own render yet, link fallback to model's baseline s_max-4 render
                has_own_render = bool((renders_by_dd.get(dd) or {}).get("items"))
                render_dd = dd if has_own_render else model_baseline_dd.get(model, dd)

                sweep_rows_out.append({
                    "row_id": rid,
                    "model": model,
                    "data_dir": dd,
                    "render_data_dir": render_dd,
                    "has_own_render": has_own_render,
                    "param_value_raw": raw_val,
                    "param_value_display": disp_val,
                    "is_default": is_def,
                    "complexity": dec["complexity"],
                    "metrics_raw": dec["metrics_raw"],
                    "metrics_scaled": dec["metrics_scaled"],
                    "frame_stats_raw": dec["frame_stats_raw"],
                    "frame_stats_scaled": dec["frame_stats_scaled"],
                })

            sweep_rows_out.sort(key=lambda x: (x["model"], _sort_key_val(x["param_value_raw"])))

            # Collect subset of renders_by_dd & summary_by_dd referenced by this sweep page
            page_dds = {r["data_dir"] for r in sweep_rows_out} | {r["render_data_dir"] for r in sweep_rows_out}
            payload = {
                "meta": {
                    "title": title,
                    "family": family,
                    "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "row_count": len(sweep_rows_out),
                },
                "param_family": family,
                "param_display": spec["param_display"],
                "models": sorted(models_seen),
                "metricsMeta": METRICS_META,
                "rows": sweep_rows_out,
                "rendersByDataDir": {k: renders_by_dd[k] for k in page_dds if k in renders_by_dd},
                "entrySummaryByDataDir": {k: summary_by_dd[k] for k in page_dds if k in summary_by_dd},
            }
            strict_json_write(json_path, payload)

        nav_html = report_core.build_suite_nav_html(html_name)
        html_text = (
            tpl_text.replace("__TITLE_HTML__", html.escape(title))
            .replace("__PARAM_LABEL__", html.escape(spec["param_display"]))
            .replace("__SUITE_NAV_HTML__", nav_html)
            .replace("__SHARED_CORE_CSS__", shared_assets["core_css"])
            .replace("__SHARED_CHART_ENGINE_JS__", shared_assets["chart_engine_js"])
            .replace("__SHARED_RENDER_DOCK_JS__", shared_assets["render_dock_js"])
            .replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False, allow_nan=False))
        )
        out_html = output_dir / html_name
        out_html.write_text(html_text, encoding="utf-8")
        results.append({"family": family, "html": str(out_html), "rows": len(payload.get("rows") or [])})

    return results


def build_unified_suite_hub_index(output_dir: Path, total_renders: int) -> Path:
    nav_html = report_core.build_suite_nav_html("index.html")
    shared_assets = report_core.load_shared_frontend_assets(output_dir)

    cards_spec = [
        (
            "⚔️ UE5 Equal-Bone-Budget Multi-Stage Comparison",
            "UE_Comparison_Report.html",
            "Step 02 (Bone Reduction) → Step 03 (Bone + QEM) → Step 04 (Bone + QEM + SSDR) across 7 equal bone-quota benchmarks with interactive Right-Side Render Dock.",
            "#1C328A",
        ),
        (
            "🧪 4-Fold Cross-Validation & Paired t-Test",
            "CV_Report.html",
            "60-case held-out unseen test frame evaluation comparing ssdr12, ssdr14, and ssdr14v ★ with Holm-Bonferroni p-values, Cohen's dz, and Win-Rate Pareto.",
            "#0f766e",
        ),
        (
            "📈 OFAT Sweep: VDB Offset (δ)",
            "Delta_Report.html",
            "Data Prep Stage 02 VDB offset sweep on Griffin_A2 (5..30) and Winter_Samurai (0.5..3.0).",
            "#2563eb",
        ),
        (
            "📈 OFAT Sweep: Weight Allocation (α / β / γ)",
            "Gamma_Report.html",
            "Step 02 parent/bone/child weight transfer ratio sweep across all 5 models.",
            "#7c3aed",
        ),
        (
            "📈 OFAT Sweep: Sparsity Limit (s_max)",
            "S_max_Report.html",
            "Step 04 maximum bones per vertex sweep (s_max = 1, 2, 4 ★, 6, 8) across all 5 models.",
            "#d97706",
        ),
        (
            "📈 OFAT Sweep: Pruning Threshold (τ)",
            "Tau_Report.html",
            "Step 04 weight mask pruning threshold sweep (adaptive ★ vs fixed 1e-4 .. 0.05).",
            "#059669",
        ),
        (
            "📈 OFAT Sweep: QEM Boundary Penalty (ω_b)",
            "Omega_b_Report.html",
            "Step 03 QEM boundary preservation penalty sweep (100 .. 5000).",
            "#dc2626",
        ),
        (
            "🔍 Render QA Inspector (#ID Reverse Lookup)",
            "Render_QA.html",
            f"Browse all {total_renders} numbered 3D RGBA renders with Optical & Neural Guardrail badges, layer stacking, and one-click faulty ID lookup.",
            "#0284c7",
        ),
    ]

    cards_html = "".join(
        f'<div class="section-card" style="border-top:4px solid {color};display:flex;flex-direction:column;justify-content:space-between;">'
        f'<div><h3 style="margin-top:0;color:#0f172a;">{html.escape(title)}</h3>'
        f'<p style="color:#475569;font-size:0.86rem;">{html.escape(desc)}</p></div>'
        f'<div style="margin-top:12px;"><a class="btn btn-primary" href="{html.escape(href)}">Open Report →</a></div>'
        f"</div>"
        for title, href, desc, color in cards_spec
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OGD-LBS Unified Experiment Report Suite</title>
<style>
{shared_assets["core_css"]}
.hub-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 18px;
  margin-top: 18px;
}}
</style>
</head>
<body>
{nav_html}
<div style="max-width:1480px;margin:24px auto;padding:0 20px 60px;">
  <div class="section-card" style="background:linear-gradient(135deg,#0f172a,#1e293b);color:#f8fafc;border:none;padding:26px 30px;">
    <h1 style="margin:0 0 8px 0;font-size:1.55rem;color:#fff;">📊 OGD-LBS Unified Experiment &amp; Visual Augmentation Suite</h1>
    <p style="margin:0 0 18px 0;color:#cbd5e1;font-size:0.92rem;">
      All reports share the unified frontend architecture, 7 core metrics, and {total_renders} in-pool 3D RGBA renders.
    </p>
    <div style="display:flex;flex-wrap:wrap;gap:10px;">
      <a class="btn btn-teal" href="UE_Comparison_Report.html">⚔️ Open UE5 Comparison</a>
      <a class="btn btn-primary" href="CV_Report.html">🧪 Open 4-Fold CV Report</a>
      <a class="btn btn-outline" href="Render_QA.html" style="background:#334155;color:#fff;border-color:#475569;">🔍 Open Render QA (#ID)</a>
      <a class="btn btn-outline" href="downloads/ogd_lbs_ue5_equal_bone_budget_report_bundle.zip" download>📦 Download Full ZIP Bundle</a>
    </div>
  </div>

  <div class="hub-grid">{cards_html}</div>
</div>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(doc, encoding="utf-8")
    return index_path


def generate_unified_suite(
    *,
    output_dir: Path = Path("/workspace/ogd_reports/pdrew_ue_comparison_report_latest"),
    ui_only: bool = False,
) -> dict[str, Any]:
    src = DEFAULT_PD_REW_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import datalad.api as dl
    from pd_rew_v2.experiment import expand_experiment_table, query_cache_status
    from pd_rew_v2.pool import PipelinePool

    output_dir = output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. UE5 Equal-Bone-Budget Report
    if ui_only and (output_dir / "ue_comparison_dashboard_data.json").is_file():
        ue_res = render_ue_html_from_cached_payload(output_dir=output_dir)
    else:
        ue_res = generate_ue_comparison_dashboard(output_dir=output_dir)

    ue_payload = json.loads((output_dir / "ue_comparison_dashboard_data.json").read_text(encoding="utf-8"))
    renders_by_dd: dict[str, Any] = dict(ue_payload.get("rendersByDataDir") or {})
    summary_by_dd: dict[str, Any] = dict(ue_payload.get("entrySummaryByDataDir") or {})

    # Map each model to its default s_max-4 stage04 data_dir for fallback renders
    model_baseline_dd: dict[str, str] = {}
    for bm in ue_payload.get("benchmarks") or []:
        s04_ours = (bm.get("step04") or {}).get("s04_v1_v1")
        if s04_ours and bm["model"] not in model_baseline_dd:
            model_baseline_dd[bm["model"]] = s04_ours["data_dir"]

    # 2. 4-Fold CV Report (generated directly into the same unified output_dir without wiping UE files!)
    cv_json_path = output_dir / "cv_dashboard_data.json"
    if ui_only and cv_json_path.is_file():
        cv_payload = json.loads(cv_json_path.read_text(encoding="utf-8"))
        cv_html = render_cv_dashboard_html(
            title=str((cv_payload.get("meta") or {}).get("title") or "OGD-LBS 4-Fold CV Report"),
            payload=cv_payload,
            output_dir=output_dir,
        )
        (output_dir / "CV_Report.html").write_text(cv_html, encoding="utf-8")
    else:
        cv_tmp_dir = output_dir / "_tmp_cv_build"
        cv_res = generate_cv_dashboard(output_dir=cv_tmp_dir)
        import shutil
        shutil.copyfile(cv_tmp_dir / "CV_Report.html", output_dir / "CV_Report.html")
        shutil.copyfile(cv_tmp_dir / "cv_dashboard_data.json", output_dir / "cv_dashboard_data.json")
        for f_agg in (cv_tmp_dir / "aggregate").glob("*"):
            dst_a = output_dir / "aggregate" / f_agg.name
            dst_a.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f_agg, dst_a)
        for f_dl in (cv_tmp_dir / "downloads").glob("*"):
            dst_d = output_dir / "downloads" / f_dl.name
            dst_d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f_dl, dst_d)
        shutil.rmtree(cv_tmp_dir, ignore_errors=True)

    # 3. All 5 OFAT Parameter Sweep Reports (Delta, Gamma, S_max, Tau, Omega_b)
    pipe = PipelinePool(DEFAULT_POOL)
    expanded = expand_experiment_table(DEFAULT_BINDING, DEFAULT_TABLE)
    cache_rep = query_cache_status(DEFAULT_BINDING, expanded, pipe)
    cache_by_row = {r["row_id"]: r["stage_statuses"] for r in cache_rep["rows"]}

    sweep_pages = build_all_ofat_sweep_pages(
        dl=dl,
        pool=DEFAULT_POOL,
        expanded=expanded,
        cache_by_row=cache_by_row,
        output_dir=output_dir,
        renders_by_dd=renders_by_dd,
        summary_by_dd=summary_by_dd,
        model_baseline_dd=model_baseline_dd,
        ui_only=ui_only,
    )

    # 4. Render QA Inspector (Render_QA.html) in the same output_dir via ogd-lbs python
    ogd_py_bin = "/opt/conda/envs/ogd-lbs/bin/python"
    qa_cmd = [
        ogd_py_bin if Path(ogd_py_bin).is_file() else sys.executable,
        str(HERE / "render_qa_inspector.py"),
        "--sheet-dir",
        str(output_dir),
        "--build-sheet",
        "--json",
    ]
    if ui_only and (output_dir / "contact_sheet_index.json").is_file():
        qa_cmd.append("--ui-only")
    qa_proc = subprocess.run(qa_cmd, check=True, capture_output=True, text=True)
    qa_idx = json.loads((output_dir / "contact_sheet_index.json").read_text(encoding="utf-8"))

    # 5. Rebuild unified Suite Hub index.html (since build_contact_sheets wrote index.html, overwrite index.html with Suite Hub!)
    hub_index = build_unified_suite_hub_index(output_dir, total_renders=qa_idx["total_thumbnails"])

    return {
        "output_dir": str(output_dir),
        "hub_index": str(hub_index),
        "ue_report": str(output_dir / "UE_Comparison_Report.html"),
        "cv_report": str(output_dir / "CV_Report.html"),
        "sweep_pages": sweep_pages,
        "render_qa_report": str(output_dir / "Render_QA.html"),
        "total_thumbnails": qa_idx["total_thumbnails"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Unified Multi-Page OGD-LBS Report Suite.")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/ogd_reports/pdrew_ue_comparison_report_latest"))
    parser.add_argument("--ui-only", action="store_true", help="Fast-rebuild HTML from cached JSON where available.")
    parser.add_argument("--serve-daemon", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    res = generate_unified_suite(output_dir=args.output_dir, ui_only=args.ui_only)
    print("=" * 92, flush=True)
    print("UNIFIED 9-PAGE OGD-LBS REPORT SUITE GENERATED (ZERO 404s)", flush=True)
    print(f"  1. Suite Hub      : {res['hub_index']}", flush=True)
    print(f"  2. UE5 Comparison : {res['ue_report']}", flush=True)
    print(f"  3. 4-Fold CV      : {res['cv_report']}", flush=True)
    for idx, sp in enumerate(res["sweep_pages"], start=4):
        print(f"  {idx}. OFAT ({sp['family']:7s}): {sp['html']} (rows={sp['rows']})", flush=True)
    print(f"  9. Render QA      : {res['render_qa_report']} ({res['total_thumbnails']} thumbnails)", flush=True)
    print("=" * 92, flush=True)

    if args.serve_daemon:
        srv = start_daemon_http_server(
            Path(res["output_dir"]),
            "UE_Comparison_Report.html",
            port=int(args.port),
        )
        print(f"🌐 HTTP Server Active (PID={srv['pid']}, Container Port={srv['port']})", flush=True)
        print("👉 Host LAN Suite Hub : http://10.241.1.178:8081/index.html", flush=True)
        print("👉 Host LAN UE5 Report: http://10.241.1.178:8081/UE_Comparison_Report.html", flush=True)


if __name__ == "__main__":
    main()
