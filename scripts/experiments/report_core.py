#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Shared Backend Core for OGD-LBS Interactive Reports.

Centralizes:
  1. Automatic pd-rew environment switching when datalad/pd_rew_v2 is required
  2. In-pool render discovery, materialization, and 0644 safe copying
  3. Multi-page Suite Navigation Bar HTML generation
  4. Shared frontend CSS/JS asset loading & standalone HTML assembly
"""

import csv
import html
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SHARED_TPL_DIR = HERE / "report_templates" / "shared"

SUITE_PAGES = [
    ("index.html", "🏠 Suite Hub"),
    ("UE_Comparison_Report.html", "⚔️ UE5 Equal-Bone-Budget (Step 02/03/04)"),
    ("CV_Report.html", "🧪 4-Fold CV & Paired t-Test"),
    ("Delta_Report.html", "📈 OFAT: δ (VDB Offset)"),
    ("Gamma_Report.html", "📈 OFAT: α/β/γ (Weights)"),
    ("S_max_Report.html", "📈 OFAT: s_max (Sparsity)"),
    ("Tau_Report.html", "📈 OFAT: τ (Pruning)"),
    ("Omega_b_Report.html", "📈 OFAT: ω_b (QEM Boundary)"),
    ("Render_QA.html", "🔍 Render QA Inspector (#ID)"),
]


def ensure_pdrew_python_env(script_file: str | Path) -> None:
    try:
        import datalad.api as _dl  # noqa: F401
    except ModuleNotFoundError:
        pdrew_py = Path("/opt/conda/envs/pd-rew/bin/python")
        if pdrew_py.is_file() and Path(sys.executable).resolve() != pdrew_py.resolve():
            os.execv(str(pdrew_py), [str(pdrew_py), str(Path(script_file).resolve()), *sys.argv[1:]])


def build_suite_nav_html(active_filename: str) -> str:
    links = []
    for href, label in SUITE_PAGES:
        cls = ' class="active"' if href == active_filename else ""
        links.append(f'<a href="{html.escape(href)}"{cls}>{html.escape(label)}</a>')
    return (
        '<nav class="suite-nav-bar" aria-label="OGD-LBS Unified Report Navigation">'
        '<span class="suite-nav-brand">📊 OGD-LBS Experiment Suite</span>'
        f'<div class="suite-nav-links">{"".join(links)}</div>'
        "</nav>"
    )


def load_shared_frontend_assets(output_dir: Path | None = None) -> dict[str, str]:
    css_path = SHARED_TPL_DIR / "ogd_core_styles.css"
    chart_js_path = SHARED_TPL_DIR / "ogd_chart_engine.js"
    dock_js_path = SHARED_TPL_DIR / "ogd_render_dock.js"

    css_text = css_path.read_text(encoding="utf-8")
    chart_js = chart_js_path.read_text(encoding="utf-8")
    dock_js = dock_js_path.read_text(encoding="utf-8")

    if output_dir is not None:
        dst_shared = output_dir / "shared"
        dst_shared.mkdir(parents=True, exist_ok=True)
        (dst_shared / "ogd_core_styles.css").write_text(css_text, encoding="utf-8")
        (dst_shared / "ogd_chart_engine.js").write_text(chart_js, encoding="utf-8")
        (dst_shared / "ogd_render_dock.js").write_text(dock_js, encoding="utf-8")

    return {
        "core_css": css_text,
        "chart_engine_js": chart_js,
        "render_dock_js": dock_js,
    }


def chmod_tree_writable(root_dir: Path) -> None:
    if not root_dir.exists():
        return
    for r_p, _dirs, files in os.walk(root_dir):
        for fn in files:
            try:
                os.chmod(os.path.join(r_p, fn), 0o644)
            except Exception:
                pass


def collect_entry_renders(
    *,
    dl: Any,
    pool: Path,
    data_dir: str,
    expected_subdir: str,
    output_dir: Path,
    bundled_rel_files: list[str],
    materialize_fn: Any,
    safe_dir_fn: Any,
) -> dict[str, Any]:
    r_root = safe_dir_fn(pool, data_dir)
    entry_name = r_root.name

    search_roots = []
    for cand_sub in (expected_subdir, "data4", "data3", "data2", "data1", "data0_json", "data_prep_00"):
        if cand_sub and (r_root / cand_sub / "renders").is_dir():
            if (r_root / cand_sub) not in search_roots:
                search_roots.append(r_root / cand_sub)
    if (r_root / "renders").is_dir():
        search_roots.append(r_root)

    seen_keys: set[str] = set()
    items: list[dict[str, Any]] = []
    cameras_set: set[str] = set()
    frames_set: set[str] = set()
    types_set: set[str] = set()

    for s_root in search_roots:
        renders_dir = s_root / "renders"
        for meta_path in sorted(renders_dir.rglob("render_meta.json")):
            png_path = meta_path.parent / "render.png"
            if not os.path.lexists(png_path):
                continue
            if dl is not None and (pool / ".datalad").is_dir():
                materialize_fn(datalad_api=dl, pool=pool, path=meta_path, required=False)
                materialize_fn(datalad_api=dl, pool=pool, path=png_path, required=False)
            if not meta_path.is_file() or not png_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            r_key = str(meta.get("render_key") or meta_path.parent.name)
            if r_key in seen_keys:
                continue
            seen_keys.add(r_key)

            r_type = str(meta.get("render_type") or "")
            r_mode = str(meta.get("render_mode") or "")
            f_token = str(meta.get("frame_token") or "f0000")
            f_idx = int(meta.get("frame_index_effective", meta.get("frame_index_requested", 0)))
            cam_id = str(meta.get("camera_id") or "front")
            cam_short = cam_id.split("_")[0]

            rel_dir = f"renders/{entry_name}/{r_type}/{f_token}_cam-{cam_id}"
            rel_png = f"{rel_dir}/render.png"
            dst_png = output_dir / rel_png
            dst_png.parent.mkdir(parents=True, exist_ok=True)
            if not dst_png.is_file():
                if dst_png.exists() or dst_png.is_symlink():
                    dst_png.unlink()
                shutil.copyfile(png_path, dst_png)
                dst_png.chmod(0o644)
            if rel_png not in bundled_rel_files:
                bundled_rel_files.append(rel_png)

            rel_cb = None
            cb_src = meta_path.parent / "colorbar.png"
            if os.path.lexists(cb_src):
                if dl is not None and (pool / ".datalad").is_dir():
                    materialize_fn(datalad_api=dl, pool=pool, path=cb_src, required=False)
                if cb_src.is_file():
                    rel_cb = f"{rel_dir}/colorbar.png"
                    dst_cb = output_dir / rel_cb
                    if not dst_cb.is_file():
                        if dst_cb.exists() or dst_cb.is_symlink():
                            dst_cb.unlink()
                        shutil.copyfile(cb_src, dst_cb)
                        dst_cb.chmod(0o644)
                    if rel_cb not in bundled_rel_files:
                        bundled_rel_files.append(rel_cb)

            cameras_set.add(cam_short)
            frames_set.add(f_token)
            types_set.add(r_type)

            items.append({
                "render_key": r_key,
                "render_type": r_type,
                "render_mode": r_mode,
                "frame_index": f_idx,
                "frame_token": f_token,
                "camera_id": cam_id,
                "camera_short": cam_short,
                "resolution": meta.get("resolution") or [1024, 1024],
                "rel_png": rel_png,
                "rel_colorbar": rel_cb,
                "heatmap_range": (meta.get("camera_framing") or {}).get("heatmap_range"),
                "frame_l2_mean": (meta.get("geometry_stats") or {}).get("frame_l2_mean"),
                "frame_l2_max": (meta.get("geometry_stats") or {}).get("frame_l2_max"),
                "alpha_coverage_pct": (meta.get("optical_qa") or {}).get("alpha_coverage_pct"),
            })

    return {
        "data_dir": data_dir,
        "entry_name": entry_name,
        "cameras": sorted(cameras_set),
        "frames": sorted(frames_set),
        "render_types": sorted(types_set),
        "items": items,
    }
