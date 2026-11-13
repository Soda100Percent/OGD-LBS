#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
LAN Interactive Render QA Inspector & Reverse-Lookup Tool.
"""

import argparse
import csv
import html
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_SRC_DIR, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from generate_pdrew_gamma_report import (  # noqa: E402
    materialize_pool_file,
    start_daemon_http_server,
)

CONTACT_SHEET_INDEX_SCHEMA = "ogd_lbs_render_contact_sheet_index_v1"
DEFAULT_POOL_DIR = Path("/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1")
DEFAULT_PDREW_ROOT = Path("/workspace/pd-rew-v2")
DEFAULT_SHEET_DIR = Path("/workspace/ogd_reports/render_qa_latest")


def evaluate_fast_neural_salience_qa(png_path: Path, *, render_mode: str = "clay") -> dict[str, Any]:
    rgba = plt.imread(str(png_path))
    if rgba.ndim == 2:
        rgba = np.stack([rgba, rgba, rgba, np.ones_like(rgba)], axis=-1)
    elif rgba.shape[2] == 3:
        rgba = np.concatenate([rgba, np.ones((*rgba.shape[:2], 1), dtype=rgba.dtype)], axis=-1)

    h, w = int(rgba.shape[0]), int(rgba.shape[1])
    alpha = rgba[:, :, 3]
    fg = alpha > 0.02
    fg_total = int(np.count_nonzero(fg))
    if fg_total == 0:
        return {
            "passed": False,
            "subject_score": 0.0,
            "num_significant_components": 0,
            "largest_component_mass_ratio": 0.0,
            "center_salience_ratio": 0.0,
            "warnings": ["empty_foreground"],
        }

    struct = ndimage.generate_binary_structure(2, 2)
    dilated_fg = ndimage.binary_dilation(fg, structure=struct, iterations=4)
    labeled, n_comp = ndimage.label(dilated_fg)
    comp_sizes = ndimage.sum(fg, labeled, index=range(1, n_comp + 1))
    comp_sizes = np.asarray(comp_sizes, dtype=np.float64) if len(comp_sizes) else np.zeros((0,), dtype=np.float64)

    min_sig_px = max(16.0, 0.0005 * float(h * w))
    sig_comps = int(np.count_nonzero(comp_sizes >= min_sig_px))
    largest_ratio = float(np.max(comp_sizes) / float(fg_total)) if comp_sizes.size else 0.0

    luma = 0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]
    gx = ndimage.sobel(luma, axis=1)
    gy = ndimage.sobel(luma, axis=0)
    grad_mag = np.hypot(gx, gy) * (alpha > 0.02)
    total_grad = float(np.sum(grad_mag))

    y0, y1 = int(0.08 * h), int(0.92 * h)
    x0, x1 = int(0.08 * w), int(0.92 * w)
    center_grad = float(np.sum(grad_mag[y0:y1, x0:x1]))
    center_salience_ratio = (center_grad / total_grad) if total_grad > 1e-9 else 0.0

    warnings: list[str] = []
    min_largest_ratio = 0.35 if render_mode in {"bone", "wireframe", "gt_wireframe"} else 0.55
    if largest_ratio < min_largest_ratio:
        warnings.append(f"fragmented_subject(largest_mass={largest_ratio:.2f}<{min_largest_ratio:.2f})")
    if sig_comps > 25:
        warnings.append(f"excessive_scattered_fragments(components={sig_comps})")
    if center_salience_ratio < 0.70:
        warnings.append(f"low_center_salience({center_salience_ratio:.2f}<0.70)")

    score = float(np.clip(0.55 * largest_ratio + 0.45 * center_salience_ratio, 0.0, 1.0))
    return {
        "passed": len(warnings) == 0 and score >= 0.55,
        "subject_score": round(score, 4),
        "num_significant_components": sig_comps,
        "largest_component_mass_ratio": round(largest_ratio, 4),
        "center_salience_ratio": round(center_salience_ratio, 4),
        "warnings": warnings,
    }


def _load_pool_metadata_table(pool_root: Path) -> dict[str, dict[str, str]]:
    csv_path = pool_root / "metadata_table.csv"
    if not csv_path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                dd = row.get("data_dir")
                if dd:
                    out[str(dd)] = dict(row)
    except Exception:
        return {}
    return out


def discover_pool_renders(
    pool_dir: Path,
    *,
    stage_filter: str | None = None,
    model_filter: str | None = None,
    render_type_filter: str | None = None,
    qa_mode: str = "fast-nn",
) -> list[dict[str, Any]]:
    pool_root = pool_dir.expanduser().resolve(strict=True)
    results_root = pool_root / "results"
    table_by_dd = _load_pool_metadata_table(pool_root)
    items: list[dict[str, Any]] = []

    try:
        import datalad.api as dl
    except Exception:
        dl = None

    for r_dir in sorted(results_root.glob("r[0-9][0-9][0-9][0-9][0-9][0-9]")):
        data_dir = f"results/{r_dir.name}"
        table_dims = table_by_dd.get(data_dir) or {}
        cp_path = r_dir / "case_parameters.json"
        cp_obj: dict[str, Any] = {}
        if os.path.lexists(cp_path):
            if dl is not None and not cp_path.is_file() and (pool_root / ".datalad").is_dir():
                materialize_pool_file(datalad_api=dl, pool=pool_root, path=cp_path, required=False)
            if cp_path.is_file():
                try:
                    cp_obj = json.loads(cp_path.read_text(encoding="utf-8"))
                except Exception:
                    cp_obj = {}

        cp_dims = cp_obj.get("pool_dimensions") or cp_obj.get("dimension_values") or cp_obj.get("flat_parameters") or {}
        pool_dims = {**cp_dims, **table_dims}
        sweep_choices = cp_obj.get("sweep_choices") or {}
        model_name = str(
            pool_dims.get("data_prep_handoff.model_name")
            or pool_dims.get("ssdr_handoff.model_name")
            or "UnknownModel"
        )
        if model_filter and model_name != model_filter:
            continue

        stage_subdirs = [
            d for d in sorted(r_dir.iterdir())
            if d.is_dir() and d.name not in {"renders", ".staging"} and not d.name.startswith(".")
        ]
        search_roots = stage_subdirs if stage_subdirs else [r_dir]

        seen_render_keys: set[str] = set()
        for s_root in search_roots:
            renders_base = s_root / "renders"
            if not renders_base.is_dir():
                continue
            for meta_path in sorted(renders_base.rglob("render_meta.json")):
                png_path = meta_path.parent / "render.png"
                if not os.path.lexists(png_path):
                    continue
                if dl is not None and (pool_root / ".datalad").is_dir():
                    if not meta_path.is_file():
                        materialize_pool_file(datalad_api=dl, pool=pool_root, path=meta_path, required=False)
                    if not png_path.is_file():
                        materialize_pool_file(datalad_api=dl, pool=pool_root, path=png_path, required=False)
                if not meta_path.is_file() or not png_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                r_key = str(meta.get("render_key") or meta_path.parent.name)
                if r_key in seen_render_keys:
                    continue
                seen_render_keys.add(r_key)

                st_name = str(meta.get("stage") or cp_obj.get("stage") or s_root.name)
                r_type = str(meta.get("render_type") or "")
                if stage_filter and st_name != stage_filter:
                    continue
                if render_type_filter and r_type != render_type_filter:
                    continue

                optical_qa = meta.get("optical_qa") or {}
                fast_nn_qa = (
                    evaluate_fast_neural_salience_qa(png_path, render_mode=str(meta.get("render_mode", "clay")))
                    if qa_mode in {"fast-nn", "vlm"}
                    else None
                )

                opt_passed = bool(optical_qa.get("passed", True))
                nn_passed = True if fast_nn_qa is None else bool(fast_nn_qa.get("passed", True))
                overall_status = "PASS" if (opt_passed and nn_passed) else ("FAIL" if not opt_passed else "WARN")

                thumb_id = len(items) + 1
                stage_input_dir = s_root if s_root != r_dir else r_dir
                cam_prefix = str(meta.get("camera_id") or "front").split("_")[0]
                frame_idx = int(meta.get("frame_index_effective", meta.get("frame_index_requested", 0)))

                rerun_cmd = (
                    f"/opt/conda/envs/ogd-lbs/bin/python /workspace/OGD-LBS/scripts/experiments/augment_stage_renders.py "
                    f"--input_dir {stage_input_dir} "
                    f"--output_dir {meta_path.parent} "
                    f"--stage {st_name} "
                    f"--camera {cam_prefix} "
                    f"--frame {frame_idx} "
                    f"--render_type {r_type} --overwrite"
                )

                variant_val = str(pool_dims.get("ssdr_stage02.variant") or "")
                if not variant_val or variant_val == "None":
                    variant_val = "default"

                items.append({
                    "thumb_id": thumb_id,
                    "thumb_code": f"#{thumb_id:04d}",
                    "overall_status": overall_status,
                    "data_dir": data_dir,
                    "entry_name": r_dir.name,
                    "stage": st_name,
                    "stage_subdir": s_root.name if s_root != r_dir else "",
                    "model": model_name,
                    "variant": variant_val,
                    "s02_version": str(pool_dims.get("ssdr_stage02") or "v1"),
                    "s03_version": str(pool_dims.get("ssdr_stage03") or "v1"),
                    "sweep_choices": sweep_choices,
                    "pool_dimensions": pool_dims,
                    "output_subdir": str(meta.get("output_subdir") or ""),
                    "render_key": r_key,
                    "render_type": r_type,
                    "render_mode": str(meta.get("render_mode") or ""),
                    "frame_index": frame_idx,
                    "frame_token": str(meta.get("frame_token") or ""),
                    "camera_id": str(meta.get("camera_id") or ""),
                    "camera_spec": meta.get("camera_spec") or {},
                    "png_path": str(png_path),
                    "meta_path": str(meta_path),
                    "rel_png": f"thumbs/thumb_{thumb_id:04d}.png",
                    "optical_qa": optical_qa,
                    "fast_nn_qa": fast_nn_qa,
                    "rerun_single_command": rerun_cmd,
                })

    return items


def build_contact_sheets(
    items: list[dict[str, Any]],
    output_dir: Path,
    *,
    cols: int = 5,
    rows_per_page: int = 3,
    run_vlm: bool = False,
    title: str = "OGD-LBS Render QA Inspector (Numbered Thumbnails + Optical & Neural Guardrails)",
) -> dict[str, Any]:
    out_root = output_dir.expanduser().resolve(strict=False)
    out_root.mkdir(parents=True, exist_ok=True)
    thumbs_dir = out_root / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    if thumbs_dir.exists():
        for old_f in thumbs_dir.glob("*"):
            try:
                old_f.chmod(0o644)
                old_f.unlink()
            except Exception:
                pass
    for it in items:
        dst = out_root / it["rel_png"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copyfile(it["png_path"], dst)
        dst.chmod(0o644)
        cb_src = Path(it["png_path"]).parent / "colorbar.png"
        if cb_src.is_file():
            cb_rel = f"thumbs/colorbar_{it['thumb_id']:04d}.png"
            cb_dst = out_root / cb_rel
            if cb_dst.exists() or cb_dst.is_symlink():
                cb_dst.unlink()
            shutil.copyfile(cb_src, cb_dst)
            cb_dst.chmod(0o644)
            it["rel_colorbar_png"] = cb_rel

    per_page = max(1, int(cols) * int(rows_per_page))
    pages: list[dict[str, Any]] = []
    for page_idx, start in enumerate(range(0, len(items), per_page), start=1):
        chunk = items[start : start + per_page]
        n_rows = max(1, math.ceil(len(chunk) / float(cols)))
        fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 3.2, n_rows * 3.5), dpi=120, squeeze=False)
        fig.patch.set_facecolor("#0f172a")

        for idx_p in range(n_rows * cols):
            ax = axes[idx_p // cols][idx_p % cols]
            ax.set_xticks([])
            ax.set_yticks([])
            if idx_p >= len(chunk):
                ax.set_facecolor("#0f172a")
                for sp in ax.spines.values():
                    sp.set_visible(False)
                continue
            it = chunk[idx_p]
            rgba = plt.imread(it["png_path"])
            h_i, w_i = rgba.shape[0], rgba.shape[1]
            yy, xx = np.indices((h_i, w_i))
            checker = (((xx // 32) + (yy // 32)) % 2).astype(np.float32)
            bg = np.where(checker[:, :, None] == 0, 0.16, 0.22)
            alpha = rgba[:, :, 3:4] if rgba.shape[2] == 4 else np.ones((h_i, w_i, 1), dtype=np.float32)
            comp = rgba[:, :, :3] * alpha + bg * (1.0 - alpha)
            ax.imshow(np.clip(comp, 0.0, 1.0))

            st = it["overall_status"]
            b_col = "#10b981" if st == "PASS" else ("#f59e0b" if st == "WARN" else "#ef4444")
            for sp in ax.spines.values():
                sp.set_edgecolor(b_col)
                sp.set_linewidth(2.8)
            ax.text(
                0.03,
                0.96,
                f"{it['thumb_code']} [{st}]",
                transform=ax.transAxes,
                fontsize=9.5,
                fontweight="bold",
                color="#fff",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=b_col, edgecolor="#0f172a", linewidth=1.0),
            )
            ax.set_xlabel(f"{it['entry_name']} | {it['model']}\n{it['render_type']}", color="#e2e8f0", fontsize=7.8)

        plt.tight_layout(pad=0.9)
        sheet_name = f"contact_sheet_page{page_idx:02d}.png"
        sheet_path = out_root / sheet_name
        fig.savefig(str(sheet_path), facecolor=fig.get_facecolor())
        plt.close(fig)
        pages.append({
            "page": page_idx,
            "sheet_png": sheet_name,
            "sheet_path": str(sheet_path),
            "thumb_ids": [x["thumb_id"] for x in chunk],
        })

    index_obj = {
        "schema": CONTACT_SHEET_INDEX_SCHEMA,
        "title": title,
        "pool_dir": str(DEFAULT_POOL_DIR),
        "output_dir": str(out_root),
        "total_thumbnails": len(items),
        "pass_count": sum(1 for it in items if it["overall_status"] == "PASS"),
        "warn_count": sum(1 for it in items if it["overall_status"] == "WARN"),
        "fail_count": sum(1 for it in items if it["overall_status"] == "FAIL"),
        "pages": pages,
        "items": items,
    }

    tpl_path = _THIS_DIR / "report_templates" / "ogd_render_qa_template.html"
    tpl_text = tpl_path.read_text(encoding="utf-8")
    html_text = tpl_text.replace("__TITLE_HTML__", html.escape(title)).replace(
        "__DATA_JSON__", json.dumps(index_obj, ensure_ascii=False)
    )

    for html_name in ("Render_QA.html", "contact_sheet.html", "index.html"):
        (out_root / html_name).write_text(html_text, encoding="utf-8")

    index_obj["contact_sheet_html"] = str(out_root / "Render_QA.html")
    index_path = out_root / "contact_sheet_index.json"
    index_path.write_text(json.dumps(index_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return index_obj


def parse_id_tokens(raw_tokens: list[str]) -> list[int]:
    out: list[int] = []
    for tok in raw_tokens:
        for part in str(tok).replace(",", " ").split():
            clean = part.strip().lstrip("#")
            if not clean:
                continue
            val = int(clean)
            if val not in out:
                out.append(val)
    return out


def lookup_thumbnail_ids(index_path: Path, thumb_ids: list[int]) -> list[dict[str, Any]]:
    if not index_path.is_file():
        raise FileNotFoundError(f"Contact sheet index not found: {index_path}. Run with --build-sheet first.")
    idx_obj = json.loads(index_path.read_text(encoding="utf-8"))
    by_id = {int(it["thumb_id"]): it for it in (idx_obj.get("items") or [])}

    resolved: list[dict[str, Any]] = []
    for tid in thumb_ids:
        if tid not in by_id:
            raise KeyError(f"Thumbnail ID #{tid:04d} not found in {index_path} (valid range: 1..{len(by_id)})")
        resolved.append(by_id[tid])
    return resolved


def invalidate_thumbnail_ids_in_pool(
    index_path: Path,
    thumb_ids: list[int],
    *,
    pd_rew_root: Path = DEFAULT_PDREW_ROOT,
    pd_rew_python: str = "/opt/conda/envs/pd-rew/bin/python",
) -> dict[str, Any]:
    import subprocess

    idx_obj = json.loads(index_path.read_text(encoding="utf-8"))
    pool_dir = Path(idx_obj.get("pool_dir") or DEFAULT_POOL_DIR)

    targets = lookup_thumbnail_ids(index_path, thumb_ids)
    rel_paths: list[str] = []
    for it in targets:
        dd = it["data_dir"]
        out_sub = it["output_subdir"]
        st_sub = it["stage_subdir"]
        if st_sub:
            rel_paths.append(f"{dd}/{st_sub}/{out_sub}")
        rel_paths.append(f"{dd}/{out_sub}")

    existing_rels = [r for r in dict.fromkeys(rel_paths) if (pool_dir / r).exists()]
    if not existing_rels:
        return {"status": "NOTHING_TO_REMOVE", "invalidated_ids": thumb_ids, "removed_paths": []}

    helper_lines = [
        "import json, sys",
        "from pathlib import Path",
        f"sys.path.insert(0, {str(pd_rew_root / 'src')!r})",
        "from pd_rew_v2.pool import PipelinePool",
        "from pd_rew_v2.utils.io import remove_tree_if_exists",
        "payload = json.loads(sys.stdin.read())",
        "pool_dir = Path(payload['pool_dir'])",
        "rels = payload['rels']",
        "thumb_ids = payload['thumb_ids']",
        "pipe = PipelinePool(pool_dir)",
        "def _do_rm():",
        "    for r in rels:",
        "        remove_tree_if_exists(pool_dir / r)",
        "rep = pipe.pool.save_files(",
        "    write_callback=_do_rm,",
        "    paths=rels,",
        "    rollback_paths=rels,",
        "    message=f'invalidate faulty render thumbnail IDs {thumb_ids}',",
        ")",
        "print(json.dumps({'commit_sha': rep.get('commit_sha')}))",
    ]
    helper_code = chr(10).join(helper_lines)
    py_exe = pd_rew_python if Path(pd_rew_python).is_file() else sys.executable
    proc = subprocess.run(
        [py_exe, "-c", helper_code],
        input=json.dumps({"pool_dir": str(pool_dir), "rels": existing_rels, "thumb_ids": thumb_ids}),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to invalidate paths via {py_exe}: stderr={proc.stderr} stdout={proc.stdout}")
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    save_rep = json.loads(out_lines[-1]) if out_lines else {}
    return {
        "status": "INVALIDATED",
        "invalidated_ids": thumb_ids,
        "removed_paths": existing_rels,
        "commit_sha": save_rep.get("commit_sha"),
    }

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build LAN interactive numbered Render QA dashboard and reverse-lookup or invalidate faulty IDs."
    )
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_DIR, help="PipelinePool directory.")
    parser.add_argument("--sheet-dir", type=Path, default=DEFAULT_SHEET_DIR, help="Output directory for QA dashboard.")
    parser.add_argument("--stage", default=None, help="Optional filter by stage name.")
    parser.add_argument("--model", default=None, help="Optional filter by model name.")
    parser.add_argument("--render-type", default=None, help="Optional filter by render_type.")
    parser.add_argument("--qa-mode", choices=["optical", "fast-nn", "vlm"], default="fast-nn", help="QA level to run.")
    parser.add_argument("--cols", type=int, default=5, help="Columns per static contact sheet page.")
    parser.add_argument("--rows", type=int, default=3, help="Rows per static contact sheet page.")
    parser.add_argument("--build-sheet", action="store_true", help="Scan pool renders and build QA dashboard.")
    parser.add_argument("--lookup", nargs="+", default=None, help="Reverse-lookup thumbnail ID(s), e.g. --lookup 1 5 or #0002.")
    parser.add_argument("--invalidate", nargs="+", default=None, help="Purge faulty thumbnail ID(s) from pool so they re-render.")
    parser.add_argument("--serve-daemon", action="store_true", help="Start background LAN HTTP server on --port.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port inside container (default: 8080 -> Host 8081).")
    parser.add_argument("--json", action="store_true", help="Print output as JSON.")
    args = parser.parse_args()

    sheet_dir = args.sheet_dir.expanduser().resolve(strict=False)
    index_path = sheet_dir / "contact_sheet_index.json"

    if args.build_sheet or (not args.lookup and not args.invalidate):
        items = discover_pool_renders(
            args.pool,
            stage_filter=args.stage,
            model_filter=args.model,
            render_type_filter=args.render_type,
            qa_mode=args.qa_mode,
        )
        idx_obj = build_contact_sheets(
            items,
            sheet_dir,
            cols=args.cols,
            rows_per_page=args.rows,
            run_vlm=(args.qa_mode == "vlm"),
        )
        if not args.json:
            print("=" * 90, flush=True)
            print(f"LAN RENDER QA INSPECTOR BUILT: {idx_obj['total_thumbnails']} numbered thumbnails", flush=True)
            print(f"  PASS / NN_WARN / OPTICAL_FAIL : {idx_obj['pass_count']} / {idx_obj['warn_count']} / {idx_obj['fail_count']}", flush=True)
            print(f"  INDEX JSON                    : {index_path}", flush=True)
            print(f"  DASHBOARD HTML                : {idx_obj['contact_sheet_html']}", flush=True)
            print("=" * 90, flush=True)
        elif not args.lookup and not args.invalidate:
            print(json.dumps(idx_obj, indent=2, ensure_ascii=False))

    if args.lookup:
        tids = parse_id_tokens(args.lookup)
        records = lookup_thumbnail_ids(index_path, tids)
        if args.json:
            print(json.dumps(records, indent=2, ensure_ascii=False))
        else:
            print("\n=== REVERSE LOOKUP FOR THUMBNAIL ID(s) ===", flush=True)
            for rec in records:
                opt = rec.get("optical_qa") or {}
                nn = rec.get("fast_nn_qa") or {}
                print(
                    f"  {rec['thumb_code']} [{rec['overall_status']}] -> {rec['data_dir']} ({rec['model']}, stage={rec['stage']})\n"
                    f"    Sweep / Variant : variant={rec['variant']} s02={rec['s02_version']} s03={rec['s03_version']}\n"
                    f"    Render Identity : type={rec['render_type']} frame={rec['frame_index']} ({rec['frame_token']}) cam={rec['camera_id']}\n"
                    f"    Output Subdir   : {rec['output_subdir']}\n"
                    f"    Optical QA      : passed={opt.get('passed')} cov={opt.get('alpha_coverage_pct')}% clip={opt.get('border_clipped_px')}px center_off={opt.get('bbox_center_offset')}\n"
                    f"    Fast-NN QA      : passed={nn.get('passed')} score={nn.get('subject_score')} largest_mass={nn.get('largest_component_mass_ratio')} components={nn.get('num_significant_components')}\n"
                    f"    Re-render Cmd   : {rec['rerun_single_command']}",
                    flush=True,
                )

    if args.invalidate:
        tids = parse_id_tokens(args.invalidate)
        inv_rep = invalidate_thumbnail_ids_in_pool(index_path, tids)
        print(json.dumps(inv_rep, indent=2, ensure_ascii=False), flush=True)

    if args.serve_daemon:
        srv = start_daemon_http_server(sheet_dir, "Render_QA.html", port=int(args.port))
        print(f"🌐 LAN Render QA Server Active (PID={srv['pid']}, Container Port={srv['port']})", flush=True)
        print("👉 Host LAN URL (Port 8081->8080): http://10.241.1.178:8081/Render_QA.html", flush=True)
        print(f"👉 Container Direct URL          : {srv['report_url']}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
