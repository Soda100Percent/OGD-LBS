#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generate the Interactive 3D Target vs. Ours Differentiable Rendering Comparison Report
(`Diff_Render_Report.html`) with independent Target/Ours texture sampling controls
and a real-time Frontend Texture Filter Lab.
"""

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import report_core  # noqa: E402

DEFAULT_POOL = Path("/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1")
DEFAULT_OUTPUT_DIR = Path("/workspace/ogd_reports/pdrew_ue_comparison_report_latest")
ALL_MODELS = ["Griffin_A2", "Winter_Samurai", "Vroid_18", "Vroid_19", "Vroid_20"]


BLENDER_GLB_EXPORT_SNIPPET = r"""
import json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

repo_src = "/workspace/OGD-LBS/src"
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

import bpy
from ogd_lbs.fitbox.blender_io import import_fitbox_stage_to_blend
from ogd_lbs.utils.common import dump_pkl, load_data

req = json.loads(os.environ["OGD_GLB_EXPORT_REQ"])
input_dir = Path(req["input_dir"]).resolve()
output_glb = Path(req["output_glb"]).resolve()
target_frames = int(req.get("target_frames", 200))
max_tex_size = int(req.get("max_tex_size", 1024))
import_mode = str(req.get("import_mode", "LBS"))

with tempfile.TemporaryDirectory(prefix="ogd_glb_prep_") as tmp_str:
    tmp_dir = Path(tmp_str)
    stage_dir = tmp_dir / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)

    skel_src = next(
        (input_dir / f"skeleton_hierarchy_data{ext}" for ext in (".pkl", ".json") if (input_dir / f"skeleton_hierarchy_data{ext}").is_file()),
        None,
    )
    anim_src = next(
        (input_dir / f"animation_data{ext}" for ext in (".pkl", ".json") if (input_dir / f"animation_data{ext}").is_file()),
        None,
    )
    if skel_src is None or anim_src is None:
        raise RuntimeError(f"Missing skeleton or animation data in {input_dir}")

    skel_obj = dict(load_data(skel_src))
    anim_obj = dict(load_data(anim_src))
    poses = np.asarray(anim_obj["animation_poses"])
    n_src_frames = int(poses.shape[0])

    if n_src_frames != target_frames and n_src_frames > 1:
        idx = np.round(np.linspace(0, n_src_frames - 1, target_frames)).astype(np.int64)
        idx[0] = 0
        idx[-1] = n_src_frames - 1
        anim_obj["animation_poses"] = poses[idx]
    else:
        idx = None

    dump_pkl(stage_dir / "skeleton_hierarchy_data.pkl", skel_obj)
    dump_pkl(stage_dir / "animation_data.pkl", anim_obj)

    non_mesh = {
        "skeleton_hierarchy_data",
        "animation_data",
        "target_export_summary",
        "body_bone_alpha",
        "step05_diff_render_summary",
        "normalized_metrics_v1",
    }
    mesh_files = []
    seen = set()
    for p in sorted(list(input_dir.glob("*.pkl")) + list(input_dir.glob("*.json")), key=lambda x: x.name):
        if p.stem in non_mesh or p.stem.endswith("_summary") or p.stem in seen:
            continue
        seen.add(p.stem)
        mesh_files.append(p)

    for mf in mesh_files:
        m_obj = dict(load_data(mf))
        if idx is not None:
            for k in ("baked_animation_vertices", "baked_animation_vertex_normals"):
                if k in m_obj and m_obj[k] is not None:
                    arr = np.asarray(m_obj[k])
                    if arr.shape[0] == n_src_frames:
                        m_obj[k] = arr[idx]
        dump_pkl(stage_dir / f"{mf.stem}.pkl", m_obj)

    for png in input_dir.glob("*.png"):
        with Image.open(png) as im:
            rgba = im.convert("RGBA")
            if max(rgba.size) > max_tex_size:
                rgba = rgba.resize((max_tex_size, max_tex_size), Image.Resampling.LANCZOS)
            rgba.save(stage_dir / png.name, optimize=True)

    tmp_blend = tmp_dir / "scene.blend"
    import_fitbox_stage_to_blend(
        input_dir=stage_dir,
        output_blend=tmp_blend,
        mesh_name="ALL",
        import_mode=import_mode,
        bone_length=0.08,
    )

    for obj in list(bpy.data.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    for mat in bpy.data.materials:
        if not mat or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        tex_node = next((n for n in nodes if n.type == "TEX_IMAGE"), None)
        bsdf_node = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if tex_node and bsdf_node:
            if "Base Color" in bsdf_node.inputs:
                links.new(bsdf_node.inputs["Base Color"], tex_node.outputs["Color"])
            if "Alpha" in bsdf_node.inputs and "Alpha" in tex_node.outputs:
                links.new(bsdf_node.inputs["Alpha"], tex_node.outputs["Alpha"])
            mat.blend_method = "CLIP"
            if hasattr(mat, "alpha_threshold"):
                mat.alpha_threshold = 0.5

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_glb),
        export_format="GLB",
        export_animations=True,
        export_frame_range=True,
        export_frame_step=1,
        export_force_sampling=True,
        export_skins=(import_mode == "LBS"),
        export_morph=(import_mode == "shape_key"),
        export_morph_normal=False,
        export_apply=False,
    )
"""


def _export_fitbox_dir_to_glb(
    input_dir: Path,
    output_glb: Path,
    *,
    import_mode: str = "LBS",
    target_frames: int = 200,
    max_tex_size: int = 1024,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_glb.is_file() and output_glb.stat().st_size > 1024 and not overwrite:
        return {
            "status": "CACHED",
            "glb_path": str(output_glb),
            "size_mib": round(output_glb.stat().st_size / (1024.0 * 1024.0), 2),
        }

    ogd_py = "/opt/conda/envs/ogd-lbs/bin/python"
    req = {
        "input_dir": str(input_dir),
        "output_glb": str(output_glb),
        "import_mode": import_mode,
        "target_frames": target_frames,
        "max_tex_size": max_tex_size,
    }
    env = os.environ.copy()
    env["OGD_GLB_EXPORT_REQ"] = json.dumps(req)
    proc = subprocess.run(
        [ogd_py, "-c", BLENDER_GLB_EXPORT_SNIPPET],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not output_glb.is_file():
        raise RuntimeError(
            f"GLB export failed for {input_dir} -> {output_glb}:\nSTDOUT:\n{proc.stdout[-1500:]}\nSTDERR:\n{proc.stderr[-1500:]}"
        )
    output_glb.chmod(0o644)
    return {
        "status": "EXPORTED",
        "glb_path": str(output_glb),
        "size_mib": round(output_glb.stat().st_size / (1024.0 * 1024.0), 2),
    }


def _compute_dir_mesh_totals(fitbox_dir: Path) -> dict[str, int]:
    from ogd_lbs.utils.common import load_data

    non_mesh = {
        "skeleton_hierarchy_data",
        "animation_data",
        "target_export_summary",
        "body_bone_alpha",
        "step05_diff_render_summary",
        "normalized_metrics_v1",
    }
    skel_p = next(
        (fitbox_dir / f"skeleton_hierarchy_data{ext}" for ext in (".pkl", ".json") if (fitbox_dir / f"skeleton_hierarchy_data{ext}").is_file()),
        None,
    )
    bones_cnt = 0
    if skel_p is not None:
        sk = load_data(skel_p)
        bones_cnt = len(sk.get("bone_names", []))

    seen = set()
    v_tot = 0
    f_tot = 0
    m_cnt = 0
    for p in sorted(list(fitbox_dir.glob("*.pkl")) + list(fitbox_dir.glob("*.json")), key=lambda x: x.name):
        if p.stem in non_mesh or p.stem.endswith("_summary") or p.stem in seen:
            continue
        seen.add(p.stem)
        obj = load_data(p)
        if not isinstance(obj, dict) or "vertices" not in obj:
            continue
        v_tot += len(obj["vertices"])
        f_tot += len(obj["faces"])
        m_cnt += 1
    return {
        "mesh_count": m_cnt,
        "vertices": v_tot,
        "faces": f_tot,
        "bones": bones_cnt,
    }


def _resolve_bsdf_for_stem(stem: str, rules: list[list[str]] | None, default_bsdf: str = "ambient_2sided") -> str:
    if not rules:
        return default_bsdf
    for r in rules:
        if isinstance(r, list) and len(r) == 2 and str(r[0]).lower() in stem.lower():
            return str(r[1])
    return default_bsdf


def discover_diff_render_models(
    pool_dir: Path,
    output_dir: Path,
    *,
    export_glb: bool = True,
    overwrite_glb: bool = False,
) -> list[dict[str, Any]]:
    handoff_by_model: dict[str, str] = {}
    stage01_by_model: dict[str, str] = {}

    with (pool_dir / "metadata_table.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = r.get("data_prep_handoff.model_name")
            if not m:
                continue
            if r.get("diff_render_handoff") not in (None, "", "None") and r.get("diff_render_stage01") in (None, "", "None"):
                handoff_by_model[m] = r["data_dir"]
            if r.get("diff_render_stage01") not in (None, "", "None"):
                stage01_by_model[m] = r["data_dir"]

    glb_dir = output_dir / "glb"
    glb_dir.mkdir(parents=True, exist_ok=True)
    previews_out_dir = output_dir / "diff_render_previews"
    previews_out_dir.mkdir(parents=True, exist_ok=True)

    models_data: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        h_dd = handoff_by_model.get(model)
        s_dd = stage01_by_model.get(model)
        tgt_dir = (pool_dir / h_dd / "diff_render_00" / "target") if h_dd else Path(f"/workspace/ogd_dataset/{model}/target")
        src_dir = (pool_dir / h_dd / "diff_render_00" / "source") if h_dd else None

        tgt_stats = _compute_dir_mesh_totals(tgt_dir) if tgt_dir.is_dir() else {"mesh_count": 0, "vertices": 0, "faces": 0, "bones": 0}
        src_stats = _compute_dir_mesh_totals(src_dir) if (src_dir and src_dir.is_dir()) else {"mesh_count": 0, "vertices": 0, "faces": 0, "bones": 0}

        if not s_dd:
            models_data.append({
                "model": model,
                "status": "RUNNING",
                "handoff_data_dir": h_dd,
                "stage01_data_dir": None,
                "target_stats": tgt_stats,
                "ours_stats": src_stats,
                "target_glb": None,
                "ours_glb": None,
                "metrics": {},
                "submesh_metrics": [],
            })
            continue

        d5_dir = pool_dir / s_dd / "data5"
        tf_dir = d5_dir / "target_format"
        if not tf_dir.is_dir():
            from augment_diff_render_target_format import build_target_format_from_diff_render_stage
            build_target_format_from_diff_render_stage(d5_dir, tf_dir)

        ours_stats = _compute_dir_mesh_totals(tf_dir)

        rel_tgt_glb = f"glb/{model}_target.glb"
        rel_ours_glb = f"glb/{model}_ours.glb"
        if export_glb:
            print(f"[diff-render-report] Exporting GLBs for {model} ...", flush=True)
            _export_fitbox_dir_to_glb(
                tgt_dir,
                output_dir / rel_tgt_glb,
                import_mode="LBS",
                target_frames=200,
                max_tex_size=1024,
                overwrite=overwrite_glb,
            )
            _export_fitbox_dir_to_glb(
                tf_dir,
                output_dir / rel_ours_glb,
                import_mode="LBS",
                target_frames=200,
                max_tex_size=1024,
                overwrite=overwrite_glb,
            )

        s2_cfg_path = d5_dir / "stage02" / "config_run.json"
        s2_cfg = json.loads(s2_cfg_path.read_text(encoding="utf-8")) if s2_cfg_path.is_file() else {}
        opt_bsdf_rules = s2_cfg.get("opt_submesh_bsdf") or []
        ref_bsdf_rules = s2_cfg.get("ref_submesh_bsdf") or []

        s05_path = d5_dir / "step05_diff_render_summary.json"
        s05_obj = json.loads(s05_path.read_text(encoding="utf-8")) if s05_path.is_file() else {}
        fm = s05_obj.get("final_metrics") or {}

        metrics_clean = {
            "stage01_train_psnr": fm.get("stage01_final_train_psnr"),
            "stage01_test_psnr": fm.get("stage01_final_test_psnr"),
            "stage01_train_lpips": fm.get("stage01_final_train_lpips"),
            "stage01_test_lpips": fm.get("stage01_final_test_lpips"),
            "stage01_train_hausdorff": fm.get("stage01_final_train_hausdorff"),
            "stage01_test_hausdorff": fm.get("stage01_final_test_hausdorff"),
            "stage02_train_psnr": fm.get("stage02_final_train_psnr"),
            "stage02_test_psnr": fm.get("stage02_final_test_psnr"),
            "stage02_train_lpips": fm.get("stage02_final_train_lpips"),
            "stage02_test_lpips": fm.get("stage02_final_test_lpips"),
            "stage02_train_hausdorff": fm.get("stage02_final_train_hausdorff"),
            "stage02_test_hausdorff": fm.get("stage02_final_test_hausdorff"),
        }

        tf_sum_path = tf_dir / "target_export_summary.json"
        tf_sum = json.loads(tf_sum_path.read_text(encoding="utf-8")) if tf_sum_path.is_file() else {}
        submesh_rows = []
        for m_entry in tf_sum.get("meshes") or []:
            stem = m_entry["target_stem"]
            raw_stem = m_entry["source_stem"]
            s1_psnr = fm.get(f"stage01_final_test_{stem}_psnr", fm.get(f"stage01_final_test_{raw_stem}_psnr"))
            s2_psnr = fm.get(f"stage02_final_test_{stem}_psnr", fm.get(f"stage02_final_test_{raw_stem}_psnr"))
            s2_lpips = fm.get(f"stage02_final_test_{stem}_lpips", fm.get(f"stage02_final_test_{raw_stem}_lpips"))

            raw_s2_src = next(
                (p for p in [
                    d5_dir / "opt_out" / "textures" / f"{raw_stem}_texture_kd.png",
                    d5_dir / "opt_out" / "textures" / f"{stem}_texture_kd.png",
                    tf_dir / f"{stem}.png",
                ] if p.is_file()),
                None,
            )
            rel_s2_tex = f"diff_render_previews/{model}/raw_textures/{stem}.png"
            if raw_s2_src:
                dst_s2 = output_dir / rel_s2_tex
                dst_s2.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(raw_s2_src, dst_s2)
                dst_s2.chmod(0o644)

            raw_s1_src = next(
                (p for p in [
                    d5_dir / "opt_out" / "textures" / f"{raw_stem}_texture_kd_stage1.png",
                    d5_dir / "opt_out" / "textures" / f"{stem}_texture_kd_stage1.png",
                    raw_s2_src,
                ] if p and p.is_file()),
                None,
            )
            rel_s1_tex = f"diff_render_previews/{model}/stage1_textures/{stem}.png"
            if raw_s1_src:
                dst_s1 = output_dir / rel_s1_tex
                dst_s1.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(raw_s1_src, dst_s1)
                dst_s1.chmod(0o644)

            mask_src = next(
                (p for p in [
                    d5_dir / "opt_out" / "masks" / f"{raw_stem}_body_mask.png",
                    d5_dir / "opt_out" / "masks" / f"{stem}_body_mask.png",
                ] if p.is_file()),
                None,
            )
            rel_mask_tex = None
            if mask_src:
                rel_mask_tex = f"diff_render_previews/{model}/masks/{stem}_body_mask.png"
                dst_m = output_dir / rel_mask_tex
                dst_m.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(mask_src, dst_m)
                dst_m.chmod(0o644)

            opt_bsdf = _resolve_bsdf_for_stem(stem, opt_bsdf_rules, "ambient_2sided")
            ref_bsdf = _resolve_bsdf_for_stem(stem, ref_bsdf_rules, "ambient_2sided")

            submesh_rows.append({
                "submesh": stem,
                "vertices": m_entry["vertex_count"],
                "faces": m_entry["face_count"],
                "opt_bsdf": opt_bsdf,
                "ref_bsdf": ref_bsdf,
                "body_mask_applied": bool(rel_mask_tex),
                "stage01_test_psnr": s1_psnr,
                "stage02_test_psnr": s2_psnr,
                "stage02_test_lpips": s2_lpips,
                "rel_raw_texture": rel_s2_tex,
                "rel_stage1_texture": rel_s1_tex,
                "rel_body_mask": rel_mask_tex,
            })

        models_data.append({
            "model": model,
            "status": "READY",
            "handoff_data_dir": h_dd,
            "stage01_data_dir": s_dd,
            "target_stats": tgt_stats,
            "ours_stats": ours_stats,
            "target_glb": rel_tgt_glb,
            "ours_glb": rel_ours_glb,
            "metrics": metrics_clean,
            "submesh_metrics": submesh_rows,
        })

    return models_data


def render_diff_render_html(
    models_data: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    shared_assets = report_core.load_shared_frontend_assets(output_dir)
    src_three = HERE / "report_templates" / "shared" / "three"
    dst_three = output_dir / "shared" / "three"
    dst_three.mkdir(parents=True, exist_ok=True)
    for js_f in src_three.glob("*.js"):
        shutil.copyfile(js_f, dst_three / js_f.name)
        (dst_three / js_f.name).chmod(0o644)

    nav_html = report_core.build_suite_nav_html("Diff_Render_Report.html")
    payload = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models_data,
    }
    json_path = output_dir / "diff_render_dashboard_data.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OGD-LBS Step 05 Differentiable Rendering — Target vs. Ours 3D &amp; Live Texture Filter Lab</title>
<style>
{shared_assets["core_css"]}
.dr-container {{
  max-width: 1880px;
  margin: 0 auto;
  padding: 16px 20px 60px;
}}
.model-tabs {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}}
.model-tab-btn {{
  padding: 7px 14px;
  border-radius: 7px;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  color: #1e293b;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: all 0.15s;
}}
.model-tab-btn:hover {{ background: #e2e8f0; }}
.model-tab-btn.active {{
  background: #1C328A;
  color: #fff;
  border-color: #1C328A;
  box-shadow: 0 2px 8px rgba(28, 50, 138, 0.25);
}}
.filter-lab-bar {{
  background: linear-gradient(135deg, #0f172a, #1e293b);
  color: #f8fafc;
  border: 1px solid #38bdf8;
  border-radius: 10px;
  padding: 12px 16px;
  margin-bottom: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.filter-lab-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  justify-content: space-between;
}}
.fl-item {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(15, 23, 42, 0.78);
  padding: 5px 10px;
  border-radius: 6px;
  border: 1px solid #334155;
  font-size: 0.79rem;
}}
.fl-item label {{ color: #94a3b8; font-weight: 600; }}
.fl-item select {{
  background: #0f172a;
  color: #38bdf8;
  border: 1px solid #475569;
  border-radius: 4px;
  padding: 3px 7px;
  font-weight: 700;
  font-size: 0.79rem;
}}
.fl-item input[type="range"] {{
  width: 90px;
  accent-color: #38bdf8;
  cursor: pointer;
}}
.fl-val {{
  font-family: ui-monospace, monospace;
  color: #38bdf8;
  font-weight: 700;
  min-width: 34px;
  text-align: right;
}}
.dual-viewport-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 8px;
}}
@media (max-width: 1100px) {{
  .dual-viewport-grid {{ grid-template-columns: minmax(0, 1fr); }}
}}
.viewport-panel {{
  border: 2px solid #334155;
  border-radius: 10px;
  overflow: hidden;
  background: #0b1120;
  display: flex;
  flex-direction: column;
}}
.viewport-panel.ours-panel {{ border-color: #0284c7; }}
.vp-header {{
  background: #0f172a;
  color: #f8fafc;
  padding: 9px 14px;
  border-bottom: 1px solid #334155;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}}
.vp-title {{ font-weight: 700; font-size: 0.92rem; display: flex; align-items: center; gap: 8px; }}
.vp-stats {{ font-size: 0.78rem; color: #94a3b8; font-family: ui-monospace, monospace; }}
.vp-stats b {{ color: #38bdf8; }}
.vp-canvas-wrap {{
  position: relative;
  width: 100%;
  height: 560px;
  cursor: grab;
  overflow: hidden;
}}
.vp-canvas-wrap:active {{ cursor: grabbing; }}
.vp-canvas-wrap canvas {{
  display: block;
  width: 100% !important;
  height: 100% !important;
}}
.vp-overlay-status {{
  position: absolute;
  top: 12px;
  left: 12px;
  background: rgba(15, 23, 42, 0.85);
  color: #f8fafc;
  padding: 6px 11px;
  border-radius: 6px;
  font-size: 0.78rem;
  font-weight: 600;
  border: 1px solid #475569;
  pointer-events: none;
}}
.timeline-bar {{
  margin-top: 12px;
  background: #0f172a;
  color: #f8fafc;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.timeline-row {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}}
.frame-slider {{
  flex: 1;
  min-width: 220px;
  accent-color: #38bdf8;
  cursor: pointer;
}}
.frame-readout {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.86rem;
  font-weight: 700;
  background: #1e293b;
  padding: 5px 10px;
  border-radius: 6px;
  border: 1px solid #475569;
  min-width: 215px;
  text-align: center;
}}
.split-tag-train {{ color: #38bdf8; }}
.split-tag-test {{ color: #f59e0b; }}
.tex-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(175px, 1fr));
  gap: 12px;
  margin-top: 10px;
}}
.tex-card {{
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  overflow: hidden;
  background: #0f172a;
  color: #f8fafc;
  font-size: 0.75rem;
}}
.tex-card canvas {{
  width: 100%;
  aspect-ratio: 1 / 1;
  display: block;
  image-rendering: pixelated;
}}
.tex-card-meta {{
  padding: 6px 8px;
  border-top: 1px solid #334155;
}}
</style>
<script>
window.DIFF_RENDER_DATA = {json.dumps(payload, ensure_ascii=False)};
</script>
</head>
<body>
{nav_html}

<div class="top-bar">
  <h1>🎨 Step 05 Differentiable Rendering — 3D Compare &amp; Live Frontend Texture Filter Lab</h1>
  <div class="top-controls">
    <div id="model-tabs-mount" class="model-tabs"></div>
  </div>
</div>

<div class="dr-container">
  <div class="section-card">
    <div class="section-header">
      <h2>🕹️ Synchronized 3D WebGL Inspector (Independent Target / Ours Sampling + Live Filter Lab)</h2>
      <div class="top-controls">
        <div class="ctrl-group">
          <label>Shading:</label>
          <select id="sel-shading-mode">
            <option value="albedo" selected>🎨 Pure Albedo (Unlit Base Texture)</option>
            <option value="albedo_wire">🕸️ Albedo + Wireframe Overlay</option>
            <option value="clay_wire">🗿 Clay + Wireframe (Geometry Only)</option>
          </select>
        </div>
        <div class="ctrl-group">
          <label>Target Sampling (Left):</label>
          <select id="sel-target-sampling">
            <option value="trilinear" selected>✨ Trilinear Mipmap + Aniso 8x (Smooth GT)</option>
            <option value="bilinear">🌊 Bilinear (No Mipmap)</option>
            <option value="nearest_mip">🎯 Nearest Mag + Mipmap Min</option>
            <option value="nearest">🔲 Nearest Point (No Mipmap — render.py)</option>
          </select>
        </div>
        <div class="ctrl-group">
          <label>Ours Sampling (Right):</label>
          <select id="sel-ours-sampling">
            <option value="trilinear" selected>✨ Trilinear Mipmap + Aniso 8x (Smooth)</option>
            <option value="bilinear">🌊 Bilinear (No Mipmap)</option>
            <option value="nearest_mip">🎯 Nearest Mag + Mipmap Min</option>
            <option value="nearest">🔲 Nearest Point (No Mipmap — render.py)</option>
          </select>
        </div>
        <div class="ctrl-group">
          <label>Alpha / Cull (render.py L255):</label>
          <select id="sel-alpha-cull">
            <option value="bsdf_clip" selected>🎯 BSDF 1/2-Sided + Alpha Clip</option>
            <option value="bsdf_blend">💧 BSDF 1/2-Sided + Alpha Blend (lerp)</option>
            <option value="double_clip">DoubleSide + Alpha Clip</option>
          </select>
        </div>
        <div class="ctrl-group">
          <label>BG:</label>
          <select id="sel-bg-mode">
            <option value="bg-checker" selected>Checkerboard</option>
            <option value="bg-dark">Dark Studio</option>
            <option value="bg-white">Paper White</option>
          </select>
        </div>
        <div class="ctrl-group">
          <button class="btn btn-sm" data-cam="front">Front 0°</button>
          <button class="btn btn-sm" data-cam="iso45">Iso 45°</button>
          <button class="btn btn-sm" data-cam="side90">Side 90°</button>
          <button class="btn btn-sm" data-cam="back180">Back 180°</button>
          <button class="btn btn-sm" data-cam="closeup">🔍 Close-up</button>
          <button class="btn btn-sm btn-primary" data-cam="reset">↺ Reset</button>
        </div>
      </div>
    </div>

    <!-- LIVE FRONTEND TEXTURE FILTER LAB -->
    <div class="filter-lab-bar">
      <div class="filter-lab-row">
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
          <span style="font-weight:800;color:#38bdf8;font-size:0.88rem;">🧪 LIVE TEXTURE FILTER LAB (Right Viewport):</span>
          <div class="fl-item">
            <label>Tex Source:</label>
            <select id="sel-tex-source">
              <option value="stage1" selected>★ Stage 1 Isolated Bake (it=14000 — Clean Alpha)</option>
              <option value="stage2">Stage 2 Final (it=20000 — Multi-Layer Alpha)</option>
            </select>
          </div>
          <div class="fl-item">
            <label>Filter Mode:</label>
            <select id="sel-filter-algo">
              <option value="local_window_avg" selected>1. ★ Local Valid Ring Diffusion (Preserves All Local Details)</option>
              <option value="cc_pinhole_pyramid">2. Pinhole CC + Rainbow Ring Pull-Push Pyramid</option>
              <option value="all_holes_pyramid">3. All-Holes Pull-Push Multi-Scale Pyramid</option>
              <option value="opaque_only">4. Force Opaque Alpha=255 Only (Keep Raw RGB)</option>
              <option value="raw">0. Raw Unfiltered (Exact Stage Output)</option>
            </select>
          </div>
          <label class="fl-item" style="cursor:pointer;border-color:#f59e0b;">
            <input type="checkbox" id="chk-show-defects">
            <span style="color:#fbbf24;font-weight:700;">🔴 Highlight Detected Holes (Red) &amp; Rings/Spikes (Yellow)</span>
          </label>
        </div>
        <div id="lbl-filter-perf" style="font-size:0.78rem;color:#38bdf8;font-family:ui-monospace,monospace;">Filter ready</div>
      </div>

      <div class="filter-lab-row" style="justify-content:flex-start;">
        <div class="fl-item" title="Texels with raw Alpha below this threshold are detected as transparent hole cores">
          <label>Hole Alpha &lt;</label>
          <input type="range" id="rng-hole-alpha" min="10" max="240" step="5" value="128">
          <span class="fl-val" id="val-hole-alpha">128</span>
        </div>
        <div class="fl-item" title="Width (in pixels) of the rainbow ring around each transparent hole to strip and heal">
          <label>Rainbow Ring:</label>
          <input type="range" id="rng-ring-px" min="0" max="4" step="1" value="1">
          <span class="fl-val" id="val-ring-px">1 px</span>
        </div>
        <div class="fl-item" title="Maximum connected-component area (in texels) of a pinhole to heal in Mode 2">
          <label>Max Pinhole Area:</label>
          <input type="range" id="rng-max-area" min="20" max="5000" step="20" value="1200">
          <span class="fl-val" id="val-max-area">1200</span>
        </div>
        <div class="fl-item" title="Color jump threshold vs. local 5x5 valid neighborhood to catch isolated opaque rainbow speckles (set 120 to disable)">
          <label>RGB Spike Thr:</label>
          <input type="range" id="rng-spike-thr" min="15" max="120" step="5" value="42">
          <span class="fl-val" id="val-spike-thr">42</span>
        </div>
        <button class="btn btn-sm" id="btn-reset-filter" style="background:#334155;color:#fff;border-color:#475569;">↺ Default Filter</button>
      </div>
    </div>

    <div class="dual-viewport-grid">
      <div class="viewport-panel">
        <div class="vp-header">
          <div class="vp-title">
            <span class="rc-badge" style="background:#475569;color:#fff;">LEFT · TARGET (GT)</span>
            <span id="lbl-target-title">Ground Truth High-Poly</span>
          </div>
          <div class="vp-stats" id="lbl-target-stats">V: - | F: - | Bones: -</div>
        </div>
        <div class="vp-canvas-wrap bg-checker" id="vp-left-wrap">
          <div class="vp-overlay-status" id="vp-left-status">Loading Target GLB...</div>
        </div>
      </div>

      <div class="viewport-panel ours-panel">
        <div class="vp-header">
          <div class="vp-title">
            <span class="rc-badge rc-badge-current">RIGHT · OURS (STEP 05 + LIVE FILTER)</span>
            <span id="lbl-ours-title">QEM + SSDR + Diff-Render Low-Poly</span>
          </div>
          <div class="vp-stats" id="lbl-ours-stats">V: - | F: - | Bones: -</div>
        </div>
        <div class="vp-canvas-wrap bg-checker" id="vp-right-wrap">
          <div class="vp-overlay-status" id="vp-right-status">Loading Ours GLB...</div>
        </div>
      </div>
    </div>

    <!-- Animation Timeline Scrubber -->
    <div class="timeline-bar">
      <div class="timeline-row">
        <button class="btn btn-teal" id="btn-play-toggle" style="min-width:95px;">⏸ Pause</button>
        <button class="btn btn-sm" id="btn-prev-frame">⏮ -1 Frame</button>
        <button class="btn btn-sm" id="btn-next-frame">⏭ +1 Frame</button>
        <input type="range" id="slider-frame" class="frame-slider" min="0" max="199" step="1" value="0">
        <div class="frame-readout" id="lbl-frame-readout">Frame: 000 / 199 (TRAIN)</div>
      </div>
      <div class="timeline-row" style="justify-content:space-between;font-size:0.79rem;color:#cbd5e1;">
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <span>Quick Jump Frame:</span>
          <button class="btn btn-sm" data-jump-frame="0">Frame 0</button>
          <button class="btn btn-sm" data-jump-frame="50">Frame 50</button>
          <button class="btn btn-sm" data-jump-frame="100">Frame 100</button>
          <button class="btn btn-sm" data-jump-frame="150">Frame 150</button>
          <button class="btn btn-sm" data-jump-frame="190" style="background:#f59e0b;color:#0f172a;border-color:#f59e0b;">★ Frame 190 (Unseen Test Split)</button>
          <button class="btn btn-sm" data-jump-frame="195" style="background:#f59e0b;color:#0f172a;border-color:#f59e0b;">★ Frame 195 (Test)</button>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <span>Speed:</span>
          <select id="sel-speed" style="background:#1e293b;color:#fff;border:1px solid #475569;border-radius:4px;padding:2px 6px;">
            <option value="0.25">0.25x</option>
            <option value="0.5">0.50x</option>
            <option value="1.0" selected>1.00x (24 FPS)</option>
            <option value="1.5">1.50x</option>
          </select>
          <a id="link-dl-target-glb" class="btn btn-sm" href="#" download>📥 Target .glb</a>
          <a id="link-dl-ours-glb" class="btn btn-sm" href="#" download>📥 Ours .glb</a>
        </div>
      </div>
    </div>
  </div>

  <!-- Live 2D Filtered Texture Cards -->
  <div class="section-card">
    <div class="section-header">
      <h2 id="hdr-submesh-section">🧩 Live 2D Submesh Texture Inspector (Updates Synchronously with Filter Lab Above)</h2>
    </div>
    <div class="tex-grid" id="tex-grid-mount"></div>
    <div class="table-wrap" id="table-submesh-mount" style="margin-top:16px;"></div>
  </div>

  <!-- Cross-Model Quantitative Summary Table -->
  <div class="section-card">
    <div class="section-header">
      <h2>📊 All-Model Differentiable Rendering Summary</h2>
    </div>
    <div class="table-wrap" id="table-all-models-mount"></div>
  </div>
</div>

<script type="module">
import * as THREE from './shared/three/three.module.min.js';
import {{ OrbitControls }} from './shared/three/OrbitControls.js';
import {{ GLTFLoader }} from './shared/three/GLTFLoader.js';

const DATA = window.DIFF_RENDER_DATA || {{ models: [] }};
const models = DATA.models || [];
const readyModels = models.filter(m => m.status === 'READY');
let activeModelName = (readyModels[0] && readyModels[0].model) || (models[0] && models[0].model) || 'Winter_Samurai';

const TOTAL_FRAMES = 200;
const FPS = 24.0;
let currentFrame = 0;
let isPlaying = true;
let playbackSpeed = 1.0;
let shadingMode = 'albedo';
let targetSamplingMode = 'trilinear';
let oursSamplingMode = 'trilinear';
let alphaCullMode = 'bsdf_clip';
let lastTimeMs = performance.now();
let frameAccumulator = 0.0;

const filterCfg = {{
  texSource: 'stage1',
  algo: 'local_window_avg',
  holeAlpha: 128,
  ringPx: 1,
  maxArea: 1200,
  spikeThr: 42,
  showDefects: false,
}};

let activeSubmeshAssets = {{}};

const wrapL = document.getElementById('vp-left-wrap');
const wrapR = document.getElementById('vp-right-wrap');
const statusL = document.getElementById('vp-left-status');
const statusR = document.getElementById('vp-right-status');

function createViewport(container) {{
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, container.clientWidth / Math.max(1, container.clientHeight), 0.01, 500);
  const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true, preserveDrawingBuffer: true }});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const ambLight = new THREE.AmbientLight(0xffffff, 1.1);
  const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.4);
  dirLight1.position.set(2, 4, 3);
  const dirLight2 = new THREE.DirectionalLight(0x94a3b8, 0.8);
  dirLight2.position.set(-2, 2, -3);
  scene.add(ambLight, dirLight1, dirLight2);

  const grid = new THREE.GridHelper(2.4, 12, 0x38bdf8, 0x334155);
  grid.position.y = -0.9;
  scene.add(grid);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.target.set(0, 0, 0);
  camera.position.set(0, 0.15, 2.45);
  controls.update();

  return {{ container, scene, camera, renderer, controls, rootGroup: null, mixer: null }};
}}

const vpL = createViewport(wrapL);
const vpR = createViewport(wrapR);

let syncingCamera = false;
function syncCamera(srcVp, dstVp) {{
  if (syncingCamera) return;
  syncingCamera = true;
  dstVp.camera.position.copy(srcVp.camera.position);
  dstVp.camera.quaternion.copy(srcVp.camera.quaternion);
  dstVp.camera.zoom = srcVp.camera.zoom;
  dstVp.camera.updateProjectionMatrix();
  dstVp.controls.target.copy(srcVp.controls.target);
  dstVp.controls.update();
  syncingCamera = false;
}}

vpL.controls.addEventListener('change', () => syncCamera(vpL, vpR));
vpR.controls.addEventListener('change', () => syncCamera(vpR, vpL));

function setCameraPreset(preset) {{
  const dist = 2.45;
  let pos = new THREE.Vector3(0, 0.12, dist);
  let tgt = new THREE.Vector3(0, 0.0, 0);
  if (preset === 'iso45') pos.set(dist * Math.sin(Math.PI / 4), 0.22, dist * Math.cos(Math.PI / 4));
  else if (preset === 'side90') pos.set(dist, 0.12, 0);
  else if (preset === 'back180') pos.set(0, 0.12, -dist);
  else if (preset === 'closeup') {{
    pos.set(0, 0.42, 1.05);
    tgt.set(0, 0.38, 0);
  }}
  vpL.camera.position.copy(pos);
  vpL.controls.target.copy(tgt);
  vpL.controls.update();
  syncCamera(vpL, vpR);
}}

// ============================================================================
// REAL-TIME JAVASCRIPT TEXTURE FILTER ENGINE
// ============================================================================

function findConnectedHoleMask(alphaArr, w, h, holeThr, maxArea, useMaxArea) {{
  const n = w * h;
  const isHole = new Uint8Array(n);
  for (let i = 0; i < n; i++) {{
    if (alphaArr[i] < holeThr) isHole[i] = 1;
  }}
  if (!useMaxArea) return isHole;

  const visited = new Uint8Array(n);
  const outPinholes = new Uint8Array(n);
  const queue = new Int32Array(n);

  for (let i = 0; i < n; i++) {{
    if (!isHole[i] || visited[i]) continue;
    let head = 0, tail = 0;
    queue[tail++] = i;
    visited[i] = 1;

    while (head < tail) {{
      const cur = queue[head++];
      const cx = cur % w;
      const cy = (cur / w) | 0;
      if (cx > 0) {{ const nb = cur - 1; if (isHole[nb] && !visited[nb]) {{ visited[nb] = 1; queue[tail++] = nb; }} }}
      if (cx + 1 < w) {{ const nb = cur + 1; if (isHole[nb] && !visited[nb]) {{ visited[nb] = 1; queue[tail++] = nb; }} }}
      if (cy > 0) {{ const nb = cur - w; if (isHole[nb] && !visited[nb]) {{ visited[nb] = 1; queue[tail++] = nb; }} }}
      if (cy + 1 < h) {{ const nb = cur + w; if (isHole[nb] && !visited[nb]) {{ visited[nb] = 1; queue[tail++] = nb; }} }}
    }}

    if (tail <= maxArea) {{
      for (let k = 0; k < tail; k++) outPinholes[queue[k]] = 1;
    }}
  }}
  return outPinholes;
}}

function dilateBinaryMask(mask, w, h, radius) {{
  if (radius <= 0) return new Uint8Array(mask);
  let cur = new Uint8Array(mask);
  let nxt = new Uint8Array(w * h);
  for (let it = 0; it < radius; it++) {{
    nxt.fill(0);
    for (let y = 0; y < h; y++) {{
      const row = y * w;
      for (let x = 0; x < w; x++) {{
        const idx = row + x;
        if (
          cur[idx] ||
          (x > 0 && cur[idx - 1]) ||
          (x + 1 < w && cur[idx + 1]) ||
          (y > 0 && cur[idx - w]) ||
          (y + 1 < h && cur[idx + w]) ||
          (x > 0 && y > 0 && cur[idx - w - 1]) ||
          (x + 1 < w && y > 0 && cur[idx - w + 1]) ||
          (x > 0 && y + 1 < h && cur[idx + w - 1]) ||
          (x + 1 < w && y + 1 < h && cur[idx + w + 1])
        ) {{
          nxt[idx] = 1;
        }}
      }}
    }}
    cur.set(nxt);
  }}
  return cur;
}}

function pullPushPyramidInpaint(rgbaBytes, defectMask, w, h) {{
  const levels = [];
  const l0_r = new Float32Array(w * h);
  const l0_g = new Float32Array(w * h);
  const l0_b = new Float32Array(w * h);
  const l0_w = new Float32Array(w * h);

  for (let i = 0; i < w * h; i++) {{
    if (defectMask[i] === 0 && rgbaBytes[i * 4 + 3] >= 32) {{
      l0_r[i] = rgbaBytes[i * 4];
      l0_g[i] = rgbaBytes[i * 4 + 1];
      l0_b[i] = rgbaBytes[i * 4 + 2];
      l0_w[i] = 1.0;
    }}
  }}
  levels.push({{ w, h, r: l0_r, g: l0_g, b: l0_b, wt: l0_w }});

  for (let lv = 0; lv < 8; lv++) {{
    const prev = levels[lv];
    if (prev.w <= 4 || prev.h <= 4) break;
    const nw = prev.w >> 1;
    const nh = prev.h >> 1;
    const nr = new Float32Array(nw * nh);
    const ng = new Float32Array(nw * nh);
    const nb = new Float32Array(nw * nh);
    const nwt = new Float32Array(nw * nh);

    for (let y = 0; y < nh; y++) {{
      const py0 = y * 2, py1 = py0 + 1;
      for (let x = 0; x < nw; x++) {{
        const px0 = x * 2, px1 = px0 + 1;
        const i00 = py0 * prev.w + px0;
        const i10 = py0 * prev.w + px1;
        const i01 = py1 * prev.w + px0;
        const i11 = py1 * prev.w + px1;
        const wSum = prev.wt[i00] + prev.wt[i10] + prev.wt[i01] + prev.wt[i11];
        const dstIdx = y * nw + x;
        if (wSum > 0) {{
          nr[dstIdx] = (prev.r[i00] * prev.wt[i00] + prev.r[i10] * prev.wt[i10] + prev.r[i01] * prev.wt[i01] + prev.r[i11] * prev.wt[i11]) / wSum;
          ng[dstIdx] = (prev.g[i00] * prev.wt[i00] + prev.g[i10] * prev.wt[i10] + prev.g[i01] * prev.wt[i01] + prev.g[i11] * prev.wt[i11]) / wSum;
          nb[dstIdx] = (prev.b[i00] * prev.wt[i00] + prev.b[i10] * prev.wt[i10] + prev.b[i01] * prev.wt[i01] + prev.b[i11] * prev.wt[i11]) / wSum;
          nwt[dstIdx] = 1.0;
        }}
      }}
    }}
    levels.push({{ w: nw, h: nh, r: nr, g: ng, b: nb, wt: nwt }});
  }}

  for (let lv = levels.length - 2; lv >= 0; lv--) {{
    const fine = levels[lv];
    const coarse = levels[lv + 1];
    for (let y = 0; y < fine.h; y++) {{
      const cy = Math.min(coarse.h - 1, y >> 1);
      for (let x = 0; x < fine.w; x++) {{
        const idx = y * fine.w + x;
        if (fine.wt[idx] > 0) continue;
        let sr = 0, sg = 0, sb = 0, sw = 0;
        for (let dy = -1; dy <= 1; dy++) {{
          const ny = y + dy;
          if (ny < 0 || ny >= fine.h) continue;
          for (let dx = -1; dx <= 1; dx++) {{
            const nx = x + dx;
            if (nx < 0 || nx >= fine.w) continue;
            const nIdx = ny * fine.w + nx;
            if (fine.wt[nIdx] > 0) {{
              sr += fine.r[nIdx]; sg += fine.g[nIdx]; sb += fine.b[nIdx]; sw += 1.0;
            }}
          }}
        }}
        const cx = Math.min(coarse.w - 1, x >> 1);
        const cIdx = cy * coarse.w + cx;
        if (sw > 0 && coarse.wt[cIdx] > 0) {{
          fine.r[idx] = 0.65 * (sr / sw) + 0.35 * coarse.r[cIdx];
          fine.g[idx] = 0.65 * (sg / sw) + 0.35 * coarse.g[cIdx];
          fine.b[idx] = 0.65 * (sb / sw) + 0.35 * coarse.b[cIdx];
          fine.wt[idx] = 1.0;
        }} else if (sw > 0) {{
          fine.r[idx] = sr / sw; fine.g[idx] = sg / sw; fine.b[idx] = sb / sw; fine.wt[idx] = 1.0;
        }} else if (coarse.wt[cIdx] > 0) {{
          fine.r[idx] = coarse.r[cIdx]; fine.g[idx] = coarse.g[cIdx]; fine.b[idx] = coarse.b[cIdx]; fine.wt[idx] = 1.0;
        }}
      }}
    }}
  }}
  return levels[0];
}}

function applyLiveFilterToSubmesh(asset) {{
  const srcImgData = (filterCfg.texSource === 'stage1' && asset.stage1ImgData) ? asset.stage1ImgData : asset.rawImgData;
  const rawData = srcImgData.data;
  const w = srcImgData.width;
  const h = srcImgData.height;
  const n = w * h;
  const outImg = asset.ctx.createImageData(w, h);
  const out = outImg.data;

  const algo = filterCfg.algo;
  const holeThr = filterCfg.holeAlpha;
  const ringPx = filterCfg.ringPx;
  const maxArea = filterCfg.maxArea;
  const spikeThr = filterCfg.spikeThr;
  const showDefects = filterCfg.showDefects;
  const maskGray = asset.maskGray;

  if (algo === 'raw' && !showDefects) {{
    out.set(rawData);
    if (maskGray) {{
      for (let i = 0; i < n; i++) {{
        out[i * 4 + 3] = ((out[i * 4 + 3] * maskGray[i]) / 255) | 0;
      }}
    }}
    asset.ctx.putImageData(outImg, 0, 0);
    asset.threeTextures.forEach(t => {{ t.needsUpdate = true; }});
    return {{ holeCnt: 0, ringCnt: 0 }};
  }}

  if (algo === 'opaque_only' && !showDefects) {{
    out.set(rawData);
    for (let i = 0; i < n; i++) {{
      out[i * 4 + 3] = maskGray ? maskGray[i] : 255;
    }}
    asset.ctx.putImageData(outImg, 0, 0);
    asset.threeTextures.forEach(t => {{ t.needsUpdate = true; }});
    return {{ holeCnt: 0, ringCnt: 0 }};
  }}

  const alphaArr = new Uint8Array(n);
  for (let i = 0; i < n; i++) alphaArr[i] = rawData[i * 4 + 3];

  const useCC = (algo === 'cc_pinhole_pyramid');
  const holeCore = findConnectedHoleMask(alphaArr, w, h, holeThr, maxArea, useCC);
  const dilatedHole = dilateBinaryMask(holeCore, w, h, ringPx);
  const defectMask = new Uint8Array(dilatedHole);

  if (spikeThr < 115) {{
    const thrSq = spikeThr * spikeThr * 3.0;
    for (let y = 2; y < h - 2; y++) {{
      const row = y * w;
      for (let x = 2; x < w - 2; x++) {{
        const i = row + x;
        if (defectMask[i] || alphaArr[i] < 64) continue;
        let sr = 0, sg = 0, sb = 0, cnt = 0;
        for (let dy = -2; dy <= 2; dy += 2) {{
          for (let dx = -2; dx <= 2; dx += 2) {{
            if (dx === 0 && dy === 0) continue;
            const ni = (y + dy) * w + (x + dx);
            if (!defectMask[ni] && alphaArr[ni] >= holeThr) {{
              sr += rawData[ni * 4]; sg += rawData[ni * 4 + 1]; sb += rawData[ni * 4 + 2]; cnt++;
            }}
          }}
        }}
        if (cnt >= 3) {{
          const dr = rawData[i * 4] - sr / cnt;
          const dg = rawData[i * 4 + 1] - sg / cnt;
          const db = rawData[i * 4 + 2] - sb / cnt;
          if (dr * dr + dg * dg + db * db > thrSq) {{
            defectMask[i] = 1;
          }}
        }}
      }}
    }}
  }}

  let holeCnt = 0, ringCnt = 0;
  for (let i = 0; i < n; i++) {{
    if (holeCore[i]) holeCnt++;
    else if (defectMask[i]) ringCnt++;
  }}

  if (showDefects) {{
    for (let i = 0; i < n; i++) {{
      if (holeCore[i]) {{
        out[i * 4] = 255; out[i * 4 + 1] = 20; out[i * 4 + 2] = 60; out[i * 4 + 3] = 255;
      }} else if (defectMask[i]) {{
        out[i * 4] = 255; out[i * 4 + 1] = 230; out[i * 4 + 2] = 0; out[i * 4 + 3] = 255;
      }} else {{
        out[i * 4] = rawData[i * 4];
        out[i * 4 + 1] = rawData[i * 4 + 1];
        out[i * 4 + 2] = rawData[i * 4 + 2];
        out[i * 4 + 3] = maskGray ? maskGray[i] : 255;
      }}
    }}
    asset.ctx.putImageData(outImg, 0, 0);
    asset.threeTextures.forEach(t => {{ t.needsUpdate = true; }});
    return {{ holeCnt, ringCnt }};
  }}

  if (algo === 'local_window_avg') {{
    const rArr = new Float32Array(n);
    const gArr = new Float32Array(n);
    const bArr = new Float32Array(n);
    const valid = new Uint8Array(n);
    for (let i = 0; i < n; i++) {{
      rArr[i] = rawData[i * 4]; gArr[i] = rawData[i * 4 + 1]; bArr[i] = rawData[i * 4 + 2];
      valid[i] = (defectMask[i] === 0 && alphaArr[i] >= holeThr) ? 1 : 0;
    }}
    for (let pass = 0; pass < 16; pass++) {{
      let changed = 0;
      for (let y = 1; y < h - 1; y++) {{
        const row = y * w;
        for (let x = 1; x < w - 1; x++) {{
          const i = row + x;
          if (valid[i]) continue;
          let sr = 0, sg = 0, sb = 0, c = 0;
          for (let dy = -1; dy <= 1; dy++) {{
            for (let dx = -1; dx <= 1; dx++) {{
              const ni = (y + dy) * w + (x + dx);
              if (valid[ni] === 1) {{ sr += rArr[ni]; sg += gArr[ni]; sb += bArr[ni]; c++; }}
            }}
          }}
          if (c > 0) {{
            rArr[i] = sr / c; gArr[i] = sg / c; bArr[i] = sb / c;
            valid[i] = 2; changed++;
          }}
        }}
      }}
      for (let i = 0; i < n; i++) if (valid[i] === 2) valid[i] = 1;
      if (changed === 0) break;
    }}
    for (let i = 0; i < n; i++) {{
      out[i * 4] = rArr[i] | 0; out[i * 4 + 1] = gArr[i] | 0; out[i * 4 + 2] = bArr[i] | 0;
      // Only make pixels opaque if they are now valid (healed or originally valid);
      // if an interior hole remained unreached after 16 passes, keep its original alpha so raw noise is never shown!
      const baseA = valid[i] ? 255 : rawData[i * 4 + 3];
      out[i * 4 + 3] = maskGray ? (((baseA * maskGray[i]) / 255) | 0) : baseA;
    }}
  }} else {{
    const allLowAlphaAndDefects = new Uint8Array(n);
    for (let i = 0; i < n; i++) {{
      if (defectMask[i] || alphaArr[i] < holeThr) allLowAlphaAndDefects[i] = 1;
    }}
    const filled = pullPushPyramidInpaint(rawData, allLowAlphaAndDefects, w, h);
    for (let i = 0; i < n; i++) {{
      if (defectMask[i]) {{
        out[i * 4] = filled.r[i] | 0;
        out[i * 4 + 1] = filled.g[i] | 0;
        out[i * 4 + 2] = filled.b[i] | 0;
        out[i * 4 + 3] = maskGray ? maskGray[i] : 255;
      }} else if (alphaArr[i] < holeThr) {{
        out[i * 4] = filled.r[i] | 0;
        out[i * 4 + 1] = filled.g[i] | 0;
        out[i * 4 + 2] = filled.b[i] | 0;
        const baseA = (algo === 'all_holes_pyramid') ? 255 : rawData[i * 4 + 3];
        out[i * 4 + 3] = maskGray ? (((baseA * maskGray[i]) / 255) | 0) : baseA;
      }} else {{
        out[i * 4] = rawData[i * 4];
        out[i * 4 + 1] = rawData[i * 4 + 1];
        out[i * 4 + 2] = rawData[i * 4 + 2];
        out[i * 4 + 3] = maskGray ? maskGray[i] : 255;
      }}
    }}
  }}

  asset.ctx.putImageData(outImg, 0, 0);
  asset.threeTextures.forEach(t => {{ t.needsUpdate = true; }});
  return {{ holeCnt, ringCnt }};
}}

function runLiveFilterOnAllActiveSubmeshes() {{
  const t0 = performance.now();
  let totHoles = 0, totRings = 0, count = 0;
  for (const stem of Object.keys(activeSubmeshAssets)) {{
    const st = applyLiveFilterToSubmesh(activeSubmeshAssets[stem]);
    totHoles += st.holeCnt;
    totRings += st.ringCnt;
    count++;
  }}
  const dtMs = (performance.now() - t0).toFixed(0);
  const perfEl = document.getElementById('lbl-filter-perf');
  if (perfEl) {{
    perfEl.textContent = `Filtered ${{count}} textures in ${{dtMs}}ms | Healed: ${{(totHoles/1000).toFixed(1)}}k hole px + ${{(totRings/1000).toFixed(1)}}k ring/spike px`;
  }}
}}

function applyShadingToViewport(vp, isOurs = false) {{
  if (!vp.rootGroup) return;

  vp.rootGroup.traverse((child) => {{
    if (!child.isMesh || child.userData.isWireOverlay) return;
    if (!child.userData.origMaterial) {{
      child.userData.origMaterial = child.material;
    }}
    const map = child.userData.liveCanvasTex || (child.userData.origMaterial && child.userData.origMaterial.map) || null;
    if (map) {{
      map.colorSpace = THREE.SRGBColorSpace;
      const sMode = isOurs ? oursSamplingMode : targetSamplingMode;
      let wantMag = THREE.LinearFilter;
      let wantMin = THREE.LinearMipmapLinearFilter;
      let wantMips = true;
      let wantAniso = 8;
      if (sMode === 'bilinear') {{
        wantMag = THREE.LinearFilter;
        wantMin = THREE.LinearFilter;
        wantMips = false;
        wantAniso = 1;
      }} else if (sMode === 'nearest_mip') {{
        wantMag = THREE.NearestFilter;
        wantMin = THREE.LinearMipmapLinearFilter;
        wantMips = true;
        wantAniso = 4;
      }} else if (sMode === 'nearest') {{
        wantMag = THREE.NearestFilter;
        wantMin = THREE.NearestFilter;
        wantMips = false;
        wantAniso = 1;
      }}
      if (map.magFilter !== wantMag || map.minFilter !== wantMin || map.generateMipmaps !== wantMips) {{
        map.magFilter = wantMag;
        map.minFilter = wantMin;
        map.generateMipmaps = wantMips;
        map.anisotropy = wantAniso;
        map.needsUpdate = true;
      }}
    }}

    const bsdf = child.userData.bsdfRule || 'ambient_2sided';
    const isOneSided = bsdf.includes('1sided');
    const useFrontCull = (alphaCullMode !== 'double_clip') && isOneSided;
    const sideMode = useFrontCull ? THREE.FrontSide : THREE.DoubleSide;
    const useBlend = (alphaCullMode === 'bsdf_blend');

    if (shadingMode === 'albedo' || shadingMode === 'albedo_wire') {{
      child.material = new THREE.MeshBasicMaterial({{
        map: map,
        color: map ? 0xffffff : 0xcccccc,
        alphaTest: useBlend ? 0.02 : 0.5,
        transparent: useBlend,
        depthWrite: true,
        side: sideMode,
      }});
    }} else if (shadingMode === 'clay_wire') {{
      child.material = new THREE.MeshStandardMaterial({{
        color: 0x94a3b8,
        roughness: 0.65,
        metalness: 0.08,
        side: sideMode,
      }});
    }}

    if (shadingMode === 'albedo_wire' || shadingMode === 'clay_wire') {{
      if (!child.userData.wireClone) {{
        const wireMat = new THREE.MeshBasicMaterial({{
          color: shadingMode === 'clay_wire' ? 0x0f172a : 0x38bdf8,
          wireframe: true,
          transparent: true,
          opacity: shadingMode === 'clay_wire' ? 0.55 : 0.38,
          depthTest: true,
        }});
        let wMesh = child.isSkinnedMesh
          ? new THREE.SkinnedMesh(child.geometry, wireMat)
          : new THREE.Mesh(child.geometry, wireMat);
        if (child.isSkinnedMesh) wMesh.bind(child.skeleton, child.bindMatrix);
        wMesh.userData.isWireOverlay = true;
        child.add(wMesh);
        child.userData.wireClone = wMesh;
      }} else {{
        child.userData.wireClone.visible = true;
        child.userData.wireClone.material.color.setHex(shadingMode === 'clay_wire' ? 0x0f172a : 0x38bdf8);
        child.userData.wireClone.material.opacity = shadingMode === 'clay_wire' ? 0.55 : 0.38;
      }}
    }} else if (child.userData.wireClone) {{
      child.userData.wireClone.visible = false;
    }}
  }});
}}

function seekBothToFrame(frameIdx) {{
  currentFrame = Math.max(0, Math.min(TOTAL_FRAMES - 1, Math.round(frameIdx)));
  const slider = document.getElementById('slider-frame');
  if (slider) slider.value = String(currentFrame);

  const isTest = currentFrame >= 190;
  const readout = document.getElementById('lbl-frame-readout');
  if (readout) {{
    const splitBadge = isTest
      ? '<span class="split-tag-test">TEST SPLIT</span>'
      : '<span class="split-tag-train">TRAIN SPLIT</span>';
    readout.innerHTML = `Frame: ${{String(currentFrame).padStart(3, '0')}} / 199 (${{splitBadge}})`;
  }}

  const timeSec = currentFrame / FPS;
  [vpL, vpR].forEach(vp => {{
    if (vp.mixer) vp.mixer.setTime(timeSec);
  }});
}}

const loader = new GLTFLoader();
let loadToken = 0;

function clearViewportModel(vp) {{
  if (vp.mixer) {{
    vp.mixer.stopAllAction();
    vp.mixer = null;
  }}
  if (vp.rootGroup) {{
    vp.scene.remove(vp.rootGroup);
    vp.rootGroup = null;
  }}
}}

function loadGltfPromise(url, onProgress) {{
  return new Promise((resolve, reject) => {{
    loader.load(url, resolve, onProgress, reject);
  }});
}}

function loadImageElement(url) {{
  return new Promise((resolve, reject) => {{
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  }});
}}

async function loadSubmeshRawTextures(rec, rootL, rootR) {{
  activeSubmeshAssets = {{}};
  const subs = rec.submesh_metrics || [];
  const texMount = document.getElementById('tex-grid-mount');
  texMount.innerHTML = '';

  const tmpCanvas = document.createElement('canvas');
  const tmpCtx = tmpCanvas.getContext('2d', {{ willReadFrequently: true }});

  await Promise.all(subs.map(async (s) => {{
    const stem = s.submesh;
    const rawImg = await loadImageElement(s.rel_raw_texture);
    const w = rawImg.width, h = rawImg.height;
    tmpCanvas.width = w; tmpCanvas.height = h;
    tmpCtx.clearRect(0, 0, w, h);
    tmpCtx.drawImage(rawImg, 0, 0);
    const rawImgData = tmpCtx.getImageData(0, 0, w, h);

    let stage1ImgData = null;
    if (s.rel_stage1_texture) {{
      try {{
        const s1Img = await loadImageElement(s.rel_stage1_texture);
        tmpCtx.clearRect(0, 0, w, h);
        tmpCtx.drawImage(s1Img, 0, 0, w, h);
        stage1ImgData = tmpCtx.getImageData(0, 0, w, h);
      }} catch (_) {{}}
    }}

    let maskGray = null;
    if (s.rel_body_mask) {{
      try {{
        const mImg = await loadImageElement(s.rel_body_mask);
        tmpCtx.clearRect(0, 0, w, h);
        tmpCtx.drawImage(mImg, 0, 0, w, h);
        const mData = tmpCtx.getImageData(0, 0, w, h).data;
        maskGray = new Uint8Array(w * h);
        for (let i = 0; i < w * h; i++) maskGray[i] = mData[i * 4];
      }} catch (_) {{}}
    }}

    const card = document.createElement('div');
    card.className = 'tex-card';
    const dispCanvas = document.createElement('canvas');
    dispCanvas.width = w;
    dispCanvas.height = h;
    dispCanvas.className = 'bg-checker';
    const dispCtx = dispCanvas.getContext('2d');

    const meta = document.createElement('div');
    meta.className = 'tex-card-meta';
    meta.innerHTML = `<div style="font-weight:700;color:#38bdf8;">${{stem}}.png (${{s.opt_bsdf}})</div>
      <div>V=${{s.vertices}} | F=${{s.faces}}${{s.body_mask_applied ? ' | 🎭 Mask' : ''}}</div>
      <div>PSNR: <b>${{fmtNum(s.stage02_test_psnr, 2)}} dB</b></div>`;

    card.appendChild(dispCanvas);
    card.appendChild(meta);
    texMount.appendChild(card);

    activeSubmeshAssets[stem] = {{
      stem,
      optBsdf: s.opt_bsdf || 'ambient_2sided',
      refBsdf: s.ref_bsdf || 'ambient_2sided',
      rawImgData,
      stage1ImgData,
      maskGray,
      canvas: dispCanvas,
      ctx: dispCtx,
      threeTextures: [],
    }};
  }}));

  const norm = (str) => String(str || '').toLowerCase().replace(/[^a-z0-9]+/g, '');

  rootL.traverse((child) => {{
    if (!child.isMesh || child.userData.isWireOverlay) return;
    const cName = child.name || '';
    for (const k of Object.keys(activeSubmeshAssets)) {{
      if (norm(cName) === norm(k) || norm(cName).includes(norm(k))) {{
        child.userData.bsdfRule = activeSubmeshAssets[k].refBsdf;
        break;
      }}
    }}
  }});

  rootR.traverse((child) => {{
    if (!child.isMesh || child.userData.isWireOverlay) return;
    const cName = child.name || '';
    const matName = (child.material && child.material.name) || '';
    let matchedAsset = activeSubmeshAssets[cName] || activeSubmeshAssets[matName];
    if (!matchedAsset) {{
      for (const k of Object.keys(activeSubmeshAssets)) {{
        if (norm(cName).includes(norm(k)) || norm(matName).includes(norm(k))) {{
          matchedAsset = activeSubmeshAssets[k];
          break;
        }}
      }}
    }}
    if (matchedAsset) {{
      child.userData.bsdfRule = matchedAsset.optBsdf;
      const origMap = (child.material && child.material.map) || null;
      const cTex = new THREE.CanvasTexture(matchedAsset.canvas);
      cTex.colorSpace = THREE.SRGBColorSpace;
      cTex.flipY = origMap ? origMap.flipY : false;
      cTex.wrapS = origMap ? origMap.wrapS : THREE.RepeatWrapping;
      cTex.wrapT = origMap ? origMap.wrapT : THREE.RepeatWrapping;
      child.userData.liveCanvasTex = cTex;
      matchedAsset.threeTextures.push(cTex);
    }}
  }});

  runLiveFilterOnAllActiveSubmeshes();
}}

async function selectModel(modelName) {{
  const rec = models.find(m => m.model === modelName);
  if (!rec) return;
  activeModelName = modelName;
  const myToken = ++loadToken;

  document.querySelectorAll('.model-tab-btn').forEach(btn => {{
    btn.classList.toggle('active', btn.getAttribute('data-model') === modelName);
  }});

  const tSt = rec.target_stats || {{}};
  const oSt = rec.ours_stats || {{}};
  const fRed = tSt.faces > 0 ? ((1.0 - oSt.faces / tSt.faces) * 100.0).toFixed(1) : '0.0';
  document.getElementById('lbl-target-title').textContent = `${{modelName}} — Target (Ground Truth)`;
  document.getElementById('lbl-ours-title').textContent = `${{modelName}} — Ours (Step 05 + Live Filter)`;
  document.getElementById('lbl-target-stats').innerHTML = `V: <b>${{(tSt.vertices||0).toLocaleString()}}</b> | F: <b>${{(tSt.faces||0).toLocaleString()}}</b> | Bones: <b>${{tSt.bones||0}}</b>`;
  document.getElementById('lbl-ours-stats').innerHTML = `V: <b>${{(oSt.vertices||0).toLocaleString()}}</b> | F: <b>${{(oSt.faces||0).toLocaleString()}} (-${{fRed}}%)</b> | Bones: <b>${{oSt.bones||0}}</b>`;

  renderSubmeshTableOnly(rec);

  const dlT = document.getElementById('link-dl-target-glb');
  const dlO = document.getElementById('link-dl-ours-glb');
  if (dlT) dlT.href = rec.target_glb || '#';
  if (dlO) dlO.href = rec.ours_glb || '#';

  clearViewportModel(vpL);
  clearViewportModel(vpR);

  if (rec.status !== 'READY' || !rec.target_glb || !rec.ours_glb) {{
    statusL.style.display = 'block';
    statusR.style.display = 'block';
    statusL.textContent = `${{modelName}} is currently training on GPU...`;
    statusR.textContent = `${{modelName}} is currently training on GPU...`;
    return;
  }}

  statusL.style.display = 'block';
  statusR.style.display = 'block';
  statusL.textContent = `Loading ${{rec.target_glb}} ...`;
  statusR.textContent = `Loading ${{rec.ours_glb}} ...`;

  try {{
    const [gltfL, gltfR] = await Promise.all([
      loadGltfPromise(rec.target_glb, (ev) => {{
        if (ev.total) statusL.textContent = `Loading Target GLB (${{Math.round(ev.loaded/ev.total*100)}}%)...`;
      }}),
      loadGltfPromise(rec.ours_glb, (ev) => {{
        if (ev.total) statusR.textContent = `Loading Ours GLB (${{Math.round(ev.loaded/ev.total*100)}}%)...`;
      }}),
    ]);
    if (myToken !== loadToken) return;

    function setupLoadedScene(vp, gltf) {{
      const root = gltf.scene;
      vp.rootGroup = new THREE.Group();
      vp.rootGroup.add(root);
      vp.scene.add(vp.rootGroup);
      if (gltf.animations && gltf.animations.length > 0) {{
        vp.mixer = new THREE.AnimationMixer(root);
        gltf.animations.forEach(clip => {{
          vp.mixer.clipAction(clip).play();
        }});
        vp.mixer.setTime(0);
      }}
      root.updateMatrixWorld(true);
      return root;
    }}

    const rootL = setupLoadedScene(vpL, gltfL);
    const rootR = setupLoadedScene(vpR, gltfR);

    const box = new THREE.Box3().setFromObject(rootR);
    if (box.isEmpty()) box.setFromObject(rootL);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1e-4);
    const normScale = 1.75 / maxDim;

    [rootL, rootR].forEach(r => {{
      r.scale.setScalar(normScale);
      r.position.set(-center.x * normScale, -center.y * normScale, -center.z * normScale);
    }});

    await loadSubmeshRawTextures(rec, rootL, rootR);
    applyShadingToViewport(vpL, false);
    applyShadingToViewport(vpR, true);
    seekBothToFrame(currentFrame);

    statusL.style.display = 'none';
    statusR.style.display = 'none';
  }} catch (err) {{
    console.error('GLB Load error:', err);
    statusL.textContent = 'Failed to load GLB: ' + err.message;
    statusR.textContent = 'Failed to load GLB: ' + err.message;
  }}
}}

function fmtNum(v, d = 2) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '-';
  return Number(v).toFixed(d);
}}

function renderModelTabsAndAllModelsTable() {{
  const tabsMount = document.getElementById('model-tabs-mount');
  tabsMount.innerHTML = models.map(m => {{
    const badge = m.status === 'READY' ? '✅' : '⏳';
    const cls = m.model === activeModelName ? 'model-tab-btn active' : 'model-tab-btn';
    return `<button class="${{cls}}" data-model="${{m.model}}">${{badge}} ${{m.model}}</button>`;
  }}).join('');

  tabsMount.querySelectorAll('.model-tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => selectModel(btn.getAttribute('data-model')));
  }});

  const tblMount = document.getElementById('table-all-models-mount');
  const rowsHtml = models.map(m => {{
    const t = m.target_stats || {{}};
    const o = m.ours_stats || {{}};
    const mt = m.metrics || {{}};
    const fRed = t.faces > 0 ? ((1.0 - o.faces / t.faces) * 100.0).toFixed(1) + '%' : '-';
    const dPsnr = (mt.stage02_test_psnr && mt.stage01_test_psnr)
      ? ('+' + (mt.stage02_test_psnr - mt.stage01_test_psnr).toFixed(2) + ' dB')
      : '-';
    return `<tr class="hoverable-row" onclick="window.selectDiffRenderModel('${{m.model}}')">
      <td><b>${{m.model}}</b></td>
      <td><span class="badge ${{m.status === 'READY' ? 'badge-ready' : 'badge-pending'}}">${{m.status}}</span></td>
      <td class="num-col">${{(t.vertices||0).toLocaleString()}} / ${{(t.faces||0).toLocaleString()}}</td>
      <td class="num-col"><b>${{(o.vertices||0).toLocaleString()}} / ${{(o.faces||0).toLocaleString()}}</b></td>
      <td class="num-col" style="color:#0f766e;font-weight:700;">-${{fRed}}</td>
      <td class="num-col">${{t.bones||0}} → <b>${{o.bones||0}}</b></td>
      <td class="num-col">${{fmtNum(mt.stage01_test_psnr, 2)}} dB</td>
      <td class="num-col best-cell">${{fmtNum(mt.stage02_test_psnr, 2)}} dB (${{dPsnr}})</td>
      <td class="num-col">${{fmtNum(mt.stage01_test_lpips, 4)}}</td>
      <td class="num-col best-cell">${{fmtNum(mt.stage02_test_lpips, 4)}}</td>
      <td class="num-col">${{fmtNum(mt.stage02_test_hausdorff, 4)}}</td>
    </tr>`;
  }}).join('');

  tblMount.innerHTML = `<table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Status</th>
        <th class="num-col">Target V / F</th>
        <th class="num-col">Ours V / F</th>
        <th class="num-col">Face Reduction</th>
        <th class="num-col">Bones (Target → Ours)</th>
        <th class="num-col">Stage 01 Test PSNR ↑</th>
        <th class="num-col">Stage 02 Test PSNR ↑ (Δ)</th>
        <th class="num-col">Stage 01 Test LPIPS ↓</th>
        <th class="num-col">Stage 02 Test LPIPS ↓</th>
        <th class="num-col">Test Hausdorff ↓</th>
      </tr>
    </thead>
    <tbody>${{rowsHtml}}</tbody>
  </table>`;
}}

function renderSubmeshTableOnly(rec) {{
  document.getElementById('hdr-submesh-section').textContent =
    `🧩 ${{rec.model}} — Live Filtered 2D Submesh Textures (${{(rec.submesh_metrics||[]).length}} Submeshes)`;
  const subMount = document.getElementById('table-submesh-mount');
  const subs = rec.submesh_metrics || [];
  if (!subs.length) {{
    subMount.innerHTML = '<div style="padding:14px;color:#64748b;">Waiting for training completion...</div>';
    return;
  }}
  subMount.innerHTML = `<table>
    <thead>
      <tr>
        <th>Submesh Name</th>
        <th>Opt BSDF</th>
        <th>Ref BSDF</th>
        <th class="num-col">Vertices</th>
        <th class="num-col">Triangles</th>
        <th>Body Alpha Mask</th>
        <th class="num-col">Stage 01 Test PSNR (dB) ↑</th>
        <th class="num-col">Stage 02 Test PSNR (dB) ↑</th>
        <th class="num-col">Stage 02 Test LPIPS ↓</th>
      </tr>
    </thead>
    <tbody>
      ${{subs.map(s => `<tr>
        <td><b>${{s.submesh}}</b></td>
        <td><code>${{s.opt_bsdf}}</code></td>
        <td><code>${{s.ref_bsdf}}</code></td>
        <td class="num-col">${{(s.vertices||0).toLocaleString()}}</td>
        <td class="num-col">${{(s.faces||0).toLocaleString()}}</td>
        <td>${{s.body_mask_applied ? '<span class="badge badge-ready">YES (Body Cutout Mask)</span>' : '-'}}</td>
        <td class="num-col">${{fmtNum(s.stage01_test_psnr, 2)}}</td>
        <td class="num-col best-cell">${{fmtNum(s.stage02_test_psnr, 2)}}</td>
        <td class="num-col">${{fmtNum(s.stage02_test_lpips, 4)}}</td>
      </tr>`).join('')}}
    </tbody>
  </table>`;
}}

window.selectDiffRenderModel = selectModel;

document.getElementById('sel-tex-source').addEventListener('change', (e) => {{
  filterCfg.texSource = e.target.value;
  runLiveFilterOnAllActiveSubmeshes();
}});

document.getElementById('sel-filter-algo').addEventListener('change', (e) => {{
  filterCfg.algo = e.target.value;
  runLiveFilterOnAllActiveSubmeshes();
}});

document.getElementById('chk-show-defects').addEventListener('change', (e) => {{
  filterCfg.showDefects = Boolean(e.target.checked);
  runLiveFilterOnAllActiveSubmeshes();
}});

function bindRangeControl(id, valId, key, suffix = '') {{
  const el = document.getElementById(id);
  const vEl = document.getElementById(valId);
  el.addEventListener('input', () => {{
    filterCfg[key] = Number(el.value);
    vEl.textContent = el.value + suffix;
    runLiveFilterOnAllActiveSubmeshes();
  }});
}}
bindRangeControl('rng-hole-alpha', 'val-hole-alpha', 'holeAlpha', '');
bindRangeControl('rng-ring-px', 'val-ring-px', 'ringPx', ' px');
bindRangeControl('rng-max-area', 'val-max-area', 'maxArea', '');
bindRangeControl('rng-spike-thr', 'val-spike-thr', 'spikeThr', '');

document.getElementById('btn-reset-filter').addEventListener('click', () => {{
  filterCfg.texSource = 'stage1';
  filterCfg.algo = 'local_window_avg';
  filterCfg.holeAlpha = 128;
  filterCfg.ringPx = 1;
  filterCfg.maxArea = 1200;
  filterCfg.spikeThr = 42;
  filterCfg.showDefects = false;
  document.getElementById('sel-tex-source').value = 'stage1';
  document.getElementById('sel-filter-algo').value = 'local_window_avg';
  document.getElementById('chk-show-defects').checked = false;
  document.getElementById('rng-hole-alpha').value = '128';
  document.getElementById('val-hole-alpha').textContent = '128';
  document.getElementById('rng-ring-px').value = '1';
  document.getElementById('val-ring-px').textContent = '1 px';
  document.getElementById('rng-max-area').value = '1200';
  document.getElementById('val-max-area').textContent = '1200';
  document.getElementById('rng-spike-thr').value = '42';
  document.getElementById('val-spike-thr').textContent = '42';
  runLiveFilterOnAllActiveSubmeshes();
}});

document.getElementById('sel-shading-mode').addEventListener('change', (e) => {{
  shadingMode = e.target.value;
  applyShadingToViewport(vpL, false);
  applyShadingToViewport(vpR, true);
}});

document.getElementById('sel-target-sampling').addEventListener('change', (e) => {{
  targetSamplingMode = e.target.value;
  applyShadingToViewport(vpL, false);
}});

document.getElementById('sel-ours-sampling').addEventListener('change', (e) => {{
  oursSamplingMode = e.target.value;
  applyShadingToViewport(vpR, true);
}});

document.getElementById('sel-alpha-cull').addEventListener('change', (e) => {{
  alphaCullMode = e.target.value;
  applyShadingToViewport(vpL, false);
  applyShadingToViewport(vpR, true);
}});

document.getElementById('sel-bg-mode').addEventListener('change', (e) => {{
  const cls = e.target.value;
  [wrapL, wrapR].forEach(w => {{
    w.classList.remove('bg-checker', 'bg-dark', 'bg-white');
    w.classList.add(cls);
  }});
}});

document.querySelectorAll('[data-cam]').forEach(btn => {{
  btn.addEventListener('click', () => setCameraPreset(btn.getAttribute('data-cam')));
}});

const btnPlay = document.getElementById('btn-play-toggle');
btnPlay.addEventListener('click', () => {{
  isPlaying = !isPlaying;
  btnPlay.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
}});

document.getElementById('btn-prev-frame').addEventListener('click', () => {{
  isPlaying = false;
  btnPlay.textContent = '▶ Play';
  seekBothToFrame((currentFrame - 1 + TOTAL_FRAMES) % TOTAL_FRAMES);
}});

document.getElementById('btn-next-frame').addEventListener('click', () => {{
  isPlaying = false;
  btnPlay.textContent = '▶ Play';
  seekBothToFrame((currentFrame + 1) % TOTAL_FRAMES);
}});

document.getElementById('slider-frame').addEventListener('input', (e) => {{
  isPlaying = false;
  btnPlay.textContent = '▶ Play';
  seekBothToFrame(Number(e.target.value));
}});

document.querySelectorAll('[data-jump-frame]').forEach(btn => {{
  btn.addEventListener('click', () => {{
    isPlaying = false;
    btnPlay.textContent = '▶ Play';
    seekBothToFrame(Number(btn.getAttribute('data-jump-frame')));
  }});
}});

document.getElementById('sel-speed').addEventListener('change', (e) => {{
  playbackSpeed = Number(e.target.value) || 1.0;
}});

window.addEventListener('resize', () => {{
  [vpL, vpR].forEach(vp => {{
    const w = vp.container.clientWidth;
    const h = Math.max(1, vp.container.clientHeight);
    vp.camera.aspect = w / h;
    vp.camera.updateProjectionMatrix();
    vp.renderer.setSize(w, h);
  }});
}});

function animate(nowMs) {{
  requestAnimationFrame(animate);
  const dtSec = Math.min(0.1, Math.max(0.0, (nowMs - lastTimeMs) / 1000.0));
  lastTimeMs = nowMs;

  if (isPlaying) {{
    frameAccumulator += dtSec * FPS * playbackSpeed;
    if (frameAccumulator >= 1.0) {{
      const step = Math.floor(frameAccumulator);
      frameAccumulator -= step;
      seekBothToFrame((currentFrame + step) % TOTAL_FRAMES);
    }}
  }}

  vpL.renderer.render(vpL.scene, vpL.camera);
  vpR.renderer.render(vpR.scene, vpR.camera);
}}

renderModelTabsAndAllModelsTable();
selectModel(activeModelName);
requestAnimationFrame(animate);
</script>
</body>
</html>
"""
    out_html = output_dir / "Diff_Render_Report.html"
    out_html.write_text(html_doc, encoding="utf-8")
    return out_html


def generate_diff_render_report(
    *,
    pool_dir: Path = DEFAULT_POOL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    export_glb: bool = True,
    overwrite_glb: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_data = discover_diff_render_models(
        pool_dir.expanduser().resolve(strict=True),
        output_dir,
        export_glb=export_glb,
        overwrite_glb=overwrite_glb,
    )
    out_html = render_diff_render_html(models_data, output_dir)
    ready_cnt = sum(1 for m in models_data if m["status"] == "READY")
    return {
        "status": "GENERATED",
        "html": str(out_html),
        "ready_models": ready_cnt,
        "total_models": len(models_data),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Step 05 Differentiable Rendering 3D Comparison Report.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ui-only", action="store_true", help="Skip GLB export if GLBs already exist.")
    parser.add_argument("--overwrite-glb", action="store_true", help="Force re-exporting GLBs.")
    args = parser.parse_args()

    res = generate_diff_render_report(
        pool_dir=args.pool,
        output_dir=args.output_dir,
        export_glb=True,
        overwrite_glb=args.overwrite_glb,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
