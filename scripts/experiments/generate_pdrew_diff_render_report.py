#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generate the Interactive 3D Triple-Viewport Differentiable Rendering & UE5 Baseline
Comparison Report (`Diff_Render_Report.html`) integrating:
  1. Target (Ground Truth)
  2. Ours (VDB + SSDR + Step 05 Diff-Render) from `ogd_lbs_ssdr_25_v1`
  3. UE-qem-lod (No Mesh Merge, Native UV) from `ogd_lbs_ue_simp_v1`
  4. UE-remesh-lod (UE5 Voxel Remesh + Weight Transfer + AutoUV + Bake) from `ogd_lbs_ue_simp_v1`
"""

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import zipfile
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
DEFAULT_UE_SIMP_POOL = Path("/workspace/pd_rew_exp_data/pools/ogd_lbs_ue_simp_v1")
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
        "skeleton_hierarchy_data_bone_reduced",
        "animation_data_bone_reduced",
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
        "skeleton_hierarchy_data_bone_reduced",
        "animation_data_bone_reduced",
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


def _discover_ue_simp_entries(ue_simp_pool: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {m: {} for m in ALL_MODELS}
    csv_path = ue_simp_pool / "metadata_table.csv"
    if not csv_path.is_file():
        return out

    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            model = r.get("ue_simp_handoff.model_name", "")
            if model not in out:
                continue
            if r.get("ue_simp_stage04") in (None, "", "None"):
                continue
            quota = r.get("ue_simp_stage02.variant", "")
            method = r.get("ue_simp_stage03", "")
            is_mod = (quota == "bone_reduced_moderate")
            suffix = "mod" if is_mod else "cons"
            m_prefix = "ue_remesh" if method == "remesh_v1" else "ue_qem"
            key = f"{m_prefix}_{suffix}"
            out[model][key] = ue_simp_pool / r["data_dir"] / "data4"
    return out


def _find_ours_stage04_geom_metrics(pool_dir: Path, model: str, variant: str = "bone_reduced_conservative") -> dict[str, Any]:
    csv_path = pool_dir / "metadata_table.csv"
    if not csv_path.is_file():
        return {}
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("data_prep_handoff.model_name") != model:
                continue
            if r.get("ssdr_stage04") in (None, "", "None"):
                continue
            if r.get("ssdr_stage02") != "v1" or r.get("ssdr_stage03") != "v1":
                continue
            if int(float(r.get("ssdr_stage04.kfold") or 0)) != 0:
                continue
            if int(float(r.get("ssdr_stage04.org_type") or 3)) != 3:
                continue
            row_var = r.get("ssdr_stage02.variant") or ""
            if model in ("Griffin_A2", "Winter_Samurai") and row_var != variant:
                continue
            nm_p = pool_dir / r["data_dir"] / "data4" / "normalized_metrics_v1.json"
            if not nm_p.is_file():
                nm_p = pool_dir / r["data_dir"] / "normalized_metrics_v1.json"
            if nm_p.is_file():
                return json.loads(nm_p.read_text(encoding="utf-8"))
    return {}


def discover_diff_render_models(
    pool_dir: Path,
    output_dir: Path,
    *,
    ue_simp_pool: Path = DEFAULT_UE_SIMP_POOL,
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

    ue_simp_by_model = _discover_ue_simp_entries(ue_simp_pool)

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
                "methods": {},
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
        ours_nm = _find_ours_stage04_geom_metrics(pool_dir, model, "bone_reduced_conservative")
        ours_comp = ours_nm.get("complexity") or {}
        ours_geom = ours_nm.get("metrics") or {}
        if "num_loose_parts" in ours_comp:
            ours_stats["loose_parts"] = ours_comp["num_loose_parts"]

        rel_tgt_glb = f"glb/{model}_target.glb"
        rel_ours_glb = f"glb/{model}_ours.glb"
        if export_glb:
            print(f"[diff-render-report] Exporting Target & Ours GLBs for {model} ...", flush=True)
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
        fm = s05_obj.get("metrics") or s05_obj.get("final_metrics") or {}

        def _get_m(key_overall: str, key_legacy: str) -> Any:
            return fm.get(key_overall, fm.get(key_legacy))

        metrics_clean = {
            "stage01_train_psnr": _get_m("stage01_final_train_overall_psnr", "stage01_final_train_psnr"),
            "stage01_test_psnr": _get_m("stage01_final_test_overall_psnr", "stage01_final_test_psnr"),
            "stage01_train_lpips": _get_m("stage01_final_train_overall_lpips", "stage01_final_train_lpips"),
            "stage01_test_lpips": _get_m("stage01_final_test_overall_lpips", "stage01_final_test_lpips"),
            "stage01_train_hausdorff": _get_m("stage01_final_train_overall_hausdorff", "stage01_final_train_hausdorff"),
            "stage01_test_hausdorff": _get_m("stage01_final_test_overall_hausdorff", "stage01_final_test_hausdorff"),
            "stage02_train_psnr": _get_m("stage02_final_train_overall_psnr", "stage02_final_train_psnr"),
            "stage02_test_psnr": _get_m("stage02_final_test_overall_psnr", "stage02_final_test_psnr"),
            "stage02_train_lpips": _get_m("stage02_final_train_overall_lpips", "stage02_final_train_lpips"),
            "stage02_test_lpips": _get_m("stage02_final_test_overall_lpips", "stage02_final_test_lpips"),
            "stage02_train_hausdorff": _get_m("stage02_final_train_overall_hausdorff", "stage02_final_train_hausdorff"),
            "stage02_test_hausdorff": _get_m("stage02_final_test_overall_hausdorff", "stage02_final_test_hausdorff"),
        }

        methods_dict: dict[str, Any] = {
            "target": {
                "key": "target",
                "label": "Target (Ground Truth High-Poly)",
                "short_label": "Target (GT)",
                "pipeline_tag": "Ground Truth",
                "glb": rel_tgt_glb,
                "stats": tgt_stats,
                "geom_metrics": {},
                "render_metrics": {},
            },
            "ours": {
                "key": "ours",
                "label": "★ Ours (VDB + SSDR + Step 05 Diff-Render)",
                "short_label": "Ours (Step 05)",
                "pipeline_tag": "Ours (VDB+SSDR+DiffRender)",
                "glb": rel_ours_glb,
                "stats": ours_stats,
                "geom_metrics": ours_geom,
                "render_metrics": metrics_clean,
            },
        }

        ue_variants_meta = [
            ("ue_qem_cons", "UE-qem-lod (No Merge, Native UV — Conservative)", "UE-qem-lod (Cons)", "UE5 QEM (No Merge)"),
            ("ue_remesh_cons", "UE-remesh-lod (UE5 Remesh + Weight Transfer + Bake — Conservative)", "UE-remesh-lod (Cons)", "UE5 Remesh + Bake"),
            ("ue_qem_mod", "UE-qem-lod (No Merge, Native UV — Moderate Quota)", "UE-qem-lod (Mod)", "UE5 QEM (Moderate)"),
            ("ue_remesh_mod", "UE-remesh-lod (UE5 Remesh + Weight Transfer + Bake — Moderate Quota)", "UE-remesh-lod (Mod)", "UE5 Remesh (Moderate)"),
        ]
        for v_key, v_label, v_short, v_tag in ue_variants_meta:
            d4_dir = ue_simp_by_model.get(model, {}).get(v_key)
            if d4_dir is None or not (d4_dir / "target_format").is_dir():
                continue
            ue_tf_dir = d4_dir / "target_format"
            rel_ue_glb = f"glb/{model}_{v_key}.glb"
            if export_glb:
                print(f"[diff-render-report] Exporting {v_key} GLB for {model} ...", flush=True)
                _export_fitbox_dir_to_glb(
                    ue_tf_dir,
                    output_dir / rel_ue_glb,
                    import_mode="LBS",
                    target_frames=200,
                    max_tex_size=1024,
                    overwrite=overwrite_glb,
                )
            ue_nm_path = d4_dir / "normalized_metrics_v1.json"
            ue_nm = json.loads(ue_nm_path.read_text(encoding="utf-8")) if ue_nm_path.is_file() else {}
            ue_comp = ue_nm.get("complexity") or {}
            ue_geom = ue_nm.get("metrics") or {}
            ue_stats = {
                "mesh_count": ue_comp.get("num_meshes", 0),
                "vertices": ue_comp.get("num_vertices", 0),
                "faces": ue_comp.get("num_triangles", 0),
                "bones": ue_comp.get("num_bones_total", 0),
                "loose_parts": ue_comp.get("num_loose_parts"),
            }
            methods_dict[v_key] = {
                "key": v_key,
                "label": v_label,
                "short_label": v_short,
                "pipeline_tag": v_tag,
                "glb": rel_ue_glb,
                "stats": ue_stats,
                "geom_metrics": ue_geom,
                "render_metrics": {},
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
            "methods": methods_dict,
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

    tpl_path = HERE / "report_templates" / "ogd_diff_render_template.html"
    tpl_text = tpl_path.read_text(encoding="utf-8")
    html_doc = (
        tpl_text
        .replace("__CORE_CSS__", shared_assets["core_css"])
        .replace("__SUITE_NAV_HTML__", nav_html)
        .replace("__DIFF_RENDER_DATA_JSON__", json.dumps(payload, ensure_ascii=False))
    )

    out_html = output_dir / "Diff_Render_Report.html"
    out_html.write_text(html_doc, encoding="utf-8")
    return out_html



def build_diff_render_standalone_zip(
    output_dir: Path,
    models_data: list[dict[str, Any]],
) -> dict[str, Any]:
    dl_dir = output_dir / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dl_dir / "ogd_lbs_diff_render_3d_report_bundle.zip"
    tmp_zip = dl_dir / ".ogd_lbs_diff_render_3d_report_bundle.zip.tmp"
    if tmp_zip.exists():
        tmp_zip.unlink()

    rel_files: list[str] = [
        "Diff_Render_Report.html",
        "diff_render_dashboard_data.json",
    ]

    shared_dir = output_dir / "shared"
    if shared_dir.is_dir():
        for p in sorted(shared_dir.rglob("*")):
            if p.is_file():
                rel_files.append(str(p.relative_to(output_dir)))

    for m in models_data:
        for k_glb in ("target_glb", "ours_glb"):
            rel_g = m.get(k_glb)
            if rel_g and (output_dir / rel_g).is_file():
                rel_files.append(str(rel_g))
        for m_obj in (m.get("methods") or {}).values():
            rel_g = m_obj.get("glb")
            if rel_g and (output_dir / rel_g).is_file():
                rel_files.append(str(rel_g))
        for sub in m.get("submesh_metrics") or []:
            for k_tex in ("rel_raw_texture", "rel_stage1_texture", "rel_body_mask"):
                rel_t = sub.get(k_tex)
                if rel_t and (output_dir / rel_t).is_file():
                    rel_files.append(str(rel_t))

    seen_rel: set[str] = set()
    ordered_rel: list[str] = []
    for rf in rel_files:
        norm_rf = rf.replace("\\", "/")
        if norm_rf not in seen_rel and (output_dir / norm_rf).is_file():
            seen_rel.add(norm_rf)
            ordered_rel.append(norm_rf)

    viewer_py = (
        "#!/usr/bin/env python3\n"
        "import http.server, socketserver, webbrowser\n"
        "PORT = 8000\n"
        "Handler = http.server.SimpleHTTPRequestHandler\n"
        "with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:\n"
        "    url = f'http://127.0.0.1:{PORT}/Diff_Render_Report.html'\n"
        "    print('Serving standalone Diff-Render 3D Report at:', url)\n"
        "    try:\n"
        "        webbrowser.open(url)\n"
        "    except Exception:\n"
        "        pass\n"
        "    httpd.serve_forever()\n"
    )
    viewer_bat = (
        "@echo off\r\n"
        "echo Starting local HTTP server for OGD-LBS Diff-Render 3D Report...\r\n"
        "python start_local_viewer.py\r\n"
        "pause\r\n"
    )

    with zipfile.ZipFile(tmp_zip, "w") as zf:
        for rf in ordered_rel:
            full_p = output_dir / rf
            ext = full_p.suffix.lower()
            comp = zipfile.ZIP_STORED if ext in {".glb", ".png", ".jpg", ".jpeg"} else zipfile.ZIP_DEFLATED
            zf.write(full_p, arcname=rf, compress_type=comp)
        zf.writestr("start_local_viewer.py", viewer_py, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("start_local_viewer.bat", viewer_bat, compress_type=zipfile.ZIP_DEFLATED)

    tmp_zip.chmod(0o644)
    tmp_zip.replace(zip_path)
    return {
        "zip_path": str(zip_path),
        "rel_zip": "downloads/ogd_lbs_diff_render_3d_report_bundle.zip",
        "file_count": len(ordered_rel) + 2,
        "size_mib": round(zip_path.stat().st_size / (1024.0 * 1024.0), 2),
    }


def generate_diff_render_report(
    *,
    pool_dir: Path = DEFAULT_POOL,
    ue_simp_pool: Path = DEFAULT_UE_SIMP_POOL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    export_glb: bool = True,
    overwrite_glb: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_data = discover_diff_render_models(
        pool_dir.expanduser().resolve(strict=True),
        output_dir,
        ue_simp_pool=ue_simp_pool.expanduser().resolve(strict=False),
        export_glb=export_glb,
        overwrite_glb=overwrite_glb,
    )
    out_html = render_diff_render_html(models_data, output_dir)
    zip_info = build_diff_render_standalone_zip(output_dir, models_data)
    ready_cnt = sum(1 for m in models_data if m["status"] == "READY")
    return {
        "status": "GENERATED",
        "html": str(out_html),
        "bundle_zip": zip_info["zip_path"],
        "bundle_size_mib": zip_info["size_mib"],
        "bundle_files": zip_info["file_count"],
        "ready_models": ready_cnt,
        "total_models": len(models_data),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Step 05 Differentiable Rendering & UE5 Triple-Viewport 3D Comparison Report.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--ue-simp-pool", type=Path, default=DEFAULT_UE_SIMP_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ui-only", action="store_true", help="Skip GLB export if GLBs already exist.")
    parser.add_argument("--overwrite-glb", action="store_true", help="Force re-exporting GLBs.")
    args = parser.parse_args()

    res = generate_diff_render_report(
        pool_dir=args.pool,
        ue_simp_pool=args.ue_simp_pool,
        output_dir=args.output_dir,
        export_glb=not args.ui_only,
        overwrite_glb=args.overwrite_glb,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
