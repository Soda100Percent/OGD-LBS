#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
UE5-Simp Step 04: Bake/Bind textures, reconstruct 200-frame LBS poses, generate
`target_format/`, and compute normalized geometry & rendering metrics (`data3 -> data4`).

For meshes processed with `ue_remesh_lod` (which have UE5 AutoUVs):
    Bakes the multi-material high-poly textures from `data3/high_poly_sources/<mesh>.pkl`
    onto the UE5 AutoUV layout (`1024x1024` RGBA PNG) using C-accelerated `BVHTree`
    nearest-surface barycentric UV transfer + gutter dilation.

For meshes processed with `ue_qem_lod` or `ue_qem_unmerged` (which preserve native UVs & material slots):
    Directly binds the original `.png` / `_mtl*.png` textures into `data4/target_format/`.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _env import ensure_repo_paths_in_syspath  # noqa: E402
ensure_repo_paths_in_syspath(__file__)

from _stage_io import prepare_output_dir, write_stage_summary  # noqa: E402
from paths import (  # noqa: E402
    print_ssdr_resolution_info,
    resolve_ssdr_stage_dirs,
    validate_ssdr_stage_resolution,
)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

from ogd_lbs.fitbox.schema import (  # noqa: E402
    KEY_ANIMATION_POSES,
    KEY_BAKED_NORMALS,
    KEY_BAKED_VERTICES,
    KEY_BONE_NAMES,
    KEY_FACE_MATERIALS,
    KEY_FACE_UVS,
    KEY_FACES,
    KEY_MATERIAL_NAMES,
    KEY_REST_POSE,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTEX_NORMALS,
    KEY_VERTICES,
)
from ogd_lbs.utils.common import (  # noqa: E402
    Timer,
    dump_metrics,
    dump_pkl,
    fail,
    load_data,
    log,
    record_metric,
    script_argv,
)

FULL_SKELETON = "skeleton_hierarchy_data.pkl"
FULL_ANIMATION = "animation_data.pkl"
TARGET_FORMAT_SUMMARY_SCHEMA = "ogd_lbs_diff_render_target_format_summary_v3"


def _reconstruct_lbs_sequence(
    v: np.ndarray,
    bt: np.ndarray,
    inv_rbt: np.ndarray,
    vbw: np.ndarray,
) -> np.ndarray:
    n_p = int(bt.shape[0])
    n_b = int(bt.shape[1])
    n_v = int(v.shape[0])
    rel_mats = np.einsum("pbij,bjk->pbik", bt, inv_rbt)
    rel_3x4 = np.ascontiguousarray(rel_mats[:, :, :3, :].reshape(n_p, n_b, 12))
    vbw_c = np.ascontiguousarray(vbw, dtype=np.float64)
    v_hom = np.hstack([v, np.ones((n_v, 1), dtype=np.float64)])
    p_r = np.empty((n_p, n_v, 3), dtype=np.float64)
    for p_id in range(n_p):
        blended_3x4 = (vbw_c @ rel_3x4[p_id]).reshape(n_v, 3, 4)
        p_r[p_id] = np.einsum("vij,vj->vi", blended_3x4, v_hom)
    return p_r


def _compute_vertex_normals_sequence(
    verts_seq: np.ndarray,
    tri_faces: np.ndarray,
) -> np.ndarray:
    n_p, n_v, _ = verts_seq.shape
    v_nrms = np.zeros((n_p, n_v, 3), dtype=np.float64)
    if tri_faces.size == 0 or n_v == 0:
        v_nrms[..., 2] = 1.0
        return v_nrms

    f0 = tri_faces[:, 0]
    f1 = tri_faces[:, 1]
    f2 = tri_faces[:, 2]
    v0 = verts_seq[:, f0, :]
    v1 = verts_seq[:, f1, :]
    v2 = verts_seq[:, f2, :]
    fn = np.cross(v1 - v0, v2 - v0)

    for p_id in range(n_p):
        np.add.at(v_nrms[p_id], f0, fn[p_id])
        np.add.at(v_nrms[p_id], f1, fn[p_id])
        np.add.at(v_nrms[p_id], f2, fn[p_id])

    lengths = np.linalg.norm(v_nrms, axis=-1, keepdims=True)
    zero_mask = lengths <= 1e-12
    lengths = np.where(zero_mask, 1.0, lengths)
    v_nrms = v_nrms / lengths
    v_nrms[zero_mask[..., 0]] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v_nrms


def _load_material_textures_for_mesh(
    mesh_stem: str,
    num_materials: int,
    textures_dir: Path,
) -> list[np.ndarray]:
    clean_stem = mesh_stem[:-len("_merged")] if mesh_stem.endswith("_merged") else mesh_stem
    imgs: list[np.ndarray] = []

    for m_i in range(max(1, num_materials)):
        cands = [
            textures_dir / f"{clean_stem}_mtl{m_i + 1}.png",
            textures_dir / f"{mesh_stem}_mtl{m_i + 1}.png",
            textures_dir / f"{clean_stem}.png",
            textures_dir / f"{mesh_stem}.png",
        ]
        found = next((p for p in cands if p.is_file()), None)
        if found is not None:
            with Image.open(found) as im:
                imgs.append(np.asarray(im.convert("RGBA"), dtype=np.uint8).copy())
        else:
            imgs.append(np.full((256, 256, 4), 220, dtype=np.uint8))
    return imgs


def _sample_rgba_bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    u_wrapped = np.mod(u, 1.0)
    v_wrapped = np.mod(1.0 - v, 1.0)
    x = u_wrapped * (w - 1)
    y = v_wrapped * (h - 1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wx = (x - x0)[:, np.newaxis]
    wy = (y - y0)[:, np.newaxis]

    c00 = img[y0, x0].astype(np.float64)
    c10 = img[y0, x1].astype(np.float64)
    c01 = img[y1, x0].astype(np.float64)
    c11 = img[y1, x1].astype(np.float64)

    c = (
        (1.0 - wx) * (1.0 - wy) * c00
        + wx * (1.0 - wy) * c10
        + (1.0 - wx) * wy * c01
        + wx * wy * c11
    )
    return np.clip(np.rint(c), 0, 255).astype(np.uint8)


def bake_texture_from_high_poly_bvh(
    *,
    low_verts: np.ndarray,
    low_faces: np.ndarray,
    low_face_uvs: np.ndarray,
    high_mesh_obj: dict[str, Any],
    mesh_stem: str,
    textures_dir: Path,
    resolution: int = 1024,
) -> np.ndarray:
    """
    Rasterize low-poly UV triangles onto a (resolution, resolution) texture grid,
    find the closest surface point on the high-poly source mesh via C-accelerated
    `mathutils.bvhtree.BVHTree`, interpolate high-poly UVs via barycentric coordinates,
    sample the corresponding high-poly material texture, and apply EDT gutter padding.
    """
    import bpy  # noqa: F401 - required to register mathutils in sys.modules
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    high_verts = np.asarray(high_mesh_obj[KEY_VERTICES], dtype=np.float64)
    high_faces = np.asarray(high_mesh_obj[KEY_FACES], dtype=np.int64)
    high_uvs = np.asarray(high_mesh_obj.get(KEY_FACE_UVS), dtype=np.float64)
    high_mats = np.asarray(
        high_mesh_obj.get(KEY_FACE_MATERIALS, np.zeros((high_faces.shape[0],), dtype=np.int64)),
        dtype=np.int64,
    )
    mat_names = high_mesh_obj.get(KEY_MATERIAL_NAMES) or [mesh_stem]
    mtl_imgs = _load_material_textures_for_mesh(mesh_stem, len(mat_names), textures_dir)

    bvh = BVHTree.FromPolygons(high_verts.tolist(), high_faces.tolist(), all_triangles=True)

    res = int(resolution)
    covered_mask = np.zeros((res, res), dtype=bool)
    texel_pos3d = np.zeros((res, res, 3), dtype=np.float64)

    # Rasterize UV triangles onto (res, res) image space (row y = (1 - v) * res, col x = u * res)
    n_tris = int(low_faces.shape[0])
    for t_i in range(n_tris):
        uv = low_face_uvs[t_i]  # (3, 2)
        px = uv[:, 0] * (res - 1)
        py = (1.0 - uv[:, 1]) * (res - 1)

        min_x = max(0, int(np.floor(np.min(px))))
        max_x = min(res - 1, int(np.ceil(np.max(px))))
        min_y = max(0, int(np.floor(np.min(py))))
        max_y = min(res - 1, int(np.ceil(np.max(py))))
        if min_x > max_x or min_y > max_y:
            continue

        x0, y0 = px[0], py[0]
        x1, y1 = px[1], py[1]
        x2, y2 = px[2], py[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-10:
            continue

        gx, gy = np.meshgrid(
            np.arange(min_x, max_x + 1, dtype=np.float64),
            np.arange(min_y, max_y + 1, dtype=np.float64),
        )
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            continue

        f = low_faces[t_i]
        v0, v1, v2 = low_verts[f[0]], low_verts[f[1]], low_verts[f[2]]
        pts = (
            w0[inside, np.newaxis] * v0
            + w1[inside, np.newaxis] * v1
            + w2[inside, np.newaxis] * v2
        )
        sub_mask = covered_mask[min_y : max_y + 1, min_x : max_x + 1]
        sub_pos = texel_pos3d[min_y : max_y + 1, min_x : max_x + 1]
        sub_mask[inside] = True
        sub_pos[inside] = pts

    ys, xs = np.nonzero(covered_mask)
    if len(ys) == 0:
        return mtl_imgs[0]

    query_pts = texel_pos3d[ys, xs]
    n_q = int(query_pts.shape[0])
    hit_locs = np.empty((n_q, 3), dtype=np.float64)
    hit_tris = np.zeros((n_q,), dtype=np.int64)

    for i in range(n_q):
        p = query_pts[i]
        loc, _, tri_idx, _ = bvh.find_nearest(Vector((float(p[0]), float(p[1]), float(p[2]))))
        if loc is not None and tri_idx is not None:
            hit_locs[i] = (loc.x, loc.y, loc.z)
            hit_tris[i] = int(tri_idx)
        else:
            hit_locs[i] = p
            hit_tris[i] = 0

    # Vectorized 3D barycentric coordinates on high-poly triangles
    tri_v = high_faces[hit_tris]
    a = high_verts[tri_v[:, 0]]
    b = high_verts[tri_v[:, 1]]
    c = high_verts[tri_v[:, 2]]

    v0 = b - a
    v1 = c - a
    v2 = hit_locs - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-14, 1.0, denom)

    bw1 = np.clip((d11 * d20 - d01 * d21) / denom, 0.0, 1.0)
    bw2 = np.clip((d00 * d21 - d01 * d20) / denom, 0.0, 1.0)
    bw0 = np.clip(1.0 - bw1 - bw2, 0.0, 1.0)
    bw_sum = bw0 + bw1 + bw2
    bw_sum = np.where(bw_sum <= 1e-12, 1.0, bw_sum)
    bw0, bw1, bw2 = bw0 / bw_sum, bw1 / bw_sum, bw2 / bw_sum

    h_uv = high_uvs[hit_tris]  # (N_q, 3, 2)
    u_interp = bw0 * h_uv[:, 0, 0] + bw1 * h_uv[:, 1, 0] + bw2 * h_uv[:, 2, 0]
    v_interp = bw0 * h_uv[:, 0, 1] + bw1 * h_uv[:, 1, 1] + bw2 * h_uv[:, 2, 1]

    mat_ids = np.clip(high_mats[hit_tris], 0, len(mtl_imgs) - 1)
    sampled_rgba = np.zeros((n_q, 4), dtype=np.uint8)
    for m_i, m_img in enumerate(mtl_imgs):
        sel = mat_ids == m_i
        if np.any(sel):
            sampled_rgba[sel] = _sample_rgba_bilinear(m_img, u_interp[sel], v_interp[sel])

    out_img = np.zeros((res, res, 4), dtype=np.uint8)
    out_img[ys, xs] = sampled_rgba

    # Gutter padding via nearest covered texel (up to 8px) to prevent UV seam bleeding
    dist, nearest_indices = distance_transform_edt(~covered_mask, return_indices=True)
    pad_mask = (~covered_mask) & (dist <= 8.0)
    out_img[pad_mask] = out_img[nearest_indices[0][pad_mask], nearest_indices[1][pad_mask]]

    return out_img


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UE5-Simp Step 04: bake/bind textures, reconstruct LBS sequence, and output target_format + metrics (data3 -> data4)."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    parser.add_argument("-m", "--metrics_dir", type=str, default=None)

    parser.add_argument("--bake_resolution", type=int, default=1024)
    parser.add_argument("--compute_metrics", action="store_true", default=True)
    parser.add_argument("--no_compute_metrics", dest="compute_metrics", action="store_false")

    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--allow_merge", action="store_true")
    parser.add_argument("--no_summary", action="store_true")
    parser.add_argument("--enable_profiling", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(script_argv())

    resolution = resolve_ssdr_stage_dirs(
        step_id="04",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="UE5-Simp Step 04: data3 -> data4 Texture Bake/Bind & Evaluation",
    )
    print_ssdr_resolution_info(resolution)
    validate_ssdr_stage_resolution(resolution)

    input_tree = resolution.input_tree
    output_tree = resolution.output_tree
    target_format_dir = output_tree / "target_format"
    raw_tex_out_dir = target_format_dir / "raw_textures"
    masks_out_dir = target_format_dir / "masks"

    prepare_output_dir(
        output_tree,
        overwrite_output=bool(args.overwrite_output),
        allow_merge=bool(args.allow_merge),
    )
    target_format_dir.mkdir(parents=True, exist_ok=True)
    raw_tex_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    skel_obj = load_data(input_tree / FULL_SKELETON)
    anim_obj = load_data(input_tree / FULL_ANIMATION)
    bone_names = [str(x) for x in skel_obj.get(KEY_BONE_NAMES, anim_obj.get(KEY_BONE_NAMES, []))]
    n_b = len(bone_names)
    rbt = np.asarray(anim_obj[KEY_REST_POSE], dtype=np.float64)
    bt = np.asarray(anim_obj[KEY_ANIMATION_POSES], dtype=np.float64)
    inv_rbt = np.linalg.inv(rbt)
    n_frames = int(bt.shape[0])

    dump_pkl(output_tree / FULL_SKELETON, skel_obj)
    dump_pkl(output_tree / FULL_ANIMATION, anim_obj)
    dump_pkl(target_format_dir / FULL_SKELETON, skel_obj)
    dump_pkl(target_format_dir / FULL_ANIMATION, anim_obj)

    textures_dir = input_tree / "textures"
    high_poly_dir = input_tree / "high_poly_sources"

    non_mesh = {FULL_SKELETON, FULL_ANIMATION}
    mesh_paths = [
        p
        for p in sorted(input_tree.glob("*.pkl"))
        if p.name not in non_mesh and not p.stem.endswith("_summary")
    ]
    if not mesh_paths:
        fail(f"No mesh PKL files found in {input_tree}")

    exported_meshes: list[dict[str, Any]] = []
    total_verts = 0
    total_faces = 0

    with Timer("time_04_bake_and_eval"):
        for mp in mesh_paths:
            raw_stem = mp.stem
            clean_stem = raw_stem[:-len("_merged")] if raw_stem.endswith("_merged") else raw_stem
            m_obj = dict(load_data(mp))

            v_rest = np.asarray(m_obj[KEY_VERTICES], dtype=np.float64)
            f_tri = np.asarray(m_obj[KEY_FACES], dtype=np.int64)
            vbw = np.asarray(m_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)
            mode_tag = str(m_obj.get("ue_simp_mode", "ue_qem_lod"))

            # Save mesh with GT baked_animation_vertices in data4 root for normalized_metrics_v1.json evaluation
            dump_pkl(output_tree / f"{raw_stem}.pkl", m_obj)

            # Reconstruct actual LBS posed vertices & normals for target_format (GLB & visual comparison)
            lbs_v = _reconstruct_lbs_sequence(v_rest, bt, inv_rbt, vbw)
            lbs_n = _compute_vertex_normals_sequence(lbs_v, f_tri)
            rest_n = _compute_vertex_normals_sequence(v_rest[None, ...], f_tri)[0]

            tf_obj = dict(m_obj)
            tf_obj["name"] = clean_stem
            tf_obj[KEY_BONE_NAMES] = bone_names
            tf_obj[KEY_VERTICES] = v_rest.astype(np.float32)
            tf_obj[KEY_VERTEX_NORMALS] = rest_n.astype(np.float32)
            tf_obj[KEY_VERTEX_BONE_WEIGHTS] = vbw.astype(np.float32)
            tf_obj[KEY_FACES] = f_tri.astype(np.int32)
            if KEY_FACE_UVS in tf_obj and tf_obj[KEY_FACE_UVS] is not None:
                tf_obj[KEY_FACE_UVS] = np.asarray(tf_obj[KEY_FACE_UVS], dtype=np.float32)
            tf_obj[KEY_BAKED_VERTICES] = lbs_v.astype(np.float32)
            tf_obj[KEY_BAKED_NORMALS] = lbs_n.astype(np.float32)

            tex_sources: list[str] = []
            high_src_pkl = high_poly_dir / f"{raw_stem}.pkl"

            if mode_tag == "ue_remesh_lod" and high_src_pkl.is_file():
                high_obj = load_data(high_src_pkl)
                baked_rgba = bake_texture_from_high_poly_bvh(
                    low_verts=v_rest,
                    low_faces=f_tri,
                    low_face_uvs=np.asarray(tf_obj[KEY_FACE_UVS], dtype=np.float64),
                    high_mesh_obj=high_obj,
                    mesh_stem=raw_stem,
                    textures_dir=textures_dir,
                    resolution=int(args.bake_resolution),
                )
                pil_baked = Image.fromarray(baked_rgba, mode="RGBA")
                pil_baked.save(target_format_dir / f"{clean_stem}.png")
                pil_baked.save(target_format_dir / f"{clean_stem}_mtl1.png")
                pil_baked.save(raw_tex_out_dir / f"{clean_stem}.png")
                tf_obj[KEY_MATERIAL_NAMES] = [clean_stem]
                tf_obj[KEY_FACE_MATERIALS] = np.zeros((f_tri.shape[0],), dtype=np.int32)
                tex_sources.append(f"bvh_baked({clean_stem}.png)")
            else:
                mat_names = [str(x) for x in m_obj.get(KEY_MATERIAL_NAMES, [clean_stem])]
                tf_obj[KEY_MATERIAL_NAMES] = mat_names
                tf_obj[KEY_FACE_MATERIALS] = np.asarray(
                    m_obj.get(KEY_FACE_MATERIALS, np.zeros((f_tri.shape[0],), dtype=np.int32)),
                    dtype=np.int32,
                )
                mtl_imgs = _load_material_textures_for_mesh(raw_stem, len(mat_names), textures_dir)
                for m_i, m_img in enumerate(mtl_imgs):
                    pil_m = Image.fromarray(m_img, mode="RGBA")
                    if m_i == 0:
                        pil_m.save(target_format_dir / f"{clean_stem}.png")
                        pil_m.save(raw_tex_out_dir / f"{clean_stem}.png")
                    pil_m.save(target_format_dir / f"{clean_stem}_mtl{m_i + 1}.png")
                    tex_sources.append(f"{clean_stem}_mtl{m_i + 1}.png")

            dump_pkl(target_format_dir / f"{clean_stem}.pkl", tf_obj)

            n_v = int(v_rest.shape[0])
            n_f = int(f_tri.shape[0])
            total_verts += n_v
            total_faces += n_f

            exported_meshes.append({
                "source_stem": raw_stem,
                "target_stem": clean_stem,
                "ue_simp_mode": mode_tag,
                "vertex_count": n_v,
                "face_count": n_f,
                "frame_count": n_frames,
                "material_names": tf_obj[KEY_MATERIAL_NAMES],
                "body_mask_applied": False,
                "texture_sources": tex_sources,
            })

        # Compute standardized geometry & LBS error metrics (normalized_metrics_v1.json)
        norm_metrics = None
        if bool(args.compute_metrics):
            import importlib.util
            aug_path = Path(__file__).resolve().parents[1] / "experiments" / "augment_stage_metrics.py"
            spec_aug = importlib.util.spec_from_file_location("_aug_metrics", aug_path)
            if spec_aug is not None and spec_aug.loader is not None:
                mod_aug = importlib.util.module_from_spec(spec_aug)
                sys.modules["_aug_metrics"] = mod_aug
                spec_aug.loader.exec_module(mod_aug)
                norm_metrics = mod_aug.compute_normalized_stage_metrics(
                    output_tree,
                    stage_name="ssdr_stage04",
                )
                (output_tree / "normalized_metrics_v1.json").write_text(
                    json.dumps(norm_metrics, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    tf_summary = {
        "schema": TARGET_FORMAT_SUMMARY_SCHEMA,
        "stage": "ue_simp_stage04",
        "input_dir": str(input_tree),
        "output_dir": str(target_format_dir),
        "mesh_count": len(exported_meshes),
        "total_vertices": total_verts,
        "total_faces": total_faces,
        "num_bones": n_b,
        "num_frames": n_frames,
        "diff_render_metrics": (norm_metrics.get("metrics") if isinstance(norm_metrics, dict) else {}),
        "meshes": exported_meshes,
        "generated_at_epoch": time.time(),
    }
    (target_format_dir / "target_export_summary.json").write_text(
        json.dumps(tf_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if not bool(args.no_summary):
        write_stage_summary(output_tree / "step04_bake_and_evaluate_summary.json", tf_summary)

    record_metric("04_total_vertices", total_verts)
    record_metric("04_total_faces", total_faces)
    record_metric("04_num_bones", n_b)
    if args.metrics_dir:
        dump_metrics(Path(args.metrics_dir) / "metrics_04.json")

    log(
        f"UE5-Simp Step 04 complete: meshes={len(exported_meshes)}, "
        f"V={total_verts}, F={total_faces}, B={n_b}, target_format={target_format_dir}"
    )


if __name__ == "__main__":
    main()
