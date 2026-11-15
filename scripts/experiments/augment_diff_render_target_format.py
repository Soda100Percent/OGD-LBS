#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Augment and normalize a differentiable rendering stage output directory (`data5`)
into a `target_format/` directory that is 100% schema- and layout-compatible with
`/workspace/ogd_dataset/<model>/target/`.

Preserves the raw baked RGBA texture in `raw_textures/<stem>.png` and the body mask
in `masks/<stem>_body_mask.png` so the Frontend Live Texture Filter Lab can test
and compare multiple hole/rainbow-ring filters in real time without data loss.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

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
from ogd_lbs.utils.common import dump_pkl, load_data  # noqa: E402

TARGET_FORMAT_SUMMARY_SCHEMA = "ogd_lbs_diff_render_target_format_summary_v3"

NON_MESH_STEMS = {
    "skeleton_hierarchy_data",
    "animation_data",
    "skeleton_hierarchy_data_bone_reduced",
    "animation_data_bone_reduced",
    "step00_convert_summary",
    "step01_frame_sampling_summary",
    "step02_reduce_bone_weight_summary",
    "step03_qem_summary",
    "step04_optimize_summary",
    "step04_test_summary",
    "step05_diff_render_summary",
    "target_export_summary",
    "body_bone_alpha",
    "normalized_metrics_v1",
}


def _triangulate_faces_array(raw_faces: Any) -> np.ndarray:
    if isinstance(raw_faces, np.ndarray) and raw_faces.ndim == 2 and raw_faces.shape[1] == 3:
        return raw_faces.astype(np.int64, copy=False)
    rows = raw_faces.tolist() if isinstance(raw_faces, np.ndarray) else list(raw_faces)
    tris: list[list[int]] = []
    for row in rows:
        face = [int(x) for x in (row.tolist() if isinstance(row, np.ndarray) else list(row))]
        if len(face) == 3:
            tris.append(face)
        elif len(face) > 3:
            for j in range(1, len(face) - 1):
                tris.append([face[0], face[j], face[j + 1]])
    return np.asarray(tris, dtype=np.int64)


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


def is_valid_existing_target_format_dir(out_dir: Path) -> bool:
    summary_file = out_dir / "target_export_summary.json"
    if not summary_file.is_file():
        return False
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("schema") != TARGET_FORMAT_SUMMARY_SCHEMA:
            return False
        if not (out_dir / "animation_data.pkl").is_file() or not (out_dir / "skeleton_hierarchy_data.pkl").is_file():
            return False
        meshes = summary.get("meshes") or []
        if not meshes:
            return False
        for m in meshes:
            stem = m["target_stem"]
            if not (out_dir / f"{stem}.pkl").is_file() or not (out_dir / f"{stem}.png").is_file():
                return False
        return True
    except Exception:
        return False


def build_target_format_from_diff_render_stage(
    input_dir: Path,
    output_dir: Path,
    *,
    stage_name: str = "diff_render_stage01",
) -> dict[str, Any]:
    s_dir = input_dir.expanduser().resolve(strict=True)
    opt_dir = (s_dir / "opt_out") if (s_dir / "opt_out").is_dir() else s_dir

    skel_src = next(
        (opt_dir / f"skeleton_hierarchy_data{ext}" for ext in (".pkl", ".pickle", ".json") if (opt_dir / f"skeleton_hierarchy_data{ext}").is_file()),
        None,
    )
    anim_src = next(
        (opt_dir / f"animation_data{ext}" for ext in (".pkl", ".pickle", ".json") if (opt_dir / f"animation_data{ext}").is_file()),
        None,
    )
    if skel_src is None or anim_src is None:
        raise FileNotFoundError(f"Missing skeleton_hierarchy_data or animation_data in {opt_dir}")

    skel_obj = load_data(skel_src)
    anim_obj = load_data(anim_src)
    bone_names = [str(x) for x in skel_obj.get(KEY_BONE_NAMES, anim_obj.get(KEY_BONE_NAMES, []))]
    n_b = len(bone_names)
    rbt = np.asarray(anim_obj[KEY_REST_POSE], dtype=np.float64)
    bt = np.asarray(anim_obj[KEY_ANIMATION_POSES], dtype=np.float64)
    inv_rbt = np.linalg.inv(rbt)
    n_frames = int(bt.shape[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_tex_out_dir = output_dir / "raw_textures"
    masks_out_dir = output_dir / "masks"
    raw_tex_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    dump_pkl(output_dir / "skeleton_hierarchy_data.pkl", skel_obj)
    dump_pkl(output_dir / "animation_data.pkl", anim_obj)

    if (opt_dir / "body_bone_alpha.json").is_file():
        shutil.copy2(opt_dir / "body_bone_alpha.json", output_dir / "body_bone_alpha.json")

    candidates = [
        p
        for p in sorted(list(opt_dir.glob("*.pkl")) + list(opt_dir.glob("*.json")), key=lambda x: x.name)
        if p.stem not in NON_MESH_STEMS and not p.stem.endswith("_summary")
    ]
    seen_stems: set[str] = set()
    mesh_files: list[Path] = []
    for c in candidates:
        if c.stem not in seen_stems:
            seen_stems.add(c.stem)
            mesh_files.append(c)

    if not mesh_files:
        raise RuntimeError(f"No submesh files found in {opt_dir}")

    tex_dir = opt_dir / "textures"
    mask_dir = opt_dir / "masks"

    exported_meshes: list[dict[str, Any]] = []
    total_verts = 0
    total_faces = 0

    for mf in mesh_files:
        raw_stem = mf.stem
        clean_stem = raw_stem[:-len("_merged")] if raw_stem.endswith("_merged") else raw_stem
        m_obj = dict(load_data(mf))

        v_rest = np.asarray(m_obj[KEY_VERTICES], dtype=np.float64)
        f_tri = _triangulate_faces_array(m_obj[KEY_FACES])
        vbw_raw = np.asarray(m_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)
        m_bones = [str(x) for x in m_obj.get(KEY_BONE_NAMES, bone_names)]

        if vbw_raw.shape[1] != n_b or m_bones != bone_names:
            vbw = np.zeros((v_rest.shape[0], n_b), dtype=np.float64)
            for b_idx, b_name in enumerate(bone_names):
                if b_name in m_bones:
                    vbw[:, b_idx] = vbw_raw[:, m_bones.index(b_name)]
            row_sums = vbw.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 1e-12] = 1.0
            vbw = vbw / row_sums
        else:
            vbw = vbw_raw

        baked_v = _reconstruct_lbs_sequence(v_rest, bt, inv_rbt, vbw)
        baked_n = _compute_vertex_normals_sequence(baked_v, f_tri)
        rest_n = _compute_vertex_normals_sequence(v_rest[None, ...], f_tri)[0]

        m_obj["name"] = clean_stem
        m_obj[KEY_BONE_NAMES] = bone_names
        m_obj[KEY_VERTICES] = v_rest.astype(np.float32)
        m_obj[KEY_VERTEX_NORMALS] = rest_n.astype(np.float32)
        m_obj[KEY_VERTEX_BONE_WEIGHTS] = vbw.astype(np.float32)
        m_obj[KEY_FACES] = f_tri.astype(np.int32)
        if KEY_FACE_UVS in m_obj:
            m_obj[KEY_FACE_UVS] = np.asarray(m_obj[KEY_FACE_UVS], dtype=np.float32)
        m_obj[KEY_MATERIAL_NAMES] = [clean_stem]
        m_obj[KEY_FACE_MATERIALS] = np.zeros((f_tri.shape[0],), dtype=np.int32)
        m_obj[KEY_BAKED_VERTICES] = baked_v.astype(np.float32)
        m_obj[KEY_BAKED_NORMALS] = baked_n.astype(np.float32)

        out_pkl_path = output_dir / f"{clean_stem}.pkl"
        dump_pkl(out_pkl_path, m_obj)

        tex_candidates = [
            tex_dir / f"{raw_stem}_texture_kd.png",
            tex_dir / f"{clean_stem}_texture_kd.png",
            opt_dir / f"{raw_stem}_texture_kd.png",
            opt_dir / f"{clean_stem}.png",
            opt_dir / f"{raw_stem}.png",
        ]
        tex_path = next((p for p in tex_candidates if p.is_file()), None)
        if tex_path is None:
            raise FileNotFoundError(f"Missing baked texture for submesh {raw_stem!r} in {tex_dir}")

        with Image.open(tex_path) as im:
            rgba_arr = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()

        # Save raw untouched Stage 02 RGBA texture for frontend live filter debugging
        Image.fromarray(rgba_arr, mode="RGBA").save(raw_tex_out_dir / f"{clean_stem}.png")

        mask_candidates = [
            mask_dir / f"{raw_stem}_body_mask.png",
            mask_dir / f"{clean_stem}_body_mask.png",
            opt_dir / f"{raw_stem}_body_mask.png",
            opt_dir / f"{clean_stem}_body_mask.png",
        ]
        mask_path = next((p for p in mask_candidates if p.is_file()), None)
        body_mask_applied = False
        tex_desc = f"textures/{tex_path.name}"

        if mask_path is not None:
            shutil.copyfile(mask_path, masks_out_dir / f"{clean_stem}_body_mask.png")
            with Image.open(mask_path) as m_im:
                m_gray = m_im.convert("L")
                if m_gray.size != (rgba_arr.shape[1], rgba_arr.shape[0]):
                    m_gray = m_gray.resize((rgba_arr.shape[1], rgba_arr.shape[0]), Image.Resampling.NEAREST)
                mask_u8 = np.asarray(m_gray, dtype=np.float32) / 255.0
            new_alpha = np.clip(np.rint(rgba_arr[..., 3].astype(np.float32) * mask_u8), 0, 255).astype(np.uint8)
            rgba_arr[..., 3] = new_alpha
            body_mask_applied = True
            tex_desc += f"+body_mask(masks/{mask_path.name})"

        pil_out = Image.fromarray(rgba_arr, mode="RGBA")
        pil_out.save(output_dir / f"{clean_stem}.png")
        pil_out.save(output_dir / f"{clean_stem}_mtl1.png")

        n_v = int(v_rest.shape[0])
        n_f = int(f_tri.shape[0])
        total_verts += n_v
        total_faces += n_f

        exported_meshes.append({
            "source_stem": raw_stem,
            "target_stem": clean_stem,
            "vertex_count": n_v,
            "face_count": n_f,
            "frame_count": n_frames,
            "material_names": [clean_stem],
            "body_mask_applied": body_mask_applied,
            "texture_sources": [tex_desc],
        })

    step05_sum_path = s_dir / "step05_diff_render_summary.json"
    step05_metrics = {}
    if step05_sum_path.is_file():
        try:
            s05 = json.loads(step05_sum_path.read_text(encoding="utf-8"))
            fm = s05.get("final_metrics") or {}
            for k in (
                "stage01_final_train_psnr",
                "stage01_final_test_psnr",
                "stage01_final_test_lpips",
                "stage01_final_test_hausdorff",
                "stage02_final_train_psnr",
                "stage02_final_test_psnr",
                "stage02_final_test_lpips",
                "stage02_final_test_hausdorff",
            ):
                if k in fm:
                    step05_metrics[k] = fm[k]
        except Exception:
            pass

    summary = {
        "schema": TARGET_FORMAT_SUMMARY_SCHEMA,
        "stage": stage_name,
        "input_dir": str(s_dir),
        "output_dir": str(output_dir),
        "mesh_count": len(exported_meshes),
        "total_vertices": total_verts,
        "total_faces": total_faces,
        "num_bones": n_b,
        "num_frames": n_frames,
        "diff_render_metrics": step05_metrics,
        "meshes": exported_meshes,
        "generated_at_epoch": time.time(),
    }
    summary_path = output_dir / "target_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Augment diff_render_stage01 output (data5) into a target-compatible directory (target_format)."
    )
    parser.add_argument("-i", "--input_dir", "--stage_dir", dest="input_dir", required=True, help="Stage output directory (data5).")
    parser.add_argument("-o", "--output_dir", default=None, help="Output directory (default: <input_dir>/target_format).")
    parser.add_argument("-s", "--stage", default="diff_render_stage01", help="Stage name.")
    parser.add_argument("--overwrite", action="store_true", help="Re-generate even if target_format already exists.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve(strict=True)
    out_dir = (
        Path(args.output_dir).expanduser().resolve(strict=False)
        if args.output_dir
        else (in_dir / "target_format")
    )

    if not args.overwrite and is_valid_existing_target_format_dir(out_dir):
        print(json.dumps({"status": "ALREADY_AUGMENTED", "output_dir": str(out_dir)}))
        return 0

    t0 = time.monotonic()
    summary = build_target_format_from_diff_render_stage(
        in_dir,
        out_dir,
        stage_name=str(args.stage),
    )
    elapsed = round(time.monotonic() - t0, 3)
    print(
        json.dumps({
            "status": "AUGMENTED",
            "output_dir": str(out_dir),
            "mesh_count": summary["mesh_count"],
            "total_vertices": summary["total_vertices"],
            "total_faces": summary["total_faces"],
            "elapsed_sec": elapsed,
        })
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
