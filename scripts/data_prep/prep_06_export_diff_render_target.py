#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prep 06: Build Differentiable Rendering `target/` directory for a dataset model.

Features:
  - Reads 200-frame Ground Truth geometry, normals, UVs, and material slots from
    `data_prep_05/fitbox_original` (renaming `Source_<VDB>` -> `<VDB>`).
  - Extracts packed textures from `.blend` or resolves external textures from
    `--texture_dir` using suffix path matching (last 3 components -> last 2 components
    -> filename fallback) to avoid breaking on author-specific absolute/relative prefixes.
"""

from __future__ import annotations

import argparse
import io
import colorsys
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

import _env  # noqa: F401
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ogd_lbs.utils.common import (  # noqa: E402
    dump_json,
    dump_pkl,
    load_data,
    log,
    script_argv,
    warn,
)

SOURCE_PREFIX = "Source_"
IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".tga", ".bmp", ".exr", ".webp"}


def _norm_token(s: str) -> str:
    # Normalize case and common Cyrillic/Latin homoglyphs (e.g. Cyrillic 'С' in Сloth3)
    return (
        s.strip()
        .replace("\u0421", "C")
        .replace("\u0441", "c")
        .lower()
    )


def _clean_path_parts(raw_path: str) -> list[str]:
    cleaned = str(raw_path or "").replace("\\", "/").strip()
    while cleaned.startswith("/"):
        cleaned = cleaned[1:]
    parts: list[str] = []
    for p in PurePosixPath(cleaned).parts:
        if not p or p in {".", "..", "/"}:
            continue
        if len(p) == 2 and p[1] == ":":
            # Skip Windows drive letters like C: or F:
            continue
        parts.append(p)
    return parts


def _index_texture_files(texture_dirs: list[Path]) -> list[tuple[Path, tuple[str, ...]]]:
    indexed: list[tuple[Path, tuple[str, ...]]] = []
    seen: set[Path] = set()
    for t_dir in texture_dirs:
        if not t_dir.is_dir():
            continue
        resolved_root = t_dir.resolve(strict=False)
        for p in sorted(resolved_root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            if p in seen:
                continue
            seen.add(p)
            norm_parts = tuple(_norm_token(part) for part in p.parts)
            indexed.append((p, norm_parts))
    return indexed


def _match_external_texture_file(
    raw_filepath: str,
    fallback_name: str,
    indexed_files: list[tuple[Path, tuple[str, ...]]],
) -> tuple[Path | None, str]:
    parts = _clean_path_parts(raw_filepath)
    if not parts and fallback_name:
        parts = _clean_path_parts(fallback_name)
    if not parts:
        return None, "empty_path"

    norm_query = tuple(_norm_token(p) for p in parts)

    # Priority 1: Match last 3 components (dir2 / dir1 / filename), then last 2 (dir1 / filename), then filename
    for k in (3, 2, 1):
        if len(norm_query) < k:
            continue
        suffix = norm_query[-k:]
        matches = [
            f_path
            for f_path, f_parts in indexed_files
            if len(f_parts) >= k and f_parts[-k:] == suffix
        ]
        if matches:
            return matches[0], f"suffix_k{k}:{'/'.join(parts[-k:])}"

    # Priority 2: Stem match (in case extension changed, e.g. .tif -> .png)
    query_stem = _norm_token(Path(parts[-1]).stem)
    for k in (3, 2, 1):
        if len(norm_query) < k:
            continue
        dir_suffix = norm_query[-k:-1]
        for f_path, f_parts in indexed_files:
            if len(f_parts) >= k and f_parts[-k:-1] == dir_suffix:
                if _norm_token(f_path.stem) == query_stem:
                    return f_path, f"stem_k{k}:{f_path.name}"

    return None, f"unmatched:{'/'.join(parts[-3:])}"


def _find_fitbox_original_dir(dataset_root: Path, model_name: str, pool_root: Path) -> Path:
    dp_cand = dataset_root / model_name / "data_prep_work_dir" / "data_prep_05" / "fitbox_original"
    if dp_cand.is_dir() and (dp_cand / "animation_data.pkl").is_file():
        return dp_cand

    results_dir = pool_root / "results"
    if results_dir.is_dir():
        for r_dir in sorted(results_dir.iterdir()):
            cand = r_dir / "data_prep_05" / "fitbox_original"
            if not cand.is_dir():
                continue
            has_griffin = (cand / "Source_main_mesh_VDB.pkl").is_file()
            if model_name == "Griffin_A2" and has_griffin:
                return cand
            if model_name == "Winter_Samurai" and not has_griffin and (cand / "Body2.pkl").is_file():
                return cand

    raise FileNotFoundError(f"Could not locate fitbox_original for model={model_name!r}")


# Legacy Maya/FBX material name -> Blender_Base_Mesh_Rig.blend material name aliases
MATERIAL_NAME_ALIASES: dict[str, str] = {
    "lambert28": "Body_Skin1",
    "lambert29": "Eyes",
    "lambert30": "Helmet_Skin1",
    "lambert31": "Weapon_Skin1",
    "lambert32": "Belt_Skin1",
    "lambert33": "Cloth_Up_Skin1",
    "lambert34": "Armor_Hands_Skin1",
    "lambert36": "Mask_Skin1",
    "lambert37": "Pants_Skin1",
    "lambert38": "Cloth_Skin1",
    "lambert39": "Hair",
}


def _find_model_blend(dataset_root: Path, model_name: str) -> Path:
    model_dir = dataset_root / model_name
    for cand_name in (
        "model/Blender_Base_Mesh_Rig.blend",
        f"{model_name}.blend",
        f"{model_name}_Normalized.blend",
        "model/rest_pose_base.blend",
    ):
        cand = model_dir / cand_name
        if cand.is_file():
            return cand
    blends = sorted(model_dir.glob("*.blend"))
    if blends:
        return blends[0]
    raise FileNotFoundError(f"No .blend file found for model={model_name!r} under {model_dir}")


def _load_image_file_rgba(img_path: Path, max_res: int) -> np.ndarray:
    with Image.open(img_path) as im:
        im_rgba = im.convert("RGBA")
        if max_res > 0 and (im_rgba.width > max_res or im_rgba.height > max_res):
            im_rgba = im_rgba.resize(
                (min(im_rgba.width, max_res), min(im_rgba.height, max_res)),
                Image.Resampling.LANCZOS,
            )
        return np.asarray(im_rgba, dtype=np.uint8)


def _extract_blender_image_rgba(
    bl_img: bpy.types.Image,
    indexed_files: list[tuple[Path, tuple[str, ...]]],
    max_res: int,
) -> tuple[np.ndarray | None, str]:
    # 1. Prefer reading raw packed file bytes via PIL (preserves exact sRGB colorspace and orientation)
    if bl_img.packed_file is not None and getattr(bl_img.packed_file, "data", None):
        try:
            with Image.open(io.BytesIO(bl_img.packed_file.data)) as im:
                im_rgba = im.convert("RGBA")
                if max_res > 0 and (im_rgba.width > max_res or im_rgba.height > max_res):
                    im_rgba = im_rgba.resize(
                        (min(im_rgba.width, max_res), min(im_rgba.height, max_res)),
                        Image.Resampling.LANCZOS,
                    )
                return np.asarray(im_rgba, dtype=np.uint8), f"packed_raw:{bl_img.name}"
        except Exception as exc:
            warn(f"Failed to decode packed_file.data for {bl_img.name} via PIL ({exc}), falling back to bl_img.pixels")

    # 2. Match in external texture_dirs by last 3 / last 2 / last 1 path components
    raw_path = str(getattr(bl_img, "filepath_raw", "") or getattr(bl_img, "filepath", "") or "")
    matched_path, match_desc = _match_external_texture_file(raw_path, bl_img.name, indexed_files)
    if matched_path is not None:
        return _load_image_file_rgba(matched_path, max_res), f"external({match_desc})"

    # 3. Fallback to Blender float pixel buffer (with linear -> sRGB gamma correction)
    w, h = int(bl_img.size[0]), int(bl_img.size[1])
    if w > 0 and h > 0 and len(bl_img.pixels) == w * h * 4:
        arr = np.empty(w * h * 4, dtype=np.float32)
        bl_img.pixels.foreach_get(arr)
        arr = np.flipud(arr.reshape((h, w, 4)))
        rgb = np.clip(arr[..., :3], 0.0, 1.0)
        srgb = np.where(rgb <= 0.0031308, rgb * 12.92, (np.maximum(rgb, 0.0031308) ** (1.0 / 2.4)) * 1.055 - 0.055)
        arr[..., :3] = np.clip(srgb, 0.0, 1.0)
        arr_u8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        if max_res > 0 and (w > max_res or h > max_res):
            pil_img = Image.fromarray(arr_u8, mode="RGBA")
            pil_img = pil_img.resize((min(w, max_res), min(h, max_res)), Image.Resampling.LANCZOS)
            arr_u8 = np.asarray(pil_img, dtype=np.uint8)
        return arr_u8, f"pixels_srgb:{bl_img.name}"

    # 2. Match in texture_dirs by last 3 / last 2 / last 1 path components
    raw_path = str(getattr(bl_img, "filepath_raw", "") or getattr(bl_img, "filepath", "") or "")
    matched_path, match_desc = _match_external_texture_file(raw_path, bl_img.name, indexed_files)
    if matched_path is not None:
        return _load_image_file_rgba(matched_path, max_res), f"external({match_desc})"

    return None, match_desc


def _extract_material_rgba_map(
    blend_path: Path,
    texture_dirs: list[Path],
    max_res: int = 1024,
    fallback_res: int = 512,
) -> dict[str, tuple[np.ndarray, str]]:
    bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
    indexed_files = _index_texture_files(texture_dirs)
    log(f"Indexed {len(indexed_files)} external texture file(s) from {[str(d) for d in texture_dirs if d.is_dir()]}")

    mats = [m for m in sorted(bpy.data.materials, key=lambda x: x.name)]
    result: dict[str, tuple[np.ndarray, str]] = {}

    for idx, mat in enumerate(mats):
        base_img = None
        alpha_img = None
        bsdf_color = None

        if mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    for out in node.outputs:
                        for lk in out.links:
                            to_node_type = getattr(lk.to_node, "type", "")
                            sock_name = lk.to_socket.name
                            # Ignore links going into Normal Map or Bump nodes
                            if to_node_type in {"NORMAL_MAP", "BUMP"}:
                                continue
                            if sock_name == "Base Color" or (
                                base_img is None
                                and sock_name in {"Color", "Emission Color"}
                                and "normal" not in node.image.name.lower()
                                and "roughness" not in node.image.name.lower()
                                and "metallic" not in node.image.name.lower()
                            ):
                                base_img = node.image
                            if sock_name == "Alpha" or "opacity" in node.image.name.lower():
                                alpha_img = node.image
                elif node.type == "BSDF_PRINCIPLED":
                    bc = node.inputs.get("Base Color")
                    if bc is not None:
                        bsdf_color = tuple(float(x) for x in bc.default_value[:4])

        rgba_arr, base_desc = (
            _extract_blender_image_rgba(base_img, indexed_files, max_res)
            if base_img is not None
            else (None, "no_base_img")
        )
        alpha_arr, alpha_desc = (
            _extract_blender_image_rgba(alpha_img, indexed_files, max_res)
            if alpha_img is not None
            else (None, "no_alpha_img")
        )

        if rgba_arr is not None:
            source_kind = base_desc
            if alpha_arr is not None and alpha_img != base_img:
                if alpha_arr.shape[:2] != rgba_arr.shape[:2]:
                    a_pil = Image.fromarray(alpha_arr).resize(
                        (rgba_arr.shape[1], rgba_arr.shape[0]), Image.Resampling.BILINEAR
                    )
                    alpha_arr = np.asarray(a_pil, dtype=np.uint8)
                a_chan = alpha_arr[..., 0] if np.all(alpha_arr[..., 3] == 255) else alpha_arr[..., 3]
                rgba_arr = rgba_arr.copy()
                rgba_arr[..., 3] = a_chan
                source_kind += f"+alpha({alpha_desc})"
            result[mat.name] = (rgba_arr, source_kind)
        else:
            if base_img is not None:
                warn(f"Material {mat.name!r} texture not found ({base_desc}); using procedural fallback.")
            if bsdf_color is not None and bsdf_color[:3] != (1.0, 1.0, 1.0):
                rgb = bsdf_color[:3]
            else:
                hue = (idx * 0.61803398875) % 1.0
                rgb = colorsys.hsv_to_rgb(hue, 0.35, 0.78)
            u8 = [int(np.clip(round(c * 255.0), 0, 255)) for c in rgb] + [255]
            arr = np.full((fallback_res, fallback_res, 4), u8, dtype=np.uint8)
            result[mat.name] = (arr, f"procedural_fallback({base_desc})")

    for alias_k, target_mat_name in MATERIAL_NAME_ALIASES.items():
        if alias_k not in result and target_mat_name in result:
            rgba_arr, src_desc = result[target_mat_name]
            result[alias_k] = (rgba_arr, f"alias({target_mat_name})->{src_desc}")
    return result


def build_model_target(
    *,
    dataset_root: Path,
    model_name: str,
    pool_root: Path,
    extra_texture_dirs: list[Path] | None = None,
    max_tex_res: int = 1024,
    overwrite: bool = True,
) -> dict[str, Any]:
    model_dir = dataset_root / model_name
    target_dir = model_dir / "target"
    if target_dir.exists() and not overwrite:
        log(f"Target dir already exists for {model_name}: {target_dir} (skipping)")
        return {"model": model_name, "status": "SKIPPED", "target_dir": str(target_dir)}

    fb_orig = _find_fitbox_original_dir(dataset_root, model_name, pool_root)
    blend_path = _find_model_blend(dataset_root, model_name)

    candidate_tex_dirs: list[Path] = []
    for d in (extra_texture_dirs or []):
        if d not in candidate_tex_dirs:
            candidate_tex_dirs.append(d)
    for default_d in (
        model_dir / "model" / "Textures",
        model_dir / "Textures",
        model_dir / "model",
    ):
        if default_d not in candidate_tex_dirs:
            candidate_tex_dirs.append(default_d)

    log(f"[{model_name}] Building target/ from fitbox_original={fb_orig} and blend={blend_path}")
    mat_rgba_map = _extract_material_rgba_map(
        blend_path,
        texture_dirs=candidate_tex_dirs,
        max_res=max_tex_res,
    )

    staging_dir = model_dir / f".target_staging_{model_name}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    for meta_stem in ("animation_data", "skeleton_hierarchy_data"):
        src_p = fb_orig / f"{meta_stem}.pkl"
        if not src_p.is_file():
            raise FileNotFoundError(f"Missing {src_p}")
        shutil.copy2(src_p, staging_dir / src_p.name)

    mesh_summaries: list[dict[str, Any]] = []
    for pkl_path in sorted(fb_orig.glob("*.pkl")):
        if pkl_path.stem in {"animation_data", "skeleton_hierarchy_data"}:
            continue

        obj = load_data(pkl_path)
        raw_stem = pkl_path.stem
        target_stem = raw_stem[len(SOURCE_PREFIX) :] if raw_stem.startswith(SOURCE_PREFIX) else raw_stem
        obj["name"] = target_stem

        mat_names = [str(m) if m is not None else "default" for m in (obj.get("material_names") or ["default"])]
        obj["material_names"] = mat_names
        faces = np.asarray(obj["faces"], dtype=np.int64)
        face_mats = np.asarray(obj.get("face_materials", np.zeros(len(faces), dtype=np.int64)), dtype=np.int64)
        face_mats = np.clip(face_mats, 0, max(0, len(mat_names) - 1))
        obj["face_materials"] = face_mats

        out_pkl = staging_dir / f"{target_stem}.pkl"
        dump_pkl(out_pkl, obj)

        saved_tex_files: list[str] = []
        tex_sources: list[str] = []
        for m_idx, m_name in enumerate(mat_names):
            if m_name in mat_rgba_map:
                rgba, src_kind = mat_rgba_map[m_name]
            else:
                rgba = np.full((512, 512, 4), [180, 180, 190, 255], dtype=np.uint8)
                src_kind = "default_gray"

            mtl_png_name = f"{target_stem}_mtl{m_idx + 1}.png"
            Image.fromarray(rgba, mode="RGBA").save(staging_dir / mtl_png_name)
            saved_tex_files.append(mtl_png_name)
            tex_sources.append(f"{m_name}->{src_kind}")

            if len(mat_names) == 1:
                single_png_name = f"{target_stem}.png"
                shutil.copy2(staging_dir / mtl_png_name, staging_dir / single_png_name)
                saved_tex_files.append(single_png_name)

        mesh_summaries.append({
            "source_pkl": pkl_path.name,
            "target_pkl": out_pkl.name,
            "target_stem": target_stem,
            "vertices": int(np.asarray(obj["vertices"]).shape[0]),
            "faces": int(faces.shape[0]),
            "frames": int(np.asarray(obj["baked_animation_vertices"]).shape[0]),
            "materials": mat_names,
            "texture_files": saved_tex_files,
            "texture_sources": tex_sources,
        })
        log(
            f"  Exported target mesh: {target_stem:<14} | V={mesh_summaries[-1]['vertices']:<6} "
            f"F={mesh_summaries[-1]['faces']:<6} | tex={tex_sources}"
        )

    summary = {
        "schema": "ogd_lbs_diff_render_target_export_v1",
        "model_name": model_name,
        "fitbox_original_dir": str(fb_orig),
        "blend_path": str(blend_path),
        "texture_dirs": [str(d) for d in candidate_tex_dirs if d.is_dir()],
        "target_dir": str(target_dir),
        "mesh_count": len(mesh_summaries),
        "meshes": mesh_summaries,
    }
    dump_json(staging_dir / "target_export_summary.json", summary)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    staging_dir.rename(target_dir)
    log(f"[{model_name}] Target directory ready: {target_dir} ({len(mesh_summaries)} meshes)")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build diff_render target/ directories from fitbox_original and .blend.")
    parser.add_argument("--dataset_root", default="/workspace/ogd_dataset")
    parser.add_argument("--pool_root", default="/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1")
    parser.add_argument("--models", nargs="+", default=["Griffin_A2", "Winter_Samurai"])
    parser.add_argument(
        "--texture_dir",
        dest="texture_dirs",
        action="append",
        default=[],
        help="Explicit directory to search for textures using suffix path matching (3 -> 2 -> 1 components).",
    )
    parser.add_argument("--max_tex_res", type=int, default=1024)
    parser.add_argument("--no_overwrite", action="store_true")
    args = parser.parse_args(script_argv())

    ds_root = Path(args.dataset_root).expanduser().resolve(strict=True)
    pl_root = Path(args.pool_root).expanduser().resolve(strict=False)
    tex_dirs = [Path(p).expanduser().resolve(strict=False) for p in (args.texture_dirs or [])]

    for m in args.models:
        build_model_target(
            dataset_root=ds_root,
            model_name=m,
            pool_root=pl_root,
            extra_texture_dirs=tex_dirs,
            max_tex_res=args.max_tex_res,
            overwrite=not args.no_overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
