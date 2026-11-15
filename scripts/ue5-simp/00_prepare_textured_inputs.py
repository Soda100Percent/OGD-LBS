#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
UE5-Simp Step 00: Prepare textured unmerged Fitbox inputs (data0_json -> data0).

Reads unmerged textured meshes & textures from `/workspace/ogd_dataset/<model_name>/target`
(or from `input_tree` if already present), attaches bone reduction variants from
`input_tree/bone_reduced_versions` (or legacy `*_bone_reduced` files in `/workspace/ogd_dataset/<model_name>/data0`),
converts all JSON/PKL objects to normalized `.pkl` files in `data0`, and copies all `.png`
textures into `data0/textures/`.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _env import ensure_repo_paths_in_syspath  # noqa: E402
ensure_repo_paths_in_syspath(__file__)

from _stage_constants import KNOWN_NON_MESH_STEMS  # noqa: E402
from _stage_io import discover_stage_files, prepare_output_dir, write_stage_summary  # noqa: E402
from paths import (  # noqa: E402
    print_ssdr_resolution_info,
    resolve_ssdr_stage_dirs,
    validate_ssdr_stage_resolution,
)

import numpy as np  # noqa: E402
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


def _normalize_mesh_dict(obj: dict[str, Any], stem: str) -> dict[str, Any]:
    out = dict(obj)
    out.setdefault("name", stem)
    for k, dtype in [
        ("vertices", np.float64),
        ("vertex_normals", np.float64),
        ("vertex_bone_weights", np.float64),
        ("baked_animation_vertices", np.float64),
        ("baked_animation_vertex_normals", np.float64),
        ("faces", np.int64),
        ("face_uvs", np.float64),
        ("face_materials", np.int64),
    ]:
        if k in out and out[k] is not None:
            out[k] = np.asarray(out[k], dtype=dtype)
    if "material_names" in out and out["material_names"] is not None:
        out["material_names"] = [str(x) for x in out["material_names"]]
    if "bone_names" in out and out["bone_names"] is not None:
        out["bone_names"] = [str(x) for x in out["bone_names"]]
    return out


def _normalize_anim_or_skel_dict(obj: dict[str, Any]) -> dict[str, Any]:
    out = dict(obj)
    if "bone_names" in out and out["bone_names"] is not None:
        out["bone_names"] = [str(x) for x in out["bone_names"]]
    for k in ("rest_pose", "animation_poses"):
        if k in out and out[k] is not None:
            out[k] = np.asarray(out[k], dtype=np.float64)
    if "scene_frames" in out and out["scene_frames"] is not None:
        out["scene_frames"] = np.asarray(out["scene_frames"], dtype=np.int64)
    return out


def _copy_bone_reduction_variants(
    *,
    input_tree: Path,
    dataset_model_dir: Path | None,
    output_tree: Path,
    bone_reduce_variants_json: str,
    model_name: str = "",
) -> dict[str, Any]:
    """
    Ensure `output_tree` has either `bone_reduced_versions/<variant>` or
    legacy `skeleton_hierarchy_data_bone_reduced.pkl` + `animation_data_bone_reduced.pkl`.
    Also supports building `bone_reduced_versions/<variant>` directly from the bare
    parameter `--bone_reduce_variants` JSON dict!
    """
    import importlib.util
    spec02 = importlib.util.spec_from_file_location(
        "_ssdr_step02",
        Path(__file__).resolve().parents[1] / "ssdr" / "02_reduce_bone_weight.py",
    )
    mod02 = importlib.util.module_from_spec(spec02)
    sys.modules["_ssdr_step02"] = mod02
    spec02.loader.exec_module(mod02)

    full_skel = load_data(output_tree / "skeleton_hierarchy_data.pkl")
    full_anim = load_data(output_tree / "animation_data.pkl")
    full_skel = mod02.align_skeleton_to_animation_order(full_skel, full_anim, role="full")

    # 0. Prefer exact bone_reduced_versions from input_tree, dataset_model_dir, or canonical main pool handoff
    canonical_handoffs = {
        "Griffin_A2": Path("/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1/results/r000001/data0_json/bone_reduced_versions"),
        "Winter_Samurai": Path("/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1/results/r000002/data0_json/bone_reduced_versions"),
    }
    br_candidates = [input_tree / "bone_reduced_versions"]
    if dataset_model_dir is not None:
        br_candidates.append(dataset_model_dir / "bone_reduced_versions")
        br_candidates.append(dataset_model_dir / "data0" / "bone_reduced_versions")
    if model_name in canonical_handoffs:
        br_candidates.append(canonical_handoffs[model_name])

    for src_br_root in br_candidates:
        if src_br_root.is_dir():
            dst_br_root = output_tree / "bone_reduced_versions"
            dst_br_root.mkdir(parents=True, exist_ok=True)
            copied_variants = []
            for var_dir in sorted(src_br_root.iterdir(), key=lambda p: p.name):
                if not var_dir.is_dir():
                    continue
                s_files = [
                    p for p in sorted(var_dir.iterdir(), key=lambda x: x.name)
                    if p.is_file() and p.suffix.lower() in (".pkl", ".json")
                ]
                if not s_files:
                    continue
                out_var = dst_br_root / var_dir.name
                out_var.mkdir(parents=True, exist_ok=True)
                for sf in s_files:
                    dump_pkl(out_var / f"{sf.stem}.pkl", _normalize_anim_or_skel_dict(load_data(sf)))
                copied_variants.append(var_dir.name)
            if copied_variants:
                return {"mode": "canonical_bone_reduced_versions", "source": str(src_br_root), "variants": copied_variants}

    # 1. Fallback: If bare parameter --bone_reduce_variants JSON is provided, synthesize bone_reduced_versions/
    variants_text = (bone_reduce_variants_json or "").strip()
    if variants_text and variants_text != "{}":
        variants_map = json.loads(variants_text)
        if isinstance(variants_map, dict) and variants_map:
            br_root = output_tree / "bone_reduced_versions"
            br_root.mkdir(parents=True, exist_ok=True)
            created_variants = []
            for var_name, remove_list in variants_map.items():
                pruned_skel, _ = mod02.prune_skeleton_by_removing_bones(
                    full_skel,
                    [str(x) for x in remove_list],
                    missing_ok=True,
                )
                pruned_anim = mod02.subset_animation_to_bone_names(
                    full_anim,
                    pruned_skel["bone_names"],
                    role=var_name,
                )
                var_dir = br_root / str(var_name)
                var_dir.mkdir(parents=True, exist_ok=True)
                dump_pkl(var_dir / "skeleton_hierarchy_data.pkl", pruned_skel)
                dump_pkl(var_dir / "animation_data.pkl", pruned_anim)
                created_variants.append(str(var_name))
            return {"mode": "bare_param_bone_reduce_variants", "variants": created_variants}

    # 2. Check existing bone_reduced_versions in input_tree
    src_br_root = input_tree / "bone_reduced_versions"
    if src_br_root.is_dir():
        dst_br_root = output_tree / "bone_reduced_versions"
        dst_br_root.mkdir(parents=True, exist_ok=True)
        copied_variants = []
        for var_dir in sorted(src_br_root.iterdir()):
            if not var_dir.is_dir():
                continue
            s_files = discover_stage_files(var_dir, suffixes=(".pkl", ".json"), resolve_by_stem=True)
            if not s_files:
                continue
            out_var = dst_br_root / var_dir.name
            out_var.mkdir(parents=True, exist_ok=True)
            for sf in s_files:
                dump_pkl(out_var / f"{sf.stem}.pkl", _normalize_anim_or_skel_dict(load_data(sf)))
            copied_variants.append(var_dir.name)
        if copied_variants:
            return {"mode": "copied_bone_reduced_versions", "variants": copied_variants}

    # 3. Check legacy *_bone_reduced in input_tree or dataset_model_dir/data0
    search_dirs = [input_tree]
    if dataset_model_dir is not None:
        search_dirs.extend([dataset_model_dir / "data0", dataset_model_dir / "target", dataset_model_dir])

    for s_dir in search_dirs:
        if not s_dir.is_dir():
            continue
        skel_br = next(
            (s_dir / f"skeleton_hierarchy_data_bone_reduced{ext}" for ext in (".pkl", ".json") if (s_dir / f"skeleton_hierarchy_data_bone_reduced{ext}").is_file()),
            None,
        )
        anim_br = next(
            (s_dir / f"animation_data_bone_reduced{ext}" for ext in (".pkl", ".json") if (s_dir / f"animation_data_bone_reduced{ext}").is_file()),
            None,
        )
        if skel_br is not None and anim_br is not None:
            dump_pkl(
                output_tree / "skeleton_hierarchy_data_bone_reduced.pkl",
                _normalize_anim_or_skel_dict(load_data(skel_br)),
            )
            dump_pkl(
                output_tree / "animation_data_bone_reduced.pkl",
                _normalize_anim_or_skel_dict(load_data(anim_br)),
            )
            return {"mode": "legacy_flat_bone_reduced", "source_dir": str(s_dir)}

    fail(
        f"Could not find bone reduction definitions in {input_tree}, "
        f"--bone_reduce_variants, or {dataset_model_dir}."
    )
    raise RuntimeError("unreachable")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UE5-Simp Step 00: prepare textured unmerged Fitbox inputs (data0_json -> data0)."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    parser.add_argument("-m", "--metrics_dir", type=str, default=None)

    parser.add_argument("--dataset_root", type=str, default="/workspace/ogd_dataset")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--bone_reduce_variants", type=str, default="{}")

    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--allow_merge", action="store_true")
    parser.add_argument("--no_summary", action="store_true")
    parser.add_argument("--enable_profiling", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(script_argv())

    resolution = resolve_ssdr_stage_dirs(
        step_id="00",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="UE5-Simp Step 00: data0_json -> data0 Prepare Textured Inputs",
    )
    print_ssdr_resolution_info(resolution)
    validate_ssdr_stage_resolution(resolution)

    input_tree = resolution.input_tree
    output_tree = resolution.output_tree

    # Resolve textured source directory (prefer explicit dataset_root/model_name/target if present)
    model_name = str(args.model_name or "").strip()
    dataset_model_dir: Path | None = None
    textured_source_dir = input_tree

    if model_name:
        cand_model_dir = Path(args.dataset_root).expanduser().resolve(strict=False) / model_name
        if cand_model_dir.is_dir():
            dataset_model_dir = cand_model_dir
            if (cand_model_dir / "target_200").is_dir():
                textured_source_dir = cand_model_dir / "target_200"
            elif (cand_model_dir / "target").is_dir():
                textured_source_dir = cand_model_dir / "target"
    else:
        # Infer model_name if input_tree has a target_export_summary or parent model name
        for cand_name in ("Griffin_A2", "Winter_Samurai", "Vroid_18", "Vroid_19", "Vroid_20"):
            if cand_name.lower() in str(input_tree).lower():
                model_name = cand_name
                cand_model_dir = Path(args.dataset_root).expanduser().resolve(strict=False) / model_name
                if cand_model_dir.is_dir():
                    dataset_model_dir = cand_model_dir
                    if (cand_model_dir / "target_200").is_dir():
                        textured_source_dir = cand_model_dir / "target_200"
                    elif (cand_model_dir / "target").is_dir():
                        textured_source_dir = cand_model_dir / "target"
                break

    prepare_output_dir(
        output_tree,
        overwrite_output=bool(args.overwrite_output),
        allow_merge=bool(args.allow_merge),
    )

    tex_out_dir = output_tree / "textures"
    tex_out_dir.mkdir(parents=True, exist_ok=True)

    with Timer("time_00_prepare_textured"):
        stage_files = discover_stage_files(
            textured_source_dir,
            suffixes=(".pkl", ".json"),
            exclude_stems={"target_export_summary"},
            recursive=False,
            resolve_by_stem=True,
        )
        if not stage_files:
            fail(f"No .pkl or .json files found in textured source dir: {textured_source_dir}")

        mesh_records = []
        for sf in stage_files:
            obj = load_data(sf)
            if not isinstance(obj, dict):
                continue
            if sf.stem in {"skeleton_hierarchy_data", "animation_data", "skeleton_hierarchy_data_bone_reduced", "animation_data_bone_reduced"}:
                dump_pkl(output_tree / f"{sf.stem}.pkl", _normalize_anim_or_skel_dict(obj))
            elif sf.stem not in KNOWN_NON_MESH_STEMS and "vertices" in obj and "faces" in obj:
                norm_mesh = _normalize_mesh_dict(obj, sf.stem)
                if model_name:
                    norm_mesh["model_name"] = model_name
                dump_pkl(output_tree / f"{sf.stem}.pkl", norm_mesh)
                mesh_records.append({
                    "name": sf.stem,
                    "vertices": int(norm_mesh["vertices"].shape[0]),
                    "faces": int(norm_mesh["faces"].shape[0]),
                    "has_uvs": bool("face_uvs" in norm_mesh and norm_mesh["face_uvs"] is not None),
                    "material_names": norm_mesh.get("material_names", []),
                })

        # Copy all PNG textures from textured_source_dir (and input_tree if any)
        copied_textures = []
        for png_src_dir in {textured_source_dir, input_tree}:
            if not png_src_dir.is_dir():
                continue
            for png_path in sorted(png_src_dir.glob("*.png")):
                dst_png = tex_out_dir / png_path.name
                shutil.copyfile(png_path, dst_png)
                copied_textures.append(png_path.name)

        bone_var_info = _copy_bone_reduction_variants(
            input_tree=input_tree,
            dataset_model_dir=dataset_model_dir,
            output_tree=output_tree,
            bone_reduce_variants_json=str(args.bone_reduce_variants),
            model_name=model_name,
        )

    summary = {
        "schema": "ogd_lbs_ue5_simp_step00_prepare_v1",
        "model_name": model_name,
        "input_tree": str(input_tree),
        "textured_source_dir": str(textured_source_dir),
        "output_tree": str(output_tree),
        "mesh_count": len(mesh_records),
        "texture_count": len(set(copied_textures)),
        "textures": sorted(set(copied_textures)),
        "bone_reduction_setup": bone_var_info,
        "meshes": mesh_records,
    }
    if not bool(args.no_summary):
        write_stage_summary(output_tree / "step00_convert_summary.json", summary)

    record_metric("00_mesh_count", len(mesh_records))
    record_metric("00_texture_count", len(set(copied_textures)))
    if args.metrics_dir:
        dump_metrics(Path(args.metrics_dir) / "metrics_00.json")

    log(
        f"UE5-Simp Step 00 complete: model={model_name!r}, meshes={len(mesh_records)}, "
        f"textures={len(set(copied_textures))}, out={output_tree}"
    )


if __name__ == "__main__":
    main()
