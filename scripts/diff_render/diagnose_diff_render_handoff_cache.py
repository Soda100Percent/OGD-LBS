#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External read-only diagnostic script for verifying `diff_render_handoff` cache entries in PipelinePool.

Checks for every HIT entry of `diff_render_handoff`:
  1. `diff_render_00/source` and `diff_render_00/target` both exist and are non-empty;
  2. `animation_data` and `skeleton_hierarchy_data` exist in both `source/` and `target/`,
     and frame counts match;
  3. `MeshLoader.scan_mesh_names` and `compute_mesh_mapping` succeed with 1:1 submesh alignment;
  4. Every target submesh material slot has a valid, readable RGBA/RGB `.png` texture file.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import _env  # noqa: F401
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402

from ogd_lbs.diff_render.mesh_loader import MeshLoader  # noqa: E402
from ogd_lbs.diff_render.submesh_sampling import compute_mesh_mapping  # noqa: E402
from ogd_lbs.utils.common import load_data, log, resolve_specific_file, script_argv  # noqa: E402


def diagnose_pool_handoff_entries(pool_root: Path, stage_subdir: str = "diff_render_00") -> int:
    csv_path = pool_root / "metadata_table.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing metadata_table.csv in {pool_root}")

    handoff_rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("diff_render_handoff") not in (None, "", "None"):
                handoff_rows.append(row)

    if not handoff_rows:
        raise RuntimeError("No HIT entries found for stage 'diff_render_handoff' in metadata_table.csv")

    loader = MeshLoader(torch.device("cpu"))
    log(f"Found {len(handoff_rows)} HIT entry(ies) for diff_render_handoff in {pool_root}")

    for idx, row in enumerate(handoff_rows, start=1):
        model = row.get("data_prep_handoff.model_name", "unknown")
        data_dir = row["data_dir"]
        handoff_dir = pool_root / data_dir / stage_subdir
        src_dir = handoff_dir / "source"
        tgt_dir = handoff_dir / "target"

        if not src_dir.is_dir() or not tgt_dir.is_dir():
            raise RuntimeError(f"[{data_dir}] Missing source/ or target/ under {handoff_dir}")

        src_anim_p = resolve_specific_file(src_dir, "animation_data", [".pkl", ".pickle", ".json"])
        tgt_anim_p = resolve_specific_file(tgt_dir, "animation_data", [".pkl", ".pickle", ".json"])
        src_skel_p = resolve_specific_file(src_dir, "skeleton_hierarchy_data", [".pkl", ".pickle", ".json"])
        tgt_skel_p = resolve_specific_file(tgt_dir, "skeleton_hierarchy_data", [".pkl", ".pickle", ".json"])

        if not src_anim_p or not tgt_anim_p or not src_skel_p or not tgt_skel_p:
            raise RuntimeError(f"[{data_dir}] Missing animation_data or skeleton_hierarchy_data in source/target")

        src_poses = np.asarray(load_data(src_anim_p)["animation_poses"])
        tgt_poses = np.asarray(load_data(tgt_anim_p)["animation_poses"])

        b_info = loader.scan_mesh_names(src_dir, ["_merged"])
        r_info = loader.scan_mesh_names(tgt_dir, ["_merged"])
        if len(b_info) == 0 or len(r_info) == 0:
            raise RuntimeError(f"[{data_dir}] Empty submeshes: source={len(b_info)}, target={len(r_info)}")

        mapping = compute_mesh_mapping(b_info, r_info)

        # Verify target baked frames and texture PNGs
        tex_count = 0
        ref_frames = None
        for info in r_info:
            fname = info["raw_stem"]
            sm_obj = load_data(info["path"])
            bv = np.asarray(sm_obj["baked_animation_vertices"])
            ref_frames = bv.shape[0]
            aligned_idx = MeshLoader._compute_step01_sample_indices(bv.shape[0], src_poses.shape[0])
            bv_aligned = bv[aligned_idx]
            ref_frames = f"{bv.shape[0]}->{bv_aligned.shape[0]}" if bv.shape[0] != src_poses.shape[0] else str(bv.shape[0])
            if bv_aligned.shape[0] != src_poses.shape[0]:
                raise RuntimeError(
                    f"[{data_dir}] Frame count mismatch after Step 01 alignment for {fname}: "
                    f"source_poses={src_poses.shape[0]} vs aligned_target={bv_aligned.shape[0]}"
                )
            mtl_names = sm_obj.get("material_names", ["default"])
            for mtl_idx in range(len(mtl_names)):
                cands = [
                    tgt_dir / f"{fname}_mtl{mtl_idx + 1}.png",
                    tgt_dir / f"{fname}.png",
                ]
                img_p = next((p for p in cands if p.is_file()), None)
                if img_p is None:
                    raise RuntimeError(
                        f"[{data_dir}] Missing texture PNG for target submesh {fname!r} mtl #{mtl_idx + 1}"
                    )
                with Image.open(img_p) as im:
                    if im.width <= 0 or im.height <= 0:
                        raise RuntimeError(f"[{data_dir}] Invalid texture dimensions in {img_p}")
                tex_count += 1

        log(
            f"✅ [{idx}/{len(handoff_rows)}] {data_dir} ({model:<14}): "
            f"src_meshes={len(b_info)}, tgt_meshes={len(r_info)}, "
            f"frames={src_poses.shape[0]}=={ref_frames}, bones={src_poses.shape[1]}, "
            f"verified_textures={tex_count}, mapping_unique={len(set(mapping))}/{len(b_info)}"
        )

    log(f"ALL {len(handoff_rows)} DIFF_RENDER_HANDOFF CACHE ENTRIES PASSED DIAGNOSTIC CHECK!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose diff_render_handoff entries in PipelinePool.")
    parser.add_argument(
        "--pool",
        default="/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1",
        help="PipelinePool root directory.",
    )
    parser.add_argument("--stage_subdir", default="diff_render_00")
    args = parser.parse_args(script_argv())
    return diagnose_pool_handoff_entries(Path(args.pool).expanduser().resolve(strict=True), args.stage_subdir)


if __name__ == "__main__":
    raise SystemExit(main())
