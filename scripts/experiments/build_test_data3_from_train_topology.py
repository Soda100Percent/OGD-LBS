#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Build test data3 by reusing train data3 low-poly topology and weights.

Purpose:
    In 4-fold CV, Step 03 QEM must be run only on train frames.  Test frames
    should reuse the exact train low-poly topology / weights and replace only
    the baked target animation sequence.

Inputs:
    --train_data3
        Fold train data3 produced by SSDR Step 03. Meshes must contain
        qem_vertex_map persisted by StepA.

    --test_data2
        Fold test data2 produced by SSDR Step 02. Meshes are full topology
        and contain held-out baked_animation_vertices.

Output:
    data3_test/
        skeleton_hierarchy_data.pkl
        animation_data.pkl
        <mesh>.pkl
        build_test_data3_summary.json
"""

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np

try:
    from _env import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from ._env import REPO_ROOT  # noqa: F401

from ogd_lbs.utils.common import load_data, dump_pkl, dump_json, script_argv, fail
from ogd_lbs.fitbox.schema import (
    KEY_ANIMATION_POSES,
    KEY_BAKED_NORMALS,
    KEY_BAKED_VERTICES,
    KEY_BONE_NAMES,
    KEY_FACES,
    KEY_SCENE_FRAMES,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTICES,
)
from ogd_lbs.fitbox.validation import validate_skeleton_and_animation, validate_fitbox_mesh


CORE_STEMS = {"skeleton_hierarchy_data", "animation_data"}
SUMMARY_STEMS = {
    "step00_convert_summary",
    "step01_frame_sampling_summary",
    "step02_reduce_bone_weight_summary",
    "step03_qem_summary",
    "step04_optimize_summary",
    "build_test_data3_summary",
}


def discover_mesh_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.glob("*.pkl"), key=lambda p: p.name):
        if not path.is_file():
            continue
        if path.stem in CORE_STEMS or path.stem in SUMMARY_STEMS:
            continue
        paths.append(path)
    return paths


def require_file(path: Path, role: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")
    return path


def load_required_dict(path: Path, role: str) -> dict[str, Any]:
    obj = load_data(path)
    if not isinstance(obj, dict):
        fail(f"{role} must contain a dict object: {path}")
    return obj


def build_single_test_mesh(
    *,
    train_mesh: dict[str, Any],
    test_mesh: dict[str, Any],
    mesh_name: str,
) -> dict[str, Any]:
    if "qem_vertex_map" not in train_mesh:
        fail(
            f"[{mesh_name}] train data3 mesh is missing qem_vertex_map. "
            "Re-run Step 03 after StepA qem-lineage patch."
        )

    qem_vertex_map = np.asarray(train_mesh["qem_vertex_map"], dtype=np.int64)
    if qem_vertex_map.ndim != 1:
        fail(f"[{mesh_name}] qem_vertex_map must be 1D, got {qem_vertex_map.shape}")

    test_baked = np.asarray(test_mesh[KEY_BAKED_VERTICES], dtype=np.float64)
    if test_baked.ndim != 3 or test_baked.shape[2] != 3:
        fail(f"[{mesh_name}] test baked vertices expected (P,V,3), got {test_baked.shape}")

    if np.any(qem_vertex_map < 0) or np.any(qem_vertex_map >= test_baked.shape[1]):
        fail(
            f"[{mesh_name}] qem_vertex_map out of bounds for test data2 vertices: "
            f"map_min={int(np.min(qem_vertex_map))}, "
            f"map_max={int(np.max(qem_vertex_map))}, "
            f"test_vertices={test_baked.shape[1]}"
        )

    out = copy.deepcopy(train_mesh)

    out[KEY_BAKED_VERTICES] = test_baked[:, qem_vertex_map, :]

    if KEY_BAKED_NORMALS in test_mesh:
        test_normals = np.asarray(test_mesh[KEY_BAKED_NORMALS], dtype=np.float64)
        if test_normals.ndim == 3 and test_normals.shape[:2] == test_baked.shape[:2]:
            out[KEY_BAKED_NORMALS] = test_normals[:, qem_vertex_map, :]

    if KEY_SCENE_FRAMES in test_mesh:
        out[KEY_SCENE_FRAMES] = np.asarray(test_mesh[KEY_SCENE_FRAMES], dtype=np.int64)

    out["cv_test_data3_source"] = {
        "schema": "ogd_lbs_cv_test_data3_source_v1",
        "mode": "train_topology_test_animation",
        "mesh_name": mesh_name,
        "qem_vertex_map_length": int(qem_vertex_map.shape[0]),
        "test_frame_count": int(test_baked.shape[0]),
    }

    return out


def build_test_data3_tree(
    *,
    train_data3: Path,
    test_data2: Path,
    output_dir: Path,
    overwrite_output: bool,
    validate_output: bool,
    weak_validate: bool,
) -> dict[str, Any]:
    train_data3 = Path(train_data3).expanduser().resolve(strict=False)
    test_data2 = Path(test_data2).expanduser().resolve(strict=False)
    output_dir = Path(output_dir).expanduser().resolve(strict=False)

    if not train_data3.is_dir():
        raise FileNotFoundError(f"train_data3 is not a directory: {train_data3}")
    if not test_data2.is_dir():
        raise FileNotFoundError(f"test_data2 is not a directory: {test_data2}")

    if output_dir.exists() and overwrite_output:
        import shutil
        shutil.rmtree(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite_output:
        raise RuntimeError(f"output_dir exists and is non-empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_skel = load_required_dict(train_data3 / "skeleton_hierarchy_data.pkl", "train skeleton")
    train_anim = load_required_dict(train_data3 / "animation_data.pkl", "train animation")
    test_skel = load_required_dict(test_data2 / "skeleton_hierarchy_data.pkl", "test skeleton")
    test_anim = load_required_dict(test_data2 / "animation_data.pkl", "test animation")

    if [str(x) for x in train_skel.get(KEY_BONE_NAMES, [])] != [str(x) for x in test_skel.get(KEY_BONE_NAMES, [])]:
        fail("Train data3 and test data2 skeleton bone_names differ.")

    # Output skeleton can be copied from train; output animation must be test.
    n_poses, n_bones = validate_skeleton_and_animation(test_anim, test_skel)

    dump_pkl(output_dir / "skeleton_hierarchy_data.pkl", train_skel)
    dump_pkl(output_dir / "animation_data.pkl", test_anim)

    train_mesh_paths = discover_mesh_files(train_data3)
    if not train_mesh_paths:
        fail(f"No train data3 mesh files found: {train_data3}")

    processed = []

    for train_mesh_path in train_mesh_paths:
        test_mesh_path = test_data2 / train_mesh_path.name
        require_file(test_mesh_path, role=f"test data2 mesh for {train_mesh_path.name}")

        train_mesh = load_required_dict(train_mesh_path, f"train mesh {train_mesh_path.name}")
        test_mesh = load_required_dict(test_mesh_path, f"test mesh {test_mesh_path.name}")

        mesh_name = str(train_mesh.get("name", train_mesh_path.stem))

        out_mesh = build_single_test_mesh(
            train_mesh=train_mesh,
            test_mesh=test_mesh,
            mesh_name=mesh_name,
        )

        if validate_output:
            validate_fitbox_mesh(
                out_mesh,
                n_poses=n_poses,
                n_bones=n_bones,
                name=f"{mesh_name}/cv_test_data3",
                weak_validate=bool(weak_validate),
            )

        out_path = output_dir / train_mesh_path.name
        dump_pkl(out_path, out_mesh)

        processed.append(
            {
                "mesh_name": mesh_name,
                "train_mesh": str(train_mesh_path),
                "test_mesh": str(test_mesh_path),
                "output_mesh": str(out_path),
                "output_vertices": int(np.asarray(out_mesh[KEY_VERTICES]).shape[0]),
                "output_faces": int(np.asarray(out_mesh[KEY_FACES]).shape[0]),
                "test_frames": int(np.asarray(out_mesh[KEY_BAKED_VERTICES]).shape[0]),
                "qem_vertex_map_length": int(np.asarray(out_mesh["qem_vertex_map"]).shape[0]),
                "weight_shape": list(np.asarray(out_mesh[KEY_VERTEX_BONE_WEIGHTS]).shape),
            }
        )

    summary = {
        "schema": "ogd_lbs_cv_build_test_data3_summary_v1",
        "train_data3": str(train_data3),
        "test_data2": str(test_data2),
        "output_dir": str(output_dir),
        "counts": {
            "frames": int(n_poses),
            "bones": int(n_bones),
            "meshes": int(len(processed)),
        },
        "files": processed,
        "notes": [
            "Static low-poly topology and weights are copied from train data3.",
            "Baked animation vertices/normals are sampled from test data2 using train qem_vertex_map.",
        ],
    }

    dump_json(output_dir / "build_test_data3_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build test data3 from train topology and test data2 baked animation."
    )
    parser.add_argument("--train_data3", type=str, required=True)
    parser.add_argument("--test_data2", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--validate_output", action="store_true")
    parser.add_argument("--weak_validate", action="store_true")

    args = parser.parse_args(script_argv())

    summary = build_test_data3_tree(
        train_data3=Path(args.train_data3),
        test_data2=Path(args.test_data2),
        output_dir=Path(args.output_dir),
        overwrite_output=bool(args.overwrite_output),
        validate_output=bool(args.validate_output),
        weak_validate=bool(args.weak_validate),
    )

    print("Built CV test data3:", flush=True)
    print(f"  output_dir: {summary['output_dir']}", flush=True)
    print(f"  frames    : {summary['counts']['frames']}", flush=True)
    print(f"  bones     : {summary['counts']['bones']}", flush=True)
    print(f"  meshes    : {summary['counts']['meshes']}", flush=True)


if __name__ == "__main__":
    main()
