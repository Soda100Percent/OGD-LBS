# -*- coding: utf-8 -*-
from __future__ import annotations

"""
4-fold CV utilities for OGD-LBS experiment scripts.

This module is pure Python / NumPy and operates on SSDR stage trees.
It intentionally avoids Blender dependencies.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ogd_lbs.utils.common import load_data, dump_pkl
from ogd_lbs.ssdr.frame_sampling import (
    apply_frame_sample_plan_to_animation,
    apply_frame_sample_plan_to_mesh,
)


CORE_STEMS = {
    "skeleton_hierarchy_data",
    "animation_data",
    "skeleton_hierarchy_data_bone_reduced",
    "animation_data_bone_reduced",
}

SUMMARY_STEMS = {
    "step00_convert_summary",
    "step01_frame_sampling_summary",
    "step02_reduce_bone_weight_summary",
    "step03_qem_summary",
    "step04_optimize_summary",
}


@dataclass(frozen=True)
class FoldFrameSplit:
    fold_index: int
    frame_count: int
    kfold: int
    special_indices: np.ndarray
    pool_indices: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    stats: dict[str, Any]


def discover_pkl_files(input_tree: Path) -> list[Path]:
    return sorted(
        [p for p in Path(input_tree).rglob("*.pkl") if p.is_file()],
        key=lambda p: str(p.relative_to(input_tree)),
    )


def classify_pkl_object(obj: Any) -> str:
    if not isinstance(obj, dict):
        return "copy"

    if "animation_poses" in obj:
        return "animation"

    if "baked_animation_vertices" in obj or "baked_animation_vertex_normals" in obj:
        return "mesh"

    return "copy"


def infer_frame_count_from_obj(obj: Any) -> int | None:
    if not isinstance(obj, dict):
        return None

    if "animation_poses" in obj:
        arr = np.asarray(obj["animation_poses"])
        if arr.ndim >= 1:
            return int(arr.shape[0])

    if "baked_animation_vertices" in obj:
        arr = np.asarray(obj["baked_animation_vertices"])
        if arr.ndim >= 1:
            return int(arr.shape[0])

    return None


def infer_reference_frame_count(input_tree: Path) -> tuple[int, Path]:
    pkl_files = discover_pkl_files(input_tree)
    for path in pkl_files:
        obj = load_data(path)
        n = infer_frame_count_from_obj(obj)
        if n is not None:
            return int(n), path

    raise RuntimeError(
        f"No time-series PKL found under {input_tree}; expected animation_poses "
        "or baked_animation_vertices."
    )


def parse_ranges_to_indices(
    ranges: list[list[int]] | tuple[tuple[int, int], ...] | None,
    *,
    frame_count: int,
) -> np.ndarray:
    mask = np.zeros(int(frame_count), dtype=bool)

    for item in ranges or []:
        if len(item) != 2:
            raise ValueError(f"Range item must be [start, end], got {item!r}")

        start, end = int(item[0]), int(item[1])
        if start < 0 or end < 0:
            raise ValueError(f"Range indices must be non-negative, got {item!r}")
        if end < start:
            raise ValueError(f"Range end must be >= start, got {item!r}")
        if end > int(frame_count):
            raise ValueError(f"Range {item!r} exceeds frame_count={frame_count}")

        mask[start:end] = True

    return np.flatnonzero(mask).astype(np.int64)


def make_kfold_frame_split(
    frame_count: int,
    *,
    kfold: int,
    fold_index: int,
    special_train_ranges: list[list[int]] | None = None,
) -> FoldFrameSplit:
    frame_count = int(frame_count)
    kfold = int(kfold)
    fold_index = int(fold_index)

    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if kfold <= 1:
        raise ValueError(f"kfold must be > 1, got {kfold}")
    if fold_index < 0 or fold_index >= kfold:
        raise ValueError(f"fold_index must be in [0, {kfold}), got {fold_index}")

    all_indices = np.arange(frame_count, dtype=np.int64)
    special = parse_ranges_to_indices(
        special_train_ranges,
        frame_count=frame_count,
    )

    special_set = set(int(x) for x in special.tolist())
    pool = np.asarray([idx for idx in all_indices.tolist() if int(idx) not in special_set], dtype=np.int64)

    if pool.size < kfold:
        raise ValueError(
            f"Pool frame count must be >= kfold after removing special frames; "
            f"pool={pool.size}, kfold={kfold}"
        )

    chunks = np.array_split(pool, kfold)
    test = np.asarray(chunks[fold_index], dtype=np.int64)
    test_set = set(int(x) for x in test.tolist())

    train = np.asarray(
        [
            idx
            for idx in all_indices.tolist()
            if int(idx) in special_set or int(idx) not in test_set
        ],
        dtype=np.int64,
    )

    # Defensive ordering.
    train = np.asarray(sorted(set(int(x) for x in train.tolist())), dtype=np.int64)
    test = np.asarray(sorted(set(int(x) for x in test.tolist())), dtype=np.int64)

    if set(train.tolist()) & set(test.tolist()):
        raise RuntimeError("Internal split error: train and test overlap.")

    if not set(special.tolist()).issubset(set(train.tolist())):
        raise RuntimeError("Internal split error: special frames are not all in train.")

    stats = {
        "schema": "ogd_lbs_cv_frame_split_v1",
        "fold_index": int(fold_index),
        "kfold": int(kfold),
        "frame_count": int(frame_count),
        "special_train_ranges": special_train_ranges or [],
        "special_count": int(special.shape[0]),
        "pool_count": int(pool.shape[0]),
        "train_count": int(train.shape[0]),
        "test_count": int(test.shape[0]),
        "special_indices": [int(x) for x in special.tolist()],
        "pool_indices": [int(x) for x in pool.tolist()],
        "train_indices": [int(x) for x in train.tolist()],
        "test_indices": [int(x) for x in test.tolist()],
        "range_semantics": "left_closed_right_open",
    }

    return FoldFrameSplit(
        fold_index=int(fold_index),
        frame_count=int(frame_count),
        kfold=int(kfold),
        special_indices=special,
        pool_indices=pool,
        train_indices=train,
        test_indices=test,
        stats=stats,
    )


def _make_plan(frame_count: int, selected_indices: np.ndarray, *, label: str) -> Any:
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    removal_mask = np.ones(int(frame_count), dtype=bool)
    removal_mask[selected_indices] = False

    stats = {
        "schema": "ogd_lbs_cv_selected_frame_plan_v1",
        "label": label,
        "original_frame_count": int(frame_count),
        "selected_count": int(selected_indices.shape[0]),
        "removed_count": int(np.count_nonzero(removal_mask)),
        "selected_indices": [int(x) for x in selected_indices.tolist()],
    }

    return SimpleNamespace(
        original_frame_count=int(frame_count),
        max_frames=int(frame_count),
        target_frames=int(selected_indices.shape[0]),
        selected_indices=selected_indices,
        removal_mask=removal_mask,
        reduced=bool(selected_indices.shape[0] != int(frame_count)),
        stats=stats,
    )


def write_split_tree(
    *,
    input_tree: Path,
    train_output_tree: Path,
    test_output_tree: Path,
    split: FoldFrameSplit,
    overwrite: bool = True,
) -> dict[str, Any]:
    input_tree = Path(input_tree).expanduser().resolve(strict=False)
    train_output_tree = Path(train_output_tree).expanduser().resolve(strict=False)
    test_output_tree = Path(test_output_tree).expanduser().resolve(strict=False)

    if overwrite:
        for out in [train_output_tree, test_output_tree]:
            if out.exists():
                shutil.rmtree(out)

    train_output_tree.mkdir(parents=True, exist_ok=True)
    test_output_tree.mkdir(parents=True, exist_ok=True)

    pkl_files = discover_pkl_files(input_tree)

    train_plan = _make_plan(
        split.frame_count,
        split.train_indices,
        label=f"fold_{split.fold_index}_train",
    )
    test_plan = _make_plan(
        split.frame_count,
        split.test_indices,
        label=f"fold_{split.fold_index}_test",
    )

    processed = []

    for pkl_path in pkl_files:
        rel = pkl_path.relative_to(input_tree)
        train_out = train_output_tree / rel
        test_out = test_output_tree / rel
        train_out.parent.mkdir(parents=True, exist_ok=True)
        test_out.parent.mkdir(parents=True, exist_ok=True)

        obj = load_data(pkl_path)
        kind = classify_pkl_object(obj)

        if kind == "animation":
            train_obj = apply_frame_sample_plan_to_animation(obj, train_plan, copy_obj=True)
            test_obj = apply_frame_sample_plan_to_animation(obj, test_plan, copy_obj=True)
            operation = "split_animation"

        elif kind == "mesh":
            train_obj = apply_frame_sample_plan_to_mesh(obj, train_plan, copy_obj=True)
            test_obj = apply_frame_sample_plan_to_mesh(obj, test_plan, copy_obj=True)
            operation = "split_mesh"

        else:
            train_obj = obj
            test_obj = obj
            operation = "copy_pkl"

        dump_pkl(train_out, train_obj)
        dump_pkl(test_out, test_obj)

        processed.append(
            {
                "relative_path": str(rel),
                "operation": operation,
                "input": str(pkl_path),
                "train_output": str(train_out),
                "test_output": str(test_out),
            }
        )

    summary = {
        "schema": "ogd_lbs_cv_split_data1_summary_v1",
        "input_tree": str(input_tree),
        "train_output_tree": str(train_output_tree),
        "test_output_tree": str(test_output_tree),
        "split": split.stats,
        "processed_count": int(len(processed)),
        "files": processed,
    }

    return summary
