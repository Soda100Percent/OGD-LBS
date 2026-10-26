# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic rebase / virtual mount math helpers.
"""

from typing import Any

import numpy as np

from ogd_lbs.blender.transforms import (
    assert_affine_matrix,
    make_reflection_matrix,
    aim_local_axis_at_target,
)


def compute_rest_basis_correction(old_rest_world: Any, new_rest_world: Any) -> np.ndarray:
    old_rest = np.asarray(old_rest_world, dtype=np.float64)
    new_rest = np.asarray(new_rest_world, dtype=np.float64)

    if old_rest.shape != new_rest.shape:
        raise RuntimeError(f"Rest shape mismatch: {old_rest.shape} vs {new_rest.shape}")

    assert_affine_matrix(old_rest, label="old_rest_world")
    assert_affine_matrix(new_rest, label="new_rest_world")

    return np.linalg.inv(old_rest) @ new_rest


def apply_rest_basis_correction(old_pose_world: Any, correction: Any) -> np.ndarray:
    old_pose = np.asarray(old_pose_world, dtype=np.float64)
    corr = np.asarray(correction, dtype=np.float64)

    assert_affine_matrix(old_pose, label="old_pose_world")
    assert_affine_matrix(corr, label="correction")

    return old_pose @ corr


def compute_lbs_operator_world(pose_world: Any, rest_world: Any) -> np.ndarray:
    pose = np.asarray(pose_world, dtype=np.float64)
    rest = np.asarray(rest_world, dtype=np.float64)

    assert_affine_matrix(pose, label="pose_world")
    assert_affine_matrix(rest, label="rest_world")

    return pose @ np.linalg.inv(rest)


def compute_direct_mount_offset(
    mount_world_ref: Any,
    driven_world_ref: Any,
) -> np.ndarray:
    mount = np.asarray(mount_world_ref, dtype=np.float64)
    driven = np.asarray(driven_world_ref, dtype=np.float64)

    assert_affine_matrix(mount, label="mount_world_ref")
    assert_affine_matrix(driven, label="driven_world_ref")

    return np.linalg.inv(mount) @ driven


def compute_mirror_mount_offset(
    source_offset: Any,
    *,
    reflection_signs: tuple[float, float, float] | None = None,
    reflection_axes: list[str] | tuple[str, ...] | str | None = None,
    translation_offset: list[float] | tuple[float, float, float] | None = None,
) -> np.ndarray:
    offset = np.asarray(source_offset, dtype=np.float64)
    assert_affine_matrix(offset, label="source_offset")

    reflect = make_reflection_matrix(signs=reflection_signs, axes=reflection_axes)
    mirrored = reflect @ offset @ reflect

    if translation_offset is not None:
        vec = np.asarray(translation_offset, dtype=np.float64).reshape(3)
        mirrored[:3, 3] += vec

    assert_affine_matrix(mirrored, label="mirrored_offset")
    return mirrored


def make_virtual_mount_world_sequence(
    mount_world_sequence: Any,
    offset_mount_to_driven: Any,
) -> np.ndarray:
    mounts = np.asarray(mount_world_sequence, dtype=np.float64)
    offset = np.asarray(offset_mount_to_driven, dtype=np.float64)

    assert_affine_matrix(mounts, label="mount_world_sequence")
    assert_affine_matrix(offset, label="offset_mount_to_driven")

    return mounts @ offset


def make_aimed_world_sequence(
    baseline_world_sequence: Any,
    target_world_points: Any,
    *,
    axis: str = "-Y",
    preserve_axis: str = "+Z",
) -> tuple[np.ndarray, list[str]]:
    baseline = np.asarray(baseline_world_sequence, dtype=np.float64)
    targets = np.asarray(target_world_points, dtype=np.float64)

    if baseline.ndim != 3 or baseline.shape[1:] != (4, 4):
        raise RuntimeError(f"Expected baseline shape (T,4,4), got {baseline.shape}")

    if targets.shape != (baseline.shape[0], 3):
        raise RuntimeError(f"Expected target points shape {(baseline.shape[0], 3)}, got {targets.shape}")

    out = np.empty_like(baseline)
    branches: list[str] = []

    for i in range(baseline.shape[0]):
        out[i], branch = aim_local_axis_at_target(
            baseline[i],
            targets[i],
            axis=axis,
            preserve_axis=preserve_axis,
        )
        branches.append(branch)

    return out, branches
