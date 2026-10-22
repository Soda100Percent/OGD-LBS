# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Dynamic frame sampling helpers for SSDR preprocessing.

Design for migrated SSDR Step 01:
    - If frame_count <= max_frames, keep all frames.
    - If frame_count > max_frames, uniformly sample target_frames frames.
    - The same selected frame indices should be applied to:
        animation_data["animation_poses"]
        animation_data["scene_frames"] if present
        mesh["baked_animation_vertices"]
        mesh["baked_animation_vertex_normals"]
        mesh["scene_frames"] if present

Important:
    - rest_pose is never modified.
    - This module does not infer or synthesize rest_pose.
    - This module is pure Python / NumPy and has no bpy dependency.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DynamicFrameSamplePlan:
    original_frame_count: int
    max_frames: int
    target_frames: int
    selected_indices: np.ndarray
    removal_mask: np.ndarray
    reduced: bool
    stats: dict[str, Any]


def _validate_plan_params(
    *,
    frame_count: int,
    max_frames: int,
    target_frames: int,
) -> None:
    if int(frame_count) <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")

    if int(max_frames) <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    if int(target_frames) <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")

    if int(target_frames) > int(max_frames):
        raise ValueError(
            f"target_frames should be <= max_frames for stable dynamic sampling, "
            f"got target_frames={target_frames}, max_frames={max_frames}"
        )


def make_dynamic_frame_sample_plan(
    frame_count: int,
    *,
    max_frames: int = 210,
    target_frames: int = 200,
) -> DynamicFrameSamplePlan:
    """
    Build a deterministic uniform sampling plan.

    Args:
        frame_count:
            Number of animation frames before Step 01.
        max_frames:
            If frame_count <= max_frames, no reduction is applied.
        target_frames:
            If frame_count > max_frames, sample exactly target_frames frames.

    Returns:
        DynamicFrameSamplePlan.

    Notes:
        For frame_count > target_frames, np.linspace over index range [0, N-1]
        produces an approximately uniform subset while preserving first/last
        frames.
    """
    frame_count = int(frame_count)
    max_frames = int(max_frames)
    target_frames = int(target_frames)

    _validate_plan_params(
        frame_count=frame_count,
        max_frames=max_frames,
        target_frames=target_frames,
    )

    if frame_count <= max_frames:
        selected = np.arange(frame_count, dtype=np.int64)
        reduced = False
    else:
        if target_frames >= frame_count:
            selected = np.arange(frame_count, dtype=np.int64)
            reduced = False
        else:
            selected = np.rint(
                np.linspace(0, frame_count - 1, num=target_frames)
            ).astype(np.int64)

            # Defensive unique pass. With target_frames < frame_count and step > 1,
            # duplicates should not occur, but this keeps behavior explicit.
            selected = np.unique(selected)

            if selected.shape[0] != target_frames:
                # Fallback that guarantees exact count by taking evenly spaced
                # integer positions from a larger arange.
                selected = np.linspace(
                    0,
                    frame_count - 1,
                    num=target_frames,
                    dtype=np.int64,
                )
                selected = np.unique(selected)

            if selected.shape[0] != target_frames:
                raise RuntimeError(
                    f"Could not construct exact dynamic sampling indices: "
                    f"frame_count={frame_count}, target_frames={target_frames}, "
                    f"unique_count={selected.shape[0]}"
                )

            reduced = True

    removal_mask = np.ones(frame_count, dtype=bool)
    removal_mask[selected] = False

    removed_count = int(np.count_nonzero(removal_mask))
    retained_count = int(selected.shape[0])

    stats = {
        "schema": "ogd_lbs_ssdr_dynamic_frame_sampling_v1",
        "original_frame_count": frame_count,
        "max_frames": max_frames,
        "target_frames": target_frames,
        "reduced": bool(reduced),
        "removed_count": removed_count,
        "retained_count": retained_count,
        "effective_removal_rate": (
            float(removed_count) / float(frame_count)
            if frame_count > 0
            else 0.0
        ),
        "selected_indices_first": int(selected[0]),
        "selected_indices_last": int(selected[-1]),
        "selected_indices": [int(x) for x in selected.tolist()],
    }

    return DynamicFrameSamplePlan(
        original_frame_count=frame_count,
        max_frames=max_frames,
        target_frames=target_frames,
        selected_indices=selected,
        removal_mask=removal_mask,
        reduced=bool(reduced),
        stats=stats,
    )


def infer_animation_frame_count(animation_obj: dict[str, Any]) -> int:
    if "animation_poses" not in animation_obj:
        raise KeyError("animation object is missing required key: animation_poses")

    poses = np.asarray(animation_obj["animation_poses"])
    if poses.ndim < 1:
        raise ValueError(f"animation_poses must have at least 1 dimension, got {poses.shape}")

    return int(poses.shape[0])


def _slice_first_axis(
    value: Any,
    indices: np.ndarray,
    *,
    key_name: str,
    expected_count: int,
) -> Any:
    arr = np.asarray(value)

    if arr.ndim < 1:
        raise ValueError(f"{key_name} must have at least 1 dimension, got {arr.shape}")

    if int(arr.shape[0]) != int(expected_count):
        raise ValueError(
            f"{key_name} first axis mismatch: expected {expected_count}, got {arr.shape[0]}"
        )

    return arr[indices]


def apply_frame_sample_plan_to_animation(
    animation_obj: dict[str, Any],
    plan: DynamicFrameSamplePlan,
    *,
    copy_obj: bool = True,
) -> dict[str, Any]:
    """
    Apply sampling to one animation_data object.

    Sliced keys:
        animation_poses
        scene_frames, if present

    Unchanged:
        rest_pose
        bone_names
        rest_pose_semantics
        any unknown metadata fields
    """
    out = dict(animation_obj) if copy_obj else animation_obj
    n = int(plan.original_frame_count)
    idx = plan.selected_indices

    out["animation_poses"] = _slice_first_axis(
        out["animation_poses"],
        idx,
        key_name="animation_poses",
        expected_count=n,
    )

    if "scene_frames" in out:
        out["scene_frames"] = _slice_first_axis(
            out["scene_frames"],
            idx,
            key_name="scene_frames",
            expected_count=n,
        )

    out.setdefault("ssdr_frame_sampling", plan.stats)
    return out


def apply_frame_sample_plan_to_mesh(
    mesh_obj: dict[str, Any],
    plan: DynamicFrameSamplePlan,
    *,
    copy_obj: bool = True,
) -> dict[str, Any]:
    """
    Apply sampling to one mesh Fitbox object.

    Sliced keys:
        baked_animation_vertices
        baked_animation_vertex_normals
        scene_frames, if present

    Unchanged:
        vertices
        faces
        vertex_bone_weights
        bone_names
        static metadata
    """
    out = dict(mesh_obj) if copy_obj else mesh_obj
    n = int(plan.original_frame_count)
    idx = plan.selected_indices

    if "baked_animation_vertices" not in out:
        raise KeyError("mesh object is missing required key: baked_animation_vertices")

    if "baked_animation_vertex_normals" not in out:
        raise KeyError("mesh object is missing required key: baked_animation_vertex_normals")

    out["baked_animation_vertices"] = _slice_first_axis(
        out["baked_animation_vertices"],
        idx,
        key_name="baked_animation_vertices",
        expected_count=n,
    )
    out["baked_animation_vertex_normals"] = _slice_first_axis(
        out["baked_animation_vertex_normals"],
        idx,
        key_name="baked_animation_vertex_normals",
        expected_count=n,
    )

    if "scene_frames" in out:
        out["scene_frames"] = _slice_first_axis(
            out["scene_frames"],
            idx,
            key_name="scene_frames",
            expected_count=n,
        )

    out.setdefault("ssdr_frame_sampling", plan.stats)
    return out
