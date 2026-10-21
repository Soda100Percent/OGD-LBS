# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Frame sampling utilities migrated from the SSDR preprocessing step.

The old SSDR script first exported all frames, then removed many frames from
animation_poses and baked mesh animations. Here we compute the retained scene
frames before Blender baking.

Index-range convention:
    - removal_idx_ranges use candidate-frame indices, not Blender scene frame IDs.
    - ranges are 0-based and left-closed, right-open: start:end.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class FrameSamplingResult:
    candidate_frames: list[int]
    retained_frames_before_debug: list[int]
    retained_frames: list[int]
    removal_mask_before_debug: list[bool]
    explicit_removed_count: int
    rate_removed_count: int
    debug_truncated_count: int
    stats: dict


def calc_mask_idx_ranges(
    mask: np.ndarray,
    idx_ranges: set[tuple[int, int]],
) -> np.ndarray:
    n = int(mask.shape[0])

    for start, end in sorted(idx_ranges):
        if start < 0 or end < 0:
            raise ValueError(f"Removal range must be non-negative: {start}:{end}")
        if end < start:
            raise ValueError(f"Removal range end must be >= start: {start}:{end}")
        if end > n:
            raise ValueError(
                f"Removal range {start}:{end} exceeds candidate frame count {n}."
            )

        mask[start:end] = True

    return mask


def calc_idxs_by_rate(item_count: int, rate: float, bias: float = 0.0) -> list[int]:
    if item_count < 0:
        raise ValueError(f"item_count must be non-negative, got {item_count}")

    if rate < 0.0 or rate >= 1.0:
        raise ValueError(
            f"removal_rate must be in [0, 1). Use 0.0 to keep all frames. Got {rate}."
        )

    n = int(item_count * rate)

    if n == 0:
        return []

    step = item_count / n
    idxs = [int(i * step + bias) for i in range(n)]

    bad = [idx for idx in idxs if idx < 0 or idx >= item_count]
    if bad:
        raise ValueError(
            f"Calculated removal indices out of bounds for item_count={item_count}: "
            f"{bad[:10]} (rate={rate}, bias={bias})"
        )

    return idxs


def calc_mask_by_rate(
    mask: np.ndarray,
    rate: float,
    bias: float = 0.0,
) -> np.ndarray:
    idxs = calc_idxs_by_rate(len(mask), rate, bias)
    mask[idxs] = True
    return mask


def calc_mask_frames(
    mask: np.ndarray,
    idx_ranges: set[tuple[int, int]],
    rate: float,
    bias: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """
    Reproduce the old SSDR frame-removal semantics:

        1. remove explicit index ranges
        2. remove rate-based indices from the remaining items

    Returns:
        removal_mask, explicit_removed_count, rate_removed_count
    """
    before_explicit = mask.copy()
    mask = calc_mask_idx_ranges(mask, idx_ranges)
    explicit_removed_count = int(np.count_nonzero(mask & ~before_explicit))

    idxs_submask_to_mask = np.arange(mask.shape[0])[~mask]
    submask = mask[~mask]

    before_rate = mask.copy()
    mask[idxs_submask_to_mask] = calc_mask_by_rate(submask, rate, bias)

    rate_removed_count = int(np.count_nonzero(mask & ~before_rate))
    return mask, explicit_removed_count, rate_removed_count


def resolve_sampled_frames(
    *,
    frame_start: int,
    frame_end_exclusive: int,
    removal_rate: float = 0.9,
    removal_bias: float = 1.0,
    removal_idx_ranges: set[tuple[int, int]] | None = None,
    debug_frame_max: int = -1,
) -> FrameSamplingResult:
    if frame_end_exclusive <= frame_start:
        raise ValueError(
            f"Invalid frame range: start={frame_start}, end_exclusive={frame_end_exclusive}"
        )

    candidate_frames = list(range(int(frame_start), int(frame_end_exclusive)))
    candidate_count = len(candidate_frames)

    idx_ranges = set(removal_idx_ranges or set())
    removal_mask = np.zeros(candidate_count, dtype=bool)

    removal_mask, explicit_removed_count, rate_removed_count = calc_mask_frames(
        removal_mask,
        idx_ranges,
        float(removal_rate),
        float(removal_bias),
    )

    retained_before_debug = [
        frame for frame, removed in zip(candidate_frames, removal_mask.tolist())
        if not removed
    ]

    if not retained_before_debug:
        raise RuntimeError(
            "Frame sampling produced zero retained frames before debug truncation. "
            f"candidate_count={candidate_count}, removal_rate={removal_rate}, "
            f"removal_idx_ranges={sorted(idx_ranges)}"
        )

    if int(debug_frame_max) > 0:
        retained_frames = retained_before_debug[: int(debug_frame_max)]
    else:
        retained_frames = list(retained_before_debug)

    if not retained_frames:
        raise RuntimeError(
            "Frame sampling produced zero retained frames after debug truncation. "
            f"debug_frame_max={debug_frame_max}"
        )

    debug_truncated_count = len(retained_before_debug) - len(retained_frames)

    stats = {
        "candidate_frame_start": int(frame_start),
        "candidate_frame_end_exclusive": int(frame_end_exclusive),
        "candidate_frame_count": int(candidate_count),
        "removal_rate": float(removal_rate),
        "removal_bias": float(removal_bias),
        "removal_idx_ranges": [list(x) for x in sorted(idx_ranges)],
        "explicit_removed_count": int(explicit_removed_count),
        "rate_removed_count": int(rate_removed_count),
        "retained_before_debug_count": int(len(retained_before_debug)),
        "debug_frame_max": int(debug_frame_max),
        "debug_truncated_count": int(debug_truncated_count),
        "final_retained_count": int(len(retained_frames)),
        "retained_scene_frames": [int(x) for x in retained_frames],
        "rest_pose_semantics": (
            "rest_pose is exported separately from armature.data.bones[*].matrix_local; "
            "animation_poses[0] corresponds to retained_scene_frames[0], not rest_pose."
        ),
    }

    return FrameSamplingResult(
        candidate_frames=[int(x) for x in candidate_frames],
        retained_frames_before_debug=[int(x) for x in retained_before_debug],
        retained_frames=[int(x) for x in retained_frames],
        removal_mask_before_debug=[bool(x) for x in removal_mask.tolist()],
        explicit_removed_count=int(explicit_removed_count),
        rate_removed_count=int(rate_removed_count),
        debug_truncated_count=int(debug_truncated_count),
        stats=stats,
    )
