# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Frame sampling utilities migrated from the SSDR preprocessing step.

Index-range convention:
    - removal_idx_ranges use candidate-frame indices, not Blender scene frame IDs.
    - ranges are 0-based and left-closed, right-open: start:end.

Sampling priority:
    1. Candidate frame range is resolved from Blender scene/NLA frames.
    2. Explicit removal_idx_ranges are applied first.
    3. If target_frame_count > 0, uniformly retain that many frames from the
       remaining candidates. This has higher priority than removal_rate.
    4. Otherwise, use legacy removal_rate / removal_bias semantics.
    5. debug_frame_max is applied last as a debug truncation.
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
    target_removed_count: int
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


def calc_idxs_by_target_count(item_count: int, target_count: int) -> list[int]:
    """
    Select exactly target_count indices uniformly from [0, item_count).

    If target_count >= item_count, keep all indices.
    """
    item_count = int(item_count)
    target_count = int(target_count)

    if item_count < 0:
        raise ValueError(f"item_count must be non-negative, got {item_count}")

    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")

    if item_count == 0:
        return []

    if target_count >= item_count:
        return list(range(item_count))

    if target_count == 1:
        return [0]

    raw = np.rint(
        np.linspace(0, item_count - 1, num=target_count)
    ).astype(np.int64)

    out: list[int] = []
    used: set[int] = set()

    for idx_raw in raw.tolist():
        idx = max(0, min(int(idx_raw), item_count - 1))
        if idx not in used:
            out.append(idx)
            used.add(idx)

    # Defensive fill. Normally linspace+rint is already unique for
    # target_count <= item_count, but keep exact-count semantics explicit.
    if len(out) < target_count:
        for idx in range(item_count):
            if idx in used:
                continue
            out.append(idx)
            used.add(idx)
            if len(out) >= target_count:
                break

    out = sorted(out[:target_count])

    if len(out) != target_count:
        raise RuntimeError(
            f"Could not select exact target_count indices: "
            f"item_count={item_count}, target_count={target_count}, got={len(out)}"
        )

    return out


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


def calc_mask_frames_by_target_count(
    mask: np.ndarray,
    idx_ranges: set[tuple[int, int]],
    target_frame_count: int,
) -> tuple[np.ndarray, int, int]:
    """
    Apply explicit index-range removal, then uniformly keep target_frame_count
    frames from the remaining candidates.

    Returns:
        removal_mask, explicit_removed_count, target_removed_count
    """
    before_explicit = mask.copy()
    mask = calc_mask_idx_ranges(mask, idx_ranges)
    explicit_removed_count = int(np.count_nonzero(mask & ~before_explicit))

    remaining_candidate_indices = np.arange(mask.shape[0])[~mask]

    if remaining_candidate_indices.size <= 0:
        raise RuntimeError(
            "Explicit removal ranges removed all candidate frames before target-frame sampling."
        )

    selected_sub_indices = calc_idxs_by_target_count(
        int(remaining_candidate_indices.size),
        int(target_frame_count),
    )
    selected_global = set(remaining_candidate_indices[selected_sub_indices].tolist())

    before_target = mask.copy()
    for idx in remaining_candidate_indices.tolist():
        if int(idx) not in selected_global:
            mask[int(idx)] = True

    target_removed_count = int(np.count_nonzero(mask & ~before_target))
    return mask, explicit_removed_count, target_removed_count


def resolve_sampled_frames(
    *,
    frame_start: int,
    frame_end_exclusive: int,
    removal_rate: float = 0.9,
    removal_bias: float = 1.0,
    removal_idx_ranges: set[tuple[int, int]] | None = None,
    target_frame_count: int = -1,
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

    target_frame_count_i = int(target_frame_count)

    if target_frame_count_i > 0:
        removal_mask, explicit_removed_count, target_removed_count = calc_mask_frames_by_target_count(
            removal_mask,
            idx_ranges,
            target_frame_count_i,
        )
        rate_removed_count = 0
        sampling_mode = "target_frame_count"
    else:
        removal_mask, explicit_removed_count, rate_removed_count = calc_mask_frames(
            removal_mask,
            idx_ranges,
            float(removal_rate),
            float(removal_bias),
        )
        target_removed_count = 0
        sampling_mode = "removal_rate"

    retained_before_debug = [
        frame for frame, removed in zip(candidate_frames, removal_mask.tolist())
        if not removed
    ]

    if not retained_before_debug:
        raise RuntimeError(
            "Frame sampling produced zero retained frames before debug truncation. "
            f"candidate_count={candidate_count}, removal_rate={removal_rate}, "
            f"target_frame_count={target_frame_count_i}, "
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
        "sampling_mode": sampling_mode,
        "candidate_frame_start": int(frame_start),
        "candidate_frame_end_exclusive": int(frame_end_exclusive),
        "candidate_frame_count": int(candidate_count),
        "removal_rate": float(removal_rate),
        "removal_bias": float(removal_bias),
        "target_frame_count": int(target_frame_count_i),
        "target_frame_count_enabled": bool(target_frame_count_i > 0),
        "removal_idx_ranges": [list(x) for x in sorted(idx_ranges)],
        "explicit_removed_count": int(explicit_removed_count),
        "rate_removed_count": int(rate_removed_count),
        "target_removed_count": int(target_removed_count),
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
        target_removed_count=int(target_removed_count),
        debug_truncated_count=int(debug_truncated_count),
        stats=stats,
    )
