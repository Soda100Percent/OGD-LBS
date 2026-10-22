# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Bone-weight reduction helpers for SSDR preprocessing.

This module migrates the core algorithm from the old SSDR Step 02 script into
a reusable pure NumPy library.

Primary use in migrated scripts/ssdr/02_reduce_bone_weight.py:
    1. Load full skeleton hierarchy.
    2. Load selected reduced skeleton hierarchy.
    3. Build SkeletonContext.
    4. For every mesh:
        full-column vertex_bone_weights -> reduced-column vertex_bone_weights.

No bpy dependency.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


def normalize_parent_indices(parent_indices: list[Any]) -> list[int | None]:
    out: list[int | None] = []

    for value in parent_indices:
        if value is None:
            out.append(None)
        else:
            ivalue = int(value)
            if ivalue < 0:
                out.append(None)
            else:
                out.append(ivalue)

    return out


def require_skeleton_fields(obj: dict[str, Any], *, name: str) -> tuple[list[str], list[int | None]]:
    if "bone_names" not in obj:
        raise KeyError(f"{name} skeleton is missing required key: bone_names")

    if "parent_indices" not in obj:
        raise KeyError(f"{name} skeleton is missing required key: parent_indices")

    bone_names = [str(x) for x in obj["bone_names"]]
    parent_indices = normalize_parent_indices(list(obj["parent_indices"]))

    if len(bone_names) != len(parent_indices):
        raise ValueError(
            f"{name} skeleton length mismatch: "
            f"bone_names={len(bone_names)}, parent_indices={len(parent_indices)}"
        )

    return bone_names, parent_indices


def align_skeleton_hierarchy_to_bone_names(
    skeleton_obj: dict[str, Any],
    target_bone_names: list[str],
) -> dict[str, Any]:
    """
    Reorder skeleton_hierarchy_data to match a target bone_names order.

    This preserves old Step 01 behavior where skeleton hierarchy was aligned to
    animation_data["bone_names"] when the two orders differed.
    """
    src_bone_names, src_parent_indices = require_skeleton_fields(
        skeleton_obj,
        name="source",
    )
    target_bone_names = [str(x) for x in target_bone_names]

    missing = [name for name in target_bone_names if name not in src_bone_names]
    if missing:
        raise ValueError(
            f"Cannot align skeleton hierarchy; target bone names missing in source skeleton: {missing}"
        )

    src_name_to_idx = {name: i for i, name in enumerate(src_bone_names)}
    target_name_to_idx = {name: i for i, name in enumerate(target_bone_names)}

    aligned_parent_indices: list[int | None] = []

    for bone_name in target_bone_names:
        src_idx = src_name_to_idx[bone_name]
        src_parent_idx = src_parent_indices[src_idx]

        if src_parent_idx is None:
            aligned_parent_indices.append(None)
            continue

        parent_name = src_bone_names[src_parent_idx]
        aligned_parent_indices.append(target_name_to_idx.get(parent_name, None))

    out = dict(skeleton_obj)
    out["bone_names"] = target_bone_names
    out["parent_indices"] = aligned_parent_indices
    return out


@dataclass
class WeightReductionStats:
    mesh_name: str
    full_bone_count: int
    reduced_bone_count: int
    vertex_count: int
    invalid_affinity_count: int
    nonzero_input_rows: int
    nonzero_output_rows: int


class SkeletonContext:
    """
    Mapping context between a full skeleton and a reduced skeleton.

    The reduced skeleton is expected to contain a subset of full skeleton bone
    names. Removed full bones transfer their weights to retained parent / self
    alternatives / retained children, following the legacy SSDR Step 02 logic.
    """

    def __init__(self, skelhier_obj: dict[str, Any], skelhier_reduced_obj: dict[str, Any]):
        self.sh_bone_names, self.sh_par_idxs = require_skeleton_fields(
            skelhier_obj,
            name="full",
        )
        self.sh_bones_count = len(self.sh_par_idxs)
        self.sh_child_idxs = [
            self._find_child_idxs(self.sh_par_idxs, b_id)
            for b_id in range(self.sh_bones_count)
        ]

        self.sh_br_bone_names, self.sh_br_par_idxs = require_skeleton_fields(
            skelhier_reduced_obj,
            name="reduced",
        )
        self.sh_br_bones_count = len(self.sh_br_par_idxs)
        self.sh_br_child_idxs = [
            self._find_child_idxs(self.sh_br_par_idxs, b_id)
            for b_id in range(self.sh_br_bones_count)
        ]

        missing_reduced = [
            b_name
            for b_name in self.sh_br_bone_names
            if b_name not in self.sh_bone_names
        ]
        if missing_reduced:
            raise ValueError(
                "Reduced skeleton contains bones not present in full skeleton: "
                f"{missing_reduced}"
            )

        self.sh_idxs_br2full = np.array(
            [self.sh_bone_names.index(b_name) for b_name in self.sh_br_bone_names],
            dtype=np.int64,
        )
        self.sh_idxs_full2br = [
            self.sh_br_bone_names.index(b_name)
            if b_name in self.sh_br_bone_names
            else -1
            for b_name in self.sh_bone_names
        ]

        self.sh_retained_par_idxs = [
            self._find_sh_retained_par_idx(b_id)
            for b_id in range(self.sh_bones_count)
        ]
        self.sh_retained_child_idxs = [
            self._find_sh_retained_child_idxs(b_id)
            for b_id in range(self.sh_bones_count)
        ]

        self.sh_branch_bone_flags = [
            self._is_branch_bone(self.sh_par_idxs, b_id)
            for b_id in range(self.sh_bones_count)
        ]
        self.sh_branch_ids = [
            self._calc_bone_branch_id(self.sh_par_idxs, self.sh_branch_bone_flags, b_id)
            for b_id in range(self.sh_bones_count)
        ]

        self.sh_bones_by_branch_id: dict[tuple[int | None, int], set[int]] = {}
        for b_id, branch_id in enumerate(self.sh_branch_ids):
            self.sh_bones_by_branch_id.setdefault(branch_id, set()).add(b_id)

        self.sh_alternative_par_idxs: list[set[int]] = [
            set()
            for _ in range(self.sh_bones_count)
        ]
        self.sh_alternative_bone_idxs: list[set[int]] = [
            set()
            for _ in range(self.sh_bones_count)
        ]
        self.sh_alternative_child_idxs: list[set[int]] = [
            set()
            for _ in range(self.sh_bones_count)
        ]

        for b_id in range(self.sh_bones_count):
            self._add_to_alternative(b_id, b_id)

            par_idx = self.sh_retained_par_idxs[b_id]
            child_idxs = self.sh_retained_child_idxs[b_id]

            if (
                self.sh_idxs_full2br[b_id] < 0
                and par_idx is not None
                and self.sh_branch_bone_flags[par_idx]
                and len(child_idxs) == 0
            ):
                branch_id = self.sh_branch_ids[b_id]
                branch_id_cluster = self.sh_bones_by_branch_id[branch_id]

                for sibling_id in branch_id_cluster:
                    if sibling_id == b_id:
                        continue
                    self._add_to_alternative(sibling_id, b_id)

    def _find_child_idxs(self, par_idxs: list[int | None], idx: int) -> list[int]:
        return [i for i, par in enumerate(par_idxs) if par == idx]

    def _find_sh_retained_idx_upward(self, sh_idx: int | None) -> int | None:
        if sh_idx is None:
            return None

        if self.sh_idxs_full2br[int(sh_idx)] >= 0:
            return int(sh_idx)

        return self._find_sh_retained_idx_upward(self.sh_par_idxs[int(sh_idx)])

    def _find_sh_retained_idxs_downward(self, sh_idx: int) -> set[int]:
        if self.sh_idxs_full2br[int(sh_idx)] >= 0:
            return {int(sh_idx)}

        children = self.sh_child_idxs[int(sh_idx)]
        result_set: set[int] = set()

        for child_idx in children:
            result_set.update(self._find_sh_retained_idxs_downward(child_idx))

        return result_set

    def _find_sh_retained_par_idx(self, sh_idx: int) -> int | None:
        return self._find_sh_retained_idx_upward(self.sh_par_idxs[int(sh_idx)])

    def _find_sh_retained_child_idxs(self, sh_idx: int) -> list[int]:
        children = self.sh_child_idxs[int(sh_idx)]
        result_set: set[int] = set()

        for child_idx in children:
            result_set.update(self._find_sh_retained_idxs_downward(child_idx))

        return sorted(result_set)

    def _is_branch_bone(self, par_idxs: list[int | None], idx: int) -> bool:
        return par_idxs.count(int(idx)) > 1

    def _calc_bone_branch_id(
        self,
        par_idxs: list[int | None],
        branch_bone_flags: list[bool],
        idx: int,
    ) -> tuple[int | None, int]:
        depth = 1
        temp_idx = par_idxs[int(idx)]

        while temp_idx is not None and not branch_bone_flags[int(temp_idx)]:
            depth += 1
            temp_idx = par_idxs[int(temp_idx)]

        return temp_idx, depth

    def _add_to_alternative(self, source_id: int, target_id: int) -> None:
        source_id = int(source_id)
        target_id = int(target_id)

        if self.sh_idxs_full2br[source_id] < 0:
            source_retained_par_idx = self.sh_retained_par_idxs[source_id]
            if source_retained_par_idx is not None:
                self.sh_alternative_par_idxs[target_id].add(int(source_retained_par_idx))

            self.sh_alternative_child_idxs[target_id].update(
                int(x)
                for x in self.sh_retained_child_idxs[source_id]
            )
        else:
            self.sh_alternative_bone_idxs[target_id].add(source_id)


def compute_weights_matrix(
    vbw_full_order: np.ndarray,
    bone_candidates: set[int],
    ctx: SkeletonContext,
    *,
    par_weight: float = 0.5,
    bone_weight: float = 0.3,
    child_weight: float = 0.2,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Convert full-skeleton weights to reduced-skeleton weights.

    Args:
        vbw_full_order:
            Vertex weights ordered by ctx.sh_bone_names, shape V x B_full.
        bone_candidates:
            Full-skeleton bone indices that have nonzero input weights.
        ctx:
            SkeletonContext.
        par_weight / bone_weight / child_weight:
            Legacy redistribution weights.

    Returns:
        Reduced weights ordered by ctx.sh_br_bone_names, shape V x B_reduced.
    """
    vbw = np.asarray(vbw_full_order, dtype=np.float64)

    if vbw.ndim != 2:
        raise ValueError(f"vbw_full_order must be 2D, got shape={vbw.shape}")

    if vbw.shape[1] != ctx.sh_bones_count:
        raise ValueError(
            f"vbw full bone axis mismatch: expected {ctx.sh_bones_count}, got {vbw.shape[1]}"
        )

    b_full = int(vbw.shape[1])
    transform = np.eye(b_full, dtype=np.float64)

    candidates = {int(x) for x in bone_candidates}

    for b_id in range(b_full):
        if ctx.sh_idxs_full2br[b_id] >= 0:
            continue

        transform[b_id, b_id] = 0.0

        if b_id not in candidates:
            continue

        alt_par = np.array(
            sorted(ctx.sh_alternative_par_idxs[b_id].intersection(candidates)),
            dtype=np.int64,
        )
        alt_bone = np.array(
            sorted(ctx.sh_alternative_bone_idxs[b_id].intersection(candidates)),
            dtype=np.int64,
        )
        alt_child = np.array(
            sorted(ctx.sh_alternative_child_idxs[b_id].intersection(candidates)),
            dtype=np.int64,
        )

        if alt_par.size + alt_bone.size + alt_child.size == 0:
            alt_par = np.array(sorted(ctx.sh_alternative_par_idxs[b_id]), dtype=np.int64)
            alt_bone = np.array(sorted(ctx.sh_alternative_bone_idxs[b_id]), dtype=np.int64)
            alt_child = np.array(sorted(ctx.sh_alternative_child_idxs[b_id]), dtype=np.int64)

        if alt_par.size > 0:
            transform[b_id, alt_par] += float(par_weight) / (float(alt_par.size) + epsilon)

        if alt_bone.size > 0:
            transform[b_id, alt_bone] += float(bone_weight) / (float(alt_bone.size) + epsilon)

        if alt_child.size > 0:
            transform[b_id, alt_child] += float(child_weight) / (float(alt_child.size) + epsilon)

    vbw_new_full = vbw @ transform

    row_sums = vbw_new_full.sum(axis=1, keepdims=True)
    mask = row_sums[:, 0] > epsilon

    vbw_new_full[mask] /= row_sums[mask]

    return vbw_new_full[:, ctx.sh_idxs_br2full]


def reorder_mesh_weights_to_full_skeleton(
    *,
    mesh_weights: np.ndarray,
    mesh_bone_names: list[str],
    full_bone_names: list[str],
) -> np.ndarray:
    weights = np.asarray(mesh_weights, dtype=np.float64)

    if weights.ndim != 2:
        raise ValueError(f"mesh_weights must be 2D, got shape={weights.shape}")

    mesh_bone_names = [str(x) for x in mesh_bone_names]
    full_bone_names = [str(x) for x in full_bone_names]

    missing = [name for name in full_bone_names if name not in mesh_bone_names]
    if missing:
        raise ValueError(
            f"Mesh weights are missing columns for full skeleton bones: {missing[:20]}"
        )

    indices = np.array(
        [mesh_bone_names.index(name) for name in full_bone_names],
        dtype=np.int64,
    )

    if weights.shape[1] < len(mesh_bone_names):
        raise ValueError(
            f"mesh weight column count {weights.shape[1]} is smaller than "
            f"mesh_bone_names count {len(mesh_bone_names)}"
        )

    return weights[:, indices]


def reduce_mesh_weights(
    mesh_obj: dict[str, Any],
    ctx: SkeletonContext,
    *,
    par_weight: float = 0.5,
    bone_weight: float = 0.3,
    child_weight: float = 0.2,
    affinity_eps: float = 1e-5,
    copy_obj: bool = True,
) -> tuple[dict[str, Any], WeightReductionStats]:
    """
    Reduce one Fitbox mesh object's vertex_bone_weights to ctx.sh_br_bone_names.
    """
    if "vertex_bone_weights" not in mesh_obj:
        raise KeyError("mesh object is missing required key: vertex_bone_weights")

    if "bone_names" not in mesh_obj:
        raise KeyError("mesh object is missing required key: bone_names")

    name = str(mesh_obj.get("name", "<unnamed_mesh>"))

    vbw_orig = np.asarray(mesh_obj["vertex_bone_weights"], dtype=np.float64)
    mesh_bone_names = [str(x) for x in mesh_obj["bone_names"]]

    vbw_full_order = reorder_mesh_weights_to_full_skeleton(
        mesh_weights=vbw_orig,
        mesh_bone_names=mesh_bone_names,
        full_bone_names=ctx.sh_bone_names,
    )

    bone_candidates = set(int(x) for x in np.where(vbw_full_order.any(axis=0))[0])

    vbw_reduced = compute_weights_matrix(
        vbw_full_order,
        bone_candidates,
        ctx,
        par_weight=par_weight,
        bone_weight=bone_weight,
        child_weight=child_weight,
    )

    input_row_sums = vbw_full_order.sum(axis=1)
    output_row_sums = vbw_reduced.sum(axis=1)

    nonzero_input_rows = int(np.count_nonzero(input_row_sums > affinity_eps))
    nonzero_output_rows = int(np.count_nonzero(output_row_sums > affinity_eps))

    invalid_mask = (
        (input_row_sums > affinity_eps)
        & (np.abs(output_row_sums - 1.0) > affinity_eps)
    )
    invalid_count = int(np.count_nonzero(invalid_mask))

    out = dict(mesh_obj) if copy_obj else mesh_obj
    out["vertex_bone_weights"] = vbw_reduced
    out["bone_names"] = list(ctx.sh_br_bone_names)

    stats = WeightReductionStats(
        mesh_name=name,
        full_bone_count=ctx.sh_bones_count,
        reduced_bone_count=ctx.sh_br_bones_count,
        vertex_count=int(vbw_reduced.shape[0]),
        invalid_affinity_count=invalid_count,
        nonzero_input_rows=nonzero_input_rows,
        nonzero_output_rows=nonzero_output_rows,
    )

    return out, stats
