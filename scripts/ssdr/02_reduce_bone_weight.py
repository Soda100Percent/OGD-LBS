#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSDR Step 02: Reduce mesh bone weights from full skeleton to a selected reduced skeleton.

Default pipeline semantics:
    --input_dir <work_dir>
        reads : <work_dir>/data1
        writes: <work_dir>/data2

    --input_dir <work_dir>/data1
        reads : <work_dir>/data1
        writes: <work_dir>/data2

Explicit automation mode:
    --input_dir <custom_input_tree> --output_dir <custom_output_tree>
        reads : <custom_input_tree>
        writes: <custom_output_tree>

Supported input layouts are intentionally limited to two forms.

1. New SSDR / fitbox_vdb-root layout:

    data1/
    ├── skeleton_hierarchy_data.pkl
    ├── animation_data.pkl
    ├── main_mesh_VDB.pkl
    └── bone_reduced_versions/
        ├── cleaned_full/
        │   ├── skeleton_hierarchy_data.pkl
        │   └── animation_data.pkl
        ├── bone_reduced_001/
        │   ├── skeleton_hierarchy_data.pkl
        │   └── animation_data.pkl
        └── ...

This is the expected result when data_prep exports:

    fitbox_exports/fitbox_vdb/

and SSDR Step 00 receives that fitbox_vdb directory as data0_json/input tree.

2. Legacy flat layout:

    data1/
    ├── skeleton_hierarchy_data.pkl
    ├── animation_data.pkl
    ├── skeleton_hierarchy_data_bone_reduced.pkl
    ├── animation_data_bone_reduced.pkl
    └── <mesh>.pkl

Unsupported on purpose:
    data1/fitbox_vdb/... wrapper layout.

For new layout, use fitbox_vdb itself as the SSDR input tree.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from _env import ensure_repo_paths_in_syspath
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from ._env import ensure_repo_paths_in_syspath

ensure_repo_paths_in_syspath(__file__)

try:
    from _stage_io import (
        load_required_dict,
        prepare_output_dir,
        require_file,
        write_stage_summary,
    )
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from ._stage_io import (
        load_required_dict,
        prepare_output_dir,
        require_file,
        write_stage_summary,
    )



import numpy as np  # noqa: E402

from ogd_lbs.utils.common import (  # noqa: E402
    log,
    warn,
    fail,
    script_argv,
    Timer,
    load_data,
    dump_pkl,
    record_metric,
    dump_metrics,
)
from ogd_lbs.ssdr.weight_reduction import (  # noqa: E402
    SkeletonContext,
    align_skeleton_hierarchy_to_bone_names,
    reduce_mesh_weights,
)
from paths import (  # noqa: E402
    resolve_ssdr_stage_dirs,
    validate_ssdr_stage_resolution,
    print_ssdr_resolution_info,
)


FULL_SKELETON = "skeleton_hierarchy_data.pkl"
FULL_ANIMATION = "animation_data.pkl"
LEGACY_REDUCED_SKELETON = "skeleton_hierarchy_data_bone_reduced.pkl"
LEGACY_REDUCED_ANIMATION = "animation_data_bone_reduced.pkl"

REDUCED_ROOT = "bone_reduced_versions"

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
}


@dataclass(frozen=True)
class SelectedDatasetLayout:
    layout_kind: str
    variant_name: str

    full_root: Path
    reduced_root: Path | None

    full_skeleton_path: Path
    full_animation_path: Path
    reduced_skeleton_path: Path
    reduced_animation_path: Path

    mesh_paths: list[Path]


def has_legacy_flat_layout(input_tree: Path) -> bool:
    return all(
        path.is_file()
        for path in [
            input_tree / FULL_SKELETON,
            input_tree / FULL_ANIMATION,
            input_tree / LEGACY_REDUCED_SKELETON,
            input_tree / LEGACY_REDUCED_ANIMATION,
        ]
    )


def has_new_vdb_root_layout(input_tree: Path) -> bool:
    return (
        (input_tree / FULL_SKELETON).is_file()
        and (input_tree / FULL_ANIMATION).is_file()
        and (input_tree / REDUCED_ROOT).is_dir()
    )


def variant_sort_key(path: Path) -> tuple[int, str]:
    """
    Sort variants lexicographically, but put cleaned_full last.

    If cleaned_full is the only valid variant, it will still be selected.
    """
    return (1 if path.name == "cleaned_full" else 0, path.name)


def discover_reduced_variant_dirs(input_tree: Path) -> list[Path]:
    root = input_tree / REDUCED_ROOT

    if not root.exists() or not root.is_dir():
        return []

    variants: list[Path] = []

    for item in root.iterdir():
        if not item.is_dir():
            continue

        if (item / FULL_SKELETON).is_file() and (item / FULL_ANIMATION).is_file():
            variants.append(item)

    return sorted(variants, key=variant_sort_key)


def should_skip_mesh_candidate(path: Path, *, input_tree: Path, layout_kind: str) -> bool:
    if path.stem in CORE_STEMS or path.stem in SUMMARY_STEMS:
        return True

    if layout_kind == "new_vdb_root":
        rel = path.relative_to(input_tree)
        if rel.parts and rel.parts[0] == REDUCED_ROOT:
            return True

    return False


def discover_mesh_paths(input_tree: Path, *, layout_kind: str) -> list[Path]:
    """
    Discover mesh pkl candidates.

    For both supported layouts, output data2 is flat. To avoid accidental
    collisions and ambiguity, only pkl files under the input tree are scanned,
    while core files and reduced variant contents are excluded.
    """
    paths: list[Path] = []

    for path in sorted(input_tree.rglob("*.pkl"), key=lambda p: str(p.relative_to(input_tree))):
        if not path.is_file():
            continue

        if should_skip_mesh_candidate(path, input_tree=input_tree, layout_kind=layout_kind):
            continue

        paths.append(path)

    return paths


def validate_flat_output_name_collisions(mesh_paths: list[Path]) -> None:
    by_name: dict[str, list[Path]] = {}

    for path in mesh_paths:
        by_name.setdefault(path.name, []).append(path)

    collisions = {name: paths for name, paths in by_name.items() if len(paths) > 1}

    if collisions:
        lines = []
        for name, paths in sorted(collisions.items()):
            lines.append(f"{name}: {[str(p) for p in paths]}")
        fail(
            "Multiple mesh files would collide in flat data2 output:\n"
            + "\n".join(lines)
        )



def _append_unique_bone_name(out: list[str], seen: set[str], name: str) -> None:
    name = str(name).strip()
    if not name:
        return
    if name in seen:
        return
    seen.add(name)
    out.append(name)


def _parse_bone_names_text(text: str | None) -> list[str]:
    """
    Parse a bone-name list from comma-separated and/or newline-separated text.

    Comments are supported with '#'. Example:

        Bip01_Spine1
        Bip01_L_Finger0, Bip01_R_Finger0
        # ignored
    """
    if text is None:
        return []

    out: list[str] = []
    seen: set[str] = set()

    normalized = str(text).replace(",", "\n")
    for raw_line in normalized.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        _append_unique_bone_name(out, seen, line)

    return out


def load_extra_remove_bone_names(
    *,
    inline_text: str | None,
    file_path: str | None,
) -> list[str]:
    """
    Load extra bone removals from CLI text and/or a text file.

    The returned list preserves first occurrence order and removes duplicates.
    """
    out: list[str] = []
    seen: set[str] = set()

    for name in _parse_bone_names_text(inline_text):
        _append_unique_bone_name(out, seen, name)

    if file_path is not None and str(file_path).strip():
        path = Path(file_path).expanduser().resolve(strict=False)
        if not path.is_file():
            fail(f"--extra_remove_bones_file does not exist: {path}")

        for name in _parse_bone_names_text(path.read_text(encoding="utf-8")):
            _append_unique_bone_name(out, seen, name)

    return out


def prune_skeleton_by_removing_bones(
    skeleton_obj: dict[str, Any],
    remove_bone_names: list[str],
    *,
    missing_ok: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Remove named bones from a skeleton and repair parent_indices.

    Parent repair policy:
        If a kept bone's direct parent is removed, climb upward until the
        nearest kept ancestor is found. If none exists, parent becomes None.
    """
    bone_names = [str(x) for x in skeleton_obj.get("bone_names", [])]
    raw_parent_indices = list(skeleton_obj.get("parent_indices", []))

    if len(bone_names) != len(raw_parent_indices):
        fail(
            "Cannot extra-remove bones from skeleton with mismatched fields: "
            f"bone_names={len(bone_names)}, parent_indices={len(raw_parent_indices)}"
        )

    parent_indices: list[int | None] = []
    for value in raw_parent_indices:
        if value is None:
            parent_indices.append(None)
        else:
            idx = int(value)
            parent_indices.append(idx if 0 <= idx < len(bone_names) else None)

    requested: list[str] = []
    seen: set[str] = set()
    for name in remove_bone_names:
        _append_unique_bone_name(requested, seen, name)

    bone_set = set(bone_names)
    missing = [name for name in requested if name not in bone_set]
    removed = [name for name in requested if name in bone_set]

    if missing and not bool(missing_ok):
        fail(
            "Requested extra-remove bones are missing from selected base variant: "
            f"{missing}"
        )

    remove_set = set(removed)
    kept_old_indices = [
        idx
        for idx, name in enumerate(bone_names)
        if name not in remove_set
    ]

    if not kept_old_indices:
        fail("extra_remove_bones would remove all bones from the selected skeleton.")

    old_to_new = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(kept_old_indices)
    }

    def nearest_kept_parent_new_index(old_idx: int) -> int | None:
        parent = parent_indices[int(old_idx)]
        visited: set[int] = set()

        while parent is not None:
            parent_i = int(parent)
            if parent_i in visited:
                warn(
                    "Cycle detected while repairing parent_indices after extra bone removal; "
                    f"old_idx={old_idx}, parent={parent_i}. Treating as root."
                )
                return None

            visited.add(parent_i)

            if parent_i in old_to_new:
                return int(old_to_new[parent_i])

            if parent_i < 0 or parent_i >= len(parent_indices):
                return None

            parent = parent_indices[parent_i]

        return None

    pruned_bone_names = [bone_names[old_idx] for old_idx in kept_old_indices]
    pruned_parent_indices = [
        nearest_kept_parent_new_index(old_idx)
        for old_idx in kept_old_indices
    ]

    out = dict(skeleton_obj)
    out["bone_names"] = pruned_bone_names
    out["parent_indices"] = pruned_parent_indices

    children_of_removed: dict[str, list[str]] = {}
    for child_idx, parent_idx in enumerate(parent_indices):
        if parent_idx is None:
            continue
        parent_name = bone_names[int(parent_idx)]
        if parent_name in remove_set:
            children_of_removed.setdefault(parent_name, []).append(bone_names[child_idx])

    info = {
        "requested_bones": requested,
        "removed_bones": removed,
        "missing_bones": missing,
        "base_bone_count": int(len(bone_names)),
        "target_bone_count": int(len(pruned_bone_names)),
        "removed_count": int(len(removed)),
        "missing_count": int(len(missing)),
        "parent_repair_mode": "nearest_kept_ancestor",
        "children_of_removed_bones": children_of_removed,
    }

    return out, info


def subset_animation_to_bone_names(
    animation_obj: dict[str, Any],
    target_bone_names: list[str],
    *,
    role: str,
) -> dict[str, Any]:
    """
    Subset animation-side bone arrays to target_bone_names.

    This is mainly used for virtual Step02 variants generated by
    --extra_remove_bones*. build_reduced_animation_output() will still rebuild
    animation_poses from the full animation later, but keeping this object
    internally consistent makes summaries/debugging safer.
    """
    out = dict(animation_obj)

    source_bone_names = [str(x) for x in out.get("bone_names", [])]
    target_bone_names = [str(x) for x in target_bone_names]

    if not source_bone_names:
        fail(f"{role} animation is missing bone_names; cannot apply extra_remove_bones.")

    missing = [name for name in target_bone_names if name not in source_bone_names]
    if missing:
        fail(
            f"{role} animation is missing target bones after extra_remove_bones: "
            f"{missing}"
        )

    indices = np.asarray(
        [source_bone_names.index(name) for name in target_bone_names],
        dtype=np.int64,
    )

    out["bone_names"] = list(target_bone_names)

    if "rest_pose" in out:
        rest_pose = np.asarray(out["rest_pose"])
        if rest_pose.ndim >= 1 and rest_pose.shape[0] == len(source_bone_names):
            out["rest_pose"] = rest_pose[indices, ...]
        else:
            warn(
                f"{role} rest_pose shape is not aligned with bone_names; "
                f"shape={rest_pose.shape}, bone_count={len(source_bone_names)}. "
                "Leaving rest_pose unchanged."
            )

    if "animation_poses" in out:
        poses = np.asarray(out["animation_poses"])
        if poses.ndim >= 2 and poses.shape[1] == len(source_bone_names):
            out["animation_poses"] = poses[:, indices, ...]
        else:
            warn(
                f"{role} animation_poses shape is not aligned with bone_names; "
                f"shape={poses.shape}, bone_count={len(source_bone_names)}. "
                "Leaving animation_poses unchanged; Step02 output animation_poses "
                "will still be rebuilt from the full animation."
            )

    if "fixed_bones" in out:
        fixed = out.get("fixed_bones", [])
        filtered_fixed: list[Any] = []
        target_set = set(target_bone_names)
        target_name_to_idx = {name: i for i, name in enumerate(target_bone_names)}

        if isinstance(fixed, np.ndarray):
            fixed_items = fixed.tolist()
        elif isinstance(fixed, (list, tuple, set)):
            fixed_items = list(fixed)
        else:
            fixed_items = [fixed]

        for item in fixed_items:
            if isinstance(item, str):
                if item in target_set:
                    filtered_fixed.append(item)
                continue

            try:
                src_idx = int(item)
            except Exception:
                continue

            if 0 <= src_idx < len(source_bone_names):
                name = source_bone_names[src_idx]
                if name in target_name_to_idx:
                    filtered_fixed.append(int(target_name_to_idx[name]))

        out["fixed_bones"] = filtered_fixed

    return out


def apply_extra_remove_bones_to_variant(
    *,
    base_variant_name: str,
    reduced_skeleton: dict[str, Any],
    reduced_animation: dict[str, Any],
    extra_remove_bones: list[str],
    extra_variant_name: str | None,
    missing_ok: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """
    Build a virtual reduced variant by deleting extra bones from a selected base variant.
    """
    requested = list(extra_remove_bones or [])

    if not requested:
        metadata = {
            "enabled": False,
            "base_variant_name": base_variant_name,
            "effective_variant_name": base_variant_name,
            "requested_bones": [],
            "removed_bones": [],
            "missing_bones": [],
            "removed_count": 0,
            "missing_count": 0,
            "base_bone_count": None,
            "target_bone_count": None,
            "parent_repair_mode": None,
        }
        return reduced_skeleton, reduced_animation, metadata, base_variant_name

    pruned_skeleton, prune_info = prune_skeleton_by_removing_bones(
        reduced_skeleton,
        requested,
        missing_ok=bool(missing_ok),
    )

    target_bone_names = [str(x) for x in pruned_skeleton.get("bone_names", [])]
    pruned_animation = subset_animation_to_bone_names(
        reduced_animation,
        target_bone_names,
        role=f"variant {base_variant_name!r}",
    )

    if extra_variant_name is not None and str(extra_variant_name).strip():
        effective_variant_name = str(extra_variant_name).strip()
    else:
        effective_variant_name = (
            f"{base_variant_name}_minus_{int(prune_info['removed_count'])}_extra_bones"
        )

    metadata = {
        "enabled": True,
        "base_variant_name": base_variant_name,
        "effective_variant_name": effective_variant_name,
        "missing_ok": bool(missing_ok),
        **prune_info,
    }

    pruned_skeleton = dict(pruned_skeleton)
    pruned_skeleton["ssdr_step02_extra_remove_bones"] = dict(metadata)

    pruned_animation = dict(pruned_animation)
    pruned_animation["ssdr_step02_extra_remove_bones"] = dict(metadata)

    return pruned_skeleton, pruned_animation, metadata, effective_variant_name

def select_dataset_layout(
    *,
    input_tree: Path,
    variant: str | None,
) -> SelectedDatasetLayout:
    """
    Select one of exactly two supported layouts:
        legacy_flat
        new_vdb_root
    """
    variant = str(variant).strip() if variant is not None else ""

    if variant:
        if not has_new_vdb_root_layout(input_tree):
            fail(
                f"--variant={variant!r} requires the new vdb-root layout:\n"
                f"  {input_tree}/skeleton_hierarchy_data.pkl\n"
                f"  {input_tree}/animation_data.pkl\n"
                f"  {input_tree}/bone_reduced_versions/<variant>/..."
            )

        reduced_root = input_tree / REDUCED_ROOT / variant
        require_file(reduced_root / FULL_SKELETON, role=f"variant {variant} skeleton")
        require_file(reduced_root / FULL_ANIMATION, role=f"variant {variant} animation")

        mesh_paths = discover_mesh_paths(input_tree, layout_kind="new_vdb_root")
        validate_flat_output_name_collisions(mesh_paths)

        return SelectedDatasetLayout(
            layout_kind="new_vdb_root_explicit",
            variant_name=variant,
            full_root=input_tree,
            reduced_root=reduced_root,
            full_skeleton_path=input_tree / FULL_SKELETON,
            full_animation_path=input_tree / FULL_ANIMATION,
            reduced_skeleton_path=reduced_root / FULL_SKELETON,
            reduced_animation_path=reduced_root / FULL_ANIMATION,
            mesh_paths=mesh_paths,
        )

    if has_legacy_flat_layout(input_tree):
        mesh_paths = discover_mesh_paths(input_tree, layout_kind="legacy_flat")
        validate_flat_output_name_collisions(mesh_paths)

        return SelectedDatasetLayout(
            layout_kind="legacy_flat",
            variant_name="legacy_bone_reduced",
            full_root=input_tree,
            reduced_root=None,
            full_skeleton_path=input_tree / FULL_SKELETON,
            full_animation_path=input_tree / FULL_ANIMATION,
            reduced_skeleton_path=input_tree / LEGACY_REDUCED_SKELETON,
            reduced_animation_path=input_tree / LEGACY_REDUCED_ANIMATION,
            mesh_paths=mesh_paths,
        )

    if not has_new_vdb_root_layout(input_tree):
        fail(
            "Could not detect supported Step 02 input layout.\n"
            "Supported layouts are exactly:\n"
            "  1) legacy flat layout with skeleton_hierarchy_data_bone_reduced.pkl\n"
            "  2) new vdb-root layout where input_tree itself contains "
            "skeleton_hierarchy_data.pkl, animation_data.pkl, meshes, and bone_reduced_versions/.\n"
            f"Input tree was: {input_tree}"
        )

    variants = discover_reduced_variant_dirs(input_tree)
    if not variants:
        fail(
            f"No valid reduced variant directories found under {input_tree / REDUCED_ROOT}. "
            "Each variant must contain skeleton_hierarchy_data.pkl and animation_data.pkl."
        )

    selected = variants[0]

    mesh_paths = discover_mesh_paths(input_tree, layout_kind="new_vdb_root")
    validate_flat_output_name_collisions(mesh_paths)

    return SelectedDatasetLayout(
        layout_kind="new_vdb_root_auto",
        variant_name=selected.name,
        full_root=input_tree,
        reduced_root=selected,
        full_skeleton_path=input_tree / FULL_SKELETON,
        full_animation_path=input_tree / FULL_ANIMATION,
        reduced_skeleton_path=selected / FULL_SKELETON,
        reduced_animation_path=selected / FULL_ANIMATION,
        mesh_paths=mesh_paths,
    )


def get_bone_names(obj: dict[str, Any], *, role: str) -> list[str]:
    names = obj.get("bone_names")
    if not isinstance(names, list):
        fail(f"{role} missing list field 'bone_names'.")
    return [str(x) for x in names]


def align_skeleton_to_animation_order(
    skeleton_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    skel_names = get_bone_names(skeleton_obj, role=f"{role} skeleton")
    anim_names = get_bone_names(animation_obj, role=f"{role} animation")

    if skel_names == anim_names:
        return dict(skeleton_obj)

    log(f"Aligning {role} skeleton hierarchy to animation bone order.")
    return align_skeleton_hierarchy_to_bone_names(skeleton_obj, anim_names)


def build_reduced_animation_output(
    *,
    full_animation: dict[str, Any],
    reduced_animation: dict[str, Any],
    ctx: SkeletonContext,
    variant_name: str,
) -> dict[str, Any]:
    if "animation_poses" not in full_animation:
        fail("Full animation object missing required key: animation_poses")

    full_poses = np.asarray(full_animation["animation_poses"])

    if full_poses.ndim != 4 or full_poses.shape[2:] != (4, 4):
        fail(
            "Full animation_poses must have shape (frames, bones, 4, 4), "
            f"got {full_poses.shape}"
        )

    if full_poses.shape[1] != ctx.sh_bones_count:
        fail(
            f"Full animation bone axis mismatch: expected {ctx.sh_bones_count}, "
            f"got {full_poses.shape[1]}"
        )

    if "animation_poses" in reduced_animation:
        reduced_poses = np.asarray(reduced_animation["animation_poses"])
        if reduced_poses.ndim >= 1 and reduced_poses.shape[0] != full_poses.shape[0]:
            fail(
                "Reduced animation frame count mismatch: "
                f"full={full_poses.shape[0]}, reduced={reduced_poses.shape[0]}"
            )

    out = dict(reduced_animation)
    out["bone_names"] = list(ctx.sh_br_bone_names)
    out["animation_poses"] = full_poses[:, ctx.sh_idxs_br2full, :, :]

    if "scene_frames" in full_animation:
        out["scene_frames"] = np.asarray(full_animation["scene_frames"])
    elif "scene_frames" in reduced_animation:
        out["scene_frames"] = np.asarray(reduced_animation["scene_frames"])

    if "rest_pose" in out:
        rest_pose = np.asarray(out["rest_pose"])
        if rest_pose.ndim >= 1 and rest_pose.shape[0] != ctx.sh_br_bones_count:
            warn(
                f"Reduced rest_pose bone count mismatch: expected {ctx.sh_br_bones_count}, "
                f"got {rest_pose.shape[0]}. Trying to subset full rest_pose instead."
            )
            if "rest_pose" in full_animation:
                full_rest = np.asarray(full_animation["rest_pose"])
                if full_rest.shape[0] == ctx.sh_bones_count:
                    out["rest_pose"] = full_rest[ctx.sh_idxs_br2full, :, :]
                else:
                    warn("Full rest_pose is also incompatible; leaving reduced rest_pose unchanged.")
    elif "rest_pose" in full_animation:
        full_rest = np.asarray(full_animation["rest_pose"])
        if full_rest.shape[0] == ctx.sh_bones_count:
            out["rest_pose"] = full_rest[ctx.sh_idxs_br2full, :, :]

    out.setdefault(
        "rest_pose_semantics",
        (
            "rest_pose is stored separately from animation_poses; animation_poses[0] "
            "corresponds to scene_frames[0] if scene_frames is present."
        ),
    )

    out["ssdr_step02_variant"] = {
        "variant_name": variant_name,
        "note": (
            "animation_poses were rebuilt by selecting reduced bone columns "
            "from the full animation according to the selected reduced skeleton."
        ),
    }

    return out


def process_single_mesh_file(
    mesh_path: Path,
    output_tree: Path,
    ctx: SkeletonContext,
    par_weight: float,
    bone_weight: float,
    child_weight: float,
    affinity_eps: float,
) -> dict[str, Any]:
    obj = load_data(mesh_path)

    if not isinstance(obj, dict):
        return {
            "operation": "skip_non_dict",
            "input": str(mesh_path),
            "output": None,
            "mesh_name": mesh_path.stem,
            "reason": "object_is_not_dict",
            "invalid_affinity_count": 0,
        }

    if "vertex_bone_weights" not in obj or "bone_names" not in obj:
        return {
            "operation": "skip_non_mesh",
            "input": str(mesh_path),
            "output": None,
            "mesh_name": str(obj.get("name", mesh_path.stem)),
            "reason": "missing vertex_bone_weights or bone_names",
            "invalid_affinity_count": 0,
        }

    reduced_obj, stats = reduce_mesh_weights(
        obj,
        ctx,
        par_weight=float(par_weight),
        bone_weight=float(bone_weight),
        child_weight=float(child_weight),
        affinity_eps=float(affinity_eps),
        copy_obj=True,
    )

    out_path = output_tree / mesh_path.name
    dump_pkl(out_path, reduced_obj)

    return {
        "operation": "reduce_mesh_weights",
        "input": str(mesh_path),
        "output": str(out_path),
        "mesh_name": stats.mesh_name,
        "full_bone_count": int(stats.full_bone_count),
        "reduced_bone_count": int(stats.reduced_bone_count),
        "vertex_count": int(stats.vertex_count),
        "invalid_affinity_count": int(stats.invalid_affinity_count),
        "nonzero_input_rows": int(stats.nonzero_input_rows),
        "nonzero_output_rows": int(stats.nonzero_output_rows),
    }


def reduce_bone_weight_tree(
    *,
    input_tree: Path,
    output_tree: Path,
    variant: str | None,
    par_weight: float,
    bone_weight: float,
    child_weight: float,
    affinity_eps: float,
    max_workers: int | None,
    overwrite_output: bool,
    allow_merge: bool,
    write_summary_file: bool,
    extra_remove_bones: list[str] | None = None,
    extra_variant_name: str | None = None,
    extra_remove_missing_ok: bool = False,
) -> dict[str, Any]:
    with Timer("time_02_select_layout"):
        layout = select_dataset_layout(
            input_tree=input_tree,
            variant=variant,
        )

    prepare_output_dir(
        output_tree,
        overwrite_output=overwrite_output,
        allow_merge=allow_merge,
    )

    log("Selected Step 02 dataset layout:")
    log(f"  layout_kind            : {layout.layout_kind}")
    log(f"  variant_name           : {layout.variant_name}")
    log(f"  full_root              : {layout.full_root}")
    log(f"  reduced_root           : {layout.reduced_root}")
    log(f"  full_skeleton_path     : {layout.full_skeleton_path}")
    log(f"  full_animation_path    : {layout.full_animation_path}")
    log(f"  reduced_skeleton_path  : {layout.reduced_skeleton_path}")
    log(f"  reduced_animation_path : {layout.reduced_animation_path}")
    log(f"  mesh_candidates        : {len(layout.mesh_paths)}")

    if not layout.mesh_paths:
        fail(f"No mesh candidate PKL files found under: {layout.full_root}")

    with Timer("time_02_load_core_data"):
        full_skeleton_raw = load_required_dict(layout.full_skeleton_path, role="full skeleton")
        full_animation = load_required_dict(layout.full_animation_path, role="full animation")
        reduced_skeleton_raw = load_required_dict(layout.reduced_skeleton_path, role="reduced skeleton")
        reduced_animation = load_required_dict(layout.reduced_animation_path, role="reduced animation")

        full_skeleton = align_skeleton_to_animation_order(
            full_skeleton_raw,
            full_animation,
            role="full",
        )
        reduced_skeleton = align_skeleton_to_animation_order(
            reduced_skeleton_raw,
            reduced_animation,
            role="reduced",
        )

        (
            reduced_skeleton,
            reduced_animation,
            extra_remove_info,
            effective_variant_name,
        ) = apply_extra_remove_bones_to_variant(
            base_variant_name=layout.variant_name,
            reduced_skeleton=reduced_skeleton,
            reduced_animation=reduced_animation,
            extra_remove_bones=list(extra_remove_bones or []),
            extra_variant_name=extra_variant_name,
            missing_ok=bool(extra_remove_missing_ok),
        )

    if extra_remove_info.get("enabled"):
        log("Applied Step 02 extra bone removals:")
        log(f"  base_variant_name      : {extra_remove_info.get('base_variant_name')}")
        log(f"  effective_variant_name : {extra_remove_info.get('effective_variant_name')}")
        log(f"  removed_count          : {extra_remove_info.get('removed_count')}")
        log(f"  target_bone_count      : {extra_remove_info.get('target_bone_count')}")

    with Timer("time_02_prepare_skeleton_context"):
        ctx = SkeletonContext(full_skeleton, reduced_skeleton)

    log(f"Full bones    : {ctx.sh_bones_count}")
    log(f"Reduced bones : {ctx.sh_br_bones_count}")

    with Timer("time_02_write_reduced_core"):
        reduced_animation_out = build_reduced_animation_output(
            full_animation=full_animation,
            reduced_animation=reduced_animation,
            ctx=ctx,
            variant_name=effective_variant_name,
        )

        dump_pkl(output_tree / FULL_SKELETON, reduced_skeleton)
        dump_pkl(output_tree / FULL_ANIMATION, reduced_animation_out)

    processed: list[dict[str, Any]] = []

    with Timer("time_02_process_meshes"):
        if max_workers == 1:
            for mesh_path in layout.mesh_paths:
                result = process_single_mesh_file(
                    mesh_path,
                    output_tree,
                    ctx,
                    float(par_weight),
                    float(bone_weight),
                    float(child_weight),
                    float(affinity_eps),
                )
                processed.append(result)
                log(f"Processed ({result['operation']}): {mesh_path.name}")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_single_mesh_file,
                        mesh_path,
                        output_tree,
                        ctx,
                        float(par_weight),
                        float(bone_weight),
                        float(child_weight),
                        float(affinity_eps),
                    )
                    for mesh_path in layout.mesh_paths
                ]

                for future in as_completed(futures):
                    result = future.result()
                    processed.append(result)
                    log(f"Processed ({result['operation']}): {Path(result['input']).name}")

    processed = sorted(processed, key=lambda item: str(item["input"]))

    reduced_mesh_count = sum(1 for item in processed if item["operation"] == "reduce_mesh_weights")
    skipped_count = len(processed) - reduced_mesh_count
    total_invalid_affinity_count = int(sum(int(item.get("invalid_affinity_count", 0)) for item in processed))

    if reduced_mesh_count <= 0:
        fail("No mesh files were reduced. Check mesh files for vertex_bone_weights and bone_names.")

    summary = {
        "schema": "ogd_lbs_ssdr_step02_reduce_bone_weight_v2",
        "input_tree": str(input_tree),
        "output_tree": str(output_tree),
        "layout": {
            "layout_kind": layout.layout_kind,
            "variant_name": layout.variant_name,
            "effective_variant_name": effective_variant_name,
            "extra_remove_bones_enabled": bool(extra_remove_info.get("enabled", False)),
            "full_root": str(layout.full_root),
            "reduced_root": str(layout.reduced_root) if layout.reduced_root is not None else None,
            "full_skeleton_path": str(layout.full_skeleton_path),
            "full_animation_path": str(layout.full_animation_path),
            "reduced_skeleton_path": str(layout.reduced_skeleton_path),
            "reduced_animation_path": str(layout.reduced_animation_path),
        },
        "weights": {
            "par_weight": float(par_weight),
            "bone_weight": float(bone_weight),
            "child_weight": float(child_weight),
            "affinity_eps": float(affinity_eps),
        },
        "extra_remove_bones": extra_remove_info,
        "skeleton": {
            "full_bone_count": int(ctx.sh_bones_count),
            "reduced_bone_count": int(ctx.sh_br_bones_count),
            "full_bone_names": list(ctx.sh_bone_names),
            "reduced_bone_names": list(ctx.sh_br_bone_names),
        },
        "outputs": {
            "skeleton_hierarchy_data": str(output_tree / FULL_SKELETON),
            "animation_data": str(output_tree / FULL_ANIMATION),
        },
        "mesh_candidate_count": int(len(layout.mesh_paths)),
        "reduced_mesh_count": int(reduced_mesh_count),
        "skipped_count": int(skipped_count),
        "total_invalid_affinity_count": int(total_invalid_affinity_count),
        "files": processed,
    }

    if write_summary_file:
        summary_path = output_tree / "step02_reduce_bone_weight_summary.json"
        write_stage_summary(summary_path, summary)
        log(f"Step 02 summary written: {summary_path}")

    record_metric("02_full_bone_count", int(ctx.sh_bones_count))
    record_metric("02_reduced_bone_count", int(ctx.sh_br_bones_count))
    record_metric("02_mesh_candidate_count", int(len(layout.mesh_paths)))
    record_metric("02_reduced_mesh_count", int(reduced_mesh_count))
    record_metric("02_skipped_count", int(skipped_count))
    record_metric("02_invalid_affinity_count", int(total_invalid_affinity_count))
    record_metric("02_extra_remove_bone_count", int(extra_remove_info.get("removed_count", 0) or 0))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSDR Step 02: reduce full-skeleton mesh weights to selected reduced skeleton."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Primary input. Normally work_dir or work_dir/data1. "
            "With explicit --output_dir, this may also be a custom input tree."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Explicit output tree. If omitted, Step 02 writes to <work_dir>/data2.",
    )
    parser.add_argument(
        "-m",
        "--metrics_dir",
        type=str,
        default=None,
        help="Optional directory to save metrics_02.json.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "For new vdb-root layout, select bone_reduced_versions/<variant>. "
            "If omitted, auto-select lexicographically smallest variant with cleaned_full last."
        ),
    )
    parser.add_argument(
        "--extra_remove_bones",
        type=str,
        default="",
        help=(
            "Optional comma/newline-separated bone names to remove from the selected "
            "base variant before Step 02 weight reduction."
        ),
    )
    parser.add_argument(
        "--extra_remove_bones_file",
        type=str,
        default=None,
        help=(
            "Optional text file containing one bone name per line, with '#' comments. "
            "Applied in addition to --extra_remove_bones."
        ),
    )
    parser.add_argument(
        "--extra_variant_name",
        type=str,
        default=None,
        help=(
            "Optional effective variant name recorded in Step02 outputs when using "
            "--extra_remove_bones*."
        ),
    )
    parser.add_argument(
        "--extra_remove_missing_ok",
        action="store_true",
        help="Do not fail if requested extra-remove bones are missing from the selected base variant.",
    )

    parser.add_argument("--par_weight", "--alpha", dest="par_weight", type=float, default=0.5)
    parser.add_argument("--bone_weight", "--beta", dest="bone_weight", type=float, default=0.3)
    parser.add_argument("--child_weight", "--gamma", dest="child_weight", type=float, default=0.2)
    parser.add_argument("--affinity_eps", type=float, default=1e-5)

    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--allow_merge", action="store_true")
    parser.add_argument("--no_summary", action="store_true")
    parser.add_argument("--enable_profiling", action="store_true")

    args = parser.parse_args(script_argv())

    resolution = resolve_ssdr_stage_dirs(
        step_id="02",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="SSDR Step 02: data1 -> data2 Bone Weight Reduction",
    )

    print_ssdr_resolution_info(resolution)

    try:
        validate_ssdr_stage_resolution(resolution)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    extra_remove_bones = load_extra_remove_bone_names(
        inline_text=args.extra_remove_bones,
        file_path=args.extra_remove_bones_file,
    )

    log("=" * 80)
    log("SSDR Step 02: data1 -> data2 Bone Weight Reduction")
    log(f"work_dir          : {resolution.work_dir}")
    log(f"input_tree        : {resolution.input_tree}")
    log(f"output_tree       : {resolution.output_tree}")
    log(f"input_source      : {resolution.input_source}")
    log(f"output_source     : {resolution.output_source}")
    log(f"variant           : {args.variant}")
    log(f"extra_remove_bones: {extra_remove_bones}")
    log(f"extra_remove_file : {args.extra_remove_bones_file}")
    log(f"extra_variant_name: {args.extra_variant_name}")
    log(f"extra_missing_ok  : {bool(args.extra_remove_missing_ok)}")
    log(f"par_weight        : {float(args.par_weight)}")
    log(f"bone_weight       : {float(args.bone_weight)}")
    log(f"child_weight      : {float(args.child_weight)}")
    log(f"affinity_eps      : {float(args.affinity_eps)}")
    log(f"max_workers       : {args.max_workers}")
    log(f"overwrite_output  : {bool(args.overwrite_output)}")
    log(f"allow_merge       : {bool(args.allow_merge)}")
    log("=" * 80)

    with Timer("time_02_total"):
        summary = reduce_bone_weight_tree(
            input_tree=resolution.input_tree,
            output_tree=resolution.output_tree,
            variant=args.variant,
            par_weight=float(args.par_weight),
            bone_weight=float(args.bone_weight),
            child_weight=float(args.child_weight),
            affinity_eps=float(args.affinity_eps),
            max_workers=args.max_workers,
            overwrite_output=bool(args.overwrite_output),
            allow_merge=bool(args.allow_merge),
            write_summary_file=not bool(args.no_summary),
            extra_remove_bones=extra_remove_bones,
            extra_variant_name=args.extra_variant_name,
            extra_remove_missing_ok=bool(args.extra_remove_missing_ok),
        )

    if args.metrics_dir:
        dump_metrics(Path(args.metrics_dir) / "metrics_02.json")

    log("SSDR Step 02 complete.")
    log(f"layout_kind                  : {summary['layout']['layout_kind']}")
    log(f"variant_name                 : {summary['layout']['variant_name']}")
    log(f"effective_variant_name       : {summary['layout'].get('effective_variant_name')}")
    log(f"extra_remove_bones_enabled   : {summary['layout'].get('extra_remove_bones_enabled')}")
    log(f"full_bone_count              : {summary['skeleton']['full_bone_count']}")
    log(f"reduced_bone_count           : {summary['skeleton']['reduced_bone_count']}")
    log(f"reduced_mesh_count           : {summary['reduced_mesh_count']}")
    log(f"total_invalid_affinity_count : {summary['total_invalid_affinity_count']}")
    log(f"output_tree                  : {resolution.output_tree}")


if __name__ == "__main__":
    main()
