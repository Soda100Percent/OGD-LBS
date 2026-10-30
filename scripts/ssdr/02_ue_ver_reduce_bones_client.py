#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
SSDR Step 02 UE version: reduce bones through a remote UE SkeletalMesh LOD service.

Default pipeline semantics:
    --input_dir <work_dir>
        reads : <work_dir>/data1
        writes: <work_dir>/data2_ue_lod

    --input_dir <work_dir>/data1
        reads : <work_dir>/data1
        writes: <work_dir>/data2_ue_lod

Explicit automation mode:
    --input_dir <custom_data1> --output_dir <custom_data2_ue_lod>
        reads : <custom_data1>
        writes: <custom_data2_ue_lod>

This script is intentionally placed under scripts/ssdr because it is a Step 02
baseline variant, parallel to:

    scripts/ssdr/02_reduce_bone_weight.py

Expected input layout, new vdb-root layout:

    data1/
    ├── skeleton_hierarchy_data.pkl
    ├── animation_data.pkl
    ├── <mesh>.pkl
    └── bone_reduced_versions/
        ├── cleaned_full/
        │   ├── skeleton_hierarchy_data.pkl
        │   └── animation_data.pkl
        ├── bone_reduced_001/
        │   ├── skeleton_hierarchy_data.pkl
        │   └── animation_data.pkl
        └── ...

Legacy flat layout is also supported:

    data1/
    ├── skeleton_hierarchy_data.pkl
    ├── animation_data.pkl
    ├── skeleton_hierarchy_data_bone_reduced.pkl
    ├── animation_data_bone_reduced.pkl
    └── <mesh>.pkl

Output layout:

    data2_ue_lod/
    ├── skeleton_hierarchy_data.pkl
    ├── animation_data.pkl
    ├── <mesh>.pkl
    ├── step02_ue_lod_summary.json
    └── ue_lod_artifacts/
        ├── original.fbx
        ├── ue_bone_lod.fbx
        ├── bones_to_remove.json
        ├── request.json
        └── manifest.json

Notes:
    - dry-run mode does not import bpy and can be run with normal Python.
    - full mode must run inside Blender Python because it exports/imports FBX.
    - UE service is expected to return an FBX whose main armature has the
      requested bones removed. If removed bones are still present, this script
      fails.
"""

import argparse
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from _env import ensure_repo_paths_in_syspath
except ModuleNotFoundError:  # pragma: no cover
    from ._env import ensure_repo_paths_in_syspath

ensure_repo_paths_in_syspath(__file__)

try:
    from _stage_io import (
        json_safe,
        load_required_dict,
        prepare_output_dir,
        require_file,
        write_stage_summary,
    )
except ModuleNotFoundError:  # pragma: no cover
    from ._stage_io import (
        json_safe,
        load_required_dict,
        prepare_output_dir,
        require_file,
        write_stage_summary,
    )

import numpy as np  # noqa: E402

from ogd_lbs.utils.common import (  # noqa: E402
    dump_json,
    dump_pkl,
    fail,
    load_data,
    load_pkl,
    log,
    script_argv,
    warn,
)
from ogd_lbs.ssdr.weight_reduction import align_skeleton_hierarchy_to_bone_names  # noqa: E402


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
    "step02_ue_lod_summary",
    "step03_qem_summary",
    "step04_optimize_summary",
    "metrics_history_per_frame",
}


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Step02UEResolution:
    input_arg: Path
    output_arg: Path | None
    work_dir: Path
    input_tree: Path
    output_tree: Path
    artifact_dir: Path
    input_source: str
    output_source: str
    mode: str


def resolve_path_no_strict(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def path_is_relative_to(child: Path, parent: Path) -> bool:
    child = child.resolve(strict=False)
    parent = parent.resolve(strict=False)
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_step02_ue_dirs(
    *,
    input_dir: str | Path,
    output_dir: str | Path | None,
) -> Step02UEResolution:
    raw_input = resolve_path_no_strict(input_dir)
    raw_output = (
        resolve_path_no_strict(output_dir)
        if output_dir is not None and str(output_dir).strip()
        else None
    )

    if raw_input.name == "data1":
        work_dir = raw_input.parent
        input_tree = raw_input
        input_source = "EXPLICIT_STAGE_DIR"
        mode = "input_arg_is_data1"

    elif (raw_input / "data1").exists() and (raw_input / "data1").is_dir():
        work_dir = raw_input
        input_tree = raw_input / "data1"
        input_source = "WORK_DIR_DEFAULT"
        mode = "input_arg_is_work_dir_with_data1"

    elif raw_output is not None:
        work_dir = raw_input.parent
        input_tree = raw_input
        input_source = "EXPLICIT_OVERRIDE"
        mode = "explicit_input_tree_and_output_tree"

    else:
        work_dir = raw_input
        input_tree = raw_input / "data1"
        input_source = "WORK_DIR_DEFAULT"
        mode = "input_arg_treated_as_work_dir_missing_or_unvalidated_data1"

    if raw_output is not None:
        output_tree = raw_output
        output_source = "EXPLICIT_OVERRIDE"
    else:
        output_tree = work_dir / "data2_ue_lod"
        output_source = "WORK_DIR_DEFAULT"

    return Step02UEResolution(
        input_arg=raw_input,
        output_arg=raw_output,
        work_dir=work_dir.resolve(strict=False),
        input_tree=input_tree.resolve(strict=False),
        output_tree=output_tree.resolve(strict=False),
        artifact_dir=(output_tree / "ue_lod_artifacts").resolve(strict=False),
        input_source=input_source,
        output_source=output_source,
        mode=mode,
    )


def validate_step02_ue_resolution(resolution: Step02UEResolution) -> None:
    errors: list[str] = []

    if not resolution.input_tree.exists():
        errors.append(f"Missing input tree: {resolution.input_tree}")
    elif not resolution.input_tree.is_dir():
        errors.append(f"Input tree is not a directory: {resolution.input_tree}")

    if resolution.output_tree == resolution.input_tree:
        errors.append(f"Output tree would equal input tree: {resolution.output_tree}")

    if path_is_relative_to(resolution.output_tree, resolution.input_tree):
        errors.append(
            f"Output tree must not be inside input tree: "
            f"output={resolution.output_tree}, input={resolution.input_tree}"
        )

    if errors:
        raise FileNotFoundError(
            "SSDR Step 02 UE path resolution validation failed:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )


def print_step02_ue_resolution_info(resolution: Step02UEResolution) -> None:
    line = "=" * 96
    subline = "-" * 96
    print(line, flush=True)
    print("📂 [SSDR Step 02 UE LOD] WorkDir Resolution Info", flush=True)
    print(f"   mode          : {resolution.mode}", flush=True)
    print(f"   work_dir      : {resolution.work_dir}", flush=True)
    print(subline, flush=True)
    print(f"   [INPUT_ARG ] input_dir    : {resolution.input_arg}", flush=True)
    print(
        f"   [INPUT     ] input_tree   : {resolution.input_tree} "
        f"({resolution.input_source} | exists={resolution.input_tree.exists()} | type=dir)",
        flush=True,
    )
    if resolution.output_arg is not None:
        print(f"   [OUTPUT_ARG] output_dir   : {resolution.output_arg}", flush=True)
    print(
        f"   [OUTPUT    ] output_tree  : {resolution.output_tree} "
        f"({resolution.output_source} | exists={resolution.output_tree.exists()} | type=dir)",
        flush=True,
    )
    print(f"   [ARTIFACTS ] artifact_dir : {resolution.artifact_dir}", flush=True)
    print(line, flush=True)


# -----------------------------------------------------------------------------
# Step02 layout selection / variant logic
# -----------------------------------------------------------------------------

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
    # Same convention as 02_reduce_bone_weight.py:
    # cleaned_full is valid but sorted last unless it is the only candidate.
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
        fail("Multiple mesh files would collide in flat output:\n" + "\n".join(lines))


def select_dataset_layout(
    *,
    input_tree: Path,
    variant: str | None,
) -> SelectedDatasetLayout:
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
            "Unsupported on purpose: wrapper layout data1/fitbox_vdb/...\n"
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


# -----------------------------------------------------------------------------
# Extra remove bones
# -----------------------------------------------------------------------------

def _append_unique_bone_name(out: list[str], seen: set[str], name: str) -> None:
    name = str(name).strip()
    if not name:
        return
    if name in seen:
        return
    seen.add(name)
    out.append(name)


def _parse_bone_names_text(text: str | None) -> list[str]:
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
    kept_old_indices = [idx for idx, name in enumerate(bone_names) if name not in remove_set]

    if not kept_old_indices:
        fail("extra_remove_bones would remove all bones from the selected skeleton.")

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_old_indices)}

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
    pruned_parent_indices = [nearest_kept_parent_new_index(old_idx) for old_idx in kept_old_indices]

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
    out = dict(animation_obj)

    source_bone_names = [str(x) for x in out.get("bone_names", [])]
    target_bone_names = [str(x) for x in target_bone_names]

    if not source_bone_names:
        fail(f"{role} animation is missing bone_names; cannot apply extra_remove_bones.")

    missing = [name for name in target_bone_names if name not in source_bone_names]
    if missing:
        fail(f"{role} animation is missing target bones after extra_remove_bones: {missing}")

    indices = np.asarray([source_bone_names.index(name) for name in target_bone_names], dtype=np.int64)

    out["bone_names"] = list(target_bone_names)

    if "rest_pose" in out:
        rest_pose = np.asarray(out["rest_pose"])
        if rest_pose.ndim >= 1 and rest_pose.shape[0] == len(source_bone_names):
            out["rest_pose"] = rest_pose[indices, ...]

    if "animation_poses" in out:
        poses = np.asarray(out["animation_poses"])
        if poses.ndim >= 2 and poses.shape[1] == len(source_bone_names):
            out["animation_poses"] = poses[:, indices, ...]

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
    pruned_skeleton["ssdr_step02_ue_extra_remove_bones"] = dict(metadata)

    pruned_animation = dict(pruned_animation)
    pruned_animation["ssdr_step02_ue_extra_remove_bones"] = dict(metadata)

    return pruned_skeleton, pruned_animation, metadata, effective_variant_name


# -----------------------------------------------------------------------------
# Bones-to-remove derivation
# -----------------------------------------------------------------------------

def derive_bones_to_remove(
    *,
    full_skeleton: dict[str, Any],
    target_skeleton: dict[str, Any],
    allow_remove_root: bool,
    allow_empty_remove: bool,
) -> dict[str, Any]:
    full_bones = [str(x) for x in full_skeleton.get("bone_names", [])]
    target_bones = [str(x) for x in target_skeleton.get("bone_names", [])]

    if not full_bones:
        fail("Full skeleton has no bone_names.")
    if not target_bones:
        fail("Target reduced skeleton has no bone_names.")

    full_set = set(full_bones)
    target_set = set(target_bones)

    extra_in_target = [name for name in target_bones if name not in full_set]
    if extra_in_target:
        fail(
            "Target reduced skeleton contains bones not present in full skeleton: "
            f"{extra_in_target}"
        )

    bones_to_remove = [name for name in full_bones if name not in target_set]

    if len(bones_to_remove) >= len(full_bones):
        fail("bones_to_remove would remove all full skeleton bones.")

    root_name = full_bones[0]
    if root_name in bones_to_remove and not bool(allow_remove_root):
        fail(
            f"Refusing to remove root bone {root_name!r}. "
            "Pass --allow_remove_root if this is intentional."
        )

    if not bones_to_remove:
        msg = "Derived empty bones_to_remove list."
        if allow_empty_remove:
            warn(msg + " Continuing because empty removal is allowed.")
        else:
            fail(msg + " Pass --allow_empty_remove to permit no-op UE reduction.")

    return {
        "full_bone_names": full_bones,
        "target_bone_names": target_bones,
        "bones_to_remove": bones_to_remove,
        "full_bone_count": int(len(full_bones)),
        "target_bone_count": int(len(target_bones)),
        "bones_to_remove_count": int(len(bones_to_remove)),
        "root_bone_name": root_name,
    }


# -----------------------------------------------------------------------------
# Blender helpers
# -----------------------------------------------------------------------------

def require_bpy():
    try:
        import bpy  # type: ignore
        return bpy
    except Exception as exc:
        fail(
            "This operation requires Blender Python with bpy available. "
            "Run with:\n"
            "  blender -b --python scripts/ssdr/02_ue_ver_reduce_bones_client.py -- <args>\n"
            f"Original import error: {exc}"
        )
        raise RuntimeError("unreachable")


def ensure_object_mode() -> None:
    bpy = require_bpy()
    try:
        active = bpy.context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def export_scene_to_fbx(
    *,
    fbx_path: Path,
) -> None:
    bpy = require_bpy()
    ensure_object_mode()

    bpy.ops.object.select_all(action="DESELECT")

    selected_count = 0
    for obj in bpy.context.scene.objects:
        if obj.type in {"ARMATURE", "MESH"}:
            obj.select_set(True)
            selected_count += 1

    if selected_count <= 0:
        fail("No ARMATURE or MESH objects found for FBX export.")

    fbx_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        filepath=str(fbx_path),
        use_selection=True,
        global_scale=1.0,
        apply_unit_scale=True,
        bake_space_transform=False,
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="NULL",
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
    )

    try:
        result = bpy.ops.export_scene.fbx(**kwargs)
    except TypeError as exc:
        warn(f"FBX export kwargs failed with TypeError; retrying with reduced kwargs: {exc}")
        for key in ["armature_nodetype", "bake_anim_use_all_bones"]:
            kwargs.pop(key, None)
        result = bpy.ops.export_scene.fbx(**kwargs)

    if result != {"FINISHED"}:
        fail(f"FBX export failed: {fbx_path}, result={result}")

    log(f"FBX export success: {fbx_path} ({fbx_path.stat().st_size} bytes)")


def build_full_data1_lbs_scene_and_export_fbx(
    *,
    input_tree: Path,
    output_fbx: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    bpy = require_bpy()

    from ogd_lbs.fitbox.blender_io import (  # noqa: WPS433
        _fitbox_load_stage_objects_multi,
        add_fitbox_lbs_vertex_groups_and_modifier_plugin_style,
        animate_fitbox_armature_lbs_plugin_style,
        clear_blender_scene_for_fitbox_import,
        create_fitbox_armature_lbs_plugin_style,
        create_fitbox_mesh_object_plugin_style,
    )

    clear_blender_scene_for_fitbox_import()

    bpy.context.scene.render.fps = int(args.fps)

    skeleton_path, animation_path, mesh_paths, skeleton_obj, animation_obj, mesh_objs = (
        _fitbox_load_stage_objects_multi(input_tree, args.mesh)
    )

    armature_obj = create_fitbox_armature_lbs_plugin_style(
        skeleton_obj=skeleton_obj,
        animation_obj=animation_obj,
        armature_name="Armature",
        bone_length=float(args.bone_length),
    )

    keyed_count = animate_fitbox_armature_lbs_plugin_style(
        armature_obj=armature_obj,
        skeleton_obj=skeleton_obj,
        animation_obj=animation_obj,
        mesh_obj=mesh_objs[0],
        max_lbs_frames=int(args.max_lbs_frames),
        use_scene_frames=bool(args.use_scene_frames),
        matrix_space=str(args.matrix_space),
    )

    mesh_object_names: list[str] = []

    for mesh_path, mesh_obj in zip(mesh_paths, mesh_objs):
        object_name = str(mesh_obj.get("name", mesh_path.stem))
        mesh_bl_obj = create_fitbox_mesh_object_plugin_style(
            mesh_obj=mesh_obj,
            object_name=object_name,
        )
        mesh_object_names.append(mesh_bl_obj.name)

        add_fitbox_lbs_vertex_groups_and_modifier_plugin_style(
            obj=mesh_bl_obj,
            mesh_obj=mesh_obj,
            skeleton_obj=skeleton_obj,
            armature_obj=armature_obj,
            weight_eps=float(args.weight_eps),
        )

    if args.save_debug_blend:
        debug_blend = output_fbx.parent / "debug_full_data1_lbs_scene.blend"
        result = bpy.ops.wm.save_as_mainfile(filepath=str(debug_blend))
        if result != {"FINISHED"}:
            warn(f"Failed to save debug blend: {debug_blend}, result={result}")
        else:
            log(f"Saved debug full scene blend: {debug_blend}")

    export_scene_to_fbx(fbx_path=output_fbx)

    return {
        "skeleton_path": str(skeleton_path),
        "animation_path": str(animation_path),
        "mesh_paths": [str(p) for p in mesh_paths],
        "mesh_object_names": mesh_object_names,
        "armature_object_name": armature_obj.name,
        "lbs_keyed_frame_count": int(keyed_count),
    }


def import_ue_fbx_scene(
    *,
    fbx_path: Path,
) -> None:
    bpy = require_bpy()
    from ogd_lbs.fitbox.blender_io import clear_blender_scene_for_fitbox_import  # noqa: WPS433

    clear_blender_scene_for_fitbox_import()

    if not fbx_path.is_file():
        fail(f"UE FBX does not exist: {fbx_path}")

    try:
        result = bpy.ops.import_scene.fbx(
            filepath=str(fbx_path),
            use_anim=False,
            automatic_bone_orientation=False,
        )
    except TypeError:
        result = bpy.ops.import_scene.fbx(filepath=str(fbx_path))

    if result != {"FINISHED"}:
        fail(f"Failed to import UE FBX: {fbx_path}, result={result}")

    log(f"Imported UE FBX: {fbx_path}")


def choose_single_armature() -> Any:
    bpy = require_bpy()
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]

    if not armatures:
        fail("No ARMATURE object found after importing UE FBX.")

    if len(armatures) == 1:
        return armatures[0]

    non_original = [obj for obj in armatures if obj.name != "Armature_Original"]
    if len(non_original) == 1:
        warn(
            "Multiple armatures found; auto-selected non-Armature_Original: "
            f"{non_original[0].name}"
        )
        return non_original[0]

    fail(
        "Could not auto-select UE output armature. "
        f"Found armatures: {[obj.name for obj in armatures]}"
    )
    raise RuntimeError("unreachable")


def collect_mesh_objects() -> list[Any]:
    bpy = require_bpy()
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        fail("No MESH objects found after importing UE FBX.")
    return meshes


def build_parent_first_bone_order(armature_obj: Any) -> list[str]:
    bones = list(armature_obj.data.bones)
    names = [bone.name for bone in bones]
    name_set = set(names)

    children: dict[str, list[str]] = {name: [] for name in names}
    roots: list[str] = []

    for bone in bones:
        if bone.parent is None or bone.parent.name not in name_set:
            roots.append(bone.name)
        else:
            children[bone.parent.name].append(bone.name)

    out: list[str] = []
    seen: set[str] = set()
    queue = list(roots)

    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        queue.extend(children.get(name, []))

    for name in names:
        if name not in seen:
            out.append(name)

    return out


def keyframe_armature_from_full_animation(
    *,
    armature_obj: Any,
    full_animation: dict[str, Any],
    source_bone_names: list[str],
    matrix_space: str,
) -> dict[str, Any]:
    bpy = require_bpy()
    from mathutils import Matrix  # noqa: WPS433

    poses = np.asarray(full_animation.get("animation_poses"), dtype=np.float64)
    if poses.ndim != 4 or poses.shape[1] != len(source_bone_names) or poses.shape[2:] != (4, 4):
        fail(
            "Full animation_poses shape mismatch. Expected "
            f"(frames, {len(source_bone_names)}, 4, 4), got {poses.shape}"
        )

    frame_count = int(poses.shape[0])
    if frame_count <= 0:
        fail("Full animation has zero frames.")

    source_index = {name: idx for idx, name in enumerate(source_bone_names)}
    ue_bone_order = build_parent_first_bone_order(armature_obj)

    missing_from_source = [name for name in ue_bone_order if name not in source_index]
    if missing_from_source:
        fail(
            "UE output armature contains bones not present in full animation; "
            f"cannot key animation: {missing_from_source}"
        )

    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="OBJECT")
    armature_obj.data.pose_position = "POSE"

    for pose_bone in armature_obj.pose.bones:
        try:
            pose_bone.rotation_mode = "QUATERNION"
        except Exception:
            pass

    deps = bpy.context.evaluated_depsgraph_get()
    matrix_space_norm = str(matrix_space).upper()

    log(
        f"Keyframing UE output armature from full animation: "
        f"frames={frame_count}, bones={len(ue_bone_order)}, matrix_space={matrix_space_norm}"
    )

    for frame_idx in range(frame_count):
        frame_number = frame_idx + 1
        bpy.context.scene.frame_set(frame_number)

        if frame_idx == 0 or frame_idx == frame_count - 1 or frame_idx % 25 == 0:
            log(f"  keyframe {frame_idx + 1}/{frame_count}")

        for bone_name in ue_bone_order:
            pose_bone = armature_obj.pose.bones.get(bone_name)
            if pose_bone is None:
                continue

            src_idx = source_index[bone_name]
            mat = Matrix(poses[frame_idx, src_idx].tolist())

            if matrix_space_norm == "WORLD":
                mat = armature_obj.convert_space(
                    pose_bone=pose_bone,
                    matrix=mat,
                    from_space="WORLD",
                    to_space="POSE",
                )

            pose_bone.matrix = mat

            try:
                deps.update()
            except Exception:
                pass

            pose_bone.keyframe_insert("location", frame=frame_number)
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame_number)
            pose_bone.keyframe_insert("scale", frame=frame_number)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frame_count
    bpy.context.scene.frame_set(1)

    original_scene_frames = full_animation.get("scene_frames")
    if original_scene_frames is not None:
        scene_frames = np.asarray(original_scene_frames, dtype=np.int64)
        if scene_frames.ndim != 1 or scene_frames.shape[0] != frame_count:
            warn(
                f"Input scene_frames shape mismatch: {scene_frames.shape}, "
                f"expected ({frame_count},). Using contiguous 1..N."
            )
            scene_frames = np.arange(1, frame_count + 1, dtype=np.int64)
    else:
        scene_frames = np.arange(1, frame_count + 1, dtype=np.int64)

    return {
        "frame_numbers": list(range(1, frame_count + 1)),
        "scene_frames": scene_frames,
        "frame_count": frame_count,
        "ue_bone_order_parent_first": ue_bone_order,
    }


def safe_stem(name: str) -> str:
    out = str(name).strip().replace("/", "_").replace("\\", "_")
    if not out:
        out = "Mesh"
    return out


def convert_imported_ue_scene_to_fitbox_stage(
    *,
    output_tree: Path,
    armature_obj: Any,
    mesh_objects: list[Any],
    full_skeleton: dict[str, Any],
    full_animation: dict[str, Any],
    target_skeleton: dict[str, Any],
    bones_to_remove: list[str],
    effective_variant_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from ogd_lbs.fitbox.blender_io import export_animation, export_mesh, export_skeleton  # noqa: WPS433

    output_tree.mkdir(parents=True, exist_ok=True)

    full_bone_names = [str(x) for x in full_skeleton.get("bone_names", full_animation.get("bone_names", []))]
    if not full_bone_names:
        fail("Cannot determine full bone names for animation keying.")

    ue_bone_names = [bone.name for bone in armature_obj.data.bones]
    target_bone_names = [str(x) for x in target_skeleton.get("bone_names", [])]

    ue_set = set(ue_bone_names)
    target_set = set(target_bone_names)
    removed_set = set(bones_to_remove)

    still_present_removed = [name for name in bones_to_remove if name in ue_set]
    if still_present_removed:
        fail(
            "UE output still contains requested removed bones. "
            f"First offending bones: {still_present_removed[:50]}"
        )

    missing_target = [name for name in target_bone_names if name not in ue_set]
    if missing_target:
        fail(
            "UE output armature is missing target reduced bones: "
            f"{missing_target[:50]}"
        )

    extra_ue_bones = [name for name in ue_bone_names if name not in target_set]
    if extra_ue_bones and not bool(args.allow_ue_extra_bones):
        fail(
            "UE output armature contains extra bones not in target reduced skeleton: "
            f"{extra_ue_bones[:50]}\n"
            "Pass --allow_ue_extra_bones if this is expected."
        )

    if bool(args.require_mesh_names_match):
        expected_mesh_names = {
            safe_stem(Path(path).stem)
            for path in getattr(args, "_input_mesh_paths_for_validation", [])
        }
        actual_mesh_names = {safe_stem(obj.name) for obj in mesh_objects}
        if expected_mesh_names and expected_mesh_names != actual_mesh_names:
            fail(
                "UE output mesh names do not match input mesh names.\n"
                f"expected={sorted(expected_mesh_names)}\n"
                f"actual={sorted(actual_mesh_names)}"
            )

    key_info = keyframe_armature_from_full_animation(
        armature_obj=armature_obj,
        full_animation=full_animation,
        source_bone_names=full_bone_names,
        matrix_space=str(args.matrix_space),
    )

    frame_numbers = key_info["frame_numbers"]
    scene_frames = key_info["scene_frames"]

    out_skel_path = output_tree / FULL_SKELETON
    out_anim_path = output_tree / FULL_ANIMATION

    exported_bone_names = export_skeleton(
        armature_obj,
        out_skel_path,
        fmt="pkl",
    )

    export_animation(
        armature_obj,
        out_anim_path,
        fmt="pkl",
        frames=frame_numbers,
    )

    # Patch animation metadata to preserve original SSDR scene_frames and record provenance.
    anim_out = load_pkl(out_anim_path)
    anim_out["scene_frames"] = scene_frames
    anim_out["ssdr_step02_ue_lod"] = {
        "variant_name": effective_variant_name,
        "source": "UE SkeletalMesh LOD bone reduction",
        "note": (
            "UE output provided reduced skeleton, mesh, and remapped weights. "
            "animation_poses were re-keyed from the full data1 animation using UE output bone names."
        ),
    }
    dump_pkl(out_anim_path, anim_out)

    mesh_outputs: dict[str, str] = {}

    for obj in mesh_objects:
        mesh_name = safe_stem(obj.name)
        out_mesh_path = output_tree / f"{mesh_name}.pkl"

        export_mesh(
            obj,
            armature_obj,
            out_mesh_path,
            fmt="pkl",
            frames=frame_numbers,
        )

        mesh_obj = load_pkl(out_mesh_path)
        mesh_obj["ssdr_step02_ue_lod"] = {
            "variant_name": effective_variant_name,
            "source": "UE SkeletalMesh LOD bone reduction",
        }
        dump_pkl(out_mesh_path, mesh_obj)
        mesh_outputs[mesh_name] = str(out_mesh_path)

    return {
        "skeleton_hierarchy_data": str(out_skel_path),
        "animation_data": str(out_anim_path),
        "meshes": mesh_outputs,
        "ue_bone_names": exported_bone_names,
        "ue_bone_count": int(len(exported_bone_names)),
        "target_bone_set_matches": bool(set(exported_bone_names) == target_set),
        "ue_bone_order_differs_from_target": bool(exported_bone_names != target_bone_names),
        "removed_bones_absent": bool(not still_present_removed),
        "extra_ue_bones": extra_ue_bones,
        "mesh_count": int(len(mesh_outputs)),
        "keyframing": {
            "frame_count": int(key_info["frame_count"]),
            "scene_frames_first": int(scene_frames[0]) if len(scene_frames) else None,
            "scene_frames_last": int(scene_frames[-1]) if len(scene_frames) else None,
        },
    }


# -----------------------------------------------------------------------------
# HTTP multipart request
# -----------------------------------------------------------------------------

def encode_multipart_formdata(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----ogd-lbs-ue-lod-{uuid.uuid4().hex}"
    crlf = "\r\n"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}{crlf}".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"{crlf}{crlf}'.encode("utf-8")
        )
        body.extend(str(value).encode("utf-8"))
        body.extend(crlf.encode("utf-8"))

    for name, (filename, data, content_type) in files.items():
        body.extend(f"--{boundary}{crlf}".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'
                f"{crlf}"
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}{crlf}{crlf}".encode("utf-8"))
        body.extend(data)
        body.extend(crlf.encode("utf-8"))

    body.extend(f"--{boundary}--{crlf}".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def post_ue_reduce_bones(
    *,
    url: str,
    input_fbx: Path,
    output_fbx: Path,
    bones_to_remove: list[str],
    timeout: int,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not input_fbx.is_file():
        fail(f"Input FBX for UE request does not exist: {input_fbx}")

    request_obj = {
        "url": url,
        "input_fbx": str(input_fbx),
        "output_fbx": str(output_fbx),
        "bones_to_remove": list(bones_to_remove),
        "options": options or {},
        "timeout": int(timeout),
    }

    fields = {
        "bones_to_remove": json.dumps(bones_to_remove, ensure_ascii=False),
    }
    if options is not None:
        fields["options"] = json.dumps(options, ensure_ascii=False)

    fbx_bytes = input_fbx.read_bytes()

    body, content_type = encode_multipart_formdata(
        fields=fields,
        files={
            "file": (
                input_fbx.name,
                fbx_bytes,
                "application/octet-stream",
            )
        },
    )

    log(f"Posting UE reduce-bones request: {url}")
    log(f"  input_fbx          : {input_fbx}")
    log(f"  input_fbx_size     : {len(fbx_bytes)}")
    log(f"  bones_to_remove    : {len(bones_to_remove)}")
    log(f"  timeout            : {int(timeout)}")

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "Accept": "application/octet-stream",
        },
        method="POST",
    )

    started = time.time()

    try:
        with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
            status_code = int(resp.getcode())
            response_bytes = resp.read()
            response_headers = dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        fail(
            f"UE service returned HTTP {exc.code}:\n"
            f"{detail[-8000:]}"
        )
    except urllib.error.URLError as exc:
        fail(f"UE service request failed: {exc}")
    except TimeoutError as exc:
        fail(f"UE service request timeout: {exc}")

    elapsed = time.time() - started

    if status_code < 200 or status_code >= 300:
        fail(f"UE service returned non-success HTTP status: {status_code}")

    if not response_bytes:
        fail("UE service returned empty response body.")

    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    output_fbx.write_bytes(response_bytes)

    if output_fbx.stat().st_size <= 0:
        fail(f"UE output FBX is empty after write: {output_fbx}")

    log(
        f"UE reduce-bones response saved: {output_fbx} "
        f"({output_fbx.stat().st_size} bytes, {elapsed:.2f}s)"
    )

    return {
        **request_obj,
        "status_code": status_code,
        "elapsed_sec": float(elapsed),
        "response_headers": response_headers,
        "response_bytes": int(len(response_bytes)),
    }


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SSDR Step 02 UE version: data1 -> data2_ue_lod via UE bone reduction service."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Primary input. Normally work_dir or work_dir/data1.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Explicit output tree. If omitted, writes to <work_dir>/data2_ue_lod.",
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
        help="Optional comma/newline-separated bone names to remove from the selected base variant.",
    )
    parser.add_argument(
        "--extra_remove_bones_file",
        type=str,
        default=None,
        help="Optional text file containing one bone name per line, with '#' comments.",
    )
    parser.add_argument(
        "--extra_variant_name",
        type=str,
        default=None,
        help="Optional effective variant name recorded when using extra removals.",
    )
    parser.add_argument(
        "--extra_remove_missing_ok",
        action="store_true",
        help="Do not fail if requested extra-remove bones are missing from selected base variant.",
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:9999/api/reduce_bones",
        help="UE reduce-bones service endpoint.",
    )
    parser.add_argument("--timeout", type=int, default=900)

    parser.add_argument("--mesh", type=str, default=None, help="Mesh stem to export. Omit or ALL for all meshes.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_lbs_frames", type=int, default=-1)
    parser.add_argument("--use_scene_frames", action="store_true")
    parser.add_argument("--matrix_space", choices=["POSE", "WORLD"], default="POSE")
    parser.add_argument("--bone_length", type=float, default=0.5)
    parser.add_argument("--weight_eps", type=float, default=1e-8)

    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--allow_merge", action="store_true")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_fbx_export", action="store_true")
    parser.add_argument("--skip_request", action="store_true")
    parser.add_argument("--skip_reimport", action="store_true")
    parser.add_argument("--save_debug_blend", action="store_true")

    parser.add_argument("--allow_remove_root", action="store_true")
    parser.add_argument("--allow_empty_remove", dest="allow_empty_remove", action="store_true", default=True)
    parser.add_argument("--no_allow_empty_remove", dest="allow_empty_remove", action="store_false")
    parser.add_argument("--allow_ue_extra_bones", action="store_true")
    parser.add_argument("--require_mesh_names_match", action="store_true")

    parser.add_argument("--no_summary", action="store_true")

    return parser


def load_and_prepare_layout(args: argparse.Namespace, resolution: Step02UEResolution) -> dict[str, Any]:
    layout = select_dataset_layout(
        input_tree=resolution.input_tree,
        variant=args.variant,
    )

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

    extra_remove_bones = load_extra_remove_bone_names(
        inline_text=args.extra_remove_bones,
        file_path=args.extra_remove_bones_file,
    )

    reduced_skeleton, reduced_animation, extra_remove_info, effective_variant_name = (
        apply_extra_remove_bones_to_variant(
            base_variant_name=layout.variant_name,
            reduced_skeleton=reduced_skeleton,
            reduced_animation=reduced_animation,
            extra_remove_bones=extra_remove_bones,
            extra_variant_name=args.extra_variant_name,
            missing_ok=bool(args.extra_remove_missing_ok),
        )
    )

    bone_plan = derive_bones_to_remove(
        full_skeleton=full_skeleton,
        target_skeleton=reduced_skeleton,
        allow_remove_root=bool(args.allow_remove_root),
        allow_empty_remove=bool(args.allow_empty_remove),
    )

    return {
        "layout": layout,
        "full_skeleton": full_skeleton,
        "full_animation": full_animation,
        "reduced_skeleton": reduced_skeleton,
        "reduced_animation": reduced_animation,
        "extra_remove_info": extra_remove_info,
        "effective_variant_name": effective_variant_name,
        "bone_plan": bone_plan,
    }


def write_artifact_jsons(
    *,
    resolution: Step02UEResolution,
    bone_plan: dict[str, Any],
    request_obj: dict[str, Any],
    manifest_obj: dict[str, Any],
) -> dict[str, str]:
    resolution.artifact_dir.mkdir(parents=True, exist_ok=True)

    bones_path = resolution.artifact_dir / "bones_to_remove.json"
    request_path = resolution.artifact_dir / "request.json"
    manifest_path = resolution.artifact_dir / "manifest.json"

    dump_json(bones_path, bone_plan["bones_to_remove"])
    dump_json(request_path, request_obj)
    dump_json(manifest_path, manifest_obj)

    return {
        "bones_to_remove_json": str(bones_path),
        "request_json": str(request_path),
        "manifest_json": str(manifest_path),
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(script_argv())

    resolution = resolve_step02_ue_dirs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    print_step02_ue_resolution_info(resolution)

    try:
        validate_step02_ue_resolution(resolution)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    prepared = load_and_prepare_layout(args, resolution)
    layout: SelectedDatasetLayout = prepared["layout"]
    full_skeleton = prepared["full_skeleton"]
    full_animation = prepared["full_animation"]
    reduced_skeleton = prepared["reduced_skeleton"]
    extra_remove_info = prepared["extra_remove_info"]
    effective_variant_name = prepared["effective_variant_name"]
    bone_plan = prepared["bone_plan"]

    args._input_mesh_paths_for_validation = [str(p) for p in layout.mesh_paths]

    log("=" * 80)
    log("SSDR Step 02 UE LOD plan")
    log(f"input_tree              : {resolution.input_tree}")
    log(f"output_tree             : {resolution.output_tree}")
    log(f"artifact_dir            : {resolution.artifact_dir}")
    log(f"layout_kind             : {layout.layout_kind}")
    log(f"variant_name            : {layout.variant_name}")
    log(f"effective_variant_name  : {effective_variant_name}")
    log(f"mesh_candidates         : {len(layout.mesh_paths)}")
    log(f"full_bone_count         : {bone_plan['full_bone_count']}")
    log(f"target_bone_count       : {bone_plan['target_bone_count']}")
    log(f"bones_to_remove_count   : {bone_plan['bones_to_remove_count']}")
    log(f"url                     : {args.url}")
    log(f"dry_run                 : {bool(args.dry_run)}")
    log(f"skip_fbx_export         : {bool(args.skip_fbx_export)}")
    log(f"skip_request            : {bool(args.skip_request)}")
    log(f"skip_reimport           : {bool(args.skip_reimport)}")
    log("=" * 80)

    plan_preview = {
        "schema": "ogd_lbs_ssdr_step02_ue_lod_plan_v1",
        "input_tree": str(resolution.input_tree),
        "output_tree": str(resolution.output_tree),
        "artifact_dir": str(resolution.artifact_dir),
        "layout": {
            "layout_kind": layout.layout_kind,
            "variant_name": layout.variant_name,
            "effective_variant_name": effective_variant_name,
            "full_skeleton_path": str(layout.full_skeleton_path),
            "full_animation_path": str(layout.full_animation_path),
            "reduced_skeleton_path": str(layout.reduced_skeleton_path),
            "reduced_animation_path": str(layout.reduced_animation_path),
            "mesh_paths": [str(p) for p in layout.mesh_paths],
        },
        "extra_remove_bones": extra_remove_info,
        "skeleton": bone_plan,
        "ue_service": {
            "url": args.url,
            "timeout": int(args.timeout),
        },
        "flags": {
            "dry_run": bool(args.dry_run),
            "skip_fbx_export": bool(args.skip_fbx_export),
            "skip_request": bool(args.skip_request),
            "skip_reimport": bool(args.skip_reimport),
        },
    }

    if args.dry_run:
        print(json.dumps(json_safe(plan_preview), indent=4, ensure_ascii=False), flush=True)
        log("Dry-run complete. No files were written.")
        return

    prepare_output_dir(
        resolution.output_tree,
        overwrite_output=bool(args.overwrite_output),
        allow_merge=bool(args.allow_merge),
    )
    resolution.artifact_dir.mkdir(parents=True, exist_ok=True)

    original_fbx = resolution.artifact_dir / "original.fbx"
    ue_bone_lod_fbx = resolution.artifact_dir / "ue_bone_lod.fbx"

    request_record: dict[str, Any] = {
        "schema": "ogd_lbs_ssdr_step02_ue_lod_request_v1",
        "url": args.url,
        "input_fbx": str(original_fbx),
        "output_fbx": str(ue_bone_lod_fbx),
        "bones_to_remove": list(bone_plan["bones_to_remove"]),
        "timeout": int(args.timeout),
    }

    manifest_obj: dict[str, Any] = {
        **plan_preview,
        "artifacts": {
            "original_fbx": str(original_fbx),
            "ue_bone_lod_fbx": str(ue_bone_lod_fbx),
        },
    }

    artifact_json_paths = write_artifact_jsons(
        resolution=resolution,
        bone_plan=bone_plan,
        request_obj=request_record,
        manifest_obj=manifest_obj,
    )

    export_info: dict[str, Any] | None = None
    ue_request_info: dict[str, Any] | None = None
    stage_export_info: dict[str, Any] | None = None
    validation_info: dict[str, Any] = {
        "warnings": [],
        "removed_bones_absent": None,
        "target_bone_set_matches": None,
        "ue_bone_order_differs_from_target": None,
        "mesh_count": None,
    }

    if args.skip_fbx_export:
        if not original_fbx.is_file():
            fail(f"--skip_fbx_export was set but original FBX does not exist: {original_fbx}")
        warn(f"Skipping FBX export and reusing existing original FBX: {original_fbx}")
    else:
        export_info = build_full_data1_lbs_scene_and_export_fbx(
            input_tree=resolution.input_tree,
            output_fbx=original_fbx,
            args=args,
        )

    if args.skip_request:
        warn("--skip_request set. Stopping after original FBX export / artifact generation.")
        validation_info["warnings"].append("skip_request enabled; data2_ue_lod Fitbox stage was not generated.")
    else:
        ue_request_info = post_ue_reduce_bones(
            url=str(args.url),
            input_fbx=original_fbx,
            output_fbx=ue_bone_lod_fbx,
            bones_to_remove=list(bone_plan["bones_to_remove"]),
            timeout=int(args.timeout),
            options={
                "preserve_mesh": True,
                "keep_triangle_count_percentage": 1.0,
                "export_reduced_lod_only": True,
            },
        )

        # Update request JSON after response.
        request_record.update(ue_request_info)
        dump_json(Path(artifact_json_paths["request_json"]), request_record)

        if args.skip_reimport:
            warn("--skip_reimport set. UE FBX was downloaded but Fitbox data2_ue_lod stage was not generated.")
            validation_info["warnings"].append("skip_reimport enabled; Fitbox stage was not generated.")
        else:
            import_ue_fbx_scene(fbx_path=ue_bone_lod_fbx)

            armature_obj = choose_single_armature()
            mesh_objects = collect_mesh_objects()

            stage_export_info = convert_imported_ue_scene_to_fitbox_stage(
                output_tree=resolution.output_tree,
                armature_obj=armature_obj,
                mesh_objects=mesh_objects,
                full_skeleton=full_skeleton,
                full_animation=full_animation,
                target_skeleton=reduced_skeleton,
                bones_to_remove=list(bone_plan["bones_to_remove"]),
                effective_variant_name=effective_variant_name,
                args=args,
            )

            validation_info.update(
                {
                    "removed_bones_absent": stage_export_info["removed_bones_absent"],
                    "target_bone_set_matches": stage_export_info["target_bone_set_matches"],
                    "ue_bone_order_differs_from_target": stage_export_info["ue_bone_order_differs_from_target"],
                    "mesh_count": stage_export_info["mesh_count"],
                    "extra_ue_bones": stage_export_info["extra_ue_bones"],
                }
            )

            if args.save_debug_blend:
                bpy = require_bpy()
                debug_blend = resolution.artifact_dir / "debug_imported_ue_bone_lod_scene.blend"
                result = bpy.ops.wm.save_as_mainfile(filepath=str(debug_blend))
                if result != {"FINISHED"}:
                    warn(f"Failed to save debug UE blend: {debug_blend}, result={result}")
                else:
                    log(f"Saved debug UE imported blend: {debug_blend}")

    summary = {
        "schema": "ogd_lbs_ssdr_step02_ue_lod_v1",
        "input_tree": str(resolution.input_tree),
        "output_tree": str(resolution.output_tree),
        "artifact_dir": str(resolution.artifact_dir),
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
        "extra_remove_bones": extra_remove_info,
        "skeleton": {
            "full_bone_count": int(bone_plan["full_bone_count"]),
            "target_bone_count": int(bone_plan["target_bone_count"]),
            "bones_to_remove_count": int(bone_plan["bones_to_remove_count"]),
            "root_bone_name": bone_plan["root_bone_name"],
            "full_bone_names": bone_plan["full_bone_names"],
            "target_bone_names": bone_plan["target_bone_names"],
            "bones_to_remove": bone_plan["bones_to_remove"],
        },
        "ue_service": {
            "url": args.url,
            "timeout": int(args.timeout),
            "request": ue_request_info,
        },
        "artifacts": {
            "original_fbx": str(original_fbx),
            "ue_bone_lod_fbx": str(ue_bone_lod_fbx),
            **artifact_json_paths,
        },
        "export": export_info,
        "outputs": stage_export_info,
        "validation": validation_info,
        "flags": {
            "skip_fbx_export": bool(args.skip_fbx_export),
            "skip_request": bool(args.skip_request),
            "skip_reimport": bool(args.skip_reimport),
            "allow_remove_root": bool(args.allow_remove_root),
            "allow_empty_remove": bool(args.allow_empty_remove),
            "allow_ue_extra_bones": bool(args.allow_ue_extra_bones),
            "require_mesh_names_match": bool(args.require_mesh_names_match),
        },
    }

    if not bool(args.no_summary):
        summary_path = resolution.output_tree / "step02_ue_lod_summary.json"
        write_stage_summary(summary_path, summary)
        summary["summary_path"] = str(summary_path)
        log(f"Step 02 UE LOD summary written: {summary_path}")

    manifest_obj.update(
        {
            "completed_at_unix": time.time(),
            "summary": summary,
        }
    )
    dump_json(Path(artifact_json_paths["manifest_json"]), manifest_obj)

    log("SSDR Step 02 UE LOD complete.")
    log(f"output_tree             : {resolution.output_tree}")
    log(f"original_fbx            : {original_fbx}")
    log(f"ue_bone_lod_fbx         : {ue_bone_lod_fbx}")
    log(f"bones_to_remove_count   : {bone_plan['bones_to_remove_count']}")
    log(f"stage_generated         : {stage_export_info is not None}")


if __name__ == "__main__":
    main()
