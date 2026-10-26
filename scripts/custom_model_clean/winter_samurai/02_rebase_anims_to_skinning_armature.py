#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Winter Samurai custom clean Step 02.

Input:
    clean01 blend from 01_import_model_and_anims.py

Goal:
    - identify the mesh-bound skinning armature
    - sample imported animation armatures' evaluated bone world matrices
    - bake clean WS_* Actions onto the skinning armature
    - remove imported animation armatures
    - remove pre-rebase Actions
    - validate before save and after reload

Generic operations are delegated to:
    ogd_lbs.blender.pose_bake
    ogd_lbs.blender.armature_ops
    ogd_lbs.blender.transforms
"""

import argparse
import csv
import json
import math
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bpy
import numpy as np

# -----------------------------------------------------------------------------
# Standalone script bootstrap:
# Blender may not honor PYTHONPATH by default. Add <repo>/src before importing
# ogd_lbs so this script can run via:
#   blender -b --python this_script.py -- ...
# -----------------------------------------------------------------------------
def _ensure_repo_src_on_syspath() -> None:
    current = Path(__file__).expanduser().resolve(strict=False)
    for parent in [current.parent, *current.parents]:
        src = parent / "src"
        if (parent / "pyproject.toml").exists() and (src / "ogd_lbs").is_dir():
            src_str = str(src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return

_ensure_repo_src_on_syspath()
# -----------------------------------------------------------------------------

from ogd_lbs.blender.context import update_view_layer
from ogd_lbs.blender.object import ensure_object_mode
from ogd_lbs.blender.armature_ops import mesh_binding_info
from ogd_lbs.blender.pose_bake import (
    capture_bone_world_matrices,
    bake_world_matrices_to_action,
    validate_action_world_matrices,
    frame_set_subframe,
)
from ogd_lbs.blender.action_utils import set_armature_action
from ogd_lbs.blender.transforms import matrix_error_arrays


TAG = "[ws-custom-clean-02]"


def log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{TAG}[WARN] {msg}", flush=True)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"Expected JSON object: {path}")
    return obj


def resolve_repo_root() -> Path | None:
    this_file = Path(__file__).resolve(strict=False)
    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return None


def open_blend(path: Path) -> None:
    require(path.is_file(), f"Input blend not found: {path}")
    log(f"Opening blend: {path}")
    result = bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    require(result == {"FINISHED"}, f"open_mainfile failed: {result}")
    ensure_object_mode()
    update_view_layer()


def save_blend(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Saving blend: {path}")
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    require(result == {"FINISHED"}, f"save_as_mainfile failed: {result}")


def get_armatures() -> list[bpy.types.Object]:
    return sorted([obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"], key=lambda obj: obj.name)


def get_meshes() -> list[bpy.types.Object]:
    return sorted([obj for obj in bpy.context.scene.objects if obj.type == "MESH"], key=lambda obj: obj.name)


def object_animation_info(obj: bpy.types.ID) -> dict[str, Any]:
    ad = getattr(obj, "animation_data", None)

    if ad is None:
        return {
            "present": False,
            "action": None,
            "nla_tracks": 0,
            "nla_strips": 0,
            "drivers": 0,
            "score": 0,
        }

    action = ad.action.name if ad.action else None
    nla_tracks = int(len(ad.nla_tracks))
    nla_strips = int(sum(len(track.strips) for track in ad.nla_tracks))

    try:
        drivers = int(len(ad.drivers))
    except Exception:
        drivers = 0

    score = 0
    if action:
        score += 10
    score += nla_strips * 5
    score += drivers

    return {
        "present": bool(action or nla_strips or drivers),
        "action": action,
        "nla_tracks": nla_tracks,
        "nla_strips": nla_strips,
        "drivers": drivers,
        "score": score,
    }


def summarize_armature(obj: bpy.types.Object) -> dict[str, Any]:
    return {
        "name": obj.name,
        "data_name": obj.data.name,
        "bone_count": int(len(obj.data.bones)),
        "root_bones": [bone.name for bone in obj.data.bones if bone.parent is None],
        "bone_names_first_100": [bone.name for bone in obj.data.bones][:100],
        "mesh_binding": mesh_binding_info(obj),
        "object_animation": object_animation_info(obj),
        "data_animation": object_animation_info(obj.data),
    }


def collect_scene_summary() -> dict[str, Any]:
    object_type_counts: dict[str, int] = {}

    for obj in bpy.data.objects:
        object_type_counts[obj.type] = object_type_counts.get(obj.type, 0) + 1

    armatures = get_armatures()
    meshes = get_meshes()

    return {
        "object_count": int(len(bpy.data.objects)),
        "object_type_counts": object_type_counts,
        "armature_count": int(len(armatures)),
        "mesh_count": int(len(meshes)),
        "empty_count": int(len([obj for obj in bpy.data.objects if obj.type == "EMPTY"])),
        "action_count": int(len(bpy.data.actions)),
        "armature_names": [obj.name for obj in armatures],
        "mesh_names_first_200": [obj.name for obj in meshes[:200]],
        "action_names_first_200": [action.name for action in sorted(bpy.data.actions, key=lambda a: a.name)[:200]],
        "armatures": [summarize_armature(obj) for obj in armatures],
    }


def select_roles(
    *,
    skinning_armature_name: str | None,
    animation_armature_names: list[str],
) -> tuple[bpy.types.Object, list[bpy.types.Object], list[dict[str, Any]]]:
    armatures = get_armatures()
    require(len(armatures) >= 2, f"Expected at least 2 armatures, got {len(armatures)}.")

    evidence: list[dict[str, Any]] = []

    for arm in armatures:
        evidence.append(
            {
                "name": arm.name,
                "bone_count": int(len(arm.data.bones)),
                "mesh_binding": mesh_binding_info(arm),
                "object_animation": object_animation_info(arm),
                "data_animation": object_animation_info(arm.data),
            }
        )

    if skinning_armature_name:
        skin = bpy.data.objects.get(skinning_armature_name)
        require(
            skin is not None and skin.type == "ARMATURE",
            f"Invalid --skinning_armature_name: {skinning_armature_name}",
        )
    else:
        bound = [arm for arm in armatures if mesh_binding_info(arm)["binding_score"] > 0]
        require(
            len(bound) == 1,
            "Expected exactly one mesh-bound skinning armature. "
            f"Evidence={json.dumps(evidence, indent=2, ensure_ascii=False)}",
        )
        skin = bound[0]

    if animation_armature_names:
        anims: list[bpy.types.Object] = []
        for name in animation_armature_names:
            obj = bpy.data.objects.get(name)
            require(obj is not None and obj.type == "ARMATURE", f"Invalid animation armature: {name}")
            require(obj is not skin, f"Animation armature cannot equal skinning armature: {name}")
            anims.append(obj)
    else:
        anims = [arm for arm in armatures if arm is not skin]

    require(anims, "No animation armatures selected.")

    skin_names = {bone.name for bone in skin.data.bones}
    require(skin_names, "Skinning armature has no bones.")

    bad_sets = []

    for anim in anims:
        anim_names = {bone.name for bone in anim.data.bones}
        if anim_names != skin_names:
            bad_sets.append(
                {
                    "animation_armature": anim.name,
                    "skinning_only": sorted(skin_names - anim_names)[:100],
                    "animation_only": sorted(anim_names - skin_names)[:100],
                }
            )

    require(
        not bad_sets,
        "Bone name sets differ between skinning and animation armatures:\n"
        + json.dumps(bad_sets, indent=2, ensure_ascii=False),
    )

    return skin, anims, evidence


def sanitize_label(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^Anim_Winter_Samurai_", "", text)
    text = re.sub(r"^Winter_Samurai_", "", text)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "Anim"


def label_from_fbx_path(path_text: str | None, fallback: str) -> str:
    if path_text:
        stem = Path(str(path_text)).stem
        return sanitize_label(stem)
    return sanitize_label(fallback)


def load_import_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None

    if not path.is_file():
        warn(f"Import report not found: {path}")
        return None

    obj = load_json(path)
    return obj


def action_frame_range_from_summaries(actions: list[dict[str, Any]]) -> tuple[int, int] | None:
    starts: list[int] = []
    ends: list[int] = []

    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("status") != "ok":
            continue

        fcurve_count = int(action.get("fcurve_count", 0) or 0)
        keyframe_count = int(action.get("keyframe_count", 0) or 0)

        if fcurve_count <= 0 and keyframe_count <= 0:
            continue

        frame_range = action.get("frame_range")
        if not isinstance(frame_range, list) or len(frame_range) != 2:
            continue

        starts.append(int(math.floor(float(frame_range[0]))))
        ends.append(int(math.ceil(float(frame_range[1]))))

    if not starts or not ends:
        return None

    return min(starts), max(ends)


def build_report_armature_metadata(import_report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if import_report is None:
        return {}

    out: dict[str, dict[str, Any]] = {}

    for idx, item in enumerate(import_report.get("animation_imports", []) or []):
        if not isinstance(item, dict):
            continue

        fbx_path = str(item.get("fbx_path", "") or "")
        label = label_from_fbx_path(fbx_path, fallback=f"anim_{idx:03d}")
        frame_range = action_frame_range_from_summaries(item.get("actions", []) or [])

        armature_names: list[str] = []

        for arm in item.get("armatures", []) or []:
            if not isinstance(arm, dict):
                continue
            name = arm.get("name")
            if isinstance(name, str) and name:
                armature_names.append(name)

        for arm_name in armature_names:
            out[arm_name] = {
                "source": "import_report",
                "import_index": int(idx),
                "fbx_path": fbx_path,
                "label": label,
                "frame_range": frame_range,
                "new_action_count": int(item.get("new_action_count", 0) or 0),
                "new_object_count": int(item.get("new_object_count", 0) or 0),
            }

    return out


def parse_repeated_names(items: list[str] | None) -> list[str]:
    out: list[str] = []

    for item in items or []:
        for chunk in str(item).split(","):
            text = chunk.strip()
            if text:
                out.append(text)

    return out


def build_source_plans(
    *,
    anims: list[bpy.types.Object],
    import_report: dict[str, Any] | None,
    scene_frame_start: int,
    scene_frame_end: int,
    frame_start_override: int | None,
    frame_end_override: int | None,
    samples_per_frame: int,
    action_name_template: str,
    only_labels: set[str],
) -> list[dict[str, Any]]:
    meta_by_arm = build_report_armature_metadata(import_report)

    plans: list[dict[str, Any]] = []
    used_action_names: set[str] = set()
    used_labels: set[str] = set()

    for idx, arm in enumerate(anims):
        meta = meta_by_arm.get(arm.name, {})
        label = str(meta.get("label") or sanitize_label(arm.name))

        base_label = label
        suffix_id = 2
        while label in used_labels:
            label = f"{base_label}_{suffix_id}"
            suffix_id += 1
        used_labels.add(label)

        if only_labels and label not in only_labels and base_label not in only_labels:
            continue

        if frame_start_override is not None:
            frame_start = int(frame_start_override)
        elif meta.get("frame_range") is not None:
            frame_start = int(meta["frame_range"][0])
        else:
            frame_start = int(scene_frame_start)

        if frame_end_override is not None:
            frame_end = int(frame_end_override)
        elif meta.get("frame_range") is not None:
            frame_end = int(meta["frame_range"][1])
        else:
            frame_end = int(scene_frame_end)

        require(frame_end >= frame_start, f"Invalid frame range for {arm.name}: {frame_start}..{frame_end}")
        require(samples_per_frame >= 1, "--samples_per_frame must be >= 1")

        sample_count = (frame_end - frame_start) * int(samples_per_frame) + 1
        times = frame_start + np.arange(sample_count, dtype=np.float64) / float(samples_per_frame)

        action_name = action_name_template.format(
            label=label,
            armature=arm.name,
            index=idx,
        )
        action_name = str(action_name).strip()
        require(action_name, f"Resolved empty action name for animation armature {arm.name}")

        base_action_name = action_name
        action_suffix = 2
        while action_name in used_action_names:
            action_name = f"{base_action_name}_{action_suffix}"
            action_suffix += 1
        used_action_names.add(action_name)

        plans.append(
            {
                "index": int(idx),
                "armature_name": arm.name,
                "label": label,
                "base_label": base_label,
                "action_name": action_name,
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "samples_per_frame": int(samples_per_frame),
                "sample_count": int(sample_count),
                "times": times,
                "metadata": meta,
            }
        )

    require(plans, "No animation source plans were selected.")
    return plans


def remove_animation_armatures(source_armature_names: list[str]) -> dict[str, Any]:
    ensure_object_mode()

    removed: list[str] = []
    missing: list[str] = []
    failed: list[dict[str, str]] = []

    for name in source_armature_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
            continue

        try:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed.append(name)
        except Exception as exc:
            failed.append({"name": name, "error": str(exc)})

    update_view_layer()

    return {
        "requested": source_armature_names,
        "removed": removed,
        "missing": missing,
        "failed": failed,
        "removed_count": int(len(removed)),
        "missing_count": int(len(missing)),
        "failed_count": int(len(failed)),
    }


def remove_old_actions(*, keep_action_names: set[str]) -> dict[str, Any]:
    removed: list[str] = []
    kept: list[str] = []
    failed: list[dict[str, str]] = []

    for action in list(bpy.data.actions):
        if action.name in keep_action_names:
            action.use_fake_user = True
            kept.append(action.name)
            continue

        name = action.name
        try:
            bpy.data.actions.remove(action, do_unlink=True)
            removed.append(name)
        except Exception as exc:
            failed.append({"name": name, "error": str(exc)})

    return {
        "keep_action_names": sorted(keep_action_names),
        "kept": sorted(kept),
        "removed": sorted(removed),
        "failed": failed,
        "kept_count": int(len(kept)),
        "removed_count": int(len(removed)),
        "failed_count": int(len(failed)),
    }


def purge_orphans() -> dict[str, Any]:
    before = {
        "objects": int(len(bpy.data.objects)),
        "armatures": int(len(bpy.data.armatures)),
        "actions": int(len(bpy.data.actions)),
        "meshes": int(len(bpy.data.meshes)),
        "materials": int(len(bpy.data.materials)),
    }

    results: list[str] = []

    for _ in range(3):
        try:
            result = bpy.ops.outliner.orphans_purge(
                do_local_ids=True,
                do_linked_ids=True,
                do_recursive=True,
            )
            results.append(str(result))
        except Exception as exc:
            results.append(f"failed: {exc}")
            break

    after = {
        "objects": int(len(bpy.data.objects)),
        "armatures": int(len(bpy.data.armatures)),
        "actions": int(len(bpy.data.actions)),
        "meshes": int(len(bpy.data.meshes)),
        "materials": int(len(bpy.data.materials)),
    }

    return {"before": before, "after": after, "operator_results": results}


def set_final_active_action(
    *,
    skinning_name: str,
    generated_action_names: list[str],
    mode: str,
) -> dict[str, Any]:
    skinning = bpy.data.objects.get(skinning_name)
    require(skinning is not None and skinning.type == "ARMATURE", f"Missing skinning armature: {skinning_name}")

    if mode == "none":
        skinning.animation_data_clear()
        return {"mode": mode, "active_action": None}

    if mode == "first":
        require(generated_action_names, "No generated actions available for final_active_action=first.")
        action = bpy.data.actions.get(generated_action_names[0])
        require(action is not None, f"Generated action missing: {generated_action_names[0]}")
        set_armature_action(skinning, action, reset_pose=True)
        return {"mode": mode, "active_action": action.name}

    raise RuntimeError(f"Unsupported final active action mode: {mode}")


def save_cache_npz(
    path: Path,
    *,
    bone_names: list[str],
    plans: list[dict[str, Any]],
    source_world_by_action: dict[str, np.ndarray],
    source_object_world_by_action: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "bone_names": np.asarray(bone_names, dtype=str),
        "action_names": np.asarray([plan["action_name"] for plan in plans], dtype=str),
        "labels": np.asarray([plan["label"] for plan in plans], dtype=str),
        "source_armature_names": np.asarray([plan["armature_name"] for plan in plans], dtype=str),
    }

    for idx, plan in enumerate(plans):
        key = f"source_{idx:03d}"
        action_name = plan["action_name"]
        data[f"{key}_action_name"] = np.asarray(action_name, dtype=str)
        data[f"{key}_label"] = np.asarray(plan["label"], dtype=str)
        data[f"{key}_armature_name"] = np.asarray(plan["armature_name"], dtype=str)
        data[f"{key}_times"] = np.asarray(plan["times"], dtype=np.float64)
        data[f"{key}_source_world"] = source_world_by_action[action_name]
        data[f"{key}_source_object_world"] = source_object_world_by_action[action_name]

    np.savez_compressed(path, **data)


def write_validation_csv(
    path: Path,
    *,
    validation_errors: dict[str, dict[str, np.ndarray]],
    validation_times: dict[str, np.ndarray],
    bone_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "action_name",
        "bone",
        "translation_l2_max",
        "translation_l2_frame",
        "rotation_deg_max",
        "rotation_deg_frame",
        "linear_relative_max",
        "linear_relative_frame",
        "matrix_abs_max",
        "matrix_abs_frame",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for action_name, errors in validation_errors.items():
            times = validation_times[action_name]

            for bi, bone_name in enumerate(bone_names):
                row: dict[str, Any] = {
                    "action_name": action_name,
                    "bone": bone_name,
                }

                for key in ["translation_l2", "rotation_deg", "linear_relative", "matrix_abs"]:
                    values = np.asarray(errors[key])
                    fi = int(np.argmax(values[:, bi]))
                    row[f"{key}_max"] = float(values[fi, bi])
                    row[f"{key}_frame"] = float(times[fi])

                writer.writerow(row)


def write_text_block(
    *,
    output_blend: Path,
    report_path: Path,
    generated_action_names: list[str],
) -> None:
    text = bpy.data.texts.new("WINTER_SAMURAI_CUSTOM_CLEAN02_REBASE_NOTES")
    text.write("Winter Samurai custom clean Step 02\n")
    text.write("=" * 80 + "\n")
    text.write(f"output_blend: {output_blend}\n")
    text.write(f"report_json : {report_path}\n")
    text.write("\n")
    text.write("Generated rebased Actions on the skinning armature:\n")
    for name in generated_action_names:
        text.write(f"  - {name}\n")
    text.write("\n")
    text.write("This file should contain only the mesh-bound skinning armature and the new rebased Actions.\n")


def self_test() -> None:
    log(f"Self-test passed. bpy={bpy.app.version_string}")
    log(f"Repo root: {resolve_repo_root()}")


def execute(args: argparse.Namespace) -> None:
    input_blend = Path(args.input_blend).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_blend = (
        Path(args.output_blend).expanduser().resolve(strict=False)
        if args.output_blend
        else (output_dir / "Winter_Samurai_clean02_rebased.blend").resolve(strict=False)
    )

    require(input_blend.is_file(), f"Missing input blend: {input_blend}")
    require(output_blend != input_blend, "Refusing to overwrite input blend.")
    require(output_blend.suffix.lower() == ".blend", "Output must be a .blend file.")

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "rebase_report.json"
    cache_path = output_dir / "source_world_cache.npz"
    csv_path = output_dir / "world_pose_error_per_action_bone.csv"

    for p in [output_blend, report_path, cache_path, csv_path]:
        require(not p.exists(), f"Output already exists; use a fresh output directory: {p}")

    if args.import_report:
        import_report_path = Path(args.import_report).expanduser().resolve(strict=False)
    else:
        import_report_path = (input_blend.parent / "import_report.json").resolve(strict=False)

    import_report = load_import_report(import_report_path)

    report: dict[str, Any] = {
        "schema": "ogd_lbs_winter_samurai_custom_clean02_rebase_v1",
        "created_at_utc": utc_now_iso(),
        "repo_root": str(resolve_repo_root()),
        "bpy_version": bpy.app.version_string,
        "input_blend": str(input_blend),
        "output_dir": str(output_dir),
        "output_blend": str(output_blend),
        "import_report": str(import_report_path) if import_report_path is not None else None,
        "report_json": str(report_path),
        "source_world_cache": str(cache_path),
        "world_pose_error_csv": str(csv_path),
        "status": "running",
        "stage": "open",
    }

    try:
        open_blend(input_blend)

        scene = bpy.context.scene
        scene_frame_start = int(scene.frame_start)
        scene_frame_end = int(scene.frame_end)

        report["scene_initial"] = collect_scene_summary()

        report["stage"] = "select_roles"
        skinning_armature_name = str(args.skinning_armature_name).strip() if args.skinning_armature_name else None
        animation_armature_names = parse_repeated_names(args.animation_armature_name)

        skinning, anims, role_evidence = select_roles(
            skinning_armature_name=skinning_armature_name,
            animation_armature_names=animation_armature_names,
        )

        skinning_name = skinning.name
        bone_names = [bone.name for bone in skinning.data.bones]

        only_labels = set(sanitize_label(x) for x in parse_repeated_names(args.only_label))

        plans = build_source_plans(
            anims=anims,
            import_report=import_report,
            scene_frame_start=scene_frame_start,
            scene_frame_end=scene_frame_end,
            frame_start_override=args.frame_start,
            frame_end_override=args.frame_end,
            samples_per_frame=int(args.samples_per_frame),
            action_name_template=str(args.action_name_template),
            only_labels=only_labels,
        )

        generated_action_names = [plan["action_name"] for plan in plans]
        source_armature_names = [plan["armature_name"] for plan in plans]

        report["selected"] = {
            "skinning_armature": skinning_name,
            "animation_armatures": source_armature_names,
            "bone_count": int(len(bone_names)),
        }
        report["role_evidence"] = role_evidence
        report["config"] = {
            "samples_per_frame": int(args.samples_per_frame),
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "position_tol": float(args.position_tol),
            "rotation_tol_deg": float(args.rotation_tol_deg),
            "linear_tol": float(args.linear_tol),
            "action_name_template": str(args.action_name_template),
            "remove_animation_armatures": not bool(args.keep_animation_armatures),
            "remove_old_actions": not bool(args.keep_old_actions),
            "purge_orphans": not bool(args.no_orphan_purge),
            "final_active_action": str(args.final_active_action),
            "only_label": sorted(only_labels),
        }
        report["plans"] = [
            {k: v for k, v in plan.items() if k not in {"times"}}
            | {"times_first": float(plan["times"][0]), "times_last": float(plan["times"][-1])}
            for plan in plans
        ]

        log("=" * 96)
        log("Winter Samurai custom clean Step 02 begin")
        log(f"input_blend      : {input_blend}")
        log(f"output_blend     : {output_blend}")
        log(f"skinning armature: {skinning_name}")
        log(f"animation count  : {len(plans)}")
        for plan in plans:
            log(
                f"  - {plan['armature_name']} -> {plan['action_name']} "
                f"frames={plan['frame_start']}..{plan['frame_end']} samples={plan['sample_count']}"
            )
        log("=" * 96)

        report["stage"] = "capture_sources"
        source_world_by_action: dict[str, np.ndarray] = {}
        source_object_world_by_action: dict[str, np.ndarray] = {}

        for plan in plans:
            action_name = plan["action_name"]
            armature_name = plan["armature_name"]
            times = np.asarray(plan["times"], dtype=np.float64)

            source_arm = bpy.data.objects.get(armature_name)
            require(source_arm is not None and source_arm.type == "ARMATURE", f"Source armature missing: {armature_name}")

            world, obj_world = capture_bone_world_matrices(
                source_arm,
                bone_names,
                times,
                label=f"Capture SOURCE {action_name}",
            )

            source_world_by_action[action_name] = world
            source_object_world_by_action[action_name] = obj_world

        save_cache_npz(
            cache_path,
            bone_names=bone_names,
            plans=plans,
            source_world_by_action=source_world_by_action,
            source_object_world_by_action=source_object_world_by_action,
        )

        report["stage"] = "bake_actions"

        frame_set_subframe(float(plans[0]["frame_start"]))
        target_object_world = np.asarray(skinning.matrix_world, dtype=np.float64).copy()

        bake_summaries: list[dict[str, Any]] = []

        for plan in plans:
            action_name = plan["action_name"]
            times = np.asarray(plan["times"], dtype=np.float64)

            bake_summary = bake_world_matrices_to_action(
                target_armature=skinning,
                bone_names=bone_names,
                times=times,
                target_world_matrices=source_world_by_action[action_name],
                target_object_world=target_object_world,
                action_name=action_name,
                position_tol=float(args.position_tol),
                rotation_tol_deg=float(args.rotation_tol_deg),
                linear_tol=float(args.linear_tol),
                overwrite_action=False,
            )

            bake_summary["source_armature_name"] = plan["armature_name"]
            bake_summary["label"] = plan["label"]
            bake_summaries.append(bake_summary)

        report["bake"] = bake_summaries

        report["stage"] = "validate_before_cleanup"
        before_cleanup_validations: list[dict[str, Any]] = []
        before_cleanup_errors: dict[str, dict[str, np.ndarray]] = {}
        validation_times_by_action: dict[str, np.ndarray] = {}

        for plan in plans:
            action_name = plan["action_name"]
            times = np.asarray(plan["times"], dtype=np.float64)
            action = bpy.data.actions.get(action_name)
            require(action is not None, f"Generated action missing: {action_name}")

            validation, errors = validate_action_world_matrices(
                armature_obj=skinning,
                action=action,
                bone_names=bone_names,
                times=times,
                expected_world_matrices=source_world_by_action[action_name],
                position_tol=float(args.position_tol),
                rotation_tol_deg=float(args.rotation_tol_deg),
                linear_tol=float(args.linear_tol),
                label=f"Validate BEFORE CLEANUP {action_name}",
            )

            before_cleanup_validations.append(validation)
            before_cleanup_errors[action_name] = errors
            validation_times_by_action[action_name] = times

        report["validation_before_cleanup"] = before_cleanup_validations
        failed_before = [v for v in before_cleanup_validations if not v.get("passed", False)]
        require(not failed_before, f"Validation before cleanup failed. See {report_path}")

        report["stage"] = "cleanup"
        cleanup: dict[str, Any] = {}

        if bool(args.keep_animation_armatures):
            cleanup["animation_armatures"] = {
                "skipped": True,
                "reason": "--keep_animation_armatures",
                "source_armature_names": source_armature_names,
            }
        else:
            cleanup["animation_armatures"] = remove_animation_armatures(source_armature_names)

        if bool(args.keep_old_actions):
            cleanup["old_actions"] = {
                "skipped": True,
                "reason": "--keep_old_actions",
                "kept_all_action_count": int(len(bpy.data.actions)),
            }
        else:
            cleanup["old_actions"] = remove_old_actions(keep_action_names=set(generated_action_names))

        if bool(args.no_orphan_purge):
            cleanup["orphan_purge"] = {"skipped": True, "reason": "--no_orphan_purge"}
        else:
            cleanup["orphan_purge"] = purge_orphans()

        final_active = set_final_active_action(
            skinning_name=skinning_name,
            generated_action_names=generated_action_names,
            mode=str(args.final_active_action),
        )
        cleanup["final_active_action"] = final_active

        report["cleanup"] = cleanup
        report["scene_after_cleanup"] = collect_scene_summary()

        report["stage"] = "validate_before_save"
        before_save_validations: list[dict[str, Any]] = []
        before_save_errors: dict[str, dict[str, np.ndarray]] = {}

        for plan in plans:
            action_name = plan["action_name"]
            times = np.asarray(plan["times"], dtype=np.float64)
            action = bpy.data.actions.get(action_name)
            require(action is not None, f"Generated action missing before save: {action_name}")

            validation, errors = validate_action_world_matrices(
                armature_obj=skinning,
                action=action,
                bone_names=bone_names,
                times=times,
                expected_world_matrices=source_world_by_action[action_name],
                position_tol=float(args.position_tol),
                rotation_tol_deg=float(args.rotation_tol_deg),
                linear_tol=float(args.linear_tol),
                label=f"Validate BEFORE SAVE {action_name}",
            )

            before_save_validations.append(validation)
            before_save_errors[action_name] = errors

        report["validation_before_save"] = before_save_validations
        failed_before_save = [v for v in before_save_validations if not v.get("passed", False)]
        require(not failed_before_save, f"Validation before save failed. See {report_path}")

        write_validation_csv(
            csv_path,
            validation_errors=before_save_errors,
            validation_times=validation_times_by_action,
            bone_names=bone_names,
        )

        write_text_block(
            output_blend=output_blend,
            report_path=report_path,
            generated_action_names=generated_action_names,
        )

        report["stage"] = "save"
        dump_json(report_path, report)
        save_blend(output_blend)

        report["stage"] = "validate_after_reload"
        open_blend(output_blend)

        after_reload_scene = collect_scene_summary()
        report["scene_after_reload"] = after_reload_scene

        skinning_reloaded = bpy.data.objects.get(skinning_name)
        require(skinning_reloaded is not None and skinning_reloaded.type == "ARMATURE", f"Reloaded skinning armature missing: {skinning_name}")

        after_reload_validations: list[dict[str, Any]] = []
        after_reload_errors: dict[str, dict[str, np.ndarray]] = {}

        for plan in plans:
            action_name = plan["action_name"]
            times = np.asarray(plan["times"], dtype=np.float64)
            action = bpy.data.actions.get(action_name)
            require(action is not None, f"Generated action missing after reload: {action_name}")

            validation, errors = validate_action_world_matrices(
                armature_obj=skinning_reloaded,
                action=action,
                bone_names=bone_names,
                times=times,
                expected_world_matrices=source_world_by_action[action_name],
                position_tol=float(args.position_tol),
                rotation_tol_deg=float(args.rotation_tol_deg),
                linear_tol=float(args.linear_tol),
                label=f"Validate AFTER RELOAD {action_name}",
            )

            after_reload_validations.append(validation)
            after_reload_errors[action_name] = errors

        report["validation_after_reload"] = after_reload_validations
        failed_after_reload = [v for v in after_reload_validations if not v.get("passed", False)]
        require(not failed_after_reload, f"Validation after reload failed. See {report_path}")

        write_validation_csv(
            csv_path,
            validation_errors=after_reload_errors,
            validation_times=validation_times_by_action,
            bone_names=bone_names,
        )

        report["status"] = "passed"
        report["stage"] = "complete"
        report["generated_actions"] = generated_action_names
        report["scope"] = (
            "Generated Actions on the skinning armature reproduce sampled evaluated bone world matrices "
            "of imported animation armatures. Imported animation armatures and pre-rebase Actions are "
            "removed by default. Mesh topology and skinning rest data are not modified."
        )

        dump_json(report_path, report)

        log("=" * 96)
        log("Winter Samurai custom clean Step 02 PASSED")
        log(f"output_blend       : {output_blend}")
        log(f"rebase_report      : {report_path}")
        log(f"source_world_cache : {cache_path}")
        log(f"error_csv          : {csv_path}")
        log(f"final armatures    : {after_reload_scene.get('armature_count')}")
        log(f"final meshes       : {after_reload_scene.get('mesh_count')}")
        log(f"final empties      : {after_reload_scene.get('empty_count')}")
        log(f"final actions      : {after_reload_scene.get('action_count')}")
        log(f"generated actions  : {generated_action_names}")
        log("=" * 96)

    except Exception as exc:
        report["status"] = "failed"
        report["stage_failed"] = report.get("stage")
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        try:
            dump_json(report_path, report)
            log(f"Failure report written: {report_path}")
        except Exception:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Winter Samurai custom clean Step 02: rebase imported animation armatures to skinning armature Actions."
    )

    parser.add_argument("--self_test", action="store_true")

    parser.add_argument("--input_blend", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_blend", type=str, default=None)

    parser.add_argument(
        "--import_report",
        type=str,
        default=None,
        help="Step 01 import_report.json. If omitted, uses <input_blend_dir>/import_report.json if present.",
    )

    parser.add_argument("--skinning_armature_name", type=str, default=None)
    parser.add_argument("--animation_armature_name", type=str, action="append", default=None)
    parser.add_argument("--only_label", type=str, action="append", default=None)

    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--samples_per_frame", type=int, default=1)

    parser.add_argument("--action_name_template", type=str, default="WS_{label}")

    parser.add_argument("--position_tol", type=float, default=1e-4)
    parser.add_argument("--rotation_tol_deg", type=float, default=0.05)
    parser.add_argument("--linear_tol", type=float, default=1e-5)

    parser.add_argument("--keep_animation_armatures", action="store_true")
    parser.add_argument("--keep_old_actions", action="store_true")
    parser.add_argument("--no_orphan_purge", action="store_true")

    parser.add_argument("--final_active_action", type=str, choices=["first", "none"], default="first")

    args = parser.parse_args(script_argv())

    self_test()

    if args.self_test:
        return

    if not args.input_blend or not args.output_dir:
        parser.error("--input_blend and --output_dir are required unless --self_test is used.")

    for key in ["position_tol", "rotation_tol_deg", "linear_tol"]:
        value = float(getattr(args, key))
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{key} must be finite and positive.")

    if int(args.samples_per_frame) < 1:
        parser.error("--samples_per_frame must be >= 1.")

    execute(args)


if __name__ == "__main__":
    main()
