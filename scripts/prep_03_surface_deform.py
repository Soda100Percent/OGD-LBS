#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 03: Animation-Guided Surface Deform Binding + Initial LBS Weight Transfer.

Responsibilities:
    1. Align Armature to a specified bind frame, or REST pose.
    2. Transfer LBS weights from Original_Source_Mesh to main_mesh_VDB.
    3. Validate transferred weights using data-driven effective-weight classification:
        - non-deform bones
        - deform bones with effective weights
        - deform bones without effective weights
    4. Validate missing effective deform-bone mappings with a practical tolerance:
        - Missing bones with small source influence ratio can be tolerated.
        - Target mesh must have no unweighted / floating vertices.
        - Large missing mappings still fail.
        - Extra mappings still fail by default.
    5. Add Surface Deform modifier and bind it.
    6. Add Armature modifier for LBS baseline.
    7. Toggle final output drive mode: SURFACE or LBS.

Important:
    bpy.ops.object.data_transfer transfers data from ACTIVE object to SELECTED meshes.
    Therefore Original_Source_Mesh must be active, and main_mesh_VDB must be selected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import bpy


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def ensure_object_mode() -> None:
    try:
        active = bpy.context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def deselect_all() -> None:
    ensure_object_mode()
    try:
        bpy.ops.object.select_all(action="DESELECT")
    except Exception:
        pass


def set_active(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj


def update_depsgraph() -> None:
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
    except Exception:
        pass


def unhide_for_ops(obj: bpy.types.Object) -> dict[str, Any]:
    state = {
        "hide_get": obj.hide_get(),
        "hide_viewport": bool(obj.hide_viewport),
        "hide_render": bool(obj.hide_render),
    }

    try:
        obj.hide_set(False)
    except Exception:
        pass

    obj.hide_viewport = False
    obj.hide_render = False

    return state


def restore_visibility(obj: bpy.types.Object, state: dict[str, Any]) -> None:
    try:
        obj.hide_set(bool(state.get("hide_get", False)))
    except Exception:
        pass

    obj.hide_viewport = bool(state.get("hide_viewport", False))
    obj.hide_render = bool(state.get("hide_render", False))


def find_primary_armature() -> bpy.types.Object:
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]

    if not armatures:
        raise RuntimeError("No Armature found in the scene.")

    if len(armatures) > 1:
        print(
            f"⚠️ Multiple armatures found; using first one: {[obj.name for obj in armatures]}",
            flush=True,
        )

    return armatures[0]


def align_bind_pose(armature: bpy.types.Object, bind_frame: int) -> None:
    if bind_frame >= 0:
        print(f"\n🛑 Aligning to POSE frame {bind_frame} for transfer / binding...", flush=True)
        armature.data.pose_position = "POSE"
        bpy.context.scene.frame_set(int(bind_frame))
    else:
        print("\n🛑 Aligning to REST pose for transfer / binding...", flush=True)
        armature.data.pose_position = "REST"

    update_depsgraph()


# -----------------------------------------------------------------------------
# Data-driven bone / weight classification
# -----------------------------------------------------------------------------

def get_bone_sets(armature: bpy.types.Object) -> dict[str, set[str]]:
    all_bones = {bone.name for bone in armature.data.bones}
    deform_bones = {
        bone.name
        for bone in armature.data.bones
        if bool(getattr(bone, "use_deform", True))
    }
    non_deform_bones = all_bones - deform_bones

    return {
        "all_bones": all_bones,
        "deform_bones": deform_bones,
        "non_deform_bones": non_deform_bones,
    }


def collect_effective_vertex_group_names(
    mesh_obj: bpy.types.Object,
    *,
    eps: float,
) -> set[str]:
    """
    A vertex group is effective if at least one vertex assignment has weight > eps.
    """

    group_by_index = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    effective_names: set[str] = set()

    for vertex in mesh_obj.data.vertices:
        for assignment in vertex.groups:
            if float(assignment.weight) <= eps:
                continue

            group_name = group_by_index.get(assignment.group)
            if group_name is not None:
                effective_names.add(group_name)

    return effective_names


def compute_per_group_weight_stats(
    mesh_obj: bpy.types.Object,
    *,
    group_names: set[str] | None = None,
    eps: float,
) -> dict[str, dict[str, Any]]:
    """
    Compute per-vertex-group nonzero vertex count, total weight, and max weight.
    """

    vertex_count = len(mesh_obj.data.vertices)
    group_by_index = {vg.index: vg.name for vg in mesh_obj.vertex_groups}

    if group_names is None:
        allowed = set(group_by_index.values())
    else:
        allowed = set(group_names)

    stats: dict[str, dict[str, Any]] = {
        name: {
            "exists": mesh_obj.vertex_groups.get(name) is not None,
            "nonzero_vertices": 0,
            "vertex_count": vertex_count,
            "vertex_ratio": 0.0,
            "total_weight": 0.0,
            "max_weight": 0.0,
        }
        for name in allowed
    }

    for vertex in mesh_obj.data.vertices:
        seen_this_vertex: set[str] = set()

        for assignment in vertex.groups:
            if float(assignment.weight) <= eps:
                continue

            group_name = group_by_index.get(assignment.group)
            if group_name is None or group_name not in allowed:
                continue

            item = stats.setdefault(
                group_name,
                {
                    "exists": True,
                    "nonzero_vertices": 0,
                    "vertex_count": vertex_count,
                    "vertex_ratio": 0.0,
                    "total_weight": 0.0,
                    "max_weight": 0.0,
                },
            )

            item["total_weight"] += float(assignment.weight)
            item["max_weight"] = max(float(item["max_weight"]), float(assignment.weight))

            if group_name not in seen_this_vertex:
                item["nonzero_vertices"] += 1
                seen_this_vertex.add(group_name)

    for item in stats.values():
        item["vertex_ratio"] = float(item["nonzero_vertices"]) / float(max(vertex_count, 1))

    return stats


def count_weighted_vertices_for_groups(
    mesh_obj: bpy.types.Object,
    *,
    group_name_filter: set[str],
    eps: float,
) -> dict[str, Any]:
    group_by_index = {vg.index: vg.name for vg in mesh_obj.vertex_groups}

    vertex_count = len(mesh_obj.data.vertices)
    nonzero_vertices = 0
    unweighted_vertices = 0
    total_assignments = 0
    total_weight_sum = 0.0
    min_positive_weight_sum = None
    max_weight_sum = 0.0

    for vertex in mesh_obj.data.vertices:
        vertex_weight_sum = 0.0
        vertex_assignments = 0

        for assignment in vertex.groups:
            group_name = group_by_index.get(assignment.group)
            if group_name not in group_name_filter:
                continue

            weight = float(assignment.weight)
            if weight <= eps:
                continue

            vertex_weight_sum += weight
            vertex_assignments += 1

        if vertex_weight_sum > eps:
            nonzero_vertices += 1
            total_assignments += vertex_assignments
            total_weight_sum += vertex_weight_sum
            max_weight_sum = max(max_weight_sum, vertex_weight_sum)

            if min_positive_weight_sum is None:
                min_positive_weight_sum = vertex_weight_sum
            else:
                min_positive_weight_sum = min(min_positive_weight_sum, vertex_weight_sum)
        else:
            unweighted_vertices += 1

    return {
        "vertex_count": vertex_count,
        "nonzero_vertices": nonzero_vertices,
        "unweighted_vertices": unweighted_vertices,
        "nonzero_vertex_ratio": nonzero_vertices / max(vertex_count, 1),
        "unweighted_vertex_ratio": unweighted_vertices / max(vertex_count, 1),
        "total_assignments": total_assignments,
        "avg_assignments_per_nonzero_vertex": total_assignments / max(nonzero_vertices, 1),
        "avg_weight_sum_per_nonzero_vertex": total_weight_sum / max(nonzero_vertices, 1),
        "min_positive_weight_sum": min_positive_weight_sum,
        "max_weight_sum": max_weight_sum,
    }


def classify_mesh_bone_weights(
    *,
    armature: bpy.types.Object,
    mesh_obj: bpy.types.Object,
    eps: float,
) -> dict[str, Any]:
    """
    Classify bones against a mesh by real vertex weights.

    Categories:
        non_deform_bones:
            bone.use_deform == False

        deform_effective_bones:
            bone.use_deform == True
            and mesh has same-name vertex group
            and at least one vertex weight > eps

        deform_invalid_bones:
            bone.use_deform == True
            and either:
                - mesh has no same-name vertex group
                - or same-name vertex group exists but has no weight > eps

    This function does not use name heuristics such as Nub/End/root/leaf.
    """

    bone_sets = get_bone_sets(armature)
    all_bones = bone_sets["all_bones"]
    deform_bones = bone_sets["deform_bones"]
    non_deform_bones = bone_sets["non_deform_bones"]

    vertex_group_names = {vg.name for vg in mesh_obj.vertex_groups}
    effective_group_names = collect_effective_vertex_group_names(mesh_obj, eps=eps)

    deform_effective_bones = deform_bones & effective_group_names
    deform_missing_group_bones = deform_bones - vertex_group_names
    deform_zero_weight_bones = (deform_bones & vertex_group_names) - effective_group_names
    deform_invalid_bones = deform_missing_group_bones | deform_zero_weight_bones

    non_deform_with_groups = non_deform_bones & vertex_group_names
    non_deform_effective_groups = non_deform_bones & effective_group_names

    effective_non_bone_groups = effective_group_names - all_bones
    all_non_bone_groups = vertex_group_names - all_bones

    deform_effective_stats = count_weighted_vertices_for_groups(
        mesh_obj,
        group_name_filter=deform_effective_bones,
        eps=eps,
    )

    per_deform_group_stats = compute_per_group_weight_stats(
        mesh_obj,
        group_names=deform_bones,
        eps=eps,
    )

    return {
        "mesh_name": mesh_obj.name,
        "all_bones": all_bones,
        "deform_bones": deform_bones,
        "non_deform_bones": non_deform_bones,
        "vertex_group_names": vertex_group_names,
        "effective_group_names": effective_group_names,
        "deform_effective_bones": deform_effective_bones,
        "deform_invalid_bones": deform_invalid_bones,
        "deform_missing_group_bones": deform_missing_group_bones,
        "deform_zero_weight_bones": deform_zero_weight_bones,
        "non_deform_with_groups": non_deform_with_groups,
        "non_deform_effective_groups": non_deform_effective_groups,
        "all_non_bone_groups": all_non_bone_groups,
        "effective_non_bone_groups": effective_non_bone_groups,
        "deform_effective_stats": deform_effective_stats,
        "per_deform_group_stats": per_deform_group_stats,
    }


def print_name_sample(
    label: str,
    names: set[str],
    *,
    limit: int = 20,
    prefix: str = "      ",
) -> None:
    names_sorted = sorted(names)
    print(f"{prefix}{label}: {len(names_sorted)}", flush=True)

    if names_sorted:
        print(f"{prefix}  first {min(limit, len(names_sorted))}: {names_sorted[:limit]}", flush=True)


def print_classification_report(title: str, cls: dict[str, Any]) -> None:
    stats = cls["deform_effective_stats"]

    print(f"\n📊 {title}: {cls['mesh_name']}", flush=True)
    print(f"   Total bones:        {len(cls['all_bones'])}", flush=True)
    print(f"   Non-deform bones:   {len(cls['non_deform_bones'])}", flush=True)
    print(f"   Deform bones:       {len(cls['deform_bones'])}", flush=True)
    print(f"   Vertex groups:      {len(cls['vertex_group_names'])}", flush=True)
    print(f"   Effective groups:   {len(cls['effective_group_names'])}", flush=True)

    print("\n   Deform-bone classification:", flush=True)
    print(f"      Effective deform bones:      {len(cls['deform_effective_bones'])}", flush=True)
    print(f"      Invalid deform bones:        {len(cls['deform_invalid_bones'])}", flush=True)
    print(f"        - Missing vertex group:    {len(cls['deform_missing_group_bones'])}", flush=True)
    print(f"        - Zero/empty vertex group: {len(cls['deform_zero_weight_bones'])}", flush=True)

    print("\n   Effective deform-weight vertex coverage:", flush=True)
    print(
        f"      nonzero vertices: {stats['nonzero_vertices']} / {stats['vertex_count']} "
        f"({stats['nonzero_vertex_ratio']:.6f})",
        flush=True,
    )
    print(
        f"      unweighted vertices: {stats['unweighted_vertices']} / {stats['vertex_count']} "
        f"({stats['unweighted_vertex_ratio']:.6f})",
        flush=True,
    )
    print(
        f"      avg assignments / nonzero vertex: "
        f"{stats['avg_assignments_per_nonzero_vertex']:.6f}",
        flush=True,
    )
    print(
        f"      avg weight sum / nonzero vertex: "
        f"{stats['avg_weight_sum_per_nonzero_vertex']:.6f}",
        flush=True,
    )

    # Informational diagnostics only.
    if cls["non_deform_effective_groups"]:
        print("\n   ℹ️ Non-deform bones with nonzero same-name vertex groups:", flush=True)
        print_name_sample(
            "non_deform_effective_groups",
            cls["non_deform_effective_groups"],
            prefix="      ",
        )

    if cls["effective_non_bone_groups"]:
        print("\n   ℹ️ Effective vertex groups not matching any bone:", flush=True)
        print_name_sample(
            "effective_non_bone_groups",
            cls["effective_non_bone_groups"],
            prefix="      ",
        )


def print_missing_bone_stats(
    *,
    title: str,
    bone_names: set[str],
    source_cls: dict[str, Any],
    limit: int = 50,
) -> None:
    per_stats = source_cls["per_deform_group_stats"]
    sorted_names = sorted(bone_names)

    print(f"   {title}: {len(sorted_names)}", flush=True)

    for name in sorted_names[:limit]:
        item = per_stats.get(name, {})
        exists = item.get("exists", False)
        nonzero = int(item.get("nonzero_vertices", 0))
        vertex_count = int(item.get("vertex_count", 0))
        ratio = float(item.get("vertex_ratio", 0.0))
        total_weight = float(item.get("total_weight", 0.0))
        max_weight = float(item.get("max_weight", 0.0))

        print(
            f"     - {name}: "
            f"source_group_exists={exists}, "
            f"source_nonzero={nonzero}/{vertex_count}, "
            f"source_ratio={ratio:.6f}, "
            f"source_total_weight={total_weight:.6f}, "
            f"source_max_weight={max_weight:.6f}",
            flush=True,
        )

    if len(sorted_names) > limit:
        print(f"     ... truncated {len(sorted_names) - limit} more", flush=True)


def validate_effective_deform_weight_mapping(
    *,
    source_cls: dict[str, Any],
    target_cls: dict[str, Any],
    min_nonzero_weight_ratio: float,
    missing_bone_max_source_vertex_ratio: float,
    strict_effective_bone_set: bool,
    allow_weight_mismatch: bool,
) -> None:
    """
    Validation logic:

    Strict mode:
        Source_Deform_Effective must equal Target_Deform_Effective.

    Default practical mode:
        Missing mappings are tolerated only if:
            1. each missing bone influences less than
               missing_bone_max_source_vertex_ratio of source vertices;
            2. target has zero unweighted vertices under effective deform-bone groups.

        Extra mappings still fail by default.

    Missing:
        source has effective deform-bone weights, target lost them.

    Extra:
        source has no effective deform-bone weights, target gained them.
    """

    source_effective = set(source_cls["deform_effective_bones"])
    target_effective = set(target_cls["deform_effective_bones"])

    missing_mappings = source_effective - target_effective
    extra_mappings = target_effective - source_effective

    source_invalid = set(source_cls["deform_invalid_bones"])
    target_invalid = set(target_cls["deform_invalid_bones"])

    target_stats = target_cls["deform_effective_stats"]
    target_unweighted_vertices = int(target_stats["unweighted_vertices"])

    print("\n🔎 Effective deform-weight validation:", flush=True)
    print(f"   Source effective deform bones: {len(source_effective)}", flush=True)
    print(f"   Target effective deform bones: {len(target_effective)}", flush=True)
    print(f"   Source invalid deform bones:   {len(source_invalid)}", flush=True)
    print(f"   Target invalid deform bones:   {len(target_invalid)}", flush=True)

    print(f"   Missing mappings: {len(missing_mappings)}", flush=True)
    print(f"   Extra mappings:   {len(extra_mappings)}", flush=True)

    print(
        f"   Target nonzero effective-deform vertex ratio: "
        f"{target_stats['nonzero_vertex_ratio']:.6f}",
        flush=True,
    )
    print(f"   Target unweighted vertices: {target_unweighted_vertices}", flush=True)

    problems: list[str] = []

    if not source_effective:
        problems.append(
            "Source has zero effective deform-bone weights. "
            "There is nothing meaningful to transfer."
        )

    if target_stats["nonzero_vertex_ratio"] < min_nonzero_weight_ratio:
        problems.append(
            f"Too few target vertices have effective deform weights: "
            f"{target_stats['nonzero_vertex_ratio']:.6f} < "
            f"{min_nonzero_weight_ratio:.6f}."
        )

    if extra_mappings:
        print_missing_bone_stats(
            title="❌ Extra mappings (source invalid/empty -> target gained)",
            bone_names=extra_mappings,
            source_cls=source_cls,
        )
        problems.append(
            f"Target gained {len(extra_mappings)} unexpected effective deform-bone mappings."
        )

    tolerated_missing: set[str] = set()
    large_missing: set[str] = set()

    if missing_mappings:
        source_per_stats = source_cls["per_deform_group_stats"]

        for bone_name in missing_mappings:
            item = source_per_stats.get(bone_name, {})
            ratio = float(item.get("vertex_ratio", 0.0))

            if ratio < float(missing_bone_max_source_vertex_ratio):
                tolerated_missing.add(bone_name)
            else:
                large_missing.add(bone_name)

        if strict_effective_bone_set:
            print_missing_bone_stats(
                title="❌ Missing mappings under strict mode",
                bone_names=missing_mappings,
                source_cls=source_cls,
            )
            problems.append(
                f"Strict mode requires exact effective-bone-set equality; "
                f"target lost {len(missing_mappings)} mappings."
            )

        else:
            if large_missing:
                print_missing_bone_stats(
                    title=(
                        "❌ Large missing mappings "
                        f"(source ratio >= {missing_bone_max_source_vertex_ratio:.6f})"
                    ),
                    bone_names=large_missing,
                    source_cls=source_cls,
                )
                problems.append(
                    f"Target lost {len(large_missing)} large effective deform-bone mappings."
                )

            if tolerated_missing:
                if target_unweighted_vertices > 0:
                    print_missing_bone_stats(
                        title=(
                            "❌ Small missing mappings cannot be tolerated "
                            "because target has unweighted vertices"
                        ),
                        bone_names=tolerated_missing,
                        source_cls=source_cls,
                    )
                    problems.append(
                        f"Target lost {len(tolerated_missing)} small mappings, "
                        f"and has {target_unweighted_vertices} unweighted vertices."
                    )
                else:
                    print(
                        "\n⚠️ Tolerated small missing deform-bone mappings:",
                        flush=True,
                    )
                    print(
                        f"   count={len(tolerated_missing)}, "
                        f"threshold source_vertex_ratio < "
                        f"{missing_bone_max_source_vertex_ratio:.6f}, "
                        f"target_unweighted_vertices={target_unweighted_vertices}",
                        flush=True,
                    )
                    print_missing_bone_stats(
                        title="tolerated missing mappings",
                        bone_names=tolerated_missing,
                        source_cls=source_cls,
                    )

    if problems:
        msg = "\n".join(f"   - {p}" for p in problems)

        if allow_weight_mismatch:
            print(
                "\n⚠️ Weight validation found real mismatches, "
                "but --allow_weight_mismatch is set:",
                flush=True,
            )
            print(msg, flush=True)
            return

        raise RuntimeError(
            "Effective deform-weight validation failed:\n"
            f"{msg}\n"
            "Use --allow_weight_mismatch only if you intentionally want to inspect a broken transfer."
        )

    if missing_mappings and not strict_effective_bone_set:
        print("   ✅ Missing mappings are within tolerance and target has no floating vertices.", flush=True)
    else:
        print("   ✅ Effective deform-bone sets match exactly.", flush=True)

    print(
        f"   ✅ Target vertex coverage passes threshold "
        f"({target_stats['nonzero_vertex_ratio']:.6f} >= {min_nonzero_weight_ratio:.6f}).",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Weight transfer
# -----------------------------------------------------------------------------

def clear_and_create_target_vertex_groups(
    *,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> None:
    print("\n🧩 Rebuilding target vertex groups from source vertex-group names...", flush=True)

    target_mesh.vertex_groups.clear()

    created_count = 0
    for src_group in source_mesh.vertex_groups:
        target_mesh.vertex_groups.new(name=src_group.name)
        created_count += 1

    print(f"   -> Created {created_count} target vertex groups.", flush=True)


def transfer_lbs_weights_operator(
    *,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> None:
    """
    Correct Blender data_transfer direction:
        ACTIVE object is source.
        SELECTED non-active objects are destinations.
    """

    print("\n⚖️ Transferring LBS weights:", flush=True)
    print(f"   Source ACTIVE:        {source_mesh.name}", flush=True)
    print(f"   Destination selected: {target_mesh.name}", flush=True)
    print("   Mapping: POLYINTERP_NEAREST / Nearest Face Interpolated", flush=True)

    ensure_object_mode()
    deselect_all()

    source_mesh.select_set(True)
    target_mesh.select_set(True)
    set_active(source_mesh)

    update_depsgraph()

    try:
        result = bpy.ops.object.data_transfer(
            use_create=True,
            data_type="VGROUP_WEIGHTS",
            vert_mapping="POLYINTERP_NEAREST",
            layers_select_src="ALL",
            layers_select_dst="NAME",
            mix_mode="REPLACE",
            mix_factor=1.0,
        )
    except TypeError as exc:
        print(f"   ⚠️ Explicit data_transfer call failed with TypeError: {exc}", flush=True)
        print("   -> Retrying with reduced parameter set...", flush=True)

        result = bpy.ops.object.data_transfer(
            use_create=True,
            data_type="VGROUP_WEIGHTS",
            vert_mapping="POLYINTERP_NEAREST",
            layers_select_src="ALL",
            mix_mode="REPLACE",
            mix_factor=1.0,
        )

    print(f"   -> data_transfer result: {result}", flush=True)

    if result != {"FINISHED"}:
        raise RuntimeError(f"bpy.ops.object.data_transfer did not finish successfully: {result}")

    update_depsgraph()


def normalize_target_vertex_groups(target_mesh: bpy.types.Object) -> None:
    print("\n🧮 Normalizing target vertex groups...", flush=True)

    ensure_object_mode()
    deselect_all()

    target_mesh.select_set(True)
    set_active(target_mesh)

    try:
        result = bpy.ops.object.vertex_group_normalize_all(
            group_select_mode="ALL",
            lock_active=False,
        )
        print(f"   -> vertex_group_normalize_all result: {result}", flush=True)
    except Exception as exc:
        print(f"   ⚠️ vertex_group_normalize_all failed: {exc}", flush=True)

    update_depsgraph()


# -----------------------------------------------------------------------------
# Modifier setup
# -----------------------------------------------------------------------------

def remove_modifier_if_exists(obj: bpy.types.Object, name: str) -> None:
    for mod in list(obj.modifiers):
        if mod.name == name:
            obj.modifiers.remove(mod)


def ensure_source_triangulate(source_mesh: bpy.types.Object) -> bpy.types.Modifier:
    print("\n📐 Adding / refreshing source Triangulate modifier...", flush=True)

    remove_modifier_if_exists(source_mesh, "OGD_Triangulate")

    tri_mod = source_mesh.modifiers.new(name="OGD_Triangulate", type="TRIANGULATE")

    try:
        tri_mod.keep_custom_normals = True
    except Exception:
        pass

    try:
        tri_mod.quad_method = "BEAUTY"
    except Exception:
        pass

    try:
        tri_mod.ngon_method = "BEAUTY"
    except Exception:
        pass

    return tri_mod


def bind_surface_deform(
    *,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
    max_falloff: float,
    falloff_start: float,
    falloff_step: float,
) -> bpy.types.Modifier:
    print("\n🔗 Setting up Surface Deform modifier on target mesh...", flush=True)

    remove_modifier_if_exists(target_mesh, "OGD_SurfaceDeform")

    surf_mod = target_mesh.modifiers.new(name="OGD_SurfaceDeform", type="SURFACE_DEFORM")
    surf_mod.target = source_mesh

    ensure_object_mode()
    deselect_all()

    target_mesh.select_set(True)
    set_active(target_mesh)

    update_depsgraph()

    current_falloff = float(falloff_start)
    is_bound = False

    print("⚙️ Starting Surface Deform falloff auto-tuning...", flush=True)

    while current_falloff <= float(max_falloff) + 1e-9:
        surf_mod.falloff = current_falloff
        print(f"   -> Attempting bind with falloff = {current_falloff:.3f}", flush=True)

        try:
            with bpy.context.temp_override(
                active_object=target_mesh,
                object=target_mesh,
                selected_objects=[target_mesh],
                selected_editable_objects=[target_mesh],
            ):
                result = bpy.ops.object.surfacedeform_bind(modifier=surf_mod.name)

            print(f"      operator result: {result}", flush=True)

        except Exception as exc:
            print(f"      ⚠️ Bind operator exception: {exc}", flush=True)

        if bool(getattr(surf_mod, "is_bound", False)):
            print(
                f"   ✅ Surface Deform bound successfully with falloff = {current_falloff:.3f}",
                flush=True,
            )
            is_bound = True
            break

        print("      ❌ Bind failed; increasing falloff...", flush=True)
        current_falloff += float(falloff_step)

    if not is_bound:
        raise RuntimeError(
            f"Failed to bind Surface Deform up to falloff={max_falloff}. "
            "Try increasing --max_falloff or inspect source/target overlap."
        )

    return surf_mod


def add_armature_modifier(
    *,
    armature: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> bpy.types.Modifier:
    print("\n🦴 Adding / refreshing target Armature modifier...", flush=True)

    remove_modifier_if_exists(target_mesh, "OGD_Armature")

    arm_mod = target_mesh.modifiers.new(name="OGD_Armature", type="ARMATURE")
    arm_mod.object = armature

    if hasattr(arm_mod, "use_vertex_groups"):
        arm_mod.use_vertex_groups = True

    if hasattr(arm_mod, "use_bone_envelopes"):
        arm_mod.use_bone_envelopes = False

    if hasattr(arm_mod, "vertex_group"):
        arm_mod.vertex_group = ""

    return arm_mod


def apply_drive_mode(
    *,
    drive_mode: str,
    surf_mod: bpy.types.Modifier,
    arm_mod: bpy.types.Modifier,
) -> None:
    drive_mode = drive_mode.upper()

    print(f"\n🎛️ Applying output drive mode: {drive_mode}", flush=True)

    if drive_mode == "SURFACE":
        surf_mod.show_viewport = True
        surf_mod.show_render = True
        arm_mod.show_viewport = False
        arm_mod.show_render = False

        print("   Surface Deform: enabled", flush=True)
        print("   Armature/LBS:   disabled", flush=True)

    elif drive_mode == "LBS":
        surf_mod.show_viewport = False
        surf_mod.show_render = False
        arm_mod.show_viewport = True
        arm_mod.show_render = True

        print("   Surface Deform: disabled", flush=True)
        print("   Armature/LBS:   enabled", flush=True)

    else:
        raise ValueError(f"Unsupported drive_mode: {drive_mode}")


def print_modifier_summary(target_mesh: bpy.types.Object) -> None:
    print("\n🧾 Target modifier summary:", flush=True)

    for mod in target_mesh.modifiers:
        extra = ""

        if mod.type == "ARMATURE":
            extra = f", object={getattr(getattr(mod, 'object', None), 'name', None)}"

        if mod.type == "SURFACE_DEFORM":
            extra = (
                f", target={getattr(getattr(mod, 'target', None), 'name', None)}, "
                f"is_bound={getattr(mod, 'is_bound', 'unknown')}, "
                f"falloff={getattr(mod, 'falloff', 'unknown')}"
            )

        print(
            f"   - {mod.name}: type={mod.type}, "
            f"show_viewport={mod.show_viewport}, show_render={mod.show_render}{extra}",
            flush=True,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prep 03: Surface Deform binding + initial LBS weight transfer "
            "with data-driven effective-weight validation and small-missing tolerance."
        )
    )

    parser.add_argument(
        "--input_blend",
        type=str,
        required=True,
        help="Path to input _prep_02.blend file.",
    )
    parser.add_argument(
        "--output_blend",
        type=str,
        default=None,
        help="Path to save output _prep_03.blend file.",
    )

    parser.add_argument(
        "--source_mesh",
        type=str,
        default="Original_Source_Mesh",
        help="Original weighted mesh used as weight/source-deform driver.",
    )
    parser.add_argument(
        "--target_mesh",
        type=str,
        default="main_mesh_VDB",
        help="Fused mesh used as target.",
    )

    parser.add_argument(
        "--bind_frame",
        type=int,
        default=-1,
        help="Animation frame to use for transfer and Surface Deform binding. -1 means REST.",
    )
    parser.add_argument(
        "--drive_mode",
        type=str,
        choices=["SURFACE", "LBS"],
        default="SURFACE",
        help="Which modifier drives the output blend viewport/render.",
    )

    parser.add_argument(
        "--max_falloff",
        type=float,
        default=16.0,
        help="Maximum Surface Deform falloff to try.",
    )
    parser.add_argument(
        "--falloff_start",
        type=float,
        default=4.0,
        help="Initial Surface Deform falloff.",
    )
    parser.add_argument(
        "--falloff_step",
        type=float,
        default=2.0,
        help="Surface Deform falloff increment.",
    )

    parser.add_argument(
        "--weight_eps",
        type=float,
        default=1e-8,
        help="Weight threshold. A weight <= eps is treated as zero. Default: 1e-8.",
    )
    parser.add_argument(
        "--min_nonzero_weight_ratio",
        type=float,
        default=0.50,
        help=(
            "Minimum ratio of target vertices that must have non-zero weights "
            "in effective deform-bone groups. Default: 0.50."
        ),
    )
    parser.add_argument(
        "--missing_bone_max_source_vertex_ratio",
        type=float,
        default=0.02,
        help=(
            "A missing deform-bone mapping can be tolerated only if that bone "
            "influences less than this ratio of source vertices, and the target "
            "has no unweighted vertices. Default: 0.02."
        ),
    )
    parser.add_argument(
        "--strict_effective_bone_set",
        action="store_true",
        help=(
            "Require exact equality between source and target effective deform-bone sets. "
            "Disables small-missing tolerance."
        ),
    )
    parser.add_argument(
        "--allow_weight_mismatch",
        action="store_true",
        help=(
            "Do not fail when validation finds real mismatches. "
            "Only use this for debugging broken transfers."
        ),
    )

    args = parser.parse_args(script_argv())

    input_path = Path(args.input_blend)
    if not input_path.exists():
        raise FileNotFoundError(f"Input blend not found: {input_path}")

    if args.output_blend:
        output_path = Path(args.output_blend)
    else:
        stem = input_path.stem.replace("_prep_02", "_prep_03")
        if "_prep_03" not in stem:
            stem += "_prep_03"
        output_path = input_path.parent / f"{stem}{input_path.suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading: {input_path}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    ensure_object_mode()

    armature = find_primary_armature()

    source_mesh = bpy.data.objects.get(args.source_mesh)
    target_mesh = bpy.data.objects.get(args.target_mesh)

    if source_mesh is None or source_mesh.type != "MESH":
        raise RuntimeError(f"Source mesh not found or not MESH: {args.source_mesh}")

    if target_mesh is None or target_mesh.type != "MESH":
        raise RuntimeError(f"Target mesh not found or not MESH: {args.target_mesh}")

    print("\n🎯 Core objects:", flush=True)
    print(f"   Armature: {armature.name} (bones={len(armature.data.bones)})", flush=True)
    print(
        f"   Source:   {source_mesh.name} "
        f"(verts={len(source_mesh.data.vertices)}, groups={len(source_mesh.vertex_groups)})",
        flush=True,
    )
    print(
        f"   Target:   {target_mesh.name} "
        f"(verts={len(target_mesh.data.vertices)}, groups={len(target_mesh.vertex_groups)})",
        flush=True,
    )

    source_vis = unhide_for_ops(source_mesh)
    target_vis = unhide_for_ops(target_mesh)

    try:
        align_bind_pose(armature, int(args.bind_frame))

        # Source classification before transfer.
        source_cls_before = classify_mesh_bone_weights(
            armature=armature,
            mesh_obj=source_mesh,
            eps=float(args.weight_eps),
        )
        print_classification_report(
            "Source classification before transfer",
            source_cls_before,
        )

        # Remove old OGD modifiers on target before rebuilding.
        for mod_name in ["OGD_SurfaceDeform", "OGD_Armature", "OGD_WeightTransfer"]:
            remove_modifier_if_exists(target_mesh, mod_name)

        # Correct-direction weight transfer.
        clear_and_create_target_vertex_groups(
            source_mesh=source_mesh,
            target_mesh=target_mesh,
        )
        transfer_lbs_weights_operator(
            source_mesh=source_mesh,
            target_mesh=target_mesh,
        )
        normalize_target_vertex_groups(target_mesh)

        # Target classification after transfer.
        target_cls_after = classify_mesh_bone_weights(
            armature=armature,
            mesh_obj=target_mesh,
            eps=float(args.weight_eps),
        )
        print_classification_report(
            "Target classification after transfer",
            target_cls_after,
        )

        validate_effective_deform_weight_mapping(
            source_cls=source_cls_before,
            target_cls=target_cls_after,
            min_nonzero_weight_ratio=float(args.min_nonzero_weight_ratio),
            missing_bone_max_source_vertex_ratio=float(args.missing_bone_max_source_vertex_ratio),
            strict_effective_bone_set=bool(args.strict_effective_bone_set),
            allow_weight_mismatch=bool(args.allow_weight_mismatch),
        )

        # Surface Deform.
        ensure_source_triangulate(source_mesh)

        surf_mod = bind_surface_deform(
            source_mesh=source_mesh,
            target_mesh=target_mesh,
            max_falloff=float(args.max_falloff),
            falloff_start=float(args.falloff_start),
            falloff_step=float(args.falloff_step),
        )

        # Armature/LBS.
        arm_mod = add_armature_modifier(
            armature=armature,
            target_mesh=target_mesh,
        )

        apply_drive_mode(
            drive_mode=args.drive_mode,
            surf_mod=surf_mod,
            arm_mod=arm_mod,
        )

        print_modifier_summary(target_mesh)

        print("\n▶️ Setting Armature to POSE mode for playback...", flush=True)
        armature.data.pose_position = "POSE"

        # Restore source hidden state, keep target visible.
        restore_visibility(source_mesh, source_vis)

        try:
            target_mesh.hide_set(False)
        except Exception:
            pass
        target_mesh.hide_viewport = False
        target_mesh.hide_render = False

        update_depsgraph()

        print("\n✅ Prep 03 complete.", flush=True)
        print(f"💾 Saving to: {output_path}", flush=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        print("🎉 Done!", flush=True)

    except Exception:
        try:
            restore_visibility(source_mesh, source_vis)
            restore_visibility(target_mesh, target_vis)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
