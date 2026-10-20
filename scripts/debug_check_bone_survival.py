#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug script: inspect why a specific bone survives Prep 04a bone pruning.

Typical use:
    python scripts/debug_check_bone_survival.py \
        --input_blend /path/to/Griffin_A2_prep_02.blend \
        --bone CC_Base_Pivot_0

It checks:
    - bone existence
    - use_deform
    - bone length
    - parent / children
    - same-name vertex group existence
    - weight stats under eps
    - classification under Prep 04a logic
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy


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


def find_armature(name: str | None = None) -> bpy.types.Object:
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]

    if not armatures:
        raise RuntimeError("No ARMATURE object found in scene.")

    if name:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise RuntimeError(f"Requested armature not found: {name}")
        if obj.type != "ARMATURE":
            raise RuntimeError(f"Requested object is not ARMATURE: {name}, type={obj.type}")
        return obj

    if len(armatures) > 1:
        print(
            f"⚠️ Multiple armatures found; using first one by default: "
            f"{[obj.name for obj in armatures]}",
            flush=True,
        )

    return armatures[0]


def find_mesh_for_bone_group(
    *,
    requested_mesh_name: str | None,
    bone_name: str,
) -> bpy.types.Object:
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]

    if not meshes:
        raise RuntimeError("No MESH object found in scene.")

    if requested_mesh_name:
        obj = bpy.data.objects.get(requested_mesh_name)
        if obj is None:
            raise RuntimeError(f"Requested mesh not found: {requested_mesh_name}")
        if obj.type != "MESH":
            raise RuntimeError(f"Requested object is not MESH: {requested_mesh_name}, type={obj.type}")
        return obj

    # Prefer current pipeline standard names.
    preferred_names = [
        "Original_Source_Mesh",
        "Griffin_A",
        "main_mesh_VDB",
    ]

    for name in preferred_names:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "MESH":
            return obj

    # Otherwise choose a mesh that has the target bone vertex group.
    for obj in meshes:
        if obj.vertex_groups.get(bone_name) is not None:
            return obj

    # Last fallback: first mesh.
    return meshes[0]


def vec3_tuple(v: Any) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def get_bone_info(
    *,
    armature: bpy.types.Object,
    bone_name: str,
    zero_length_eps: float,
) -> dict[str, Any]:
    bone = armature.data.bones.get(bone_name)

    if bone is None:
        return {
            "exists": False,
            "bone_name": bone_name,
        }

    head = bone.head_local
    tail = bone.tail_local
    length = float((tail - head).length)

    children = [child.name for child in bone.children]

    info = {
        "exists": True,
        "bone_name": bone.name,
        "armature_object": armature.name,
        "armature_data": armature.data.name,
        "use_deform": bool(getattr(bone, "use_deform", True)),
        "parent": bone.parent.name if bone.parent else None,
        "children": children,
        "child_count": len(children),
        "head_local": vec3_tuple(head),
        "tail_local": vec3_tuple(tail),
        "length": length,
        "is_zero_or_tiny_length": bool(length <= zero_length_eps),
        "zero_length_eps": float(zero_length_eps),
        "hide": bool(getattr(bone, "hide", False)),
        "use_connect": bool(getattr(bone, "use_connect", False)),
    }

    # Bone collections in Blender 4.x.
    try:
        info["collections"] = [coll.name for coll in bone.collections]
    except Exception:
        info["collections"] = []

    return info


def collect_vertex_group_weight_stats(
    *,
    mesh_obj: bpy.types.Object,
    group_name: str,
    eps: float,
    top_n: int,
) -> dict[str, Any]:
    vertex_count = len(mesh_obj.data.vertices)
    vg = mesh_obj.vertex_groups.get(group_name)

    result: dict[str, Any] = {
        "mesh_name": mesh_obj.name,
        "mesh_data": mesh_obj.data.name,
        "vertex_count": vertex_count,
        "group_name": group_name,
        "group_exists": vg is not None,
        "group_index": None,
        "eps": float(eps),
        "nonzero_vertices": 0,
        "vertex_ratio": 0.0,
        "total_weight": 0.0,
        "max_weight": 0.0,
        "min_positive_weight": None,
        "avg_weight_over_nonzero": 0.0,
        "top_vertices": [],
        "threshold_counts": {},
    }

    thresholds = [
        0.0,
        1e-12,
        1e-10,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
    ]
    threshold_counts = {str(t): 0 for t in thresholds}

    if vg is None:
        result["threshold_counts"] = threshold_counts
        return result

    group_index = int(vg.index)
    result["group_index"] = group_index

    weighted_items: list[dict[str, Any]] = []

    # Map group index -> group name for top-vertex neighbor assignments.
    group_name_by_index = {group.index: group.name for group in mesh_obj.vertex_groups}

    for vertex in mesh_obj.data.vertices:
        weight_for_group = 0.0

        for assignment in vertex.groups:
            if int(assignment.group) == group_index:
                weight_for_group = float(assignment.weight)
                break

        for threshold in thresholds:
            if weight_for_group > threshold:
                threshold_counts[str(threshold)] += 1

        if weight_for_group > eps:
            result["nonzero_vertices"] += 1
            result["total_weight"] += weight_for_group
            result["max_weight"] = max(float(result["max_weight"]), weight_for_group)

            if result["min_positive_weight"] is None:
                result["min_positive_weight"] = weight_for_group
            else:
                result["min_positive_weight"] = min(float(result["min_positive_weight"]), weight_for_group)

            all_assignments = []
            weight_sum_all_groups = 0.0

            for assignment in vertex.groups:
                gname = group_name_by_index.get(int(assignment.group), f"<group_{assignment.group}>")
                w = float(assignment.weight)
                weight_sum_all_groups += w
                all_assignments.append((gname, w))

            all_assignments.sort(key=lambda item: item[1], reverse=True)

            weighted_items.append(
                {
                    "vertex_index": int(vertex.index),
                    "weight": weight_for_group,
                    "co": vec3_tuple(vertex.co),
                    "all_group_weight_sum": weight_sum_all_groups,
                    "weight_fraction_of_all_groups": (
                        weight_for_group / weight_sum_all_groups
                        if weight_sum_all_groups > 0.0
                        else None
                    ),
                    "top_assignments": [
                        {"group": gname, "weight": w}
                        for gname, w in all_assignments[:10]
                    ],
                }
            )

    result["vertex_ratio"] = float(result["nonzero_vertices"]) / float(max(vertex_count, 1))
    result["avg_weight_over_nonzero"] = (
        float(result["total_weight"]) / float(max(result["nonzero_vertices"], 1))
    )
    result["threshold_counts"] = threshold_counts

    weighted_items.sort(key=lambda item: item["weight"], reverse=True)
    result["top_vertices"] = weighted_items[:top_n]

    return result


def classify_bone_under_prep04a_logic(
    *,
    armature: bpy.types.Object,
    mesh_obj: bpy.types.Object,
    bone_name: str,
    eps: float,
) -> dict[str, Any]:
    bone = armature.data.bones.get(bone_name)

    if bone is None:
        return {
            "bone_name": bone_name,
            "classification": "BONE_NOT_FOUND",
            "reason": "Bone does not exist in this armature.",
        }

    deform_bones = {
        b.name
        for b in armature.data.bones
        if bool(getattr(b, "use_deform", True))
    }
    non_deform_bones = {
        b.name
        for b in armature.data.bones
        if not bool(getattr(b, "use_deform", True))
    }

    vg_by_index = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    effective_vgs: set[str] = set()

    for vertex in mesh_obj.data.vertices:
        for assignment in vertex.groups:
            if float(assignment.weight) <= eps:
                continue
            gname = vg_by_index.get(int(assignment.group))
            if gname is not None:
                effective_vgs.add(gname)

    is_deform = bone_name in deform_bones
    is_non_deform = bone_name in non_deform_bones
    has_same_name_group = mesh_obj.vertex_groups.get(bone_name) is not None
    is_effective_group = bone_name in effective_vgs

    if is_non_deform:
        classification = "DELETE_NON_DEFORM"
        reason = "bone.use_deform is False, so Prep 04a deletes it regardless of vertex-group weights."
    elif is_deform and is_effective_group:
        classification = "KEEP_DEFORM_EFFECTIVE"
        reason = f"bone.use_deform is True and same-name vertex group has at least one weight > eps ({eps})."
    elif is_deform and not has_same_name_group:
        classification = "DELETE_DEFORM_MISSING_GROUP"
        reason = "bone.use_deform is True but mesh has no same-name vertex group."
    elif is_deform and has_same_name_group and not is_effective_group:
        classification = "DELETE_DEFORM_ZERO_OR_TINY_WEIGHT"
        reason = f"bone.use_deform is True and group exists, but no weight > eps ({eps})."
    else:
        classification = "UNKNOWN"
        reason = "Unexpected classification state."

    return {
        "bone_name": bone_name,
        "classification": classification,
        "reason": reason,
        "is_deform": is_deform,
        "is_non_deform": is_non_deform,
        "has_same_name_group": has_same_name_group,
        "is_effective_group": is_effective_group,
        "eps": float(eps),
        "total_bones": len(armature.data.bones),
        "deform_bone_count": len(deform_bones),
        "non_deform_bone_count": len(non_deform_bones),
        "effective_vertex_group_count": len(effective_vgs),
    }


def get_armature_modifier_info(mesh_obj: bpy.types.Object) -> list[dict[str, Any]]:
    infos = []
    for mod in mesh_obj.modifiers:
        if mod.type != "ARMATURE":
            continue
        infos.append(
            {
                "name": mod.name,
                "object": getattr(getattr(mod, "object", None), "name", None),
                "show_viewport": bool(mod.show_viewport),
                "show_render": bool(mod.show_render),
                "use_vertex_groups": bool(getattr(mod, "use_vertex_groups", False)),
                "use_bone_envelopes": bool(getattr(mod, "use_bone_envelopes", False)),
                "vertex_group": str(getattr(mod, "vertex_group", "")),
            }
        )
    return infos


def print_section(title: str) -> None:
    print("\n" + "=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def print_json_like(title: str, obj: Any) -> None:
    print(f"\n{title}", flush=True)
    print(json.dumps(obj, indent=2, ensure_ascii=False), flush=True)


def inspect_one_file(
    *,
    blend_path: Path,
    bone_name: str,
    armature_name: str | None,
    mesh_name: str | None,
    eps: float,
    zero_length_eps: float,
    top_n: int,
) -> dict[str, Any]:
    print_section(f"📂 Loading blend: {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    ensure_object_mode()

    print(f"📦 bpy version: {bpy.app.version_string}", flush=True)

    armature = find_armature(armature_name)
    mesh_obj = find_mesh_for_bone_group(
        requested_mesh_name=mesh_name,
        bone_name=bone_name,
    )

    print(f"🎯 Armature: {armature.name} / data={armature.data.name}", flush=True)
    print(
        f"🎯 Mesh:     {mesh_obj.name} / data={mesh_obj.data.name} "
        f"(verts={len(mesh_obj.data.vertices)}, groups={len(mesh_obj.vertex_groups)})",
        flush=True,
    )

    bone_info = get_bone_info(
        armature=armature,
        bone_name=bone_name,
        zero_length_eps=zero_length_eps,
    )
    weight_stats = collect_vertex_group_weight_stats(
        mesh_obj=mesh_obj,
        group_name=bone_name,
        eps=eps,
        top_n=top_n,
    )
    classification = classify_bone_under_prep04a_logic(
        armature=armature,
        mesh_obj=mesh_obj,
        bone_name=bone_name,
        eps=eps,
    )
    arm_mods = get_armature_modifier_info(mesh_obj)

    print_json_like("🦴 Bone info:", bone_info)
    print_json_like("⚖️ Vertex-group weight stats:", weight_stats)
    print_json_like("🧪 Prep 04a classification:", classification)
    print_json_like("🔗 Mesh Armature modifiers:", arm_mods)

    print("\n📌 Human-readable conclusion:", flush=True)

    if not bone_info.get("exists", False):
        print(f"   ❌ Bone '{bone_name}' does not exist in armature '{armature.name}'.", flush=True)
    else:
        if bone_info["is_zero_or_tiny_length"]:
            print(
                f"   ⚠️ Bone length is tiny/zero: length={bone_info['length']:.12g} "
                f"<= zero_length_eps={zero_length_eps:.12g}.",
                flush=True,
            )
        else:
            print(f"   ✅ Bone has non-zero length: {bone_info['length']:.12g}.", flush=True)

        print(f"   use_deform = {bone_info['use_deform']}", flush=True)
        print(f"   parent     = {bone_info['parent']}", flush=True)
        print(f"   children   = {bone_info['children']}", flush=True)

    cls = classification["classification"]
    reason = classification["reason"]

    if cls == "KEEP_DEFORM_EFFECTIVE":
        print(f"   ✅ It survives Prep 04a because: {reason}", flush=True)
        print(
            f"      nonzero_vertices={weight_stats['nonzero_vertices']} / "
            f"{weight_stats['vertex_count']} "
            f"({weight_stats['vertex_ratio']:.8f})",
            flush=True,
        )
        print(
            f"      max_weight={weight_stats['max_weight']:.12g}, "
            f"total_weight={weight_stats['total_weight']:.12g}, "
            f"avg_weight={weight_stats['avg_weight_over_nonzero']:.12g}",
            flush=True,
        )
    elif cls.startswith("DELETE"):
        print(f"   ❌ It should be deleted by Prep 04a because: {reason}", flush=True)
        print(
            "      If it still appears in Armature_Cleaned, then inspect whether you are "
            "looking at the latest output file, or whether the script used a different "
            "armature/mesh/eps setting.",
            flush=True,
        )
    else:
        print(f"   ⚠️ Classification is {cls}: {reason}", flush=True)

    return {
        "blend_path": str(blend_path),
        "bpy_version": bpy.app.version_string,
        "armature": armature.name,
        "mesh": mesh_obj.name,
        "bone_info": bone_info,
        "weight_stats": weight_stats,
        "classification": classification,
        "armature_modifiers": arm_mods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect why a specific bone survives Prep 04a bone pruning."
    )

    parser.add_argument(
        "--input_blend",
        type=str,
        nargs="+",
        required=True,
        help="One or more .blend files to inspect.",
    )
    parser.add_argument(
        "--bone",
        type=str,
        default="CC_Base_Pivot_0",
        help="Bone name to inspect. Default: CC_Base_Pivot_0.",
    )
    parser.add_argument(
        "--armature",
        type=str,
        default=None,
        help="Optional armature object name. If omitted, first armature is used.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help=(
            "Optional mesh object name. If omitted, prefer Original_Source_Mesh, "
            "then Griffin_A, then a mesh containing the bone vertex group."
        ),
    )
    parser.add_argument(
        "--weight_eps",
        type=float,
        default=1e-8,
        help="Same effective-weight threshold as Prep 04a. Default: 1e-8.",
    )
    parser.add_argument(
        "--zero_length_eps",
        type=float,
        default=1e-7,
        help="Bone length <= this value is reported as zero/tiny. Default: 1e-7.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Print top-N vertices by this bone's weight. Default: 20.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to write full diagnostic JSON.",
    )

    args = parser.parse_args(script_argv())

    all_results = []

    for item in args.input_blend:
        blend_path = Path(item)
        if not blend_path.exists():
            raise FileNotFoundError(f"Input blend does not exist: {blend_path}")

        result = inspect_one_file(
            blend_path=blend_path,
            bone_name=args.bone,
            armature_name=args.armature,
            mesh_name=args.mesh,
            eps=float(args.weight_eps),
            zero_length_eps=float(args.zero_length_eps),
            top_n=int(args.top_n),
        )
        all_results.append(result)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Diagnostic JSON saved to: {out_path}", flush=True)

    print("\n✅ Debug inspection complete.", flush=True)


if __name__ == "__main__":
    main()
