# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Reusable Blender helpers for Surface Deform binding and LBS weight transfer.

This module contains no data-prep path conventions. It operates on Blender
objects that are already resolved by the caller.
"""

from collections import defaultdict
from typing import Any

import bpy

from ogd_lbs.blender.context import update_view_layer
from ogd_lbs.blender.object import (
    ensure_object_mode,
    deselect_all_objects,
    set_active_object,
    unhide_object_for_ops,
    restore_source_visibility,
    remove_modifier_if_exists,
)


def clear_and_create_target_vertex_groups(
    *,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> int:
    if source_mesh.type != "MESH" or target_mesh.type != "MESH":
        raise RuntimeError("clear_and_create_target_vertex_groups expects MESH objects.")

    print("   🧩 Rebuilding target vertex groups from source...", flush=True)

    target_mesh.vertex_groups.clear()

    count = 0
    for src_group in source_mesh.vertex_groups:
        target_mesh.vertex_groups.new(name=src_group.name)
        count += 1

    print(f"      created groups: {count}", flush=True)
    return count


def transfer_lbs_weights_operator(
    *,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> None:
    if source_mesh.type != "MESH" or target_mesh.type != "MESH":
        raise RuntimeError("transfer_lbs_weights_operator expects MESH objects.")

    print("   ⚖️ Transferring LBS weights: source ACTIVE -> target selected", flush=True)

    ensure_object_mode()
    deselect_all_objects()

    source_mesh.select_set(True)
    target_mesh.select_set(True)
    set_active_object(source_mesh)

    update_view_layer()

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
        print(f"      ⚠️ data_transfer TypeError: {exc}", flush=True)
        print("      -> retrying reduced parameter set", flush=True)
        result = bpy.ops.object.data_transfer(
            use_create=True,
            data_type="VGROUP_WEIGHTS",
            vert_mapping="POLYINTERP_NEAREST",
            layers_select_src="ALL",
            mix_mode="REPLACE",
            mix_factor=1.0,
        )

    print(f"      data_transfer result: {result}", flush=True)

    if result != {"FINISHED"}:
        raise RuntimeError(f"bpy.ops.object.data_transfer failed: {result}")

    update_view_layer()


def normalize_target_vertex_groups(target_mesh: bpy.types.Object) -> None:
    if target_mesh.type != "MESH":
        raise RuntimeError("normalize_target_vertex_groups expects a MESH object.")

    print("   🧮 Normalizing target vertex groups...", flush=True)

    ensure_object_mode()
    deselect_all_objects()

    target_mesh.select_set(True)
    set_active_object(target_mesh)

    try:
        result = bpy.ops.object.vertex_group_normalize_all(
            group_select_mode="ALL",
            lock_active=False,
        )
        print(f"      normalize result: {result}", flush=True)
    except Exception as exc:
        print(f"      ⚠️ vertex_group_normalize_all failed: {exc}", flush=True)

    update_view_layer()


def count_vertices_with_any_weight(
    mesh_obj: bpy.types.Object,
    eps: float,
) -> tuple[int, int]:
    if mesh_obj.type != "MESH":
        raise RuntimeError("count_vertices_with_any_weight expects a MESH object.")

    total = len(mesh_obj.data.vertices)
    nonzero = 0

    for vertex in mesh_obj.data.vertices:
        weight_sum = 0.0
        for assignment in vertex.groups:
            weight_sum += float(assignment.weight)

        if weight_sum > eps:
            nonzero += 1

    return nonzero, total


def _mesh_update_edges(mesh: bpy.types.Mesh) -> None:
    try:
        mesh.update(calc_edges=True)
    except TypeError:
        try:
            mesh.update()
        except Exception:
            pass
    except Exception:
        pass

    try:
        mesh.calc_loop_triangles()
    except Exception:
        pass


def _edge_key(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _collect_edges_with_more_than_two_faces(
    mesh: bpy.types.Mesh,
) -> list[tuple[tuple[int, int], list[int]]]:
    _mesh_update_edges(mesh)

    edge_to_polys: dict[tuple[int, int], list[int]] = defaultdict(list)

    for poly in mesh.polygons:
        verts = [int(v) for v in poly.vertices]
        if len(verts) < 2:
            continue

        for idx, a in enumerate(verts):
            b = verts[(idx + 1) % len(verts)]
            if int(a) == int(b):
                continue
            edge_to_polys[_edge_key(int(a), int(b))].append(int(poly.index))

    return [
        (key, polys)
        for key, polys in edge_to_polys.items()
        if len(polys) > 2
    ]


def _surface_deform_target_topology_stats_from_mesh(
    mesh: bpy.types.Mesh,
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    bad_edges = _collect_edges_with_more_than_two_faces(mesh)

    samples: list[dict[str, Any]] = []
    for (a, b), polys in bad_edges[: int(sample_limit)]:
        item: dict[str, Any] = {
            "edge_vertices": [int(a), int(b)],
            "face_count": int(len(polys)),
            "polygon_indices": [int(x) for x in polys[:50]],
        }

        try:
            va = mesh.vertices[int(a)].co
            vb = mesh.vertices[int(b)].co
            item["edge_length"] = float((va - vb).length)
            item["v0"] = [float(va[0]), float(va[1]), float(va[2])]
            item["v1"] = [float(vb[0]), float(vb[1]), float(vb[2])]
        except Exception:
            pass

        samples.append(item)

    return {
        "vertex_count": int(len(mesh.vertices)),
        "edge_count": int(len(mesh.edges)),
        "face_count": int(len(mesh.polygons)),
        "edges_with_more_than_2_faces_count": int(len(bad_edges)),
        "edges_with_more_than_2_faces_samples": samples,
    }


def surface_deform_target_topology_stats(
    obj: bpy.types.Object,
    *,
    evaluated: bool = False,
    add_temp_triangulate: bool = False,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """
    Audit the topology condition that blocks Blender Surface Deform binding:
    target edges shared by more than two polygons.

    Args:
        obj: Mesh object used as Surface Deform target.
        evaluated: Audit evaluated mesh instead of raw mesh data.
        add_temp_triangulate: Temporarily add a Triangulate modifier during
            evaluated audit. This approximates Prep 03's bind target state.
        sample_limit: Number of offending-edge samples to keep.

    Returns:
        JSON-serializable topology stats.
    """
    if obj.type != "MESH":
        raise RuntimeError(f"surface_deform_target_topology_stats expects MESH: {obj.name}")

    ensure_object_mode()
    temp_mod = None

    try:
        if add_temp_triangulate:
            remove_modifier_if_exists(obj, "__OGD_SurfaceBindProxy_Audit_Triangulate")
            temp_mod = obj.modifiers.new(
                name="__OGD_SurfaceBindProxy_Audit_Triangulate",
                type="TRIANGULATE",
            )
            try:
                temp_mod.keep_custom_normals = True
            except Exception:
                pass
            try:
                temp_mod.quad_method = "BEAUTY"
            except Exception:
                pass
            try:
                temp_mod.ngon_method = "BEAUTY"
            except Exception:
                pass
            update_view_layer()

        if not evaluated:
            stats = _surface_deform_target_topology_stats_from_mesh(
                obj.data,
                sample_limit=sample_limit,
            )
        else:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            try:
                stats = _surface_deform_target_topology_stats_from_mesh(
                    eval_mesh,
                    sample_limit=sample_limit,
                )
            finally:
                try:
                    eval_obj.to_mesh_clear()
                except Exception:
                    pass

        stats.update(
            {
                "object_name": obj.name,
                "data_name": obj.data.name,
                "evaluated": bool(evaluated),
                "temp_triangulate": bool(add_temp_triangulate),
            }
        )
        return stats

    finally:
        if temp_mod is not None:
            try:
                obj.modifiers.remove(temp_mod)
            except Exception:
                pass
            update_view_layer()


def _make_surface_bind_proxy_name(source_mesh: bpy.types.Object) -> str:
    return f"{source_mesh.name}__SurfaceBindProxy"


def _require_proxy_name_available(proxy_name: str) -> None:
    collisions = []

    if bpy.data.objects.get(proxy_name) is not None:
        collisions.append(f"object:{proxy_name}")

    if bpy.data.meshes.get(proxy_name) is not None:
        collisions.append(f"mesh:{proxy_name}")

    if collisions:
        raise RuntimeError(
            "Surface Deform bind proxy name collision. "
            f"Collisions={collisions}. Remove stale proxy or choose a clean input blend."
        )


def _link_proxy_to_source_collection(
    *,
    source_mesh: bpy.types.Object,
    proxy: bpy.types.Object,
) -> None:
    collections = list(getattr(source_mesh, "users_collection", []))

    if collections:
        collections[0].objects.link(proxy)
        return

    bpy.context.scene.collection.objects.link(proxy)


def _triangulate_mesh_data_in_place(obj: bpy.types.Object) -> None:
    if obj.type != "MESH":
        raise RuntimeError(f"_triangulate_mesh_data_in_place expects MESH: {obj.name}")

    ensure_object_mode()
    deselect_all_objects()

    try:
        obj.hide_set(False)
    except Exception:
        pass
    obj.hide_viewport = False

    obj.select_set(True)
    set_active_object(obj)
    update_view_layer()

    result = bpy.ops.object.mode_set(mode="EDIT")
    if result != {"FINISHED"}:
        raise RuntimeError(f"Failed to enter EDIT mode for triangulation: {obj.name}, result={result}")

    try:
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_all(action="SELECT")

        try:
            tri_result = bpy.ops.mesh.quads_convert_to_tris(
                quad_method="BEAUTY",
                ngon_method="BEAUTY",
            )
        except TypeError:
            tri_result = bpy.ops.mesh.quads_convert_to_tris()

        if tri_result != {"FINISHED"}:
            raise RuntimeError(
                f"quads_convert_to_tris failed for proxy={obj.name}, result={tri_result}"
            )

    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    _mesh_update_edges(obj.data)
    update_view_layer()


def _bad_edge_indices(mesh: bpy.types.Mesh) -> list[int]:
    bad = _collect_edges_with_more_than_two_faces(mesh)
    bad_keys = {key for key, _polys in bad}

    _mesh_update_edges(mesh)

    edge_by_key: dict[tuple[int, int], int] = {}
    for edge in mesh.edges:
        a = int(edge.vertices[0])
        b = int(edge.vertices[1])
        edge_by_key[_edge_key(a, b)] = int(edge.index)

    indices = []
    missing_keys = []

    for key in bad_keys:
        idx = edge_by_key.get(key)
        if idx is None:
            missing_keys.append(key)
            continue
        indices.append(int(idx))

    if missing_keys:
        raise RuntimeError(
            "Could not map offending polygon-edge keys to mesh.edge indices. "
            f"Missing={missing_keys[:20]}"
        )

    return sorted(set(indices))


def _split_selected_edges_in_place(
    obj: bpy.types.Object,
    edge_indices: list[int],
) -> None:
    if obj.type != "MESH":
        raise RuntimeError(f"_split_selected_edges_in_place expects MESH: {obj.name}")

    if not edge_indices:
        return

    ensure_object_mode()
    deselect_all_objects()

    try:
        obj.hide_set(False)
    except Exception:
        pass
    obj.hide_viewport = False

    obj.select_set(True)
    set_active_object(obj)
    update_view_layer()

    mesh = obj.data
    _mesh_update_edges(mesh)

    for vertex in mesh.vertices:
        vertex.select = False
    for edge in mesh.edges:
        edge.select = False
    for poly in mesh.polygons:
        poly.select = False

    valid = set(int(idx) for idx in edge_indices)
    for edge in mesh.edges:
        if int(edge.index) in valid:
            edge.select = True

    result = bpy.ops.object.mode_set(mode="EDIT")
    if result != {"FINISHED"}:
        raise RuntimeError(f"Failed to enter EDIT mode for edge split: {obj.name}, result={result}")

    try:
        bpy.ops.mesh.select_mode(type="EDGE")

        try:
            split_result = bpy.ops.mesh.edge_split(type="EDGE")
        except TypeError:
            split_result = bpy.ops.mesh.edge_split()

        if split_result != {"FINISHED"}:
            raise RuntimeError(
                f"mesh.edge_split failed for proxy={obj.name}, result={split_result}"
            )

    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    _mesh_update_edges(mesh)
    update_view_layer()


def repair_surface_bind_proxy_topology(
    proxy: bpy.types.Object,
    *,
    max_iters: int = 8,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """
    Repair a proxy mesh for Surface Deform binding.

    Strategy:
        1. Triangulate mesh data in place.
        2. Repeatedly split edges shared by more than two polygons.

    This intentionally operates only on a copied proxy mesh. It does not mutate
    the original Source_<VDB> lineage object.
    """
    if proxy.type != "MESH":
        raise RuntimeError(f"repair_surface_bind_proxy_topology expects MESH: {proxy.name}")

    print(f"   🧰 Repairing Surface Deform bind proxy topology: {proxy.name}", flush=True)
    print("      triangulating proxy mesh data...", flush=True)

    _triangulate_mesh_data_in_place(proxy)

    iteration_summaries: list[dict[str, Any]] = []

    for iteration in range(int(max_iters)):
        stats = surface_deform_target_topology_stats(
            proxy,
            evaluated=False,
            add_temp_triangulate=False,
            sample_limit=sample_limit,
        )
        bad_count = int(stats["edges_with_more_than_2_faces_count"])

        print(
            f"      repair iteration {iteration}: edge>2={bad_count}, "
            f"V={stats['vertex_count']}, F={stats['face_count']}",
            flush=True,
        )

        iteration_summaries.append(
            {
                "iteration": int(iteration),
                "before_split": stats,
            }
        )

        if bad_count <= 0:
            return {
                "proxy_name": proxy.name,
                "status": "ok",
                "max_iters": int(max_iters),
                "iterations": iteration_summaries,
                "final_stats": stats,
            }

        edge_indices = _bad_edge_indices(proxy.data)
        print(f"         splitting offending edges: {edge_indices}", flush=True)

        _split_selected_edges_in_place(proxy, edge_indices)

    final_stats = surface_deform_target_topology_stats(
        proxy,
        evaluated=False,
        add_temp_triangulate=False,
        sample_limit=sample_limit,
    )
    final_bad = int(final_stats["edges_with_more_than_2_faces_count"])

    if final_bad > 0:
        raise RuntimeError(
            f"Surface bind proxy repair failed for {proxy.name}: "
            f"edge>2 remains {final_bad} after max_iters={max_iters}. "
            f"samples={final_stats.get('edges_with_more_than_2_faces_samples')}"
        )

    return {
        "proxy_name": proxy.name,
        "status": "ok",
        "max_iters": int(max_iters),
        "iterations": iteration_summaries,
        "final_stats": final_stats,
    }


def create_surface_deform_bind_proxy(
    *,
    source_mesh: bpy.types.Object,
    proxy_name: str | None = None,
    repair_max_iters: int = 8,
    sample_limit: int = 20,
) -> tuple[bpy.types.Object, dict[str, Any]]:
    """
    Create a hidden repaired Surface Deform bind target proxy.

    The proxy:
        - is a copied mesh object;
        - preserves object transform, parent, vertex groups, and modifiers as
          copied by Blender object.copy();
        - is triangulated and non-manifold edge>2 cases are split;
        - stays hidden from user/render but remains viewport-evaluable.

    It is intended to be used as Surface Deform modifier target instead of the
    original Source_<VDB> object when the original has edge>2 topology.
    """
    if source_mesh.type != "MESH":
        raise RuntimeError(f"create_surface_deform_bind_proxy expects MESH: {source_mesh.name}")

    if proxy_name is None:
        proxy_name = _make_surface_bind_proxy_name(source_mesh)

    proxy_name = str(proxy_name).strip()
    if not proxy_name:
        raise RuntimeError("Surface bind proxy name cannot be empty.")

    _require_proxy_name_available(proxy_name)

    print("   🧪 Creating Surface Deform bind proxy...", flush=True)
    print(f"      source: {source_mesh.name}", flush=True)
    print(f"      proxy : {proxy_name}", flush=True)

    ensure_object_mode()
    update_view_layer()

    proxy = source_mesh.copy()
    proxy.data = source_mesh.data.copy()
    proxy.name = proxy_name
    proxy.data.name = proxy_name

    _link_proxy_to_source_collection(source_mesh=source_mesh, proxy=proxy)

    proxy.matrix_world = source_mesh.matrix_world.copy()

    try:
        proxy.hide_set(False)
    except Exception:
        pass
    proxy.hide_viewport = False
    proxy.hide_render = True

    update_view_layer()

    repair_report = repair_surface_bind_proxy_topology(
        proxy,
        max_iters=int(repair_max_iters),
        sample_limit=int(sample_limit),
    )

    # Hidden from the user/render, but NOT disabled from viewport evaluation.
    try:
        proxy.hide_set(True)
    except Exception:
        pass
    proxy.hide_viewport = False
    proxy.hide_render = True

    update_view_layer()

    return proxy, repair_report


def resolve_surface_deform_bind_source(
    *,
    source_mesh: bpy.types.Object,
    proxy_mode: str,
    repair_max_iters: int = 8,
    sample_limit: int = 20,
) -> tuple[bpy.types.Object, dict[str, Any]]:
    """
    Resolve which mesh object should be used as Surface Deform bind target.

    proxy_mode:
        off
            Always use source_mesh directly.
        auto
            Audit source_mesh with temporary triangulation. Create repaired proxy
            only if edge>2 exists.
        always
            Always create repaired proxy.
    """
    mode = str(proxy_mode or "off").strip().lower()

    if mode not in {"off", "auto", "always"}:
        raise RuntimeError(
            f"Unsupported surface bind proxy mode: {proxy_mode!r}. "
            "Expected off, auto, or always."
        )

    report: dict[str, Any] = {
        "mode": mode,
        "source_name": source_mesh.name,
        "proxy_created": False,
        "bind_source_name": source_mesh.name,
    }

    if mode == "off":
        return source_mesh, report

    audit = surface_deform_target_topology_stats(
        source_mesh,
        evaluated=True,
        add_temp_triangulate=True,
        sample_limit=sample_limit,
    )
    report["source_evaluated_triangulated_audit"] = audit

    edge_gt2 = int(audit.get("edges_with_more_than_2_faces_count", 0))

    print(
        f"   🔎 Surface bind source audit: {source_mesh.name}, "
        f"mode=evaluated+temp_triangulate, edge>2={edge_gt2}",
        flush=True,
    )

    if mode == "auto" and edge_gt2 <= 0:
        print("      proxy not needed.", flush=True)
        return source_mesh, report

    proxy, repair_report = create_surface_deform_bind_proxy(
        source_mesh=source_mesh,
        repair_max_iters=int(repair_max_iters),
        sample_limit=int(sample_limit),
    )

    report.update(
        {
            "proxy_created": True,
            "proxy_name": proxy.name,
            "bind_source_name": proxy.name,
            "proxy_repair": repair_report,
        }
    )

    final_bad = int(
        repair_report.get("final_stats", {}).get("edges_with_more_than_2_faces_count", -1)
    )
    if final_bad != 0:
        raise RuntimeError(
            f"Surface bind proxy repair did not fully remove edge>2 topology: "
            f"proxy={proxy.name}, final_edge_gt2={final_bad}"
        )

    print(
        f"      proxy ready: {proxy.name}, final edge>2={final_bad}",
        flush=True,
    )

    return proxy, report


def ensure_source_triangulate(source_mesh: bpy.types.Object) -> bpy.types.Modifier:
    if source_mesh.type != "MESH":
        raise RuntimeError("ensure_source_triangulate expects a MESH object.")

    print("   📐 Refreshing source Triangulate modifier...", flush=True)

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
    if source_mesh.type != "MESH" or target_mesh.type != "MESH":
        raise RuntimeError("bind_surface_deform expects MESH objects.")

    print("   🔗 Binding Surface Deform...", flush=True)

    remove_modifier_if_exists(target_mesh, "OGD_SurfaceDeform")

    surf_mod = target_mesh.modifiers.new(name="OGD_SurfaceDeform", type="SURFACE_DEFORM")
    surf_mod.target = source_mesh

    ensure_object_mode()
    deselect_all_objects()

    target_mesh.select_set(True)
    set_active_object(target_mesh)

    update_view_layer()

    current_falloff = float(falloff_start)

    while current_falloff <= float(max_falloff) + 1e-9:
        surf_mod.falloff = current_falloff
        print(f"      trying falloff={current_falloff:.3f}", flush=True)

        try:
            with bpy.context.temp_override(
                active_object=target_mesh,
                object=target_mesh,
                selected_objects=[target_mesh],
                selected_editable_objects=[target_mesh],
            ):
                result = bpy.ops.object.surfacedeform_bind(modifier=surf_mod.name)

            print(f"         bind result: {result}", flush=True)
        except Exception as exc:
            print(f"         ⚠️ bind exception: {exc}", flush=True)

        update_view_layer()

        if bool(getattr(surf_mod, "is_bound", False)):
            print(f"      ✅ Surface Deform bound with falloff={current_falloff:.3f}", flush=True)
            return surf_mod

        current_falloff += float(falloff_step)

    raise RuntimeError(
        f"Failed to bind Surface Deform for target={target_mesh.name}, "
        f"source={source_mesh.name}. Tried falloff up to {max_falloff}."
    )


def add_armature_modifier(
    *,
    armature: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> bpy.types.Modifier:
    if armature.type != "ARMATURE":
        raise RuntimeError("add_armature_modifier expects an ARMATURE object.")

    if target_mesh.type != "MESH":
        raise RuntimeError("add_armature_modifier expects a MESH target object.")

    print("   🦴 Adding Armature modifier for LBS drive mode...", flush=True)

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
    drive = str(drive_mode).upper()

    if drive == "SURFACE":
        surf_mod.show_viewport = True
        surf_mod.show_render = True
        arm_mod.show_viewport = False
        arm_mod.show_render = False
    elif drive == "LBS":
        surf_mod.show_viewport = False
        surf_mod.show_render = False
        arm_mod.show_viewport = True
        arm_mod.show_render = True
    else:
        raise ValueError(f"Unsupported single drive mode: {drive_mode}")


def apply_drive_mode_to_target(
    target_mesh: bpy.types.Object,
    *,
    drive_mode: str,
) -> None:
    surf_mod = target_mesh.modifiers.get("OGD_SurfaceDeform")
    arm_mod = target_mesh.modifiers.get("OGD_Armature")

    if surf_mod is None:
        raise RuntimeError(f"Missing OGD_SurfaceDeform on target mesh: {target_mesh.name}")

    if arm_mod is None:
        raise RuntimeError(f"Missing OGD_Armature on target mesh: {target_mesh.name}")

    apply_drive_mode(
        drive_mode=drive_mode,
        surf_mod=surf_mod,
        arm_mod=arm_mod,
    )


def print_modifier_summary(obj: bpy.types.Object) -> None:
    print(f"   🧾 Modifier summary for {obj.name}:", flush=True)

    for mod in obj.modifiers:
        print(
            f"      - {mod.name}: type={mod.type}, "
            f"show_viewport={mod.show_viewport}, "
            f"show_render={mod.show_render}, "
            f"target={getattr(getattr(mod, 'target', None), 'name', None)}, "
            f"object={getattr(getattr(mod, 'object', None), 'name', None)}, "
            f"is_bound={getattr(mod, 'is_bound', None)}",
            flush=True,
        )


def process_surface_deform_pair(
    *,
    armature: bpy.types.Object,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
    drive_mode: str,
    max_falloff: float,
    falloff_start: float,
    falloff_step: float,
    weight_eps: float,
    surface_bind_proxy_mode: str = "off",
    surface_bind_proxy_repair_max_iters: int = 8,
) -> None:
    """
    Process one Source_<VDB> -> <VDB> pair.

    drive_mode must be SURFACE or LBS. For BOTH workflows, bind once with
    SURFACE first, save, then let caller switch targets to LBS and save again.
    """
    drive = str(drive_mode).upper()

    if drive not in {"SURFACE", "LBS"}:
        raise ValueError(
            f"process_surface_deform_pair expects SURFACE or LBS, got {drive_mode!r}."
        )

    print("\n" + "=" * 80, flush=True)
    print(f"🔁 Processing VDB pair: {source_mesh.name} -> {target_mesh.name}", flush=True)

    source_state = unhide_object_for_ops(source_mesh)
    _ = unhide_object_for_ops(target_mesh)

    try:
        for mod_name in ["OGD_SurfaceDeform", "OGD_Armature"]:
            remove_modifier_if_exists(target_mesh, mod_name)

        clear_and_create_target_vertex_groups(
            source_mesh=source_mesh,
            target_mesh=target_mesh,
        )

        transfer_lbs_weights_operator(
            source_mesh=source_mesh,
            target_mesh=target_mesh,
        )

        normalize_target_vertex_groups(target_mesh)

        nonzero, total = count_vertices_with_any_weight(target_mesh, eps=float(weight_eps))
        ratio = nonzero / max(total, 1)

        print(
            f"   📊 Target weighted vertices after transfer: "
            f"{nonzero}/{total} ({ratio:.6f})",
            flush=True,
        )

        if nonzero <= 0:
            raise RuntimeError(
                f"Target mesh has zero weighted vertices after transfer: {target_mesh.name}"
            )

        bind_source_mesh, bind_source_report = resolve_surface_deform_bind_source(
            source_mesh=source_mesh,
            proxy_mode=str(surface_bind_proxy_mode),
            repair_max_iters=int(surface_bind_proxy_repair_max_iters),
        )

        if bind_source_mesh is not source_mesh:
            print(
                f"   🔧 Surface Deform bind target proxy enabled: "
                f"{source_mesh.name} -> {bind_source_mesh.name}",
                flush=True,
            )
        else:
            print(
                f"   🔧 Surface Deform bind target: {bind_source_mesh.name} "
                f"(proxy_mode={surface_bind_proxy_mode})",
                flush=True,
            )

        ensure_source_triangulate(bind_source_mesh)

        surf_mod = bind_surface_deform(
            source_mesh=bind_source_mesh,
            target_mesh=target_mesh,
            max_falloff=float(max_falloff),
            falloff_start=float(falloff_start),
            falloff_step=float(falloff_step),
        )

        arm_mod = add_armature_modifier(
            armature=armature,
            target_mesh=target_mesh,
        )

        apply_drive_mode(
            drive_mode=drive,
            surf_mod=surf_mod,
            arm_mod=arm_mod,
        )

        target_mesh.hide_set(False)
        target_mesh.hide_viewport = False
        target_mesh.hide_render = False

        print_modifier_summary(target_mesh)

    finally:
        restore_source_visibility(source_mesh, source_state)
        update_view_layer()
