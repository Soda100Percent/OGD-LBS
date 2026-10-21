# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Reusable Blender SDF Grid utilities.

This module is intentionally generic:
    - no OGD-LBS data-prep path conventions
    - no manifest assumptions
    - no pipeline stage naming

It provides reusable helpers for Blender 5.x SDF Grid fusion:
    Mesh -> SDF Grid -> SDF Grid Offset -> Grid to Mesh
"""

import math
from pathlib import Path
from typing import Any

import bpy

from ogd_lbs.blender.object import ensure_object_mode
from ogd_lbs.blender.context import update_view_layer


try:
    import numpy as np
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    np = None
    trimesh = None
    HAS_TRIMESH = False


def create_sdf_grid_geonodes(
    obj: bpy.types.Object,
    voxel_size: float,
    offset_distance: float,
    *,
    band_width_padding: int,
) -> bpy.types.Modifier:
    """
    Create a Blender 5.x Geometry Nodes modifier:

        Mesh -> SDF Grid -> SDF Grid Offset -> Grid to Mesh
    """
    modifier = obj.modifiers.new(name="OGD_SDF_Offset", type="NODES")
    tree = bpy.data.node_groups.new(name=f"SDF_Remesh_Tree_{obj.name}", type="GeometryNodeTree")
    modifier.node_group = tree

    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    node_in = nodes.new(type="NodeGroupInput")
    node_in.location = (-500, 0)

    node_mesh_to_sdf = nodes.new(type="GeometryNodeMeshToSDFGrid")
    node_mesh_to_sdf.name = "Node_MeshToSDF"
    node_mesh_to_sdf.location = (-250, 0)
    node_mesh_to_sdf.inputs["Voxel Size"].default_value = float(voxel_size)

    band_width_voxels = int(math.ceil(abs(float(offset_distance)) / float(voxel_size))) + int(
        band_width_padding
    )
    node_mesh_to_sdf.inputs["Band Width"].default_value = int(band_width_voxels)

    node_sdf_offset = nodes.new(type="GeometryNodeSDFGridOffset")
    node_sdf_offset.name = "Node_SDFOffset"
    node_sdf_offset.location = (0, 0)
    node_sdf_offset.inputs["Distance"].default_value = float(offset_distance)

    node_grid_to_mesh = nodes.new(type="GeometryNodeGridToMesh")
    node_grid_to_mesh.name = "Node_GridToMesh"
    node_grid_to_mesh.location = (250, 0)

    node_out = nodes.new(type="NodeGroupOutput")
    node_out.location = (500, 0)

    links.new(node_in.outputs["Geometry"], node_mesh_to_sdf.inputs["Mesh"])
    links.new(node_mesh_to_sdf.outputs["SDF Grid"], node_sdf_offset.inputs["Grid"])
    links.new(node_sdf_offset.outputs["Grid"], node_grid_to_mesh.inputs["Grid"])
    links.new(node_grid_to_mesh.outputs["Mesh"], node_out.inputs["Geometry"])

    return modifier


def update_sdf_nodes(
    modifier: bpy.types.Modifier,
    *,
    voxel_size: float,
    offset_distance: float,
    band_width_padding: int,
) -> None:
    tree = modifier.node_group
    if tree is None:
        raise RuntimeError(f"Modifier has no node_group: {modifier.name}")

    node_mesh_to_sdf = tree.nodes["Node_MeshToSDF"]
    node_sdf_offset = tree.nodes["Node_SDFOffset"]

    node_mesh_to_sdf.inputs["Voxel Size"].default_value = float(voxel_size)

    band_width_voxels = int(math.ceil(abs(float(offset_distance)) / float(voxel_size))) + int(
        band_width_padding
    )
    node_mesh_to_sdf.inputs["Band Width"].default_value = int(band_width_voxels)

    node_sdf_offset.inputs["Distance"].default_value = float(offset_distance)


def duplicate_join_and_staticize_sources(
    source_objects: list[bpy.types.Object],
    *,
    temp_name: str,
    apply_scale: bool,
) -> bpy.types.Object:
    """
    Duplicate source mesh objects, join if needed, convert evaluated result to a
    static Mesh object, optionally apply scale, and clear vertex groups.

    This bakes the current evaluated deformation into geometry.
    """
    if not source_objects:
        raise RuntimeError("duplicate_join_and_staticize_sources requires at least one source object.")

    for obj in source_objects:
        if obj.type != "MESH":
            raise RuntimeError(f"Expected MESH source object, got {obj.type}: {obj.name}")

    ensure_object_mode()

    bpy.ops.object.select_all(action="DESELECT")

    for obj in source_objects:
        try:
            obj.hide_set(False)
        except Exception:
            pass
        obj.hide_viewport = False
        obj.select_set(True)

    bpy.context.view_layer.objects.active = source_objects[0]
    bpy.ops.object.duplicate()

    duplicated = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not duplicated:
        raise RuntimeError("No duplicated mesh object found.")

    bpy.context.view_layer.objects.active = duplicated[0]

    if len(duplicated) > 1:
        bpy.ops.object.join()

    temp_obj = bpy.context.view_layer.objects.active
    temp_obj.name = temp_name
    temp_obj.data.name = temp_name

    bpy.ops.object.convert(target="MESH")

    temp_obj = bpy.context.object
    temp_obj.name = temp_name
    temp_obj.data.name = temp_name

    if apply_scale:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    try:
        temp_obj.vertex_groups.clear()
    except Exception:
        pass

    update_view_layer()
    return temp_obj


def make_empty_topology_stats(reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "reason": reason,
        "components_filtered": 0,
        "main_genus": None,
        "main_is_watertight": False,
        "is_triangle_soup_like": False,
    }


def evaluate_topology_from_evaluated_mesh(
    obj: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph,
    *,
    min_component_faces: int,
    min_component_face_ratio: float,
) -> dict[str, Any]:
    """
    Evaluate topology directly from Blender evaluated mesh memory.

    This avoids OBJ round-trip triangle-soup artifacts.
    """
    if not HAS_TRIMESH:
        return make_empty_topology_stats("trimesh_not_available")

    eval_obj = obj.evaluated_get(depsgraph)

    try:
        eval_mesh = eval_obj.to_mesh()
        eval_mesh.calc_loop_triangles()

        verts = np.empty(len(eval_mesh.vertices) * 3, dtype=np.float64)
        eval_mesh.vertices.foreach_get("co", verts)

        faces = np.empty(len(eval_mesh.loop_triangles) * 3, dtype=np.int64)
        eval_mesh.loop_triangles.foreach_get("vertices", faces)

        mesh = trimesh.Trimesh(
            vertices=verts.reshape((-1, 3)),
            faces=faces.reshape((-1, 3)),
            process=False,
            validate=False,
        )

        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass

        components = mesh.split(only_watertight=False)

        face_threshold = max(
            int(min_component_faces),
            int(math.ceil(len(mesh.faces) * float(min_component_face_ratio))),
        )

        if face_threshold > 0:
            components_filtered_list = [
                c for c in components if len(c.faces) >= face_threshold
            ]
        else:
            components_filtered_list = list(components)

        if not components_filtered_list and components:
            components_filtered_list = [max(components, key=lambda c: len(c.faces))]

        main = max(components_filtered_list, key=lambda c: len(c.faces)) if components_filtered_list else None

        main_genus = int(round(1.0 - (float(main.euler_number) / 2.0))) if main else None
        main_is_watertight = bool(main.is_watertight) if main else False
        is_triangle_soup_like = (len(components) / float(max(len(mesh.faces), 1))) > 0.25

        return {
            "ok": True,
            "reason": "ok",
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "components_raw": len(components),
            "components_filtered": len(components_filtered_list),
            "main_genus": main_genus,
            "main_is_watertight": main_is_watertight,
            "is_triangle_soup_like": is_triangle_soup_like,
        }

    except Exception as exc:
        return make_empty_topology_stats(f"failed: {exc}")

    finally:
        try:
            eval_obj.to_mesh_clear()
        except Exception:
            pass


def topology_is_success(stats: dict[str, Any], *, require_watertight: bool) -> bool:
    if not stats.get("ok", False):
        return False

    if stats.get("is_triangle_soup_like", False):
        return False

    if stats.get("components_filtered", 999) != 1:
        return False

    if stats.get("main_genus", None) != 0:
        return False

    if require_watertight and not stats.get("main_is_watertight", False):
        return False

    return True


def format_topology_line(stats: dict[str, Any]) -> str:
    if not stats.get("ok", False):
        return f"FAILED: {stats.get('reason')}"

    return (
        f"V={stats['vertices']}, F={stats['faces']}, "
        f"C_raw={stats['components_raw']}, "
        f"C_filtered={stats['components_filtered']}, "
        f"g_main={stats['main_genus']}, "
        f"watertight={stats['main_is_watertight']}"
    )


def export_selected_obj(obj: bpy.types.Object, out_path: str | Path) -> None:
    """
    Export one selected object to OBJ using Blender's wm.obj_export.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    result = bpy.ops.wm.obj_export(
        filepath=str(out_path),
        export_selected_objects=True,
    )

    if result != {"FINISHED"}:
        raise RuntimeError(f"OBJ export failed for {out_path}: {result}")
