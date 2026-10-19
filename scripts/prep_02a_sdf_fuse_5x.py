#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 02a: SDF Grid fusion in Blender 5.x.

Main responsibilities:
    1. Open a Prep 01 .blend file.
    2. Convert selected source meshes into static rest-pose mesh copies.
    3. Fuse them through Blender 5.x SDF Grid Geometry Nodes.
    4. Optionally auto-tune positive SDF offset using topology diagnostics.
    5. Export fused static mesh OBJ files and a manifest.json.

Auto-offset strategy:
    - voxel_size is manually supplied by CLI or config.
    - If --offset is explicitly supplied, use it and skip auto tuning.
    - If config contains offset and --auto_offset is not set, use config offset.
    - Otherwise, auto tune:
        offset starts at 0.5 * voxel_size
        step is 0.5 * voxel_size
        max offset is max_offset_multiplier * voxel_size (default 8.0)
      Stop when filtered component count is 1 and main component genus is 0.
      If no perfect topology is found, use the last attempted offset.

Important fix:
    - Topology analysis no longer exports temporary OBJ files.
    - It reads the evaluated mesh directly from Blender depsgraph and constructs
      a trimesh object from shared vertices and triangle indices.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy

try:
    import numpy as np
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    np = None
    trimesh = None
    HAS_TRIMESH = False


# -----------------------------------------------------------------------------
# Basic Blender helpers
# -----------------------------------------------------------------------------

def ensure_object_mode() -> None:
    try:
        active = bpy.context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def set_all_armatures_rest_pose() -> None:
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            try:
                obj.data.pose_position = "REST"
            except Exception:
                pass


def print_scene_summary() -> None:
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]

    print(f"📦 bpy version: {bpy.app.version_string}", flush=True)
    print(f"🦴 Armatures: {[(a.name, len(a.data.bones)) for a in armatures]}", flush=True)
    print(
        "📦 Meshes: "
        + str([(m.name, len(m.data.vertices), len(m.data.polygons), tuple(round(v, 6) for v in m.scale)) for m in meshes]),
        flush=True,
    )


# -----------------------------------------------------------------------------
# Geometry Nodes SDF Grid
# -----------------------------------------------------------------------------

def create_sdf_grid_geonodes(
    obj: bpy.types.Object,
    voxel_size: float,
    offset_distance: float,
    *,
    band_width_padding: int,
) -> bpy.types.Modifier:
    """
    Build:
        Group Input Geometry
          -> Mesh to SDF Grid
          -> SDF Grid Offset
          -> Grid to Mesh
          -> Group Output Geometry
    """

    modifier = obj.modifiers.new(name="OGD_SDF_Offset", type="NODES")
    tree = bpy.data.node_groups.new(
        name=f"SDF_Remesh_Tree_{obj.name}",
        type="GeometryNodeTree",
    )
    modifier.node_group = tree

    tree.interface.new_socket(
        name="Geometry",
        in_out="INPUT",
        socket_type="NodeSocketGeometry",
    )
    tree.interface.new_socket(
        name="Geometry",
        in_out="OUTPUT",
        socket_type="NodeSocketGeometry",
    )

    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    node_in = nodes.new(type="NodeGroupInput")
    node_in.location = (-500, 0)

    node_mesh_to_sdf = nodes.new(type="GeometryNodeMeshToSDFGrid")
    node_mesh_to_sdf.name = "Node_MeshToSDF"
    node_mesh_to_sdf.label = "Mesh to SDF Grid"
    node_mesh_to_sdf.location = (-250, 0)
    node_mesh_to_sdf.inputs["Voxel Size"].default_value = float(voxel_size)

    band_width_voxels = compute_band_width_voxels(
        voxel_size=voxel_size,
        offset_distance=offset_distance,
        band_width_padding=band_width_padding,
    )
    node_mesh_to_sdf.inputs["Band Width"].default_value = int(band_width_voxels)

    node_sdf_offset = nodes.new(type="GeometryNodeSDFGridOffset")
    node_sdf_offset.name = "Node_SDFOffset"
    node_sdf_offset.label = "SDF Grid Offset"
    node_sdf_offset.location = (0, 0)
    node_sdf_offset.inputs["Distance"].default_value = float(offset_distance)

    node_grid_to_mesh = nodes.new(type="GeometryNodeGridToMesh")
    node_grid_to_mesh.name = "Node_GridToMesh"
    node_grid_to_mesh.label = "Grid to Mesh"
    node_grid_to_mesh.location = (250, 0)

    node_out = nodes.new(type="NodeGroupOutput")
    node_out.location = (500, 0)

    links.new(node_in.outputs["Geometry"], node_mesh_to_sdf.inputs["Mesh"])
    links.new(node_mesh_to_sdf.outputs["SDF Grid"], node_sdf_offset.inputs["Grid"])
    links.new(node_sdf_offset.outputs["Grid"], node_grid_to_mesh.inputs["Grid"])
    links.new(node_grid_to_mesh.outputs["Mesh"], node_out.inputs["Geometry"])

    return modifier


def compute_band_width_voxels(
    *,
    voxel_size: float,
    offset_distance: float,
    band_width_padding: int,
) -> int:
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    return int(math.ceil(abs(float(offset_distance)) / float(voxel_size))) + int(band_width_padding)


def update_sdf_nodes(
    modifier: bpy.types.Modifier,
    *,
    voxel_size: float,
    offset_distance: float,
    band_width_padding: int,
) -> None:
    tree = modifier.node_group
    node_mesh_to_sdf = tree.nodes["Node_MeshToSDF"]
    node_sdf_offset = tree.nodes["Node_SDFOffset"]

    node_mesh_to_sdf.inputs["Voxel Size"].default_value = float(voxel_size)
    node_mesh_to_sdf.inputs["Band Width"].default_value = compute_band_width_voxels(
        voxel_size=voxel_size,
        offset_distance=offset_distance,
        band_width_padding=band_width_padding,
    )
    node_sdf_offset.inputs["Distance"].default_value = float(offset_distance)


# -----------------------------------------------------------------------------
# Source mesh preparation
# -----------------------------------------------------------------------------

def duplicate_join_and_staticize_sources(
    source_objects: list[bpy.types.Object],
    *,
    temp_name: str,
    apply_scale: bool,
) -> bpy.types.Object:
    """
    Duplicate selected source meshes, join them if needed, then convert to static
    mesh before adding the SDF Geometry Nodes modifier.

    This is important because commercial character meshes usually have Armature
    modifiers or other modifiers. We do not want OGD_SDF_Offset to be a non-first
    modifier in the stack.
    """

    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")

    for obj in source_objects:
        obj.hide_set(False)
        obj.select_set(True)

    bpy.context.view_layer.objects.active = source_objects[0]

    print("     -> Duplicating source mesh objects...", flush=True)
    bpy.ops.object.duplicate()

    duplicated = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not duplicated:
        raise RuntimeError("Duplication produced no mesh objects.")

    bpy.context.view_layer.objects.active = duplicated[0]

    if len(duplicated) > 1:
        print(f"     -> Joining {len(duplicated)} duplicated meshes...", flush=True)
        bpy.ops.object.join()

    temp_obj = bpy.context.view_layer.objects.active
    if temp_obj is None or temp_obj.type != "MESH":
        raise RuntimeError("Failed to create active temporary mesh after duplicate/join.")

    temp_obj.name = temp_name
    temp_obj.data.name = temp_name

    # Collapse existing modifier stack, including Armature, before adding SDF GN.
    # Armatures have already been set to REST pose.
    print("     -> Converting duplicate to static mesh before SDF...", flush=True)
    bpy.ops.object.convert(target="MESH")
    temp_obj = bpy.context.object

    if temp_obj is None or temp_obj.type != "MESH":
        raise RuntimeError("Object convert(target='MESH') failed.")

    temp_obj.name = temp_name
    temp_obj.data.name = temp_name

    if apply_scale:
        scale_tuple = tuple(float(v) for v in temp_obj.scale)
        if any(abs(v - 1.0) > 1e-6 for v in scale_tuple):
            print(f"     -> Applying object scale before SDF: {scale_tuple}", flush=True)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        else:
            print("     -> Object scale already identity.", flush=True)

    # These are no longer useful for a temporary static SDF source.
    try:
        temp_obj.vertex_groups.clear()
    except Exception:
        pass

    return temp_obj


# -----------------------------------------------------------------------------
# Topology diagnostics
# -----------------------------------------------------------------------------

def make_empty_topology_stats(reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "reason": reason,
        "vertices": 0,
        "faces": 0,
        "edges": 0,
        "components_raw": 0,
        "components_filtered": 0,
        "euler_number": None,
        "global_genus": None,
        "main_faces": 0,
        "main_face_ratio": 0.0,
        "main_euler_number": None,
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
    Evaluate topology from Blender memory.

    This bypasses temporary OBJ export/import and avoids false topology caused by:
        - split normals
        - UV seams
        - material splits
        - OBJ precision or importer/exporter behavior
    """

    if not HAS_TRIMESH:
        return make_empty_topology_stats("trimesh_or_numpy_not_available")

    eval_obj = obj.evaluated_get(depsgraph)

    try:
        eval_mesh = eval_obj.to_mesh()
    except Exception as exc:
        return make_empty_topology_stats(f"to_mesh_failed: {exc}")

    try:
        eval_mesh.calc_loop_triangles()

        vertex_count = len(eval_mesh.vertices)
        tri_count = len(eval_mesh.loop_triangles)

        if vertex_count <= 0 or tri_count <= 0:
            return make_empty_topology_stats("empty_evaluated_mesh")

        verts = np.empty(vertex_count * 3, dtype=np.float64)
        eval_mesh.vertices.foreach_get("co", verts)
        verts = verts.reshape((-1, 3))

        faces = np.empty(tri_count * 3, dtype=np.int64)
        eval_mesh.loop_triangles.foreach_get("vertices", faces)
        faces = faces.reshape((-1, 3))

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False,
            validate=False,
        )

        # Remove unused vertices after constructing from evaluated mesh.
        # This should not merge split vertices; it only removes unused data.
        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass

        total_faces = int(len(mesh.faces))
        total_vertices = int(len(mesh.vertices))
        total_edges = int(len(mesh.edges_unique))

        if total_faces <= 0:
            return make_empty_topology_stats("trimesh_empty_faces")

        components = mesh.split(only_watertight=False)
        components_raw = len(components)

        face_threshold = int(min_component_faces)
        if min_component_face_ratio > 0.0:
            ratio_threshold = int(math.ceil(total_faces * float(min_component_face_ratio)))
            face_threshold = max(face_threshold, ratio_threshold)

        if face_threshold > 0:
            components_filtered_list = [
                comp for comp in components
                if len(comp.faces) >= face_threshold
            ]
        else:
            components_filtered_list = list(components)

        if not components_filtered_list and components:
            components_filtered_list = [max(components, key=lambda comp: len(comp.faces))]

        components_filtered = len(components_filtered_list)

        main = None
        if components_filtered_list:
            main = max(components_filtered_list, key=lambda comp: len(comp.faces))

        euler_number = int(mesh.euler_number)
        global_genus_float = components_raw - (float(euler_number) / 2.0)
        global_genus = int(round(global_genus_float))

        main_faces = 0
        main_face_ratio = 0.0
        main_euler = None
        main_genus = None
        main_is_watertight = False

        if main is not None and len(main.faces) > 0:
            try:
                main.remove_unreferenced_vertices()
            except Exception:
                pass

            main_faces = int(len(main.faces))
            main_face_ratio = float(main_faces) / float(max(total_faces, 1))
            main_euler = int(main.euler_number)
            main_genus_float = 1.0 - (float(main_euler) / 2.0)
            main_genus = int(round(main_genus_float))
            main_is_watertight = bool(main.is_watertight)

        # Diagnostic: if components are close to faces, it is likely face soup.
        is_triangle_soup_like = False
        if total_faces > 0 and components_raw / float(total_faces) > 0.25:
            is_triangle_soup_like = True

        return {
            "ok": True,
            "reason": "ok",
            "vertices": total_vertices,
            "faces": total_faces,
            "edges": total_edges,
            "components_raw": int(components_raw),
            "components_filtered": int(components_filtered),
            "euler_number": euler_number,
            "global_genus": global_genus,
            "main_faces": main_faces,
            "main_face_ratio": main_face_ratio,
            "main_euler_number": main_euler,
            "main_genus": main_genus,
            "main_is_watertight": main_is_watertight,
            "is_triangle_soup_like": bool(is_triangle_soup_like),
        }

    except Exception as exc:
        return make_empty_topology_stats(f"topology_eval_failed: {exc}")

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

    if int(stats.get("components_filtered", 999999)) != 1:
        return False

    if stats.get("main_genus", None) != 0:
        return False

    if require_watertight and not bool(stats.get("main_is_watertight", False)):
        return False

    return True


def format_topology_line(stats: dict[str, Any]) -> str:
    if not stats.get("ok", False):
        return f"topology_eval=FAILED reason={stats.get('reason')}"

    return (
        f"V={stats['vertices']}, F={stats['faces']}, "
        f"C_raw={stats['components_raw']}, C_filtered={stats['components_filtered']}, "
        f"g_main={stats['main_genus']}, watertight={stats['main_is_watertight']}, "
        f"main_face_ratio={stats['main_face_ratio']:.4f}, "
        f"triangle_soup_like={stats['is_triangle_soup_like']}"
    )


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------

def load_merge_rules(args: argparse.Namespace) -> dict[str, Any]:
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"📄 Loading config from: {config_path}", flush=True)
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    print("ℹ️ No config provided. Defaulting to merge all meshes into 'main_mesh_VDB'.", flush=True)
    all_mesh_names = [obj.name for obj in bpy.data.objects if obj.type == "MESH"]

    return {
        "main_mesh_VDB": {
            "source_meshes": all_mesh_names,
        }
    }


def resolve_voxel_size(args: argparse.Namespace, config: dict[str, Any]) -> float:
    voxel_size = args.voxel_size if args.voxel_size is not None else config.get("voxel_size", 0.05)
    voxel_size = float(voxel_size)

    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    return voxel_size


def resolve_offset_mode(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[bool, float | None, str]:
    """
    Returns:
        use_auto_offset, manual_offset, reason
    """

    if args.offset is not None:
        return False, float(args.offset), "manual_cli_offset"

    if args.auto_offset:
        return True, None, "forced_auto_offset"

    if "offset" in config:
        return False, float(config["offset"]), "manual_config_offset"

    return True, None, "default_auto_offset"


def export_selected_obj(obj: bpy.types.Object, obj_path: Path) -> None:
    ensure_object_mode()

    obj_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    print(f"     -> Exporting final OBJ: {obj_path}", flush=True)
    bpy.ops.wm.obj_export(
        filepath=str(obj_path),
        export_selected_objects=True,
    )


def cleanup_object(obj: bpy.types.Object) -> None:
    try:
        ensure_object_mode()
        bpy.ops.object.select_all(action="DESELECT")
        obj.hide_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.delete()
    except Exception as exc:
        print(f"     ⚠️ Failed to delete temporary object {obj.name}: {exc}", flush=True)


def process_one_rule(
    target_name: str,
    config: dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
    depsgraph: bpy.types.Depsgraph,
) -> dict[str, Any] | None:
    source_names = list(config.get("source_meshes", []))

    valid_sources = []
    missing_sources = []

    for name in source_names:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            missing_sources.append(name)
        else:
            valid_sources.append(obj)

    if not valid_sources:
        print(f"  ⏭️ Skip '{target_name}': no valid source meshes. Missing={missing_sources}", flush=True)
        return None

    voxel_size = resolve_voxel_size(args, config)
    use_auto_offset, manual_offset, offset_mode_reason = resolve_offset_mode(args, config)

    print(f"\n  🔄 Processing '{target_name}'...", flush=True)
    print(f"     Sources: {[obj.name for obj in valid_sources]}", flush=True)
    if missing_sources:
        print(f"     ⚠️ Missing sources ignored: {missing_sources}", flush=True)

    print(f"     voxel_size={voxel_size}", flush=True)
    print(f"     offset_mode={offset_mode_reason}", flush=True)

    temp_obj = duplicate_join_and_staticize_sources(
        valid_sources,
        temp_name=f"TEMP_{target_name}",
        apply_scale=bool(args.apply_scale),
    )

    final_obj_path = out_dir / f"{target_name}.obj"

    # Add SDF modifier after staticization, so it should be first and clean.
    initial_offset = float(manual_offset) if manual_offset is not None else 0.0
    modifier = create_sdf_grid_geonodes(
        temp_obj,
        voxel_size=voxel_size,
        offset_distance=initial_offset,
        band_width_padding=int(args.band_width_padding),
    )

    selected_offset = initial_offset
    selected_stats: dict[str, Any] | None = None
    auto_attempts: list[dict[str, Any]] = []

    if use_auto_offset:
        if not HAS_TRIMESH:
            print(
                "     ⚠️ trimesh/numpy not available. "
                "Auto offset disabled. Fallback to offset = 0.5 * voxel_size.",
                flush=True,
            )
            selected_offset = 0.5 * voxel_size
            update_sdf_nodes(
                modifier,
                voxel_size=voxel_size,
                offset_distance=selected_offset,
                band_width_padding=int(args.band_width_padding),
            )
            depsgraph.update()
            selected_stats = evaluate_topology_from_evaluated_mesh(
                temp_obj,
                depsgraph,
                min_component_faces=int(args.min_component_faces),
                min_component_face_ratio=float(args.min_component_face_ratio),
            )
        else:
            print(
                "     -> Auto-tuning offset by evaluated-mesh topology "
                f"(start=0.5x, step=0.5x, max={args.max_offset_multiplier}x)",
                flush=True,
            )

            multiplier = 0.5
            step = 0.5
            max_multiplier = float(args.max_offset_multiplier)
            success = False

            while multiplier <= max_multiplier + 1e-9:
                current_offset = voxel_size * multiplier

                update_sdf_nodes(
                    modifier,
                    voxel_size=voxel_size,
                    offset_distance=current_offset,
                    band_width_padding=int(args.band_width_padding),
                )
                depsgraph.update()

                stats = evaluate_topology_from_evaluated_mesh(
                    temp_obj,
                    depsgraph,
                    min_component_faces=int(args.min_component_faces),
                    min_component_face_ratio=float(args.min_component_face_ratio),
                )

                ok = topology_is_success(
                    stats,
                    require_watertight=bool(args.require_watertight),
                )

                mark = "✅ success" if ok else "❌ continue"
                print(
                    f"     [Attempt] multiplier={multiplier:.1f}x "
                    f"offset={current_offset:.6g} | {format_topology_line(stats)} | {mark}",
                    flush=True,
                )

                attempt_record = {
                    "multiplier": multiplier,
                    "offset": current_offset,
                    "success": bool(ok),
                    "topology": stats,
                }
                auto_attempts.append(attempt_record)

                selected_offset = current_offset
                selected_stats = stats

                if ok:
                    success = True
                    break

                multiplier += step

            if not success:
                print(
                    f"     ⚠️ No perfect topology found before max multiplier "
                    f"{max_multiplier}x. Using last attempted offset={selected_offset:.6g}.",
                    flush=True,
                )

            # Write selected offset back before final apply/export.
            update_sdf_nodes(
                modifier,
                voxel_size=voxel_size,
                offset_distance=selected_offset,
                band_width_padding=int(args.band_width_padding),
            )
            depsgraph.update()

    else:
        assert manual_offset is not None
        selected_offset = float(manual_offset)

        print(
            f"     -> Using manual offset={selected_offset:.6g}; auto tuning disabled.",
            flush=True,
        )

        update_sdf_nodes(
            modifier,
            voxel_size=voxel_size,
            offset_distance=selected_offset,
            band_width_padding=int(args.band_width_padding),
        )
        depsgraph.update()

        selected_stats = evaluate_topology_from_evaluated_mesh(
            temp_obj,
            depsgraph,
            min_component_faces=int(args.min_component_faces),
            min_component_face_ratio=float(args.min_component_face_ratio),
        )

        print(f"     Manual topology check | {format_topology_line(selected_stats)}", flush=True)

    # Final apply and export only once.
    print(
        f"     -> Applying final SDF modifier with voxel_size={voxel_size}, "
        f"offset={selected_offset:.6g}",
        flush=True,
    )

    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    temp_obj.select_set(True)
    bpy.context.view_layer.objects.active = temp_obj

    result = bpy.ops.object.modifier_apply(modifier=modifier.name)
    if result != {"FINISHED"}:
        print(f"     ⚠️ modifier_apply returned {result}", flush=True)

    temp_obj.name = target_name
    temp_obj.data.name = target_name

    try:
        temp_obj.modifiers.clear()
    except Exception:
        pass

    try:
        temp_obj.vertex_groups.clear()
    except Exception:
        pass

    export_selected_obj(temp_obj, final_obj_path)

    cleanup_object(temp_obj)

    print(f"     -> Final mesh saved: {final_obj_path}", flush=True)

    return {
        "obj_path": str(final_obj_path.resolve()),
        "source_meshes": [obj.name for obj in valid_sources],
        "missing_source_meshes": missing_sources,
        "voxel_size": voxel_size,
        "offset": selected_offset,
        "offset_mode": offset_mode_reason,
        "band_width_padding": int(args.band_width_padding),
        "topology": selected_stats,
        "auto_attempts": auto_attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prep 02a: SDF Grid fusion in Blender 5.x with topology-aware auto offset."
    )

    parser.add_argument(
        "--input_blend",
        type=str,
        required=True,
        help="Path to input Prep 01 .blend file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config. If omitted, all scene meshes are fused into main_mesh_VDB.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for OBJ files and manifest.json.",
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Override voxel size. If omitted, use config voxel_size or fallback 0.05.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="Manual SDF offset. If provided, disables auto offset.",
    )
    parser.add_argument(
        "--auto_offset",
        action="store_true",
        help="Force auto offset even if config contains an offset.",
    )
    parser.add_argument(
        "--max_offset_multiplier",
        type=float,
        default=8.0,
        help="Max auto offset as multiplier of voxel_size. Default: 8.0.",
    )

    parser.add_argument(
        "--band_width_padding",
        type=int,
        default=8,
        help="Extra voxel layers for Mesh to SDF Grid Band Width. Default: 8.",
    )
    parser.add_argument(
        "--min_component_faces",
        type=int,
        default=10,
        help="Ignore connected components with fewer faces than this during component filtering. Default: 10.",
    )
    parser.add_argument(
        "--min_component_face_ratio",
        type=float,
        default=0.0,
        help="Also ignore components smaller than this ratio of total faces. Default: 0.0.",
    )
    parser.add_argument(
        "--require_watertight",
        action="store_true",
        help="Require main component to be watertight for auto-offset success.",
    )
    parser.add_argument(
        "--no_apply_scale",
        dest="apply_scale",
        action="store_false",
        help="Do not apply object scale before SDF. By default scale is applied.",
    )
    parser.set_defaults(apply_scale=True)

    args = parser.parse_args(script_argv())

    input_path = Path(args.input_blend)
    if not input_path.exists():
        raise FileNotFoundError(f"Input blend file not found: {input_path}")

    if args.max_offset_multiplier < 0.5:
        raise ValueError("--max_offset_multiplier must be >= 0.5")

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent / f"{input_path.stem}_sdf_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading: {input_path}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    ensure_object_mode()

    set_all_armatures_rest_pose()
    print_scene_summary()

    if not HAS_TRIMESH:
        print(
            "⚠️ trimesh/numpy is not available in this Python environment. "
            "Auto-offset topology analysis will not work.",
            flush=True,
        )

    merge_rules = load_merge_rules(args)

    manifest: dict[str, Any] = {
        "source_blend": str(input_path.resolve()),
        "bpy_version": bpy.app.version_string,
        "exports": {},
        "details": {},
    }

    depsgraph = bpy.context.evaluated_depsgraph_get()

    print("\n🛠️ Starting SDF Grid Fusion (Blender 5.x)...", flush=True)

    for target_name, config in merge_rules.items():
        result = process_one_rule(
            target_name=target_name,
            config=config,
            args=args,
            out_dir=out_dir,
            depsgraph=depsgraph,
        )

        if result is None:
            continue

        manifest["exports"][target_name] = result["obj_path"]
        manifest["details"][target_name] = {
            k: v for k, v in result.items()
            if k != "obj_path"
        }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(f"\n✅ Prep 02a complete. Manifest saved to: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
