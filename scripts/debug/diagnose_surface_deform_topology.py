#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Diagnose mesh topology issues that can block Blender Surface Deform binding.

Primary use case:
    Surface Deform bind fails with:
        "Target has edges with more than two polygons"

This script is diagnostic-only:
    - does not modify project Data Prep code
    - does not save modified .blend files
    - may temporarily add/remove an in-memory Triangulate modifier for analysis
    - writes a JSON report and prints concise suspect summaries

Typical use:

    conda run --no-capture-output -n lao-lbs \
      python scripts/debug/diagnose_surface_deform_topology.py \
        --prep02_blend /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/data_prep_02/Winter_Samurai_prep_02.blend \
        --prep01_blend /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/data_prep_01/Winter_Samurai_prep_01.blend \
        --manifest /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/data_prep_02/sdf_exports/manifest.json \
        --vdb top \
        --output_json /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/debug_reports/source_top_topology_diag.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def ensure_repo_paths_in_syspath() -> Path:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            src_dir = parent / "src"
            src_s = str(src_dir)
            if src_s not in sys.path:
                sys.path.insert(0, src_s)
            return parent

    raise RuntimeError("Could not locate repository root containing pyproject.toml and src/.")


REPO_ROOT = ensure_repo_paths_in_syspath()

import bpy  # noqa: E402

from ogd_lbs.utils.common import script_argv  # noqa: E402
from ogd_lbs.blender.blend_io import open_blend_file  # noqa: E402
from ogd_lbs.blender.context import update_view_layer  # noqa: E402
from ogd_lbs.blender.object import ensure_object_mode  # noqa: E402


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(int(n)))
        self.size = [1] * int(n)

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve(strict=False)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"JSON root must be object: {p}")
    return obj


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path).expanduser().resolve(strict=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
        f.write("\n")
    print(f"\nSaved topology report: {p}", flush=True)


def unique_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = str(item)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def mesh_update_edges(mesh: bpy.types.Mesh) -> None:
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


def polygon_area(poly: bpy.types.MeshPolygon) -> float:
    try:
        return float(poly.area)
    except Exception:
        return 0.0


def vector_to_list(v: Any) -> list[float]:
    return [float(v[0]), float(v[1]), float(v[2])]


def analyze_mesh_data(
    *,
    mesh: bpy.types.Mesh,
    object_name: str,
    mode: str,
    area_eps: float,
    duplicate_eps: float,
    sample_limit: int,
) -> dict[str, Any]:
    mesh_update_edges(mesh)

    vertex_count = int(len(mesh.vertices))
    edge_count = int(len(mesh.edges))
    face_count = int(len(mesh.polygons))

    coords = [mesh.vertices[i].co.copy() for i in range(vertex_count)]

    edge_to_polys: dict[tuple[int, int], list[int]] = defaultdict(list)
    used_edge_keys: set[tuple[int, int]] = set()
    used_vertices: set[int] = set()

    degenerate_faces: list[dict[str, Any]] = []
    face_areas: dict[int, float] = {}

    dsu = DisjointSet(vertex_count)

    for poly in mesh.polygons:
        verts = [int(v) for v in poly.vertices]
        area = polygon_area(poly)
        face_areas[int(poly.index)] = float(area)

        for v in verts:
            used_vertices.add(v)

        repeated_vertex = len(set(verts)) != len(verts)
        too_few_unique = len(set(verts)) < 3
        tiny_area = float(area) <= float(area_eps)

        if repeated_vertex or too_few_unique or tiny_area:
            reasons = []
            if repeated_vertex:
                reasons.append("repeated_vertex")
            if too_few_unique:
                reasons.append("less_than_3_unique_vertices")
            if tiny_area:
                reasons.append(f"area<=eps({area_eps:g})")

            if len(degenerate_faces) < sample_limit:
                degenerate_faces.append(
                    {
                        "polygon_index": int(poly.index),
                        "vertices": verts,
                        "area": float(area),
                        "reasons": reasons,
                    }
                )

        if len(verts) >= 2:
            for i in range(len(verts)):
                a = int(verts[i])
                b = int(verts[(i + 1) % len(verts)])
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                edge_to_polys[key].append(int(poly.index))
                used_edge_keys.add(key)
                if 0 <= a < vertex_count and 0 <= b < vertex_count:
                    dsu.union(a, b)

    mesh_edge_keys: set[tuple[int, int]] = set()
    loose_edges: list[tuple[int, int]] = []

    for edge in mesh.edges:
        try:
            a = int(edge.vertices[0])
            b = int(edge.vertices[1])
        except Exception:
            continue
        key = (a, b) if a < b else (b, a)
        mesh_edge_keys.add(key)
        if key not in used_edge_keys:
            loose_edges.append(key)

    boundary_edges = [
        key for key, polys in edge_to_polys.items()
        if len(polys) == 1
    ]
    manifold_edges = [
        key for key, polys in edge_to_polys.items()
        if len(polys) == 2
    ]
    nonmanifold_edges_gt2 = [
        (key, polys)
        for key, polys in edge_to_polys.items()
        if len(polys) > 2
    ]

    nonmanifold_samples = []
    for key, polys in nonmanifold_edges_gt2[:sample_limit]:
        a, b = key
        if 0 <= a < vertex_count and 0 <= b < vertex_count:
            length = float((coords[a] - coords[b]).length)
            va = vector_to_list(coords[a])
            vb = vector_to_list(coords[b])
        else:
            length = None
            va = None
            vb = None

        nonmanifold_samples.append(
            {
                "edge_vertices": [int(a), int(b)],
                "face_count": int(len(polys)),
                "polygon_indices": [int(x) for x in polys[:50]],
                "edge_length": length,
                "v0": va,
                "v1": vb,
                "face_areas_first_20": [
                    float(face_areas.get(int(idx), 0.0))
                    for idx in polys[:20]
                ],
            }
        )

    loose_edge_samples = [
        {
            "edge_vertices": [int(a), int(b)],
            "v0": vector_to_list(coords[a]) if 0 <= a < vertex_count else None,
            "v1": vector_to_list(coords[b]) if 0 <= b < vertex_count else None,
        }
        for a, b in loose_edges[:sample_limit]
    ]

    # Connected components over vertices used by polygon edges.
    component_roots: dict[int, int] = {}
    for v in used_vertices:
        root = dsu.find(v)
        component_roots[root] = component_roots.get(root, 0) + 1

    face_component_roots: dict[int, int] = {}
    for poly in mesh.polygons:
        verts = [int(v) for v in poly.vertices]
        if not verts:
            continue
        root = dsu.find(verts[0])
        face_component_roots[root] = face_component_roots.get(root, 0) + 1

    component_vertex_sizes = sorted(component_roots.values(), reverse=True)
    component_face_sizes = sorted(face_component_roots.values(), reverse=True)

    loose_vertices = sorted(set(range(vertex_count)) - used_vertices)

    # Near duplicate vertex candidates by simple quantization.
    duplicate_groups: list[dict[str, Any]] = []
    duplicate_candidate_vertex_count = 0

    if duplicate_eps > 0.0 and vertex_count > 0:
        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        inv = 1.0 / float(duplicate_eps)

        for idx, co in enumerate(coords):
            key = (
                int(round(float(co[0]) * inv)),
                int(round(float(co[1]) * inv)),
                int(round(float(co[2]) * inv)),
            )
            buckets[key].append(int(idx))

        for indices in buckets.values():
            if len(indices) <= 1:
                continue
            duplicate_candidate_vertex_count += len(indices)
            if len(duplicate_groups) < sample_limit:
                first = indices[0]
                duplicate_groups.append(
                    {
                        "count": int(len(indices)),
                        "vertex_indices_first_50": [int(x) for x in indices[:50]],
                        "representative_coord": vector_to_list(coords[first]),
                    }
                )

    try:
        loop_triangle_count = int(len(mesh.loop_triangles))
    except Exception:
        loop_triangle_count = 0

    blocking_for_surface_deform = bool(len(nonmanifold_edges_gt2) > 0)

    return {
        "object_name": object_name,
        "mode": mode,
        "vertex_count": vertex_count,
        "edge_count": edge_count,
        "face_count": face_count,
        "loop_triangle_count": loop_triangle_count,
        "used_polygon_edge_count": int(len(edge_to_polys)),
        "manifold_edge_count": int(len(manifold_edges)),
        "boundary_edge_count": int(len(boundary_edges)),
        "loose_edge_count": int(len(loose_edges)),
        "loose_vertex_count": int(len(loose_vertices)),
        "degenerate_face_count": int(len(degenerate_faces)),
        "edges_with_more_than_2_faces_count": int(len(nonmanifold_edges_gt2)),
        "surface_deform_blocking_edges_gt2": blocking_for_surface_deform,
        "connected_component_count_by_vertices": int(len(component_vertex_sizes)),
        "connected_component_vertex_sizes_first_20": [
            int(x) for x in component_vertex_sizes[:20]
        ],
        "connected_component_count_by_faces": int(len(component_face_sizes)),
        "connected_component_face_sizes_first_20": [
            int(x) for x in component_face_sizes[:20]
        ],
        "duplicate_eps": float(duplicate_eps),
        "duplicate_candidate_group_count": int(len(duplicate_groups)),
        "duplicate_candidate_vertex_count": int(duplicate_candidate_vertex_count),
        "samples": {
            "edges_with_more_than_2_faces": nonmanifold_samples,
            "loose_edges": loose_edge_samples,
            "degenerate_faces": degenerate_faces,
            "duplicate_vertex_groups": duplicate_groups,
            "loose_vertices_first_50": [int(x) for x in loose_vertices[:50]],
        },
    }


def get_evaluated_mesh(
    obj: bpy.types.Object,
) -> tuple[bpy.types.Object, bpy.types.Mesh]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    update_view_layer()

    eval_obj = obj.evaluated_get(depsgraph)
    try:
        mesh = eval_obj.to_mesh()
    except TypeError:
        mesh = eval_obj.to_mesh(depsgraph=depsgraph)

    if mesh is None:
        raise RuntimeError(f"Failed to get evaluated mesh for object: {obj.name}")

    return eval_obj, mesh


def analyze_object(
    *,
    obj: bpy.types.Object,
    modes: list[str],
    area_eps: float,
    duplicate_eps: float,
    sample_limit: int,
) -> dict[str, Any]:
    if obj.type != "MESH":
        raise RuntimeError(f"Expected MESH object: {obj.name}, got {obj.type}")

    out: dict[str, Any] = {
        "object_name": obj.name,
        "object_type": obj.type,
        "data_name": obj.data.name,
        "parent": obj.parent.name if obj.parent else None,
        "modifiers": [
            {
                "name": mod.name,
                "type": mod.type,
                "show_viewport": bool(getattr(mod, "show_viewport", False)),
                "show_render": bool(getattr(mod, "show_render", False)),
            }
            for mod in obj.modifiers
        ],
        "vertex_group_count": int(len(obj.vertex_groups)),
        "modes": {},
    }

    for mode in modes:
        mode = str(mode).strip()
        temp_modifier = None
        eval_obj = None

        try:
            if mode == "raw":
                stats = analyze_mesh_data(
                    mesh=obj.data,
                    object_name=obj.name,
                    mode=mode,
                    area_eps=area_eps,
                    duplicate_eps=duplicate_eps,
                    sample_limit=sample_limit,
                )

            elif mode == "evaluated":
                eval_obj, mesh = get_evaluated_mesh(obj)
                try:
                    stats = analyze_mesh_data(
                        mesh=mesh,
                        object_name=obj.name,
                        mode=mode,
                        area_eps=area_eps,
                        duplicate_eps=duplicate_eps,
                        sample_limit=sample_limit,
                    )
                finally:
                    try:
                        eval_obj.to_mesh_clear()
                    except Exception:
                        pass

            elif mode == "evaluated_triangulated":
                # Approximate the state used before Surface Deform bind:
                # Prep 03 adds a Triangulate modifier to Source_<VDB> before binding.
                temp_modifier = obj.modifiers.new(
                    name="__OGD_TOPO_DIAG_TEMP_TRIANGULATE__",
                    type="TRIANGULATE",
                )
                try:
                    temp_modifier.keep_custom_normals = True
                except Exception:
                    pass
                try:
                    temp_modifier.quad_method = "BEAUTY"
                except Exception:
                    pass
                try:
                    temp_modifier.ngon_method = "BEAUTY"
                except Exception:
                    pass

                update_view_layer()
                eval_obj, mesh = get_evaluated_mesh(obj)
                try:
                    stats = analyze_mesh_data(
                        mesh=mesh,
                        object_name=obj.name,
                        mode=mode,
                        area_eps=area_eps,
                        duplicate_eps=duplicate_eps,
                        sample_limit=sample_limit,
                    )
                finally:
                    try:
                        eval_obj.to_mesh_clear()
                    except Exception:
                        pass

            else:
                raise RuntimeError(f"Unsupported analysis mode: {mode!r}")

            out["modes"][mode] = stats

        except Exception as exc:
            out["modes"][mode] = {
                "object_name": obj.name,
                "mode": mode,
                "error": str(exc),
            }

        finally:
            if temp_modifier is not None:
                try:
                    obj.modifiers.remove(temp_modifier)
                except Exception:
                    pass
                update_view_layer()

    return out


def diagnose_blend_objects(
    *,
    blend_path: Path,
    object_names: list[str],
    modes: list[str],
    area_eps: float,
    duplicate_eps: float,
    sample_limit: int,
    role: str,
) -> dict[str, Any]:
    open_blend_file(blend_path, role=role, load_ui=False)
    ensure_object_mode()
    update_view_layer()

    objects_report: dict[str, Any] = {}

    for name in unique_keep_order(object_names):
        obj = bpy.data.objects.get(name)

        if obj is None:
            objects_report[name] = {
                "object_name": name,
                "missing": True,
                "error": f"Object not found in blend: {name}",
            }
            continue

        if obj.type != "MESH":
            objects_report[name] = {
                "object_name": name,
                "missing": False,
                "object_type": obj.type,
                "error": f"Object is not MESH: {obj.type}",
            }
            continue

        print(f"Diagnosing {role}: {name}", flush=True)
        objects_report[name] = analyze_object(
            obj=obj,
            modes=modes,
            area_eps=area_eps,
            duplicate_eps=duplicate_eps,
            sample_limit=sample_limit,
        )

    return {
        "blend_path": str(blend_path),
        "role": role,
        "object_names": unique_keep_order(object_names),
        "objects": objects_report,
    }


def get_mode_stat(item: dict[str, Any], preferred_mode: str) -> dict[str, Any] | None:
    modes = item.get("modes")
    if not isinstance(modes, dict):
        return None

    if preferred_mode in modes and isinstance(modes[preferred_mode], dict):
        return modes[preferred_mode]

    for value in modes.values():
        if isinstance(value, dict) and "error" not in value:
            return value

    return None


def collect_rankings(section: dict[str, Any], *, preferred_mode: str) -> list[dict[str, Any]]:
    out = []
    objects = section.get("objects", {})
    if not isinstance(objects, dict):
        return out

    for name, item in objects.items():
        if not isinstance(item, dict) or item.get("missing"):
            continue

        stat = get_mode_stat(item, preferred_mode)
        if not stat:
            continue

        out.append(
            {
                "object_name": str(name),
                "mode": stat.get("mode"),
                "edges_with_more_than_2_faces_count": int(
                    stat.get("edges_with_more_than_2_faces_count", 0)
                ),
                "degenerate_face_count": int(stat.get("degenerate_face_count", 0)),
                "loose_edge_count": int(stat.get("loose_edge_count", 0)),
                "boundary_edge_count": int(stat.get("boundary_edge_count", 0)),
                "connected_component_count_by_faces": int(
                    stat.get("connected_component_count_by_faces", 0)
                ),
                "face_count": int(stat.get("face_count", 0)),
                "vertex_count": int(stat.get("vertex_count", 0)),
            }
        )

    out.sort(
        key=lambda x: (
            int(x["edges_with_more_than_2_faces_count"]),
            int(x["degenerate_face_count"]),
            int(x["loose_edge_count"]),
            int(x["connected_component_count_by_faces"]),
            int(x["face_count"]),
        ),
        reverse=True,
    )
    return out


def print_table(title: str, rankings: list[dict[str, Any]], *, limit: int = 30) -> None:
    print("\n" + "=" * 96, flush=True)
    print(title, flush=True)
    print("=" * 96, flush=True)

    if not rankings:
        print("<empty>", flush=True)
        return

    print(
        f"{'object':<32} {'mode':<24} {'edge>2':>8} {'degen':>8} "
        f"{'looseE':>8} {'boundary':>9} {'compF':>7} {'V':>8} {'F':>8}",
        flush=True,
    )

    for item in rankings[:limit]:
        print(
            f"{item['object_name']:<32} "
            f"{str(item['mode']):<24} "
            f"{int(item['edges_with_more_than_2_faces_count']):>8} "
            f"{int(item['degenerate_face_count']):>8} "
            f"{int(item['loose_edge_count']):>8} "
            f"{int(item['boundary_edge_count']):>9} "
            f"{int(item['connected_component_count_by_faces']):>7} "
            f"{int(item['vertex_count']):>8} "
            f"{int(item['face_count']):>8}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose topology issues that may block Surface Deform binding."
    )

    parser.add_argument("--prep02_blend", type=str, required=True)
    parser.add_argument("--prep01_blend", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--vdb", type=str, default="top")
    parser.add_argument(
        "--object",
        dest="extra_objects",
        action="append",
        default=[],
        help="Additional Prep02 object name to diagnose. Can be repeated.",
    )
    parser.add_argument(
        "--component",
        dest="extra_components",
        action="append",
        default=[],
        help="Additional Prep01 component object name to diagnose. Can be repeated.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=["raw", "evaluated", "evaluated_triangulated"],
        default=[],
        help=(
            "Analysis mode. Can be repeated. Default: raw + evaluated + "
            "evaluated_triangulated."
        ),
    )
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--area_eps", type=float, default=1e-12)
    parser.add_argument("--duplicate_eps", type=float, default=1e-6)
    parser.add_argument("--sample_limit", type=int, default=30)
    parser.add_argument(
        "--fail_on_edges_gt2",
        action="store_true",
        help="Exit nonzero if any diagnosed object has edges with >2 faces.",
    )

    args = parser.parse_args(script_argv())

    prep02_blend = Path(args.prep02_blend).expanduser().resolve(strict=False)
    prep01_blend = (
        Path(args.prep01_blend).expanduser().resolve(strict=False)
        if args.prep01_blend
        else None
    )
    manifest_path = (
        Path(args.manifest).expanduser().resolve(strict=False)
        if args.manifest
        else None
    )

    modes = args.mode or ["raw", "evaluated", "evaluated_triangulated"]
    vdb_name = str(args.vdb)

    manifest_obj: dict[str, Any] = {}
    component_names: list[str] = []

    if manifest_path is not None:
        manifest_obj = load_json(manifest_path)
        mapping = manifest_obj.get("vdb_mapping", {})
        if not isinstance(mapping, dict):
            raise RuntimeError(f"manifest vdb_mapping must be object: {manifest_path}")

        raw_components = mapping.get(vdb_name, [])
        if raw_components is None:
            raw_components = []
        if not isinstance(raw_components, list):
            raise RuntimeError(f"manifest vdb_mapping[{vdb_name!r}] must be list.")

        component_names.extend(str(x) for x in raw_components)

    component_names.extend(str(x) for x in args.extra_components)
    component_names = unique_keep_order(component_names)

    prep02_object_names = unique_keep_order(
        [
            f"Source_{vdb_name}",
            vdb_name,
            *[str(x) for x in args.extra_objects],
        ]
    )

    print("=" * 96, flush=True)
    print("Surface Deform topology diagnostic", flush=True)
    print("=" * 96, flush=True)
    print(f"repo_root      : {REPO_ROOT}", flush=True)
    print(f"prep02_blend   : {prep02_blend}", flush=True)
    print(f"prep01_blend   : {prep01_blend}", flush=True)
    print(f"manifest       : {manifest_path}", flush=True)
    print(f"vdb            : {vdb_name}", flush=True)
    print(f"prep02 objects : {prep02_object_names}", flush=True)
    print(f"components     : {component_names}", flush=True)
    print(f"modes          : {modes}", flush=True)
    print(f"output_json    : {args.output_json}", flush=True)
    print("=" * 96, flush=True)

    prep02_report = diagnose_blend_objects(
        blend_path=prep02_blend,
        object_names=prep02_object_names,
        modes=modes,
        area_eps=float(args.area_eps),
        duplicate_eps=float(args.duplicate_eps),
        sample_limit=int(args.sample_limit),
        role="Prep02 blend",
    )

    prep01_report: dict[str, Any] | None = None

    if prep01_blend is not None and component_names:
        prep01_report = diagnose_blend_objects(
            blend_path=prep01_blend,
            object_names=component_names,
            modes=modes,
            area_eps=float(args.area_eps),
            duplicate_eps=float(args.duplicate_eps),
            sample_limit=int(args.sample_limit),
            role="Prep01 source-components blend",
        )

    report: dict[str, Any] = {
        "schema": "ogd_lbs_surface_deform_topology_diagnostic_v1",
        "repo_root": str(REPO_ROOT),
        "inputs": {
            "prep02_blend": str(prep02_blend),
            "prep01_blend": str(prep01_blend) if prep01_blend else None,
            "manifest": str(manifest_path) if manifest_path else None,
            "vdb": vdb_name,
            "modes": modes,
            "area_eps": float(args.area_eps),
            "duplicate_eps": float(args.duplicate_eps),
            "sample_limit": int(args.sample_limit),
        },
        "manifest_vdb_mapping_for_vdb": component_names,
        "prep02": prep02_report,
        "prep01_components": prep01_report,
        "rankings": {},
        "notes": [
            "Surface Deform bind is known to fail when the modifier target has edges shared by more than two polygons.",
            "For Prep03 Source_<VDB> targets, evaluated_triangulated approximates the temporary Triangulate modifier added before binding.",
            "This script is diagnostic-only and does not save modified blend files.",
        ],
    }

    prep02_rank = collect_rankings(
        prep02_report,
        preferred_mode="evaluated_triangulated",
    )
    report["rankings"]["prep02_by_edges_gt2"] = prep02_rank

    print_table(
        "Prep02 objects ranked by Surface-Deform-blocking edge>2 count",
        prep02_rank,
    )

    any_edges_gt2 = any(
        int(item.get("edges_with_more_than_2_faces_count", 0)) > 0
        for item in prep02_rank
    )

    if prep01_report is not None:
        prep01_rank = collect_rankings(
            prep01_report,
            preferred_mode="evaluated_triangulated",
        )
        report["rankings"]["prep01_components_by_edges_gt2"] = prep01_rank

        print_table(
            "Prep01 source components ranked by Surface-Deform-blocking edge>2 count",
            prep01_rank,
        )

        any_edges_gt2 = any_edges_gt2 or any(
            int(item.get("edges_with_more_than_2_faces_count", 0)) > 0
            for item in prep01_rank
        )

    dump_json(args.output_json, report)

    print("\n" + "=" * 96, flush=True)
    print("Diagnostic complete.", flush=True)
    print("=" * 96, flush=True)

    if any_edges_gt2:
        print(
            "Found at least one object/mode with edges shared by more than two faces.",
            flush=True,
        )
        print(
            "Inspect report.samples.edges_with_more_than_2_faces for offending edge samples.",
            flush=True,
        )
    else:
        print(
            "No edges with more than two faces were found in the diagnosed modes.",
            flush=True,
        )
        print(
            "If Surface Deform still fails, next suspects are concavity, doubles, "
            "collinear edges/faces, or Blender operator-specific target validation.",
            flush=True,
        )

    if args.fail_on_edges_gt2 and any_edges_gt2:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
