#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSDR Step 03: Create low-poly skinned mesh shapes with QEM.

Default pipeline semantics:
    --input_dir <work_dir>
        reads : <work_dir>/data2
        writes: <work_dir>/data3

    --input_dir <work_dir>/data2
        reads : <work_dir>/data2
        writes: <work_dir>/data3

Explicit automation mode:
    --input_dir <custom_input_tree> --output_dir <custom_output_tree>
        reads : <custom_input_tree>
        writes: <custom_output_tree>

Configuration:
    Optional config file:
        --config /path/to/ssdr_03_qem_config.json

    If --config is omitted, this script checks:
        <work_dir>/ssdr_03_qem_config.json

    If no config exists, built-in defaults are used.

Precedence:
    explicit CLI overrides
      > config mesh_overrides[mesh_name]
      > config global
      > built-in defaults

Input layout:
    data2/
    ├── skeleton_hierarchy_data.pkl
    ├── animation_data.pkl
    └── <mesh>.pkl

Output layout:
    data3/
    ├── skeleton_hierarchy_data.pkl
    ├── animation_data.pkl
    ├── <mesh>.pkl
    └── step03_qem_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from _env import ensure_repo_paths_in_syspath
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from ._env import ensure_repo_paths_in_syspath

ensure_repo_paths_in_syspath(__file__)

try:
    from _stage_constants import (
        CORE_STEMS,
        STEP_SUMMARY_STEMS as SUMMARY_STEMS,
    )
    from _stage_io import prepare_output_dir, write_stage_summary
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from ._stage_constants import (
        CORE_STEMS,
        STEP_SUMMARY_STEMS as SUMMARY_STEMS,
    )
    from ._stage_io import prepare_output_dir, write_stage_summary



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
from ogd_lbs.fitbox.schema import (  # noqa: E402
    KEY_ANIMATION_POSES,
    KEY_BAKED_NORMALS,
    KEY_BAKED_VERTICES,
    KEY_FACE_MATERIALS,
    KEY_FACE_UVS,
    KEY_FACES,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTEX_NORMALS,
    KEY_VERTICES,
)
from ogd_lbs.fitbox.validation import (  # noqa: E402
    validate_fitbox_mesh,
    validate_skeleton_and_animation,
)
from ogd_lbs.qem.qem import (  # noqa: E402
    QEMOptions,
    quadric_edge_collapse_decimation,
)
from paths import (  # noqa: E402
    resolve_ssdr_stage_dirs,
    validate_ssdr_stage_resolution,
    print_ssdr_resolution_info,
)


FULL_SKELETON = "skeleton_hierarchy_data.pkl"
FULL_ANIMATION = "animation_data.pkl"


BUILTIN_DEFAULTS: dict[str, Any] = {
    "simp_rate_base": 0.1,
    "min_target_faces": 100,
    "maximum_error": float("inf"),
    "qem_frames": "all",
    "qem_sample_count": 16,
    "qem_profile_detail": False,
    "preserve_topology": False,
    "preserve_boundary_mode": "soft",
    "boundary_weight": 1000.0,
    "preserve_normal": True,
    "normal_flip_threshold": 0.0,
    "quality_check": True,
    "quality_thr": 0.3,
    "hard_quality_check": False,
    "attribute_merge": "edge_linear",
    "cleanup_mesh": True,
    "validate_output": True,
    "weak_validate": False,
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"yes", "true", "t", "y", "1", "on"}:
        return True
    if text in {"no", "false", "f", "n", "0", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}.")


def add_optional_bool_arg(parser: argparse.ArgumentParser, name: str, *, help_text: str = "") -> None:
    parser.add_argument(
        f"--{name}",
        dest=name,
        nargs="?",
        const=True,
        default=None,
        type=str2bool,
        help=help_text,
    )
    parser.add_argument(
        f"--no_{name}",
        dest=name,
        action="store_false",
        help=argparse.SUPPRESS,
    )


def coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if text in {"-inf", "-infinity"}:
        return float("-inf")

    try:
        return float(text)
    except Exception as exc:
        raise ValueError(f"Could not parse {field_name} as float: {value!r}") from exc


def coerce_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Could not parse {field_name} as int: {value!r}") from exc


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(f"QEM config must be a JSON object: {path}")

    return obj


def resolve_config_path(
    *,
    explicit_config: str | None,
    work_dir: Path,
) -> tuple[Path | None, str, dict[str, Any]]:
    if explicit_config is not None and str(explicit_config).strip():
        path = Path(explicit_config).expanduser().resolve(strict=False)
        if not path.is_file():
            raise FileNotFoundError(f"Explicit Step 03 config does not exist: {path}")
        return path, "EXPLICIT_OVERRIDE", load_json_file(path)

    default_path = (work_dir / "ssdr_03_qem_config.json").resolve(strict=False)
    if default_path.is_file():
        return default_path, "WORK_DIR_DEFAULT", load_json_file(default_path)

    return None, "MISSING_USING_BUILTIN_DEFAULTS", {}


def normalize_config(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    global_cfg = raw.get("global", {})
    mesh_overrides = raw.get("mesh_overrides", {})

    if global_cfg is None:
        global_cfg = {}
    if mesh_overrides is None:
        mesh_overrides = {}

    if not isinstance(global_cfg, dict):
        raise RuntimeError("Step 03 config field 'global' must be an object.")

    if not isinstance(mesh_overrides, dict):
        raise RuntimeError("Step 03 config field 'mesh_overrides' must be an object.")

    normalized_overrides: dict[str, dict[str, Any]] = {}
    for name, value in mesh_overrides.items():
        if not isinstance(value, dict):
            raise RuntimeError(f"mesh_overrides[{name!r}] must be an object.")
        normalized_overrides[str(name)] = dict(value)

    return dict(global_cfg), normalized_overrides


def parse_specific_rates(text: str | None) -> dict[str, float]:
    if text is None or not str(text).strip():
        return {}

    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise RuntimeError("--specific_rates must be a JSON object mapping mesh name to multiplier.")

    return {str(k): float(v) for k, v in obj.items()}


def parse_mesh_rate_items(items: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}

    for item in items or []:
        if "=" not in item:
            raise RuntimeError(f"--mesh_rate item must use mesh=rate format, got {item!r}")

        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()

        if not name:
            raise RuntimeError(f"--mesh_rate mesh name cannot be empty: {item!r}")

        out[name] = float(value)

    return out


def cli_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "simp_rate_base",
        "min_target_faces",
        "maximum_error",
        "qem_frames",
        "qem_sample_count",
        "qem_profile_detail",
        "preserve_topology",
        "preserve_boundary_mode",
        "boundary_weight",
        "preserve_normal",
        "normal_flip_threshold",
        "quality_check",
        "quality_thr",
        "hard_quality_check",
        "attribute_merge",
        "cleanup_mesh",
        "validate_output",
        "weak_validate",
    ]

    out: dict[str, Any] = {}

    for key in keys:
        value = getattr(args, key)
        if value is not None:
            out[key] = value

    return out


def merge_global_config(
    *,
    config_global: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(BUILTIN_DEFAULTS)
    merged.update(config_global)
    merged.update(cli_overrides)

    merged["simp_rate_base"] = coerce_float(merged["simp_rate_base"], field_name="simp_rate_base")
    merged["min_target_faces"] = coerce_int(merged["min_target_faces"], field_name="min_target_faces")
    merged["maximum_error"] = coerce_float(merged["maximum_error"], field_name="maximum_error")
    merged["qem_sample_count"] = coerce_int(merged["qem_sample_count"], field_name="qem_sample_count")
    merged["boundary_weight"] = coerce_float(merged["boundary_weight"], field_name="boundary_weight")
    merged["normal_flip_threshold"] = coerce_float(
        merged["normal_flip_threshold"],
        field_name="normal_flip_threshold",
    )
    merged["quality_thr"] = coerce_float(merged["quality_thr"], field_name="quality_thr")

    for key in [
        "qem_profile_detail",
        "preserve_topology",
        "preserve_normal",
        "quality_check",
        "hard_quality_check",
        "cleanup_mesh",
        "validate_output",
        "weak_validate",
    ]:
        merged[key] = bool(merged[key])

    if merged["qem_frames"] not in {"all", "sampled", "bind"}:
        raise RuntimeError(f"Unsupported qem_frames={merged['qem_frames']!r}.")

    if merged["preserve_boundary_mode"] not in {"none", "soft", "hard"}:
        raise RuntimeError(
            f"Unsupported preserve_boundary_mode={merged['preserve_boundary_mode']!r}."
        )

    if merged["attribute_merge"] not in {"survivor", "average", "edge_linear"}:
        raise RuntimeError(f"Unsupported attribute_merge={merged['attribute_merge']!r}.")

    return merged


def effective_mesh_config(
    *,
    mesh_name: str,
    global_cfg: dict[str, Any],
    mesh_overrides: dict[str, dict[str, Any]],
    cli_specific_rates: dict[str, float],
    cli_mesh_rates: dict[str, float],
) -> dict[str, Any]:
    cfg = dict(global_cfg)

    override = dict(mesh_overrides.get(mesh_name, {}))
    cfg.update(override)

    if mesh_name in cli_specific_rates:
        cfg["simp_rate_multiplier"] = float(cli_specific_rates[mesh_name])

    if mesh_name in cli_mesh_rates:
        cfg["simp_rate_multiplier"] = float(cli_mesh_rates[mesh_name])

    cfg.setdefault("simp_rate_multiplier", 1.0)

    cfg["simp_rate_base"] = coerce_float(cfg["simp_rate_base"], field_name=f"{mesh_name}.simp_rate_base")
    cfg["simp_rate_multiplier"] = coerce_float(
        cfg["simp_rate_multiplier"],
        field_name=f"{mesh_name}.simp_rate_multiplier",
    )
    cfg["min_target_faces"] = coerce_int(cfg["min_target_faces"], field_name=f"{mesh_name}.min_target_faces")
    cfg["maximum_error"] = coerce_float(cfg["maximum_error"], field_name=f"{mesh_name}.maximum_error")
    cfg["qem_sample_count"] = coerce_int(cfg["qem_sample_count"], field_name=f"{mesh_name}.qem_sample_count")

    if "target_faces" in cfg and cfg["target_faces"] is not None:
        cfg["target_faces"] = coerce_int(cfg["target_faces"], field_name=f"{mesh_name}.target_faces")

    return cfg


def qem_options_from_config(cfg: dict[str, Any]) -> QEMOptions:
    return QEMOptions(
        preserve_boundary_mode=str(cfg["preserve_boundary_mode"]),
        boundary_weight=float(cfg["boundary_weight"]),
        preserve_topology=bool(cfg["preserve_topology"]),
        preserve_normal=bool(cfg["preserve_normal"]),
        normal_flip_threshold=float(cfg["normal_flip_threshold"]),
        quality_check=bool(cfg["quality_check"]),
        quality_thr=float(cfg["quality_thr"]),
        hard_quality_check=bool(cfg["hard_quality_check"]),
        maximum_error=float(cfg["maximum_error"]),
        attribute_merge=str(cfg["attribute_merge"]),
        qem_frames=str(cfg["qem_frames"]),
        qem_sample_count=int(cfg["qem_sample_count"]),
        qem_profile_detail=bool(cfg["qem_profile_detail"]),
        cleanup_mesh=bool(cfg["cleanup_mesh"]),
    )


def discover_mesh_files(input_tree: Path) -> list[Path]:
    paths: list[Path] = []

    for path in sorted(input_tree.glob("*.pkl"), key=lambda p: p.name):
        if not path.is_file():
            continue

        if path.stem in CORE_STEMS:
            continue

        if path.stem in SUMMARY_STEMS:
            continue

        paths.append(path)

    return paths


def safe_normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)

    if weights.ndim != 2:
        raise RuntimeError(f"vertex_bone_weights must be 2D after QEM, got {weights.shape}")

    out = np.maximum(weights, 0.0)
    row_sums = out.sum(axis=1, keepdims=True)

    zero_rows = row_sums[:, 0] <= 1e-10
    if np.any(zero_rows):
        if out.shape[1] <= 0:
            raise RuntimeError("Cannot normalize vertex_bone_weights with zero bone columns.")
        out[zero_rows, :] = 1.0 / float(out.shape[1])
        row_sums = out.sum(axis=1, keepdims=True)

    out = out / np.maximum(row_sums, 1e-10)
    return out


def maybe_index_face_sidecar(value: Any, face_map: np.ndarray | None) -> Any:
    if face_map is None:
        return value

    arr = np.asarray(value, dtype=object)

    if arr.size == 0:
        return value

    if arr.shape[0] <= int(np.max(face_map, initial=-1)):
        return value

    return arr[face_map]


def triangulate_faces_for_qem(
    mesh_obj: dict[str, Any],
    faces: np.ndarray,
) -> tuple[np.ndarray, Any | None, Any | None, dict[str, Any]]:
    """
    Convert polygon faces to triangles for QEM.

    QEM core expects triangle faces (F, 3). Fitbox input may contain quads or
    other fixed-width polygon faces. We triangulate each polygon by fan:

        [v0, v1, v2, v3] -> [v0, v1, v2], [v0, v2, v3]

    Sidecars are expanded consistently:
        face_uvs       : per-face UV polygon data
        face_materials : duplicated per generated triangle
    """
    faces = np.asarray(faces, dtype=np.int64)

    if faces.ndim != 2 or faces.shape[1] < 3:
        raise RuntimeError(f"Cannot triangulate invalid faces shape: {faces.shape}")

    polygon_size = int(faces.shape[1])
    original_face_count = int(faces.shape[0])

    if polygon_size == 3:
        return faces, mesh_obj.get(KEY_FACE_UVS, None), mesh_obj.get(KEY_FACE_MATERIALS, None), {
            "triangulated": False,
            "polygon_size": 3,
            "original_face_count": original_face_count,
            "triangulated_face_count": original_face_count,
            "triangles_per_polygon": 1,
        }

    face_uvs_raw = mesh_obj.get(KEY_FACE_UVS, None)
    face_mats_raw = mesh_obj.get(KEY_FACE_MATERIALS, None)

    face_uvs_arr = None
    if face_uvs_raw is not None:
        candidate = np.asarray(face_uvs_raw, dtype=object)
        if candidate.shape[0] == original_face_count:
            face_uvs_arr = candidate

    face_mats_arr = None
    if face_mats_raw is not None:
        candidate = np.asarray(face_mats_raw, dtype=object)
        if candidate.shape[0] == original_face_count:
            face_mats_arr = candidate

    tri_faces: list[list[int]] = []
    tri_uvs: list[Any] = []
    tri_mats: list[Any] = []

    for face_idx, face in enumerate(faces):
        for j in range(1, polygon_size - 1):
            tri_faces.append([
                int(face[0]),
                int(face[j]),
                int(face[j + 1]),
            ])

            if face_uvs_arr is not None:
                uv_poly = np.asarray(face_uvs_arr[face_idx], dtype=object)
                if uv_poly.ndim >= 1 and uv_poly.shape[0] >= polygon_size:
                    tri_uvs.append([
                        uv_poly[0],
                        uv_poly[j],
                        uv_poly[j + 1],
                    ])

            if face_mats_arr is not None:
                tri_mats.append(face_mats_arr[face_idx])

    tri_faces_arr = np.asarray(tri_faces, dtype=np.int64)

    tri_uvs_out = None
    if face_uvs_arr is not None and len(tri_uvs) == len(tri_faces):
        tri_uvs_out = np.asarray(tri_uvs, dtype=object)

    tri_mats_out = None
    if face_mats_arr is not None and len(tri_mats) == len(tri_faces):
        tri_mats_out = np.asarray(tri_mats, dtype=object)

    info = {
        "triangulated": True,
        "polygon_size": polygon_size,
        "original_face_count": original_face_count,
        "triangulated_face_count": int(tri_faces_arr.shape[0]),
        "triangles_per_polygon": polygon_size - 2,
        "face_uvs_expanded": tri_uvs_out is not None,
        "face_materials_expanded": tri_mats_out is not None,
    }

    return tri_faces_arr, tri_uvs_out, tri_mats_out, info


def process_single_mesh(
    *,
    mesh_path: Path,
    output_tree: Path,
    global_cfg: dict[str, Any],
    mesh_overrides: dict[str, dict[str, Any]],
    cli_specific_rates: dict[str, float],
    cli_mesh_rates: dict[str, float],
    n_poses: int,
    n_bones: int,
) -> dict[str, Any]:
    mesh_obj = load_data(mesh_path)

    if not isinstance(mesh_obj, dict):
        raise RuntimeError(f"Mesh file must contain a dict object: {mesh_path}")

    mesh_name = str(mesh_obj.get("name", mesh_path.stem))

    cfg = effective_mesh_config(
        mesh_name=mesh_name,
        global_cfg=global_cfg,
        mesh_overrides=mesh_overrides,
        cli_specific_rates=cli_specific_rates,
        cli_mesh_rates=cli_mesh_rates,
    )

    validate_output = bool(cfg["validate_output"])
    weak_validate = bool(cfg["weak_validate"])

    validate_fitbox_mesh(
        mesh_obj,
        n_poses=n_poses,
        n_bones=n_bones,
        name=f"{mesh_name}/input",
        weak_validate=weak_validate,
    )

    faces = np.asarray(mesh_obj[KEY_FACES], dtype=np.int64)
    vertices = np.asarray(mesh_obj[KEY_VERTICES], dtype=np.float64)
    vertex_normals = np.asarray(mesh_obj[KEY_VERTEX_NORMALS], dtype=np.float64)
    baked_vertices = np.asarray(mesh_obj[KEY_BAKED_VERTICES], dtype=np.float64)
    baked_normals = np.asarray(mesh_obj[KEY_BAKED_NORMALS], dtype=np.float64)
    vertex_bone_weights = np.asarray(mesh_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)

    input_vertices = int(vertices.shape[0])
    input_faces = int(faces.shape[0])

    faces, triangulated_face_uvs, triangulated_face_materials, triangulation_info = triangulate_faces_for_qem(
        mesh_obj,
        faces,
    )

    qem_input_faces = int(faces.shape[0])

    if bool(triangulation_info.get("triangulated", False)):
        log(
            f"[{mesh_name}] triangulated polygon faces for QEM: "
            f"{input_faces} polygons with K={triangulation_info['polygon_size']} "
            f"-> {qem_input_faces} triangles"
        )

    if "target_faces" in cfg and cfg["target_faces"] is not None:
        target_faces = int(cfg["target_faces"])
        target_source = "target_faces"
    else:
        effective_rate = float(cfg["simp_rate_base"]) * float(cfg["simp_rate_multiplier"])
        target_faces = int(max(math.floor(qem_input_faces * effective_rate), int(cfg["min_target_faces"])))
        target_source = "simp_rate"

    target_faces = max(1, min(int(target_faces), qem_input_faces))

    shape_vertices = np.concatenate((vertices[np.newaxis, ...], baked_vertices), axis=0)
    shape_normals = np.concatenate((vertex_normals[np.newaxis, ...], baked_normals), axis=0)

    options = qem_options_from_config(cfg)

    result = quadric_edge_collapse_decimation(
        vertices=shape_vertices,
        vertex_normals=shape_normals,
        faces=faces,
        target_num_faces=target_faces,
        vertex_attributes={"vbw": vertex_bone_weights},
        options=options,
    )

    stats = result.stats

    log(
        f"[{mesh_name}] {input_faces} -> {stats.actual_faces} faces "
        f"(target={stats.target_faces}, source={target_source}, "
        f"stop={stats.stop_reason})"
    )

    mesh_obj[KEY_VERTICES] = result.vertices[0]
    mesh_obj[KEY_VERTEX_NORMALS] = result.vertex_normals[0]
    mesh_obj[KEY_BAKED_VERTICES] = result.vertices[1:]
    mesh_obj[KEY_BAKED_NORMALS] = result.vertex_normals[1:]
    mesh_obj[KEY_FACES] = result.faces

    if triangulated_face_uvs is not None:
        mesh_obj[KEY_FACE_UVS] = maybe_index_face_sidecar(triangulated_face_uvs, result.face_map)
    elif KEY_FACE_UVS in mesh_obj:
        warn(f"[{mesh_name}] Dropping face_uvs because they could not be expanded through triangulation.")
        mesh_obj.pop(KEY_FACE_UVS, None)

    if triangulated_face_materials is not None:
        mesh_obj[KEY_FACE_MATERIALS] = maybe_index_face_sidecar(triangulated_face_materials, result.face_map)
    elif KEY_FACE_MATERIALS in mesh_obj:
        warn(f"[{mesh_name}] Dropping face_materials because they could not be expanded through triangulation.")
        mesh_obj.pop(KEY_FACE_MATERIALS, None)

    mesh_obj[KEY_VERTEX_BONE_WEIGHTS] = safe_normalize_weights(result.vertex_attributes["vbw"])

    # Persist QEM lineage so downstream experiments can reuse the exact
    # train-fold low-poly topology on held-out test motions.
    mesh_obj["qem_vertex_map"] = np.asarray(result.vertex_map, dtype=np.int64)
    mesh_obj["qem_face_map"] = (
        None
        if result.face_map is None
        else np.asarray(result.face_map, dtype=np.int64)
    )
    mesh_obj["qem_lineage"] = {
        "schema": "ogd_lbs_qem_lineage_v1",
        "source_mesh": str(mesh_path),
        "mesh_name": mesh_name,
        "input_vertices": int(input_vertices),
        "input_faces": int(input_faces),
        "qem_input_faces": int(qem_input_faces),
        "target_faces": int(stats.target_faces),
        "actual_faces": int(stats.actual_faces),
        "target_source": str(target_source),
        "stop_reason": str(stats.stop_reason),
        "vertex_map_semantics": (
            "qem_vertex_map maps output low-poly vertex index to the input "
            "triangulated data2 vertex index."
        ),
        "face_map_semantics": (
            "qem_face_map maps output low-poly face index to the input "
            "triangulated data2 face index when available."
        ),
    }

    if validate_output:
        validate_fitbox_mesh(
            mesh_obj,
            n_poses=n_poses,
            n_bones=n_bones,
            name=f"{mesh_name}/output",
            weak_validate=weak_validate,
        )

    out_path = output_tree / f"{mesh_name}.pkl"
    dump_pkl(out_path, mesh_obj)

    return {
        "operation": "qem_decimate_mesh",
        "name": mesh_name,
        "input": str(mesh_path),
        "output": str(out_path),
        "input_vertices": int(input_vertices),
        "input_faces": int(input_faces),
        "qem_input_faces": int(qem_input_faces),
        "triangulation": triangulation_info,
        "target_faces": int(stats.target_faces),
        "actual_faces": int(stats.actual_faces),
        "has_qem_vertex_map": bool(result.vertex_map is not None),
        "qem_vertex_map_length": int(0 if result.vertex_map is None else len(result.vertex_map)),
        "has_qem_face_map": bool(result.face_map is not None),
        "qem_face_map_length": int(0 if result.face_map is None else len(result.face_map)),
        "target_source": target_source,
        "simp_rate_base": float(cfg["simp_rate_base"]),
        "simp_rate_multiplier": float(cfg.get("simp_rate_multiplier", 1.0)),
        "effective_simp_rate": float(cfg["simp_rate_base"]) * float(cfg.get("simp_rate_multiplier", 1.0)),
        "stop_reason": str(stats.stop_reason),
        "fallback_cleaned_verts": int(stats.fallback_cleaned_verts),
        "fallback_cleaned_faces": int(stats.fallback_cleaned_faces),
        "collapse_success": int(stats.collapse_success),
        "heap_pushes": int(stats.heap_pushes),
        "heap_pops": int(stats.heap_pops),
        "rejected": {
            "invalid": int(stats.rejected_invalid),
            "boundary": int(stats.rejected_boundary),
            "topology": int(stats.rejected_topology),
            "normal_flip": int(stats.rejected_normal_flip),
            "quality": int(stats.rejected_quality),
            "max_error": int(stats.rejected_max_error),
            "stale_edge": int(stats.rejected_stale_edge),
        },
    }


def create_low_poly_tree(
    *,
    input_tree: Path,
    output_tree: Path,
    work_dir: Path,
    config_arg: str | None,
    cli_overrides: dict[str, Any],
    cli_specific_rates: dict[str, float],
    cli_mesh_rates: dict[str, float],
    max_workers: int | None,
    overwrite_output: bool,
    allow_merge: bool,
    write_summary_file: bool,
) -> dict[str, Any]:
    config_path, config_source, raw_config = resolve_config_path(
        explicit_config=config_arg,
        work_dir=work_dir,
    )
    config_global, mesh_overrides = normalize_config(raw_config)
    effective_global = merge_global_config(
        config_global=config_global,
        cli_overrides=cli_overrides,
    )

    prepare_output_dir(
        output_tree,
        overwrite_output=overwrite_output,
        allow_merge=allow_merge,
    )

    skeleton_path = input_tree / FULL_SKELETON
    animation_path = input_tree / FULL_ANIMATION

    if not skeleton_path.is_file():
        fail(f"Missing required skeleton file: {skeleton_path}")

    if not animation_path.is_file():
        fail(f"Missing required animation file: {animation_path}")

    with Timer("time_03_load_core_data"):
        skeleton_obj = load_data(skeleton_path)
        animation_obj = load_data(animation_path)

        if not isinstance(skeleton_obj, dict) or not isinstance(animation_obj, dict):
            fail("skeleton_hierarchy_data.pkl and animation_data.pkl must contain dict objects.")

        n_poses, n_bones = validate_skeleton_and_animation(animation_obj, skeleton_obj)

    with Timer("time_03_discover_mesh_files"):
        mesh_paths = discover_mesh_files(input_tree)

    if not mesh_paths:
        fail(f"No mesh PKL files found under input tree: {input_tree}")

    log("Selected Step 03 QEM inputs:")
    log(f"  input_tree       : {input_tree}")
    log(f"  output_tree      : {output_tree}")
    log(f"  config_path      : {config_path}")
    log(f"  config_source    : {config_source}")
    log(f"  mesh_candidates  : {len(mesh_paths)}")
    log(f"  n_poses          : {n_poses}")
    log(f"  n_bones          : {n_bones}")
    log(f"  max_workers      : {max_workers}")

    with Timer("time_03_copy_core_data"):
        dump_pkl(output_tree / FULL_SKELETON, skeleton_obj)
        dump_pkl(output_tree / FULL_ANIMATION, animation_obj)

    processed: list[dict[str, Any]] = []

    with Timer("time_03_process_meshes"):
        if max_workers == 1:
            for mesh_path in mesh_paths:
                result = process_single_mesh(
                    mesh_path=mesh_path,
                    output_tree=output_tree,
                    global_cfg=effective_global,
                    mesh_overrides=mesh_overrides,
                    cli_specific_rates=cli_specific_rates,
                    cli_mesh_rates=cli_mesh_rates,
                    n_poses=n_poses,
                    n_bones=n_bones,
                )
                processed.append(result)
                log(f"Processed QEM mesh: {result['name']}")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_single_mesh,
                        mesh_path=mesh_path,
                        output_tree=output_tree,
                        global_cfg=effective_global,
                        mesh_overrides=mesh_overrides,
                        cli_specific_rates=cli_specific_rates,
                        cli_mesh_rates=cli_mesh_rates,
                        n_poses=n_poses,
                        n_bones=n_bones,
                    )
                    for mesh_path in mesh_paths
                ]

                for future in as_completed(futures):
                    result = future.result()
                    processed.append(result)
                    log(f"Processed QEM mesh: {result['name']}")

    processed = sorted(processed, key=lambda item: item["name"])

    total_input_faces = int(sum(int(item["input_faces"]) for item in processed))
    total_target_faces = int(sum(int(item["target_faces"]) for item in processed))
    total_actual_faces = int(sum(int(item["actual_faces"]) for item in processed))
    total_fallback_cleaned_verts = int(sum(int(item["fallback_cleaned_verts"]) for item in processed))
    total_fallback_cleaned_faces = int(sum(int(item["fallback_cleaned_faces"]) for item in processed))

    summary = {
        "schema": "ogd_lbs_ssdr_step03_qem_v1",
        "input_tree": str(input_tree),
        "output_tree": str(output_tree),
        "config": {
            "config_path": str(config_path) if config_path is not None else None,
            "config_source": config_source,
            "effective_global": effective_global,
            "mesh_overrides": mesh_overrides,
            "cli_specific_rates": cli_specific_rates,
            "cli_mesh_rates": cli_mesh_rates,
        },
        "core_files": {
            "skeleton_hierarchy_data": str(output_tree / FULL_SKELETON),
            "animation_data": str(output_tree / FULL_ANIMATION),
        },
        "mesh_candidate_count": int(len(mesh_paths)),
        "processed_mesh_count": int(len(processed)),
        "totals": {
            "input_faces": total_input_faces,
            "target_faces": total_target_faces,
            "actual_faces": total_actual_faces,
            "fallback_cleaned_verts": total_fallback_cleaned_verts,
            "fallback_cleaned_faces": total_fallback_cleaned_faces,
        },
        "files": processed,
    }

    if write_summary_file:
        summary_path = output_tree / "step03_qem_summary.json"
        write_stage_summary(summary_path, summary)
        log(f"Step 03 summary written: {summary_path}")

    record_metric("03_mesh_candidate_count", int(len(mesh_paths)))
    record_metric("03_processed_mesh_count", int(len(processed)))
    record_metric("03_total_input_faces", total_input_faces)
    record_metric("03_total_target_faces", total_target_faces)
    record_metric("03_total_actual_faces", total_actual_faces)
    record_metric("03_total_fallback_cleaned_verts", total_fallback_cleaned_verts)
    record_metric("03_total_fallback_cleaned_faces", total_fallback_cleaned_faces)

    if total_input_faces > 0:
        record_metric("03_actual_to_input_face_ratio", float(total_actual_faces) / float(total_input_faces))

    if total_target_faces > 0:
        record_metric("03_actual_to_target_face_ratio", float(total_actual_faces) / float(total_target_faces))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSDR Step 03: create low-poly skinned mesh shapes with QEM."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Primary input. Normally work_dir or work_dir/data2. "
            "With explicit --output_dir, this may also be a custom input tree."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Explicit output tree. If omitted, Step 03 writes to <work_dir>/data3.",
    )
    parser.add_argument(
        "-m",
        "--metrics_dir",
        type=str,
        default=None,
        help="Optional directory to save metrics_03.json.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional Step 03 QEM config JSON. If omitted, the script uses "
            "<work_dir>/ssdr_03_qem_config.json if it exists; otherwise built-in defaults."
        ),
    )

    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--allow_merge", action="store_true")
    parser.add_argument("--no_summary", action="store_true")
    parser.add_argument("--enable_profiling", action="store_true")

    parser.add_argument("--simp_rate_base", type=float, default=None)
    parser.add_argument(
        "--specific_rates",
        type=str,
        default=None,
        help='JSON object mapping mesh_name to simp_rate_multiplier, e.g. {"main_mesh_VDB": 0.5}',
    )
    parser.add_argument(
        "--mesh_rate",
        type=str,
        action="append",
        default=None,
        help="Per-mesh simplification multiplier override in mesh=rate format. Can be repeated.",
    )
    parser.add_argument("--min_target_faces", type=int, default=None)
    parser.add_argument("--maximum_error", type=str, default=None)

    parser.add_argument("--qem_frames", choices=["all", "sampled", "bind"], default=None)
    parser.add_argument("--qem_sample_count", type=int, default=None)
    add_optional_bool_arg(parser, "qem_profile_detail")

    add_optional_bool_arg(parser, "preserve_topology")
    parser.add_argument("--preserve_boundary_mode", choices=["none", "soft", "hard"], default=None)
    parser.add_argument("--boundary_weight", type=float, default=None)
    add_optional_bool_arg(parser, "preserve_normal")
    parser.add_argument("--normal_flip_threshold", type=float, default=None)
    add_optional_bool_arg(parser, "quality_check")
    parser.add_argument("--quality_thr", type=float, default=None)
    add_optional_bool_arg(parser, "hard_quality_check")
    parser.add_argument("--attribute_merge", choices=["survivor", "average", "edge_linear"], default=None)
    add_optional_bool_arg(parser, "cleanup_mesh")
    add_optional_bool_arg(parser, "validate_output")
    add_optional_bool_arg(parser, "weak_validate")

    args = parser.parse_args(script_argv())

    resolution = resolve_ssdr_stage_dirs(
        step_id="03",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="SSDR Step 03: data2 -> data3 QEM Low-Poly Shapes",
    )

    print_ssdr_resolution_info(resolution)

    try:
        validate_ssdr_stage_resolution(resolution)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    cli_overrides = cli_overrides_from_args(args)
    cli_specific_rates = parse_specific_rates(args.specific_rates)
    cli_mesh_rates = parse_mesh_rate_items(args.mesh_rate)

    log("=" * 80)
    log("SSDR Step 03: data2 -> data3 QEM Low-Poly Shapes")
    log(f"work_dir          : {resolution.work_dir}")
    log(f"input_tree        : {resolution.input_tree}")
    log(f"output_tree       : {resolution.output_tree}")
    log(f"input_source      : {resolution.input_source}")
    log(f"output_source     : {resolution.output_source}")
    log(f"config            : {args.config}")
    log(f"cli_overrides     : {sorted(cli_overrides.keys())}")
    log(f"specific_rates    : {cli_specific_rates}")
    log(f"mesh_rates        : {cli_mesh_rates}")
    log(f"max_workers       : {args.max_workers}")
    log(f"overwrite_output  : {bool(args.overwrite_output)}")
    log(f"allow_merge       : {bool(args.allow_merge)}")
    log("=" * 80)

    with Timer("time_03_total"):
        summary = create_low_poly_tree(
            input_tree=resolution.input_tree,
            output_tree=resolution.output_tree,
            work_dir=resolution.work_dir,
            config_arg=args.config,
            cli_overrides=cli_overrides,
            cli_specific_rates=cli_specific_rates,
            cli_mesh_rates=cli_mesh_rates,
            max_workers=args.max_workers,
            overwrite_output=bool(args.overwrite_output),
            allow_merge=bool(args.allow_merge),
            write_summary_file=not bool(args.no_summary),
        )

    if args.metrics_dir:
        dump_metrics(Path(args.metrics_dir) / "metrics_03.json")

    log("SSDR Step 03 complete.")
    log(f"mesh_candidate_count : {summary['mesh_candidate_count']}")
    log(f"processed_mesh_count : {summary['processed_mesh_count']}")
    log(f"total_input_faces    : {summary['totals']['input_faces']}")
    log(f"total_target_faces   : {summary['totals']['target_faces']}")
    log(f"total_actual_faces   : {summary['totals']['actual_faces']}")
    log(f"output_tree          : {resolution.output_tree}")


if __name__ == "__main__":
    main()
