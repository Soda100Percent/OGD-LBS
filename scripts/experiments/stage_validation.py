"""Strict Step02 object validation, separate from generic legacy validators.

No pickle loading, file mutation or algorithm execution in this module.
The caller must authenticate/verify artifact provenance before loading pickle.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from ogd_lbs.fitbox.validation import validate_fitbox_mesh
from ogd_lbs.ssdr.weight_reduction import align_skeleton_hierarchy_to_bone_names

ATOL = 1e-6
GEOMETRY_KEYS = (
    "vertices", "faces", "vertex_normals",
    "baked_animation_vertices", "baked_animation_vertex_normals",
    "scene_frames", "face_uvs", "face_materials", "material_names", "name",
)


class StageValidationError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise StageValidationError(message)


def names(obj, role):
    require(isinstance(obj, dict), f"{role} must be a dictionary")
    values = obj.get("bone_names")
    require(isinstance(values, list) and values
            and all(isinstance(x, str) and x for x in values),
            f"{role}: missing/non-string bone names")
    require(len(set(values)) == len(values), f"{role}: duplicate bone names")
    return values


def skeleton(obj, role):
    bones = names(obj, role)
    parents = obj.get("parent_indices")
    require(isinstance(parents, (list, tuple, np.ndarray))
            and len(parents) == len(bones), f"{role}: parent count mismatch")
    normalized = []
    for p in parents:
        if p is None:
            normalized.append(None)
            continue
        require(isinstance(p, (int, np.integer)) and not isinstance(p, (bool, np.bool_)),
                f"{role}: parent index must be integer")
        p = int(p)
        require(-1 <= p < len(bones), f"{role}: parent index out of bounds")
        normalized.append(None if p == -1 else p)

    for start in range(len(bones)):
        visited = set()
        current = start
        while current is not None:
            require(current not in visited, f"{role}: skeleton cycle")
            visited.add(current)
            current = normalized[current]
    return {**obj, "bone_names": list(bones), "parent_indices": normalized}


def numeric(value, role):
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise StageValidationError(f"{role}: cannot form an array") from exc
    require(array.dtype.kind in "fiu", f"{role}: expected real numeric values")
    require(np.isfinite(array).all(), f"{role}: NaN/Inf")
    return array


def animation(obj, expected_bones, role):
    bones = names(obj, role)
    require(set(bones) == set(expected_bones), f"{role}: bone set mismatch")
    poses = numeric(obj.get("animation_poses"), f"{role}.animation_poses")
    rest = numeric(obj.get("rest_pose"), f"{role}.rest_pose")
    require(poses.ndim == 4 and poses.shape[0] > 0
            and poses.shape[1:] == (len(bones), 4, 4), f"{role}: invalid pose shape")
    require(rest.shape == (len(bones), 4, 4), f"{role}: invalid rest shape")
    last = np.asarray([0.0, 0.0, 0.0, 1.0])
    require(np.allclose(rest[..., 3, :], last, atol=ATOL, rtol=0)
            and np.allclose(poses[..., 3, :], last, atol=ATOL, rtol=0),
            f"{role}: invalid affine matrix bottom row")
    try:
        inverse = np.linalg.inv(rest)
    except np.linalg.LinAlgError as exc:
        raise StageValidationError(f"{role}: singular rest pose") from exc
    require(np.isfinite(inverse).all(), f"{role}: invalid inverse rest pose")
    frames = None
    if "scene_frames" in obj:
        frames = numeric(obj["scene_frames"], f"{role}.scene_frames")
        require(frames.dtype.kind in "iu" and frames.shape == (poses.shape[0],),
                f"{role}: invalid scene frame axis")
    return bones, poses, rest, frames


def mesh(obj, *, n_poses, expected_bones, role, exact_order=False):
    bones = names(obj, role)
    require(set(bones) == set(expected_bones), f"{role}: mesh bone set mismatch")
    if exact_order:
        require(bones == expected_bones, f"{role}: output bone order mismatch")

    arrays = {}
    for key in [
        "vertices", "faces", "vertex_normals", "vertex_bone_weights",
        "baked_animation_vertices", "baked_animation_vertex_normals",
    ]:
        require(key in obj, f"{role}: required field missing: {key}")
        arrays[key] = numeric(obj[key], f"{role}.{key}")
    v, f = arrays["vertices"], arrays["faces"]
    require(v.ndim == 2 and v.shape[0] > 0 and v.shape[1] == 3,
            f"{role}: invalid vertex shape")
    require(f.dtype.kind in "iu" and f.ndim == 2 and f.shape[0] > 0
            and f.shape[1] in (3, 4), f"{role}: faces must be integral triangles/quads")
    require(np.all((f >= 0) & (f < len(v))), f"{role}: face index out of bounds")
    require(arrays["vertex_normals"].shape == v.shape, f"{role}: normal shape mismatch")
    for key in ["baked_animation_vertices", "baked_animation_vertex_normals"]:
        require(arrays[key].shape == (n_poses, len(v), 3), f"{role}: invalid {key} shape")
    weights = arrays["vertex_bone_weights"]
    require(weights.shape == (len(v), len(bones)), f"{role}: weight shape mismatch")
    require(np.all(weights >= 0), f"{role}: negative weights")
    sums = weights.sum(axis=1)
    require(np.all(sums > 0), f"{role}: zero weight rows")
    require(np.allclose(sums, 1.0, atol=ATOL, rtol=0),
            f"{role}: weight rows are not normalized")

    frames = None
    if "scene_frames" in obj:
        frames = numeric(obj["scene_frames"], f"{role}.scene_frames")
        require(frames.dtype.kind in "iu" and frames.shape == (n_poses,),
                f"{role}: invalid scene_frames")
    if "vertex_bone_weights_mask" in obj:
        mask = np.asarray(obj["vertex_bone_weights_mask"])
        if mask.size:
            require(mask.dtype.kind in "biuf" and np.isfinite(mask).all()
                    and mask.shape == weights.shape, f"{role}: weight-mask axis mismatch")

    # Avoid legacy weak_validate mutation. This call cannot drop fields.
    try:
        validate_fitbox_mesh(
            obj, n_poses=n_poses, n_bones=len(bones),
            name=role, weak_validate=False,
        )
    except Exception as exc:
        raise StageValidationError(f"{role}: {exc}") from exc
    return frames


def same_array(left, right):
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except Exception:
        return left == right


def validate_step02_inputs(
    *, full_skeleton, full_animation, reduced_skeleton, reduced_animation, meshes,
):
    full = skeleton(full_skeleton, "full_skeleton")
    reduced = skeleton(reduced_skeleton, "reduced_skeleton")
    full_names, poses, _, frames = animation(
        full_animation, full["bone_names"], "full_animation"
    )
    reduced_names, reduced_poses, _, reduced_frames = animation(
        reduced_animation, reduced["bone_names"], "reduced_animation"
    )
    require(set(reduced_names) <= set(full_names), "Reduced bones are not a full-skeleton subset")
    require(reduced_poses.shape[0] == poses.shape[0], "Full/reduced frame count mismatch")
    if frames is not None and reduced_frames is not None:
        require(same_array(frames, reduced_frames), "Full/reduced scene frames differ")

    require(isinstance(meshes, dict) and meshes, "Expected nonempty mesh mapping")
    for key, obj in meshes.items():
        require(isinstance(key, str) and key.endswith(".pkl")
                and "/" not in key and "\\" not in key, "Mesh mapping requires flat PKL names")
        mesh_frames = mesh(
            obj, n_poses=len(poses), expected_bones=full_names,
            role=f"input_mesh:{key}",
        )
        if frames is not None and mesh_frames is not None:
            require(same_array(frames, mesh_frames), f"{key}: scene frames differ from animation")

    return {
        "schema": "ogd_step02_input_validation_v1",
        "status": "VALID",
        "frames": len(poses),
        "full_bone_names": full_names,
        "reduced_bone_names": reduced_names,
        "mesh_files": sorted(meshes),
        "scene_frames_present": frames is not None,
        "repair_performed": False,
    }


def validate_step02_outputs(
    *, full_skeleton, full_animation, reduced_skeleton, reduced_animation,
    input_meshes, output_skeleton, output_animation, output_meshes,
    summary, expected_parameters, expected_variant,
):
    checked = validate_step02_inputs(
        full_skeleton=full_skeleton, full_animation=full_animation,
        reduced_skeleton=reduced_skeleton, reduced_animation=reduced_animation,
        meshes=input_meshes,
    )
    expected_names = checked["reduced_bone_names"]
    expected_skeleton = align_skeleton_hierarchy_to_bone_names(
        skeleton(reduced_skeleton, "reduced"), expected_names
    )
    out_skeleton = skeleton(output_skeleton, "output_skeleton")
    require(out_skeleton["bone_names"] == expected_skeleton["bone_names"]
            and out_skeleton["parent_indices"] == expected_skeleton["parent_indices"],
            "Output skeleton differs from selected reduced skeleton")
    out_names, out_poses, out_rest, out_frames = animation(
        output_animation, expected_names, "output_animation"
    )
    require(out_names == expected_names, "Output animation bone order differs")
    full_names = checked["full_bone_names"]
    indices = [full_names.index(name) for name in expected_names]
    expected_poses = np.asarray(full_animation["animation_poses"])[:, indices, :, :]
    require(same_array(out_poses, expected_poses), "Output animation is not the expected full-animation subset")
    require(same_array(out_rest, reduced_animation["rest_pose"]),
            "Output rest pose differs from the selected reduced animation")
    expected_frames = full_animation.get("scene_frames", reduced_animation.get("scene_frames"))
    if expected_frames is not None:
        require(out_frames is not None and same_array(out_frames, expected_frames),
                "Output animation scene frames differ")
    else:
        require(out_frames is None, "Unexpected synthesized scene frame labels")

    require(isinstance(output_meshes, dict) and set(output_meshes) == set(input_meshes),
            "Output mesh file set differs from expected set")
    for key, obj in output_meshes.items():
        mesh(obj, n_poses=checked["frames"], expected_bones=expected_names,
             role=f"output_mesh:{key}", exact_order=True)
        original = input_meshes[key]
        for field in GEOMETRY_KEYS:
            require((field in original) == (field in obj),
                    f"{key}: field presence changed: {field}")
            if field in original:
                require(same_array(original[field], obj[field]),
                        f"{key}: preserved field changed: {field}")

    require(isinstance(summary, dict)
            and summary.get("schema") == "ogd_lbs_ssdr_step02_reduce_bone_weight_v2",
            "Unsupported Step02 summary")
    weights = summary.get("weights") or {}
    for key, actual_name in [
        ("alpha", "par_weight"), ("beta", "bone_weight"),
        ("gamma", "child_weight"), ("affinity_eps", "affinity_eps"),
    ]:
        require(type(weights.get(actual_name)) in (int, float)
                and weights[actual_name] == expected_parameters[key],
                f"Summary parameter mismatch: {key}")
    layout = summary.get("layout") or {}
    require(layout.get("effective_variant_name") == expected_variant,
            "Actual variant differs from expected variant")
    require(layout.get("extra_remove_bones_enabled") is False,
            "Unexpected extra bone removals")
    require(type(summary.get("reduced_mesh_count")) is int
            and summary["reduced_mesh_count"] == len(input_meshes),
            "Reduced mesh count mismatch")
    require(type(summary.get("skipped_count")) is int
            and summary["skipped_count"] == 0, "Unexpected skipped mesh candidates")
    require(type(summary.get("total_invalid_affinity_count")) is int
            and summary["total_invalid_affinity_count"] == 0,
            "Invalid affinity rows were reported")
    files = summary.get("files")
    require(isinstance(files, list) and len(files) == len(input_meshes)
            and all(isinstance(x, dict) and x.get("operation") == "reduce_mesh_weights"
                    for x in files), "Unexpected mesh processing records")

    return {
        "schema": "ogd_step02_output_validation_v1",
        "status": "VALID",
        "frames": checked["frames"],
        "bones": len(expected_names),
        "mesh_files": sorted(output_meshes),
        "variant": expected_variant,
        "repair_performed": False,
        "validation_scope": "loaded object invariants and Step02 summary consistency",
        "artifact_hash_and_provenance_verification": "caller_required",
        "ingest_performed": False,
    }
