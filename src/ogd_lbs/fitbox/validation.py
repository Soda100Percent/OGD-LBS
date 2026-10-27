# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Fitbox validation helpers.

This module contains pure Python / NumPy validators used by SSDR preprocessing
and later optimization stages. It intentionally has no bpy dependency.
"""

from typing import Any

import numpy as np

from ogd_lbs.utils.common import fail, warn
from ogd_lbs.fitbox.schema import (
    KEY_ANIMATION_POSES,
    KEY_BAKED_VERTICES,
    KEY_BONE_NAMES,
    KEY_FACE_MATERIALS,
    KEY_FACE_UVS,
    KEY_FACES,
    KEY_VERTICES,
    KEY_VERTEX_BONE_WEIGHTS,
)


def _as_array(value: Any, *, name: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except Exception as exc:
        fail(f"Could not convert {name} to numpy array: {exc}")
        raise RuntimeError("unreachable")


def validate_skeleton_and_animation(anim_obj: dict, skel_obj: dict) -> tuple[int, int]:
    """
    Validate consistency between animation_data and skeleton_hierarchy_data.

    Returns:
        (n_poses, n_bones)
    """
    anim_bones = anim_obj.get(KEY_BONE_NAMES, [])
    skel_bones = skel_obj.get(KEY_BONE_NAMES, [])

    if not anim_bones or not skel_bones:
        fail("Missing bone_names in animation or skeleton data.")

    anim_bones = [str(x) for x in anim_bones]
    skel_bones = [str(x) for x in skel_bones]

    if len(anim_bones) != len(skel_bones):
        fail(
            f"Bone count mismatch: animation has {len(anim_bones)}, "
            f"skeleton has {len(skel_bones)}."
        )

    if anim_bones != skel_bones:
        fail("Bone names order or content mismatch between animation and skeleton data.")

    poses = _as_array(anim_obj.get(KEY_ANIMATION_POSES, []), name=KEY_ANIMATION_POSES)
    n_poses = int(poses.shape[0]) if poses.ndim >= 1 else 0
    n_bones = len(anim_bones)

    if n_poses > 0:
        if poses.ndim != 4 or poses.shape[1:] != (n_bones, 4, 4):
            fail(
                f"Animation poses shape mismatch: expected "
                f"(frames, {n_bones}, 4, 4), got {poses.shape}."
            )

    return n_poses, n_bones


def validate_fitbox_mesh(
    mesh_obj: dict,
    n_poses: int | None = None,
    n_bones: int | None = None,
    name: str = "Mesh",
    weak_validate: bool = False,
) -> None:
    """
    Validate one Fitbox skinned mesh object.

    When weak_validate=True, non-fatal topology sidecar mismatches such as
    face_uvs / face_materials length mismatch are downgraded to warnings and
    the dirty fields are removed from mesh_obj.
    """
    v = _as_array(mesh_obj.get(KEY_VERTICES, []), name=f"{name}.{KEY_VERTICES}")
    f = _as_array(mesh_obj.get(KEY_FACES, []), name=f"{name}.{KEY_FACES}")
    uvs = _as_array(mesh_obj.get(KEY_FACE_UVS, []), name=f"{name}.{KEY_FACE_UVS}")
    face_mats = _as_array(mesh_obj.get(KEY_FACE_MATERIALS, []), name=f"{name}.{KEY_FACE_MATERIALS}")
    vbw = _as_array(mesh_obj.get(KEY_VERTEX_BONE_WEIGHTS, []), name=f"{name}.{KEY_VERTEX_BONE_WEIGHTS}")
    bkd_v = _as_array(mesh_obj.get(KEY_BAKED_VERTICES, []), name=f"{name}.{KEY_BAKED_VERTICES}")

    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] <= 0:
        fail(f"[{name}] vertices must have shape (V, 3), got {v.shape}.")

    if f.ndim != 2 or f.shape[0] <= 0 or f.shape[1] not in (3, 4):
        fail(f"[{name}] faces must have shape (F, 3) or (F, 4), got {f.shape}.")

    if not np.issubdtype(f.dtype, np.integer):
        f = f.astype(np.int64)

    if np.any(f < 0) or np.any(f >= v.shape[0]):
        fail(f"[{name}] face indices out of range for vertex count {v.shape[0]}.")

    if len(uvs) > 0 and len(f) != len(uvs):
        msg = f"[{name}] faces count ({len(f)}) != face_uvs count ({len(uvs)})"
        if weak_validate:
            warn(msg + " -> dropped face_uvs due to weak_validate.")
            mesh_obj.pop(KEY_FACE_UVS, None)
        else:
            fail(msg)

    if len(face_mats) > 0 and len(f) != len(face_mats):
        msg = f"[{name}] faces count ({len(f)}) != face_materials count ({len(face_mats)})"
        if weak_validate:
            warn(msg + " -> dropped face_materials due to weak_validate.")
            mesh_obj.pop(KEY_FACE_MATERIALS, None)
        else:
            fail(msg)

    if vbw.size > 0:
        if vbw.ndim != 2:
            fail(f"[{name}] vertex_bone_weights must be 2D, got {vbw.shape}.")

        if v.shape[0] != vbw.shape[0]:
            fail(
                f"[{name}] vertices count ({v.shape[0]}) != "
                f"vertex_bone_weights rows ({vbw.shape[0]})."
            )

        row_sums = vbw.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            bad_count = int(np.count_nonzero(np.abs(row_sums - 1.0) > 1e-5))
            fail(f"[{name}] vertex_bone_weights do not sum to 1.0; bad rows={bad_count}.")

        if n_bones is not None and vbw.shape[1] != int(n_bones):
            fail(
                f"[{name}] vertex_bone_weights columns ({vbw.shape[1]}) "
                f"!= expected bone count ({n_bones})."
            )

    if n_poses is not None and bkd_v.size > 0:
        if bkd_v.ndim != 3 or bkd_v.shape[2] != 3:
            fail(f"[{name}] baked_animation_vertices must have shape (P, V, 3), got {bkd_v.shape}.")

        if bkd_v.shape[0] != int(n_poses):
            fail(
                f"[{name}] baked_animation_vertices frames ({bkd_v.shape[0]}) "
                f"!= expected poses count ({n_poses})."
            )

        if bkd_v.shape[1] != v.shape[0]:
            fail(
                f"[{name}] baked_animation_vertices vertex axis ({bkd_v.shape[1]}) "
                f"!= vertex count ({v.shape[0]})."
            )

    for arr_name, arr in [
        (KEY_VERTICES, v),
        (KEY_FACES, f),
        (KEY_VERTEX_BONE_WEIGHTS, vbw),
        (KEY_BAKED_VERTICES, bkd_v),
    ]:
        if arr.size > 0 and not np.isfinite(arr).all():
            fail(f"[{name}] NaN or Inf found in {arr_name}.")
