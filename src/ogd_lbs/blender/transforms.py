# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic Blender / NumPy transform helpers.

Asset-agnostic by design:
    - no dataset paths
    - no character names
    - no action names
    - no model-specific bone names
"""

from typing import Any
import math

import numpy as np
try:
    from mathutils import Matrix
except ModuleNotFoundError:
    Matrix = None  # type: ignore[assignment]


def to_numpy_matrix(value: Any, *, dtype=np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype).copy()


def to_blender_matrix(value: Any):
    if Matrix is None:
        raise RuntimeError(
            "mathutils.Matrix is not available. "
            "Run this function inside Blender Python, or install a compatible mathutils package."
        )
    return Matrix(np.asarray(value, dtype=np.float64).tolist())


def assert_affine_matrix(value: Any, *, label: str = "matrix", atol: float = 1e-6) -> None:
    arr = np.asarray(value, dtype=np.float64)

    if arr.shape[-2:] != (4, 4):
        raise RuntimeError(f"{label}: expected (..., 4, 4), got {arr.shape}.")

    if not np.isfinite(arr).all():
        raise RuntimeError(f"{label}: matrix contains NaN/Inf.")

    if not np.allclose(arr[..., 3, :], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise RuntimeError(f"{label}: non-affine homogeneous bottom row.")

    singular = np.linalg.svd(arr[..., :3, :3], compute_uv=False)

    if not np.all(singular[..., -1] > 1e-12):
        raise RuntimeError(f"{label}: singular or near-singular linear transform.")


def require_similarity_transform(value: Any, *, label: str = "matrix", tol: float = 1e-5) -> float:
    arr = np.asarray(value, dtype=np.float64)
    assert_affine_matrix(arr, label=label)

    linear = arr[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    det = float(np.linalg.det(linear))

    if det <= 0.0:
        raise RuntimeError(f"{label}: reflected or degenerate transform is unsupported.")

    mean = float(np.mean(singular))
    rel = float((np.max(singular) - np.min(singular)) / max(mean, 1e-12))

    if rel > float(tol):
        raise RuntimeError(
            f"{label}: non-uniform scale or shear is unsupported; "
            f"singular_values={singular.tolist()}, relative={rel}"
        )

    return mean


def closest_rotation(linear: Any) -> np.ndarray:
    arr = np.asarray(linear, dtype=np.float64)
    u, _s, vt = np.linalg.svd(arr)
    rot = u @ vt
    det = np.linalg.det(rot)

    if rot.ndim == 2:
        if det < 0:
            u = u.copy()
            u[:, -1] *= -1.0
            rot = u @ vt
        return rot

    mask = det < 0
    if np.any(mask):
        u = u.copy()
        u[mask, :, -1] *= -1.0
        rot = u @ vt

    return rot


def matrix_error_arrays(actual: Any, expected: Any) -> dict[str, np.ndarray]:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)

    if a.shape != b.shape:
        raise RuntimeError(f"Matrix comparison shape mismatch: {a.shape} vs {b.shape}")

    assert_affine_matrix(a, label="actual")
    assert_affine_matrix(b, label="expected")

    translation_l2 = np.linalg.norm(a[..., :3, 3] - b[..., :3, 3], axis=-1)

    linear_relative = (
        np.linalg.norm(a[..., :3, :3] - b[..., :3, :3], axis=(-2, -1))
        / np.maximum(np.linalg.norm(b[..., :3, :3], axis=(-2, -1)), 1e-12)
    )

    ra = closest_rotation(a[..., :3, :3])
    rb = closest_rotation(b[..., :3, :3])
    delta = np.swapaxes(ra, -1, -2) @ rb

    trace = np.trace(delta, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)

    skew = np.stack(
        [
            delta[..., 2, 1] - delta[..., 1, 2],
            delta[..., 0, 2] - delta[..., 2, 0],
            delta[..., 1, 0] - delta[..., 0, 1],
        ],
        axis=-1,
    )
    sin = 0.5 * np.linalg.norm(skew, axis=-1)
    rotation_deg = np.degrees(np.arctan2(sin, cos))

    matrix_abs = np.max(np.abs(a - b), axis=(-2, -1))

    return {
        "translation_l2": translation_l2,
        "rotation_deg": rotation_deg,
        "linear_relative": linear_relative,
        "matrix_abs": matrix_abs,
    }


def summarize_matrix_errors(
    errors: dict[str, np.ndarray],
    *,
    times: Any | None = None,
    names: list[str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key, values in errors.items():
        arr = np.asarray(values)
        idx = int(np.argmax(arr))
        unraveled = np.unravel_index(idx, arr.shape)

        item: dict[str, Any] = {
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

        if times is not None and arr.ndim >= 1:
            item["worst_time"] = float(times[unraveled[0]])

        if names is not None and arr.ndim >= 2:
            item["worst_name"] = str(names[unraveled[1]])

        out[key] = item
        out[f"{key}_max"] = item["max"]
        out[f"{key}_mean"] = item["mean"]

    return out


def is_error_within_tolerance(
    summary_or_maxima: dict[str, Any],
    *,
    position: float,
    rotation_deg: float,
    linear: float,
) -> bool:
    def read(name: str) -> float:
        if name in summary_or_maxima:
            value = summary_or_maxima[name]
            if isinstance(value, dict):
                return float(value.get("max", math.inf))
            return float(value)
        return float(summary_or_maxima.get(f"{name}_max", math.inf))

    return bool(
        read("translation_l2") <= float(position)
        and read("rotation_deg") <= float(rotation_deg)
        and read("linear_relative") <= float(linear)
    )


def make_reflection_matrix(
    *,
    signs: tuple[float, float, float] | None = None,
    axes: list[str] | tuple[str, ...] | str | None = None,
) -> np.ndarray:
    if signs is not None:
        if len(signs) != 3:
            raise RuntimeError("signs must contain exactly 3 values.")
        sx, sy, sz = [float(x) for x in signs]
        if abs(sx) != 1.0 or abs(sy) != 1.0 or abs(sz) != 1.0:
            raise RuntimeError("reflection signs must be +/-1.")
    else:
        if axes is None:
            axes = ["X"]
        if isinstance(axes, str):
            axes = [axes]
        axes_set = {str(a).upper() for a in axes}
        sx = -1.0 if "X" in axes_set else 1.0
        sy = -1.0 if "Y" in axes_set else 1.0
        sz = -1.0 if "Z" in axes_set else 1.0

    out = np.eye(4, dtype=np.float64)
    out[0, 0] = sx
    out[1, 1] = sy
    out[2, 2] = sz
    return out


def _unit(value: Any, *, label: str = "vector", eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if not np.isfinite(v).all():
        raise RuntimeError(f"{label}: nonfinite vector.")

    length = float(np.linalg.norm(v))
    if length <= eps:
        raise RuntimeError(f"{label}: degenerate vector.")

    return v / length


def _axis_to_index_sign(axis: str) -> tuple[int, float]:
    text = str(axis).strip().upper()

    if text in {"X", "+X"}:
        return 0, 1.0
    if text == "-X":
        return 0, -1.0
    if text in {"Y", "+Y"}:
        return 1, 1.0
    if text == "-Y":
        return 1, -1.0
    if text in {"Z", "+Z"}:
        return 2, 1.0
    if text == "-Z":
        return 2, -1.0

    raise RuntimeError(f"Unsupported axis: {axis!r}")


def _complete_right_handed_basis(columns: list[np.ndarray | None]) -> np.ndarray:
    missing = [idx for idx, value in enumerate(columns) if value is None]
    if len(missing) != 1:
        raise RuntimeError("Expected exactly one missing axis to complete basis.")

    miss = missing[0]
    x, y, z = columns

    if miss == 0:
        assert y is not None and z is not None
        x = _unit(np.cross(y, z), label="completed_x")
    elif miss == 1:
        assert x is not None and z is not None
        y = _unit(np.cross(z, x), label="completed_y")
    else:
        assert x is not None and y is not None
        z = _unit(np.cross(x, y), label="completed_z")

    basis = np.column_stack([x, y, z])

    if float(np.linalg.det(basis)) < 0.0:
        if miss == 0:
            x = -x
        elif miss == 1:
            y = -y
        else:
            z = -z
        basis = np.column_stack([x, y, z])

    det = float(np.linalg.det(basis))
    if abs(det - 1.0) > 1e-8:
        raise RuntimeError(f"Improper basis determinant: {det}")

    return basis


def aim_local_axis_at_target(
    base_world: Any,
    target_world_point: Any,
    *,
    axis: str = "-Y",
    preserve_axis: str = "+Z",
) -> tuple[np.ndarray, str]:
    """
    Keep origin and positive column scales, orient requested local axis toward
    target_world_point, and keep roll as close as possible to preserve_axis.

    Returns:
        (new_world_matrix, branch_name)
    """

    base = np.asarray(base_world, dtype=np.float64).copy()
    target = np.asarray(target_world_point, dtype=np.float64).reshape(3)

    assert_affine_matrix(base, label="base_world")

    linear = base[:3, :3]
    scales = np.linalg.norm(linear, axis=0)

    if not np.all(scales > 1e-12):
        raise RuntimeError("base_world: zero scale axis.")

    rotation = linear / scales[None, :]

    if float(np.linalg.det(rotation)) <= 0.0:
        raise RuntimeError("base_world: reflected baseline is unsupported.")

    if float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))) > 1e-4:
        raise RuntimeError("base_world: baseline contains shear; refusing approximation.")

    origin = base[:3, 3]
    direction = _unit(target - origin, label="aim_direction")

    axis_idx, axis_sign = _axis_to_index_sign(axis)
    preserve_idx, _ = _axis_to_index_sign(preserve_axis)

    target_column = axis_sign * direction

    columns: list[np.ndarray | None] = [None, None, None]
    columns[axis_idx] = target_column

    candidate_indices = [preserve_idx, 0, 1, 2]
    branch = "PROJECT_PRESERVE_AXIS"
    chosen_idx = None

    for idx in candidate_indices:
        if idx == axis_idx:
            continue
        raw = rotation[:, idx]
        projected = raw - np.dot(raw, target_column) * target_column
        if float(np.linalg.norm(projected)) >= 1e-4:
            columns[idx] = _unit(projected, label=f"projected_axis_{idx}")
            chosen_idx = idx
            if idx != preserve_idx:
                branch = "FALLBACK_PROJECT_OTHER_AXIS"
            break

    if chosen_idx is None:
        raise RuntimeError("Could not construct stable aim roll axis.")

    rot = _complete_right_handed_basis(columns)

    out = base.copy()
    out[:3, :3] = rot * scales[None, :]
    out[:3, 3] = origin
    out[3, :] = [0.0, 0.0, 0.0, 1.0]

    return out, branch


def self_test() -> None:
    ident = np.eye(4)[None, None]
    rot250 = np.eye(4)
    c = math.cos(math.radians(250))
    s = math.sin(math.radians(250))
    rot250[:3, :3] = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    e = matrix_error_arrays(rot250[None, None], ident)
    angle = float(e["rotation_deg"][0, 0])
    if abs(angle - 110.0) > 1e-3:
        raise RuntimeError(f"Shortest rotation self-test failed: {angle}")

    base = np.eye(4)
    base[:3, :3] *= 2.0
    aimed, _branch = aim_local_axis_at_target(base, [1.0, 2.0, 3.0], axis="-Y")
    direction = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    minus_y = -aimed[:3, 1] / np.linalg.norm(aimed[:3, 1])
    if not np.allclose(minus_y, direction, atol=1e-8):
        raise RuntimeError("Aim self-test failed.")

    if not np.allclose(aimed[:3, 3], base[:3, 3], atol=0.0):
        raise RuntimeError("Aim origin preservation self-test failed.")

    if not np.allclose(np.linalg.norm(aimed[:3, :3], axis=0), [2.0, 2.0, 2.0]):
        raise RuntimeError("Aim scale preservation self-test failed.")
