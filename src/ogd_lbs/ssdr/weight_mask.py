from __future__ import annotations

import numpy as np

from ogd_lbs.utils.common import warn
from ogd_lbs.fitbox.schema import KEY_VERTEX_BONE_WEIGHTS_MASK


def _normalize_mask_mode(mask_mode: str) -> str:
    mode = str(mask_mode or "adaptive").strip().lower()
    if mode not in {"adaptive", "fixed"}:
        raise ValueError(
            f"Unsupported mask_mode={mask_mode!r}. Expected 'adaptive' or 'fixed'."
        )
    return mode


def _validate_mask_shape(mask: np.ndarray, vbw: np.ndarray, name: str) -> None:
    if mask.shape != vbw.shape:
        raise ValueError(
            f"[{name}] vertex_bone_weights_mask shape mismatch: "
            f"mask={mask.shape}, weights={vbw.shape}"
        )


def _repair_zero_mask_rows(mask: np.ndarray, vbw: np.ndarray, name: str) -> np.ndarray:
    """
    Ensure every vertex has at least one allowed bone in the mask.
    """
    row_sums = mask.sum(axis=1)
    zero_mask_rows = np.where(row_sums == 0)[0]

    if len(zero_mask_rows) > 0:
        warn(f"[{name}] Found {len(zero_mask_rows)} vertices with all-zero masks. Applying fallback.")

        for v_id in zero_mask_rows:
            # Fallback 1: use initially non-zero weights.
            fallback_row = (vbw[v_id] > 0).astype(np.float64)

            if fallback_row.sum() == 0:
                # Fallback 2: allow all bones.
                warn(
                    f"[{name}] Vertex {int(v_id)} has all-zero initial weights too. "
                    "Falling back to all 1s."
                )
                fallback_row = np.ones_like(fallback_row, dtype=np.float64)

            mask[v_id] = fallback_row

    return mask


def build_weight_mask(
    mesh_obj: dict,
    vbw: np.ndarray,
    use_weight_mask: bool,
    name: str,
    *,
    mask_mode: str = "adaptive",
    mask_threshold: float = -1.0,
) -> np.ndarray:
    """
    Build and validate a vertex-bone weight mask.

    Args:
        mesh_obj:
            Fitbox mesh dictionary.
        vbw:
            Current vertex-bone weights, shape (V, B).
        use_weight_mask:
            If False, returns an all-ones mask.
        name:
            Mesh name used in warnings.
        mask_mode:
            "adaptive":
                Backward-compatible behavior. Use mesh-provided
                vertex_bone_weights_mask when available. If absent or all-ones,
                auto-generate mask from initial non-zero weights: vbw > 0.
            "fixed":
                Ignore mesh-provided mask and generate mask from a fixed
                threshold: vbw > mask_threshold.
        mask_threshold:
            Fixed threshold for mask_mode="fixed". Use -1.0 for adaptive mode.

    Returns:
        Float mask with shape equal to vbw.
    """
    vbw = np.asarray(vbw, dtype=np.float64)

    if vbw.ndim != 2:
        raise ValueError(f"[{name}] vbw must be 2D, got {vbw.shape}")

    if not use_weight_mask:
        return np.ones_like(vbw, dtype=np.float64)

    mode = _normalize_mask_mode(mask_mode)

    if mode == "fixed":
        threshold = float(mask_threshold)
        if threshold < 0.0:
            raise ValueError(
                f"[{name}] mask_mode='fixed' requires mask_threshold >= 0, "
                f"got {mask_threshold}"
            )

        mask = (vbw > threshold).astype(np.float64)
        return _repair_zero_mask_rows(mask, vbw, name)

    # Adaptive mode: preserve previous behavior.
    mask = mesh_obj.get(KEY_VERTEX_BONE_WEIGHTS_MASK, None)

    if mask is None:
        mask = np.ones_like(vbw, dtype=np.float64)
    else:
        mask = np.asarray(mask, dtype=np.float64)
        _validate_mask_shape(mask, vbw, name)

    if np.all(mask == 1.0):
        warn(
            f"[{name}] Input mask is all 1s. "
            "Auto-generating mask based on initial non-zero weights."
        )
        mask = (vbw > 0).astype(np.float64)

    return _repair_zero_mask_rows(mask, vbw, name)
