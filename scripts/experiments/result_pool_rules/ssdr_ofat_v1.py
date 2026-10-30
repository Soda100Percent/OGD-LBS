# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _set_none(out: dict[str, Any], changed: dict[str, Any], key: str) -> None:
    old = out.get(key)
    if old is not None:
        out[key] = None
        changed[key] = {"old": old, "new": None}


def normalize_disabled_weight_mask(dims: dict[str, Any], *, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    If weight mask is disabled, tau/mask-mode/mask-threshold do not affect the run.
    """
    out = dict(dims)
    changed: dict[str, Any] = {}

    if not _as_bool(out.get("step04_use_weight_mask")):
        for key in ["tau", "step04_mask_mode", "step04_mask_threshold"]:
            _set_none(out, changed, key)

    return out, {"rule": "normalize_disabled_weight_mask", "changed": changed}


def normalize_adaptive_tau(dims: dict[str, Any], *, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    If tau/mask mode is adaptive, the concrete numeric threshold is irrelevant.
    """
    out = dict(dims)
    changed: dict[str, Any] = {}

    tau_text = str(out.get("tau", "")).strip().lower()
    mode_text = str(out.get("step04_mask_mode", "")).strip().lower()

    if tau_text == "adaptive" or mode_text == "adaptive":
        old_tau = out.get("tau")
        if old_tau != "adaptive":
            out["tau"] = "adaptive"
            changed["tau"] = {"old": old_tau, "new": "adaptive"}

        old_mode = out.get("step04_mask_mode")
        if old_mode != "adaptive":
            out["step04_mask_mode"] = "adaptive"
            changed["step04_mask_mode"] = {"old": old_mode, "new": "adaptive"}

        _set_none(out, changed, "step04_mask_threshold")

    return out, {"rule": "normalize_adaptive_tau", "changed": changed}


def normalize_disabled_weight_smoothness(dims: dict[str, Any], *, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    If weight smoothness is disabled, smoothness strength is irrelevant.
    """
    out = dict(dims)
    changed: dict[str, Any] = {}

    if not _as_bool(out.get("step04_use_weight_smoothness")):
        _set_none(out, changed, "step04_weights_smooth_step")

    return out, {"rule": "normalize_disabled_weight_smoothness", "changed": changed}


def normalize_zero_trans_affine(dims: dict[str, Any], *, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    If trans_affine is zero, trans_affine_norm is irrelevant.
    """
    out = dict(dims)
    changed: dict[str, Any] = {}

    if _as_float(out.get("step04_trans_affine"), 0.0) == 0.0:
        _set_none(out, changed, "step04_trans_affine_norm")

    return out, {"rule": "normalize_zero_trans_affine", "changed": changed}


def normalize_zero_roll_affinity(dims: dict[str, Any], *, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    If roll_affinity is zero, roll_affinity_norm is irrelevant.
    """
    out = dict(dims)
    changed: dict[str, Any] = {}

    if _as_float(out.get("step04_roll_affinity"), 0.0) == 0.0:
        _set_none(out, changed, "step04_roll_affinity_norm")

    return out, {"rule": "normalize_zero_roll_affinity", "changed": changed}
