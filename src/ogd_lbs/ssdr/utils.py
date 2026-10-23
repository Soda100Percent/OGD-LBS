from __future__ import annotations

import numpy as np

from ogd_lbs.utils.common import fail


def ensure_finite_array(arr: np.ndarray, name: str) -> None:
    """确保数组中没有 NaN 或 Inf"""
    if not np.isfinite(arr).all():
        fail(f"[{name}] Array contains NaN or Inf values. Optimization cannot proceed.")


def validate_combined_arrays(
    v_cb: np.ndarray, 
    ba_v_cb: np.ndarray, 
    vbw_cb: np.ndarray, 
    bt: np.ndarray, 
    rbt: np.ndarray
) -> None:
    """验证拼接后的全局大型矩阵，防止越界"""
    ensure_finite_array(v_cb, "v_cb")
    ensure_finite_array(ba_v_cb, "ba_v_cb")
    ensure_finite_array(vbw_cb, "vbw_cb")
    ensure_finite_array(bt, "bt")
    ensure_finite_array(rbt, "rbt")
    
    v_count = v_cb.shape[0]
    n_p = bt.shape[0]
    n_b = rbt.shape[0]
    
    if ba_v_cb.shape != (n_p, v_count, 3):
        fail(f"Combined ba_v_cb shape mismatch: expected ({n_p}, {v_count}, 3), got {ba_v_cb.shape}")
    if vbw_cb.shape != (v_count, n_b):
        fail(f"Combined vbw_cb shape mismatch: expected ({v_count}, {n_b}), got {vbw_cb.shape}")