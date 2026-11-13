from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ogd_lbs.ssdr.skinning import skinning_s, skinning_t, skinning_w


def calc_hausdorff_directed(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the directed Hausdorff distance max_{a in A} min_{b in B} ||a - b||_2
    using cKDTree to avoid O(N^2) memory allocation on high-poly meshes.
    """
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    if A_arr.size == 0 or B_arr.size == 0:
        return 0.0
    tree_b = cKDTree(B_arr)
    dists, _ = tree_b.query(A_arr, k=1, workers=-1)
    return float(np.max(dists))


def calc_hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the symmetric Hausdorff distance between two N-D point sets.
    """
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    if A_arr.size == 0 or B_arr.size == 0:
        return 0.0
    tree_a = cKDTree(A_arr)
    tree_b = cKDTree(B_arr)
    d_ab, _ = tree_b.query(A_arr, k=1, workers=-1)
    d_ba, _ = tree_a.query(B_arr, k=1, workers=-1)
    return float(max(np.max(d_ab), np.max(d_ba)))


def recon_err(
    v: np.ndarray,
    p: np.ndarray,
    bt: np.ndarray,
    inv_rbt: np.ndarray,
    vbw: np.ndarray,
    f: np.ndarray | None = None,
) -> float:
    """
    Compute standard L2 reconstruction error (mean Euclidean distance).
    """
    n_b, n_v, n_p = inv_rbt.shape[0], vbw.shape[0], p.shape[0]
    p_r = np.zeros((n_p, n_v, 3), dtype=np.float64)

    for p_id in range(n_p):
        constructed = skinning_t(v, bt, inv_rbt, p_id)
        constructed = skinning_w(constructed, vbw)
        p_r[p_id] = skinning_s(constructed)

    p_e = p - p_r
    return float(np.mean(np.linalg.norm(p_e, axis=2)))


def recon_err_hausdorff_per_frame(
    v: np.ndarray,
    p: np.ndarray,
    bt: np.ndarray,
    inv_rbt: np.ndarray,
    vbw: np.ndarray,
) -> np.ndarray:
    """
    Compute Hausdorff distance per frame.
    Returns: 1D array of shape (n_p,)
    """
    n_p = p.shape[0]
    hausdorff_dists = np.zeros((n_p,), dtype=np.float64)

    for p_id in range(n_p):
        constructed = skinning_t(v, bt, inv_rbt, p_id)
        constructed = skinning_w(constructed, vbw)
        p_r = skinning_s(constructed)
        hausdorff_dists[p_id] = calc_hausdorff(p[p_id], p_r)

    return hausdorff_dists


def recon_err_hausdorff(
    v: np.ndarray,
    p: np.ndarray,
    bt: np.ndarray,
    inv_rbt: np.ndarray,
    vbw: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Hausdorff distance frame by frame.
    Returns: (mean_hausdorff, max_hausdorff) across all frames.
    """
    hausdorff_dists = recon_err_hausdorff_per_frame(v, p, bt, inv_rbt, vbw)
    return float(np.mean(hausdorff_dists)), float(np.max(hausdorff_dists))


def recon_err_per_frame(
    v: np.ndarray,
    p: np.ndarray,
    bt: np.ndarray,
    inv_rbt: np.ndarray,
    vbw: np.ndarray,
) -> np.ndarray:
    """
    Compute L2 reconstruction error per frame.
    Returns: 1D array of shape (n_p,)
    """
    n_b, n_v, n_p = inv_rbt.shape[0], vbw.shape[0], p.shape[0]
    p_r = np.zeros((n_p, n_v, 3), dtype=np.float64)

    for p_id in range(n_p):
        constructed = skinning_t(v, bt, inv_rbt, p_id)
        constructed = skinning_w(constructed, vbw)
        p_r[p_id] = skinning_s(constructed)

    p_e = p - p_r
    return np.mean(np.linalg.norm(p_e, axis=2), axis=1)
