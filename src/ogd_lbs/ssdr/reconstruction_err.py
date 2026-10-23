from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ogd_lbs.ssdr.skinning import skinning_t, skinning_w, skinning_s

def calc_hausdorff_directed(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the directed Hausdorff distance between two N-D point sets.
    """
    distances = cdist(A, B)
    return float(np.max(np.min(distances, axis=1)))

def calc_hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the Hausdorff distance between two N-D point sets.
    """
    return max(calc_hausdorff_directed(A, B), calc_hausdorff_directed(B, A))

def recon_err(
    v: np.ndarray, 
    p: np.ndarray, 
    bt: np.ndarray, 
    inv_rbt: np.ndarray, 
    vbw: np.ndarray, 
    f: np.ndarray | None = None
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

def recon_err_hausdorff(
    v: np.ndarray, 
    p: np.ndarray, 
    bt: np.ndarray, 
    inv_rbt: np.ndarray, 
    vbw: np.ndarray
) -> tuple[float, float]:
    """
    Compute Hausdorff distance frame by frame to avoid OOM.
    Returns: (mean_hausdorff, max_hausdorff) across all frames.
    """
    n_p = p.shape[0]
    hausdorff_dists = []
    
    for p_id in range(n_p):
        constructed = skinning_t(v, bt, inv_rbt, p_id)
        constructed = skinning_w(constructed, vbw)
        p_r = skinning_s(constructed)
        
        # Calculate Hausdorff for the current frame
        h_dist = calc_hausdorff(p[p_id], p_r)
        hausdorff_dists.append(h_dist)
        
    return float(np.mean(hausdorff_dists)), float(np.max(hausdorff_dists))

def recon_err_per_frame(
    v: np.ndarray, 
    p: np.ndarray, 
    bt: np.ndarray, 
    inv_rbt: np.ndarray, 
    vbw: np.ndarray
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
    # axis=2 是坐标(x,y,z)求norm, axis=1 是对所有顶点求平均，最终保留帧维度
    return np.mean(np.linalg.norm(p_e, axis=2), axis=1)