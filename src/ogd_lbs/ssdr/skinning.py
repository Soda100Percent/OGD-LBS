from __future__ import annotations

import numpy as np


def skinning_t(v: np.ndarray, bt: np.ndarray, inv_rbt: np.ndarray, p_id: int) -> np.ndarray:
    """向量化版本的 skinning_t"""
    n_v = v.shape[0]
    v_homo = np.concatenate((v, np.ones((n_v, 1), dtype=np.float64)), axis=1)
    
    # bt[p_id] 形状: (n_b, 4, 4), inv_rbt 形状: (n_b, 4, 4)
    M = bt[p_id] @ inv_rbt  # 批量矩阵乘法，结果形状: (n_b, 4, 4)
    constructed = M @ v_homo.T  # (n_b, 4, 4) @ (4, n_v) -> (n_b, 4, n_v)
    
    return constructed[:, :3, :].transpose(0, 2, 1)


def skinning_b(v: np.ndarray, inv_rbt: np.ndarray, b_id: int) -> np.ndarray:
    n_v = v.shape[0]
    v_homo = np.concatenate((v, np.ones((n_v, 1), dtype=np.float64)), axis=1)
    w2rb = inv_rbt[b_id]
    return (w2rb @ v_homo.T).T[:, :3]


def skinning_t_single(u: np.ndarray, bt: np.ndarray, inv_rbt: np.ndarray, b_id: int, p_id: int) -> np.ndarray:
    u_homo = np.concatenate((u, [1.0]))
    b2w = bt[p_id, b_id]
    w2rb = inv_rbt[b_id]
    return (b2w @ w2rb @ u_homo)[:3]


def skinning_w(constructed: np.ndarray, vbw: np.ndarray) -> np.ndarray:
    return constructed * vbw.transpose((1, 0))[:, :, np.newaxis]


def skinning_s(constructed: np.ndarray) -> np.ndarray:
    return np.sum(constructed, axis=0)