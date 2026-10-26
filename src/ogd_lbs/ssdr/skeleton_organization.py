from __future__ import annotations

import numpy as np


def calc_bone_lens(bone_pose: np.ndarray, par_idxs: np.ndarray | list) -> np.ndarray:
    b_pos = np.array([(bone_pose[b_id] @ np.array([0, 0, 0, 1.0]))[:3] for b_id in range(bone_pose.shape[0])])
    return np.array([
        np.linalg.norm(b_pos[par_idxs[b_id] if par_idxs[b_id] is not None else b_id] - b_pos[b_id]) 
        for b_id in range(bone_pose.shape[0])
    ])


def calc_org_loss(orig_bone_lens: np.ndarray, bone_lens: np.ndarray) -> float:
    epsilon = 1e-10
    loss = np.array([
        abs((bone_len - orig_bone_len) / (orig_bone_len + epsilon)) 
        for orig_bone_len, bone_len in zip(orig_bone_lens.tolist(), bone_lens.tolist())
    ])
    return float(np.mean(loss))