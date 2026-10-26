from __future__ import annotations

import numpy as np

# 修正内部导入路径
from ogd_lbs.utils.common import warn
from ogd_lbs.fitbox.schema import KEY_VERTEX_BONE_WEIGHTS_MASK


def build_weight_mask(mesh_obj: dict, vbw: np.ndarray, use_weight_mask: bool, name: str) -> np.ndarray:
    """
    构建并安全检查权重遮罩
    """
    # 1. 如果未启用遮罩，返回全 1 矩阵
    if not use_weight_mask:
        return np.ones_like(vbw, dtype=np.float64)

    # 2. 启用遮罩，尝试读取
    mask = mesh_obj.get(KEY_VERTEX_BONE_WEIGHTS_MASK, None)
    if mask is None:
        mask = np.ones_like(vbw, dtype=np.float64)
    else:
        mask = np.array(mask, dtype=np.float64)

    # 3. 检查是否全为 1
    if np.all(mask == 1.0):
        warn(f"[{name}] Input mask is all 1s. Auto-generating mask based on initial non-zero weights.")
        mask = (vbw > 0).astype(np.float64)

    # 4. 安全机制：检查是否有顶点的 mask 全为 0
    row_sums = mask.sum(axis=1)
    zero_mask_rows = np.where(row_sums == 0)[0]
    
    if len(zero_mask_rows) > 0:
        warn(f"[{name}] Found {len(zero_mask_rows)} vertices with all-zero masks. Applying fallback.")
        for v_id in zero_mask_rows:
            # Fallback 1: 尝试使用初始权重大于 0 的骨骼
            fallback_row = (vbw[v_id] > 0).astype(np.float64)
            if fallback_row.sum() == 0:
                # Fallback 2: 如果初始权重也全为 0，只能 fallback 到全 1
                warn(f"[{name}] Vertex {v_id} has all-zero initial weights too. Falling back to all 1s.")
                fallback_row = np.ones_like(fallback_row)
            mask[v_id] = fallback_row

    return mask