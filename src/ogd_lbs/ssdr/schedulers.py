from __future__ import annotations

from typing import Tuple


def get_legacy_schedule(
    it: int,
    max_iter: int,
    org_type: int,
    base_opt_weight: bool
) -> Tuple[bool, int, float]:
    """
    根据当前迭代轮次和 org_type 调度器预设，返回本轮的优化参数。
    Returns:
        to_opt_weight (bool): 本轮是否优化权重
        weight_calc_mode (int): 权重计算模式 (0 或 1)
        jnt_factor (float): 关节约束系数
    """
    jnt_factor = 0.0
    weight_calc_mode = 0
    to_opt_weight = base_opt_weight

    if org_type == 1:
        jnt_factor = min(1.0 * (1.5 ** it), 10.0)
        
    elif org_type in (2, 3):
        half_iter = int(max(max_iter - 10, 0) / 2)

        if it < half_iter:
            to_opt_weight = False
            jnt_factor = 1.0 * (0.666 ** it)
            
        elif it < 2 * half_iter:
            to_opt_weight = True
            weight_calc_mode = 1 if org_type == 2 else 0
            jnt_factor = 1.0 * (0.666 ** (2 * half_iter - (it + 1)))
            
        else:
            to_opt_weight = True
            weight_calc_mode = 0
            jnt_factor = min((1.0 * 0.666 ** (2 * half_iter - (it + 1))), 10.0)

    return to_opt_weight, weight_calc_mode, jnt_factor