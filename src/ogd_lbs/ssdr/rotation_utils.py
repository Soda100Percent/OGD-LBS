from __future__ import annotations
import numpy as np

def normalize_vec(v: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, bool]:
    """归一化向量，并返回是否有效的布尔值"""
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v), False
    return v / norm, True

def _get_arbitrary_orthogonal(v: np.ndarray) -> np.ndarray:
    """获取一个与 v 垂直的任意单位向量"""
    axis = np.cross(v, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    if np.linalg.norm(axis) < 1e-6:
        axis = np.cross(v, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    axis, _ = normalize_vec(axis)
    return axis

def shortest_arc_rotation(a: np.ndarray, b: np.ndarray, fallback_axis: np.ndarray | None = None, eps: float = 1e-4) -> np.ndarray:
    """
    计算将向量 a 旋转到向量 b 的最短弧旋转矩阵 R，使得 R @ a ≈ b。
    包含对平行 (0度) 和反向平行 (180度奇点) 的绝对安全处理。
    """
    a_norm, a_valid = normalize_vec(a)
    b_norm, b_valid = normalize_vec(b)
    
    if not a_valid or not b_valid:
        return np.eye(3, dtype=np.float64)
        
    dot_val = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    
    # 情况 1: 几乎平行 (0度)
    if dot_val > 1.0 - eps:
        return np.eye(3, dtype=np.float64)
        
    # 情况 2: 几乎反向平行 (180度奇点)
    if dot_val < -1.0 + eps:
        # 使用 fallback_axis 寻找一个正交的旋转轴
        if fallback_axis is not None:
            # 将 fallback_axis 投影到 a 的法平面上，确保严格正交
            axis = fallback_axis - np.dot(fallback_axis, a_norm) * a_norm
            axis, valid = normalize_vec(axis)
            if not valid:
                axis = _get_arbitrary_orthogonal(a_norm)
        else:
            axis = _get_arbitrary_orthogonal(a_norm)
                
        # 绕 axis 旋转 180 度的旋转矩阵: R = 2 * outer(axis, axis) - I
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
        
    # 情况 3: 正常的最短弧旋转 (罗德里格斯公式变体)
    cross_ab = np.cross(a_norm, b_norm)
    s = np.linalg.norm(cross_ab)
    c = dot_val
    
    vx = np.array([
        [0, -cross_ab[2], cross_ab[1]],
        [cross_ab[2], 0, -cross_ab[0]],
        [-cross_ab[1], cross_ab[0], 0]
    ], dtype=np.float64)
    
    R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return R

def rot_y(phi: float) -> np.ndarray:
    """生成绕 Y 轴旋转 phi 弧度的旋转矩阵"""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ], dtype=np.float64)

def extract_twist_y_angle(H: np.ndarray) -> float:
    """
    将任意 3x3 矩阵 H 投影到最近的 Y 轴纯扭转 (Twist-Y)，并提取其旋转角。
    比直接使用 atan2(H[0,2], H[0,0]) 更稳健。
    """
    return float(np.arctan2(H[0, 2] - H[2, 0], H[0, 0] + H[2, 2]))

def wrap_angle_pi(x: float) -> float:
    """将角度限制在 [-pi, pi] 之间"""
    return float((x + np.pi) % (2 * np.pi) - np.pi)

def solve_rotation_from_covariance(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """通过 SVD 从协方差矩阵求解最佳旋转矩阵，包含反射修复"""
    if not np.isfinite(M).all() or np.linalg.norm(M) < eps:
        return np.eye(3, dtype=np.float64)
    try:
        u, _, vt = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)
        
    R = (u @ vt).T
    if np.linalg.det(R) < 0:
        vt[2, :] *= -1.0
        R = (u @ vt).T
    return R