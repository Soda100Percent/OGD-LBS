from __future__ import annotations

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# 修正内部导入路径
from ogd_lbs.ssdr.skinning import skinning_t_single, skinning_t, skinning_w, skinning_s, skinning_b
from ogd_lbs.ssdr.rotation_utils import shortest_arc_rotation, rot_y, solve_rotation_from_covariance, extract_twist_y_angle
from ogd_lbs.utils.common import log


def _calculate_q(v, bt, inv_rbt, vbw, p, b_id, p_id):
    constructed = skinning_t(v, bt, inv_rbt, p_id)
    weighted = skinning_w(constructed, vbw)
    
    total_skinned = skinning_s(weighted)
    skinned_pose_without_b = total_skinned - weighted[b_id]
    return p[p_id] - skinned_pose_without_b


def build_bone_bfs_order(par_idxs: np.ndarray) -> list[int]:
    """构建广度优先搜索 (BFS) 的骨骼更新顺序，确保父骨骼先于子骨骼更新"""
    n_b = len(par_idxs)
    children = [[] for _ in range(n_b)]
    roots = []
    for i, p in enumerate(par_idxs):
        if p is not None and p >= 0:
            children[p].append(i)
        else:
            roots.append(i)
            
    order = []
    queue = roots.copy()
    while queue:
        curr = queue.pop(0)
        order.append(curr)
        queue.extend(children[curr])
    return order


def calc_bt_parallel(
    v, p, bt_slice, inv_rbt, vbw, bj_l, fixed_bones, p_id, 
    v_b_cache, bone_jnts_local, trans_affine, trans_affine_norm, bt_covariance_mode,
    par_idxs, rest_roll_angles, rest_roll_valid, bone_update_order,
    roll_affinity, roll_affinity_norm
):
    n_b, n_v = vbw.shape[1], vbw.shape[0]
    bt_work = np.copy(bt_slice)
    p_slice = p[np.newaxis, p_id]
    epsilon = 1e-10

    q_time = 0.0
    svd_time = 0.0

    # 按照 BFS 顺序更新
    for b_id in bone_update_order:
        if b_id in fixed_bones:
            continue

        v_b = v_b_cache[b_id]
        
        t0 = time.perf_counter()
        q = _calculate_q(v, bt_work[np.newaxis, ...], inv_rbt, vbw, p_slice, b_id, 0)
        t1 = time.perf_counter()
        q_time += (t1 - t0)
        
        jnts, c = bone_jnts_local[b_id]
        n_jnts = len(jnts)
        psi = np.zeros((n_jnts, 3))

        for i, (nbr_b_id, jnt) in enumerate(jnts):
            psi[i] = skinning_t_single(jnt, bt_work[np.newaxis, ...], inv_rbt, nbr_b_id, 0)

        w_b = vbw[:, b_id]
        w_b_sqr = np.square(w_b)
        sum_sqr_w = np.sum(w_b_sqr) / n_v
        
        # ---------------------------------------------------------
        # 1. 计算平移亲和力质心 (Translation Affinity Centroids)
        # ---------------------------------------------------------
        alpha = 0.0
        u_aff = np.zeros(3)
        v_aff = np.zeros(3)
        
        if trans_affine > 0.0:
            W_p = np.power(w_b, trans_affine_norm)
            sum_W_p = np.sum(W_p)
            if sum_W_p > epsilon:
                alpha = trans_affine * sum_sqr_w
                u_aff = np.sum(W_p[:, np.newaxis] * v_b, axis=0) / sum_W_p
                v_aff = np.sum(W_p[:, np.newaxis] * p_slice[0], axis=0) / sum_W_p

        # ---------------------------------------------------------
        # 2. 计算全局加权旋转中心 (Global Weighted Centroids)
        # ---------------------------------------------------------
        W_total = sum_sqr_w + bj_l * n_jnts + alpha
        if W_total < epsilon:
            continue

        p_star = (np.sum(w_b_sqr[:, np.newaxis] * v_b, axis=0) / n_v + bj_l * np.sum(c, axis=0) + alpha * u_aff) / W_total
        q_star = (np.sum(w_b[:, np.newaxis] * q, axis=0) / n_v + bj_l * np.sum(psi, axis=0) + alpha * v_aff) / W_total

        # ---------------------------------------------------------
        # 3. 构造基础协方差矩阵 M_base
        # ---------------------------------------------------------
        p_bar = v_b - p_star
        q_bar = q - w_b[:, np.newaxis] * q_star[np.newaxis, :]
        c_bar = c - p_star
        psi_bar = psi - q_star
        
        if bt_covariance_mode == "legacy":
            M_data = (p_bar * w_b[:, np.newaxis]).T @ q_bar / (n_v * n_v)
            M_jnts = (bj_l * bj_l) * (c_bar.T @ psi_bar)
            M_base = M_data + M_jnts
            if alpha > 0.0:
                M_base += (alpha * alpha) * np.outer(u_aff - p_star, v_aff - q_star)
        else:
            M_data = (p_bar * w_b[:, np.newaxis]).T @ q_bar / n_v
            M_jnts = bj_l * (c_bar.T @ psi_bar)
            M_base = M_data + M_jnts
            if alpha > 0.0:
                M_base += alpha * np.outer(u_aff - p_star, v_aff - q_star)

        if not np.isfinite(M_base).all() or np.linalg.norm(M_base) < epsilon:
            continue

        # ---------------------------------------------------------
        # 4. 两阶段 SVD 与 Roll Prior 约束注入 (支持 p-norm)
        # ---------------------------------------------------------
        t2 = time.perf_counter()
        
        # 阶段一：初步估算当前数据项期望的旋转 R0
        R0 = solve_rotation_from_covariance(M_base)
        y_c_current = R0[:, 1]  # 当前期望的子骨骼主轴
        
        M_roll = np.zeros((3, 3), dtype=np.float64)
        
        has_parent = rest_roll_valid[b_id]
        if has_parent and roll_affinity > 0.0:
            parent_id = par_idxs[b_id]
            R_parent = bt_work[parent_id, :3, :3]
            y_p = R_parent[:, 1]
            
            # 最短弧对齐 (严格传入父骨骼 X 轴作为 180度奇点 Fallback)
            S = shortest_arc_rotation(y_p, y_c_current, fallback_axis=R_parent[:, 0])
            T_rest = rot_y(rest_roll_angles[b_id])
            
            # 构造理想先验：继承父骨骼 Roll + Rest 偏移，同时指向当前期望的 Swing
            R_prior = S @ R_parent @ T_rest
            
            # --- 动态权重计算 (IRLS for p-norm) ---
            if roll_affinity_norm == 2.0:
                dynamic_multiplier = 1.0
            else:
                # 提取初步解 R0 与理想先验 R_prior 之间的纯 Roll 偏差
                H_diff = R0.T @ R_prior
                theta = extract_twist_y_angle(H_diff)
                # 归一化到 [0, 1] 区间
                theta_norm = min(abs(theta) / np.pi, 1.0)
                # 计算动态乘子: (theta_norm) ^ (p - 2)
                dynamic_multiplier = np.power(theta_norm + 1e-8, roll_affinity_norm - 2.0)
            # --------------------------------------
            
            # 相对强度缩放
            base_norm = np.linalg.norm(M_base, ord='fro')
            gamma_eff = roll_affinity * dynamic_multiplier * max(base_norm, epsilon)
            
            # 只约束 X 和 Z 轴 (Roll)
            D_xz = np.diag([1.0, 0.0, 1.0])
            M_roll = gamma_eff * (D_xz @ R_prior.T)

        # 阶段二：终极 SVD 求解
        M_final = M_base + M_roll
        R_final = solve_rotation_from_covariance(M_final)
        
        svd_time += (time.perf_counter() - t2)

        t = q_star - R_final @ p_star

        if np.isfinite(R_final).all() and np.isfinite(t).all():
            bt_work[b_id] = np.eye(4)
            bt_work[b_id, :3, :3] = R_final
            bt_work[b_id, :3, 3] = t

    return p_id, bt_work, {"q_time": q_time, "svd_time": svd_time}


def opt_bt_parallel(
    v, p, bt, inv_rbt, vbw, bj, bj_l, fixed_bones, 
    max_workers=None, enable_profiling=False, 
    trans_affine=10.0, trans_affine_norm=4.0, bt_covariance_mode="corrected",
    rbt=None, par_idxs=None, roll_affinity=0.0, roll_affinity_norm=2.0
):
    n_p = p.shape[0]
    n_b = inv_rbt.shape[0]
    bt_new = np.copy(bt)

    t_start = time.perf_counter()
    total_q_time = 0.0
    total_svd_time = 0.0

    # 预计算全局共享的不变量
    v_b_cache = [skinning_b(v, inv_rbt, b_id) for b_id in range(n_b)]
    bone_jnts_local = []
    for b_id in range(n_b):
        jnts = bj[b_id]
        c_local = np.zeros((len(jnts), 3))
        for i, (nbr_b_id, jnt) in enumerate(jnts):
            c_local[i] = skinning_b(jnt[np.newaxis, :], inv_rbt, b_id)[0]
        bone_jnts_local.append((jnts, c_local))

    # 预计算 BFS 顺序
    bone_update_order = build_bone_bfs_order(par_idxs)
    
    # 预计算 Rest Roll 偏移 (复用 BoneHealthPrecompute 的逻辑)
    from ogd_lbs.ssdr.bone_health import BoneHealthPrecompute
    if rbt is not None and par_idxs is not None:
        bh_pre = BoneHealthPrecompute(rbt, par_idxs)
        rest_roll_angles = bh_pre.rest_roll_angles
        rest_roll_valid = bh_pre.rest_roll_valid
    else:
        rest_roll_angles = np.zeros(n_b, dtype=np.float64)
        rest_roll_valid = np.zeros(n_b, dtype=bool)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                calc_bt_parallel, v, p, bt[p_id], inv_rbt, vbw, bj_l, fixed_bones, p_id, 
                v_b_cache, bone_jnts_local, trans_affine, trans_affine_norm, bt_covariance_mode,
                par_idxs, rest_roll_angles, rest_roll_valid, bone_update_order,
                roll_affinity, roll_affinity_norm
            )
            for p_id in range(n_p)
        ]
        for future in as_completed(futures):
            p_id, bt_out, stats = future.result()
            bt_new[p_id] = bt_out
            total_q_time += stats["q_time"]
            total_svd_time += stats["svd_time"]

    if enable_profiling:
        log(f"  -> [Profiling] Bone Trans - Wall Time: {time.perf_counter() - t_start:.2f}s | Sum of Q: {total_q_time:.2f}s | Sum of SVD: {total_svd_time:.2f}s")

    return bt_new