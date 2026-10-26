from __future__ import annotations

import time
import os
import numpy as np
from scipy.optimize import lsq_linear
from concurrent.futures import ProcessPoolExecutor, as_completed

# 修正内部导入路径
from ogd_lbs.ssdr.skinning import skinning_t_single
from ogd_lbs.utils.common import log


def _lsp_parm_vbp(v, p, bt, inv_rbt, v_id, b_indices, p_id):
    A = np.zeros((3, len(b_indices)), dtype=np.float64)
    B = p[p_id, v_id]
    for i, b_id in enumerate(b_indices):
        A[:, i] = skinning_t_single(v[v_id], bt, inv_rbt, b_id, p_id)
    return A, B


def _lsp_parm_vb(v, p, bt, inv_rbt, v_id, b_indices):
    n_p = p.shape[0]
    n_b_idx = len(b_indices)
    
    # 预分配内存，避免循环 concatenate
    A = np.zeros((3 * n_p, n_b_idx), dtype=np.float64)
    B = np.zeros(3 * n_p, dtype=np.float64)
    
    for p_id in range(n_p):
        a, b = _lsp_parm_vbp(v, p, bt, inv_rbt, v_id, b_indices, p_id)
        A[p_id*3 : (p_id+1)*3, :] = a
        B[p_id*3 : (p_id+1)*3] = b
    return A, B


def calc_weight_parallel(v, p, bt, inv_rbt, s, vbw_mask, v_id, mode=1, prev_bw=None, weight_solver='legacy', target_w=None, lambda_reg=0.0):
    t0 = time.perf_counter()
    b_indices = np.where(vbw_mask[v_id] > 0)[0]
    
    if len(b_indices) == 0:
        if prev_bw is not None: return v_id, prev_bw, {"matrix_build_time": 0.0, "solver_time": 0.0}
        return v_id, np.zeros(bt.shape[1]), {"matrix_build_time": 0.0, "solver_time": 0.0}

    A, B = _lsp_parm_vb(v, p, bt, inv_rbt, v_id, b_indices)
    
    if lambda_reg > 0.0 and target_w is not None:
        sqrt_lambda = np.sqrt(lambda_reg)
        A_reg = np.eye(len(b_indices)) * sqrt_lambda
        B_reg = target_w[b_indices] * sqrt_lambda
        A = np.concatenate((A, A_reg), axis=0)
        B = np.concatenate((B, B_reg), axis=0)
        
    t1 = time.perf_counter()
    matrix_build_time = t1 - t0

    x_opt = None
    if weight_solver == 'simplex':
        # 修正动态导入路径
        from ogd_lbs.ssdr.convex_ls import solve_simplex
        x_opt, success = solve_simplex(A, B)
        if not success: x_opt = None
            
    if x_opt is None:
        try:
            result = lsq_linear(A, B, bounds=(0, 1), method='bvls')
            x_opt = result.x
        except ValueError:
            if prev_bw is not None: return v_id, prev_bw, {"matrix_build_time": matrix_build_time, "solver_time": time.perf_counter() - t1}
            return v_id, np.zeros(bt.shape[1]), {"matrix_build_time": matrix_build_time, "solver_time": time.perf_counter() - t1}

    w_retained_local_idxs = np.argsort(-x_opt)[:min(s, len(b_indices))]
    bw = np.zeros((bt.shape[1]))
    
    if mode == 1:
        bw[b_indices[w_retained_local_idxs]] = x_opt[w_retained_local_idxs]
    else:
        b_indices_sparse = b_indices[w_retained_local_idxs]
        A, B = _lsp_parm_vb(v, p, bt, inv_rbt, v_id, b_indices_sparse)
        
        if lambda_reg > 0.0 and target_w is not None:
            sqrt_lambda = np.sqrt(lambda_reg)
            A_reg = np.eye(len(b_indices_sparse)) * sqrt_lambda
            B_reg = target_w[b_indices_sparse] * sqrt_lambda
            A = np.concatenate((A, A_reg), axis=0)
            B = np.concatenate((B, B_reg), axis=0)
            
        if weight_solver == 'simplex':
            # 修正动态导入路径
            from ogd_lbs.ssdr.convex_ls import solve_simplex
            x_opt_sparse, success = solve_simplex(A, B)
            if success: bw[b_indices_sparse] = x_opt_sparse
            else: weight_solver = 'legacy'
                
        if weight_solver != 'simplex':
            try:
                result = lsq_linear(A, B, bounds=(0, 1), method='bvls')
                bw[b_indices_sparse] = result.x
            except ValueError:
                if prev_bw is not None: return v_id, prev_bw, {"matrix_build_time": matrix_build_time, "solver_time": time.perf_counter() - t1}

    w_sum = np.sum(bw)
    if w_sum > 1e-10:
        bw /= w_sum
    else:
        if prev_bw is not None and np.sum(prev_bw) > 1e-10:
            bw = prev_bw / np.sum(prev_bw)
        else:
            bw[b_indices] = 1.0 / len(b_indices)

    t2 = time.perf_counter()
    solver_time = t2 - t1

    return v_id, bw, {"matrix_build_time": matrix_build_time, "solver_time": solver_time}


def _calc_weight_chunk(v_ids, v, p, bt, inv_rbt, s, vbw_mask, mode, prev_vbw, weight_solver, target_w_full, lambda_reg):
    """批量处理顶点，减少 IPC 开销"""
    results = []
    for v_id in v_ids:
        res = calc_weight_parallel(
            v, p, bt, inv_rbt, s, vbw_mask, v_id, mode, 
            prev_vbw[v_id] if prev_vbw is not None else None,
            weight_solver,
            target_w_full[v_id] if target_w_full is not None else None,
            lambda_reg
        )
        results.append(res)
    return results


def opt_vbw_parallel(v, p, bt, inv_rbt, s, vbw_mask=None, mode=1, prev_vbw=None, max_workers=None, weight_solver='legacy', target_w_full=None, lambda_reg=0.0, enable_profiling=False):
    v_count, b_count = v.shape[0], inv_rbt.shape[0]
    vbw = np.zeros((v_count, b_count))

    t_start = time.perf_counter()
    total_matrix_time = 0.0
    total_solver_time = 0.0

    # 将顶点划分为 Chunk
    worker_count = max_workers if max_workers else (os.cpu_count() or 4)
    chunk_size = max(1, v_count // (worker_count * 4))
    chunks = [range(i, min(i + chunk_size, v_count)) for i in range(0, v_count, chunk_size)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _calc_weight_chunk, chunk, v, p, bt, inv_rbt, s, vbw_mask, mode, prev_vbw, weight_solver, target_w_full, lambda_reg
            ) 
            for chunk in chunks
        ]
        for future in as_completed(futures):
            for v_id, bw, stats in future.result():
                vbw[v_id] = bw
                total_matrix_time += stats["matrix_build_time"]
                total_solver_time += stats["solver_time"]

    t_end = time.perf_counter()
    wall_time = t_end - t_start
    
    if enable_profiling:
        log(f"  -> [Profiling] Weight Opt - Wall Time: {wall_time:.2f}s | Sum of Matrix Build: {total_matrix_time:.2f}s | Sum of Solver: {total_solver_time:.2f}s")

    return vbw