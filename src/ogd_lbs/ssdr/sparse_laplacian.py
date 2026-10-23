from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class SparseLaplacianSolver:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, step: float = 1.0, eps: float = 1e-8):
        self.v_count = vertices.shape[0]
        self.step = step
        self.eps = eps
        self.solver = self._build_and_factorize(vertices, faces)
        
    def _build_and_factorize(self, vertices, faces):
        # 构建邻接表
        adj = [set() for _ in range(self.v_count)]
        for f in faces:
            for i in range(3):
                v1, v2 = f[i], f[(i + 1) % 3]
                adj[v1].add(v2)
                adj[v2].add(v1)
                
        row_indices, col_indices, data = [], [], []
        d = np.zeros(self.v_count)
        
        # 构建拉普拉斯矩阵 (基于反距离权重)
        for i in range(self.v_count):
            for j in adj[i]:
                dist = np.linalg.norm(vertices[i] - vertices[j])
                val = 1.0 / (dist + self.eps)
                row_indices.append(i)
                col_indices.append(j)
                data.append(-val)
                d[i] += val
                
        for i in range(self.v_count):
            row_indices.append(i)
            col_indices.append(i)
            data.append(d[i])
            
        L = sp.csr_matrix((data, (row_indices, col_indices)), shape=(self.v_count, self.v_count))
        inv_D = sp.diags(1.0 / (d + self.eps))
        L_rw = inv_D @ L  # 随机游走归一化拉普拉斯
        
        # 预分解 (I + step * L)
        A = sp.eye(self.v_count, format='csc') + self.step * L_rw
        return spla.factorized(A.tocsc())
        
    def solve(self, w: np.ndarray) -> np.ndarray:
        """隐式求解 (I + step * L) ws = w"""
        # spla.factorized 返回的是直接可调用的函数，而不是对象
        ws = self.solver(w)
        
        ws = np.clip(ws, 0.0, 1.0)
        row_sums = ws.sum(axis=1, keepdims=True)
        # 安全归一化
        ws = np.divide(ws, row_sums, out=np.ones_like(ws)/ws.shape[1], where=row_sums>1e-8)
        return ws