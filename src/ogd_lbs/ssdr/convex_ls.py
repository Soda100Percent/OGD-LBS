from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def solve_simplex(A: np.ndarray, B: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """
    严格单纯形约束的最小二乘求解器
    Solve: min ||Ax - B||^2
    Subject to: sum(x) = 1.0, x >= 0
    """
    n = A.shape[1]
    if x0 is None:
        x0 = np.ones(n) / n
        
    def objective(x):
        res = A @ x - B
        return np.dot(res, res)
        
    def jacobian(x):
        return 2 * A.T @ (A @ x - B)
        
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bnds = [(0.0, 1.0) for _ in range(n)]
    
    # SLSQP 在处理这种极小维度 (如 top 4~8) 的单纯形约束时极其高效且稳定
    res = minimize(
        objective, x0, method='SLSQP', jac=jacobian, 
        bounds=bnds, constraints=cons, 
        options={'ftol': 1e-8, 'disp': False}
    )
    
    return res.x, res.success