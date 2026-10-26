from __future__ import annotations
import numpy as np
from typing import Any

from ogd_lbs.ssdr.rotation_utils import shortest_arc_rotation, extract_twist_y_angle, wrap_angle_pi

class BoneHealthPrecompute:
    """
    预计算 Rest Pose 下的父子骨骼相对 Roll 偏移量与相对局部方向。
    使用 Swing-Twist 分解理论。
    """
    def __init__(self, rbt: np.ndarray, par_idxs: np.ndarray):
        self.rbt = rbt
        self.par_idxs = par_idxs
        self.n_b = rbt.shape[0]
        
        self.rest_roll_angles = np.zeros(self.n_b, dtype=np.float64)
        self.rest_local_y = np.zeros((self.n_b, 3), dtype=np.float64)
        self.rest_roll_valid = np.zeros(self.n_b, dtype=bool)
        
        for j in range(self.n_b):
            p = self.par_idxs[j]
            if p is None or p < 0:
                self.rest_roll_valid[j] = False
                continue
                
            Rp0 = self.rbt[p, :3, :3]
            Rc0 = self.rbt[j, :3, :3]
            
            yp0 = Rp0[:, 1]
            yc0 = Rc0[:, 1]
            
            # 记录 Rest Pose 下子骨骼 Y 轴在父骨骼局部空间的方向
            self.rest_local_y[j] = Rp0.T @ yc0
            
            # 将子骨骼 Y 轴对齐到父骨骼 Y 轴
            A = shortest_arc_rotation(yc0, yp0, fallback_axis=Rp0[:, 0])
            # 计算对齐后的相对旋转 (理论上应该是纯 Y 轴 Roll)
            H = Rp0.T @ A @ Rc0
            
            phi0 = extract_twist_y_angle(H)
            self.rest_roll_angles[j] = phi0
            self.rest_roll_valid[j] = True

    def evaluate(self, bt_seq: np.ndarray, bone_names: list[str] | None = None, return_maps: bool = False) -> dict[str, Any]:
        """
        评估动画序列中的骨骼健康度 (Roll 翻转与 相对 Swing 折叠)
        bt_seq: P x B x 4 x 4
        """
        n_p = bt_seq.shape[0]
        
        roll_deltas = []
        swing_deltas = []
        
        # 记录每根骨骼的最大畸变及其发生的帧索引
        bone_max_roll = np.zeros(self.n_b, dtype=np.float64)
        bone_max_swing = np.zeros(self.n_b, dtype=np.float64)
        bone_max_roll_frame = np.zeros(self.n_b, dtype=int)
        bone_max_swing_frame = np.zeros(self.n_b, dtype=int)
        
        # 记录每一帧中，所有有效骨骼的最大畸变 (用于返回逐帧映射)
        roll_max_per_frame = np.zeros(n_p, dtype=np.float64)
        swing_max_per_frame = np.zeros(n_p, dtype=np.float64)
        
        for j in range(self.n_b):
            if not self.rest_roll_valid[j]:
                continue
                
            p = self.par_idxs[j]
            
            for f in range(n_p):
                Rp = bt_seq[f, p, :3, :3]
                Rc = bt_seq[f, j, :3, :3]
                
                yp = Rp[:, 1]
                yc = Rc[:, 1]
                
                # 1. 评估相对 Swing Delta (非自然折断)
                y_expected = Rp @ self.rest_local_y[j]
                dot_val = np.clip(np.dot(yc, y_expected), -1.0, 1.0)
                swing_deg = np.arccos(dot_val) * 180.0 / np.pi
                swing_deltas.append(swing_deg)
                
                # 更新骨骼最大 Swing
                if swing_deg > bone_max_swing[j]:
                    bone_max_swing[j] = swing_deg
                    bone_max_swing_frame[j] = f
                # 更新帧最大 Swing
                if swing_deg > swing_max_per_frame[f]:
                    swing_max_per_frame[f] = swing_deg
                
                # 2. 评估 Roll 翻转
                A = shortest_arc_rotation(yc, yp, fallback_axis=Rp[:, 0])
                H = Rp.T @ A @ Rc
                phi = extract_twist_y_angle(H)
                
                delta = wrap_angle_pi(phi - self.rest_roll_angles[j])
                delta_deg = abs(delta) * 180.0 / np.pi
                roll_deltas.append(delta_deg)
                
                # 更新骨骼最大 Roll
                if delta_deg > bone_max_roll[j]:
                    bone_max_roll[j] = delta_deg
                    bone_max_roll_frame[j] = f
                # 更新帧最大 Roll
                if delta_deg > roll_max_per_frame[f]:
                    roll_max_per_frame[f] = delta_deg

        if not roll_deltas:
            return {"status": "no_valid_bones"}
            
        roll_arr = np.array(roll_deltas)
        swing_arr = np.array(swing_deltas)
        
        worst_roll_idx = int(np.argmax(bone_max_roll))
        worst_swing_idx = int(np.argmax(bone_max_swing))
        
        def get_bname(idx):
            return bone_names[idx] if bone_names else str(idx)
            
        ret = {
            "status": "ok",
            "roll_abs_deg_mean": float(np.mean(roll_arr)),
            "roll_abs_deg_p95": float(np.percentile(roll_arr, 95)),
            "roll_abs_deg_max": float(np.max(roll_arr)),
            "roll_over_120_ratio": float(np.mean(roll_arr > 120.0)),
            "roll_over_150_ratio": float(np.mean(roll_arr > 150.0)),
            
            "swing_delta_mean": float(np.mean(swing_arr)),
            "swing_delta_max": float(np.max(swing_arr)),
            "swing_over_150_ratio": float(np.mean(swing_arr > 150.0)),
            
            "worst_bone_by_roll": f"{get_bname(worst_roll_idx)} (Max: {bone_max_roll[worst_roll_idx]:.1f} deg @ Frame {bone_max_roll_frame[worst_roll_idx]})",
            "worst_bone_by_swing": f"{get_bname(worst_swing_idx)} (Max: {bone_max_swing[worst_swing_idx]:.1f} deg @ Frame {bone_max_swing_frame[worst_swing_idx]})"
        }
        
        if return_maps:
            ret["roll_max_per_frame"] = roll_max_per_frame
            ret["swing_max_per_frame"] = swing_max_per_frame
            
        return ret