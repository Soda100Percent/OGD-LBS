# filepath: src/ogd_lbs/diff_render/geom_constraints.py
import torch
from typing import List

from ogd_lbs.diff_render.config import DiffRenderConfig
from ogd_lbs.diff_render.nvdiffmodeling.mesh import Mesh
from ogd_lbs.utils.common import log

class GeomConstraints:
    """几何约束工具：用于冻结特定子网格（如眼睛、牙齿）的顶点位置和蒙皮权重"""

    def __init__(self, config: DiffRenderConfig, base_mesh: Mesh, mesh_names: List[str], v_offset: List[int]):
        self.config = config
        self.active = len(self.config.fixed_submeshes) > 0
        self.mask = None
        self.cached_v_pos = None
        self.cached_vbw = None

        if not self.active:
            return

        # 计算需要冻结的顶点索引掩码
        total_vertices = base_mesh.v_pos.shape[0]
        mask_tensor = torch.zeros(total_vertices, dtype=torch.bool, device=base_mesh.v_pos.device)

        frozen_count = 0
        for target_name in self.config.fixed_submeshes:
            # 寻找匹配的子网格索引
            for i, name in enumerate(mesh_names):
                if target_name in name.split('&'):
                    v_from = v_offset[i]
                    v_to = v_offset[i + 1]
                    mask_tensor[v_from:v_to] = True
                    frozen_count += (v_to - v_from)

        if frozen_count > 0:
            self.mask = mask_tensor
            # 缓存初始状态
            self.cached_v_pos = base_mesh.v_pos[self.mask].clone().detach()
            self.cached_vbw = base_mesh.v_weights[self.mask].clone().detach()
            log(f"已启用几何约束，共冻结 {frozen_count} 个顶点 (目标部件: {self.config.fixed_submeshes})")
        else:
            log(f"未找到匹配的固定部件名称: {self.config.fixed_submeshes}，几何约束未生效。")
            self.active = False

    def apply(self, v_pos_opt: torch.Tensor, vbw_opt: torch.Tensor):
        """在 optimizer.step() 之后调用，强制覆盖回初始值"""
        if not self.active:
            return

        with torch.no_grad():
            v_pos_opt[self.mask] = self.cached_v_pos
            vbw_opt[self.mask] = self.cached_vbw
