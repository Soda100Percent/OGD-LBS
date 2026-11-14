# filepath: src/ogd_lbs/diff_render/renderer.py
import torch
import nvdiffrast.torch as dr

from .config import DiffRenderConfig
from ogd_lbs.diff_render.nvdiffmodeling import render

class DiffRendererWrapper:
    """高层渲染包装器：负责 nvdiffrast 上下文管理及调用底层渲染算子"""

    def __init__(self, config: DiffRenderConfig, device: torch.device):
        self.config = config
        self.device = device
        # 初始化 nvdiffrast 光栅化上下文
        self.glctx = dr.RasterizeGLContext(device=device)

    def forward_render(self, mesh_obj, mvp, campos, lightpos, resolution: int, spp: int, background: torch.Tensor = None, return_alpha: bool = False):
        """调用底层渲染算子"""
        return render.render_mesh(
            self.glctx,
            mesh_obj,
            mvp,
            campos,
            lightpos,
            self.config.light_power,
            resolution,
            spp=spp,
            num_layers=self.config.layers,
            msaa=True,
            background=background,
            min_roughness=self.config.min_roughness,
            return_alpha=return_alpha
        )
