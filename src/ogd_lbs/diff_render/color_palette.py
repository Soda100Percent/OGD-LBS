# filepath: src/ogd_lbs/diff_render/color_palette.py
import colorsys
import numpy as np
import torch
from typing import List, Dict, Tuple

from ogd_lbs.diff_render.config import DiffRenderConfig
from ogd_lbs.diff_render.nvdiffmodeling import texture, mesh
from ogd_lbs.utils.common import log

class ColorPalette:
    """ID 色图与调色板轮换工具"""

    def __init__(self, config: DiffRenderConfig):
        self.config = config
        self.n_palette = config.n_kd_palette
        self.opt_variants = []
        self.ref_variants = []
        self.active = config.use_color_palette
        self.base_colors = []  # 缓存生成的纯色列表，用于稳定轮换

    @staticmethod
    def get_colors(n: int) -> List[Tuple[float, float, float]]:
        """使用 HSV 色彩空间生成 N 个高对比度的 RGB 颜色"""
        hsv_tuples = [(x * 1.0 / n, 0.8, 0.5) for x in range(n)]
        return [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]

    def create_base_materials(self, num_meshes: int) -> List[Dict]:
        """为每个子网格生成独立的纯色材质，用于 loader 合并"""
        mats = []
        if not self.active:
            # 🌟 修复：即使不启用调色板，也必须返回 num_meshes 个材质！
            # 否则 merge_materials 无法横向拼接，导致所有子网格 UV 重叠，后续切片也会越界。
            for _ in range(num_meshes):
                mats.append({
                    'name': '_default_mat',
                    'bsdf': 'diffuse',
                    'kd': texture.Texture2D(torch.tensor([0.5, 0.5, 0.5, 1.0], dtype=torch.float32, device='cuda')),
                    'ks': texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')),
                    'normal': texture.Texture2D(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
                })
            return mats

        self.base_colors = self.get_colors(num_meshes)
        for c in self.base_colors:
            mats.append({
                'name': '_default_mat',
                'bsdf': 'diffuse',
                'kd': texture.Texture2D(torch.tensor([*c, 1.0], dtype=torch.float32, device='cuda')),
                'ks': texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')),
                'normal': texture.Texture2D(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
            })
        return mats

    def build_variants(self, base_mat: Dict, ref_mat: Dict):
        """提取图集颜色并生成多套轮换材质变体（使用稳定的向量化最近邻匹配）"""
        if not self.active:
            self.opt_variants = [base_mat]
            self.ref_variants = [ref_mat]
            return

        log(f"正在生成 {self.n_palette} 套 ID 色图材质变体...")

        np_kd_map = base_mat['kd'].data.clone().detach().cpu().numpy()[0]
        np_ref_kd_map = ref_mat['kd'].data.clone().detach().cpu().numpy()[0]

        np_kd_map_variants = np.zeros(((self.n_palette,) + np_kd_map.shape), dtype=np.float32)
        np_ref_kd_map_variants = np.zeros(((self.n_palette,) + np_ref_kd_map.shape), dtype=np.float32)

        base_colors_rgba = np.array([list(c) + [1.0] for c in self.base_colors], dtype=np.float32)

        flat_kd = np_kd_map.reshape(-1, 4)
        flat_ref = np_ref_kd_map.reshape(-1, 4)

        dists_kd = np.linalg.norm(flat_kd[:, None, :] - base_colors_rgba[None, :, :], axis=2)
        best_match_kd = np.argmin(dists_kd, axis=1)

        dists_ref = np.linalg.norm(flat_ref[:, None, :] - base_colors_rgba[None, :, :], axis=2)
        best_match_ref = np.argmin(dists_ref, axis=1)

        for i in range(self.n_palette):
            shifted_colors_rgba = np.roll(base_colors_rgba, -i, axis=0)

            shifted_flat_kd = shifted_colors_rgba[best_match_kd]
            np_kd_map_variants[i] = shifted_flat_kd.reshape(np_kd_map.shape)

            shifted_flat_ref = shifted_colors_rgba[best_match_ref]
            np_ref_kd_map_variants[i] = shifted_flat_ref.reshape(np_ref_kd_map.shape)

        for i in range(self.n_palette):
            torch_kd_map = texture.Texture2D(torch.tensor(np_kd_map_variants[i], dtype=torch.float32, device='cuda'))
            torch_ref_kd_map = texture.Texture2D(torch.tensor(np_ref_kd_map_variants[i], dtype=torch.float32, device='cuda'))

            self.opt_variants.append({
                'bsdf': base_mat['bsdf'],
                'kd': torch_kd_map,
                'ks': base_mat['ks'],
                'normal': base_mat['normal'],
                'back_deepen': True
            })

            self.ref_variants.append({
                'bsdf': ref_mat['bsdf'],
                'kd': torch_ref_kd_map,
                'ks': ref_mat['ks'],
                'normal': ref_mat['normal'],
                'back_deepen': False
            })

    def apply(self, iter_idx: int, opt_mesh: mesh.Mesh, ref_mesh: mesh.Mesh):
        """动态替换当前迭代的材质"""
        if not self.active or len(self.opt_variants) == 0:
            return opt_mesh, ref_mesh

        variant_idx = iter_idx % self.n_palette

        opt_mesh = mesh.mtl_override(opt_mesh, self.opt_variants[variant_idx]).eval()
        ref_mesh = mesh.mtl_override(ref_mesh, self.ref_variants[variant_idx]).eval()

        return opt_mesh, ref_mesh
