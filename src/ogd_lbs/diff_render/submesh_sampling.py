# filepath: src/ogd_lbs/diff_render/submesh_sampling.py
import numpy as np
import difflib
from typing import List, Dict

from ogd_lbs.diff_render.config import DiffRenderConfig
from ogd_lbs.diff_render.nvdiffmodeling import mesh
from ogd_lbs.utils.common import log, warn

def compute_mesh_mapping(bmesh_info: List[Dict[str, str]], rmesh_info: List[Dict[str, str]]) -> List[int]:
    """🌟 核心升级：基于字典信息的多级匹配与相似度推荐"""
    rmesh2bmesh = []

    log("\n" + "="*70)
    log("[Target -> Source 映射详情]")

    for r_info in rmesh_info:
        mapped_idx = 0
        matched = False

        r_clean_stem = r_info['clean_stem'].lower()
        r_clean_name = r_info['clean_name'].lower()

        # 1. 精确匹配 (交叉对比 clean_stem 和 clean_name)
        for i, b_info in enumerate(bmesh_info):
            b_clean_stem = b_info['clean_stem'].lower()
            b_clean_name = b_info['clean_name'].lower()
            if (r_clean_stem == b_clean_stem or r_clean_name == b_clean_name or
                r_clean_stem == b_clean_name or r_clean_name == b_clean_stem):
                mapped_idx = i
                matched = True
                break

        # 2. 子串匹配 (如果精确匹配失败)
        if not matched:
            for i, b_info in enumerate(bmesh_info):
                b_clean_stem = b_info['clean_stem'].lower()
                b_clean_name = b_info['clean_name'].lower()
                if (r_clean_stem in b_clean_stem or b_clean_stem in r_clean_stem or
                    r_clean_name in b_clean_name or b_clean_name in r_clean_name):
                    mapped_idx = i
                    matched = True
                    break

        # 3. 相似度推荐 (如果依然匹配失败)
        if not matched:
            best_ratio = -1.0
            best_idx = 0
            for i, b_info in enumerate(bmesh_info):
                b_clean_stem = b_info['clean_stem'].lower()
                ratio = difflib.SequenceMatcher(None, r_clean_stem, b_clean_stem).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i

            mapped_idx = best_idx
            s_info = bmesh_info[best_idx]
            warn(f"Target 网格 '{r_info['raw_stem']}' 未匹配到 Source！"
                 f"最相似的 Source 是 '{s_info['raw_stem']}' (相似度: {best_ratio*100:.1f}%)，已默认回退到索引 {best_idx}。")

        # 🌟 打印详尽的映射关系表
        s_info = bmesh_info[mapped_idx]
        log(f"- Target: stem='{r_info['raw_stem']}', name='{r_info['raw_name']}' (clean: '{r_info['clean_stem']}')\n"
            f"  => Source: stem='{s_info['raw_stem']}', name='{s_info['raw_name']}' (clean: '{s_info['clean_stem']}') [索引: {mapped_idx}]")

        rmesh2bmesh.append(mapped_idx)

    log("="*70 + "\n")
    return rmesh2bmesh

class SubmeshSampler:
    def __init__(self, config: DiffRenderConfig, rmesh2bmesh: List[int],
                 bmesh_f_offset: List[int], rmesh_f_offset: List[int]):
        self.config = config
        self.active = config.use_submesh_sampling
        self.bmesh_f_offset = bmesh_f_offset
        self.rmesh_f_offset = rmesh_f_offset
        self.n_sub_bmesh = len(bmesh_f_offset) - 1
        self.n_sub_rmesh = len(rmesh_f_offset) - 1
        self.rmesh2bmesh = rmesh2bmesh

        if self.active:
            log(f"已启用子网格分件采样 (模式: {self.config.submesh_mode})，Target->Source 索引映射: {self.rmesh2bmesh}")

    def _get_rmesh_mask(self, iter_idx: int, max_iter: int) -> List[int]:
        if self.config.submesh_mode == "bake":
            switch_iter = int(max_iter * getattr(self.config, 'bake_isolated_ratio', 0.5))
            if iter_idx < switch_iter:
                mask = [0] * self.n_sub_rmesh
                mask[iter_idx % self.n_sub_rmesh] = 1
                return mask
            else:
                return [1] * self.n_sub_rmesh
        else:
            mode = getattr(self.config, "submesh_sampling_mode", "off")
            if mode == "off":
                return [1] * self.n_sub_rmesh
            elif mode == "random":
                # 🌟 修复：增加 min(1.0, ...) 防止在追加训练阶段概率越界
                p_active = min(1.0, iter_idx / max(1, max_iter))
                mask = np.random.choice([1, 0], size=self.n_sub_rmesh, p=[p_active, 1 - p_active])
                if np.all(mask == 0): mask[iter_idx % self.n_sub_rmesh] = 1
                return mask.tolist()
            elif mode == "anneal":
                rest_end = max_iter * getattr(self.config, "rest_pose_ratio", 0.1)
                if iter_idx < rest_end:
                    mask = [0] * self.n_sub_rmesh
                    mask[iter_idx % self.n_sub_rmesh] = 1
                    return mask
                else:
                    # 🌟 修复：增加 min(1.0, ...) 防止在追加训练阶段概率越界
                    p_active = min(1.0, iter_idx / max(1, max_iter))
                    mask = np.random.choice([1, 0], size=self.n_sub_rmesh, p=[p_active, 1 - p_active])
                    if np.all(mask == 0): mask[iter_idx % self.n_sub_rmesh] = 1
                    return mask.tolist()
            else:
                return [1] * self.n_sub_rmesh

    def _mask_ref2base(self, ref_mask: List[int]) -> List[int]:
        idxs = np.where(ref_mask)[0]
        idxs_remap = np.array(self.rmesh2bmesh)[idxs]
        base_mask = np.zeros((self.n_sub_bmesh,), dtype=int)
        base_mask[idxs_remap] = 1
        return base_mask.tolist()

    def _extract_submesh(self, full_mesh: mesh.Mesh, mask: List[int], f_offset: List[int]) -> mesh.Mesh:
        sub_t_idx = []
        for i, is_active in enumerate(mask):
            if is_active:
                f_from, f_to = f_offset[i], f_offset[i + 1]
                sub_t_idx.extend(range(f_from, f_to))
        return mesh.sub_mesh(full_mesh, sub_t_idx).eval()

    def apply(self, iter_idx: int, opt_mesh: mesh.Mesh, ref_mesh: mesh.Mesh):
        if not self.active:
            return opt_mesh, ref_mesh
        rmesh_mask = self._get_rmesh_mask(iter_idx, self.config.iter)
        bmesh_mask = self._mask_ref2base(rmesh_mask)
        opt_sub = self._extract_submesh(opt_mesh, bmesh_mask, self.bmesh_f_offset)
        ref_sub = self._extract_submesh(ref_mesh, rmesh_mask, self.rmesh_f_offset)
        return opt_sub, ref_sub
