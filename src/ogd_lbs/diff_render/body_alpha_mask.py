
# filepath: src/ogd_lbs/diff_render/body_alpha_mask.py
import json
from pathlib import Path
from typing import List, Optional

import torch
import nvdiffrast.torch as dr

from ogd_lbs.diff_render.config import DiffRenderConfig
from ogd_lbs.diff_render.nvdiffmodeling.mesh import Mesh
from ogd_lbs.diff_render.nvdiffmodeling import util
from ogd_lbs.utils.common import log


class BodyAlphaMask:
    """
    基于骨骼 alpha 的自动 body mask 管理器。

    核心公式：
        alpha_v = sum_b W[v, b] * A[body_mesh_id, b]

    其中：
        W: 顶点 alpha 蒙皮权重，初值取普通蒙皮权重并归一化
        A: 每个 body 子网格独立的一组 bone alpha，可训练，初值为 1
        alpha_v: 顶点 alpha，初始为 1，优化后可小于 1
    """

    def __init__(
        self,
        config: DiffRenderConfig,
        base_mesh: Mesh,
        mesh_names: List[str],
        v_offset: List[int],
        f_offset: List[int],
        face_uvs,
    ):
        self.config = config
        self.base_mesh = base_mesh
        self.mesh_names = mesh_names
        self.v_offset = v_offset
        self.f_offset = f_offset
        self.face_uvs = face_uvs

        self.device = base_mesh.v_pos.device

        # 🌟 核心修改：通过 body_alpha_opt_iter 决定是否激活 Body Mask
        self.active = getattr(config, "body_alpha_opt_iter", 0) > 0

        self.body_mesh_indices: List[int] = []
        self.body_mesh_names: List[str] = []

        self.bone_alpha: Optional[torch.Tensor] = None
        self.alpha_weights: Optional[torch.Tensor] = None

        if not self.active:
            return

        if base_mesh.v_weights is None:
            log("body_alpha_opt_iter > 0，但 base_mesh.v_weights 为空，BodyAlphaMask 未启用。")
            self.active = False
            return

        self.body_mesh_indices = self._match_body_meshes()
        self.body_mesh_names = [self.mesh_names[i] for i in self.body_mesh_indices]

        if len(self.body_mesh_indices) == 0:
            log(
                f"Body Alpha 优化已开启，但未匹配到 body 子网格。"
                f"body_mask_submeshes={self.config.body_mask_submeshes}"
            )
            self.active = False
            return

        # 防御性归一化 alpha 权重。
        with torch.no_grad():
            w = torch.clamp(base_mesh.v_weights.detach(), min=0.0)
            w_sum = torch.sum(w, dim=1, keepdim=True)
            valid = w_sum > 1e-8

            w_norm = torch.where(
                valid,
                w / torch.clamp(w_sum, min=1e-8),
                torch.zeros_like(w)
            )

        self.alpha_weights = w_norm

        num_body_meshes = len(self.body_mesh_indices)
        num_bones = base_mesh.v_weights.shape[1]

        self.bone_alpha = torch.ones(
            (num_body_meshes, num_bones),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        log(
            f"✅ BodyAlphaMask 已启用 | "
            f"body meshes={self.body_mesh_names} | "
            f"num_body_meshes={num_body_meshes} | "
            f"num_bones={num_bones} | "
            f"keep_weight={self.config.body_alpha_keep_weight} | "
            f"binary_weight={self.config.body_alpha_binary_weight} | "
            f"threshold={self.config.body_alpha_threshold}"
        )

    def _match_body_meshes(self) -> List[int]:
        targets = [
            str(x).lower()
            for x in getattr(self.config, "body_mask_submeshes", ["body"])
        ]

        out = []
        for i, name in enumerate(self.mesh_names):
            parts = [p.lower() for p in str(name).split("&")]
            matched = False

            for target in targets:
                for part in parts:
                    if target in part:
                        matched = True
                        break
                if matched:
                    break

            if matched:
                out.append(i)

        return out

    def parameters(self) -> List[torch.Tensor]:
        if not self.active or self.bone_alpha is None:
            return []
        return [self.bone_alpha]

    def compute_v_alpha(self, hard: bool = False, threshold: Optional[float] = None) -> torch.Tensor:
        total_vertices = self.base_mesh.v_pos.shape[0]

        v_alpha = torch.ones(
            (total_vertices, 1),
            dtype=torch.float32,
            device=self.device,
        )

        if not self.active:
            return v_alpha

        assert self.bone_alpha is not None
        assert self.alpha_weights is not None

        alpha_min = float(getattr(self.config, "body_alpha_min", 0.0))
        alpha_max = float(getattr(self.config, "body_alpha_max", 1.0))

        for local_idx, mesh_idx in enumerate(self.body_mesh_indices):
            v_from = self.v_offset[mesh_idx]
            v_to = self.v_offset[mesh_idx + 1]

            W = self.alpha_weights[v_from:v_to, :]  # [Nv, B]
            A = self.bone_alpha[local_idx, :]        # [B]

            alpha = torch.sum(W * A[None, :], dim=1, keepdim=True)
            alpha = torch.clamp(alpha, min=alpha_min, max=alpha_max)

            v_alpha[v_from:v_to, :] = alpha

        if hard:
            if threshold is None:
                threshold = float(getattr(self.config, "body_alpha_threshold", 0.5))
            v_alpha = torch.where(
                v_alpha > threshold,
                torch.ones_like(v_alpha),
                torch.zeros_like(v_alpha)
            )

        return v_alpha

    def regularization_loss(self) -> torch.Tensor:
        if not self.active or self.bone_alpha is None:
            return torch.zeros((), dtype=torch.float32, device=self.device)

        keep_weight = float(getattr(self.config, "body_alpha_keep_weight", 0.01))
        binary_weight = float(getattr(self.config, "body_alpha_binary_weight", 0.0))

        loss = torch.zeros((), dtype=torch.float32, device=self.device)

        if keep_weight > 0.0:
            keep_loss = torch.mean((1.0 - self.bone_alpha) ** 2)
            loss = loss + keep_weight * keep_loss

        if binary_weight > 0.0:
            ba = torch.clamp(self.bone_alpha, 0.0, 1.0)
            binary_loss = torch.mean(ba * (1.0 - ba))
            loss = loss + binary_weight * binary_loss

        return loss

    def clamp_(self):
        if not self.active or self.bone_alpha is None:
            return

        alpha_min = float(getattr(self.config, "body_alpha_min", 0.0))
        alpha_max = float(getattr(self.config, "body_alpha_max", 1.0))

        with torch.no_grad():
            self.bone_alpha.clamp_(min=alpha_min, max=alpha_max)

    def save_bone_alpha_json(self, out_dir: str | Path):
        if not self.active or self.bone_alpha is None:
            return

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "body_mesh_names": self.body_mesh_names,
            "body_mesh_indices": self.body_mesh_indices,
            "bone_alpha": self.bone_alpha.detach().cpu().numpy().tolist(),
            "body_alpha_keep_weight": float(getattr(self.config, "body_alpha_keep_weight", 0.01)),
            "body_alpha_binary_weight": float(getattr(self.config, "body_alpha_binary_weight", 0.0)),
            "body_alpha_threshold": float(getattr(self.config, "body_alpha_threshold", 0.5)),
        }

        # 🌟 扁平化：存入 geometry 文件夹
        path = out_dir / "body_bone_alpha.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        log(f"Body bone alpha 已保存: {path}")

    def _bake_mask_for_body_mesh(
        self,
        ctx,
        mesh_idx: int,
        res_y: int,
        res_x: int,
        hard: bool,
    ) -> torch.Tensor:
        assert self.active

        f_from = self.f_offset[mesh_idx]
        f_to = self.f_offset[mesh_idx + 1]

        if f_to <= f_from:
            return torch.ones((res_y, res_x, 1), dtype=torch.float32, device=self.device)

        face_uvs = torch.tensor(
            self.face_uvs[f_from:f_to],
            dtype=torch.float32,
            device=self.device
        )  # [F, 3, 2]

        faces = self.base_mesh.t_pos_idx[f_from:f_to, :]  # [F, 3]

        v_alpha = self.compute_v_alpha(
            hard=hard,
            threshold=float(getattr(self.config, "body_alpha_threshold", 0.5))
        )  # [V, 1]

        corner_alpha = v_alpha[faces.reshape(-1), :]  # [F*3, 1]
        corner_uv = face_uvs.reshape(-1, 2)           # [F*3, 2]

        x = corner_uv[:, 0:1] * 2.0 - 1.0
        y = 1.0 - corner_uv[:, 1:2] * 2.0

        z = torch.zeros_like(x)
        w = torch.ones_like(x)

        pos_clip = torch.cat([x, y, z, w], dim=-1)[None, ...].contiguous()

        tri = torch.arange(
            0,
            corner_uv.shape[0],
            dtype=torch.int32,
            device=self.device
        ).reshape(-1, 3).contiguous()

        rast, _ = dr.rasterize(ctx, pos_clip, tri, resolution=[res_y, res_x])

        mask, _ = dr.interpolate(
            corner_alpha[None, ...].contiguous(),
            rast,
            tri
        )

        covered = rast[..., 3:4] > 0
        mask = torch.where(covered, mask, torch.ones_like(mask))
        mask = torch.clamp(mask[0], 0.0, 1.0)  # [H, W, 1]

        return mask

    def bake_and_save_masks(
        self,
        ctx,
        out_root: str | Path,
        res_y: int,
        res_x: int,
    ):
        if not self.active:
            return

        # 🌟 扁平化：统一存入 masks 文件夹
        masks_dir = Path(out_root) / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        threshold = float(getattr(self.config, "body_alpha_threshold", 0.5))

        for mesh_idx in self.body_mesh_indices:
            name = self.mesh_names[mesh_idx]

            soft = self._bake_mask_for_body_mesh(
                ctx=ctx,
                mesh_idx=mesh_idx,
                res_y=res_y,
                res_x=res_x,
                hard=False,
            )

            hard = torch.where(
                soft > threshold,
                torch.ones_like(soft),
                torch.zeros_like(soft)
            )

            soft_rgb = soft.repeat(1, 1, 3).detach().cpu().numpy()
            hard_rgb = hard.repeat(1, 1, 3).detach().cpu().numpy()

            # 🌟 扁平化：带前缀保存
            soft_path = masks_dir / f"{name}_body_mask_soft.png"
            hard_path = masks_dir / f"{name}_body_mask.png"

            util.save_image(str(soft_path), soft_rgb)
            util.save_image(str(hard_path), hard_rgb)

            log(f"已烘焙 Body mask: {name} | soft={soft_path} | hard={hard_path}")
