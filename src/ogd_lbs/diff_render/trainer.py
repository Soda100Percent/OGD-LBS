# filepath: src/ogd_lbs/diff_render/trainer.py
import os
import time
import shutil
import math
import torch
import numpy as np
from pathlib import Path
from typing import Tuple

from .config import DiffRenderConfig
from .mesh_loader import MeshLoader
from .renderer import DiffRendererWrapper
from .color_palette import ColorPalette
from .submesh_sampling import SubmeshSampler
from .submesh_sampling import compute_mesh_mapping
from .geom_constraints import GeomConstraints
from .body_alpha_mask import BodyAlphaMask
from . import camera

# 引入复合 Image Loss
from .image_loss import CompositeImageLoss
from ogd_lbs.utils.common import log, warn, load_data, discover_and_resolve_files, resolve_specific_file
from ogd_lbs.diff_render.nvdiffmodeling import mesh, util, regularizer, obj, texture
import ogd_lbs.diff_render.nvdiffmodeling.renderutils as ru


class DifferentiableTrainer:
    def __init__(self, config: DiffRenderConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._angle_weighting_log_printed = False
        self._body_mask_phase_log_printed = False
        self._texture_alpha_binary_log_printed = False
        self.body_alpha_mask = None

        self.final_metrics = {}
        self.history_metrics = {}  # 存储深度评估历史曲线

        # Debug skeleton data.
        self.anim_poses_abs = None
        self.rest_pose_abs = None
        self.skeleton_parent_indices = None

        # Commit 1: trainability flags.
        # 这些 flag 会在 _init_assets() 之后被正式赋值。
        # 这里先初始化，避免某些异常路径下访问属性时报错。
        self.train_position = True
        self.train_weight = True
        self.train_kd = True

        # Commit 2: Laplacian regularizer cache.
        # 旧逻辑会在每个 iteration 重建 laplace_regularizer_const，
        # 导致每步重复 GPU->CPU 同步、Python 邻接表构建、CUDA index tensor 创建。
        # 新逻辑只在资产初始化完成后构建一次。
        self.lap_loss_fn = None
        self.lap_fac = 0.0
        self.min_lap_fac = 0.0

        log(f"初始化 DifferentiableTrainer，使用设备: {self.device}")
        log(f"当前沙盒运行目录: {self.config.run_dir}")

        self.renderer = DiffRendererWrapper(self.config, self.device)
        self.color_palette = ColorPalette(self.config)
        self.image_loss_fn = CompositeImageLoss(self.config, self.device)

        self._init_assets()
        self._build_optimizer()

        self.preview_background = torch.tensor(
            util.checkerboard(self.config.display_res, 8),
            dtype=torch.float32,
            device=self.device
        )

    def _normalize_parent_indices(self, parent_indices):
        if parent_indices is None:
            return None

        out = []
        for x in parent_indices:
            if x is None:
                out.append(-1)
                continue

            if isinstance(x, str):
                xs = x.strip().lower()
                if xs in ("none", "null", ""):
                    out.append(-1)
                    continue

            out.append(int(x))

        return out

    def _log_bsdf_hist(self, name: str, t_bsdf_id: torch.Tensor):
        """
        打印 mixed BSDF ID 分布，便于确认 Stage01 / Stage02 是否使用了预期着色。

        ID 约定：
          0: diffuse
          1: ambient_1sided
          2: ambient_2sided
          3: diffuse_1sided
          4: diffuse_2sided
          5: diffuse_2sided_back_deepen
        """
        if t_bsdf_id is None:
            log(f"{name} t_bsdf_id=None")
            return

        ids, counts = torch.unique(
            t_bsdf_id.detach().flatten(),
            return_counts=True
        )

        hist = {
            int(i.item()): int(c.item())
            for i, c in zip(ids, counts)
        }

        log(
            f"{name} BSDF ID histogram: {hist} | "
            f"0=diffuse, "
            f"1=ambient_1sided, "
            f"2=ambient_2sided, "
            f"3=diffuse_1sided, "
            f"4=diffuse_2sided, "
            f"5=diffuse_2sided_back_deepen"
        )

    def _init_assets(self):
        log("准备加载 3D 资产与可训练变量...")
        self.loader = MeshLoader(self.device)

        if not self.config.source_dir or not self.config.target_dir:
            raise ValueError("必须指定 source_dir 和 target_dir 才能进行训练！")

        self.opt_use_mixed = (
            any(isinstance(r, list) and len(r) == 2 for r in self.config.opt_submesh_bsdf)
            if self.config.opt_submesh_bsdf
            else False
        )
        self.ref_use_mixed = (
            any(isinstance(r, list) and len(r) == 2 for r in self.config.ref_submesh_bsdf)
            if self.config.ref_submesh_bsdf
            else False
        )

        log(
            f"BSDF 模式 | "
            f"opt_use_mixed={self.opt_use_mixed} | "
            f"ref_use_mixed={self.ref_use_mixed} | "
            f"opt_bsdf={self.config.opt_bsdf} | "
            f"ref_bsdf={self.config.ref_bsdf} | "
            f"enable_back_deepen={getattr(self.config, 'enable_back_deepen', False)}"
        )

        raw_bmesh_info = self.loader.scan_mesh_names(self.config.source_dir, self.config.ignore_suffixes)
        num_bmeshes = len(raw_bmesh_info)
        bmesh_mats = self.color_palette.create_base_materials(num_bmeshes)

        if self.opt_use_mixed:
            for m in bmesh_mats:
                m['bsdf'] = 'mixed'

        self.base_mesh, self.bmesh_names, self.bmesh_v_offset, self.bmesh_f_offset, self.face_uvs = self.loader.load_base_mesh(
            self.config.source_dir,
            bmesh_mats,
            self.config.ignore_suffixes,
            bsdf_rules=self.config.opt_submesh_bsdf if self.opt_use_mixed else None
        )

        self._log_bsdf_hist("opt/base", self.base_mesh.t_bsdf_id)

        a_path = resolve_specific_file(self.config.source_dir, "animation_data", [".pkl", ".pickle", ".json"])
        if not a_path:
            raise FileNotFoundError(f"Missing animation_data in {self.config.source_dir}")
        a_obj = load_data(a_path)

        bm_rp = np.array(a_obj["rest_pose"])
        bm_p = np.array(a_obj["animation_poses"])

        self.anim_poses_abs = bm_p.astype(np.float32)
        self.rest_pose_abs = bm_rp.astype(np.float32)

        bones_count = bm_rp.shape[0]
        poses_count = bm_p.shape[0]
        bm_anim_mtx = np.zeros((poses_count, bones_count, 4, 4))

        self.skeleton_parent_indices = None

        skeleton_path = resolve_specific_file(
            self.config.source_dir,
            "skeleton_hierarchy_data",
            [".pkl", ".pickle", ".json"]
        )

        if skeleton_path:
            try:
                skeleton_obj = load_data(skeleton_path)

                if isinstance(skeleton_obj, dict) and "parent_indices" in skeleton_obj:
                    self.skeleton_parent_indices = self._normalize_parent_indices(
                        skeleton_obj["parent_indices"]
                    )
                else:
                    warn(f"未在 skeleton_hierarchy_data 中找到 parent_indices: {skeleton_path}")
            except Exception as e:
                warn(f"读取 skeleton_hierarchy_data 失败: {skeleton_path} | error={e}")

        for p_id in range(poses_count):
            for b_id in range(bones_count):
                b2w = bm_p[p_id, b_id]
                w2rb = np.linalg.inv(bm_rp[b_id])
                bm_anim_mtx[p_id, b_id] = (b2w @ w2rb).T

        self.base_mesh.bone_mtx = torch.tensor(
            bm_anim_mtx,
            dtype=torch.float32,
            device=self.device
        )

        raw_rmesh_info = self.loader.scan_mesh_names(self.config.target_dir, self.config.ignore_suffixes)
        rmesh2bmesh = compute_mesh_mapping(raw_bmesh_info, raw_rmesh_info)

        if self.config.use_target_textures:
            log("正在扫描并精准加载 Target 目录下的真实 PNG 贴图...")
            ref_mats = []
            target_path = Path(self.config.target_dir)
            ref_bsdf = "mixed" if self.ref_use_mixed else self.config.ref_bsdf

            for info in raw_rmesh_info:
                fname = info["raw_stem"]
                sm_path = info["path"]

                try:
                    sm_obj = load_data(sm_path)
                    mtl_names = sm_obj.get('material_names', ['default'])
                except Exception as e:
                    warn(f"无法读取 {fname} 的材质信息，默认分配 1 个材质。错误: {e}")
                    mtl_names = ['default']

                for mtl_idx in range(len(mtl_names)):
                    possible_paths = [
                        target_path / f"{fname}_mtl{mtl_idx + 1}.png",
                        target_path / fname / f"{fname}_mtl{mtl_idx + 1}.png",
                        target_path / f"{fname}.png",
                        target_path / fname / f"{fname}.png"
                    ]

                    img_path = None
                    for p in possible_paths:
                        if p.exists():
                            img_path = p
                            break

                    if img_path:
                        ref_mats.append({
                            "name": "_default_mat",
                            "bsdf": ref_bsdf,
                            "kd": texture.srgb_to_rgb(texture.load_texture2D(str(img_path))),
                            "ks": texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)),
                            "normal": texture.Texture2D(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device))
                        })
                    else:
                        log(f"警告: 未找到 {fname} 的第 {mtl_idx+1} 个材质贴图，回退为默认灰模。")
                        ref_mats.append(bmesh_mats[0])
        else:
            ref_mats = bmesh_mats
            ref_bsdf = "mixed" if self.ref_use_mixed else self.config.ref_bsdf
            for mat in ref_mats:
                mat["bsdf"] = ref_bsdf

        self.ref_mesh_sequence, self.rmesh_names, self.rmesh_v_offset, self.rmesh_f_offset = self.loader.load_ref_mesh_sequence(
            self.config.target_dir,
            ref_mats,
            self.config.ignore_suffixes,
            use_face_materials=self.config.use_target_textures,
            material_indices=rmesh2bmesh,
            bsdf_rules=self.config.ref_submesh_bsdf if self.ref_use_mixed else None,
            target_frames=poses_count,
        )

        if len(self.ref_mesh_sequence) > 0:
            self._log_bsdf_hist("ref/frame0", self.ref_mesh_sequence[0].t_bsdf_id)

        self.poses_count = len(self.ref_mesh_sequence)

        self.test_frames = []
        self.train_frames = list(range(self.poses_count))

        if len(self.config.test_frame_range) == 2:
            start_ratio, end_ratio = self.config.test_frame_range
            start_idx = int(self.poses_count * start_ratio)
            end_idx = int(self.poses_count * end_ratio)
            end_idx = min(self.poses_count, end_idx)

            self.test_frames = list(range(start_idx, end_idx))
            self.train_frames = [i for i in range(self.poses_count) if i not in self.test_frames]

            if 0 not in self.train_frames:
                self.train_frames.insert(0, 0)
                if 0 in self.test_frames:
                    self.test_frames.remove(0)

        if len(self.train_frames) == 0:
            warn("训练帧为空！强制将第 0 帧加入训练集。")
            self.train_frames = [0]

        log(f"帧划分完毕 | 总帧数: {self.poses_count} | 训练帧数: {len(self.train_frames)} | 测试帧数: {len(self.test_frames)}")

        log("正在为参考网格序列预计算法线与切线...")
        for i in range(self.poses_count):
            m = self.ref_mesh_sequence[i]
            if m.v_nrm is None:
                m = mesh.auto_normals(m)
            m = mesh.compute_tangents(m)
            self.ref_mesh_sequence[i] = m.eval().clone()

        self.color_palette.build_variants(self.base_mesh.material, self.ref_mesh_sequence[0].material)
        self.submesh_sampler = SubmeshSampler(self.config, rmesh2bmesh, self.bmesh_f_offset, self.rmesh_f_offset)
        self.geom_constraints = GeomConstraints(self.config, self.base_mesh, self.bmesh_names, self.bmesh_v_offset)

        # ======================================================================================
        # Commit 1:
        #   根据 skip_train 正确设置训练 flag 与 requires_grad。
        #
        # 旧逻辑中，即使 Stage02 skip_train 包含 position/weight，
        # v_pos_opt / vbw_opt 仍然 requires_grad=True。
        #
        # 这会导致 Stage02 只训练纹理时仍然构建几何、蒙皮、法线、切线相关计算图。
        # 新逻辑保证：
        #   - skip_train 包含 position -> v_pos_opt.requires_grad=False
        #   - skip_train 包含 weight   -> vbw_opt.requires_grad=False
        #   - skip_train 包含 kd       -> kd 不进入 optimizer
        # ======================================================================================
        self.train_position = "position" not in self.config.skip_train
        self.train_weight = "weight" not in self.config.skip_train
        self.train_kd = "kd" not in self.config.skip_train

        log(
            f"训练开关 | "
            f"position={self.train_position} | "
            f"weight={self.train_weight} | "
            f"kd={self.train_kd}"
        )

        self.v_pos_opt = (
            self.base_mesh.v_pos
            .clone()
            .detach()
            .requires_grad_(self.train_position)
        )

        self.vbw_opt = (
            self.base_mesh.v_weights
            .clone()
            .detach()
            .requires_grad_(self.train_weight)
        )

        self.vbw_mask = self.base_mesh.v_weights.clone().detach() != 0

        self.body_alpha_mask = BodyAlphaMask(
            config=self.config,
            base_mesh=self.base_mesh,
            mesh_names=self.bmesh_names,
            v_offset=self.bmesh_v_offset,
            f_offset=self.bmesh_f_offset,
            face_uvs=self.face_uvs,
        )

        actual_res_y = min(self.config.texture_res[0], 1024)
        actual_res_x = min(self.config.texture_res[1], 1024)
        full_res_x = 2 ** math.ceil(math.log2(actual_res_x * num_bmeshes))
        full_res_y = 2 ** math.ceil(math.log2(actual_res_y))

        if self.config.random_textures:
            channels = 4 if self.config.layers > 1 else 3
            init_kd = torch.rand(
                full_res_y,
                full_res_x,
                channels,
                dtype=torch.float32,
                device=self.device
            )
            self.kd_map_opt = texture.create_trainable(init_kd, [full_res_y, full_res_x], not self.config.custom_mip)
        else:
            self.kd_map_opt = texture.create_trainable(self.base_mesh.material["kd"], [full_res_y, full_res_x], not self.config.custom_mip)

        with torch.no_grad():
            self.orig_opt_bmesh = self._build_opt_mesh(body_alpha_mode="none").eval().clone()

        # Commit 2:
        # 缓存 Laplacian regularizer 拓扑。
        self._build_laplacian_regularizer_cache()

    def _build_optimizer(self):
        param_groups = []
        lr_lambdas = []

        def add_group(params, lr: float, name: str, decay: bool):
            params = list(params)
            if len(params) == 0:
                return
            param_groups.append({"params": params, "lr": lr, "name": name})
            if decay:
                lr_lambdas.append(lambda step: max(0.0, 10 ** (-step * 0.0001)))
            else:
                lr_lambdas.append(lambda step: 1.0)
            log(f"Optimizer param group added | name={name} | lr={lr:.6f} | decay={decay}")

        if "position" not in self.config.skip_train:
            add_group([self.v_pos_opt], lr=float(self.config.learning_rate), name="position", decay=True)

        if "weight" not in self.config.skip_train:
            add_group([self.vbw_opt], lr=float(self.config.learning_rate), name="weight", decay=True)

        if "kd" not in self.config.skip_train:
            tex_lr = getattr(self.config, 'tex_learning_rate', self.config.learning_rate)
            add_group(self.kd_map_opt.getMips(), lr=float(tex_lr), name="kd", decay=True)

        if self.body_alpha_mask is not None and self.body_alpha_mask.active:
            add_group(
                self.body_alpha_mask.parameters(),
                lr=float(getattr(self.config, 'body_alpha_opt_lr', 0.03)),
                name="body_alpha",
                decay=False
            )

        if len(param_groups) == 0:
            raise RuntimeError("没有任何可训练参数，请检查 skip_train 配置。")

        self.optimizer = torch.optim.Adam(param_groups)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambdas)

    def _get_lr_log_text(self) -> str:
        parts = []
        for group in self.optimizer.param_groups:
            name = group.get("name", "unnamed")
            lr = group.get("lr", 0.0)
            parts.append(f"{name}:{lr:.6f}")
        return ", ".join(parts)

    def _get_loss_str(self, loss_dict: dict) -> str:
        parts = []
        if self.config.loss_l1_weight > 0:
            parts.append(f"L1:{loss_dict.get('l1', 0.0):.4f}")
        if self.config.loss_mse_weight > 0:
            parts.append(f"MSE:{loss_dict.get('mse', 0.0):.4f}")
        if self.config.loss_ssim_weight > 0:
            parts.append(f"SSIM:{loss_dict.get('ssim', 0.0):.4f}")
        if self.config.loss_lpips_weight > 0:
            parts.append(f"LPIPS:{loss_dict.get('lpips', 0.0):.4f}")
        return "|".join(parts)

    def _is_body_mask_phase(self, iteration: int) -> bool:
        body_alpha_opt_iter = getattr(self.config, 'body_alpha_opt_iter', 0)
        if body_alpha_opt_iter <= 0:
            return False
        if self.body_alpha_mask is None or not self.body_alpha_mask.active:
            return False
        return iteration >= self.config.iter

    def _build_laplacian_regularizer_cache(self):
        """
        Commit 2:
        构建并缓存 Laplacian regularizer。

        设计原则：
          1. Laplacian 拓扑只依赖 t_pos_idx，训练过程中不变。
          2. 旧逻辑每个 iteration 调用 regularizer.laplace_regularizer_const，
             会重复执行：
               - t_pos_idx.detach().cpu().numpy()
               - Python 邻接表构建
               - CUDA index tensor 创建
          3. 这里只构建一次，后续 train_step 中直接 eval。

        注意：
          - 只有 position 可训练时才需要 Laplacian。
          - Stage02 通常 skip_train 包含 position，因此这里会跳过。
          - Body Alpha phase 会在 train_step 中跳过 Laplacian eval。
        """
        self.lap_loss_fn = None

        if not self.train_position:
            log("Laplacian regularizer cache skipped | position is not trainable.")
            return

        # Laplacian 只需要 v_pos 和 t_pos_idx，不需要 material / normal / tangent / texture。
        #
        # 这里 v_pos 使用 self.v_pos_opt 本体，保证后续 optimizer 更新 v_pos_opt 后，
        # self.lap_loss_fn.eval() 读取到的是最新 position。
        lap_opt_mesh = mesh.Mesh(
            v_pos=self.v_pos_opt,
            t_pos_idx=self.base_mesh.t_pos_idx
        )

        lap_base_mesh = None
        if self.config.relative_laplacian:
            # base mesh 只作为 reference，不需要梯度。
            lap_base_mesh = mesh.Mesh(
                v_pos=self.orig_opt_bmesh.v_pos.detach(),
                t_pos_idx=self.base_mesh.t_pos_idx
            )

        self.lap_loss_fn = regularizer.laplace_regularizer_const(
            lap_opt_mesh,
            lap_base_mesh
        )

        log(
            f"Laplacian regularizer cache built | "
            f"relative={self.config.relative_laplacian} | "
            f"n_verts={self.base_mesh.v_pos.shape[0]} | "
            f"n_faces={self.base_mesh.t_pos_idx.shape[0]}"
        )

    def _eval_laplacian_loss(self, body_mask_phase: bool) -> torch.Tensor:
        """
        Commit 2:
        统一管理 Laplacian loss 是否参与当前 iteration。

        规则：
          - position 不可训练：跳过。
          - Laplacian cache 不存在：跳过。
          - Body Alpha phase：跳过。
          - 其他情况：使用缓存的 self.lap_loss_fn.eval()。
        """
        if (
            body_mask_phase
            or not self.train_position
            or self.lap_loss_fn is None
        ):
            return torch.zeros((), dtype=torch.float32, device=self.device)

        return self.lap_loss_fn.eval()

    def _update_laplacian_factor(self, iteration: int, lap_loss: torch.Tensor):
        """
        Commit 2:
        集中管理 lap_fac 更新逻辑。

        与旧逻辑保持一致：
          - 如果 config.laplacian_factor 非 0，则使用显式配置值。
          - 如果 config.laplacian_factor 为 0，则根据第 0 步 lap_loss 自适应。

        新增保护：
          - 如果 position 不训练或 lap_loss_fn 不存在，则 lap_fac = 0。
          - 如果自适应模式下 lap_loss 太小，则避免除零。
        """
        if iteration == 0:
            if not self.train_position or self.lap_loss_fn is None:
                self.lap_fac = 0.0
                self.min_lap_fac = 0.0
            elif self.config.laplacian_factor:
                self.lap_fac = float(self.config.laplacian_factor)
                self.min_lap_fac = self.lap_fac * 0.02
            else:
                lap_val = float(lap_loss.detach().item())
                if lap_val > 1e-12:
                    self.lap_fac = (0.1 / lap_val) * 0.25
                else:
                    self.lap_fac = 0.0
                self.min_lap_fac = self.lap_fac * 0.02
        else:
            if self.lap_fac > 0.0:
                self.lap_fac = (
                    (self.lap_fac - self.min_lap_fac)
                    * 10 ** (-iteration * 0.000001)
                    + self.min_lap_fac
                )

    def _angle_weight_gamma(self) -> float:
        decay = float(getattr(self.config, "angle_weight_grazing_decay", 0.0))
        return 1.0 + 15.0 * max(0.0, min(1.0, decay))

    def _compute_texture_alpha_binary_regularization(self, iteration: int) -> torch.Tensor:
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        if "kd" in self.config.skip_train or not hasattr(self, "kd_map_opt"):
            return zero

        weight = float(getattr(self.config, "texture_alpha_binary_weight", 0.0))
        if weight <= 0.0:
            return zero

        start_ratio = float(getattr(self.config, "texture_alpha_binary_start_ratio", 0.3))
        end_ratio = float(getattr(self.config, "bake_isolated_ratio", 0.5))
        start_iter = int(self.config.iter * start_ratio)
        end_iter = int(self.config.iter * end_ratio)

        if end_iter <= start_iter or iteration < start_iter or iteration >= end_iter:
            return zero

        mips = self.kd_map_opt.getMips()
        if len(mips) == 0 or mips[0].shape[-1] < 4:
            return zero

        alpha = torch.clamp(mips[0][..., 3:4], 0.0, 1.0)
        binary_loss = torch.mean(alpha * (1.0 - alpha))
        progress = (iteration - start_iter) / max(1, end_iter - start_iter)
        return binary_loss * (weight * max(0.0, min(1.0, float(progress))))

    def _detach_texture2d(self, tex_obj: texture.Texture2D) -> texture.Texture2D:
        """
        Commit 1 helper.

        Body Alpha 追加阶段只允许 body_alpha_mask.bone_alpha 接收梯度。
        因此材质中的 kd_map_opt 需要在 forward 前 detach。

        这里返回一个新的 Texture2D 包装对象，但底层 tensor 只是 detach 视图，
        不会复制贴图数据。
        """
        data = tex_obj.data

        if isinstance(data, list):
            return texture.Texture2D([m.detach() for m in data])

        return texture.Texture2D(data.detach())

    def _state_tensors_for_phase(self, body_alpha_mode: str):
        """
        Commit 1 helper.

        根据当前训练阶段决定 Mesh 中使用 trainable tensor 还是 detached tensor。

        规则：
          - 普通 Stage01:
              position/weight 是否可导由 skip_train 决定。
          - Stage02:
              position/weight 通常 requires_grad=False。
          - Body Alpha 追加阶段:
              强制 detach position 和 weight，只让 bone_alpha 参与 autograd。
        """
        body_phase = body_alpha_mode in ("soft", "hard")

        v_pos = (
            self.v_pos_opt
            if self.train_position and not body_phase
            else self.v_pos_opt.detach()
        )

        vbw = (
            self.vbw_opt
            if self.train_weight and not body_phase
            else self.vbw_opt.detach()
        )

        return v_pos, vbw

    def _build_opt_mesh(self, body_alpha_mode: str = "none"):
        opt_mat = self.base_mesh.material.copy()

        body_phase = body_alpha_mode in ("soft", "hard")

        if hasattr(self, "kd_map_opt"):
            if body_phase or not self.train_kd:
                # Commit 1:
                # Body Alpha 阶段只优化 bone_alpha。
                # kd_map_opt 需要在 forward 前 detach，而不是 backward 之后再清 grad。
                opt_mat["kd"] = self._detach_texture2d(self.kd_map_opt)
            else:
                opt_mat["kd"] = self.kd_map_opt

        opt_mat["bsdf"] = "mixed" if self.opt_use_mixed else self.config.opt_bsdf

        # legacy/global back_deepen 开关。
        #
        # 注意：
        #   本轮 two-stage Stage01 已经改为通过 BSDF ID=5
        #   diffuse_2sided_back_deepen 携带逐面 back_deepen 语义。
        #
        #   因此正常 two-stage Stage01 不应该强制设置 enable_back_deepen=True，
        #   否则 ambient_2sided -> diffuse_2sided 的面也会被误加深。
        #
        #   这里仅保留向后兼容：如果某些旧实验或手动命令显式打开
        #   enable_back_deepen，则 opt mesh 的 diffuse family 会全局背面加深。
        if getattr(self.config, "enable_back_deepen", False):
            opt_mat["back_deepen"] = True
        else:
            opt_mat.pop("back_deepen", None)

        v_alpha = None
        t_alpha_idx = None
        if (
            body_alpha_mode in ("soft", "hard")
            and self.body_alpha_mask is not None
            and self.body_alpha_mask.active
        ):
            v_alpha = self.body_alpha_mask.compute_v_alpha(hard=False)
            t_alpha_idx = self.base_mesh.t_pos_idx
            opt_mat["use_vertex_alpha"] = True
            if body_alpha_mode == "hard":
                opt_mat["alpha_clip_threshold"] = float(self.config.body_alpha_threshold)
            else:
                opt_mat.pop("alpha_clip_threshold", None)
        else:
            opt_mat.pop("alpha_clip_threshold", None)

        v_pos_current, vbw_current = self._state_tensors_for_phase(body_alpha_mode)

        opt_bmesh = mesh.Mesh(
            v_pos=v_pos_current,
            t_pos_idx=self.base_mesh.t_pos_idx,
            v_nrm=self.base_mesh.v_nrm,
            t_nrm_idx=self.base_mesh.t_nrm_idx,
            v_tex=self.base_mesh.v_tex,
            t_tex_idx=self.base_mesh.t_tex_idx,
            v_weights=vbw_current,
            bone_mtx=self.base_mesh.bone_mtx,
            material=opt_mat,
            v_alpha=v_alpha,
            t_alpha_idx=t_alpha_idx,
            t_bsdf_id=self.base_mesh.t_bsdf_id
        )
        opt_bmesh = mesh.auto_normals(opt_bmesh)
        opt_bmesh = mesh.compute_tangents(opt_bmesh)
        return opt_bmesh

    def _make_opaque_debug_material(self, base_material: dict, bsdf: str) -> dict:
        mat = base_material.copy()
        mat["bsdf"] = bsdf
        mat["kd"] = texture.Texture2D(
            torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
        )
        if "ks" not in mat or mat["ks"] is None:
            mat["ks"] = texture.Texture2D(
                torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            )
        if "normal" not in mat or mat["normal"] is None:
            mat["normal"] = texture.Texture2D(
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
            )
        mat["use_vertex_alpha"] = False
        mat.pop("alpha_clip_threshold", None)
        return mat

    def _compute_angle_weight_mask(
        self,
        opt_bmesh_skinned: mesh.Mesh,
        mvp,
        campos,
        lightpos,
        random_bg: torch.Tensor
    ) -> torch.Tensor:
        gamma = self._angle_weight_gamma()
        debug_bg = torch.zeros_like(random_bg)
        mask_mat = self._make_opaque_debug_material(opt_bmesh_skinned.material, "mask")
        position_mat = self._make_opaque_debug_material(opt_bmesh_skinned.material, "position")
        normal_mat = self._make_opaque_debug_material(opt_bmesh_skinned.material, "normal")

        opt_bmesh_mask = mesh.mtl_override(opt_bmesh_skinned, mask_mat).eval()
        opt_bmesh_pos = mesh.mtl_override(opt_bmesh_skinned, position_mat).eval()
        opt_bmesh_norm = mesh.mtl_override(opt_bmesh_skinned, normal_mat).eval()

        with torch.no_grad():
            mask_render = self.renderer.forward_render(
                opt_bmesh_mask,
                mvp,
                campos,
                lightpos,
                self.config.train_res,
                self.config.spp,
                debug_bg
            )
            position_render = self.renderer.forward_render(
                opt_bmesh_pos,
                mvp,
                campos,
                lightpos,
                self.config.train_res,
                self.config.spp,
                debug_bg
            )
            normal_render = self.renderer.forward_render(
                opt_bmesh_norm,
                mvp,
                campos,
                lightpos,
                self.config.train_res,
                self.config.spp,
                debug_bg
            )

            N_map = torch.nn.functional.normalize(
                normal_render[..., :3] * 2.0 - 1.0,
                dim=-1,
                eps=1e-8
            )
            P_map = position_render[..., :3]
            campos_t = torch.as_tensor(
                campos,
                dtype=torch.float32,
                device=self.device
            ).view(self.config.batch, 1, 1, 3)
            V_map = torch.nn.functional.normalize(campos_t - P_map, dim=-1, eps=1e-8)

            cos_nv = torch.sum(N_map * V_map, dim=-1, keepdim=True)
            angle_weight = torch.pow(torch.abs(cos_nv).clamp(0.0, 1.0), gamma)

            valid_mask = mask_render[..., 0:1].clamp(0.0, 1.0)
            weight_mask = valid_mask * angle_weight + (1.0 - valid_mask) * torch.ones_like(angle_weight)

        return weight_mask

    def _get_debug_joint_positions(self, frame_id: int):
        if self.anim_poses_abs is None:
            return None

        poses = self.anim_poses_abs

        if poses is None or len(poses.shape) != 4 or poses.shape[-2:] != (4, 4):
            return None

        frame_id = int(frame_id) % poses.shape[0]
        pose_mtx = poses[frame_id]

        col_translation = pose_mtx[:, :3, 3]
        row_translation = pose_mtx[:, 3, :3]

        col_abs = float(np.nanmax(np.abs(col_translation))) if col_translation.size > 0 else 0.0
        row_abs = float(np.nanmax(np.abs(row_translation))) if row_translation.size > 0 else 0.0

        if col_abs < 1e-8 and row_abs > 1e-8:
            joint_pos = row_translation
        else:
            joint_pos = col_translation

        return joint_pos.astype(np.float32)

    def _make_debug_view_tensor(self, mesh_obj: mesh.Mesh, mvp, frame_id: int) -> torch.Tensor:
        joint_pos = self._get_debug_joint_positions(frame_id)

        debug_img_np = util.render_debug_view(
            v_pos=mesh_obj.v_pos,
            t_pos_idx=mesh_obj.t_pos_idx,
            mvp=mvp[0],
            resolution=self.config.display_res,
            joint_pos=joint_pos,
            parent_indices=self.skeleton_parent_indices,
            draw_3d_aabb=True,
            max_wire_edges=None,
        )

        return torch.tensor(
            debug_img_np,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

    def _chunked_image_loss(self, c_opt, c_ref, weight_mask, eval_mode=False, chunk_size=4):
        total_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        loss_dict_sum = {}
        batch_size = c_opt.shape[0]

        for b_start in range(0, batch_size, chunk_size):
            b_end = min(b_start + chunk_size, batch_size)
            wm_chunk = weight_mask[b_start:b_end] if weight_mask is not None else None

            chunk_loss, chunk_dict = self.image_loss_fn(
                c_opt[b_start:b_end],
                c_ref[b_start:b_end],
                wm_chunk,
                eval_mode=eval_mode
            )

            weight = (b_end - b_start) / batch_size
            total_loss = total_loss + chunk_loss * weight
            for k, v in chunk_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v * weight

        return total_loss, loss_dict_sum

    # 🌟 修复 1：新增辅助函数判断网格是否为空（面片数为 0）
    def _mesh_has_faces(self, m: mesh.Mesh) -> bool:
        return (
            m is not None
            and m.t_pos_idx is not None
            and m.t_pos_idx.numel() > 0
            and m.t_pos_idx.shape[0] > 0
        )

    # 🌟 修复 2：新增辅助函数提取子网格真正使用的顶点，避免 Hausdorff 距离计算越界或炸显存
    def _used_vertices_for_mesh(self, m: mesh.Mesh) -> torch.Tensor:
        if not self._mesh_has_faces(m):
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)

        idx = torch.unique(m.t_pos_idx.reshape(-1))
        idx = idx[(idx >= 0) & (idx < m.v_pos.shape[0])]

        if idx.numel() == 0:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)

        return m.v_pos[idx]

    def train_step(self, iteration: int) -> Tuple[float, float, float]:
        # Commit 1:
        # set_to_none=True 可以减少不必要的 grad tensor 写零开销和显存占用。
        self.optimizer.zero_grad(set_to_none=True)

        rest_pose_end_iter = int(self.config.iter * self.config.rest_pose_ratio)
        if iteration < rest_pose_end_iter:
            frame_id = 0
        else:
            idx = iteration % len(self.train_frames)
            frame_id = self.train_frames[idx]

        body_mask_phase = self._is_body_mask_phase(iteration)
        body_alpha_mode = "soft" if body_mask_phase else "none"

        if body_mask_phase and not self._body_mask_phase_log_printed:
            log(f"进入追加的 Body Alpha 优化阶段 | iter={iteration} | 冻结几何与 RGB | 仅优化 body_bone_alpha")
            self._body_mask_phase_log_printed = True

        opt_bmesh = self._build_opt_mesh(body_alpha_mode=body_alpha_mode)

        # Commit 2:
        # 使用缓存的 Laplacian regularizer。
        # 不再每个 iteration 重建 regularizer。
        lap_loss = self._eval_laplacian_loss(body_mask_phase)

        opt_bmesh_skinned = mesh.skinning(opt_bmesh).eval({
            "time": frame_id,
            "weight_mask": self.vbw_mask
        })
        ref_mesh = self.ref_mesh_sequence[frame_id].clone()

        opt_bmesh_skinned, ref_mesh = self.submesh_sampler.apply(iteration, opt_bmesh_skinned, ref_mesh)
        opt_bmesh_skinned, ref_mesh = self.color_palette.apply(iteration, opt_bmesh_skinned, ref_mesh)

        focus_end_iter = int(self.config.iter * self.config.cam_focus_ratio)
        cam_mode = "train_focus" if (self.config.cam_focus_mode and iteration < focus_end_iter) else "train_orbit"

        mvp, campos, lightpos = camera.get_camera_params(
            self.config,
            opt_bmesh_skinned,
            ref_mesh,
            mode=cam_mode
        )
        random_bg = torch.rand(
            self.config.batch,
            self.config.train_res,
            self.config.train_res,
            3,
            dtype=torch.float32,
            device=self.device
        )

        color_opt = self.renderer.forward_render(
            opt_bmesh_skinned,
            mvp,
            campos,
            lightpos,
            self.config.train_res,
            self.config.spp,
            random_bg
        )
        with torch.no_grad():
            color_ref = self.renderer.forward_render(
                ref_mesh,
                mvp,
                campos,
                lightpos,
                self.config.train_res,
                self.config.spp,
                random_bg
            )

        weight_mask = (
            self._compute_angle_weight_mask(opt_bmesh_skinned, mvp, campos, lightpos, random_bg)
            if self.config.use_angle_weighting
            else None
        )

        img_loss, loss_dict = self._chunked_image_loss(color_opt, color_ref, weight_mask, eval_mode=False, chunk_size=4)
        self._current_loss_dict = loss_dict

        # Commit 2:
        # 集中更新 lap_fac。
        self._update_laplacian_factor(iteration, lap_loss)

        body_alpha_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if body_mask_phase and self.body_alpha_mask is not None and self.body_alpha_mask.active:
            body_alpha_loss = self.body_alpha_mask.regularization_loss()

        texture_alpha_binary_loss = self._compute_texture_alpha_binary_regularization(iteration)

        total_loss = img_loss + lap_loss * self.lap_fac + body_alpha_loss + texture_alpha_binary_loss
        total_loss.backward()

        # Commit 1:
        # 由于 body phase 已经在 forward 前 detach geometry / texture，
        # 理论上这里不再需要清 grad。
        # 但保留作为安全保险，避免未来新增路径意外让冻结参数收到梯度。
        if body_mask_phase:
            if self.v_pos_opt.grad is not None:
                self.v_pos_opt.grad = None
            if self.vbw_opt.grad is not None:
                self.vbw_opt.grad = None
            if hasattr(self, "kd_map_opt") and "kd" not in self.config.skip_train:
                for mip in self.kd_map_opt.getMips():
                    if mip.grad is not None:
                        mip.grad = None
        elif self.config.submesh_mode == "bake" and iteration >= int(self.config.iter * getattr(self.config, 'bake_isolated_ratio', 0.5)):
            if hasattr(self, "kd_map_opt") and "kd" not in self.config.skip_train:
                for mip in self.kd_map_opt.getMips():
                    if mip.grad is not None:
                        mip.grad[..., :3] = 0.0

        self.optimizer.step()
        self.scheduler.step()

        if hasattr(self, "kd_map_opt") and "kd" not in self.config.skip_train:
            self.kd_map_opt.clamp_(min=0.0, max=1.0)

        if self.body_alpha_mask is not None and self.body_alpha_mask.active:
            self.body_alpha_mask.clamp_()

        with torch.no_grad():
            epsilon = 1e-20
            self.vbw_opt.abs_()
            self.vbw_opt.div_(torch.sum(self.vbw_opt, dim=1, keepdim=True) + epsilon)
            self.vbw_opt *= self.vbw_mask

        self.geom_constraints.apply(self.v_pos_opt, self.vbw_opt)

        return total_loss.item(), img_loss.item(), lap_loss.item()

    def fast_eval_step(self, iteration: int) -> Tuple[float, dict]:
        """轻量级评估：继承训练相机模式，保留所有数据增强，用于同轨对比"""
        if not self.test_frames:
            return 0.0, {}

        max_f = self.config.eval_frame_max_test_set
        if len(self.test_frames) > max_f:
            indices = np.linspace(0, len(self.test_frames) - 1, max_f, dtype=int)
            eval_frames = [self.test_frames[i] for i in indices]
        else:
            eval_frames = self.test_frames

        img_loss_sum = 0.0
        loss_dict_sum = {}

        body_mask_phase = self._is_body_mask_phase(iteration)
        body_alpha_mode = "soft" if body_mask_phase else "none"

        # 动态继承相机的 Focus 模式
        focus_end_iter = int(self.config.iter * self.config.cam_focus_ratio)
        cam_mode = "train_focus" if (self.config.cam_focus_mode and iteration < focus_end_iter) else "train_orbit"

        with torch.no_grad():
            opt_bmesh = self._build_opt_mesh(body_alpha_mode=body_alpha_mode)

            for frame_id in eval_frames:
                opt_bmesh_skinned = mesh.skinning(opt_bmesh).eval({
                    "time": frame_id,
                    "weight_mask": self.vbw_mask
                })
                ref_mesh = self.ref_mesh_sequence[frame_id].clone()

                opt_bmesh_skinned, ref_mesh = self.submesh_sampler.apply(iteration, opt_bmesh_skinned, ref_mesh)
                opt_bmesh_skinned, ref_mesh = self.color_palette.apply(iteration, opt_bmesh_skinned, ref_mesh)

                mvp, campos, lightpos = camera.get_camera_params(
                    self.config, opt_bmesh_skinned, ref_mesh, mode=cam_mode
                )
                random_bg = torch.rand(self.config.batch, self.config.train_res, self.config.train_res, 3, dtype=torch.float32, device=self.device)

                color_opt = self.renderer.forward_render(opt_bmesh_skinned, mvp, campos, lightpos, self.config.train_res, self.config.spp, random_bg)
                color_ref = self.renderer.forward_render(ref_mesh, mvp, campos, lightpos, self.config.train_res, self.config.spp, random_bg)

                weight_mask = self._compute_angle_weight_mask(opt_bmesh_skinned, mvp, campos, lightpos, random_bg) if self.config.use_angle_weighting else None

                img_loss, loss_dict = self._chunked_image_loss(color_opt, color_ref, weight_mask, eval_mode=False, chunk_size=4)

                img_loss_sum += img_loss.item()
                for k, v in loss_dict.items():
                    loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v

        n = len(eval_frames)
        avg_img = img_loss_sum / n
        avg_dict = {k: v / n for k, v in loss_dict_sum.items()}

        return avg_img, avg_dict

    def _calc_hausdorff_gpu(self, A: torch.Tensor, B: torch.Tensor, chunk_size=10000) -> float:
        if A.shape[0] == 0 or B.shape[0] == 0:
            return 0.0

        def directed_hausdorff(X, Y):
            max_min_dist = 0.0
            for i in range(0, X.shape[0], chunk_size):
                X_chunk = X[i:i+chunk_size]
                dists = torch.cdist(X_chunk, Y)
                min_dists, _ = torch.min(dists, dim=1)
                max_min_dist = max(max_min_dist, torch.max(min_dists).item())
            return max_min_dist

        return max(directed_hausdorff(A, B), directed_hausdorff(B, A))

    def deep_eval_step(self, iteration: int):
        """深度评估机制：临时剥离相机抖动/缩放，保存可视化对比图，计算绝对基准指标（含 Std/Max/Min）"""

        # 1. 备份并覆写相机配置，确保评估环境绝对稳定
        orig_auto_center = getattr(self.config, 'enable_auto_center', True)
        orig_jitter = getattr(self.config, 'cam_jitter_pixels', 0.0)
        orig_zoom = getattr(self.config, 'cam_zoom_max', 0.0)

        self.config.enable_auto_center = True
        self.config.cam_jitter_pixels = 0.0
        self.config.cam_zoom_max = 0.0

        # 2. 准备可视化输出目录
        eval_img_dir = Path(self.config.run_dir) / "eval_images"
        eval_img_dir.mkdir(parents=True, exist_ok=True)

        try:
            def _evaluate_set(frames, prefix):
                if not frames:
                    return

                max_f = self.config.eval_frame_max_train_set if prefix == "train" else self.config.eval_frame_max_test_set
                if len(frames) > max_f:
                    indices = np.linspace(0, len(frames) - 1, max_f, dtype=int)
                    sampled_frames = [frames[i] for i in indices]
                else:
                    sampled_frames = frames

                # 🌟 核心修改 1：使用列表收集每一帧的数据
                metrics_collect = {"overall": {"l1": [], "mse": [], "ssim": [], "lpips": [], "hausdorff": [], "psnr": []}}
                for name in self.bmesh_names:
                    metrics_collect[name] = {"l1": [], "mse": [], "ssim": [], "lpips": [], "hausdorff": [], "psnr": []}

                body_alpha_mode = "soft" if self._is_body_mask_phase(iteration) else "none"

                with torch.no_grad():
                    opt_bmesh = self._build_opt_mesh(body_alpha_mode=body_alpha_mode)

                    for frame_id in sampled_frames:
                        opt_bmesh_skinned = mesh.skinning(opt_bmesh).eval({"time": frame_id, "weight_mask": self.vbw_mask})
                        ref_mesh = self.ref_mesh_sequence[frame_id].clone()

                        # 强制使用 train_orbit (随机旋转) 评估多视角
                        mvp, campos, lightpos = camera.get_camera_params(self.config, opt_bmesh_skinned, ref_mesh, mode="train_orbit")
                        bg = torch.rand(self.config.batch, self.config.train_res, self.config.train_res, 3, dtype=torch.float32, device=self.device)

                        # -----------------------------------------------------------
                        # 1. Overall Eval & Visual Confirmation
                        # -----------------------------------------------------------
                        c_opt = self.renderer.forward_render(opt_bmesh_skinned, mvp, campos, lightpos, self.config.train_res, self.config.spp, bg)
                        c_ref = self.renderer.forward_render(ref_mesh, mvp, campos, lightpos, self.config.train_res, self.config.spp, bg)

                        _, l_dict = self._chunked_image_loss(c_opt, c_ref, None, eval_mode=True, chunk_size=4)

                        overall_opt_v = self._used_vertices_for_mesh(opt_bmesh_skinned)
                        overall_ref_v = self._used_vertices_for_mesh(ref_mesh)
                        haus = self._calc_hausdorff_gpu(overall_opt_v, overall_ref_v)

                        # 🌟 核心修改 2：计算当帧 PSNR 并存入列表
                        frame_mse = l_dict.get("mse", 0.0)
                        frame_psnr = -10.0 * math.log10(frame_mse + 1e-8)

                        metrics_collect["overall"]["psnr"].append(frame_psnr)
                        metrics_collect["overall"]["hausdorff"].append(haus)
                        for k in ["l1", "mse", "ssim", "lpips"]:
                            metrics_collect["overall"][k].append(l_dict.get(k, 0.0))

                        # 可视化存图
                        img_opt = c_opt[0, ..., :3]
                        img_ref = c_ref[0, ..., :3]
                        img_concat = torch.cat([img_opt, img_ref], dim=1)  # 左右拼接
                        img_concat = util.tonemap_srgb(img_concat)
                        np_img = img_concat.detach().cpu().numpy()
                        save_name = eval_img_dir / f"iter_{iteration:06d}_frame_{frame_id:04d}_{prefix}.png"
                        util.save_image(str(save_name), np_img)

                        # -----------------------------------------------------------
                        # 2. Per-submesh Eval
                        # -----------------------------------------------------------
                        for i, name in enumerate(self.bmesh_names):
                            b_mask = [1 if j == i else 0 for j in range(len(self.bmesh_names))]
                            r_mask = [1 if r2b == i else 0 for r2b in self.submesh_sampler.rmesh2bmesh]

                            if sum(r_mask) == 0:
                                continue

                            sub_opt = self.submesh_sampler._extract_submesh(opt_bmesh_skinned, b_mask, self.submesh_sampler.bmesh_f_offset)
                            sub_ref = self.submesh_sampler._extract_submesh(ref_mesh, r_mask, self.submesh_sampler.rmesh_f_offset)

                            if not self._mesh_has_faces(sub_opt) or not self._mesh_has_faces(sub_ref):
                                continue

                            c_opt_sub = self.renderer.forward_render(sub_opt, mvp, campos, lightpos, self.config.train_res, self.config.spp, bg)
                            c_ref_sub = self.renderer.forward_render(sub_ref, mvp, campos, lightpos, self.config.train_res, self.config.spp, bg)

                            _, l_dict_sub = self._chunked_image_loss(c_opt_sub, c_ref_sub, None, eval_mode=True, chunk_size=4)

                            sub_opt_v = self._used_vertices_for_mesh(sub_opt)
                            sub_ref_v = self._used_vertices_for_mesh(sub_ref)
                            sub_haus = self._calc_hausdorff_gpu(sub_opt_v, sub_ref_v)

                            sub_mse = l_dict_sub.get("mse", 0.0)
                            sub_psnr = -10.0 * math.log10(sub_mse + 1e-8)

                            metrics_collect[name]["psnr"].append(sub_psnr)
                            metrics_collect[name]["hausdorff"].append(sub_haus)
                            for k in ["l1", "mse", "ssim", "lpips"]:
                                metrics_collect[name][k].append(l_dict_sub.get(k, 0.0))

                # 🌟 核心修改 3：计算统计特征并写入 history_metrics
                summary_str = ""
                for part, m_dict in metrics_collect.items():
                    str_psnr, str_lpips, str_haus = "", "", ""
                    for k, v_list in m_dict.items():
                        if len(v_list) == 0:
                            continue

                        arr = np.array(v_list)
                        mean_val = float(np.mean(arr))
                        std_val = float(np.std(arr))
                        max_val = float(np.max(arr))
                        min_val = float(np.min(arr))

                        # 记录 Mean (保持原有 key 兼容性)
                        key_mean = f"history_{prefix}_{part}_{k}"
                        if key_mean not in self.history_metrics: self.history_metrics[key_mean] = []
                        self.history_metrics[key_mean].append(mean_val)

                        # 记录 Std, Max, Min
                        key_std = f"history_{prefix}_{part}_{k}_std"
                        if key_std not in self.history_metrics: self.history_metrics[key_std] = []
                        self.history_metrics[key_std].append(std_val)

                        key_max = f"history_{prefix}_{part}_{k}_max"
                        if key_max not in self.history_metrics: self.history_metrics[key_max] = []
                        self.history_metrics[key_max].append(max_val)

                        key_min = f"history_{prefix}_{part}_{k}_min"
                        if key_min not in self.history_metrics: self.history_metrics[key_min] = []
                        self.history_metrics[key_min].append(min_val)

                        # 🌟 优化排版：补充 LPIPS 和 Haus 的 Max/Min 打印，并在 ± 两侧加空格
                        if part == "overall":
                            if k == "psnr":
                                str_psnr = f"PSNR: {mean_val:.2f} ± {std_val:.2f} (Max: {max_val:.2f}, Min: {min_val:.2f})"
                            elif k == "lpips":
                                str_lpips = f"LPIPS: {mean_val:.4f} ± {std_val:.4f} (Max: {max_val:.4f}, Min: {min_val:.4f})"
                            elif k == "hausdorff":
                                str_haus = f"Haus: {mean_val:.4f} ± {std_val:.4f} (Max: {max_val:.4f}, Min: {min_val:.4f})"

                    if part == "overall":
                        summary_str = f"{str_psnr} | {str_lpips} | {str_haus}"

                log(f"[Deep Eval] {prefix.capitalize():<5} Overall -> {summary_str}")

            _evaluate_set(self.train_frames, "train")
            _evaluate_set(self.test_frames, "test")

        finally:
            # 3. 恢复相机配置，不影响后续训练
            self.config.enable_auto_center = orig_auto_center
            self.config.cam_jitter_pixels = orig_jitter
            self.config.cam_zoom_max = orig_zoom

    def _save_preview(self, iteration: int):
        rest_pose_end_iter = int(self.config.iter * self.config.rest_pose_ratio)
        frame_id = 0 if iteration < rest_pose_end_iter else iteration % self.poses_count

        body_mask_phase = self._is_body_mask_phase(iteration)

        if body_mask_phase:
            opt_bmesh_soft = self._build_opt_mesh(body_alpha_mode="soft")
            opt_bmesh_hard = self._build_opt_mesh(body_alpha_mode="hard")

            opt_bmesh_soft_skinned = mesh.skinning(opt_bmesh_soft).eval({
                "time": frame_id,
                "weight_mask": self.vbw_mask
            })
            opt_bmesh_hard_skinned = mesh.skinning(opt_bmesh_hard).eval({
                "time": frame_id,
                "weight_mask": self.vbw_mask
            })
            ref_mesh_base = self.ref_mesh_sequence[frame_id].clone()

            opt_bmesh_soft_skinned, ref_mesh_soft = self.submesh_sampler.apply(
                iteration,
                opt_bmesh_soft_skinned,
                ref_mesh_base.clone()
            )
            opt_bmesh_hard_skinned, ref_mesh_hard = self.submesh_sampler.apply(
                iteration,
                opt_bmesh_hard_skinned,
                ref_mesh_base.clone()
            )
            opt_bmesh_soft_skinned, ref_mesh_soft = self.color_palette.apply(
                iteration,
                opt_bmesh_soft_skinned,
                ref_mesh_soft
            )
            opt_bmesh_hard_skinned, ref_mesh_hard = self.color_palette.apply(
                iteration,
                opt_bmesh_hard_skinned,
                ref_mesh_hard
            )
        else:
            opt_bmesh = self._build_opt_mesh(body_alpha_mode="none")
            opt_bmesh_skinned = mesh.skinning(opt_bmesh).eval({
                "time": frame_id,
                "weight_mask": self.vbw_mask
            })
            ref_mesh = self.ref_mesh_sequence[frame_id].clone()
            opt_bmesh_skinned, ref_mesh = self.submesh_sampler.apply(
                iteration,
                opt_bmesh_skinned,
                ref_mesh
            )
            opt_bmesh_skinned, ref_mesh = self.color_palette.apply(
                iteration,
                opt_bmesh_skinned,
                ref_mesh
            )

        focus_end_iter = int(self.config.iter * self.config.cam_focus_ratio)
        raw_mode = "preview_raw_focus" if (self.config.cam_focus_mode and iteration < focus_end_iter) else "preview_raw_orbit"

        preview_modes = ["preview_fixed", "preview_orbit", raw_mode]
        row_images = []

        with torch.no_grad():
            for mode in preview_modes:
                if body_mask_phase:
                    mvp, campos, lightpos = camera.get_camera_params(
                        self.config,
                        opt_bmesh_soft_skinned,
                        ref_mesh_soft,
                        mode=mode
                    )
                    # 🌟 开启 RGBA 渲染
                    color_soft = self.renderer.forward_render(
                        opt_bmesh_soft_skinned, mvp, campos, lightpos,
                        self.config.display_res, self.config.spp, background=None, return_alpha=True
                    )
                    color_hard = self.renderer.forward_render(
                        opt_bmesh_hard_skinned, mvp, campos, lightpos,
                        self.config.display_res, self.config.spp, background=None, return_alpha=True
                    )
                    color_ref = self.renderer.forward_render(
                        ref_mesh_soft, mvp, campos, lightpos,
                        self.config.display_res, self.config.spp, background=None, return_alpha=True
                    )

                    img_soft = util.scale_img_nhwc(color_soft, [self.config.display_res, self.config.display_res])
                    img_hard = util.scale_img_nhwc(color_hard, [self.config.display_res, self.config.display_res])
                    img_ref = util.scale_img_nhwc(color_ref, [self.config.display_res, self.config.display_res])

                    debug_img_tensor = self._make_debug_view_tensor(
                        mesh_obj=opt_bmesh_soft_skinned,
                        mvp=mvp,
                        frame_id=frame_id
                    )
                    # 给 debug RGB 图像补充 Alpha 通道 (全不透明)
                    alpha_channel = torch.ones_like(debug_img_tensor[..., 0:1])
                    debug_img_rgba = torch.cat([debug_img_tensor, alpha_channel], dim=-1)

                    row_img = torch.cat(
                        [img_soft, img_hard, debug_img_rgba, img_ref],
                        dim=2
                    )
                else:
                    mvp, campos, lightpos = camera.get_camera_params(
                        self.config,
                        opt_bmesh_skinned,
                        ref_mesh,
                        mode=mode
                    )
                    # 🌟 开启 RGBA 渲染
                    color_opt = self.renderer.forward_render(
                        opt_bmesh_skinned, mvp, campos, lightpos,
                        self.config.display_res, self.config.spp, background=None, return_alpha=True
                    )
                    color_ref = self.renderer.forward_render(
                        ref_mesh, mvp, campos, lightpos,
                        self.config.display_res, self.config.spp, background=None, return_alpha=True
                    )

                    img_opt = util.scale_img_nhwc(color_opt, [self.config.display_res, self.config.display_res])
                    img_ref = util.scale_img_nhwc(color_ref, [self.config.display_res, self.config.display_res])

                    debug_img_tensor = self._make_debug_view_tensor(
                        mesh_obj=opt_bmesh_skinned,
                        mvp=mvp,
                        frame_id=frame_id
                    )
                    # 给 debug RGB 图像补充 Alpha 通道 (全不透明)
                    alpha_channel = torch.ones_like(debug_img_tensor[..., 0:1])
                    debug_img_rgba = torch.cat([debug_img_tensor, alpha_channel], dim=-1)

                    row_img = torch.cat(
                        [img_opt, debug_img_rgba, img_ref],
                        dim=2
                    )

                row_images.append(row_img)

            # 拼接所有行 [1, H_total, W_total, 4]
            result_rgba = torch.cat(row_images, dim=1)

            # 1. 生成透明背景版本
            # 仅对 RGB 通道做 tonemap，保持 Alpha 不变
            result_rgba[0, ..., :3] = util.tonemap_srgb(result_rgba[0, ..., :3])
            np_result_transparent = result_rgba[0].detach().cpu().numpy()

            # 2. 生成格子背景版本 (Alpha Blending)
            rgb = result_rgba[..., :3]
            alpha = result_rgba[..., 3:4]

            # 🌟 修复：同时在高度 (Y) 和宽度 (X) 方向平铺格子背景
            num_imgs_y = result_rgba.shape[1] // self.config.display_res
            num_imgs_x = result_rgba.shape[2] // self.config.display_res
            bg_tiled = self.preview_background.repeat(1, num_imgs_y, num_imgs_x, 1)

            result_rgb_blended = rgb * alpha + bg_tiled * (1.0 - alpha)
            np_result_checkerboard = result_rgb_blended[0].detach().cpu().numpy()

            # 保存双版本
            trans_dir = Path(self.config.run_dir) / "previews_transparent"
            check_dir = Path(self.config.run_dir) / "previews_checkerboard"
            trans_dir.mkdir(parents=True, exist_ok=True)
            check_dir.mkdir(parents=True, exist_ok=True)

            save_name = f"preview_body_mask_iter_{iteration:06d}.png" if body_mask_phase else f"preview_iter_{iteration:06d}.png"

            util.save_image(str(trans_dir / save_name), np_result_transparent)
            util.save_image(str(check_dir / save_name), np_result_checkerboard)

    def _save_per_mesh_textures(self, save_name: str):
        if not hasattr(self, "kd_map_opt") or "kd" in self.config.skip_train:
            return

        tex_dir = Path(self.config.run_dir) / "elod_mesh" / "textures"
        tex_dir.mkdir(parents=True, exist_ok=True)

        actual_res_y = min(self.config.texture_res[0], 1024)
        actual_res_x = min(self.config.texture_res[1], 1024)
        kd_data = self.kd_map_opt.data[0]

        for i, name in enumerate(self.bmesh_names):
            start_x, end_x = i * actual_res_x, (i + 1) * actual_res_x
            single_tex = kd_data[:actual_res_y, start_x:end_x, :]
            single_tex_obj = texture.Texture2D(single_tex.unsqueeze(0))

            # 🌟 扁平化：带前缀保存
            save_path = tex_dir / f"{name}_{save_name}"
            texture.save_texture2D(str(save_path), texture.rgb_to_srgb(single_tex_obj))

        log(f"已成功将图集拆解并保存逐网格贴图: {save_name} (位于 textures 目录)")

    def _save_final_mesh(self):
        mesh_dir = Path(self.config.run_dir) / "mesh"
        elod_dir = Path(self.config.run_dir) / "elod_mesh"
        geom_dir = elod_dir / "geometry"

        mesh_dir.mkdir(parents=True, exist_ok=True)
        geom_dir.mkdir(parents=True, exist_ok=True)

        opt_bmesh = self._build_opt_mesh(body_alpha_mode="none")
        obj.write_obj(str(mesh_dir), opt_bmesh.eval())

        # 🌟 扁平化：几何文件存入 geometry
        self.loader.save_base_mesh(
            geom_dir,
            opt_bmesh.eval(),
            self.bmesh_names,
            self.bmesh_v_offset,
            self.bmesh_f_offset,
            self.face_uvs
        )
        self._save_per_mesh_textures("texture_kd.png")

        if self.body_alpha_mask is not None and self.body_alpha_mask.active:
            actual_res_y, actual_res_x = min(self.config.texture_res[0], 1024), min(self.config.texture_res[1], 1024)
            # JSON 存入 geometry
            self.body_alpha_mask.save_bone_alpha_json(geom_dir)
            # Masks 存入 masks 文件夹
            self.body_alpha_mask.bake_and_save_masks(
                ctx=self.renderer.glctx,
                out_root=elod_dir,
                res_y=actual_res_y,
                res_x=actual_res_x
            )

        # 🌟 扁平化：元数据存入 geometry
        for meta_stem in ["animation_data", "skeleton_hierarchy_data"]:
            src_file = resolve_specific_file(self.config.source_dir, meta_stem, [".pkl", ".pickle", ".json"])
            if src_file:
                shutil.copy2(src_file, geom_dir / src_file.name)

        log(f"最终 3D 资产已保存至: {self.config.run_dir}")

    def run(self):
        body_alpha_opt_iter = getattr(self.config, 'body_alpha_opt_iter', 0)
        total_iters = self.config.iter + body_alpha_opt_iter
        log(f"开始训练管线，主阶段: {self.config.iter} iter, 追加阶段: {body_alpha_opt_iter} iter, 总计: {total_iters} iter")

        sum_total_loss, sum_img_loss, sum_lap_loss = 0.0, 0.0, 0.0
        interval_start_time = time.perf_counter()
        bake_switch_iter = int(self.config.iter * getattr(self.config, 'bake_isolated_ratio', 0.5))

        for it in range(total_iters + 1):
            iter_start_time = time.perf_counter()
            total_loss, img_loss, lap_loss = self.train_step(it)

            sum_total_loss += total_loss
            sum_img_loss += img_loss
            sum_lap_loss += lap_loss

            if it == total_iters:
                self.final_metrics = {
                    "final_total_loss": total_loss,
                    "final_img_loss": img_loss,
                    "final_lap_loss": lap_loss * self.lap_fac
                }
                for k, v in self._current_loss_dict.items():
                    self.final_metrics[f"final_{k}_loss"] = v

                if self.test_frames:
                    test_img, test_dict = self.fast_eval_step(it)
                    self.final_metrics["final_test_img_loss"] = test_img
                    for k, v in test_dict.items():
                        self.final_metrics[f"final_test_{k}_loss"] = v

                if self.config.eval_interval > 0:
                    self.deep_eval_step(it)
                    for k, v_list in self.history_metrics.items():
                        final_key = k.replace("history_", "final_")
                        self.final_metrics[final_key] = v_list[-1]
                        self.final_metrics[k] = v_list

            if it % self.config.print_interval == 0:
                lr_text = self._get_lr_log_text()

                avg_total = total_loss if it == 0 else sum_total_loss / self.config.print_interval
                avg_img = img_loss if it == 0 else sum_img_loss / self.config.print_interval
                avg_lap = (lap_loss * self.lap_fac) if it == 0 else (sum_lap_loss / self.config.print_interval) * self.lap_fac
                iter_dur = (time.perf_counter() - iter_start_time) * 1000 if it == 0 else ((time.perf_counter() - interval_start_time) * 1000) / self.config.print_interval

                train_loss_str = self._get_loss_str(self._current_loss_dict)
                train_img_str = f"{avg_img:.4f}" + (f" ({train_loss_str})" if train_loss_str else "")
                train_tot = avg_img + avg_lap

                test_str = ""
                if self.test_frames:
                    test_img, test_dict = self.fast_eval_step(it)
                    test_loss_str = self._get_loss_str(test_dict)
                    test_img_str = f"{test_img:.4f}" + (f" ({test_loss_str})" if test_loss_str else "")
                    test_tot = test_img + avg_lap
                    test_str = f"\n  └─ Test  : Tot:{test_tot:.4f} | Img:{test_img_str}"
                    train_prefix = "  ├─ Train :"
                else:
                    train_prefix = "  └─ Train :"

                log(f"Iter: {it:05d} | Lap: {avg_lap:.4f} | LR: [{lr_text}] | {iter_dur:.1f}ms/iter\n"
                    f"{train_prefix} Tot:{train_tot:.4f} | Img:{train_img_str}{test_str}")

                if it != 0:
                    sum_total_loss, sum_img_loss, sum_lap_loss = 0.0, 0.0, 0.0
                    interval_start_time = time.perf_counter()

            if self.config.eval_interval > 0 and it > 0 and it % self.config.eval_interval == 0 and it != total_iters:
                self.deep_eval_step(it)

            if self.config.save_interval > 0 and it % self.config.save_interval == 0:
                log(f"保存预览图 (Iter {it})...")
                self._save_preview(it)

            if self.config.submesh_mode == "bake" and "kd" not in self.config.skip_train and it == bake_switch_iter:
                log(f"达到二段式烘焙切换点 (Iter {it})，保存阶段一漫反射贴图...")
                self._save_per_mesh_textures("texture_kd_stage1.png")

        self._save_final_mesh()
        log("训练管线执行完毕！")
