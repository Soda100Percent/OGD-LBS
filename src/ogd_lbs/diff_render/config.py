# filepath: src/ogd_lbs/diff_render/config.py
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
from pathlib import Path

from ogd_lbs.utils.common import load_json, dump_json, log, warn


@dataclass
class DiffRenderConfig:
    """可微渲染管线全局配置类"""

    # --- 基础配置 ---
    out_dir: str = "out/default"
    run_dir: str = ""
    iter: int = 5000
    batch: int = 1

    # --- 学习率与迭代控制 ---
    learning_rate: float = 0.01
    tex_learning_rate: float = 0.01
    body_alpha_opt_iter: int = 0
    body_alpha_opt_lr: float = 0.03

    # --- 渲染与贴图配置 ---
    train_res: int = 512
    display_res: Optional[int] = None
    texture_res: Tuple[int, int] = (1024, 1024)
    layers: int = 1  # 运行时会被 geom_opt_layers 或 tex_opt_layers 覆盖
    geom_opt_layers: int = 2
    tex_opt_layers: int = 3
    spp: int = 1
    random_train_res: bool = False
    custom_mip: bool = False
    random_textures: bool = False

    # --- 优化控制与复合 Loss 权重 ---
    skip_train: List[str] = field(default_factory=list)
    loss: str = "mse"  # 遗留参数，现由复合 Loss 替代

    loss_l1_weight: float = 0.0
    loss_mse_weight: float = 1.0
    loss_ssim_weight: float = 0.0
    loss_lpips_weight: float = 0.0

    # --- 物理与材质参数 ---
    light_power: float = 5.0
    min_roughness: float = 0.08
    displacement: float = 0.15
    opt_bsdf: str = "diffuse"
    ref_bsdf: str = "diffuse"

    # Stage02 / texture optimization BSDF override.
    # 可选值：
    #   none
    #   ambient_2sided
    #   2sided 作为 ambient_2sided 的别名
    tex_opt_bsdf_override: str = "none"

    # 🌟 几何阶段/调试阶段可选：对背面 diffuse 颜色加深。
    # 注意：
    # - 这个参数主要影响 opt mesh。
    # - 在 two-stage 的 Stage01 几何优化阶段会被脚本自动打开。
    # - mixed 管线中只对 diffuse / diffuse_1sided / diffuse_2sided 生效。
    enable_back_deepen: bool = False

    # 🌟 混合材质系统：为特定子网格指定 BSDF
    opt_submesh_bsdf: List[List[str]] = field(default_factory=list)
    ref_submesh_bsdf: List[List[str]] = field(default_factory=list)

    # --- 几何与拉普拉斯正则化 ---
    subdivision: int = 0
    relative_laplacian: bool = True
    laplacian_factor: float = 1.0

    # --- 相机配置 ---
    camera_eye: List[float] = field(default_factory=lambda: [0.0, -28.0, 0.0])
    camera_up: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    camera_radius: float = 28.0

    # --- 相机高级策略 ---
    enable_auto_center: bool = True
    cam_zoom_max: float = 0.0
    cam_jitter_pixels: float = 0.0
    cam_focus_mode: bool = False
    cam_focus_offset: float = 2.0
    cam_focus_fluctuation_in: float = 0.2
    cam_focus_fluctuation_out: float = 1.5
    cam_focus_ratio: float = 1.0

    # --- 优化策略控制 ---
    use_color_palette: bool = True
    n_kd_palette: int = 5
    use_submesh_sampling: bool = True
    submesh_mode: str = "geom"
    submesh_sampling_mode: str = "off"  # off, random, anneal
    fixed_submeshes: List[str] = field(default_factory=list)
    use_target_textures: bool = False
    use_angle_weighting: bool = False
    angle_weight_grazing_decay: float = 0.0

    # --- Texture Alpha 二值化退火正则 ---
    texture_alpha_binary_weight: float = 0.0
    texture_alpha_binary_start_ratio: float = 0.3

    # --- Body Alpha Mask ---
    body_mask_submeshes: List[str] = field(default_factory=lambda: ["body"])
    body_alpha_keep_weight: float = 0.01
    body_alpha_binary_weight: float = 0.0
    body_alpha_threshold: float = 0.5
    body_alpha_min: float = 0.0
    body_alpha_max: float = 1.0
    body_mask_force_layers: int = 4

    # --- 阶段比例控制 ---
    rest_pose_ratio: float = 0.1
    bake_isolated_ratio: float = 0.5

    # --- 资产路径 ---
    base_mesh: Optional[str] = None
    ref_mesh: Optional[str] = None
    source_dir: Optional[str] = None
    target_dir: Optional[str] = None

    # 🌟 忽略的网格名称后缀
    ignore_suffixes: List[str] = field(default_factory=list)

    # 🌟 测试帧区间（归一化比例，如 [0.8, 1.0]）
    test_frame_range: List[float] = field(default_factory=list)

    # --- 辅助控制与评估 ---
    save_interval: int = 1000
    display_interval: int = 0
    print_interval: int = 50

    # 🌟 深度评估配置
    eval_interval: int = 0
    eval_frame_max_train_set: int = 5
    eval_frame_max_test_set: int = 5

    @classmethod
    def from_json(cls, json_path: str | Path) -> "DiffRenderConfig":
        data = load_json(json_path)
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {}

        for k, v in data.items():
            if k in valid_keys:
                filtered_data[k] = v
            else:
                warn(f"配置项 '{k}' 在 DiffRenderConfig 中未定义，已自动忽略。")

        config = cls(**filtered_data)

        config.ignore_suffixes = [str(x) for x in config.ignore_suffixes]

        if config.test_frame_range:
            try:
                parsed_range = [float(x) for x in config.test_frame_range]
                if len(parsed_range) == 2:
                    start = max(0.0, min(1.0, parsed_range[0]))
                    end = max(0.0, min(1.0, parsed_range[1]))
                    if start > end:
                        start, end = end, start
                    config.test_frame_range = [start, end]
                else:
                    warn("test_frame_range 必须包含两个浮点数，已重置为空。")
                    config.test_frame_range = []
            except ValueError:
                warn("test_frame_range 解析失败，已重置为空。")
                config.test_frame_range = []

        if config.display_res is None:
            config.display_res = config.train_res

        config.loss_l1_weight = max(0.0, float(config.loss_l1_weight))
        config.loss_mse_weight = max(0.0, float(config.loss_mse_weight))
        config.loss_ssim_weight = max(0.0, float(config.loss_ssim_weight))
        config.loss_lpips_weight = max(0.0, float(config.loss_lpips_weight))

        config.angle_weight_grazing_decay = max(0.0, min(1.0, float(config.angle_weight_grazing_decay)))
        config.texture_alpha_binary_weight = max(0.0, float(config.texture_alpha_binary_weight))
        config.texture_alpha_binary_start_ratio = max(0.0, min(1.0, float(config.texture_alpha_binary_start_ratio)))

        config.body_alpha_keep_weight = max(0.0, float(config.body_alpha_keep_weight))
        config.body_alpha_binary_weight = max(0.0, float(config.body_alpha_binary_weight))
        config.body_alpha_threshold = max(0.0, min(1.0, float(config.body_alpha_threshold)))
        config.body_alpha_min = max(0.0, min(1.0, float(config.body_alpha_min)))
        config.body_alpha_max = max(0.0, min(1.0, float(config.body_alpha_max)))

        if config.body_alpha_min > config.body_alpha_max:
            warn(f"body_alpha_min > body_alpha_max，已交换: {config.body_alpha_min} > {config.body_alpha_max}")
            config.body_alpha_min, config.body_alpha_max = config.body_alpha_max, config.body_alpha_min

        config.body_mask_force_layers = max(1, int(config.body_mask_force_layers))

        config.eval_interval = max(0, int(config.eval_interval))
        config.eval_frame_max_train_set = max(1, int(config.eval_frame_max_train_set))
        config.eval_frame_max_test_set = max(1, int(config.eval_frame_max_test_set))

        config.enable_back_deepen = bool(config.enable_back_deepen)

        config.tex_opt_bsdf_override = str(config.tex_opt_bsdf_override).strip().lower()
        if config.tex_opt_bsdf_override == "2sided":
            config.tex_opt_bsdf_override = "ambient_2sided"
        if config.tex_opt_bsdf_override not in ("none", "ambient_2sided"):
            warn(
                f"tex_opt_bsdf_override='{config.tex_opt_bsdf_override}' 非法，"
                "已回退为 'none'。可用值: none, ambient_2sided"
            )
            config.tex_opt_bsdf_override = "none"

        return config

    def save_snapshot(self, path: str | Path):
        dump_json(path, asdict(self))
        log(f"配置快照已保存至: {path}")
