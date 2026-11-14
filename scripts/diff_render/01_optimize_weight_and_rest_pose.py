# filepath: scripts/diff_render_pipeline/01_optimize_weight_and_rest_pose.py
import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime

import _env  # noqa: F401
from ogd_lbs.diff_render.config import DiffRenderConfig
from ogd_lbs.diff_render.trainer import DifferentiableTrainer
from ogd_lbs.utils.common import log, fail, script_argv, Timer, load_data, dump_pkl, record_metric, dump_metrics
from ogd_lbs.uv_mapping import generate_fitbox_uvs
from ogd_lbs.fitbox.schema import SKELETON_STEM, ANIMATION_STEM
from ogd_lbs.fitbox.validation import validate_fitbox_mesh

def convert_ambient_bsdf_rules_to_diffuse(
    rules,
    enable_back_deepen_for_ambient_1sided: bool
):
    """
    Stage01 几何优化阶段使用。

    opt mesh:
        ambient_1sided -> diffuse_2sided_back_deepen
        ambient_2sided -> diffuse_2sided
        ambient        -> diffuse_2sided

    ref mesh:
        ambient_1sided -> diffuse_2sided
        ambient_2sided -> diffuse_2sided
        ambient        -> diffuse_2sided

    也就是说，ref_mesh 永远不启用 back_deepen。
    """
    if rules in (None, "N/A"):
        return rules

    ambient_1sided_target = (
        "diffuse_2sided_back_deepen"
        if enable_back_deepen_for_ambient_1sided
        else "diffuse_2sided"
    )

    mapping = {
        "ambient_1sided": ambient_1sided_target,
        "ambient_2sided": "diffuse_2sided",
        "ambient": "diffuse_2sided",
    }

    out = []
    for rule in rules:
        if not isinstance(rule, (list, tuple)) or len(rule) != 2:
            out.append(rule)
            continue

        name, bsdf = rule
        bsdf_key = str(bsdf).strip().lower()
        new_bsdf = mapping.get(bsdf_key, bsdf)

        out.append([name, new_bsdf])

    return out

def main():
    parser = argparse.ArgumentParser(description="Stage 01: Optimize Geometry and Bone Weights.")

    parser.add_argument("-c", "--config", type=str, default=None, help="Path to the JSON configuration file.")
    parser.add_argument("-i", "--input_dir", type=str, default=None, help="Input directory containing source/target data.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Base output directory.")
    parser.add_argument("--source_dir", type=str, default=None, help="Direct path to the source directory.")
    parser.add_argument("--target_dir", type=str, default=None, help="Direct path to the target directory.")

    parser.add_argument("--iter", type=int, default=None, help="Main phase iterations.")
    parser.add_argument("--batch", type=int, default=None, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Base learning rate.")
    parser.add_argument("--train_textures", action="store_true", help="Enable texture training in stage 01 (One-Stage mode).")
    parser.add_argument("--tex_learning_rate", type=float, default=None, help="Texture learning rate if train_textures is enabled.")
    parser.add_argument("--layers", type=int, default=None, help="Depth peeling layers.")

    parser.add_argument("--body_alpha_opt_iter", type=int, default=None, help="Additional iterations for body alpha optimization.")
    parser.add_argument("--body_alpha_opt_lr", type=float, default=None, help="Learning rate for body alpha.")

    parser.add_argument("--train_res", type=int, default=None, help="Training resolution.")
    parser.add_argument("--texture_res", type=int, nargs=2, default=None, help="Texture resolution, e.g., 1024 1024.")

    parser.add_argument("--loss_l1_weight", type=float, default=None, help="Weight for L1 Loss.")
    parser.add_argument("--loss_mse_weight", type=float, default=None, help="Weight for MSE Loss.")
    parser.add_argument("--loss_ssim_weight", type=float, default=None, help="Weight for SSIM Loss.")
    parser.add_argument("--loss_lpips_weight", type=float, default=None, help="Weight for LPIPS Loss.")
    parser.add_argument("--laplacian_factor", type=float, default=None, help="Laplacian regularization factor.")
    parser.add_argument("--enable_relative_laplacian", action="store_true", help="Enable relative laplacian.")
    parser.add_argument("--disable_relative_laplacian", action="store_true", help="Disable relative laplacian.")
    parser.add_argument("--light_power", type=float, default=None, help="Light power multiplier.")
    parser.add_argument("--min_roughness", type=float, default=None, help="Minimum roughness.")
    parser.add_argument("--enable_back_deepen", action="store_true", help="Enable back-face darkening for diffuse/mixed diffuse rendering.")

    parser.add_argument("--opt_submesh_bsdf", type=str, default=None, help="JSON string of opt_submesh_bsdf rules.")
    parser.add_argument("--ref_submesh_bsdf", type=str, default=None, help="JSON string of ref_submesh_bsdf rules.")

    parser.add_argument("--random_textures", action="store_true", help="Initialize textures with random noise.")
    parser.add_argument("--disable_random_textures", action="store_true", help="Disable random textures.")

    parser.add_argument("--no_sandbox", action="store_true", help="Disable creating a new run_ timestamp sandbox directory.")
    parser.add_argument("--force_auto_uv", action="store_true", help="Force auto UV generation for source meshes.")
    parser.add_argument("--auto_uv_method", type=str, default="xatlas", help="Algorithm for auto UV generation.")

    parser.add_argument("--submesh_sampling_mode", type=str, default=None, choices=["off", "random", "anneal"], help="Submesh sampling strategy.")
    parser.add_argument("--use_color_palette", action="store_true", help="Enable ID color palette strategy.")
    parser.add_argument("--disable_color_palette", action="store_true", help="Disable ID color palette strategy.")

    parser.add_argument("--enable_auto_center", action="store_true", help="Enable auto center and scale.")
    parser.add_argument("--disable_auto_center", action="store_true", help="Disable auto center and scale.")
    parser.add_argument("--cam_zoom_max", type=float, default=None, help="Max random zoom factor for camera.")
    parser.add_argument("--cam_jitter_pixels", type=float, default=None, help="Sub-pixel jitter intensity.")
    parser.add_argument("--cam_focus_mode", action="store_true", help="Enable surface-based random focus camera mode.")
    parser.add_argument("--cam_focus_offset", type=float, default=None, help="Base normal offset distance for focus camera.")
    parser.add_argument("--cam_focus_fluctuation_in", type=float, default=None, help="Max inward fluctuation ratio.")
    parser.add_argument("--cam_focus_fluctuation_out", type=float, default=None, help="Max outward fluctuation ratio.")
    parser.add_argument("--cam_focus_ratio", type=float, default=None, help="Ratio of iterations to use focus camera mode.")

    parser.add_argument("--rest_pose_ratio", type=float, default=None, help="Ratio of iterations to keep rest pose.")

    parser.add_argument("--enable_angle_weighting", action="store_true", help="Enable two-sided per-pixel angle weighting.")
    parser.add_argument("--disable_angle_weighting", action="store_true", help="Force disable angle weighting.")
    parser.add_argument("--angle_weight_grazing_decay", type=float, default=None, help="Grazing angle decay strength for angle weighting.")

    parser.add_argument("--texture_alpha_binary_weight", type=float, default=None, help="Weight for delayed texture alpha binary regularization.")
    parser.add_argument("--texture_alpha_binary_start_ratio", type=float, default=None, help="Normalized training ratio to start texture alpha binary regularization.")

    parser.add_argument("--body_mask_submeshes", nargs="+", default=None, help="Names of source submeshes treated as body.")
    parser.add_argument("--body_alpha_keep_weight", type=float, default=None, help="Regularization weight encouraging body bone alpha to stay near 1.")
    parser.add_argument("--body_alpha_binary_weight", type=float, default=None, help="Optional binary regularization weight encouraging bone alpha near 0 or 1.")
    parser.add_argument("--body_alpha_threshold", type=float, default=None, help="Threshold for hard body mask preview/bake.")
    parser.add_argument("--body_alpha_min", type=float, default=None, help="Minimum clamp value for body bone alpha.")
    parser.add_argument("--body_alpha_max", type=float, default=None, help="Maximum clamp value for body bone alpha.")
    parser.add_argument("--body_mask_force_layers", type=int, default=None, help="Force minimum depth peeling layers when body mask is enabled.")

    parser.add_argument("--ignore_suffixes", nargs="+", default=None, help="List of suffixes to ignore when matching mesh names.")
    parser.add_argument("--test_frame_range", type=float, nargs=2, default=None, help="Normalized range [start, end] for test frames, e.g., 0.8 1.0.")

    parser.add_argument("--save_interval", type=int, default=None, help="Override the save interval for quick testing.")
    parser.add_argument("--print_interval", type=int, default=None, help="Override the print interval for loss averaging.")

    parser.add_argument("--eval_interval", type=int, default=None, help="Interval for deep evaluation (0 to disable).")
    parser.add_argument("--eval_frame_max_train_set", type=int, default=None, help="Max frames to sample from train set for deep evaluation.")
    parser.add_argument("--eval_frame_max_test_set", type=int, default=None, help="Max frames to sample from test set for deep evaluation.")

    parser.add_argument("--enable_profiling", action="store_true", help="Enable performance profiling Timer.")

    args = parser.parse_args(script_argv())

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists(): fail(f"找不到配置文件: {config_path.absolute()}")
        config = DiffRenderConfig.from_json(config_path)
    else:
        config = DiffRenderConfig()

    if args.output_dir: config.out_dir = args.output_dir
    if args.input_dir:
        input_path = Path(args.input_dir)
        config.source_dir = str(input_path / "source")
        config.target_dir = str(input_path / "target")
    if args.source_dir: config.source_dir = args.source_dir
    if args.target_dir: config.target_dir = args.target_dir

    if args.iter is not None: config.iter = args.iter
    if args.batch is not None: config.batch = args.batch
    if args.learning_rate is not None: config.learning_rate = args.learning_rate
    if args.tex_learning_rate is not None: config.tex_learning_rate = args.tex_learning_rate
    if args.layers is not None: config.layers = args.layers
    if args.body_alpha_opt_iter is not None: config.body_alpha_opt_iter = args.body_alpha_opt_iter
    if args.body_alpha_opt_lr is not None: config.body_alpha_opt_lr = args.body_alpha_opt_lr

    if args.train_res is not None:
        config.train_res = args.train_res
        if config.display_res is None: config.display_res = args.train_res
    if args.texture_res is not None: config.texture_res = tuple(args.texture_res)

    if args.loss_l1_weight is not None: config.loss_l1_weight = max(0.0, args.loss_l1_weight)
    if args.loss_mse_weight is not None: config.loss_mse_weight = max(0.0, args.loss_mse_weight)
    if args.loss_ssim_weight is not None: config.loss_ssim_weight = max(0.0, args.loss_ssim_weight)
    if args.loss_lpips_weight is not None: config.loss_lpips_weight = max(0.0, args.loss_lpips_weight)
    if args.laplacian_factor is not None: config.laplacian_factor = args.laplacian_factor
    if args.enable_relative_laplacian: config.relative_laplacian = True
    elif args.disable_relative_laplacian: config.relative_laplacian = False
    if args.light_power is not None: config.light_power = max(0.0, args.light_power)
    if args.min_roughness is not None: config.min_roughness = max(0.0, min(1.0, args.min_roughness))
    if args.enable_back_deepen: config.enable_back_deepen = True

    if args.opt_submesh_bsdf is not None: config.opt_submesh_bsdf = json.loads(args.opt_submesh_bsdf)
    if args.ref_submesh_bsdf is not None: config.ref_submesh_bsdf = json.loads(args.ref_submesh_bsdf)

    if args.random_textures: config.random_textures = True
    elif args.disable_random_textures: config.random_textures = False

    if args.submesh_sampling_mode is not None: config.submesh_sampling_mode = args.submesh_sampling_mode

    if args.use_color_palette: config.use_color_palette = True
    elif args.disable_color_palette: config.use_color_palette = False

    if args.enable_auto_center: config.enable_auto_center = True
    elif args.disable_auto_center: config.enable_auto_center = False
    if args.cam_zoom_max is not None: config.cam_zoom_max = args.cam_zoom_max
    if args.cam_jitter_pixels is not None: config.cam_jitter_pixels = args.cam_jitter_pixels
    if args.cam_focus_mode: config.cam_focus_mode = True
    if args.cam_focus_offset is not None: config.cam_focus_offset = args.cam_focus_offset
    if args.cam_focus_fluctuation_in is not None: config.cam_focus_fluctuation_in = args.cam_focus_fluctuation_in
    if args.cam_focus_fluctuation_out is not None: config.cam_focus_fluctuation_out = args.cam_focus_fluctuation_out
    if args.cam_focus_ratio is not None: config.cam_focus_ratio = args.cam_focus_ratio
    if args.rest_pose_ratio is not None: config.rest_pose_ratio = args.rest_pose_ratio

    if args.enable_angle_weighting: config.use_angle_weighting = True
    elif args.disable_angle_weighting: config.use_angle_weighting = False
    if args.angle_weight_grazing_decay is not None: config.angle_weight_grazing_decay = max(0.0, min(1.0, args.angle_weight_grazing_decay))

    if args.texture_alpha_binary_weight is not None: config.texture_alpha_binary_weight = max(0.0, args.texture_alpha_binary_weight)
    if args.texture_alpha_binary_start_ratio is not None: config.texture_alpha_binary_start_ratio = max(0.0, min(1.0, args.texture_alpha_binary_start_ratio))

    if args.body_mask_submeshes is not None: config.body_mask_submeshes = args.body_mask_submeshes
    if args.body_alpha_keep_weight is not None: config.body_alpha_keep_weight = max(0.0, args.body_alpha_keep_weight)
    if args.body_alpha_binary_weight is not None: config.body_alpha_binary_weight = max(0.0, args.body_alpha_binary_weight)
    if args.body_alpha_threshold is not None: config.body_alpha_threshold = max(0.0, min(1.0, args.body_alpha_threshold))
    if args.body_alpha_min is not None: config.body_alpha_min = max(0.0, min(1.0, args.body_alpha_min))
    if args.body_alpha_max is not None: config.body_alpha_max = max(0.0, min(1.0, args.body_alpha_max))
    if args.body_mask_force_layers is not None: config.body_mask_force_layers = max(1, args.body_mask_force_layers)

    if args.ignore_suffixes is not None: config.ignore_suffixes = args.ignore_suffixes
    if args.test_frame_range is not None: config.test_frame_range = args.test_frame_range

    if args.save_interval is not None: config.save_interval = args.save_interval
    if args.print_interval is not None: config.print_interval = args.print_interval

    if args.eval_interval is not None: config.eval_interval = args.eval_interval
    if args.eval_frame_max_train_set is not None: config.eval_frame_max_train_set = args.eval_frame_max_train_set
    if args.eval_frame_max_test_set is not None: config.eval_frame_max_test_set = args.eval_frame_max_test_set

    if not config.source_dir or not config.target_dir: fail("必须指定输入数据路径！")
    if not config.out_dir: fail("必须指定输出路径！")

    if args.train_textures:
        config.skip_train = ["normal", "displacement"]
        config.use_target_textures = True
        log("已开启贴图训练模式 (One-Stage)。")
    else:
        config.skip_train = ["kd", "ks", "normal", "displacement"]

        # 🌟 two-stage Stage01 / 几何优化阶段：
        #
        # 本轮转换策略：
        #   opt mesh:
        #       ambient_1sided -> diffuse_2sided_back_deepen
        #       ambient_2sided -> diffuse_2sided
        #       ambient        -> diffuse_2sided
        #   ref mesh:
        #       ambient_1sided -> diffuse_2sided
        #       ambient_2sided -> diffuse_2sided
        #       ambient        -> diffuse_2sided
        #
        # 关键点：
        #   - 不再使用 diffuse_1sided。
        #   - 不再全局开启 config.enable_back_deepen。
        #   - opt mesh 背面加深由 BSDF ID=5 逐面携带。
        #   - ref mesh 永远不开启背面加深。
        config.opt_submesh_bsdf = convert_ambient_bsdf_rules_to_diffuse(
            config.opt_submesh_bsdf,
            enable_back_deepen_for_ambient_1sided=True
        )
        config.ref_submesh_bsdf = convert_ambient_bsdf_rules_to_diffuse(
            config.ref_submesh_bsdf,
            enable_back_deepen_for_ambient_1sided=False
        )

        # 🌟 重要：本轮不再使用全局 back_deepen。
        # per-face back_deepen 由 diffuse_2sided_back_deepen / BSDF ID=5 表达。
        config.enable_back_deepen = False

        log(
            "仅优化几何，跳过贴图训练。"
            "opt: ambient_1sided -> diffuse_2sided_back_deepen, "
            "ambient_2sided/ambient -> diffuse_2sided；"
            "ref: ambient/ambient_1sided/ambient_2sided -> diffuse_2sided。"
            "全局 enable_back_deepen 已关闭。"
        )

    if args.no_sandbox:
        base_run_dir = Path(config.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_run_dir = Path(config.out_dir) / f"run_{timestamp}"

    preprocess_dir = base_run_dir / "01_preprocess"
    optimize_dir = base_run_dir / "02_optimize"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    optimize_dir.mkdir(parents=True, exist_ok=True)

    if args.force_auto_uv:
        log(f"正在为 Source 网格强制生成 UV (算法: {args.auto_uv_method})...")
        uv_mapped_dir = preprocess_dir / "uv_mapped"
        uv_mapped_dir.mkdir(parents=True, exist_ok=True)
        valid_exts = {".json", ".pkl", ".pickle"}
        source_path = Path(config.source_dir)

        with Timer("time_stage01_auto_uv"):
            for item in source_path.iterdir():
                if item.is_file() and item.suffix.lower() in valid_exts:
                    stem = item.stem
                    if stem in [SKELETON_STEM, ANIMATION_STEM, f"{SKELETON_STEM}_bone_reduced", f"{ANIMATION_STEM}_bone_reduced"]:
                        shutil.copy2(item, uv_mapped_dir / item.name)
                    else:
                        mesh_obj = load_data(item)
                        mesh_obj = generate_fitbox_uvs(mesh_obj, method=args.auto_uv_method, overwrite=True)
                        validate_fitbox_mesh(mesh_obj, name=mesh_obj.get("name", stem))
                        dump_pkl(uv_mapped_dir / f"{stem}.pkl", mesh_obj)
        config.source_dir = str(uv_mapped_dir)

    config.run_dir = str(optimize_dir)
    config.save_snapshot(base_run_dir / "config_run.json")

    with Timer("time_stage01_total"):
        trainer = DifferentiableTrainer(config)
        trainer.run()

    for k, v in trainer.final_metrics.items():
        record_metric(f"stage01_{k}", v)
    dump_metrics(optimize_dir / "metrics_01.json")

if __name__ == "__main__":
    main()
