#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Differentiable Rendering Stage Entrypoint for pd-rew-v2 & OGD-LBS.

Supports both:
  - pipeline_mode="one_stage" (joint geometry + weight + texture optimization)
  - pipeline_mode="two_stage" (Stage 01 geometry/weights -> Stage 02 texture baking + optional body alpha mask)

Input contract:
  --input_dir <work_dir>
      If <work_dir>/diff_render_00/{source,target} exists (pd-rew-v2 staged mode):
          source_dir = <work_dir>/diff_render_00/source
          target_dir = <work_dir>/diff_render_00/target
          output_dir = <work_dir>/data5 (unless --output_dir is explicitly provided)
      elif <work_dir>/{source,target} exists:
          source_dir = <work_dir>/source
          target_dir = <work_dir>/target
          output_dir = <work_dir>/data5 (unless --output_dir is explicitly provided)

Output contract (<output_dir>, default: <work_dir>/data5):
  ├── stage01/
  ├── stage02/ (when pipeline_mode == "two_stage")
  ├── opt_out/ (final Fitbox-compatible geometry *.pkl, animation_data, skeleton_hierarchy_data, textures/, masks/)
  └── step05_diff_render_summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import _env  # noqa: F401

from ogd_lbs.utils.common import dump_json, fail, load_json, log, script_argv

VALID_ARGS_01 = {
    "iter", "learning_rate", "tex_learning_rate", "layers", "body_alpha_opt_iter", "body_alpha_opt_lr",
    "train_res", "texture_res", "batch",
    "loss_l1_weight", "loss_mse_weight", "loss_ssim_weight", "loss_lpips_weight",
    "laplacian_factor", "relative_laplacian",
    "light_power", "min_roughness", "enable_back_deepen",
    "opt_submesh_bsdf", "ref_submesh_bsdf",
    "random_textures",
    "force_auto_uv", "auto_uv_method", "submesh_sampling_mode", "use_color_palette",
    "enable_auto_center", "cam_zoom_max", "cam_jitter_pixels",
    "cam_focus_mode", "cam_focus_offset", "cam_focus_fluctuation_in", "cam_focus_fluctuation_out", "cam_focus_ratio",
    "rest_pose_ratio", "enable_angle_weighting", "angle_weight_grazing_decay",
    "texture_alpha_binary_weight", "texture_alpha_binary_start_ratio",
    "body_mask_submeshes", "body_alpha_keep_weight", "body_alpha_binary_weight",
    "body_alpha_threshold", "body_alpha_min", "body_alpha_max", "body_mask_force_layers",
    "ignore_suffixes", "test_frame_range",
    "eval_interval", "eval_frame_max_train_set", "eval_frame_max_test_set",
    "save_interval", "print_interval", "enable_profiling", "train_textures",
}

VALID_ARGS_02 = {
    "iter", "learning_rate", "tex_learning_rate", "layers", "body_alpha_opt_iter", "body_alpha_opt_lr",
    "train_res", "texture_res", "batch",
    "loss_l1_weight", "loss_mse_weight", "loss_ssim_weight", "loss_lpips_weight",
    "laplacian_factor", "relative_laplacian",
    "light_power", "min_roughness", "tex_opt_bsdf_override",
    "opt_submesh_bsdf", "ref_submesh_bsdf",
    "random_textures",
    "enable_auto_center", "cam_zoom_max", "cam_jitter_pixels",
    "cam_focus_mode", "cam_focus_offset", "cam_focus_fluctuation_in", "cam_focus_fluctuation_out", "cam_focus_ratio",
    "rest_pose_ratio", "bake_isolated_ratio", "enable_angle_weighting", "angle_weight_grazing_decay",
    "texture_alpha_binary_weight", "texture_alpha_binary_start_ratio",
    "body_mask_submeshes", "body_alpha_keep_weight", "body_alpha_binary_weight",
    "body_alpha_threshold", "body_alpha_min", "body_alpha_max", "body_mask_force_layers",
    "ignore_suffixes", "test_frame_range",
    "eval_interval", "eval_frame_max_train_set", "eval_frame_max_test_set",
    "save_interval", "print_interval", "enable_profiling",
}


def _parse_bool(val: str | bool) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {val!r}")


def _parse_json_or_list(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (list, tuple, dict)):
        return raw
    s = str(raw).strip()
    if not s or s == "N/A":
        return default
    if s.startswith("[") or s.startswith("{"):
        return json.loads(s)
    return [x.strip() for x in s.split(",") if x.strip()]


def convert_ambient_bsdf_rules_to_diffuse(
    rules: Any,
    enable_back_deepen_for_ambient_1sided: bool,
) -> Any:
    if rules in (None, "N/A", []):
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
        out.append([name, mapping.get(bsdf_key, bsdf)])
    return out


def _run_sub_stage(script_name: str, args_dict: dict[str, Any], gpu_id: str | None = None) -> None:
    script_path = Path(__file__).resolve().parent / script_name
    cmd = [sys.executable, str(script_path)]

    for k, v in args_dict.items():
        if v in (None, "N/A"):
            continue
        if isinstance(v, bool):
            if k == "use_color_palette":
                cmd.append("--use_color_palette" if v else "--disable_color_palette")
            elif k == "relative_laplacian":
                cmd.append("--enable_relative_laplacian" if v else "--disable_relative_laplacian")
            elif k == "random_textures":
                cmd.append("--random_textures" if v else "--disable_random_textures")
            elif k in {"enable_auto_center", "enable_angle_weighting"}:
                cmd.append(f"--{k}" if v else f"--disable_{k.replace('enable_', '')}")
            else:
                if v:
                    cmd.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            if len(v) == 0:
                continue
            if k in {"opt_submesh_bsdf", "ref_submesh_bsdf"}:
                cmd.extend([f"--{k}", json.dumps(v)])
            else:
                cmd.append(f"--{k}")
                cmd.extend([str(x) for x in v])
        else:
            cmd.extend([f"--{k}", str(v)])

    log(f"Running sub-stage {script_name}: {' '.join(cmd[2:])}")
    env = os.environ.copy()
    if gpu_id is not None and str(gpu_id).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id).strip()

    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        fail(f"{script_name} exited with non-zero code: {proc.returncode}")


def _resolve_io_dirs(
    input_dir_arg: str,
    output_dir_arg: str | None,
    source_dir_arg: str | None,
    target_dir_arg: str | None,
    input_subdir: str = "diff_render_00",
    output_subdir: str = "data5",
) -> tuple[Path, Path, Path]:
    in_root = Path(input_dir_arg).expanduser().resolve(strict=False)
    if source_dir_arg and target_dir_arg:
        src_dir = Path(source_dir_arg).expanduser().resolve(strict=True)
        tgt_dir = Path(target_dir_arg).expanduser().resolve(strict=True)
    elif (in_root / input_subdir / "source").is_dir() and (in_root / input_subdir / "target").is_dir():
        src_dir = (in_root / input_subdir / "source").resolve(strict=True)
        tgt_dir = (in_root / input_subdir / "target").resolve(strict=True)
    elif (in_root / "source").is_dir() and (in_root / "target").is_dir():
        src_dir = (in_root / "source").resolve(strict=True)
        tgt_dir = (in_root / "target").resolve(strict=True)
    else:
        raise FileNotFoundError(
            f"Could not locate source/ and target/ under {in_root} or {in_root / input_subdir}"
        )

    if output_dir_arg:
        out_dir = Path(output_dir_arg).expanduser().resolve(strict=False)
    elif in_root.name == input_subdir:
        out_dir = in_root.parent / output_subdir
    else:
        out_dir = in_root / output_subdir

    return src_dir, tgt_dir, out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified Differentiable Rendering Stage (One-Stage / Two-Stage).")
    p.add_argument("-i", "--input_dir", required=True, help="Work directory or stage input directory.")
    p.add_argument("-o", "--output_dir", default=None, help="Explicit output directory (default: <work_dir>/data5).")
    p.add_argument("--source_dir", default=None, help="Explicit source mesh directory override.")
    p.add_argument("--target_dir", default=None, help="Explicit target mesh/texture directory override.")
    p.add_argument("--input_subdir", default="diff_render_00", help="Upstream handoff subdir name.")
    p.add_argument("--output_subdir", default="data5", help="Stage output subdir name.")
    p.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")

    p.add_argument("--pipeline_mode", choices=["one_stage", "two_stage"], default="two_stage")
    p.add_argument("--geom_opt_iter", type=int, default=10000)
    p.add_argument("--geom_opt_layers", type=int, default=3)
    p.add_argument("--geom_opt_lr", type=float, default=0.03)
    p.add_argument("--tex_opt_iter", type=int, default=20000)
    p.add_argument("--tex_opt_layers", type=int, default=4)
    p.add_argument("--tex_opt_lr", type=float, default=0.03)
    p.add_argument("--body_alpha_opt_iter", type=int, default=5000)
    p.add_argument("--body_alpha_opt_lr", type=float, default=0.06)

    p.add_argument("--train_res", type=int, default=512)
    p.add_argument("--texture_res", type=str, default="[1024, 1024]")
    p.add_argument("--batch", type=int, default=40)

    p.add_argument("--opt_submesh_bsdf", type=str, default="[]")
    p.add_argument("--ref_submesh_bsdf", type=str, default="[]")
    p.add_argument("--tex_opt_bsdf_override", type=str, default="ambient_2sided")
    p.add_argument("--ignore_suffixes", type=str, default='["_merged"]')
    p.add_argument("--test_frame_range", type=str, default="[0.95, 1.0]")

    p.add_argument("--force_auto_uv", type=_parse_bool, nargs="?", const=True, default=True)
    p.add_argument("--auto_uv_method", type=str, default="xatlas")
    p.add_argument("--submesh_sampling_mode", choices=["off", "random", "anneal"], default="anneal")
    p.add_argument("--use_color_palette", type=_parse_bool, nargs="?", const=True, default=True)
    p.add_argument("--bake_isolated_ratio", type=float, default=0.7)
    p.add_argument("--rest_pose_ratio", type=float, default=0.8)

    p.add_argument("--cam_zoom_max", type=float, default=0.0)
    p.add_argument("--cam_jitter_pixels", type=float, default=1.0)
    p.add_argument("--cam_focus_offset", type=float, default=1.0)
    p.add_argument("--cam_focus_fluctuation_in", type=float, default=0.2)
    p.add_argument("--cam_focus_fluctuation_out", type=float, default=0.5)
    p.add_argument("--cam_focus_ratio", type=float, default=0.6)
    p.add_argument("--enable_auto_center", type=_parse_bool, nargs="?", const=True, default=True)

    p.add_argument("--enable_angle_weighting", type=_parse_bool, nargs="?", const=True, default=True)
    p.add_argument("--angle_weight_grazing_decay", type=float, default=0.5)

    p.add_argument("--texture_alpha_binary_start_ratio", type=float, default=0.5)
    p.add_argument("--texture_alpha_binary_weight", type=float, default=0.02)

    p.add_argument("--body_mask_submeshes", type=str, default='["body"]')
    p.add_argument("--body_alpha_threshold", type=float, default=0.5)
    p.add_argument("--body_alpha_min", type=float, default=0.0)
    p.add_argument("--body_alpha_max", type=float, default=1.0)
    p.add_argument("--body_mask_force_layers", type=int, default=4)
    p.add_argument("--body_alpha_keep_weight", type=float, default=0.0)
    p.add_argument("--body_alpha_binary_weight", type=float, default=0.0)

    p.add_argument("--loss_l1_weight", type=float, default=0.0)
    p.add_argument("--loss_mse_weight", type=float, default=1.0)
    p.add_argument("--loss_ssim_weight", type=float, default=0.0)
    p.add_argument("--loss_lpips_weight", type=float, default=0.0)

    p.add_argument("--relative_laplacian", type=_parse_bool, nargs="?", const=True, default=True)
    p.add_argument("--laplacian_factor", type=float, default=1000.0)
    p.add_argument("--light_power", type=float, default=1.5)
    p.add_argument("--min_roughness", type=float, default=0.25)
    p.add_argument("--random_textures", type=_parse_bool, nargs="?", const=True, default=True)

    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--print_interval", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--eval_frame_max_train_set", type=int, default=10)
    p.add_argument("--eval_frame_max_test_set", type=int, default=10)
    p.add_argument("--enable_profiling", action="store_true", default=True)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args(script_argv())
    t0 = time.monotonic()

    src_dir, tgt_dir, out_dir = _resolve_io_dirs(
        args.input_dir,
        args.output_dir,
        args.source_dir,
        args.target_dir,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    combo: dict[str, Any] = vars(args).copy()
    combo["texture_res"] = _parse_json_or_list(args.texture_res, [1024, 1024])
    combo["opt_submesh_bsdf"] = _parse_json_or_list(args.opt_submesh_bsdf, [])
    combo["ref_submesh_bsdf"] = _parse_json_or_list(args.ref_submesh_bsdf, [])
    combo["ignore_suffixes"] = _parse_json_or_list(args.ignore_suffixes, ["_merged"])
    combo["test_frame_range"] = _parse_json_or_list(args.test_frame_range, [0.95, 1.0])
    combo["body_mask_submeshes"] = _parse_json_or_list(args.body_mask_submeshes, ["body"])

    pipeline_mode = combo["pipeline_mode"]
    stage01_out = out_dir / "stage01"
    stage02_out = out_dir / "stage02"

    if pipeline_mode == "one_stage":
        args_01 = {k: v for k, v in combo.items() if k in VALID_ARGS_01}
        args_01["train_textures"] = True
        args_01["use_color_palette"] = False
        args_01["iter"] = int(combo["geom_opt_iter"]) + int(combo["tex_opt_iter"])
        args_01["learning_rate"] = float(combo["geom_opt_lr"])
        args_01["tex_learning_rate"] = float(combo["tex_opt_lr"])
        args_01["layers"] = max(1, int(combo["geom_opt_layers"]), int(combo["tex_opt_layers"]))
        if float(combo.get("cam_focus_ratio", 0.0)) > 0.0:
            args_01["cam_focus_mode"] = True
        args_01.update({
            "source_dir": str(src_dir),
            "target_dir": str(tgt_dir),
            "output_dir": str(stage01_out),
            "no_sandbox": True,
        })
        _run_sub_stage("01_optimize_weight_and_rest_pose.py", args_01, gpu_id=args.gpu)
        final_elod_dir = stage01_out / "02_optimize" / "elod_mesh"
    else:
        args_01 = {k: v for k, v in combo.items() if k in VALID_ARGS_01}
        args_01["iter"] = int(combo["geom_opt_iter"])
        args_01["learning_rate"] = float(combo["geom_opt_lr"])
        args_01["layers"] = max(1, int(combo["geom_opt_layers"]))
        args_01["body_alpha_opt_iter"] = 0
        args_01["cam_focus_ratio"] = 0.0
        args_01["enable_angle_weighting"] = False
        args_01["opt_submesh_bsdf"] = convert_ambient_bsdf_rules_to_diffuse(
            combo["opt_submesh_bsdf"],
            enable_back_deepen_for_ambient_1sided=True,
        )
        args_01["ref_submesh_bsdf"] = convert_ambient_bsdf_rules_to_diffuse(
            combo["ref_submesh_bsdf"],
            enable_back_deepen_for_ambient_1sided=False,
        )
        args_01.pop("enable_back_deepen", None)
        args_01.update({
            "source_dir": str(src_dir),
            "target_dir": str(tgt_dir),
            "output_dir": str(stage01_out),
            "no_sandbox": True,
        })
        _run_sub_stage("01_optimize_weight_and_rest_pose.py", args_01, gpu_id=args.gpu)

        args_02 = {k: v for k, v in combo.items() if k in VALID_ARGS_02}
        args_02["iter"] = int(combo["tex_opt_iter"])
        args_02["learning_rate"] = float(combo["tex_opt_lr"])
        args_02["tex_learning_rate"] = float(combo["tex_opt_lr"])
        args_02["layers"] = max(1, int(combo["tex_opt_layers"]))
        if float(combo.get("cam_focus_ratio", 0.0)) > 0.0:
            args_02["cam_focus_mode"] = True
        args_02.update({
            "source_dir": str(stage01_out / "02_optimize" / "elod_mesh" / "geometry"),
            "target_dir": str(tgt_dir),
            "output_dir": str(stage02_out),
            "no_sandbox": True,
        })
        _run_sub_stage("02_bake_diffuse_texture.py", args_02, gpu_id=args.gpu)
        final_elod_dir = stage02_out / "elod_mesh"

    # Consolidate final geometry + textures + masks into <out_dir>/opt_out for downstream compatibility
    opt_out_dir = out_dir / "opt_out"
    if opt_out_dir.exists():
        shutil.rmtree(opt_out_dir)
    opt_out_dir.mkdir(parents=True, exist_ok=True)

    geom_dir = final_elod_dir / "geometry"
    if geom_dir.is_dir():
        for item in sorted(geom_dir.iterdir()):
            if item.is_file():
                shutil.copy2(item, opt_out_dir / item.name)
    for extra_sub in ("textures", "masks"):
        sub_p = final_elod_dir / extra_sub
        if sub_p.is_dir():
            shutil.copytree(sub_p, opt_out_dir / extra_sub)

    metrics_merged: dict[str, Any] = {}
    m01_path = stage01_out / "02_optimize" / "metrics_01.json"
    m02_path = stage02_out / "metrics_02.json"
    if m01_path.is_file():
        metrics_merged.update(load_json(m01_path))
    if m02_path.is_file():
        metrics_merged.update(load_json(m02_path))

    summary = {
        "schema": "ogd_lbs_step05_diff_render_summary_v1",
        "status": "SUCCEEDED",
        "pipeline_mode": pipeline_mode,
        "source_dir": str(src_dir),
        "target_dir": str(tgt_dir),
        "output_dir": str(out_dir),
        "opt_out_dir": str(opt_out_dir),
        "duration_sec": round(time.monotonic() - t0, 3),
        "metrics": metrics_merged,
    }
    dump_json(out_dir / "step05_diff_render_summary.json", summary)
    log(f"DiffRender stage completed in {summary['duration_sec']}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
