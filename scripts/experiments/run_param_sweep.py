#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Run OFAT sweeps for reviewer-requested OGD-LBS hyperparameters.

Parameters:
    alpha, beta, gamma  -> SSDR Step 02 aliases
    omega_b             -> SSDR Step 03 boundary_weight
    s_max               -> SSDR Step 04 max bones per vertex
    tau                 -> SSDR Step 04 fixed/adaptive mask threshold
    delta               -> Data Prep Prep 02 SDF offset, then SSDR 00-04

Default usage:
    python scripts/experiments/run_param_sweep.py \
      --config scripts/experiments/configs/ofat_sweep_config.json \
      --dry_run

Actual execution:
    python scripts/experiments/run_param_sweep.py \
      --config scripts/experiments/configs/ofat_sweep_config.json \
      --only_models Vroid_18 \
      --only_params s_max tau omega_b
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

try:
    from _env import REPO_ROOT
except ModuleNotFoundError:  # pragma: no cover
    from ._env import REPO_ROOT

try:
    from experiment_io import (
        append_optional,
        append_store_true,
        copy_tree_contents,
        dump_json,
        load_json,
        python_cmd,
        run_logged_command,
        slugify,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .experiment_io import (
        append_optional,
        append_store_true,
        copy_tree_contents,
        dump_json,
        load_json,
        python_cmd,
        run_logged_command,
        slugify,
    )


SUPPORTED_PARAMS = ("alpha", "beta", "gamma", "s_max", "tau", "omega_b", "delta")
STAGE_BY_PARAM = {
    "alpha": "ssdr_step02",
    "beta": "ssdr_step02",
    "gamma": "ssdr_step02",
    "omega_b": "ssdr_step03",
    "s_max": "ssdr_step04",
    "tau": "ssdr_step04",
    "delta": "data_prep_prep02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OFAT parameter sweeps for OGD-LBS revision experiments."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(REPO_ROOT / "scripts/experiments/configs/ofat_sweep_config.json"),
    )
    parser.add_argument(
        "-o",
        "--output_root",
        type=str,
        default=None,
        help="Override config output_root.",
    )
    parser.add_argument(
        "--only_models",
        nargs="*",
        default=None,
        help="Optional model-name filter.",
    )
    parser.add_argument(
        "--only_params",
        nargs="*",
        default=None,
        choices=SUPPORTED_PARAMS,
        help="Optional parameter filter.",
    )
    parser.add_argument(
        "--case_limit",
        type=int,
        default=-1,
        help="Debug: run/plan only the first N cases after filtering.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run cases even if done marker exists.")
    parser.add_argument("--no_conda", action="store_true", help="Use current Python instead of conda run for Python entrypoints.")
    parser.add_argument(
        "--print_manifest_only",
        action="store_true",
        help="Only write manifest and exit without running/planning commands.",
    )
    return parser.parse_args()


def normalize_config(cfg: dict[str, Any], *, output_root_override: str | None) -> dict[str, Any]:
    out = dict(cfg)

    if output_root_override:
        out["output_root"] = output_root_override

    if "output_root" not in out:
        out["output_root"] = "/workspace/ogd_experiments/ofat_sweep"

    out.setdefault("schema", "ogd_lbs_ofat_sweep_config_v1")
    out.setdefault("conda_env", "lao-lbs")
    out.setdefault("bpy5_env", "ogd-bpy52")
    out.setdefault("bpy4_env", out.get("conda_env", "lao-lbs"))

    out.setdefault("defaults", {})
    out.setdefault("sweeps", {})
    out.setdefault("delta_sweeps_by_model", {})
    out.setdefault("abc_coupling", {"mode": "keep_sum_one"})
    out.setdefault("step04_defaults", {})
    out.setdefault("ssdr_defaults", {})
    out.setdefault("data_prep_defaults", {})

    return out


def selected_datasets(cfg: dict[str, Any], only_models: list[str] | None) -> list[dict[str, Any]]:
    datasets = cfg.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise RuntimeError("Config must contain non-empty list field 'datasets'.")

    if not only_models:
        return datasets

    wanted = set(str(x) for x in only_models)
    out = [d for d in datasets if str(d.get("model")) in wanted]
    missing = wanted - {str(d.get("model")) for d in out}
    if missing:
        raise RuntimeError(f"--only_models contains unknown models in config: {sorted(missing)}")
    return out


def selected_params(cfg: dict[str, Any], only_params: list[str] | None) -> list[str]:
    params = list(only_params or SUPPORTED_PARAMS)
    sweeps = cfg.get("sweeps", {})
    delta_by_model = cfg.get("delta_sweeps_by_model", {})

    out = []
    for p in params:
        if p == "delta":
            if delta_by_model:
                out.append(p)
            continue

        if p in sweeps and isinstance(sweeps[p], list) and sweeps[p]:
            out.append(p)

    return out


def coupled_abc(
    *,
    touched: str | None,
    value: Any,
    defaults: dict[str, Any],
    abc_coupling: dict[str, Any],
) -> dict[str, float]:
    base = {
        "alpha": float(defaults.get("alpha", 0.5)),
        "beta": float(defaults.get("beta", 0.3)),
        "gamma": float(defaults.get("gamma", 0.2)),
    }

    if touched not in {"alpha", "beta", "gamma"}:
        return base

    mode = str(abc_coupling.get("mode", "keep_sum_one"))
    if mode != "keep_sum_one":
        base[touched] = float(value)
        return base

    fixed = float(value)
    if fixed < 0.0 or fixed > 1.0:
        raise ValueError(f"{touched} sweep value must be in [0,1] for keep_sum_one, got {value}")

    others = [k for k in ["alpha", "beta", "gamma"] if k != touched]
    denom = sum(base[k] for k in others)
    remain = 1.0 - fixed

    base[touched] = fixed
    if denom <= 1e-12:
        for k in others:
            base[k] = remain / 2.0
    else:
        for k in others:
            base[k] = remain * base[k] / denom

    return base


def effective_params(param: str, value: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(cfg.get("defaults", {}))
    eff = dict(defaults)

    if param in {"alpha", "beta", "gamma"}:
        eff.update(coupled_abc(
            touched=param,
            value=value,
            defaults=defaults,
            abc_coupling=dict(cfg.get("abc_coupling", {})),
        ))
    else:
        eff[param] = value

    return eff


def iter_cases(
    cfg: dict[str, Any],
    *,
    datasets: list[dict[str, Any]],
    params: list[str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for dataset in datasets:
        model = str(dataset.get("model"))
        for param in params:
            if param == "delta":
                values = cfg.get("delta_sweeps_by_model", {}).get(model, [])
            else:
                values = cfg.get("sweeps", {}).get(param, [])

            for value in values:
                eff = effective_params(param, value, cfg)
                case_id = f"{param}_{slugify(value)}"
                cases.append(
                    {
                        "schema": "ogd_lbs_ofat_case_v1",
                        "model": model,
                        "param": param,
                        "value": value,
                        "stage": STAGE_BY_PARAM[param],
                        "case_id": case_id,
                        "effective_params": eff,
                        "dataset": dataset,
                    }
                )

    return cases


def repo_script(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def get_case_root(cfg: dict[str, Any], case: dict[str, Any]) -> Path:
    output_root = Path(str(cfg["output_root"])).expanduser().resolve(strict=False)
    return output_root / str(case["model"]) / f"sweep_{case['param']}" / str(case["case_id"])


def get_qem_config(dataset: dict[str, Any]) -> str | None:
    raw = dataset.get("qem_config", None)
    if raw is None or not str(raw).strip():
        return None
    return str(Path(str(raw)).expanduser().resolve(strict=False))


def build_step02_cmd(
    cfg: dict[str, Any],
    dataset: dict[str, Any],
    *,
    input_data1: Path,
    output_data2: Path,
    metrics_dir: Path,
    eff: dict[str, Any],
    no_conda: bool,
) -> list[str]:
    cmd = python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda)
    cmd += [
        repo_script("scripts/ssdr/02_reduce_bone_weight.py"),
        "--input_dir", str(input_data1),
        "--output_dir", str(output_data2),
        "--metrics_dir", str(metrics_dir),
        "--overwrite_output",
        "--alpha", str(eff.get("alpha", 0.5)),
        "--beta", str(eff.get("beta", 0.3)),
        "--gamma", str(eff.get("gamma", 0.2)),
    ]

    ssdr_defaults = cfg.get("ssdr_defaults", {})
    step02_workers = ssdr_defaults.get("step02_max_workers", ssdr_defaults.get("max_workers", None))
    append_optional(cmd, "--max_workers", step02_workers)

    variant = dataset.get("variant", None)
    if variant:
        cmd += ["--variant", str(variant)]

    return cmd


def build_step03_cmd(
    cfg: dict[str, Any],
    dataset: dict[str, Any],
    *,
    input_data2: Path,
    output_data3: Path,
    metrics_dir: Path,
    eff: dict[str, Any],
    no_conda: bool,
) -> list[str]:
    cmd = python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda)
    cmd += [
        repo_script("scripts/ssdr/03_create_low_poly_shapes.py"),
        "--input_dir", str(input_data2),
        "--output_dir", str(output_data3),
        "--metrics_dir", str(metrics_dir),
        "--overwrite_output",
        "--boundary_weight", str(eff.get("omega_b", cfg.get("defaults", {}).get("omega_b", 1000.0))),
    ]

    qem_config = get_qem_config(dataset)
    if qem_config:
        cmd += ["--config", qem_config]

    ssdr_defaults = cfg.get("ssdr_defaults", {})
    qem_workers = ssdr_defaults.get("qem_max_workers", ssdr_defaults.get("max_workers", None))
    append_optional(cmd, "--max_workers", qem_workers)

    return cmd


def build_step04_cmd(
    cfg: dict[str, Any],
    *,
    input_data3: Path,
    output_data4: Path,
    metrics_dir: Path,
    eff: dict[str, Any],
    no_conda: bool,
) -> list[str]:
    s4 = dict(cfg.get("step04_defaults", {}))

    cmd = python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda)
    cmd += [
        repo_script("scripts/ssdr/04_optimize_weight_and_bone_trans.py"),
        "--input_dir", str(input_data3),
        "--output_dir", str(output_data4),
        "--metrics_dir", str(metrics_dir),
        "--overwrite_output",
        "--iter", str(s4.get("iter", 70)),
        "--print_every_n", str(s4.get("print_every_n", 5)),
        "--org_type", str(s4.get("org_type", 3)),
        "--weight_solver", str(s4.get("weight_solver", "simplex")),
        "--bt_covariance_mode", str(s4.get("bt_covariance_mode", "corrected")),
        "--trans_affine", str(s4.get("trans_affine", 0.0)),
        "--trans_affine_norm", str(s4.get("trans_affine_norm", 4.0)),
        "--roll_affinity", str(s4.get("roll_affinity", 0.0)),
        "--roll_affinity_norm", str(s4.get("roll_affinity_norm", 2.0)),
        "--health_level", str(s4.get("health_level", 1)),
        "--health_every", str(s4.get("health_every", 5)),
        "--health_weight_scheme", str(s4.get("health_weight_scheme", "uniform")),
        "--bone_health_every", str(s4.get("bone_health_every", 5)),
        "--s_max", str(eff.get("s_max", cfg.get("defaults", {}).get("s_max", 4))),
    ]

    tau = eff.get("tau", cfg.get("defaults", {}).get("tau", "adaptive"))
    if str(tau).lower() == "adaptive":
        cmd += ["--mask_mode", "adaptive"]
    else:
        cmd += ["--mask_mode", "fixed", "--tau", str(tau)]

    for flag, key in [
        ("--opt_weight", "opt_weight"),
        ("--input_poses", "input_poses"),
        ("--use_weight_mask", "use_weight_mask"),
        ("--use_weight_smoothness", "use_weight_smoothness"),
        ("--eval_mesh_health", "eval_mesh_health"),
        ("--eval_bone_health", "eval_bone_health"),
        ("--validate_output", "validate_output"),
        ("--weak_validate", "weak_validate"),
        ("--save_intermediate", "save_intermediate"),
    ]:
        append_store_true(cmd, flag, s4.get(key, False))

    append_optional(cmd, "--weights_smooth_step", s4.get("weights_smooth_step", None))
    append_optional(cmd, "--weight_soft_lock", s4.get("weight_soft_lock", None))
    append_optional(cmd, "--lambda_reg", s4.get("lambda_reg", None))
    append_optional(cmd, "--max_workers", s4.get("max_workers", cfg.get("ssdr_defaults", {}).get("max_workers", None)))

    if bool(s4.get("freeze_weights", False)):
        cmd.append("--freeze_weights")

    return cmd


def build_ssdr_00_cmd(cfg: dict[str, Any], *, ssdr_work_dir: Path, metrics_dir: Path, no_conda: bool) -> list[str]:
    cmd = python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda)
    cmd += [
        repo_script("scripts/ssdr/00_convert_json_to_pickle.py"),
        "--input_dir", str(ssdr_work_dir),
        "--metrics_dir", str(metrics_dir),
        "--overwrite_output",
    ]
    append_optional(cmd, "--max_workers", cfg.get("ssdr_defaults", {}).get("max_workers", None))
    return cmd


def build_ssdr_01_cmd(cfg: dict[str, Any], *, ssdr_work_dir: Path, metrics_dir: Path, no_conda: bool) -> list[str]:
    ssdr = cfg.get("ssdr_defaults", {})
    cmd = python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda)
    cmd += [
        repo_script("scripts/ssdr/01_reduce_animation_frame.py"),
        "--input_dir", str(ssdr_work_dir),
        "--metrics_dir", str(metrics_dir),
        "--overwrite_output",
        "--max_frames", str(ssdr.get("max_frames", 210)),
        "--target_frames", str(ssdr.get("target_frames", 200)),
    ]
    append_optional(cmd, "--max_workers", ssdr.get("max_workers", None))
    return cmd


def build_delta_prep_config(src_config: Path, dst_config: Path, delta: float) -> None:
    obj = load_json(src_config)

    def update_group(g: dict[str, Any]) -> None:
        if "source_meshes" in g:
            g["offset"] = float(delta)
            if "auto_offset" in g:
                g["auto_offset"] = False

    if "merge_groups" in obj and isinstance(obj["merge_groups"], dict):
        for group in obj["merge_groups"].values():
            if isinstance(group, dict):
                update_group(group)
    else:
        metadata = {"schema", "asset", "description", "notes", "unmerged_meshes"}
        for key, value in obj.items():
            if key in metadata:
                continue
            if isinstance(value, dict):
                update_group(value)

    dump_json(dst_config, obj)


def build_delta_actions(
    cfg: dict[str, Any],
    case: dict[str, Any],
    *,
    case_root: Path,
    no_conda: bool,
) -> list[dict[str, Any]]:
    dataset = case["dataset"]
    model = str(case["model"])
    value = float(case["value"])
    eff = case["effective_params"]

    data_prep_work_dir_raw = dataset.get("data_prep_work_dir", None)
    prep02_config_raw = dataset.get("prep02_config", None)

    if not data_prep_work_dir_raw:
        raise RuntimeError(f"Model {model} delta sweep requires data_prep_work_dir.")
    if not prep02_config_raw:
        raise RuntimeError(f"Model {model} delta sweep requires prep02_config.")

    data_prep_work_dir = Path(str(data_prep_work_dir_raw)).expanduser().resolve(strict=False)
    src_data_prep_01 = data_prep_work_dir / "data_prep_01"

    src_config = Path(str(prep02_config_raw)).expanduser().resolve(strict=False)
    case_config = case_root / "prep_02_sdf_merge_config_delta.json"

    build_delta_prep_config(src_config, case_config, value)

    metrics_dir = case_root / "metrics"
    ssdr_work_dir = case_root / "ssdr_work_dir"

    actions: list[dict[str, Any]] = []

    actions.append({
        "name": "data_prep_02_sdf_pipeline",
        "cmd": [
            "bash",
            repo_script("scripts/data_prep/prep_02_sdf_pipeline.sh"),
            "--input_dir", str(src_data_prep_01),
            "--output_dir", str(case_root / "data_prep_02"),
            "--config", str(case_config),
            "--bpy5_env", str(cfg.get("bpy5_env", "ogd-bpy52")),
            "--bpy4_env", str(cfg.get("bpy4_env", cfg.get("conda_env", "lao-lbs"))),
        ],
    })

    actions.append({
        "name": "data_prep_03_surface_deform",
        "cmd": python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda)
        + [
            repo_script("scripts/data_prep/prep_03_surface_deform.py"),
            "--input_dir", str(case_root / "data_prep_02"),
            "--output_dir", str(case_root / "data_prep_03"),
            "--drive_mode", "BOTH",
        ],
    })

    cmd04a = python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda) + [
        repo_script("scripts/data_prep/prep_04a_bone_pruning.py"),
        "--input_dir", str(case_root / "data_prep_03"),
        "--output_dir", str(case_root / "data_prep_04"),
    ]

    extra_reduce = dataset.get("extra_bone_reduce_config", None)
    if extra_reduce:
        cmd04a += [
            "--extra_bone_reduce_config", str(Path(str(extra_reduce)).expanduser().resolve(strict=False)),
            "--overwrite_config_variants",
            "--no_manual_template",
        ]

    actions.append({"name": "data_prep_04a_bone_pruning", "cmd": cmd04a})

    dp_defaults = cfg.get("data_prep_defaults", {})
    cmd05 = python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda) + [
        repo_script("scripts/data_prep/prep_05_export_fitbox.py"),
        "--input_dir", str(case_root / "data_prep_04"),
        "--output_dir", str(case_root / "data_prep_05"),
    ]
    target_frame_count = dp_defaults.get("target_frame_count", None)
    if target_frame_count is not None:
        cmd05 += ["--target_frame_count", str(target_frame_count)]

    actions.append({"name": "data_prep_05_export_fitbox", "cmd": cmd05})

    actions.append({
        "name": "handoff_fitbox_vdb_to_ssdr_data0_json",
        "internal": "copy_tree_contents",
        "src": str(case_root / "data_prep_05" / "fitbox_vdb"),
        "dst": str(ssdr_work_dir / "data0_json"),
    })

    actions.append({"name": "ssdr_00_convert", "cmd": build_ssdr_00_cmd(cfg, ssdr_work_dir=ssdr_work_dir, metrics_dir=metrics_dir, no_conda=no_conda)})
    actions.append({"name": "ssdr_01_frame_sampling", "cmd": build_ssdr_01_cmd(cfg, ssdr_work_dir=ssdr_work_dir, metrics_dir=metrics_dir, no_conda=no_conda)})

    actions.append({
        "name": "ssdr_02_reduce_weight",
        "cmd": build_step02_cmd(
            cfg,
            dataset,
            input_data1=ssdr_work_dir / "data1",
            output_data2=ssdr_work_dir / "data2",
            metrics_dir=metrics_dir,
            eff=eff,
            no_conda=no_conda,
        ),
    })

    actions.append({
        "name": "ssdr_03_qem",
        "cmd": build_step03_cmd(
            cfg,
            dataset,
            input_data2=ssdr_work_dir / "data2",
            output_data3=ssdr_work_dir / "data3",
            metrics_dir=metrics_dir,
            eff=eff,
            no_conda=no_conda,
        ),
    })

    actions.append({
        "name": "ssdr_04_optimize",
        "cmd": build_step04_cmd(
            cfg,
            input_data3=ssdr_work_dir / "data3",
            output_data4=ssdr_work_dir / "data4",
            metrics_dir=case_root,
            eff=eff,
            no_conda=no_conda,
        ),
    })

    return actions


def build_ssdr_only_actions(
    cfg: dict[str, Any],
    case: dict[str, Any],
    *,
    case_root: Path,
    no_conda: bool,
) -> list[dict[str, Any]]:
    dataset = case["dataset"]
    eff = case["effective_params"]
    param = str(case["param"])

    ssdr_work_dir_raw = dataset.get("ssdr_work_dir", None)
    if not ssdr_work_dir_raw:
        raise RuntimeError(f"Model {case['model']} requires ssdr_work_dir for parameter {param}.")

    ssdr_work_dir = Path(str(ssdr_work_dir_raw)).expanduser().resolve(strict=False)
    metrics_dir = case_root / "metrics"

    actions: list[dict[str, Any]] = []

    if param in {"alpha", "beta", "gamma"}:
        actions.append({
            "name": "ssdr_02_reduce_weight",
            "cmd": build_step02_cmd(
                cfg,
                dataset,
                input_data1=ssdr_work_dir / "data1",
                output_data2=case_root / "data2",
                metrics_dir=metrics_dir,
                eff=eff,
                no_conda=no_conda,
            ),
        })
        actions.append({
            "name": "ssdr_03_qem",
            "cmd": build_step03_cmd(
                cfg,
                dataset,
                input_data2=case_root / "data2",
                output_data3=case_root / "data3",
                metrics_dir=metrics_dir,
                eff=eff,
                no_conda=no_conda,
            ),
        })
        step04_input = case_root / "data3"

    elif param == "omega_b":
        actions.append({
            "name": "ssdr_03_qem",
            "cmd": build_step03_cmd(
                cfg,
                dataset,
                input_data2=ssdr_work_dir / "data2",
                output_data3=case_root / "data3",
                metrics_dir=metrics_dir,
                eff=eff,
                no_conda=no_conda,
            ),
        })
        step04_input = case_root / "data3"

    elif param in {"s_max", "tau"}:
        step04_input = ssdr_work_dir / "data3"

    else:
        raise RuntimeError(f"Unsupported SSDR-only param: {param}")

    actions.append({
        "name": "ssdr_04_optimize",
        "cmd": build_step04_cmd(
            cfg,
            input_data3=step04_input,
            output_data4=case_root / "data4",
            metrics_dir=case_root,
            eff=eff,
            no_conda=no_conda,
        ),
    })

    return actions


def build_actions(
    cfg: dict[str, Any],
    case: dict[str, Any],
    *,
    case_root: Path,
    no_conda: bool,
) -> list[dict[str, Any]]:
    if case["param"] == "delta":
        return build_delta_actions(cfg, case, case_root=case_root, no_conda=no_conda)

    return build_ssdr_only_actions(cfg, case, case_root=case_root, no_conda=no_conda)


def run_case(
    cfg: dict[str, Any],
    case: dict[str, Any],
    *,
    dry_run: bool,
    force: bool,
    no_conda: bool,
) -> None:
    case_root = get_case_root(cfg, case)
    done_path = case_root / "done"

    if done_path.exists() and not force:
        print(f"[skip] {case['model']} {case['case_id']} already done: {done_path}", flush=True)
        return

    case_root.mkdir(parents=True, exist_ok=True)
    dump_json(case_root / "case_config.json", case)

    actions = build_actions(cfg, case, case_root=case_root, no_conda=no_conda)
    dump_json(case_root / "commands.json", actions)

    print("\n" + "#" * 96, flush=True)
    print(f"[case] model={case['model']} param={case['param']} value={case['value']} id={case['case_id']}", flush=True)
    print(f"[case_root] {case_root}", flush=True)
    print("#" * 96, flush=True)

    for action in actions:
        name = str(action["name"])

        if action.get("internal") == "copy_tree_contents":
            copy_tree_contents(
                Path(str(action["src"])),
                Path(str(action["dst"])),
                dry_run=dry_run,
            )
            continue

        cmd = [str(x) for x in action["cmd"]]
        run_logged_command(
            cmd,
            log_path=case_root / "logs" / f"{name}.log",
            cwd=REPO_ROOT,
            dry_run=dry_run,
        )

    if dry_run:
        print(f"[dry-run] not writing done marker for {case_root}", flush=True)
    else:
        done_path.write_text("done\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    cfg_raw = load_json(args.config)
    cfg = normalize_config(cfg_raw, output_root_override=args.output_root)

    datasets = selected_datasets(cfg, args.only_models)
    params = selected_params(cfg, args.only_params)

    cases = iter_cases(cfg, datasets=datasets, params=params)

    if int(args.case_limit) > 0:
        cases = cases[: int(args.case_limit)]

    output_root = Path(str(cfg["output_root"])).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "ogd_lbs_ofat_sweep_manifest_v1",
        "config_path": str(Path(args.config).expanduser().resolve(strict=False)),
        "output_root": str(output_root),
        "dry_run": bool(args.dry_run),
        "selected_models": [str(d.get("model")) for d in datasets],
        "selected_params": params,
        "case_count": int(len(cases)),
        "cases": cases,
        "notes": [
            "OFAT: one reviewer-requested parameter is varied per case.",
            "For alpha/beta/gamma, keep_sum_one coupling adjusts the other two coefficients proportionally.",
            "For delta, the runner regenerates Data Prep from Prep 02 onward, then SSDR 00-04.",
        ],
    }
    dump_json(output_root / "ofat_manifest.json", manifest)

    print("=" * 96, flush=True)
    print("OGD-LBS OFAT sweep runner", flush=True)
    print(f"config       : {args.config}", flush=True)
    print(f"output_root  : {output_root}", flush=True)
    print(f"models       : {manifest['selected_models']}", flush=True)
    print(f"params       : {params}", flush=True)
    print(f"case_count   : {len(cases)}", flush=True)
    print(f"dry_run      : {bool(args.dry_run)}", flush=True)
    print("=" * 96, flush=True)

    if args.print_manifest_only:
        return

    for case in cases:
        run_case(
            cfg,
            case,
            dry_run=bool(args.dry_run),
            force=bool(args.force),
            no_conda=bool(args.no_conda),
        )

    print("\nOFAT sweep runner finished.", flush=True)


if __name__ == "__main__":
    main()
