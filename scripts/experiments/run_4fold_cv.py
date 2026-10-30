#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Run 4-fold cross-validation for unseen-motion generalization.

Pipeline per fold:
    1. Split base data1 into data1_train / data1_test.
    2. Train:
        Step02(data1_train -> data2_train)
        Step03(data2_train -> data3_train)
        Step04(data3_train -> data4_train)
    3. Test:
        Step02(data1_test -> data2_test)
        build_test_data3_from_train_topology(data3_train, data2_test -> data3_test)
        Step04(data3_test -> data4_test) with --freeze_weights

Dry-run:
    Prints commands but does not execute them and does not write split trees.
"""

import argparse
from pathlib import Path
from typing import Any

try:
    from _env import REPO_ROOT
except ModuleNotFoundError:  # pragma: no cover
    from ._env import REPO_ROOT

try:
    from cv_utils import (
        infer_reference_frame_count,
        make_kfold_frame_split,
        resolve_special_train_ranges_for_dataset,
        write_split_tree,
    )
    from experiment_io import (
        append_optional,
        append_store_true,
        dump_json,
        load_json,
        python_cmd,
        run_logged_command,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .cv_utils import (
        infer_reference_frame_count,
        make_kfold_frame_split,
        resolve_special_train_ranges_for_dataset,
        write_split_tree,
    )
    from .experiment_io import (
        append_optional,
        append_store_true,
        dump_json,
        load_json,
        python_cmd,
        run_logged_command,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OGD-LBS 4-fold CV experiments.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(REPO_ROOT / "scripts/experiments/configs/cv_4fold_config.json"),
    )
    parser.add_argument("-o", "--output_root", type=str, default=None)
    parser.add_argument("--only_models", nargs="*", default=None)
    parser.add_argument("--only_folds", nargs="*", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_conda", action="store_true")
    parser.add_argument("--print_manifest_only", action="store_true")
    return parser.parse_args()


def repo_script(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def normalize_config(cfg: dict[str, Any], output_root_override: str | None) -> dict[str, Any]:
    out = dict(cfg)

    if output_root_override:
        out["output_root"] = output_root_override

    out.setdefault("schema", "ogd_lbs_cv_4fold_config_v1")
    out.setdefault("output_root", "/workspace/ogd_experiments/cv4")
    out.setdefault("conda_env", "lao-lbs")
    out.setdefault("kfold", 4)
    out.setdefault("special_train_ranges", [])
    out.setdefault("step02_defaults", {})
    out.setdefault("step03_defaults", {})
    out.setdefault("step04_train", {})
    out.setdefault("step04_test", {})
    out.setdefault("ssdr_defaults", {})
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


def selected_folds(cfg: dict[str, Any], only_folds: list[int] | None) -> list[int]:
    kfold = int(cfg.get("kfold", 4))
    if only_folds is None:
        return list(range(kfold))

    out = sorted(set(int(x) for x in only_folds))
    bad = [x for x in out if x < 0 or x >= kfold]
    if bad:
        raise RuntimeError(f"--only_folds out of range for kfold={kfold}: {bad}")
    return out


def dataset_data1_dir(dataset: dict[str, Any]) -> Path:
    if dataset.get("data1_dir"):
        return Path(str(dataset["data1_dir"])).expanduser().resolve(strict=False)

    ssdr = dataset.get("ssdr_work_dir")
    if not ssdr:
        raise RuntimeError(f"Dataset {dataset.get('model')} requires data1_dir or ssdr_work_dir.")

    return Path(str(ssdr)).expanduser().resolve(strict=False) / "data1"


def build_step02_cmd(
    cfg: dict[str, Any],
    dataset: dict[str, Any],
    *,
    input_data1: Path,
    output_data2: Path,
    metrics_dir: Path,
    no_conda: bool,
) -> list[str]:
    s2 = dict(cfg.get("step02_defaults", {}))
    cmd = python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda)
    cmd += [
        repo_script("scripts/ssdr/02_reduce_bone_weight.py"),
        "--input_dir", str(input_data1),
        "--output_dir", str(output_data2),
        "--metrics_dir", str(metrics_dir),
        "--overwrite_output",
        "--alpha", str(s2.get("alpha", s2.get("par_weight", 0.5))),
        "--beta", str(s2.get("beta", s2.get("bone_weight", 0.3))),
        "--gamma", str(s2.get("gamma", s2.get("child_weight", 0.2))),
    ]

    append_optional(cmd, "--max_workers", s2.get("max_workers", cfg.get("ssdr_defaults", {}).get("max_workers", None)))

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
    no_conda: bool,
) -> list[str]:
    s3 = dict(cfg.get("step03_defaults", {}))
    cmd = python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda)
    cmd += [
        repo_script("scripts/ssdr/03_create_low_poly_shapes.py"),
        "--input_dir", str(input_data2),
        "--output_dir", str(output_data3),
        "--metrics_dir", str(metrics_dir),
        "--overwrite_output",
    ]

    qem_config = dataset.get("qem_config", None)
    if qem_config:
        cmd += ["--config", str(Path(str(qem_config)).expanduser().resolve(strict=False))]

    append_optional(cmd, "--boundary_weight", s3.get("omega_b", s3.get("boundary_weight", None)))
    append_optional(cmd, "--max_workers", s3.get("max_workers", cfg.get("ssdr_defaults", {}).get("qem_max_workers", cfg.get("ssdr_defaults", {}).get("max_workers", None))))

    return cmd


def build_step04_cmd(
    cfg: dict[str, Any],
    *,
    input_data3: Path,
    output_data4: Path,
    metrics_dir: Path,
    step04_cfg: dict[str, Any],
    force_freeze_weights: bool,
    no_conda: bool,
) -> list[str]:
    s4 = dict(step04_cfg)

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
        "--s_max", str(s4.get("s_max", 4)),
    ]

    tau = s4.get("tau", s4.get("mask_threshold", "adaptive"))
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

    if bool(force_freeze_weights) or bool(s4.get("freeze_weights", False)):
        cmd.append("--freeze_weights")

    return cmd


def build_test_data3_cmd(
    cfg: dict[str, Any],
    *,
    train_data3: Path,
    test_data2: Path,
    output_dir: Path,
    no_conda: bool,
) -> list[str]:
    cmd = python_cmd(conda_env=cfg.get("conda_env"), no_conda=no_conda)
    cmd += [
        repo_script("scripts/experiments/build_test_data3_from_train_topology.py"),
        "--train_data3", str(train_data3),
        "--test_data2", str(test_data2),
        "--output_dir", str(output_dir),
        "--overwrite_output",
        "--validate_output",
    ]
    return cmd


def run_action(
    *,
    name: str,
    cmd: list[str],
    fold_root: Path,
    dry_run: bool,
) -> None:
    run_logged_command(
        [str(x) for x in cmd],
        log_path=fold_root / "logs" / f"{name}.log",
        cwd=REPO_ROOT,
        dry_run=bool(dry_run),
    )


def run_fold(
    cfg: dict[str, Any],
    dataset: dict[str, Any],
    *,
    fold_index: int,
    output_root: Path,
    dry_run: bool,
    force: bool,
    no_conda: bool,
) -> None:
    model = str(dataset.get("model"))
    kfold = int(cfg.get("kfold", 4))

    model_root = output_root / model
    fold_root = model_root / f"fold_{fold_index:02d}"
    done_path = fold_root / "done"

    if done_path.exists() and not force:
        print(f"[skip] {model} fold={fold_index} already done: {done_path}", flush=True)
        return

    base_data1 = dataset_data1_dir(dataset)

    frame_count, ref_path = infer_reference_frame_count(base_data1)
    special_train_ranges = resolve_special_train_ranges_for_dataset(
        frame_count=frame_count,
        dataset=dataset,
        cfg=cfg,
    )

    split = make_kfold_frame_split(
        frame_count,
        kfold=kfold,
        fold_index=fold_index,
        special_train_ranges=special_train_ranges,
    )

    fold_root.mkdir(parents=True, exist_ok=True)

    fold_summary = {
        "schema": "ogd_lbs_cv_fold_plan_v1",
        "model": model,
        "fold_index": int(fold_index),
        "fold_root": str(fold_root),
        "base_data1": str(base_data1),
        "reference_time_series_file": str(ref_path),
        "split": split.stats,
    }
    dump_json(fold_root / "fold_plan.json", fold_summary)

    data1_train = fold_root / "data1_train"
    data1_test = fold_root / "data1_test"

    print("\n" + "#" * 96, flush=True)
    print(f"[fold] model={model} fold={fold_index}/{kfold}", flush=True)
    print(f"[fold_root] {fold_root}", flush=True)
    print(f"[split] train={split.stats['train_count']} test={split.stats['test_count']} special={split.stats['special_count']}", flush=True)
    print("#" * 96, flush=True)

    if dry_run:
        print(f"[dry-run] would split data1: {base_data1} -> {data1_train}, {data1_test}", flush=True)
    else:
        split_summary = write_split_tree(
            input_tree=base_data1,
            train_output_tree=data1_train,
            test_output_tree=data1_test,
            split=split,
            overwrite=True,
        )
        dump_json(fold_root / "split_data1_summary.json", split_summary)

    train_root = fold_root / "train"
    test_root = fold_root / "test"

    # Train branch.
    run_action(
        name="train_02_reduce_weight",
        cmd=build_step02_cmd(
            cfg,
            dataset,
            input_data1=data1_train,
            output_data2=train_root / "data2",
            metrics_dir=train_root,
            no_conda=no_conda,
        ),
        fold_root=fold_root,
        dry_run=dry_run,
    )

    run_action(
        name="train_03_qem",
        cmd=build_step03_cmd(
            cfg,
            dataset,
            input_data2=train_root / "data2",
            output_data3=train_root / "data3",
            metrics_dir=train_root,
            no_conda=no_conda,
        ),
        fold_root=fold_root,
        dry_run=dry_run,
    )

    run_action(
        name="train_04_optimize",
        cmd=build_step04_cmd(
            cfg,
            input_data3=train_root / "data3",
            output_data4=train_root / "data4",
            metrics_dir=train_root,
            step04_cfg=dict(cfg.get("step04_train", {})),
            force_freeze_weights=False,
            no_conda=no_conda,
        ),
        fold_root=fold_root,
        dry_run=dry_run,
    )

    # Test branch.
    run_action(
        name="test_02_reduce_weight",
        cmd=build_step02_cmd(
            cfg,
            dataset,
            input_data1=data1_test,
            output_data2=test_root / "data2",
            metrics_dir=test_root,
            no_conda=no_conda,
        ),
        fold_root=fold_root,
        dry_run=dry_run,
    )

    run_action(
        name="test_build_data3",
        cmd=build_test_data3_cmd(
            cfg,
            train_data3=train_root / "data3",
            test_data2=test_root / "data2",
            output_dir=test_root / "data3_test",
            no_conda=no_conda,
        ),
        fold_root=fold_root,
        dry_run=dry_run,
    )

    run_action(
        name="test_04_optimize_freeze_weights",
        cmd=build_step04_cmd(
            cfg,
            input_data3=test_root / "data3_test",
            output_data4=test_root / "data4_test",
            metrics_dir=test_root,
            step04_cfg=dict(cfg.get("step04_test", {})),
            force_freeze_weights=True,
            no_conda=no_conda,
        ),
        fold_root=fold_root,
        dry_run=dry_run,
    )

    if dry_run:
        print(f"[dry-run] not writing done marker: {done_path}", flush=True)
    else:
        done_path.write_text("done\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw = load_json(args.config)
    cfg = normalize_config(raw, args.output_root)

    output_root = Path(str(cfg["output_root"])).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = selected_datasets(cfg, args.only_models)
    folds = selected_folds(cfg, args.only_folds)

    manifest = {
        "schema": "ogd_lbs_cv_4fold_manifest_v1",
        "config_path": str(Path(args.config).expanduser().resolve(strict=False)),
        "output_root": str(output_root),
        "dry_run": bool(args.dry_run),
        "kfold": int(cfg.get("kfold", 4)),
        "special_train_ranges": cfg.get("special_train_ranges", []),
        "models": [str(d.get("model")) for d in datasets],
        "folds": [int(x) for x in folds],
        "notes": [
            "Test branch uses train Step03 qem_vertex_map to build data3_test.",
            "Test Step04 always passes --freeze_weights to prevent schedule-driven weight updates.",
        ],
    }
    dump_json(output_root / "cv4_manifest.json", manifest)

    print("=" * 96, flush=True)
    print("OGD-LBS 4-fold CV runner", flush=True)
    print(f"config      : {args.config}", flush=True)
    print(f"output_root : {output_root}", flush=True)
    print(f"models      : {manifest['models']}", flush=True)
    print(f"folds       : {folds}", flush=True)
    print(f"dry_run     : {bool(args.dry_run)}", flush=True)
    print("=" * 96, flush=True)

    if args.print_manifest_only:
        return

    for dataset in datasets:
        for fold_index in folds:
            run_fold(
                cfg,
                dataset,
                fold_index=int(fold_index),
                output_root=output_root,
                dry_run=bool(args.dry_run),
                force=bool(args.force),
                no_conda=bool(args.no_conda),
            )

    print("\n4-fold CV runner finished.", flush=True)


if __name__ == "__main__":
    main()
