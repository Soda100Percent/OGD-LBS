#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
SSDR Step 04: Optimize skinning weights and bone transforms.

Default pipeline semantics:
    --input_dir <work_dir>
        reads : <work_dir>/data3
        writes: <work_dir>/data4

    --input_dir <work_dir>/data3
        reads : <work_dir>/data3
        writes: <work_dir>/data4

Explicit automation mode:
    --input_dir <custom_input_tree> --output_dir <custom_output_tree>
        reads : <custom_input_tree>
        writes: <custom_output_tree>

Input layout:
    data3/
    ├── skeleton_hierarchy_data.pkl
    ├── animation_data.pkl
    └── <mesh>.pkl

Output layout:
    data4/
    ├── opt_out/
    │   ├── skeleton_hierarchy_data.pkl
    │   ├── animation_data.pkl
    │   ├── <mesh>.pkl
    │   └── metrics_history_per_frame.npz
    ├── opt_out_iter_<n>/              # optional --save_intermediate
    └── step04_optimize_summary.json
"""

import os

# Avoid thread explosion when NumPy/SciPy BLAS is used inside ProcessPoolExecutor.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


def ensure_repo_paths_in_syspath() -> None:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src_dir = parent / "src"
            ssdr_script_dir = parent / "scripts" / "ssdr"

            for item in [src_dir, ssdr_script_dir]:
                item_s = str(item)
                if item_s not in sys.path:
                    sys.path.insert(0, item_s)
            return

    raise RuntimeError("Could not locate repository root containing pyproject.toml and src/.")


ensure_repo_paths_in_syspath()

from ogd_lbs.utils.common import (  # noqa: E402
    log,
    warn,
    fail,
    script_argv,
    Timer,
    load_data,
    load_json,
    dump_pkl,
    dump_json,
    resolve_specific_file,
    discover_and_resolve_files,
    record_metric,
    dump_metrics,
)
from ogd_lbs.fitbox.schema import (  # noqa: E402
    KEY_ANIMATION_POSES,
    KEY_BAKED_VERTICES,
    KEY_BONE_NAMES,
    KEY_FACES,
    KEY_FIXED_BONES,
    KEY_REST_POSE,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTEX_BONE_WEIGHTS_MASK,
)
from ogd_lbs.fitbox.validation import (  # noqa: E402
    validate_skeleton_and_animation,
    validate_fitbox_mesh,
)

from ogd_lbs.ssdr.schedulers import get_legacy_schedule  # noqa: E402
from ogd_lbs.ssdr.weight_mask import build_weight_mask  # noqa: E402
from ogd_lbs.ssdr.utils import validate_combined_arrays  # noqa: E402
from ogd_lbs.ssdr.optimize_weight import opt_vbw_parallel  # noqa: E402
from ogd_lbs.ssdr.optimize_bone_trans import opt_bt_parallel  # noqa: E402
from ogd_lbs.ssdr.reconstruction_err import (  # noqa: E402
    recon_err,
    recon_err_hausdorff,
    recon_err_per_frame,
)
from ogd_lbs.ssdr.skeleton_organization import calc_bone_lens, calc_org_loss  # noqa: E402
from ogd_lbs.ssdr.skinning import skinning_t, skinning_w, skinning_s  # noqa: E402
from ogd_lbs.ssdr.mesh_health import MeshHealthPrecompute  # noqa: E402
from ogd_lbs.ssdr.bone_health import BoneHealthPrecompute  # noqa: E402

from paths import (  # noqa: E402
    resolve_ssdr_stage_dirs,
    validate_ssdr_stage_resolution,
    print_ssdr_resolution_info,
)


ANIMATION_STEM = "animation_data"
SKELETON_STEM = "skeleton_hierarchy_data"

CORE_STEMS = {
    ANIMATION_STEM,
    SKELETON_STEM,
}

SUMMARY_STEMS = {
    "step00_convert_summary",
    "step01_frame_sampling_summary",
    "step02_reduce_bone_weight_summary",
    "step03_qem_summary",
    "step04_optimize_summary",
}


def path_has_any_children(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def prepare_output_dir(
    output_tree: Path,
    *,
    overwrite_output: bool,
    allow_merge: bool,
) -> None:
    if overwrite_output and allow_merge:
        fail("--overwrite_output and --allow_merge are mutually exclusive.")

    if output_tree.exists():
        if not output_tree.is_dir():
            fail(f"Output path exists but is not a directory: {output_tree}")

        if overwrite_output:
            warn(f"Removing existing output directory because --overwrite_output is set: {output_tree}")
            shutil.rmtree(output_tree)
            output_tree.mkdir(parents=True, exist_ok=True)
            return

        if path_has_any_children(output_tree) and not allow_merge:
            fail(
                f"Output directory already exists and is non-empty: {output_tree}\n"
                "Use --overwrite_output to clear it, or --allow_merge to write into it."
            )

        output_tree.mkdir(parents=True, exist_ok=True)
        return

    output_tree.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]

    if isinstance(obj, set):
        return sorted(json_safe(x) for x in obj)

    return obj


def require_dict(obj: Any, *, role: str, path: Path) -> dict[str, Any]:
    if not isinstance(obj, dict):
        fail(f"{role} must be a dictionary object: {path}")
    return obj


def discover_step04_mesh_files(input_tree: Path) -> list[Path]:
    ext_priority = [".pkl", ".json"]
    exclude = set(CORE_STEMS) | set(SUMMARY_STEMS)
    return discover_and_resolve_files(
        input_tree,
        ext_priority,
        exclude_stems=exclude,
    )


def require_triangle_faces(mesh_obj: dict[str, Any], *, mesh_name: str) -> None:
    faces = np.asarray(mesh_obj.get(KEY_FACES, []), dtype=np.int64)

    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] <= 0:
        fail(
            f"[{mesh_name}] Step 04 expects triangulated data3 mesh faces with shape (F, 3); "
            f"got {faces.shape}. Run Step 03 QEM first."
        )


def normalize_parent_indices(parent_indices: Any, *, bone_count: int) -> np.ndarray:
    raw = list(parent_indices)

    if len(raw) != int(bone_count):
        fail(f"parent_indices length mismatch: {len(raw)} vs bone_count={bone_count}")

    out: list[int | None] = []
    for value in raw:
        if value is None:
            out.append(None)
            continue

        idx = int(value)
        out.append(idx if idx >= 0 else None)

    return np.asarray(out, dtype=object)


def normalize_fixed_bones(raw_fixed_bones: Any, bone_names: list[str]) -> set[int]:
    if raw_fixed_bones is None:
        return set()

    if isinstance(raw_fixed_bones, np.ndarray):
        items = raw_fixed_bones.tolist()
    elif isinstance(raw_fixed_bones, (list, tuple, set)):
        items = list(raw_fixed_bones)
    else:
        items = [raw_fixed_bones]

    name_to_idx = {name: idx for idx, name in enumerate(bone_names)}
    out: set[int] = set()

    for item in items:
        if isinstance(item, str):
            if item in name_to_idx:
                out.add(int(name_to_idx[item]))
                continue

            try:
                out.add(int(item))
                continue
            except ValueError:
                warn(f"Ignoring unknown fixed bone name: {item!r}")
                continue

        out.add(int(item))

    valid = {idx for idx in out if 0 <= idx < len(bone_names)}
    invalid = sorted(out - valid)
    if invalid:
        warn(f"Ignoring fixed bone indices out of range: {invalid}")

    return valid


def get_reconstructed_sequence(
    v: np.ndarray,
    bt: np.ndarray,
    inv_rbt: np.ndarray,
    vbw: np.ndarray,
) -> np.ndarray:
    n_p = bt.shape[0]
    n_v = v.shape[0]
    p_r = np.zeros((n_p, n_v, 3), dtype=np.float64)

    for p_id in range(n_p):
        constructed = skinning_t(v, bt, inv_rbt, p_id)
        constructed = skinning_w(constructed, vbw)
        p_r[p_id] = skinning_s(constructed)

    return p_r


def load_baked_schedule(path: str | None) -> list[dict[str, Any]] | None:
    if path is None or not str(path).strip():
        return None

    sched_path = Path(path).expanduser().resolve(strict=False)

    if not sched_path.exists():
        fail(f"baked_schedule does not exist: {sched_path}")

    obj = load_json(sched_path)
    if not isinstance(obj, list):
        fail(f"baked_schedule must be a JSON list: {sched_path}")

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(obj):
        if not isinstance(item, dict):
            fail(f"baked_schedule[{idx}] must be an object.")
        out.append(dict(item))

    log(f"Loaded baked schedule: {sched_path} (iterations={len(out)})")
    return out


def resolve_iteration_schedule(
    *,
    it: int,
    max_iter: int,
    org_type: int,
    opt_weight: bool,
    baked_schedule: list[dict[str, Any]] | None,
) -> tuple[bool, int, float, str]:
    if baked_schedule is not None and int(it) < len(baked_schedule):
        bs = baked_schedule[int(it)]
        to_opt_weight = bool(bs.get("to_opt_weight", False))
        weight_calc_mode = int(bs.get("weight_calc_mode", 0))
        jnt_factor = float(bs.get("jnt_factor", 0.0))
        return to_opt_weight, weight_calc_mode, jnt_factor, "baked_schedule"

    to_opt_weight, weight_calc_mode, jnt_factor = get_legacy_schedule(
        int(it),
        int(max_iter),
        int(org_type),
        bool(opt_weight),
    )
    return bool(to_opt_weight), int(weight_calc_mode), float(jnt_factor), "legacy_schedule"


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(path, json_safe(summary))


def optimize_weight_and_bone_trans_tree(
    *,
    input_tree: Path,
    output_tree: Path,
    baked_schedule_path: str | None,
    print_every_n: int,
    max_iter: int,
    org_type: int,
    opt_weight: bool,
    input_poses: bool,
    use_weight_mask: bool,
    weight_solver: str,
    use_weight_smoothness: bool,
    weights_smooth_step: float,
    weight_soft_lock: float,
    lambda_reg: float,
    bt_covariance_mode: str,
    trans_affine: float,
    trans_affine_norm: float,
    roll_affinity: float,
    roll_affinity_norm: float,
    eval_mesh_health: bool,
    health_level: int,
    health_every: int,
    health_weight_scheme: str,
    eval_bone_health: bool,
    bone_health_every: int,
    max_workers: int | None,
    enable_profiling: bool,
    validate_output: bool,
    weak_validate: bool,
    save_intermediate: bool,
    overwrite_output: bool,
    allow_merge: bool,
    write_summary_file: bool,
) -> dict[str, Any]:
    prepare_output_dir(
        output_tree,
        overwrite_output=overwrite_output,
        allow_merge=allow_merge,
    )

    baked_schedule = load_baked_schedule(baked_schedule_path)

    with Timer("time_04_file_scanning"):
        ext_priority = [".pkl", ".json"]

        anim_path = resolve_specific_file(input_tree, ANIMATION_STEM, ext_priority)
        skelhier_path = resolve_specific_file(input_tree, SKELETON_STEM, ext_priority)

        if anim_path is None:
            fail(f"Missing core animation file in {input_tree}: {ANIMATION_STEM}.pkl/json")
        if skelhier_path is None:
            fail(f"Missing core skeleton file in {input_tree}: {SKELETON_STEM}.pkl/json")

        anim_obj = require_dict(load_data(anim_path), role="animation_data", path=anim_path)
        skelhier_obj = require_dict(load_data(skelhier_path), role="skeleton_hierarchy_data", path=skelhier_path)

        n_p, n_b = validate_skeleton_and_animation(anim_obj, skelhier_obj)

        if KEY_REST_POSE not in anim_obj:
            fail(f"animation_data missing required key: {KEY_REST_POSE}")
        if KEY_ANIMATION_POSES not in anim_obj:
            fail(f"animation_data missing required key: {KEY_ANIMATION_POSES}")

        rbt = np.asarray(anim_obj[KEY_REST_POSE], dtype=np.float64)
        bt = np.asarray(anim_obj[KEY_ANIMATION_POSES], dtype=np.float64).copy()

        if rbt.shape != (n_b, 4, 4):
            fail(f"rest_pose shape mismatch: expected ({n_b}, 4, 4), got {rbt.shape}")
        if bt.shape != (n_p, n_b, 4, 4):
            fail(f"animation_poses shape mismatch: expected ({n_p}, {n_b}, 4, 4), got {bt.shape}")

        bone_names = [str(x) for x in skelhier_obj.get(KEY_BONE_NAMES, anim_obj.get(KEY_BONE_NAMES, []))]
        if len(bone_names) != n_b:
            fail(f"bone_names length mismatch: {len(bone_names)} vs n_b={n_b}")

        fixed_bones = normalize_fixed_bones(anim_obj.get(KEY_FIXED_BONES, []), bone_names)
        par_idxs = normalize_parent_indices(skelhier_obj["parent_indices"], bone_count=n_b)
        inv_rbt = np.linalg.inv(rbt)

        if not bool(input_poses):
            for p_id in range(n_p):
                bt[p_id] = rbt

        orig_bone_lens = calc_bone_lens(rbt, par_idxs)
        bone_orig_poses = np.array(
            [(rbt[b_id] @ np.array([0.0, 0.0, 0.0, 1.0]))[:3] for b_id in range(n_b)],
            dtype=np.float64,
        )

        bone_jnts = []
        for idx in range(n_b):
            par_id = par_idxs[idx]
            child_indices = [e_id for e_id, e_par_id in enumerate(par_idxs) if e_par_id == idx]

            jnts_info = []
            if par_id is not None and int(par_id) >= 0:
                jnts_info.append((int(par_id), bone_orig_poses[idx]))

            for child_id in child_indices:
                jnts_info.append((int(child_id), bone_orig_poses[int(child_id)]))

            bone_jnts.append(jnts_info)

        mesh_paths = discover_step04_mesh_files(input_tree)

        if not mesh_paths:
            fail(f"No mesh PKL/JSON files found under input tree: {input_tree}")

    bone_health_pre = BoneHealthPrecompute(rbt=rbt, par_idxs=par_idxs) if eval_bone_health else None

    with Timer("time_04_matrix_assembly"):
        name_list: list[str] = []
        v_idxs_list: list[np.ndarray] = []
        skdmeshes: list[dict[str, Any]] = []
        mesh_paths_by_name: dict[str, str] = {}

        v_cb = np.zeros((0, 3), dtype=np.float64)
        ba_v_cb = np.zeros((n_p, 0, 3), dtype=np.float64)
        vbw_cb = np.zeros((0, n_b), dtype=np.float64)

        vbw_mask_list: list[np.ndarray] = []
        laplacian_solvers: list[Any] = []
        health_pre_list: list[Any] = []

        for path in mesh_paths:
            mesh_obj = require_dict(load_data(path), role=f"mesh {path.name}", path=path)
            name = str(mesh_obj.get("name", path.stem))

            validate_fitbox_mesh(
                mesh_obj,
                n_poses=n_p,
                n_bones=n_b,
                name=f"{name}/input",
                weak_validate=bool(weak_validate),
            )
            require_triangle_faces(mesh_obj, mesh_name=name)

            v = np.asarray(mesh_obj[KEY_BAKED_VERTICES], dtype=np.float64)
            if v.ndim != 3:
                fail(f"[{name}] baked vertices expected 3D, got {v.shape}")

            rest_v = np.asarray(mesh_obj["vertices"], dtype=np.float64)
            f = np.asarray(mesh_obj[KEY_FACES], dtype=np.int64)
            ba_v = np.asarray(mesh_obj[KEY_BAKED_VERTICES], dtype=np.float64)
            vbw = np.asarray(mesh_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)

            if use_weight_smoothness:
                from ogd_lbs.ssdr.sparse_laplacian import SparseLaplacianSolver
                laplacian_solvers.append(
                    SparseLaplacianSolver(rest_v, f, step=float(weights_smooth_step))
                )
            else:
                laplacian_solvers.append(None)

            if eval_mesh_health:
                health_pre_list.append(
                    MeshHealthPrecompute(
                        rest_v=rest_v,
                        faces=f,
                        weight_scheme=str(health_weight_scheme),
                    )
                )
            else:
                health_pre_list.append(None)

            vbw_mask = build_weight_mask(mesh_obj, vbw, bool(use_weight_mask), name)

            v_id_from = v_cb.shape[0]
            v_idxs_list.append(np.arange(v_id_from, v_id_from + rest_v.shape[0], dtype=np.int64))

            v_cb = np.concatenate((v_cb, rest_v), axis=0)
            ba_v_cb = np.concatenate((ba_v_cb, ba_v), axis=1)
            vbw_cb = np.concatenate((vbw_cb, vbw), axis=0)

            vbw_mask_list.append(vbw_mask)
            name_list.append(name)
            skdmeshes.append(mesh_obj)
            mesh_paths_by_name[name] = str(path)

        validate_combined_arrays(v_cb, ba_v_cb, vbw_cb, bt, rbt)

        max_name_len = max((len(n) for n in name_list), default=0)
        name_col_width = max(max_name_len, 15)

    log("Selected Step 04 optimization inputs:")
    log(f"  input_tree       : {input_tree}")
    log(f"  output_tree      : {output_tree}")
    log(f"  animation        : {anim_path}")
    log(f"  skeleton         : {skelhier_path}")
    log(f"  mesh_candidates  : {len(mesh_paths)}")
    log(f"  mesh_names       : {name_list}")
    log(f"  n_poses          : {n_p}")
    log(f"  n_bones          : {n_b}")
    log(f"  total_vertices   : {v_cb.shape[0]}")
    log(f"  iter             : {max_iter}")
    log(f"  org_type         : {org_type}")
    log(f"  opt_weight       : {opt_weight}")
    log(f"  input_poses      : {input_poses}")
    log(f"  use_weight_mask  : {use_weight_mask}")
    log(f"  weight_solver    : {weight_solver}")
    log(f"  max_workers      : {max_workers}")
    log(f"  fixed_bones      : {sorted(fixed_bones)}")
    log(f"  eval_mesh_health : {eval_mesh_health}")
    log(f"  eval_bone_health : {eval_bone_health}")

    with Timer("time_04_optimization_loop"):
        org_loss_array = np.zeros((max_iter, n_p), dtype=np.float64)
        sparseness = 4

        loss_buffer: dict[str, list[float]] = {
            "err": [],
            "org_loss": [],
            "jnt_factor": [],
        }
        bone_health_buffer: dict[str, list[Any]] = {
            "roll_max": [],
            "swing_max": [],
            "roll_150": [],
            "worst_roll_bone": [],
            "worst_swing_bone": [],
        }
        mesh_health_buffer: dict[str, dict[str, list[float]]] = {
            name: {
                "arap": [],
                "edge": [],
                "flip": [],
            }
            for name in name_list
        }

        history_metrics: dict[str, np.ndarray] = {
            "org_loss": np.zeros((max_iter, n_p), dtype=np.float64),
            "l2_err": np.zeros((max_iter, n_p), dtype=np.float64),
        }

        if eval_bone_health:
            history_metrics["bone_roll_max"] = np.zeros((max_iter, n_p), dtype=np.float64)
            history_metrics["bone_swing_max"] = np.zeros((max_iter, n_p), dtype=np.float64)

        if eval_mesh_health:
            for name in name_list:
                history_metrics[f"mesh_{name}_arap"] = np.zeros((max_iter, n_p), dtype=np.float64)
                history_metrics[f"mesh_{name}_edge"] = np.zeros((max_iter, n_p), dtype=np.float64)

        def flush_buffers(prefix: str) -> None:
            if not loss_buffer["err"]:
                return

            avg_err = float(np.mean(loss_buffer["err"]))
            avg_org = float(np.mean(loss_buffer["org_loss"]))
            avg_jnt = float(np.mean(loss_buffer["jnt_factor"]))

            log(
                f"{prefix} JntFactor: {avg_jnt:.4f} | "
                f"Err: {avg_err:.6f} | OrgLoss: {avg_org:.6f}"
            )

            if bone_health_buffer["roll_max"]:
                roll_max = float(np.max(bone_health_buffer["roll_max"]))
                swing_max = float(np.max(bone_health_buffer["swing_max"]))
                roll_150 = float(np.mean(bone_health_buffer["roll_150"]))

                idx_roll = int(np.argmax(bone_health_buffer["roll_max"]))
                idx_swing = int(np.argmax(bone_health_buffer["swing_max"]))
                worst_roll = bone_health_buffer["worst_roll_bone"][idx_roll]
                worst_swing = bone_health_buffer["worst_swing_bone"][idx_swing]

                s_roll = "[!ROLL_FLIP]" if roll_max > 150.0 else ""
                s_swing = "[!SWING_FOLD]" if swing_max > 150.0 else ""
                has_warn = (roll_max > 150.0) or (swing_max > 150.0)

                log(
                    f"  -> [Bone Health] {s_roll:<12} {s_swing:<13} "
                    f"Roll Max: {roll_max:.1f} deg | Over 150: {roll_150:.2%} | "
                    f"Swing Max: {swing_max:.1f} deg"
                )
                if has_warn:
                    log(f"     Worst Roll: {worst_roll} | Worst Swing: {worst_swing}")

            has_mesh_data = any(len(mesh_health_buffer[name]["arap"]) > 0 for name in name_list)
            if has_mesh_data:
                header_pad = max(1, name_col_width - 13)
                log(
                    f"  -> [Mesh Health]{' ' * header_pad}"
                    "FLIP     ARAP     EDGE       ARAP Ratio   Edge p99 Excess   Normal Flip"
                )

                for name in name_list:
                    mb = mesh_health_buffer[name]
                    if not mb["arap"]:
                        continue

                    arap_max = float(np.max(mb["arap"]))
                    edge_max = float(np.max(mb["edge"]))
                    flip_max = float(np.max(mb["flip"]))

                    s_flip = "[!FLIP]" if flip_max > 0.0 else ""
                    s_arap = "[!ARAP]" if arap_max > 1.5 else ""
                    s_edge = "[!EDGE]" if edge_max > 0.1 else ""

                    log(
                        f"     {name:<{name_col_width}} "
                        f"{s_flip:<8} {s_arap:<8} {s_edge:<8} "
                        f"{arap_max:10.3f}   {edge_max:15.4f}   {flip_max:11.2%}"
                    )

            for key in loss_buffer:
                loss_buffer[key].clear()
            for key in bone_health_buffer:
                bone_health_buffer[key].clear()
            for name in name_list:
                for key in mesh_health_buffer[name]:
                    mesh_health_buffer[name][key].clear()

        initial_err = recon_err(v_cb, ba_v_cb, bt, inv_rbt, vbw_cb)
        loss_buffer["err"].append(float(initial_err))
        loss_buffer["org_loss"].append(0.0)
        loss_buffer["jnt_factor"].append(0.0)

        if eval_bone_health and bone_health_pre is not None:
            bh_res = bone_health_pre.evaluate(bt, bone_names=bone_names)
            if bh_res.get("status") != "no_valid_bones":
                bone_health_buffer["roll_max"].append(float(bh_res.get("roll_abs_deg_max", 0.0)))
                bone_health_buffer["swing_max"].append(float(bh_res.get("swing_delta_max", 0.0)))
                bone_health_buffer["roll_150"].append(float(bh_res.get("roll_over_150_ratio", 0.0)))
                bone_health_buffer["worst_roll_bone"].append(str(bh_res.get("worst_bone_by_roll", "")))
                bone_health_buffer["worst_swing_bone"].append(str(bh_res.get("worst_bone_by_swing", "")))

        if eval_mesh_health:
            for i, name in enumerate(name_list):
                v_indices = v_idxs_list[i]
                p_recon = get_reconstructed_sequence(
                    v_cb[v_indices, :],
                    bt,
                    inv_rbt,
                    vbw_cb[v_indices, :],
                )
                health_res = health_pre_list[i].evaluate(
                    recon_seq=p_recon,
                    gt_seq=ba_v_cb[:, v_indices, :],
                    level=int(health_level),
                )
                rel = health_res.get("relative", {})
                mesh_health_buffer[name]["arap"].append(float(rel.get("arap_global_mean_ratio", 0.0)))
                mesh_health_buffer[name]["edge"].append(float(rel.get("edge_log_abs_p99_excess", 0.0)))
                mesh_health_buffer[name]["flip"].append(float(rel.get("normal_vs_gt_flip_ratio", 0.0)))

        flush_buffers("[Initial]")

        interval_start = 0

        for it in range(max_iter):
            to_opt_weight, weight_calc_mode, jnt_factor, schedule_source = resolve_iteration_schedule(
                it=it,
                max_iter=max_iter,
                org_type=org_type,
                opt_weight=opt_weight,
                baked_schedule=baked_schedule,
            )

            log(
                f"[Iter {it:02d}] schedule={schedule_source}, "
                f"to_opt_weight={to_opt_weight}, weight_calc_mode={weight_calc_mode}, "
                f"jnt_factor={jnt_factor:.6g}"
            )

            with Timer(f"time_04_iter_{it}_bt"):
                bt = opt_bt_parallel(
                    v=v_cb,
                    p=ba_v_cb,
                    bt=bt,
                    inv_rbt=inv_rbt,
                    vbw=vbw_cb,
                    bj=bone_jnts,
                    bj_l=float(jnt_factor),
                    fixed_bones=fixed_bones,
                    max_workers=max_workers,
                    enable_profiling=bool(enable_profiling),
                    trans_affine=float(trans_affine),
                    trans_affine_norm=float(trans_affine_norm),
                    bt_covariance_mode=str(bt_covariance_mode),
                    rbt=rbt,
                    par_idxs=par_idxs,
                    roll_affinity=float(roll_affinity),
                    roll_affinity_norm=float(roll_affinity_norm),
                )

            for p_id in range(n_p):
                org_loss_array[it, p_id] = calc_org_loss(
                    orig_bone_lens,
                    calc_bone_lens(bt[p_id], par_idxs),
                )

            org_loss_avg = float(np.mean(org_loss_array[it, :]))

            if to_opt_weight:
                with Timer(f"time_04_iter_{it}_weight"):
                    for i, name in enumerate(name_list):
                        v_indices = v_idxs_list[i]

                        v_slice = v_cb[v_indices, :]
                        ba_v_slice = ba_v_cb[:, v_indices, :]
                        mask_slice = vbw_mask_list[i]
                        prev_vbw_slice = vbw_cb[v_indices, :]

                        target_w = None
                        local_lambda_reg = 0.0

                        if use_weight_smoothness or float(weight_soft_lock) > 0.0:
                            local_lambda_reg = float(lambda_reg)

                            if use_weight_smoothness:
                                solver = laplacian_solvers[i]
                                if solver is None:
                                    fail(f"Internal error: missing Laplacian solver for mesh {name}")
                                ws = solver.solve(prev_vbw_slice)
                            else:
                                ws = prev_vbw_slice

                            target_w = (
                                (1.0 - float(weight_soft_lock)) * ws
                                + float(weight_soft_lock) * prev_vbw_slice
                            )

                        vbw_cb[v_indices, :] = opt_vbw_parallel(
                            v=v_slice,
                            p=ba_v_slice,
                            bt=bt,
                            inv_rbt=inv_rbt,
                            s=sparseness,
                            vbw_mask=mask_slice,
                            mode=int(weight_calc_mode),
                            prev_vbw=prev_vbw_slice,
                            max_workers=max_workers,
                            weight_solver=str(weight_solver),
                            target_w_full=target_w,
                            lambda_reg=float(local_lambda_reg),
                            enable_profiling=bool(enable_profiling),
                        )

            err = recon_err(v_cb, ba_v_cb, bt, inv_rbt, vbw_cb)
            loss_buffer["err"].append(float(err))
            loss_buffer["org_loss"].append(float(org_loss_avg))
            loss_buffer["jnt_factor"].append(float(jnt_factor))

            history_metrics["org_loss"][it, :] = org_loss_array[it, :]
            history_metrics["l2_err"][it, :] = recon_err_per_frame(
                v_cb,
                ba_v_cb,
                bt,
                inv_rbt,
                vbw_cb,
            )

            if (
                eval_bone_health
                and bone_health_pre is not None
                and (it % int(bone_health_every) == 0 or it == max_iter - 1)
            ):
                bh_res = bone_health_pre.evaluate(bt, bone_names=bone_names, return_maps=True)
                if bh_res.get("status") != "no_valid_bones":
                    bone_health_buffer["roll_max"].append(float(bh_res.get("roll_abs_deg_max", 0.0)))
                    bone_health_buffer["swing_max"].append(float(bh_res.get("swing_delta_max", 0.0)))
                    bone_health_buffer["roll_150"].append(float(bh_res.get("roll_over_150_ratio", 0.0)))
                    bone_health_buffer["worst_roll_bone"].append(str(bh_res.get("worst_bone_by_roll", "")))
                    bone_health_buffer["worst_swing_bone"].append(str(bh_res.get("worst_bone_by_swing", "")))

                    history_metrics["bone_roll_max"][it, :] = bh_res["roll_max_per_frame"]
                    history_metrics["bone_swing_max"][it, :] = bh_res["swing_max_per_frame"]

            if eval_mesh_health and (it % int(health_every) == 0 or it == max_iter - 1):
                for i, name in enumerate(name_list):
                    v_indices = v_idxs_list[i]
                    p_recon = get_reconstructed_sequence(
                        v_cb[v_indices, :],
                        bt,
                        inv_rbt,
                        vbw_cb[v_indices, :],
                    )
                    health_res = health_pre_list[i].evaluate(
                        recon_seq=p_recon,
                        gt_seq=ba_v_cb[:, v_indices, :],
                        level=int(health_level),
                        return_maps=True,
                    )

                    rel = health_res.get("relative", {})
                    mesh_health_buffer[name]["arap"].append(float(rel.get("arap_global_mean_ratio", 0.0)))
                    mesh_health_buffer[name]["edge"].append(float(rel.get("edge_log_abs_p99_excess", 0.0)))
                    mesh_health_buffer[name]["flip"].append(float(rel.get("normal_vs_gt_flip_ratio", 0.0)))

                    if "arap_spokes" in health_res.get("recon", {}):
                        history_metrics[f"mesh_{name}_arap"][it, :] = (
                            health_res["recon"]["arap_spokes"]["global_energy_seq"]
                        )
                    if "edge" in health_res.get("recon", {}):
                        history_metrics[f"mesh_{name}_edge"][it, :] = (
                            health_res["recon"]["edge"]["log_abs_seq"].mean(axis=1)
                        )

            if (it + 1) % int(print_every_n) == 0 or it == max_iter - 1:
                if interval_start == it:
                    prefix = f"[Iter {it:02d}]"
                else:
                    prefix = f"[Iter {interval_start:02d}-{it:02d}]"

                flush_buffers(prefix)
                interval_start = it + 1

            if save_intermediate:
                iter_outdir = output_tree / f"opt_out_iter_{it}"
                iter_outdir.mkdir(parents=True, exist_ok=True)

                for i, mesh_obj in enumerate(skdmeshes):
                    v_indices = v_idxs_list[i]
                    name = name_list[i]
                    mesh_obj_copy = dict(mesh_obj)
                    mesh_obj_copy[KEY_VERTEX_BONE_WEIGHTS] = vbw_cb[v_indices, :]
                    mesh_obj_copy[KEY_VERTEX_BONE_WEIGHTS_MASK] = vbw_mask_list[i]

                    if validate_output:
                        validate_fitbox_mesh(
                            mesh_obj_copy,
                            n_poses=n_p,
                            n_bones=n_b,
                            name=f"{name}/intermediate_iter_{it}",
                            weak_validate=bool(weak_validate),
                        )

                    dump_pkl(iter_outdir / f"{name}.pkl", mesh_obj_copy)

                anim_obj_copy = dict(anim_obj)
                anim_obj_copy[KEY_ANIMATION_POSES] = bt
                dump_pkl(iter_outdir / f"{ANIMATION_STEM}.pkl", anim_obj_copy)
                dump_pkl(iter_outdir / f"{SKELETON_STEM}.pkl", skelhier_obj)

    with Timer("time_04_final_metrics"):
        l2_err = recon_err(v_cb, ba_v_cb, bt, inv_rbt, vbw_cb)
        h_mean, h_max = recon_err_hausdorff(v_cb, ba_v_cb, bt, inv_rbt, vbw_cb)
        org_loss_final = float(np.mean(org_loss_array[-1, :])) if max_iter > 0 else 0.0

        final_metrics: dict[str, Any] = {
            "04_L2_Err": float(l2_err),
            "04_Hausdorff_Mean": float(h_mean),
            "04_Hausdorff_Max": float(h_max),
            "04_Org_Loss": float(org_loss_final),
        }

        if eval_bone_health and bone_health_pre is not None:
            bh_res = bone_health_pre.evaluate(bt, bone_names=bone_names)
            final_metrics["04_Max_Roll"] = float(bh_res.get("roll_abs_deg_max", 0.0))
            final_metrics["04_Max_Swing"] = float(bh_res.get("swing_delta_max", 0.0))

        if eval_mesh_health:
            arap_ratios = []
            edge_excesses = []
            normal_flips = []

            for i, name in enumerate(name_list):
                v_indices = v_idxs_list[i]
                p_recon = get_reconstructed_sequence(
                    v_cb[v_indices, :],
                    bt,
                    inv_rbt,
                    vbw_cb[v_indices, :],
                )
                health_res = health_pre_list[i].evaluate(
                    recon_seq=p_recon,
                    gt_seq=ba_v_cb[:, v_indices, :],
                    level=int(health_level),
                )
                rel = health_res.get("relative", {})
                arap_ratios.append(float(rel.get("arap_global_mean_ratio", 0.0)))
                edge_excesses.append(float(rel.get("edge_log_abs_p99_excess", 0.0)))
                normal_flips.append(float(rel.get("normal_vs_gt_flip_ratio", 0.0)))

            final_metrics["04_ARAP_Ratio"] = float(np.mean(arap_ratios)) if arap_ratios else 0.0
            final_metrics["04_Edge_Excess"] = float(np.mean(edge_excesses)) if edge_excesses else 0.0
            final_metrics["04_Normal_Flip"] = float(np.mean(normal_flips)) if normal_flips else 0.0

        for key, value in final_metrics.items():
            record_metric(key, value)

        record_metric("04_Frame_Count", int(n_p))
        record_metric("04_Bone_Count", int(n_b))
        record_metric("04_Mesh_Count", int(len(name_list)))
        record_metric("04_Vertex_Count_Total", int(v_cb.shape[0]))
        record_metric("04_Iter", int(max_iter))

    with Timer("time_04_data_output"):
        final_outdir = output_tree / "opt_out"
        final_outdir.mkdir(parents=True, exist_ok=True)

        output_mesh_paths: dict[str, str] = {}

        for i, mesh_obj in enumerate(skdmeshes):
            v_indices = v_idxs_list[i]
            name = name_list[i]

            mesh_obj_out = dict(mesh_obj)
            mesh_obj_out[KEY_VERTEX_BONE_WEIGHTS] = vbw_cb[v_indices, :]
            mesh_obj_out[KEY_VERTEX_BONE_WEIGHTS_MASK] = vbw_mask_list[i]

            if validate_output:
                validate_fitbox_mesh(
                    mesh_obj_out,
                    n_poses=n_p,
                    n_bones=n_b,
                    name=f"{name}/output",
                    weak_validate=bool(weak_validate),
                )

            out_mesh_path = final_outdir / f"{name}.pkl"
            dump_pkl(out_mesh_path, mesh_obj_out)
            output_mesh_paths[name] = str(out_mesh_path)

        anim_obj_out = dict(anim_obj)
        anim_obj_out[KEY_ANIMATION_POSES] = bt

        if validate_output:
            validate_skeleton_and_animation(anim_obj_out, skelhier_obj)

        out_anim_path = final_outdir / f"{ANIMATION_STEM}.pkl"
        out_skel_path = final_outdir / f"{SKELETON_STEM}.pkl"
        out_history_path = final_outdir / "metrics_history_per_frame.npz"

        dump_pkl(out_anim_path, anim_obj_out)
        dump_pkl(out_skel_path, skelhier_obj)
        np.savez_compressed(out_history_path, **history_metrics)

        log(f"All optimized data saved to: {final_outdir}")
        log(f"Detailed history metrics saved to: {out_history_path}")

    summary = {
        "schema": "ogd_lbs_ssdr_step04_optimize_weight_and_bone_trans_v1",
        "input_tree": str(input_tree),
        "output_tree": str(output_tree),
        "opt_out": str(final_outdir),
        "inputs": {
            "animation": str(anim_path),
            "skeleton": str(skelhier_path),
            "meshes": mesh_paths_by_name,
        },
        "outputs": {
            "animation": str(out_anim_path),
            "skeleton": str(out_skel_path),
            "meshes": output_mesh_paths,
            "metrics_history_per_frame": str(out_history_path),
        },
        "counts": {
            "frames": int(n_p),
            "bones": int(n_b),
            "meshes": int(len(name_list)),
            "vertices_total": int(v_cb.shape[0]),
        },
        "config": {
            "baked_schedule": baked_schedule_path,
            "print_every_n": int(print_every_n),
            "iter": int(max_iter),
            "org_type": int(org_type),
            "opt_weight": bool(opt_weight),
            "input_poses": bool(input_poses),
            "use_weight_mask": bool(use_weight_mask),
            "weight_solver": str(weight_solver),
            "use_weight_smoothness": bool(use_weight_smoothness),
            "weights_smooth_step": float(weights_smooth_step),
            "weight_soft_lock": float(weight_soft_lock),
            "lambda_reg": float(lambda_reg),
            "bt_covariance_mode": str(bt_covariance_mode),
            "trans_affine": float(trans_affine),
            "trans_affine_norm": float(trans_affine_norm),
            "roll_affinity": float(roll_affinity),
            "roll_affinity_norm": float(roll_affinity_norm),
            "eval_mesh_health": bool(eval_mesh_health),
            "health_level": int(health_level),
            "health_every": int(health_every),
            "health_weight_scheme": str(health_weight_scheme),
            "eval_bone_health": bool(eval_bone_health),
            "bone_health_every": int(bone_health_every),
            "max_workers": max_workers,
            "validate_output": bool(validate_output),
            "weak_validate": bool(weak_validate),
            "save_intermediate": bool(save_intermediate),
        },
        "final_metrics": final_metrics,
    }

    if write_summary_file:
        summary_path = output_tree / "step04_optimize_summary.json"
        write_summary(summary_path, summary)
        log(f"Step 04 summary written: {summary_path}")
        summary["summary_path"] = str(summary_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSDR Step 04: optimize skinning weights and bone transforms."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Primary input. Normally work_dir or work_dir/data3. "
            "With explicit --output_dir, this may also be a custom input tree."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Explicit output tree. If omitted, Step 04 writes to <work_dir>/data4.",
    )
    parser.add_argument(
        "-m",
        "--metrics_dir",
        type=str,
        default=None,
        help="Optional directory to save metrics_04.json.",
    )
    parser.add_argument(
        "--baked_schedule",
        type=str,
        default=None,
        help="Optional baked schedule JSON.",
    )

    parser.add_argument("--print_every_n", type=int, default=1)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--org_type", type=int, default=3)
    parser.add_argument("--opt_weight", action="store_true")
    parser.add_argument("--input_poses", action="store_true")
    parser.add_argument("--use_weight_mask", action="store_true")

    parser.add_argument("--weight_solver", choices=["legacy", "simplex"], default="legacy")
    parser.add_argument("--use_weight_smoothness", action="store_true")
    parser.add_argument("--weights_smooth_step", type=float, default=1.0)
    parser.add_argument("--weight_soft_lock", type=float, default=0.0)
    parser.add_argument("--lambda_reg", type=float, default=0.1)

    parser.add_argument("--bt_covariance_mode", choices=["legacy", "corrected"], default="corrected")
    parser.add_argument("--trans_affine", type=float, default=10.0)
    parser.add_argument("--trans_affine_norm", type=float, default=4.0)
    parser.add_argument("--roll_affinity", type=float, default=0.0)
    parser.add_argument("--roll_affinity_norm", type=float, default=2.0)

    parser.add_argument("--eval_mesh_health", action="store_true")
    parser.add_argument("--health_level", type=int, default=1)
    parser.add_argument("--health_every", type=int, default=5)
    parser.add_argument(
        "--health_weight_scheme",
        type=str,
        default="uniform",
        choices=["uniform", "cotangent_clamped"],
    )

    parser.add_argument("--eval_bone_health", action="store_true")
    parser.add_argument("--bone_health_every", type=int, default=5)

    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--validate_output", action="store_true")
    parser.add_argument("--weak_validate", action="store_true")
    parser.add_argument("--save_intermediate", action="store_true")

    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--allow_merge", action="store_true")
    parser.add_argument("--no_summary", action="store_true")

    args = parser.parse_args(script_argv())

    if int(args.iter) <= 0:
        fail(f"--iter must be positive, got {args.iter}")

    if int(args.print_every_n) <= 0:
        fail(f"--print_every_n must be positive, got {args.print_every_n}")

    resolution = resolve_ssdr_stage_dirs(
        step_id="04",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="SSDR Step 04: data3 -> data4 Weight/Bone Optimization",
    )

    print_ssdr_resolution_info(resolution)

    try:
        validate_ssdr_stage_resolution(resolution)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    log("=" * 80)
    log("SSDR Step 04: data3 -> data4 Weight/Bone Optimization")
    log(f"work_dir          : {resolution.work_dir}")
    log(f"input_tree        : {resolution.input_tree}")
    log(f"output_tree       : {resolution.output_tree}")
    log(f"input_source      : {resolution.input_source}")
    log(f"output_source     : {resolution.output_source}")
    log(f"baked_schedule    : {args.baked_schedule}")
    log(f"iter              : {int(args.iter)}")
    log(f"org_type          : {int(args.org_type)}")
    log(f"opt_weight        : {bool(args.opt_weight)}")
    log(f"input_poses       : {bool(args.input_poses)}")
    log(f"use_weight_mask   : {bool(args.use_weight_mask)}")
    log(f"weight_solver     : {args.weight_solver}")
    log(f"max_workers       : {args.max_workers}")
    log(f"overwrite_output  : {bool(args.overwrite_output)}")
    log(f"allow_merge       : {bool(args.allow_merge)}")
    log("=" * 80)

    with Timer("time_04_total"):
        summary = optimize_weight_and_bone_trans_tree(
            input_tree=resolution.input_tree,
            output_tree=resolution.output_tree,
            baked_schedule_path=args.baked_schedule,
            print_every_n=int(args.print_every_n),
            max_iter=int(args.iter),
            org_type=int(args.org_type),
            opt_weight=bool(args.opt_weight),
            input_poses=bool(args.input_poses),
            use_weight_mask=bool(args.use_weight_mask),
            weight_solver=str(args.weight_solver),
            use_weight_smoothness=bool(args.use_weight_smoothness),
            weights_smooth_step=float(args.weights_smooth_step),
            weight_soft_lock=float(args.weight_soft_lock),
            lambda_reg=float(args.lambda_reg),
            bt_covariance_mode=str(args.bt_covariance_mode),
            trans_affine=float(args.trans_affine),
            trans_affine_norm=float(args.trans_affine_norm),
            roll_affinity=float(args.roll_affinity),
            roll_affinity_norm=float(args.roll_affinity_norm),
            eval_mesh_health=bool(args.eval_mesh_health),
            health_level=int(args.health_level),
            health_every=int(args.health_every),
            health_weight_scheme=str(args.health_weight_scheme),
            eval_bone_health=bool(args.eval_bone_health),
            bone_health_every=int(args.bone_health_every),
            max_workers=args.max_workers,
            enable_profiling=bool(args.enable_profiling),
            validate_output=bool(args.validate_output),
            weak_validate=bool(args.weak_validate),
            save_intermediate=bool(args.save_intermediate),
            overwrite_output=bool(args.overwrite_output),
            allow_merge=bool(args.allow_merge),
            write_summary_file=not bool(args.no_summary),
        )

    if args.metrics_dir:
        dump_metrics(Path(args.metrics_dir) / "metrics_04.json")

    log("SSDR Step 04 complete.")
    log(f"frames          : {summary['counts']['frames']}")
    log(f"bones           : {summary['counts']['bones']}")
    log(f"meshes          : {summary['counts']['meshes']}")
    log(f"vertices_total  : {summary['counts']['vertices_total']}")
    log(f"04_L2_Err       : {summary['final_metrics'].get('04_L2_Err')}")
    log(f"04_Hausdorff_Max: {summary['final_metrics'].get('04_Hausdorff_Max')}")
    log(f"opt_out         : {summary['opt_out']}")
    log(f"output_tree     : {resolution.output_tree}")


if __name__ == "__main__":
    main()
