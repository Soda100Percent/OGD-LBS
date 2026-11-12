#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate OGD-LBS OFAT (delta, gamma, s_max, tau, omega_b) and 4-fold CV
(ssdr12 vs ssdr14 vs ssdr14v with multi-level paired t-tests and PENDING
placeholders) reports from pd-rew-v2 PipelinePool.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.request

import numpy as np

HERE = Path(__file__).resolve().parent
OGD_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_PD_REW_ROOT = Path("/workspace/pd-rew-v2")
DEFAULT_POOL = Path("/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1")
DEFAULT_BINDING = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_repo_binding.yaml"
DEFAULT_TABLE = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_experiment_table.yaml"
DEFAULT_ALIASES = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_parameter_aliases.yaml"
DEFAULT_REPORT_ROOT = Path("/workspace/ogd_reports")

PARAM_CHOICE_PREFIXES = {
    "cv": ("cv-",),
    "delta": ("delta-",),
    "gamma": ("weight-",),
    "s_max": ("s_max-",),
    "tau": ("tau-",),
    "omega_b": ("boundary_weight-",),
}

PARAM_TITLES = {
    "cv": "OGD-LBS 4-Fold Cross-Validation & Scheduler Ablation (ssdr12 / ssdr14 / ssdr14v) — Paired t-Test Report",
    "delta": "OGD-LBS VDB Offset (δ) Sweep — pd-rew-v2 Result Pool",
    "gamma": "OGD-LBS Weight Allocation (α/β/γ) Sweep — pd-rew-v2 Result Pool",
    "s_max": "OGD-LBS Max Bones per Vertex (s_max) Sweep — pd-rew-v2 Result Pool",
    "tau": "OGD-LBS Weight Pruning Threshold (τ) Sweep — pd-rew-v2 Result Pool",
    "omega_b": "OGD-LBS QEM Boundary Penalty (ω_b) Sweep — pd-rew-v2 Result Pool",
}

ORG_TYPE_ALIASES = {
    0: "ssdr12",
    1: "ssdr14",
    3: "ssdr14v",
}

FRAME_NPZ_METRICS = [
    ("04_L2_Err", "l2_err"),
    ("04_Org_Loss", "org_loss"),
    ("04_Hausdorff_Mean", "hausdorff"),
    ("04_ARAP_Ratio", "arap_ratio"),
    ("04_Max_Roll", "bone_roll_max"),
    ("04_Max_Swing", "bone_swing_max"),
]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def strict_json_write(path: Path, obj: Any) -> None:
    def _sanitize(v: Any) -> Any:
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        if isinstance(v, dict):
            return {str(k): _sanitize(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_sanitize(x) for x in v]
        return v

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json_object(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(obj, dict), f"JSON root must be object: {path}")
    return obj


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_pool_data_dir(pool: Path, value: Any) -> Path:
    require(isinstance(value, str) and value, f"Invalid pool data_dir: {value!r}")
    rel = PurePosixPath(value)
    require(
        not rel.is_absolute()
        and ".." not in rel.parts
        and str(rel) == value
        and len(rel.parts) == 2
        and rel.parts[0] == "results"
        and rel.parts[1].startswith("r"),
        f"Unsafe or unexpected pool data_dir: {value!r}",
    )
    return pool / rel


def materialize_pool_file(*, datalad_api: Any, pool: Path, path: Path, required: bool) -> bool:
    if not os.path.lexists(path):
        if required:
            raise RuntimeError(f"Required pool report file is not tracked: {path}")
        return False
    datalad_api.get(path=str(path), dataset=str(pool), result_renderer="disabled", on_failure="stop")
    if not path.is_file():
        if required:
            raise RuntimeError(f"Required pool report file is unavailable: {path}")
        return False
    return True


def _row_matches_param_family(row: dict[str, Any], param_family: str) -> bool:
    choices = row.get("sweep_choices") or {}
    w_choice = str(choices.get("weights") or "")
    prefixes = PARAM_CHOICE_PREFIXES[param_family]
    return any(w_choice.startswith(pfx) for pfx in prefixes)


def case_from_expanded_row(
    row: dict[str, Any],
    *,
    param_family: str,
    stage04_data_dir: str | None,
    case_status: str = "READY",
    split_override: str | None = None,
) -> dict[str, Any]:
    flat = row.get("flat_parameters") or {}
    choices = row.get("sweep_choices") or {}

    model = flat.get("data_prep_handoff.model_name") or flat.get("ssdr_handoff.model_name")
    alpha = flat.get("ssdr_stage02.par_weight", flat.get("stage02.par_weight", 0.5))
    beta = flat.get("ssdr_stage02.bone_weight", flat.get("stage02.bone_weight", 0.3))
    gamma = flat.get("ssdr_stage02.child_weight", flat.get("stage02.child_weight", 0.2))
    omega_b = flat.get("ssdr_stage03.boundary_weight", flat.get("stage03.boundary_weight", 1000.0))
    s_max = flat.get("ssdr_stage04.s_max", flat.get("stage04.s_max", 4))
    mask_mode = flat.get("ssdr_stage04.mask_mode", flat.get("stage04.mask_mode", "adaptive"))
    mask_threshold = flat.get("ssdr_stage04.mask_threshold", flat.get("stage04.mask_threshold", -1.0))
    tau = "adaptive" if str(mask_mode).lower() == "adaptive" else mask_threshold
    delta = flat.get("data_prep_stage02.offset")
    org_type = int(flat.get("ssdr_stage04.org_type", 3))
    kfold = int(flat.get("ssdr_stage04.kfold", 0))
    fold_index = int(flat.get("ssdr_stage04.fold_index", -1))
    sched_alias = ORG_TYPE_ALIASES.get(org_type, f"org{org_type}")

    if param_family == "cv":
        split_tag = split_override or "test"
        swept_param = "fold_sched"
        swept_value = f"f{fold_index}_{sched_alias}"
    else:
        split_tag = "full"
        swept_param = param_family
        value_by_family = {
            "delta": delta,
            "gamma": gamma,
            "s_max": s_max,
            "tau": tau,
            "omega_b": omega_b,
        }
        swept_value = value_by_family[param_family]
        require(swept_value is not None, f"{row.get('row_id')}: missing value for {param_family}")

    return {
        "schema": "ogd_lbs_ofat_case_v1",
        "case_status": case_status,
        "model": model,
        "param": swept_param,
        "value": swept_value,
        "fold_index": fold_index,
        "kfold": kfold,
        "split": split_tag,
        "org_type": org_type,
        "sched_alias": sched_alias,
        "stage": "data_prep_prep02" if param_family == "delta" else f"ssdr_{param_family}",
        "case_id": f"{row.get('row_id')}_{split_tag}" if param_family == "cv" else row.get("row_id"),
        "effective_params": {
            "delta": delta,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "omega_b": omega_b,
            "s_max": s_max,
            "tau": tau,
            "org_type": org_type,
            "sched_alias": sched_alias,
            "kfold": kfold,
            "fold_index": fold_index,
        },
        "dataset": {"model": model},
        "pd_rew": {
            "row_id": row.get("row_id"),
            "row_index": row.get("row_index"),
            "sweep_choices": choices,
            "sweep_values": row.get("sweep_values") or {},
            "stage04_data_dir": stage04_data_dir,
        },
    }


def _load_final_frame_vector(npz_path: Path | None, npz_key: str) -> np.ndarray | None:
    if npz_path is None or not npz_path.is_file():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            if npz_key not in npz.files:
                return None
            arr = np.asarray(npz[npz_key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] > 0:
                return arr[-1, :].copy()
            if arr.ndim == 1 and arr.shape[0] > 0:
                return arr.copy()
    except Exception:
        return None
    return None


def compute_cv_multilevel_ttests(
    *,
    cv_records: list[dict[str, Any]],
    aggregate_metrics_mod: Any,
) -> list[dict[str, Any]]:
    """
    Build comprehensive paired t-test rows comparing:
      - ssdr12 (org_type=0) vs ssdr14v (org_type=3, baseline)
      - ssdr14 (org_type=1) vs ssdr14v (org_type=3, baseline)
    across 3 analysis levels:
      1. Pooled Out-of-Sample Frame-Aligned (per model & all models combined on held-out test frames)
      2. Per-Fold Frame-Aligned (within each model & fold on held-out test frames)
      3. Fold-Level (n<=20) and Model-Level (n=5) Summary Paired t-tests (on both test and train splits)
    """
    results: list[dict[str, Any]] = []

    # Index records by (split, model, fold_index, sched_alias)
    by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    models = sorted({str(r["model"]) for r in cv_records})
    folds = sorted({int(r["fold_index"]) for r in cv_records})

    for rec in cv_records:
        key = (str(rec["split"]), str(rec["model"]), int(rec["fold_index"]), str(rec["sched_alias"]))
        by_key[key] = rec

    # Level 1: Pooled Out-of-Sample Frame-Aligned Paired t-test (Test split, per model & ALL_MODELS)
    for cand_sched in ("ssdr12", "ssdr14"):
        for metric_name, npz_key in FRAME_NPZ_METRICS:
            all_base_vecs: list[np.ndarray] = []
            all_cand_vecs: list[np.ndarray] = []
            all_missing_folds = 0

            for model in models:
                m_base_vecs: list[np.ndarray] = []
                m_cand_vecs: list[np.ndarray] = []
                m_missing = 0
                for fold_idx in folds:
                    base_rec = by_key.get(("test", model, fold_idx, "ssdr14v"))
                    cand_rec = by_key.get(("test", model, fold_idx, cand_sched))
                    if (
                        base_rec is None
                        or cand_rec is None
                        or base_rec["case_status"] != "READY"
                        or cand_rec["case_status"] != "READY"
                    ):
                        m_missing += 1
                        all_missing_folds += 1
                        continue
                    vb = _load_final_frame_vector(base_rec.get("history_npz_path"), npz_key)
                    vc = _load_final_frame_vector(cand_rec.get("history_npz_path"), npz_key)
                    if vb is None or vc is None or vb.shape != vc.shape:
                        m_missing += 1
                        all_missing_folds += 1
                        continue
                    m_base_vecs.append(vb)
                    m_cand_vecs.append(vc)
                    all_base_vecs.append(vb)
                    all_cand_vecs.append(vc)

                a = np.concatenate(m_base_vecs) if m_base_vecs else np.zeros((0,), dtype=np.float64)
                b = np.concatenate(m_cand_vecs) if m_cand_vecs else np.zeros((0,), dtype=np.float64)
                st = aggregate_metrics_mod.compute_paired_stats(a, b)
                results.append({
                    "analysis_level": "1_pooled_test_frames_by_model",
                    "split": "test",
                    "model": model,
                    "fold_index": "ALL_4_FOLDS",
                    "candidate": cand_sched,
                    "baseline": "ssdr14v",
                    "metric": metric_name,
                    "n_pairs": int(a.size),
                    "pending_folds": int(m_missing),
                    "baseline_mean": float(np.mean(a)) if a.size else "",
                    "group_mean": float(np.mean(b)) if b.size else "",
                    "mean_delta_group_minus_baseline": float(np.mean(b - a)) if a.size else "",
                    "ci95_low": st["ci95_low"],
                    "ci95_high": st["ci95_high"],
                    "cohens_dz": st["cohens_dz"],
                    "t_stat": st["t_stat"],
                    "p_value": st["p_value"],
                })

            a_all = np.concatenate(all_base_vecs) if all_base_vecs else np.zeros((0,), dtype=np.float64)
            b_all = np.concatenate(all_cand_vecs) if all_cand_vecs else np.zeros((0,), dtype=np.float64)
            st_all = aggregate_metrics_mod.compute_paired_stats(a_all, b_all)
            results.append({
                "analysis_level": "1_pooled_test_frames_all_models",
                "split": "test",
                "model": "ALL_5_MODELS",
                "fold_index": "ALL_4_FOLDS",
                "candidate": cand_sched,
                "baseline": "ssdr14v",
                "metric": metric_name,
                "n_pairs": int(a_all.size),
                "pending_folds": int(all_missing_folds),
                "baseline_mean": float(np.mean(a_all)) if a_all.size else "",
                "group_mean": float(np.mean(b_all)) if b_all.size else "",
                "mean_delta_group_minus_baseline": float(np.mean(b_all - a_all)) if a_all.size else "",
                "ci95_low": st_all["ci95_low"],
                "ci95_high": st_all["ci95_high"],
                "cohens_dz": st_all["cohens_dz"],
                "t_stat": st_all["t_stat"],
                "p_value": st_all["p_value"],
            })

    # Level 2: Per-Fold Frame-Aligned Paired t-test (Test split, within each model & fold_index)
    for model in models:
        for fold_idx in folds:
            for cand_sched in ("ssdr12", "ssdr14"):
                base_rec = by_key.get(("test", model, fold_idx, "ssdr14v"))
                cand_rec = by_key.get(("test", model, fold_idx, cand_sched))
                for metric_name, npz_key in FRAME_NPZ_METRICS:
                    if (
                        base_rec is None
                        or cand_rec is None
                        or base_rec["case_status"] != "READY"
                        or cand_rec["case_status"] != "READY"
                    ):
                        results.append({
                            "analysis_level": "2_per_fold_test_frames",
                            "split": "test",
                            "model": model,
                            "fold_index": str(fold_idx),
                            "candidate": cand_sched,
                            "baseline": "ssdr14v",
                            "metric": metric_name,
                            "n_pairs": 0,
                            "pending_folds": 1,
                            "baseline_mean": "PENDING",
                            "group_mean": "PENDING",
                            "mean_delta_group_minus_baseline": "PENDING",
                            "ci95_low": float("nan"),
                            "ci95_high": float("nan"),
                            "cohens_dz": float("nan"),
                            "t_stat": float("nan"),
                            "p_value": float("nan"),
                        })
                        continue
                    vb = _load_final_frame_vector(base_rec.get("history_npz_path"), npz_key)
                    vc = _load_final_frame_vector(cand_rec.get("history_npz_path"), npz_key)
                    if vb is None or vc is None or vb.shape != vc.shape:
                        continue
                    st = aggregate_metrics_mod.compute_paired_stats(vb, vc)
                    results.append({
                        "analysis_level": "2_per_fold_test_frames",
                        "split": "test",
                        "model": model,
                        "fold_index": str(fold_idx),
                        "candidate": cand_sched,
                        "baseline": "ssdr14v",
                        "metric": metric_name,
                        "n_pairs": int(vb.size),
                        "pending_folds": 0,
                        "baseline_mean": float(np.mean(vb)),
                        "group_mean": float(np.mean(vc)),
                        "mean_delta_group_minus_baseline": float(np.mean(vc - vb)),
                        "ci95_low": st["ci95_low"],
                        "ci95_high": st["ci95_high"],
                        "cohens_dz": st["cohens_dz"],
                        "t_stat": st["t_stat"],
                        "p_value": st["p_value"],
                    })

    # Level 3: Fold-level (n<=20) and Model-level (n=5) summary paired t-tests
    summary_metrics = [
        "04_L2_Err",
        "04_Org_Loss",
        "04_Hausdorff_Mean",
        "04_Hausdorff_Max",
        "04_ARAP_Ratio",
        "04_Max_Roll",
        "04_Max_Swing",
    ]
    for split_name in ("test", "train"):
        for cand_sched in ("ssdr12", "ssdr14"):
            for metric_name in summary_metrics:
                fold_a: list[float] = []
                fold_b: list[float] = []
                model_a: list[float] = []
                model_b: list[float] = []
                missing_pairs = 0

                for model in models:
                    m_a: list[float] = []
                    m_b: list[float] = []
                    for fold_idx in folds:
                        br = by_key.get((split_name, model, fold_idx, "ssdr14v"))
                        cr = by_key.get((split_name, model, fold_idx, cand_sched))
                        if (
                            br is None
                            or cr is None
                            or br["case_status"] != "READY"
                            or cr["case_status"] != "READY"
                        ):
                            missing_pairs += 1
                            continue
                        va = aggregate_metrics_mod.maybe_float(br["metrics"].get(metric_name))
                        vb = aggregate_metrics_mod.maybe_float(cr["metrics"].get(metric_name))
                        if va is None or vb is None:
                            missing_pairs += 1
                            continue
                        fold_a.append(va)
                        fold_b.append(vb)
                        m_a.append(va)
                        m_b.append(vb)
                    if m_a and m_b:
                        model_a.append(float(np.mean(m_a)))
                        model_b.append(float(np.mean(m_b)))

                arr_fa = np.asarray(fold_a, dtype=np.float64)
                arr_fb = np.asarray(fold_b, dtype=np.float64)
                st_f = aggregate_metrics_mod.compute_paired_stats(arr_fa, arr_fb)
                results.append({
                    "analysis_level": "3_fold_level_summary_paired",
                    "split": split_name,
                    "model": "ALL_5_MODELS",
                    "fold_index": "PAIR_BY_MODEL_FOLD",
                    "candidate": cand_sched,
                    "baseline": "ssdr14v",
                    "metric": metric_name,
                    "n_pairs": int(arr_fa.size),
                    "pending_folds": int(missing_pairs),
                    "baseline_mean": float(np.mean(arr_fa)) if arr_fa.size else "",
                    "group_mean": float(np.mean(arr_fb)) if arr_fb.size else "",
                    "mean_delta_group_minus_baseline": float(np.mean(arr_fb - arr_fa)) if arr_fa.size else "",
                    "ci95_low": st_f["ci95_low"],
                    "ci95_high": st_f["ci95_high"],
                    "cohens_dz": st_f["cohens_dz"],
                    "t_stat": st_f["t_stat"],
                    "p_value": st_f["p_value"],
                })

                arr_ma = np.asarray(model_a, dtype=np.float64)
                arr_mb = np.asarray(model_b, dtype=np.float64)
                st_m = aggregate_metrics_mod.compute_paired_stats(arr_ma, arr_mb)
                results.append({
                    "analysis_level": "4_model_level_4fold_mean_paired",
                    "split": split_name,
                    "model": "ALL_5_MODELS",
                    "fold_index": "MEAN_OVER_4_FOLDS",
                    "candidate": cand_sched,
                    "baseline": "ssdr14v",
                    "metric": metric_name,
                    "n_pairs": int(arr_ma.size),
                    "pending_folds": int(missing_pairs),
                    "baseline_mean": float(np.mean(arr_ma)) if arr_ma.size else "",
                    "group_mean": float(np.mean(arr_mb)) if arr_mb.size else "",
                    "mean_delta_group_minus_baseline": float(np.mean(arr_mb - arr_ma)) if arr_ma.size else "",
                    "ci95_low": st_m["ci95_low"],
                    "ci95_high": st_m["ci95_high"],
                    "cohens_dz": st_m["cohens_dz"],
                    "t_stat": st_m["t_stat"],
                    "p_value": st_m["p_value"],
                })

    aggregate_metrics_mod.apply_holm_bonferroni(
        results,
        family_keys=["analysis_level", "split", "model", "fold_index", "metric"],
    )
    return results


def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.241.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def stop_existing_server_on_port(port: int) -> list[int]:
    port_hex = f"{int(port):04X}"
    target_inodes: set[str] = set()
    for net_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        p = Path(net_file)
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_addr, state, inode = parts[1], parts[3], parts[9]
            if state == "0A" and local_addr.upper().endswith(f":{port_hex}"):
                target_inodes.add(inode)

    killed_pids: list[int] = []
    if not target_inodes:
        return killed_pids

    my_pid = os.getpid()
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
        except ValueError:
            continue
        if pid == my_pid:
            continue
        fd_dir = proc_dir / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    link = os.readlink(fd)
                except OSError:
                    continue
                for ino in target_inodes:
                    if f"socket:[{ino}]" in link:
                        killed_pids.append(pid)
                        break
        except OSError:
            continue

    killed_pids = sorted(set(killed_pids))
    for pid in killed_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        alive = [pid for pid in killed_pids if Path(f"/proc/{pid}").exists()]
        if not alive:
            break
        time.sleep(0.1)

    for pid in killed_pids:
        if Path(f"/proc/{pid}").exists():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return killed_pids


def start_daemon_http_server(report_dir: Path, html_filename: str, port: int = 8080) -> dict[str, Any]:
    killed = stop_existing_server_on_port(port)
    if killed:
        print(f"[server] Replaced existing process(es) on port {port}: {killed}", flush=True)

    log_file = report_dir / "http_server.log"
    pid_file = report_dir / "http_server.pid"

    with log_file.open("a", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                "0.0.0.0",
                "--directory",
                str(report_dir),
            ],
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )

    pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")

    url_local = f"http://127.0.0.1:{port}/{html_filename}"
    last_err = None
    for _ in range(30):
        try:
            with urllib.request.urlopen(url_local, timeout=2.0) as resp:
                if resp.status == 200:
                    break
        except Exception as exc:
            last_err = exc
            time.sleep(0.15)
    else:
        raise RuntimeError(f"HTTP server on port {port} failed healthcheck: {last_err}")

    lan_ip = get_lan_ip()
    return {
        "pid": proc.pid,
        "port": port,
        "replaced_pids": killed,
        "lan_ip": lan_ip,
        "report_url": f"http://{lan_ip}:{port}/{html_filename}",
        "index_url": f"http://{lan_ip}:{port}/index.html",
        "log_file": str(log_file),
    }


def generate_param_report(
    *,
    param_family: str,
    pd_rew_root: Path = DEFAULT_PD_REW_ROOT,
    pool: Path = DEFAULT_POOL,
    binding_path: Path = DEFAULT_BINDING,
    table_path: Path = DEFAULT_TABLE,
    aliases_path: Path = DEFAULT_ALIASES,
    report_root: Path = DEFAULT_REPORT_ROOT,
    output_dir: Path | None = None,
    title: str | None = None,
    require_history: bool = True,
    allow_missing: bool = True,
) -> dict[str, Any]:
    src = pd_rew_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import datalad.api as dl
    import aggregate_metrics
    import generate_report
    from add_pdrew_report_downloads import build_downloads
    from pd_rew_v2.experiment import expand_experiment_table, query_cache_status
    from pd_rew_v2.pool import PipelinePool

    pipe = PipelinePool(pool)
    pipe.verify(check_annex_hashes=False)

    expanded = expand_experiment_table(binding_path, table_path)
    final_stage = expanded["stages"][-1]["name"]
    cache_status = query_cache_status(
        binding_path,
        expanded,
        pipe,
        start_stage=final_stage,
        end_stage=final_stage,
    )

    cache_rows = {r["row_id"]: r for r in cache_status["rows"]}
    selected_rows = [r for r in expanded["rows"] if _row_matches_param_family(r, param_family)]
    require(bool(selected_rows), f"No rows matched parameter family {param_family!r}")

    missing_selected = [
        r["row_id"]
        for r in selected_rows
        if cache_rows[r["row_id"]]["stage_statuses"][final_stage]["status"] != "HIT"
    ]
    if missing_selected and not allow_missing:
        raise RuntimeError(
            f"{param_family}: {len(missing_selected)} row(s) not HIT in pool: {missing_selected}"
        )

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        output_dir = report_root / f"pdrew_{param_family}_report_{timestamp}"
    else:
        output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compat_root = output_dir / "compat_input"
    aggregate_dir = output_dir / "aggregate"
    html_filename = f"{param_family.capitalize()}_Report.html"
    report_html = output_dir / html_filename
    compat_root.mkdir()
    aggregate_dir.mkdir()

    source_records: list[dict[str, Any]] = []
    cv_records: list[dict[str, Any]] = []

    for row in selected_rows:
        row_id = row["row_id"]
        st_info = cache_rows[row_id]["stage_statuses"][final_stage]
        is_hit = st_info["status"] == "HIT"

        if not is_hit:
            case_cfg = case_from_expanded_row(
                row,
                param_family=param_family,
                stage04_data_dir=None,
                case_status="PENDING",
                split_override="test" if param_family == "cv" else None,
            )
            model = str(case_cfg["model"])
            val = case_cfg["value"]
            val_slug = str(val).replace(".", "p").replace("-", "m")
            case_root = compat_root / model / f"sweep_{param_family}" / f"{param_family}_{val_slug}"
            case_root.mkdir(parents=True, exist_ok=True)
            strict_json_write(case_root / "case_config.json", case_cfg)

            source_records.append({
                "row_id": row_id,
                "status": "PENDING",
                "model": model,
                "param": case_cfg["param"],
                "value": val,
                "stage04_data_dir": None,
                "compatibility_case_root": str(case_root),
            })
            if param_family == "cv":
                for sp_name in ("test", "train"):
                    cv_records.append({
                        "row_id": row_id,
                        "case_status": "PENDING",
                        "split": sp_name,
                        "model": model,
                        "fold_index": int(case_cfg["fold_index"]),
                        "org_type": int(case_cfg["org_type"]),
                        "sched_alias": str(case_cfg["sched_alias"]),
                        "metrics": {},
                        "history_npz_path": None,
                    })
            print(f"[pending-placeholder] {row_id} model={model} {case_cfg['param']}={val} -> PENDING", flush=True)
            continue

        data_dir = st_info["data_dir"]
        data4_dir = safe_pool_data_dir(pool, data_dir) / "data4"

        if param_family == "cv":
            train_sum_src = data4_dir / "step04_optimize_summary.json"
            test_sum_src = data4_dir / "step04_test_summary.json"
            train_hist_src = data4_dir / "opt_out/metrics_history_per_frame.npz"
            test_hist_src = data4_dir / "opt_out_test/metrics_history_per_frame.npz"

            materialize_pool_file(datalad_api=dl, pool=pool, path=train_sum_src, required=True)
            materialize_pool_file(datalad_api=dl, pool=pool, path=test_sum_src, required=True)
            train_hist_ok = materialize_pool_file(
                datalad_api=dl, pool=pool, path=train_hist_src, required=require_history
            )
            test_hist_ok = materialize_pool_file(
                datalad_api=dl, pool=pool, path=test_hist_src, required=require_history
            )

            train_sum = load_json_object(train_sum_src)
            test_sum = load_json_object(test_sum_src)
            train_metrics = train_sum.get("final_metrics") or {}
            test_metrics = test_sum.get("final_metrics") or {}

            # Primary chart/table row in compat_root uses the held-out TEST split
            case_cfg = case_from_expanded_row(
                row,
                param_family="cv",
                stage04_data_dir=data_dir,
                case_status="READY",
                split_override="test",
            )
            model = str(case_cfg["model"])
            val = case_cfg["value"]
            val_slug = str(val).replace(".", "p").replace("-", "m")
            case_root = compat_root / model / "sweep_cv" / f"cv_{val_slug}"
            case_root.mkdir(parents=True, exist_ok=True)

            strict_json_write(case_root / "case_config.json", case_cfg)
            strict_json_write(case_root / "metrics_04.json", test_metrics)

            summary_dst = case_root / "data4/step04_optimize_summary.json"
            summary_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(test_sum_src, summary_dst)
            if test_hist_ok:
                history_dst = case_root / "data4/opt_out/metrics_history_per_frame.npz"
                history_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(test_hist_src, history_dst)

            cv_records.append({
                "row_id": row_id,
                "case_status": "READY",
                "split": "test",
                "model": model,
                "fold_index": int(case_cfg["fold_index"]),
                "org_type": int(case_cfg["org_type"]),
                "sched_alias": str(case_cfg["sched_alias"]),
                "metrics": test_metrics,
                "history_npz_path": test_hist_src if test_hist_ok else None,
            })
            cv_records.append({
                "row_id": row_id,
                "case_status": "READY",
                "split": "train",
                "model": model,
                "fold_index": int(case_cfg["fold_index"]),
                "org_type": int(case_cfg["org_type"]),
                "sched_alias": str(case_cfg["sched_alias"]),
                "metrics": train_metrics,
                "history_npz_path": train_hist_src if train_hist_ok else None,
            })

            source_records.append({
                "row_id": row_id,
                "status": "READY",
                "model": model,
                "param": case_cfg["param"],
                "value": val,
                "stage04_data_dir": data_dir,
                "train_summary_sha256": sha256_file(train_sum_src),
                "test_summary_sha256": sha256_file(test_sum_src),
                "compatibility_case_root": str(case_root),
            })
            print(f"[materialized-cv] {row_id} model={model} {val} -> {data_dir}", flush=True)
        else:
            summary_src = data4_dir / "step04_optimize_summary.json"
            history_src = data4_dir / "opt_out/metrics_history_per_frame.npz"
            materialize_pool_file(datalad_api=dl, pool=pool, path=summary_src, required=True)
            history_present = materialize_pool_file(
                datalad_api=dl, pool=pool, path=history_src, required=require_history
            )

            summary = load_json_object(summary_src)
            final_metrics = summary.get("final_metrics") or {}
            require(bool(final_metrics), f"{row_id}: empty final_metrics in {summary_src}")

            case_cfg = case_from_expanded_row(row, param_family=param_family, stage04_data_dir=data_dir)
            model = str(case_cfg["model"])
            val = case_cfg["value"]
            val_slug = str(val).replace(".", "p").replace("-", "m")
            case_root = compat_root / model / f"sweep_{param_family}" / f"{param_family}_{val_slug}"
            case_root.mkdir(parents=True, exist_ok=True)

            strict_json_write(case_root / "case_config.json", case_cfg)
            strict_json_write(case_root / "metrics_04.json", final_metrics)

            summary_dst = case_root / "data4/step04_optimize_summary.json"
            summary_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary_src, summary_dst)
            if history_present:
                history_dst = case_root / "data4/opt_out/metrics_history_per_frame.npz"
                history_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(history_src, history_dst)

            source_records.append({
                "row_id": row_id,
                "status": "READY",
                "model": model,
                "param": param_family,
                "value": val,
                "stage04_data_dir": data_dir,
                "summary_sha256": sha256_file(summary_src),
                "history_sha256": sha256_file(history_src) if history_present else None,
                "compatibility_case_root": str(case_root),
            })
            print(f"[materialized] {row_id} model={model} {param_family}={val} -> {data_dir}", flush=True)

    strict_json_write(
        output_dir / "pdrew_source_manifest.json",
        {
            "schema": "ogd_lbs_pdrew_report_source_v2",
            "param_family": param_family,
            "pool": str(pool),
            "binding": str(binding_path),
            "table": str(table_path),
            "row_count": len(source_records),
            "missing_row_ids": missing_selected,
            "records": source_records,
        },
    )

    if param_family == "cv":
        agg_manifest = aggregate_metrics.aggregate(
            input_root=compat_root,
            output_dir=aggregate_dir,
            mode="sweep",
            metrics_requested=None,
            group_by=["sched_alias"],
            ttest_kind="paired",
            baseline_query_text="sched_alias=ssdr14v",
            pair_by=["model", "fold_index"],
            ttest_group_by=["sched_alias"],
            where_text="",
        )
        multilevel_ttests = compute_cv_multilevel_ttests(
            cv_records=cv_records,
            aggregate_metrics_mod=aggregate_metrics,
        )
        aggregate_metrics.write_csv(aggregate_dir / "ttest_results.csv", multilevel_ttests)
        strict_json_write(aggregate_dir / "ttest_results.json", multilevel_ttests)
        aggregate_metrics.write_ttest_markdown(
            aggregate_dir / "ttest_results.md",
            multilevel_ttests,
            group_by=["analysis_level", "split", "model", "fold_index", "candidate", "baseline"],
        )
    else:
        agg_manifest = aggregate_metrics.aggregate(
            input_root=compat_root,
            output_dir=aggregate_dir,
            mode="sweep",
            metrics_requested=None,
            group_by=["model", "param", "value"] if param_family == "delta" else ["param", "value"],
            ttest_kind="none",
            baseline_query_text="",
            pair_by=None,
            ttest_group_by=None,
            where_text=f"param={param_family}",
        )

    report_title = title or PARAM_TITLES.get(param_family, f"OGD-LBS {param_family} Sweep Report")
    if missing_selected:
        report_title = f"{report_title} (Partial: {len(selected_rows) - len(missing_selected)}/{len(selected_rows)} Ready, {len(missing_selected)} Pending)"

    generate_report.generate_report(
        aggregate_dir=aggregate_dir,
        output_html=report_html,
        mode="sweep",
        title=report_title,
        default_sweep_values=None,
        model_scale_values=None,
        scaled_metrics=None,
        axis_mode=generate_report.DEFAULT_AXIS_MODE,
        pseudo_log_ratio_threshold=generate_report.DEFAULT_PSEUDO_LOG_RATIO_THRESHOLD,
    )

    strict_json_write(
        output_dir / "report_manifest.json",
        {
            "schema": "ogd_lbs_pdrew_report_v2",
            "status": "GENERATED_WITH_PLACEHOLDERS" if missing_selected else "GENERATED",
            "param_family": param_family,
            "pool": str(pool),
            "row_count": len(source_records),
            "missing_row_ids": missing_selected,
            "aggregate_dir": str(aggregate_dir),
            "report_html": str(report_html),
        },
    )
    shutil.rmtree(compat_root, ignore_errors=True)

    dl_receipt = build_downloads(output_dir, html_filename=html_filename, title=report_title)
    return {
        "param_family": param_family,
        "row_count": len(source_records),
        "missing_row_ids": missing_selected,
        "output_dir": str(output_dir),
        "html_filename": html_filename,
        "report_html": str(report_html),
        "index_html": dl_receipt["index_html"],
        "zip_path": dl_receipt["zip_path"],
        "aggregate_manifest": agg_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OGD-LBS OFAT / CV report from pd-rew-v2 pool.")
    parser.add_argument(
        "-p",
        "--param",
        choices=["cv", "delta", "gamma", "s_max", "tau", "omega_b"],
        default="cv",
        help="Parameter family to generate report for (default: cv).",
    )
    parser.add_argument("--pd-rew-root", type=Path, default=DEFAULT_PD_REW_ROOT)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--require-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--serve", action="store_true", help="Serve report in foreground on --port.")
    parser.add_argument("--serve-daemon", action="store_true", help="Serve report in background on --port.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port (default: 8080).")
    args = parser.parse_args()

    res = generate_param_report(
        param_family=args.param,
        pd_rew_root=args.pd_rew_root.expanduser().resolve(),
        pool=args.pool.expanduser().resolve(),
        binding_path=args.binding.expanduser().resolve(),
        table_path=args.table.expanduser().resolve(),
        aliases_path=args.aliases.expanduser().resolve(),
        report_root=args.report_root.expanduser().resolve(strict=False),
        output_dir=args.output_dir,
        title=args.title,
        require_history=bool(args.require_history),
        allow_missing=bool(args.allow_missing),
    )
    print("=" * 88, flush=True)
    print(f"PD-REW-V2 REPORT GENERATED ({res['param_family'].upper()})", flush=True)
    print("ROWS        :", res["row_count"], flush=True)
    print("PENDING ROWS:", res["missing_row_ids"] or "NONE (100% READY)", flush=True)
    print("REPORT HTML :", res["report_html"], flush=True)
    print("INDEX HTML  :", res["index_html"], flush=True)
    print("DOWNLOAD ZIP:", res["zip_path"], flush=True)
    print("=" * 88, flush=True)

    if args.serve_daemon:
        srv = start_daemon_http_server(
            Path(res["output_dir"]),
            res["html_filename"],
            port=int(args.port),
        )
        print(f"🌐 LAN Server Active (PID={srv['pid']}, Port={srv['port']})", flush=True)
        print(f"👉 Interactive Report : {srv['report_url']}", flush=True)
        print(f"👉 Download Center    : {srv['index_url']}", flush=True)
    elif args.serve:
        stop_existing_server_on_port(int(args.port))
        import generate_report
        generate_report.serve_dir(Path(res["output_dir"]), Path(res["report_html"]), int(args.port))


if __name__ == "__main__":
    main()
