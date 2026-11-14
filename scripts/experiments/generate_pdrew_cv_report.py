#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the redesigned OGD-LBS 4-Fold Cross-Validation & Scheduler Ablation
(ssdr12 / ssdr14 / ssdr14v) Interactive Report with Multi-Level Paired t-Tests.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import time
from typing import Any
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in os.sys.path:
    os.sys.path.insert(0, str(HERE))

import report_core  # noqa: E402
report_core.ensure_pdrew_python_env(__file__)
import aggregate_metrics  # noqa: E402
from generate_pdrew_gamma_report import (  # noqa: E402
    DEFAULT_ALIASES,
    DEFAULT_BINDING,
    DEFAULT_PD_REW_ROOT,
    DEFAULT_POOL,
    DEFAULT_REPORT_ROOT,
    DEFAULT_TABLE,
    materialize_pool_file,
    require,
    safe_pool_data_dir,
    sha256_file,
    start_daemon_http_server,
    stop_existing_server_on_port,
    strict_json_write,
)

ORG_TYPE_TO_METHOD = {
    0: "ssdr12",
    1: "ssdr14",
    3: "ssdr14v",
}
METHODS_ORDER = ["ssdr12", "ssdr14", "ssdr14v"]

MODEL_SCALES = {
    "Griffin_A2": 0.3,
    "Winter_Samurai": 10.0,
    "Vroid_18": 1.0,
    "Vroid_19": 1.0,
    "Vroid_20": 1.0,
}

SCALED_METRICS = {
    "04_L2_Err",
    "04_Hausdorff_Mean",
    "04_Hausdorff_Max",
}

METRICS_META = [
    {"key": "04_L2_Err", "short": "VD (L2 Err)", "label": "VD / L2 Error ↓", "npz": "l2_err", "scaled": True},
    {"key": "04_Org_Loss", "short": "SO (Org Loss)", "label": "SO / Bone Org Loss ↓", "npz": "org_loss", "scaled": False},
    {"key": "04_Hausdorff_Mean", "short": "HD Mean", "label": "Hausdorff Mean ↓", "npz": "hausdorff", "scaled": True},
    {"key": "04_Hausdorff_Max", "short": "HD Max", "label": "Hausdorff Max ↓", "npz": None, "scaled": True},
    {"key": "04_ARAP_Ratio", "short": "ARAP Ratio", "label": "Mesh ARAP Ratio ↓", "npz": "arap_ratio", "scaled": False},
    {"key": "04_Max_Roll", "short": "Max Roll (°)", "label": "Bone Max Roll (°) ↓", "npz": "bone_roll_max", "scaled": False},
    {"key": "04_Max_Swing", "short": "Max Swing (°)", "label": "Bone Max Swing (°) ↓", "npz": "bone_swing_max", "scaled": False},
]


def _sig_stars(p_val: float | None) -> str:
    if p_val is None or not math.isfinite(p_val):
        return ""
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        seen: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _load_npz_series(npz_path: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {"frame_indices": [], "series": {}, "stats": {}}
    if npz_path is None or not npz_path.is_file():
        return out
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            if "frame_indices" in npz.files:
                f_arr = np.asarray(npz["frame_indices"], dtype=np.int64).reshape(-1)
                out["frame_indices"] = [int(x) for x in f_arr.tolist()]
            for m in METRICS_META:
                nk = m["npz"]
                mk = m["key"]
                if not nk or nk not in npz.files:
                    continue
                arr = np.asarray(npz[nk], dtype=np.float64)
                vec = arr[-1, :] if (arr.ndim == 2 and arr.shape[0] > 0) else arr.reshape(-1)
                if vec.size == 0:
                    continue
                out["series"][mk] = [float(x) for x in vec.tolist()]
                out["stats"][mk] = {
                    "mean": float(np.mean(vec)),
                    "std": float(np.std(vec, ddof=0)),
                    "min": float(np.min(vec)),
                    "max": float(np.max(vec)),
                }
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _paired_test_on_vectors(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """
    a = baseline (ssdr14v), b = candidate (ssdr12 or ssdr14).
    Lower metric values are better, so diff = b - a > 0 means ssdr14v is better.
    """
    st = aggregate_metrics.compute_paired_stats(a, b)
    n = int(min(a.size, b.size))
    if n <= 0:
        return {
            "n_pairs": 0,
            "baseline_mean": None,
            "candidate_mean": None,
            "mean_diff": None,
            "rel_diff_pct": None,
            "ci95_low": None,
            "ci95_high": None,
            "cohens_dz": None,
            "win_rate_ssdr14v": None,
            "t_stat": None,
            "p_value": None,
        }
    base_mean = float(np.mean(a))
    cand_mean = float(np.mean(b))
    diff = b - a
    mean_diff = float(np.mean(diff))
    rel_pct = float((mean_diff / base_mean) * 100.0) if abs(base_mean) > 1e-12 else None
    wins = float(np.mean(a < b) * 100.0)
    return {
        "n_pairs": n,
        "baseline_mean": base_mean,
        "candidate_mean": cand_mean,
        "mean_diff": mean_diff,
        "rel_diff_pct": rel_pct,
        "ci95_low": st["ci95_low"] if math.isfinite(st["ci95_low"]) else None,
        "ci95_high": st["ci95_high"] if math.isfinite(st["ci95_high"]) else None,
        "cohens_dz": st["cohens_dz"] if math.isfinite(st["cohens_dz"]) else None,
        "win_rate_ssdr14v": wins,
        "t_stat": st["t_stat"] if math.isfinite(st["t_stat"]) else None,
        "p_value": st["p_value"] if math.isfinite(st["p_value"]) else None,
    }


def _apply_holm(rows: list[dict[str, Any]], group_keys: list[str]) -> None:
    buckets: dict[tuple[Any, ...], list[int]] = {}
    for idx, r in enumerate(rows):
        p = r.get("p_value")
        if p is None or not isinstance(p, (int, float)) or not math.isfinite(float(p)):
            r["p_value_holm"] = None
            r["sig"] = "PENDING" if r.get("n_pairs", 0) == 0 else ""
            continue
        key = tuple(r.get(k, "") for k in group_keys)
        buckets.setdefault(key, []).append(idx)

    for _key, idxs in buckets.items():
        m = len(idxs)
        idxs_sorted = sorted(idxs, key=lambda i: float(rows[i]["p_value"]))
        running_max = 0.0
        for rank, i in enumerate(idxs_sorted):
            adj = min(1.0, float(rows[i]["p_value"]) * float(m - rank))
            running_max = max(running_max, adj)
            rows[i]["p_value_holm"] = running_max
            rows[i]["sig"] = _sig_stars(running_max)


def build_latex_paper_table(
    summary_records: list[dict[str, Any]],
    pooled_ttests: list[dict[str, Any]],
    *,
    scale_mode: str = "scaled",
) -> str:
    sig_lookup: dict[tuple[str, str, str], str] = {}
    for t in pooled_ttests:
        if t.get("scale_mode") == scale_mode and t.get("split") == "test":
            sig_lookup[(str(t["model"]), str(t["candidate"]), str(t["metric"]))] = str(t.get("sig") or "")

    metrics_cols = [
        ("04_L2_Err", "VD ($L_2$) $\\downarrow$"),
        ("04_Org_Loss", "SO (Org) $\\downarrow$"),
        ("04_Hausdorff_Mean", "HD Mean $\\downarrow$"),
        ("04_Hausdorff_Max", "HD Max $\\downarrow$"),
        ("04_ARAP_Ratio", "ARAP $\\downarrow$"),
        ("04_Max_Roll", "Roll ($^\\circ$) $\\downarrow$"),
        ("04_Max_Swing", "Swing ($^\\circ$) $\\downarrow$"),
    ]

    lines = [
        "% Auto-generated 4-Fold Cross-Validation Summary Table (Held-out Test Split)",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{4-fold cross-validation results on held-out unseen animation frames (mean $\\pm$ SD across 4 folds). "
        "Paired $t$-test significance against \\textbf{ssdr14v (Ours)}: $^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}",
        "\\label{tab:cv4_ablation}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * len(metrics_cols) + "}",
        "\\hline",
        "Model & Method & " + " & ".join(title for _, title in metrics_cols) + " \\\\",
        "\\hline",
    ]

    models = sorted({str(r["model"]) for r in summary_records if r["model"] != "ALL_5_MODELS"})
    if any(r["model"] == "ALL_5_MODELS" for r in summary_records):
        models.append("ALL_5_MODELS")

    by_key = {
        (str(r["model"]), str(r["method"]), str(r["split"]), str(r["scale_mode"])): r
        for r in summary_records
    }

    for model in models:
        m_label = model.replace("_", "\\_")
        for m_idx, method in enumerate(METHODS_ORDER):
            rec = by_key.get((model, method, "test", scale_mode))
            row_prefix = m_label if m_idx == 0 else ""
            method_tex = "\\textbf{ssdr14v (Ours)}" if method == "ssdr14v" else method
            cells = [row_prefix, method_tex]
            for m_key, _ in metrics_cols:
                if not rec or rec["metrics"].get(m_key, {}).get("mean") is None:
                    cells.append("Pending")
                    continue
                m_info = rec["metrics"][m_key]
                mean_v = float(m_info["mean"])
                std_v = float(m_info["std"] or 0.0)
                txt = f"{mean_v:.4g} $\\pm$ {std_v:.2g}"
                if method != "ssdr14v":
                    sig = sig_lookup.get((model, method, m_key), "")
                    if sig in {"*", "**", "***"}:
                        txt += f"$^{{{sig}}}$"
                if m_info.get("is_best"):
                    txt = f"\\textbf{{{txt}}}"
                cells.append(txt)
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\hline")

    lines.extend([
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def generate_cv_dashboard(
    *,
    pd_rew_root: Path = DEFAULT_PD_REW_ROOT,
    pool: Path = DEFAULT_POOL,
    binding_path: Path = DEFAULT_BINDING,
    table_path: Path = DEFAULT_TABLE,
    report_root: Path = DEFAULT_REPORT_ROOT,
    output_dir: Path | None = None,
    title: str = "OGD-LBS 4-Fold Cross-Validation & Scheduler Significance (t-Test) Report",
) -> dict[str, Any]:
    src = pd_rew_root / "src"
    if str(src) not in os.sys.path:
        os.sys.path.insert(0, str(src))

    import datalad.api as dl
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

    cv_exp_rows = [
        r for r in expanded["rows"]
        if str((r.get("sweep_choices") or {}).get("weights") or "").startswith("cv-")
    ]
    require(bool(cv_exp_rows), "No cv-* rows found in experiment table")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        output_dir = report_root / f"pdrew_cv4_report_{timestamp}"
    else:
        output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_dir = output_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    fold_cases: list[dict[str, Any]] = []
    missing_row_ids: list[str] = []

    for row in cv_exp_rows:
        row_id = row["row_id"]
        flat = row.get("flat_parameters") or {}
        model = str(flat.get("data_prep_handoff.model_name") or flat.get("ssdr_handoff.model_name"))
        org_type = int(flat.get("ssdr_stage04.org_type", 3))
        method = ORG_TYPE_TO_METHOD.get(org_type, f"org{org_type}")
        kfold = int(flat.get("ssdr_stage04.kfold", 4))
        fold_index = int(flat.get("ssdr_stage04.fold_index", 0))
        core_frac = float(flat.get("ssdr_stage04.special_train_fraction", 0.0))
        model_scale = float(MODEL_SCALES.get(model, 1.0))

        st_info = cache_rows[row_id]["stage_statuses"][final_stage]
        if st_info["status"] != "HIT":
            missing_row_ids.append(row_id)
            for split_name in ("test", "train"):
                fold_cases.append({
                    "row_id": row_id,
                    "status": "PENDING",
                    "model": model,
                    "model_scale": model_scale,
                    "fold_index": fold_index,
                    "kfold": kfold,
                    "core_frac": core_frac,
                    "org_type": org_type,
                    "method": method,
                    "split": split_name,
                    "data_dir": None,
                    "frame_count": None,
                    "metrics_raw": {},
                    "metrics_scaled": {},
                    "frame_stats_raw": {},
                    "frame_stats_scaled": {},
                    "frame_indices": [],
                    "frame_series_raw": {},
                    "frame_series_scaled": {},
                })
            print(f"[cv-pending] {row_id} model={model} fold={fold_index} method={method} -> PENDING", flush=True)
            continue

        data_dir = st_info["data_dir"]
        data4_dir = safe_pool_data_dir(pool, data_dir) / "data4"

        train_sum_src = data4_dir / "step04_optimize_summary.json"
        test_sum_src = data4_dir / "step04_test_summary.json"
        train_npz_src = data4_dir / "opt_out/metrics_history_per_frame.npz"
        test_npz_src = data4_dir / "opt_out_test/metrics_history_per_frame.npz"

        materialize_pool_file(datalad_api=dl, pool=pool, path=train_sum_src, required=True)
        materialize_pool_file(datalad_api=dl, pool=pool, path=test_sum_src, required=True)
        materialize_pool_file(datalad_api=dl, pool=pool, path=train_npz_src, required=True)
        materialize_pool_file(datalad_api=dl, pool=pool, path=test_npz_src, required=True)

        train_sum = json.loads(train_sum_src.read_text(encoding="utf-8"))
        test_sum = json.loads(test_sum_src.read_text(encoding="utf-8"))
        train_npz = _load_npz_series(train_npz_src)
        test_npz = _load_npz_series(test_npz_src)

        for split_name, sum_obj, npz_obj in [
            ("test", test_sum, test_npz),
            ("train", train_sum, train_npz),
        ]:
            fm = sum_obj.get("final_metrics") or {}
            counts = sum_obj.get("counts") or {}
            metrics_raw: dict[str, float | None] = {}
            metrics_scaled: dict[str, float | None] = {}
            fstats_raw: dict[str, Any] = {}
            fstats_scaled: dict[str, Any] = {}
            fseries_raw: dict[str, list[float]] = {}
            fseries_scaled: dict[str, list[float]] = {}

            for m in METRICS_META:
                mk = m["key"]
                is_sc = bool(m["scaled"])
                factor = model_scale if is_sc else 1.0
                rv = aggregate_metrics.maybe_float(fm.get(mk))
                metrics_raw[mk] = rv
                metrics_scaled[mk] = (rv * factor) if rv is not None else None

                if mk in npz_obj["stats"]:
                    st_r = npz_obj["stats"][mk]
                    fstats_raw[mk] = st_r
                    fstats_scaled[mk] = {k: float(v) * factor for k, v in st_r.items()}
                if mk in npz_obj["series"]:
                    ser_r = npz_obj["series"][mk]
                    fseries_raw[mk] = ser_r
                    fseries_scaled[mk] = [float(x) * factor for x in ser_r]

            fold_cases.append({
                "row_id": row_id,
                "status": "READY",
                "model": model,
                "model_scale": model_scale,
                "fold_index": fold_index,
                "kfold": kfold,
                "core_frac": core_frac,
                "org_type": org_type,
                "method": method,
                "split": split_name,
                "data_dir": data_dir,
                "frame_count": counts.get("frames"),
                "metrics_raw": metrics_raw,
                "metrics_scaled": metrics_scaled,
                "frame_stats_raw": fstats_raw,
                "frame_stats_scaled": fstats_scaled,
                "frame_indices": npz_obj.get("frame_indices") or [],
                "frame_series_raw": fseries_raw,
                "frame_series_scaled": fseries_scaled,
            })
        print(f"[cv-ready] {row_id} model={model} fold={fold_index} method={method} -> {data_dir}", flush=True)

    models = sorted({c["model"] for c in fold_cases})
    folds = sorted({int(c["fold_index"]) for c in fold_cases})
    case_map = {
        (c["split"], c["model"], int(c["fold_index"]), c["method"]): c
        for c in fold_cases
    }

    # === Compute 3-Way Head-to-Head Frame Win Rates (%) per (split, model, fold_index, method) ===
    for split_name in ("test", "train"):
        for model in models:
            for fold_idx in folds:
                c12 = case_map.get((split_name, model, fold_idx, "ssdr12"))
                c14 = case_map.get((split_name, model, fold_idx, "ssdr14"))
                c14v = case_map.get((split_name, model, fold_idx, "ssdr14v"))
                group_cases = {"ssdr12": c12, "ssdr14": c14, "ssdr14v": c14v}
                for m_name, c_obj in group_cases.items():
                    if c_obj is not None:
                        c_obj["win_rates_h2h"] = {}
                        c_obj["win_rates_vs_14v"] = {}
                if not c12 or not c14 or not c14v:
                    continue
                if c12["status"] != "READY" or c14["status"] != "READY" or c14v["status"] != "READY":
                    continue
                for m in METRICS_META:
                    mk = m["key"]
                    if not m["npz"]:
                        continue
                    s12 = c12["frame_series_raw"].get(mk)
                    s14 = c14["frame_series_raw"].get(mk)
                    s14v = c14v["frame_series_raw"].get(mk)
                    if not s12 or not s14 or not s14v:
                        continue
                    v12 = np.asarray(s12, dtype=np.float64)
                    v14 = np.asarray(s14, dtype=np.float64)
                    v14v = np.asarray(s14v, dtype=np.float64)
                    if not (v12.shape == v14.shape == v14v.shape and v12.size > 0):
                        continue
                    c12["win_rates_h2h"][mk] = float(0.5 * (np.mean(v12 < v14) + np.mean(v12 < v14v)) * 100.0)
                    c14["win_rates_h2h"][mk] = float(0.5 * (np.mean(v14 < v12) + np.mean(v14 < v14v)) * 100.0)
                    c14v["win_rates_h2h"][mk] = float(0.5 * (np.mean(v14v < v12) + np.mean(v14v < v14)) * 100.0)

                    c12["win_rates_vs_14v"][mk] = float(np.mean(v12 < v14v) * 100.0)
                    c14["win_rates_vs_14v"][mk] = float(np.mean(v14 < v14v) * 100.0)
                    c14v["win_rates_vs_14v"][mk] = 50.0

    # === Build Module 1: Model x Method 4-Fold Summary Records ===
    summary_records: list[dict[str, Any]] = []
    for scale_mode in ("scaled", "raw"):
        m_field = "metrics_scaled" if scale_mode == "scaled" else "metrics_raw"
        for split_name in ("test", "train"):
            for model in [*models, "ALL_5_MODELS"]:
                model_rows_temp: list[dict[str, Any]] = []
                for method in METHODS_ORDER:
                    if model == "ALL_5_MODELS":
                        cand_cases = [
                            case_map[(split_name, m_name, f_idx, method)]
                            for m_name in models
                            for f_idx in folds
                            if (split_name, m_name, f_idx, method) in case_map
                        ]
                    else:
                        cand_cases = [
                            case_map[(split_name, model, f_idx, method)]
                            for f_idx in folds
                            if (split_name, model, f_idx, method) in case_map
                        ]
                    ready_cases = [c for c in cand_cases if c["status"] == "READY"]
                    pending_cnt = len(cand_cases) - len(ready_cases)

                    m_summary: dict[str, Any] = {}
                    for m in METRICS_META:
                        mk = m["key"]
                        vals = [
                            float(c[m_field][mk])
                            for c in ready_cases
                            if c[m_field].get(mk) is not None
                        ]
                        # Also compute train-test gap if split_name == 'test'
                        gap_vals: list[float] = []
                        if split_name == "test":
                            for c in ready_cases:
                                tr_c = case_map.get(("train", c["model"], int(c["fold_index"]), method))
                                if tr_c and tr_c["status"] == "READY":
                                    v_te = c[m_field].get(mk)
                                    v_tr = tr_c[m_field].get(mk)
                                    if v_te is not None and v_tr is not None:
                                        gap_vals.append(float(v_te) - float(v_tr))

                        if vals:
                            arr = np.asarray(vals, dtype=np.float64)
                            m_summary[mk] = {
                                "n": int(arr.size),
                                "mean": float(np.mean(arr)),
                                "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0,
                                "min": float(np.min(arr)),
                                "max": float(np.max(arr)),
                                "gen_gap_mean": float(np.mean(gap_vals)) if gap_vals else None,
                                "is_best": False,
                            }
                        else:
                            m_summary[mk] = {
                                "n": 0,
                                "mean": None,
                                "std": None,
                                "min": None,
                                "max": None,
                                "gen_gap_mean": None,
                                "is_best": False,
                            }

                    rec = {
                        "scale_mode": scale_mode,
                        "split": split_name,
                        "model": model,
                        "method": method,
                        "n_ready_folds": len(ready_cases),
                        "n_pending_folds": pending_cnt,
                        "metrics": m_summary,
                    }
                    model_rows_temp.append(rec)
                    summary_records.append(rec)

                # Mark best method per (model, split, scale_mode, metric)
                for m in METRICS_META:
                    mk = m["key"]
                    valid_means = [
                        r["metrics"][mk]["mean"]
                        for r in model_rows_temp
                        if r["metrics"][mk]["mean"] is not None
                    ]
                    if valid_means:
                        best_v = min(valid_means)
                        for r in model_rows_temp:
                            mv = r["metrics"][mk]["mean"]
                            if mv is not None and abs(mv - best_v) <= 1e-12:
                                r["metrics"][mk]["is_best"] = True
                        base_v = next(
                            (r["metrics"][mk]["mean"] for r in model_rows_temp if r["method"] == "ssdr14v"),
                            None,
                        )
                        for r in model_rows_temp:
                            mv = r["metrics"][mk]["mean"]
                            if mv is not None and base_v is not None and abs(base_v) > 1e-12:
                                r["metrics"][mk]["rel_vs_ssdr14v_pct"] = float(((mv - base_v) / base_v) * 100.0)
                            else:
                                r["metrics"][mk]["rel_vs_ssdr14v_pct"] = None

    # === Build Module 2: Three-Level Paired t-Tests ===
    pooled_frame_ttests: list[dict[str, Any]] = []
    per_fold_frame_ttests: list[dict[str, Any]] = []
    fold_model_summary_ttests: list[dict[str, Any]] = []

    for scale_mode in ("scaled", "raw"):
        ser_field = "frame_series_scaled" if scale_mode == "scaled" else "frame_series_raw"
        met_field = "metrics_scaled" if scale_mode == "scaled" else "metrics_raw"

        for split_name in ("test", "train"):
            # Level 1 & Level 2: Frame-aligned paired t-tests
            for m in METRICS_META:
                mk = m["key"]
                if not m["npz"]:
                    continue

                for cand_method in ("ssdr12", "ssdr14"):
                    all_a: list[np.ndarray] = []
                    all_b: list[np.ndarray] = []
                    all_pending = 0

                    for model in models:
                        mod_a: list[np.ndarray] = []
                        mod_b: list[np.ndarray] = []
                        mod_pending = 0

                        for fold_idx in folds:
                            c_base = case_map.get((split_name, model, fold_idx, "ssdr14v"))
                            c_cand = case_map.get((split_name, model, fold_idx, cand_method))
                            if (
                                not c_base
                                or not c_cand
                                or c_base["status"] != "READY"
                                or c_cand["status"] != "READY"
                                or mk not in c_base[ser_field]
                                or mk not in c_cand[ser_field]
                            ):
                                mod_pending += 1
                                all_pending += 1
                                if split_name in ("test", "train"):
                                    per_fold_frame_ttests.append({
                                        "scale_mode": scale_mode,
                                        "split": split_name,
                                        "model": model,
                                        "fold_index": fold_idx,
                                        "candidate": cand_method,
                                        "baseline": "ssdr14v",
                                        "metric": mk,
                                        "metric_short": m["short"],
                                        "pending_folds": 1,
                                        **_paired_test_on_vectors(np.zeros((0,)), np.zeros((0,))),
                                    })
                                continue

                            va = np.asarray(c_base[ser_field][mk], dtype=np.float64)
                            vb = np.asarray(c_cand[ser_field][mk], dtype=np.float64)
                            mod_a.append(va)
                            mod_b.append(vb)
                            all_a.append(va)
                            all_b.append(vb)

                            if split_name in ("test", "train"):
                                per_fold_frame_ttests.append({
                                    "scale_mode": scale_mode,
                                    "split": split_name,
                                    "model": model,
                                    "fold_index": fold_idx,
                                    "candidate": cand_method,
                                    "baseline": "ssdr14v",
                                    "metric": mk,
                                    "metric_short": m["short"],
                                    "pending_folds": 0,
                                    **_paired_test_on_vectors(va, vb),
                                })

                        vec_ma = np.concatenate(mod_a) if mod_a else np.zeros((0,))
                        vec_mb = np.concatenate(mod_b) if mod_b else np.zeros((0,))
                        pooled_frame_ttests.append({
                            "scale_mode": scale_mode,
                            "split": split_name,
                            "model": model,
                            "candidate": cand_method,
                            "baseline": "ssdr14v",
                            "metric": mk,
                            "metric_short": m["short"],
                            "pending_folds": mod_pending,
                            **_paired_test_on_vectors(vec_ma, vec_mb),
                        })

                    vec_aa = np.concatenate(all_a) if all_a else np.zeros((0,))
                    vec_ab = np.concatenate(all_b) if all_b else np.zeros((0,))
                    pooled_frame_ttests.append({
                        "scale_mode": scale_mode,
                        "split": split_name,
                        "model": "ALL_5_MODELS",
                        "candidate": cand_method,
                        "baseline": "ssdr14v",
                        "metric": mk,
                        "metric_short": m["short"],
                        "pending_folds": all_pending,
                        **_paired_test_on_vectors(vec_aa, vec_ab),
                    })

            # Level 3: Fold-level (n<=20) and Model-level (n=5) summary paired t-tests
            for m in METRICS_META:
                mk = m["key"]
                for cand_method in ("ssdr12", "ssdr14"):
                    fold_a: list[float] = []
                    fold_b: list[float] = []
                    mod_mean_a: list[float] = []
                    mod_mean_b: list[float] = []
                    pending_pairs = 0

                    for model in models:
                        ma: list[float] = []
                        mb: list[float] = []
                        for fold_idx in folds:
                            c_base = case_map.get((split_name, model, fold_idx, "ssdr14v"))
                            c_cand = case_map.get((split_name, model, fold_idx, cand_method))
                            if (
                                not c_base
                                or not c_cand
                                or c_base["status"] != "READY"
                                or c_cand["status"] != "READY"
                            ):
                                pending_pairs += 1
                                continue
                            va = c_base[met_field].get(mk)
                            vb = c_cand[met_field].get(mk)
                            if va is None or vb is None:
                                pending_pairs += 1
                                continue
                            fold_a.append(float(va))
                            fold_b.append(float(vb))
                            ma.append(float(va))
                            mb.append(float(vb))
                        if ma and mb:
                            mod_mean_a.append(float(np.mean(ma)))
                            mod_mean_b.append(float(np.mean(mb)))

                    fold_model_summary_ttests.append({
                        "scale_mode": scale_mode,
                        "unit": "fold_pairs (5 models × 4 folds, n≤20)",
                        "split": split_name,
                        "candidate": cand_method,
                        "baseline": "ssdr14v",
                        "metric": mk,
                        "metric_short": m["short"],
                        "pending_folds": pending_pairs,
                        **_paired_test_on_vectors(np.asarray(fold_a), np.asarray(fold_b)),
                    })
                    fold_model_summary_ttests.append({
                        "scale_mode": scale_mode,
                        "unit": "model_4fold_means (5 models, n=5)",
                        "split": split_name,
                        "candidate": cand_method,
                        "baseline": "ssdr14v",
                        "metric": mk,
                        "metric_short": m["short"],
                        "pending_folds": pending_pairs,
                        **_paired_test_on_vectors(np.asarray(mod_mean_a), np.asarray(mod_mean_b)),
                    })

    gen_gap_ttests: list[dict[str, Any]] = []
    for scale_mode in ("scaled", "raw"):
        met_field = "metrics_scaled" if scale_mode == "scaled" else "metrics_raw"
        for m in METRICS_META:
            mk = m["key"]
            for model in [*models, "ALL_5_MODELS"]:
                for method in METHODS_ORDER:
                    tr_vals: list[float] = []
                    te_vals: list[float] = []
                    pend = 0
                    target_models = models if model == "ALL_5_MODELS" else [model]
                    for tm in target_models:
                        for f_idx in folds:
                            c_tr = case_map.get(("train", tm, f_idx, method))
                            c_te = case_map.get(("test", tm, f_idx, method))
                            if (
                                not c_tr or not c_te
                                or c_tr["status"] != "READY"
                                or c_te["status"] != "READY"
                            ):
                                pend += 1
                                continue
                            v_tr = c_tr[met_field].get(mk)
                            v_te = c_te[met_field].get(mk)
                            if v_tr is None or v_te is None:
                                pend += 1
                                continue
                            tr_vals.append(float(v_tr))
                            te_vals.append(float(v_te))
                    gen_gap_ttests.append({
                        "scale_mode": scale_mode,
                        "model": model,
                        "method": method,
                        "metric": mk,
                        "metric_short": m["short"],
                        "pending_folds": pend,
                        **_paired_test_on_vectors(np.asarray(tr_vals), np.asarray(te_vals)),
                    })

    _apply_holm(pooled_frame_ttests, ["scale_mode", "split", "model", "metric"])
    _apply_holm(per_fold_frame_ttests, ["scale_mode", "split", "model", "fold_index", "metric"])
    _apply_holm(fold_model_summary_ttests, ["scale_mode", "unit", "split", "metric"])

    # Attach pooled test significance stars into summary_records for instant frontend rendering
    pooled_sig_map = {
        (t["scale_mode"], t["split"], t["model"], t["candidate"], t["metric"]): t.get("sig", "")
        for t in pooled_frame_ttests
    }
    for rec in summary_records:
        if rec["method"] == "ssdr14v":
            continue
        for m in METRICS_META:
            mk = m["key"]
            rec["metrics"][mk]["sig"] = pooled_sig_map.get(
                (rec["scale_mode"], rec["split"], rec["model"], rec["method"], mk),
                "",
            )

    # === Build Module 3.3: Out-of-Sample Stitched Frame Trajectories per Model ===
    trajectories: dict[str, Any] = {}
    for model in models:
        trajectories[model] = {}
        for scale_mode in ("scaled", "raw"):
            ser_field = "frame_series_scaled" if scale_mode == "scaled" else "frame_series_raw"
            model_traj: dict[str, Any] = {"folds": []}
            for fold_idx in folds:
                fold_entry: dict[str, Any] = {"fold_index": fold_idx, "methods": {}}
                for method in METHODS_ORDER:
                    c = case_map.get(("test", model, fold_idx, method))
                    if c and c["status"] == "READY":
                        fold_entry["methods"][method] = {
                            "status": "READY",
                            "frame_indices": c["frame_indices"],
                            "series": c[ser_field],
                        }
                    else:
                        fold_entry["methods"][method] = {
                            "status": "PENDING",
                            "frame_indices": [],
                            "series": {},
                        }
                model_traj["folds"].append(fold_entry)
            trajectories[model][scale_mode] = model_traj

    # Write CSV / JSON / LaTeX artifacts
    _write_csv(agg_dir / "ttest_pooled_frames.csv", pooled_frame_ttests)
    _write_csv(agg_dir / "ttest_per_fold_frames.csv", per_fold_frame_ttests)
    _write_csv(agg_dir / "ttest_fold_and_model_summary.csv", fold_model_summary_ttests)

    flat_summary_csv: list[dict[str, Any]] = []
    for r in summary_records:
        row_out = {
            "scale_mode": r["scale_mode"],
            "split": r["split"],
            "model": r["model"],
            "method": r["method"],
            "n_ready_folds": r["n_ready_folds"],
            "n_pending_folds": r["n_pending_folds"],
        }
        for m in METRICS_META:
            mk = m["key"]
            mi = r["metrics"][mk]
            row_out[f"{mk}_mean"] = mi["mean"] if mi["mean"] is not None else ""
            row_out[f"{mk}_std"] = mi["std"] if mi["std"] is not None else ""
            row_out[f"{mk}_rel_vs_ssdr14v_pct"] = (
                mi.get("rel_vs_ssdr14v_pct") if mi.get("rel_vs_ssdr14v_pct") is not None else ""
            )
            row_out[f"{mk}_sig"] = mi.get("sig", "")
        flat_summary_csv.append(row_out)
    _write_csv(agg_dir / "model_method_4fold_summary.csv", flat_summary_csv)

    latex_scaled = build_latex_paper_table(summary_records, pooled_frame_ttests, scale_mode="scaled")
    latex_raw = build_latex_paper_table(summary_records, pooled_frame_ttests, scale_mode="raw")
    (agg_dir / "paper_table_4fold_cv_scaled.tex").write_text(latex_scaled, encoding="utf-8")
    (agg_dir / "paper_table_4fold_cv_raw.tex").write_text(latex_raw, encoding="utf-8")

    # Strip heavy per-frame arrays from fold_cases_table to keep table payload light
    fold_cases_table = [
        {k: v for k, v in c.items() if k not in {"frame_series_raw", "frame_series_scaled"}}
        for c in fold_cases
    ]

    total_cv_cases = len(cv_exp_rows)
    ready_cv_cases = total_cv_cases - len(missing_row_ids)

    dashboard_payload = {
        "meta": {
            "title": title,
            "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pool": str(pool),
            "total_cv_cases": total_cv_cases,
            "ready_cv_cases": ready_cv_cases,
            "pending_cv_cases": len(missing_row_ids),
            "missing_row_ids": missing_row_ids,
            "models": models,
            "folds": folds,
            "methods": METHODS_ORDER,
            "model_scales": MODEL_SCALES,
            "metrics_meta": METRICS_META,
        },
        "summaryRecords": summary_records,
        "pooledFrameTtests": pooled_frame_ttests,
        "perFoldFrameTtests": per_fold_frame_ttests,
        "foldModelSummaryTtests": fold_model_summary_ttests,
        "genGapTtests": gen_gap_ttests,
        "foldCases": fold_cases_table,
        "trajectories": trajectories,
        "latex": {
            "scaled": latex_scaled,
            "raw": latex_raw,
        },
    }

    strict_json_write(output_dir / "cv_dashboard_data.json", dashboard_payload)

    html_text = render_cv_dashboard_html(title=title, payload=dashboard_payload, output_dir=output_dir)
    html_path = output_dir / "CV_Report.html"
    html_path.write_text(html_text, encoding="utf-8")

    # Also write index.html and ZIP package
    zip_name = "ogd_lbs_cv4_ttest_report_bundle.zip"
    zip_path = output_dir / "downloads" / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    index_html = render_cv_index_html(title=title, zip_name=zip_name, meta=dashboard_payload["meta"])
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    bundle_files = [
        "CV_Report.html",
        "index.html",
        "cv_dashboard_data.json",
        "aggregate/model_method_4fold_summary.csv",
        "aggregate/ttest_pooled_frames.csv",
        "aggregate/ttest_per_fold_frames.csv",
        "aggregate/ttest_fold_and_model_summary.csv",
        "aggregate/paper_table_4fold_cv_scaled.tex",
        "aggregate/paper_table_4fold_cv_raw.tex",
    ]
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in bundle_files:
            zf.write(output_dir / rel, arcname=rel)

    return {
        "output_dir": str(output_dir),
        "report_html": str(html_path),
        "index_html": str(output_dir / "index.html"),
        "zip_path": str(zip_path),
        "total_cv_cases": total_cv_cases,
        "ready_cv_cases": ready_cv_cases,
        "missing_row_ids": missing_row_ids,
        "pooled_frame_ttests": pooled_frame_ttests,
    }


def render_cv_index_html(*, title: str, zip_name: str, meta: dict[str, Any]) -> str:
    status_badge = (
        f"<span style='color:#0f766e;font-weight:700;'>✅ {meta['ready_cv_cases']}/{meta['total_cv_cases']} Cases Ready</span>"
        if meta["pending_cv_cases"] == 0
        else f"<span style='color:#d97706;font-weight:700;'>⏳ {meta['ready_cv_cases']}/{meta['total_cv_cases']} Cases Ready ({meta['pending_cv_cases']} Pending: {', '.join(meta['missing_row_ids'])})</span>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #f4f6f9; color: #1e293b; }}
main {{ background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.06); }}
h1 {{ margin-top: 0; color: #0f172a; }}
.btn {{ display: inline-block; padding: 11px 18px; border-radius: 7px; text-decoration: none; font-weight: 600; margin-right: 10px; }}
.btn-primary {{ background: #1C328A; color: #fff; }}
.btn-secondary {{ background: #0f766e; color: #fff; }}
li {{ margin: 10px 0; }}
a {{ color: #1C328A; font-weight: 600; }}
</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<p>Cluster Pool Status: {status_badge}</p>
<p style="margin: 22px 0;">
  <a class="btn btn-primary" href="CV_Report.html">📊 Open Interactive 4-Fold CV &amp; t-Test Dashboard</a>
  <a class="btn btn-secondary" href="downloads/{html.escape(zip_name)}" download>📦 Download Full Report ZIP</a>
</p>
<h2>Direct Data &amp; Paper Table Exports</h2>
<ul>
  <li><a href="aggregate/paper_table_4fold_cv_scaled.tex" download>📄 Paper LaTeX Table (Model-Scaled, paper_table_4fold_cv_scaled.tex)</a></li>
  <li><a href="aggregate/paper_table_4fold_cv_raw.tex" download>📄 Paper LaTeX Table (Raw Unscaled, paper_table_4fold_cv_raw.tex)</a></li>
  <li><a href="aggregate/model_method_4fold_summary.csv" download>📊 Model × Method 4-Fold Summary (CSV)</a></li>
  <li><a href="aggregate/ttest_pooled_frames.csv" download>🧪 Tab 1: Pooled Out-of-Sample Frame-Aligned Paired t-Test (CSV)</a></li>
  <li><a href="aggregate/ttest_per_fold_frames.csv" download>🧪 Tab 2: Per-Fold Frame-Aligned Paired t-Test (CSV)</a></li>
  <li><a href="aggregate/ttest_fold_and_model_summary.csv" download>🧪 Tab 3: Fold-Level &amp; Model-Level Summary Paired t-Test (CSV)</a></li>
  <li><a href="cv_dashboard_data.json" download>🗂️ Complete Dashboard Data Payload (JSON)</a></li>
</ul>
</main>
</body>
</html>
"""


def render_cv_dashboard_html(*, title: str, payload: dict[str, Any], output_dir: Path | None = None) -> str:
    def _clean(v: Any) -> Any:
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        if isinstance(v, dict):
            return {str(k): _clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_clean(x) for x in v]
        return v

    shared_assets = report_core.load_shared_frontend_assets(output_dir)
    suite_nav_html = report_core.build_suite_nav_html("CV_Report.html")
    data_json = json.dumps(_clean(payload), ensure_ascii=False, allow_nan=False)
    tpl_path = HERE / "report_templates" / "ogd_cv_dashboard_template.html"
    tpl = tpl_path.read_text(encoding="utf-8")
    return (
        tpl.replace("__TITLE_HTML__", html.escape(title))
        .replace("__SUITE_NAV_HTML__", suite_nav_html)
        .replace("__SHARED_CORE_CSS__", shared_assets["core_css"])
        .replace("__SHARED_CHART_ENGINE_JS__", shared_assets["chart_engine_js"])
        .replace("__SHARED_RENDER_DOCK_JS__", shared_assets["render_dock_js"])
        .replace("__DATA_JSON__", data_json)
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate redesigned OGD-LBS 4-Fold CV & t-test report.")
    parser.add_argument("--pd-rew-root", type=Path, default=DEFAULT_PD_REW_ROOT)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/ogd_reports/pdrew_cv4_report_latest"))
    parser.add_argument("--serve-daemon", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    res = generate_cv_dashboard(
        pd_rew_root=args.pd_rew_root.expanduser().resolve(),
        pool=args.pool.expanduser().resolve(),
        binding_path=args.binding.expanduser().resolve(),
        table_path=args.table.expanduser().resolve(),
        report_root=args.report_root.expanduser().resolve(strict=False),
        output_dir=args.output_dir,
    )
    print("=" * 92, flush=True)
    print("REDESIGNED 4-FOLD CV & PAIRED T-TEST DASHBOARD GENERATED", flush=True)
    print(f"READY / TOTAL : {res['ready_cv_cases']} / {res['total_cv_cases']}", flush=True)
    print(f"PENDING CASES : {res['missing_row_ids'] or 'NONE (100% COMPLETE)'}", flush=True)
    print(f"REPORT HTML   : {res['report_html']}", flush=True)
    print(f"DOWNLOAD ZIP  : {res['zip_path']}", flush=True)
    print("=" * 92, flush=True)

    if args.serve_daemon:
        srv = start_daemon_http_server(
            Path(res["output_dir"]),
            "CV_Report.html",
            port=int(args.port),
        )
        print(f"🌐 LAN Server Active (PID={srv['pid']}, Port={srv['port']})", flush=True)
        print(f"👉 Interactive 4-Fold CV Report : {srv['report_url']}", flush=True)
        print(f"👉 Export & Download Center     : {srv['index_url']}", flush=True)


if __name__ == "__main__":
    main()
