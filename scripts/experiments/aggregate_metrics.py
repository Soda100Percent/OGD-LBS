#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Aggregate OGD-LBS experiment metrics.

Supported layouts:

1. OFAT sweep output from scripts/experiments/run_param_sweep.py

    <input_root>/<MODEL>/sweep_<param>/<case_id>/
    ├── case_config.json
    ├── metrics_04.json
    └── data4/step04_optimize_summary.json

2. 4-fold CV output from scripts/experiments/run_4fold_cv.py

    <input_root>/<MODEL>/fold_00/
    ├── fold_plan.json
    ├── train/
    │   ├── metrics_04.json
    │   └── data4/step04_optimize_summary.json
    └── test/
        ├── metrics_04.json
        └── data4_test/step04_optimize_summary.json

3. Generic recursive metrics_04.json discovery.

Outputs:
    aggregated_metrics.csv/json
    summary_by_group.csv
    summary_table.md
    summary_table.tex
    optional ttest_results.csv/json/md
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from _env import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from ._env import REPO_ROOT  # noqa: F401

try:
    from experiment_io import dump_json, load_json, json_safe
except ModuleNotFoundError:  # pragma: no cover
    from .experiment_io import dump_json, load_json, json_safe


DEFAULT_METRICS = [
    "04_L2_Err",
    "04_Hausdorff_Mean",
    "04_Hausdorff_Max",
    "04_Org_Loss",
    "04_ARAP_Ratio",
    "04_Edge_Excess",
    "04_Normal_Flip",
    "04_Max_Roll",
    "04_Max_Swing",
    "04_Frame_Count",
    "04_Bone_Count",
    "04_Mesh_Count",
    "04_Vertex_Count_Total",
    "04_Iter",
    "04_S_Max",
    "04_Mask_NNZ_Mean",
    "04_Mask_NNZ_Max",
    "04_Final_Weight_NNZ_Mean",
    "04_Final_Weight_NNZ_Max",
    "03_total_actual_faces",
    "03_actual_to_input_face_ratio",
    "02_reduced_bone_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate OGD-LBS OFAT/CV/generic experiment metrics."
    )
    parser.add_argument("-i", "--input_root", type=str, required=True)
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Default: <input_root>/aggregated_reports",
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "cv", "generic"],
        default="generic",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric names to summarize. Default uses common OGD-LBS metrics found in rows.",
    )
    parser.add_argument(
        "--group_by",
        nargs="*",
        default=None,
        help="Grouping fields. Defaults: sweep=param value, cv=split, generic=<none>.",
    )
    parser.add_argument(
        "--ttest",
        choices=["none", "paired", "independent"],
        default="none",
    )
    parser.add_argument(
        "--baseline_query",
        type=str,
        default="",
        help='Comma-separated equality query, e.g. "param=s_max,value=4" or "split=train".',
    )
    parser.add_argument(
        "--pair_by",
        nargs="*",
        default=None,
        help="Pairing fields for paired t-test. Defaults: sweep=model, cv=model fold_index.",
    )
    parser.add_argument(
        "--ttest_group_by",
        nargs="*",
        default=None,
        help="Grouping fields for t-test candidates. Default: same as --group_by.",
    )
    return parser.parse_args()


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def maybe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def parse_query(text: str) -> dict[str, str]:
    text = str(text or "").strip()
    if not text:
        return {}

    out: dict[str, str] = {}
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid query item {item!r}; expected key=value.")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def row_matches_query(row: dict[str, Any], query: dict[str, str]) -> bool:
    for key, expected in query.items():
        if str(row.get(key, "")) != expected:
            return False
    return True


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    obj = load_json(path)
    if not isinstance(obj, dict):
        return {}
    return obj


def add_flat_scalars(
    row: dict[str, Any],
    obj: dict[str, Any],
    *,
    prefix: str = "",
    allow_existing_overwrite: bool = False,
) -> None:
    for key, value in obj.items():
        out_key = f"{prefix}{key}" if prefix else str(key)
        if not is_scalar(value):
            continue
        if out_key in row and not allow_existing_overwrite:
            continue
        row[out_key] = value


def add_history_npz_stats(row: dict[str, Any], npz_path: Path) -> None:
    if not npz_path.is_file():
        return

    try:
        with np.load(npz_path, allow_pickle=True) as npz:
            for key in npz.files:
                arr = np.asarray(npz[key])
                if not np.issubdtype(arr.dtype, np.number):
                    continue

                if arr.ndim == 2 and arr.shape[0] > 0:
                    final_vals = arr[-1, :]
                    row[f"history_{key}_final_mean"] = float(np.mean(final_vals))
                    row[f"history_{key}_final_std"] = float(np.std(final_vals))
                    row[f"history_{key}_final_min"] = float(np.min(final_vals))
                    row[f"history_{key}_final_max"] = float(np.max(final_vals))
                elif arr.ndim == 1 and arr.shape[0] > 0:
                    row[f"history_{key}_final"] = float(arr[-1])
    except Exception as exc:
        row["history_npz_error"] = str(exc)


def find_step04_summary_for_run(run_root: Path, *, split: str | None = None) -> Path | None:
    candidates = []

    if split == "test":
        candidates.extend([
            run_root / "data4_test" / "step04_optimize_summary.json",
            run_root / "data4" / "step04_optimize_summary.json",
        ])
    else:
        candidates.extend([
            run_root / "data4" / "step04_optimize_summary.json",
            run_root / "data4_test" / "step04_optimize_summary.json",
        ])

    candidates.extend(sorted(run_root.rglob("step04_optimize_summary.json")))

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path

    return None


def load_run_record(
    *,
    run_root: Path,
    metrics_path: Path,
    summary_path: Path | None,
    base: dict[str, Any],
) -> dict[str, Any]:
    row = dict(base)
    row["run_root"] = str(run_root)
    row["metrics_path"] = str(metrics_path)

    metrics_obj = load_json_if_exists(metrics_path)
    add_flat_scalars(row, metrics_obj)

    if summary_path is not None:
        row["step04_summary_path"] = str(summary_path)
        summary_obj = load_json_if_exists(summary_path)

        final_metrics = summary_obj.get("final_metrics", {})
        if isinstance(final_metrics, dict):
            add_flat_scalars(row, final_metrics)

        counts = summary_obj.get("counts", {})
        if isinstance(counts, dict):
            for k, v in counts.items():
                if is_scalar(v):
                    row[f"summary_count_{k}"] = v

        config = summary_obj.get("config", {})
        if isinstance(config, dict):
            for k, v in config.items():
                if is_scalar(v):
                    row[f"summary_config_{k}"] = v

    add_history_npz_stats(row, run_root / "opt_out" / "metrics_history_per_frame.npz")

    # Step04 normally writes history under output_tree/opt_out.
    if summary_path is not None:
        summary_parent = summary_path.parent
        add_history_npz_stats(row, summary_parent / "opt_out" / "metrics_history_per_frame.npz")

    return row


def discover_sweep_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for case_config_path in sorted(input_root.rglob("case_config.json")):
        case_root = case_config_path.parent
        metrics_path = case_root / "metrics_04.json"
        if not metrics_path.is_file():
            continue

        case_cfg = load_json_if_exists(case_config_path)

        base: dict[str, Any] = {
            "mode": "sweep",
            "case_config_path": str(case_config_path),
            "case_root": str(case_root),
            "model": case_cfg.get("model"),
            "param": case_cfg.get("param"),
            "value": case_cfg.get("value"),
            "case_id": case_cfg.get("case_id", case_root.name),
            "stage": case_cfg.get("stage"),
        }

        eff = case_cfg.get("effective_params", {})
        if isinstance(eff, dict):
            for k, v in eff.items():
                if is_scalar(v):
                    base[f"eff_{k}"] = v

        summary_path = find_step04_summary_for_run(case_root)
        rows.append(
            load_run_record(
                run_root=case_root,
                metrics_path=metrics_path,
                summary_path=summary_path,
                base=base,
            )
        )

    return rows


def discover_cv_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for fold_plan_path in sorted(input_root.rglob("fold_plan.json")):
        fold_root = fold_plan_path.parent
        fold_plan = load_json_if_exists(fold_plan_path)

        split_obj = fold_plan.get("split", {})
        split_stats = split_obj if isinstance(split_obj, dict) else {}

        base_common: dict[str, Any] = {
            "mode": "cv",
            "fold_plan_path": str(fold_plan_path),
            "fold_root": str(fold_root),
            "model": fold_plan.get("model", fold_root.parent.name),
            "fold_index": fold_plan.get("fold_index", fold_root.name.replace("fold_", "")),
            "split_train_count": split_stats.get("train_count"),
            "split_test_count": split_stats.get("test_count"),
            "split_special_count": split_stats.get("special_count"),
        }

        for split_name, output_dir_name in [("train", "data4"), ("test", "data4_test")]:
            run_root = fold_root / split_name
            metrics_path = run_root / "metrics_04.json"
            if not metrics_path.is_file():
                continue

            base = dict(base_common)
            base["split"] = split_name

            summary_path = find_step04_summary_for_run(run_root, split=split_name)

            rows.append(
                load_run_record(
                    run_root=run_root,
                    metrics_path=metrics_path,
                    summary_path=summary_path,
                    base=base,
                )
            )

    return rows


def discover_generic_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for metrics_path in sorted(input_root.rglob("metrics_04.json")):
        run_root = metrics_path.parent
        summary_path = find_step04_summary_for_run(run_root)
        base = {
            "mode": "generic",
            "run_name": run_root.name,
            "relative_run_root": str(run_root.relative_to(input_root)),
        }
        rows.append(
            load_run_record(
                run_root=run_root,
                metrics_path=metrics_path,
                summary_path=summary_path,
                base=base,
            )
        )

    return rows


def discover_rows(input_root: Path, mode: str) -> list[dict[str, Any]]:
    if mode == "sweep":
        return discover_sweep_rows(input_root)
    if mode == "cv":
        return discover_cv_rows(input_root)
    if mode == "generic":
        return discover_generic_rows(input_root)
    raise ValueError(f"Unsupported mode={mode!r}")


def union_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "mode",
        "model",
        "param",
        "value",
        "case_id",
        "stage",
        "fold_index",
        "split",
        "run_name",
        "relative_run_root",
        "run_root",
        "metrics_path",
        "step04_summary_path",
    ]

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    ordered = [k for k in preferred if k in all_keys]
    ordered.extend(sorted(k for k in all_keys if k not in set(ordered)))
    return ordered


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = union_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def default_group_by(mode: str) -> list[str]:
    if mode == "sweep":
        return ["param", "value"]
    if mode == "cv":
        return ["split"]
    return []


def default_pair_by(mode: str) -> list[str]:
    if mode == "sweep":
        return ["model"]
    if mode == "cv":
        return ["model", "fold_index"]
    return []


def row_group_key(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(f, "") for f in fields)


def choose_metric_names(rows: list[dict[str, Any]], requested: list[str] | None) -> list[str]:
    if requested:
        return [m for m in requested if any(m in row for row in rows)]

    out = [m for m in DEFAULT_METRICS if any(m in row for row in rows)]

    if out:
        return out

    numeric = []
    for key in union_fieldnames(rows):
        if any(maybe_float(row.get(key)) is not None for row in rows):
            numeric.append(key)
    return numeric


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    group_by: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    if group_by:
        for row in rows:
            groups.setdefault(row_group_key(row, group_by), []).append(row)
    else:
        groups[tuple()] = rows

    summary_rows = []

    for key, group_rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        out: dict[str, Any] = {}
        for field, value in zip(group_by, key):
            out[field] = value

        out["n_runs"] = int(len(group_rows))

        for metric in metrics:
            values = [
                x
                for x in (maybe_float(row.get(metric)) for row in group_rows)
                if x is not None
            ]
            out[f"{metric}_n"] = int(len(values))
            if values:
                arr = np.asarray(values, dtype=np.float64)
                out[f"{metric}_mean"] = float(np.mean(arr))
                out[f"{metric}_std"] = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
                out[f"{metric}_min"] = float(np.min(arr))
                out[f"{metric}_max"] = float(np.max(arr))
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
                out[f"{metric}_min"] = ""
                out[f"{metric}_max"] = ""

        summary_rows.append(out)

    return summary_rows


def format_mean_std(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean", "")
    std = row.get(f"{metric}_std", "")
    n = row.get(f"{metric}_n", 0)

    if mean == "" or mean is None:
        return ""

    try:
        mean_f = float(mean)
        std_f = float(std)
    except Exception:
        return ""

    return f"{mean_f:.6g} ± {std_f:.3g} (n={int(n)})"


def write_markdown_summary(
    path: Path,
    summary_rows: list[dict[str, Any]],
    *,
    group_by: list[str],
    metrics: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = list(group_by) + metrics
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")

    for row in summary_rows:
        cells = []
        for col in group_by:
            cells.append(str(row.get(col, "")))
        for metric in metrics:
            cells.append(format_mean_std(row, metric))
        lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(text: Any) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s


def write_latex_summary(
    path: Path,
    summary_rows: list[dict[str, Any]],
    *,
    group_by: list[str],
    metrics: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = list(group_by) + metrics
    col_spec = "l" * len(cols)

    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    lines.append(r"\hline")

    for row in summary_rows:
        cells = []
        for col in group_by:
            cells.append(latex_escape(row.get(col, "")))
        for metric in metrics:
            cells.append(latex_escape(format_mean_std(row, metric)))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scipy_ttest(kind: str, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    try:
        from scipy import stats
    except Exception:
        return float("nan"), float("nan")

    if kind == "paired":
        if a.size < 2 or b.size < 2 or a.size != b.size:
            return float("nan"), float("nan")
        res = stats.ttest_rel(a, b, nan_policy="omit")
        return float(res.statistic), float(res.pvalue)

    if kind == "independent":
        if a.size < 2 or b.size < 2:
            return float("nan"), float("nan")
        res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(res.statistic), float(res.pvalue)

    raise ValueError(f"Unsupported t-test kind: {kind}")


def paired_values(
    baseline_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    *,
    pair_by: list[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    base_map: dict[tuple[Any, ...], float] = {}
    group_map: dict[tuple[Any, ...], float] = {}

    for row in baseline_rows:
        x = maybe_float(row.get(metric))
        if x is None:
            continue
        base_map[row_group_key(row, pair_by)] = float(x)

    for row in group_rows:
        x = maybe_float(row.get(metric))
        if x is None:
            continue
        group_map[row_group_key(row, pair_by)] = float(x)

    keys = sorted(set(base_map.keys()) & set(group_map.keys()), key=lambda k: str(k))

    a = np.asarray([base_map[k] for k in keys], dtype=np.float64)
    b = np.asarray([group_map[k] for k in keys], dtype=np.float64)
    return a, b, len(keys)


def independent_values(
    baseline_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    *,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(
        [x for x in (maybe_float(row.get(metric)) for row in baseline_rows) if x is not None],
        dtype=np.float64,
    )
    b = np.asarray(
        [x for x in (maybe_float(row.get(metric)) for row in group_rows) if x is not None],
        dtype=np.float64,
    )
    return a, b


def compute_ttests(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    baseline_query: dict[str, str],
    group_by: list[str],
    pair_by: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    if kind == "none":
        return []

    if not baseline_query:
        raise RuntimeError("--ttest requires --baseline_query.")

    baseline_rows = [row for row in rows if row_matches_query(row, baseline_query)]
    if not baseline_rows:
        raise RuntimeError(f"No baseline rows matched query: {baseline_query}")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row_group_key(row, group_by), []).append(row)

    results = []

    for key, group_rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if all(row_matches_query(row, baseline_query) for row in group_rows):
            continue

        group_fields = {field: value for field, value in zip(group_by, key)}

        for metric in metrics:
            if kind == "paired":
                a, b, paired_n = paired_values(
                    baseline_rows,
                    group_rows,
                    pair_by=pair_by,
                    metric=metric,
                )
                t_stat, p_value = scipy_ttest(kind, a, b)
                n_base = int(a.size)
                n_group = int(b.size)
            else:
                a, b = independent_values(baseline_rows, group_rows, metric=metric)
                t_stat, p_value = scipy_ttest(kind, a, b)
                paired_n = 0
                n_base = int(a.size)
                n_group = int(b.size)

            results.append(
                {
                    "test": kind,
                    **group_fields,
                    "metric": metric,
                    "baseline_query": ",".join(f"{k}={v}" for k, v in baseline_query.items()),
                    "pair_by": ",".join(pair_by),
                    "n_baseline": n_base,
                    "n_group": n_group,
                    "n_pairs": int(paired_n),
                    "baseline_mean": float(np.mean(a)) if a.size else "",
                    "group_mean": float(np.mean(b)) if b.size else "",
                    "mean_delta_group_minus_baseline": (
                        float(np.mean(b) - np.mean(a)) if a.size and b.size else ""
                    ),
                    "t_stat": t_stat,
                    "p_value": p_value,
                }
            )

    return results


def write_ttest_markdown(path: Path, rows: list[dict[str, Any]], *, group_by: list[str]) -> None:
    cols = list(group_by) + [
        "metric",
        "n_baseline",
        "n_group",
        "n_pairs",
        "baseline_mean",
        "group_mean",
        "mean_delta_group_minus_baseline",
        "p_value",
    ]

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")

    for row in rows:
        cells = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                if math.isnan(val):
                    cells.append("nan")
                else:
                    cells.append(f"{val:.6g}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(
    *,
    input_root: Path,
    output_dir: Path,
    mode: str,
    metrics_requested: list[str] | None,
    group_by: list[str] | None,
    ttest_kind: str,
    baseline_query_text: str,
    pair_by: list[str] | None,
    ttest_group_by: list[str] | None,
) -> dict[str, Any]:
    input_root = Path(input_root).expanduser().resolve(strict=False)
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = discover_rows(input_root, mode)
    metrics = choose_metric_names(rows, metrics_requested)

    group_fields = list(group_by) if group_by is not None else default_group_by(mode)
    summary_rows = summarize_rows(rows, group_by=group_fields, metrics=metrics)

    aggregated_csv = output_dir / "aggregated_metrics.csv"
    aggregated_json = output_dir / "aggregated_metrics.json"
    summary_csv = output_dir / "summary_by_group.csv"
    summary_md = output_dir / "summary_table.md"
    summary_tex = output_dir / "summary_table.tex"

    write_csv(aggregated_csv, rows)
    dump_json(aggregated_json, rows)
    write_csv(summary_csv, summary_rows)
    write_markdown_summary(summary_md, summary_rows, group_by=group_fields, metrics=metrics)
    write_latex_summary(summary_tex, summary_rows, group_by=group_fields, metrics=metrics)

    baseline_query = parse_query(baseline_query_text)
    ttest_rows: list[dict[str, Any]] = []

    if ttest_kind != "none":
        ttest_group_fields = (
            list(ttest_group_by)
            if ttest_group_by is not None
            else group_fields
        )
        pair_fields = (
            list(pair_by)
            if pair_by is not None
            else default_pair_by(mode)
        )

        ttest_rows = compute_ttests(
            rows,
            kind=ttest_kind,
            baseline_query=baseline_query,
            group_by=ttest_group_fields,
            pair_by=pair_fields,
            metrics=metrics,
        )

        write_csv(output_dir / "ttest_results.csv", ttest_rows)
        dump_json(output_dir / "ttest_results.json", ttest_rows)
        write_ttest_markdown(output_dir / "ttest_results.md", ttest_rows, group_by=ttest_group_fields)

    manifest = {
        "schema": "ogd_lbs_aggregate_metrics_manifest_v1",
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "mode": mode,
        "row_count": int(len(rows)),
        "summary_group_by": group_fields,
        "metrics": metrics,
        "outputs": {
            "aggregated_metrics_csv": str(aggregated_csv),
            "aggregated_metrics_json": str(aggregated_json),
            "summary_by_group_csv": str(summary_csv),
            "summary_table_md": str(summary_md),
            "summary_table_tex": str(summary_tex),
        },
        "ttest": {
            "kind": ttest_kind,
            "baseline_query": baseline_query,
            "row_count": int(len(ttest_rows)),
            "outputs": (
                {
                    "ttest_results_csv": str(output_dir / "ttest_results.csv"),
                    "ttest_results_json": str(output_dir / "ttest_results.json"),
                    "ttest_results_md": str(output_dir / "ttest_results.md"),
                }
                if ttest_kind != "none"
                else {}
            ),
        },
    }
    dump_json(output_dir / "aggregate_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve(strict=False)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    else:
        output_dir = input_root / "aggregated_reports"

    manifest = aggregate(
        input_root=input_root,
        output_dir=output_dir,
        mode=str(args.mode),
        metrics_requested=args.metrics,
        group_by=args.group_by,
        ttest_kind=str(args.ttest),
        baseline_query_text=str(args.baseline_query),
        pair_by=args.pair_by,
        ttest_group_by=args.ttest_group_by,
    )

    print("=" * 96, flush=True)
    print("OGD-LBS aggregate metrics complete", flush=True)
    print(f"input_root : {manifest['input_root']}", flush=True)
    print(f"output_dir : {manifest['output_dir']}", flush=True)
    print(f"mode       : {manifest['mode']}", flush=True)
    print(f"rows       : {manifest['row_count']}", flush=True)
    print(f"metrics    : {manifest['metrics']}", flush=True)
    print(f"summary md : {manifest['outputs']['summary_table_md']}", flush=True)
    if manifest["ttest"]["kind"] != "none":
        print(f"ttest rows : {manifest['ttest']['row_count']}", flush=True)
        print(f"ttest md   : {manifest['ttest']['outputs']['ttest_results_md']}", flush=True)
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
