#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generate a minimal standalone HTML report from aggregate_metrics.py outputs.

Input:
    <aggregate_dir>/
    ├── aggregate_manifest.json
    ├── aggregated_metrics.csv
    ├── summary_by_group.csv
    └── ttest_results.csv, optional

Output:
    OGD_LBS_Report.html

This report intentionally does not fully clone the legacy SSDR dashboard.
It provides the minimum visualization loop needed for revision experiments:
    tables + key metric bars + Pareto scatter + SVG/PNG/PDF frontend export.
"""

import argparse
import csv
import datetime as _dt
import html
import json
import math
import socket
import http.server
import socketserver
from pathlib import Path
from typing import Any

try:
    from _env import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from ._env import REPO_ROOT  # noqa: F401


METRIC_SPECS = [
    ("04_L2_Err", "L2 Error ↓", "min"),
    ("04_Hausdorff_Mean", "Hausdorff Mean ↓", "min"),
    ("04_Hausdorff_Max", "Hausdorff Max ↓", "min"),
    ("04_Org_Loss", "Org Loss ↓", "min"),
    ("04_ARAP_Ratio", "ARAP Ratio ↓", "min"),
    ("04_Max_Roll", "Max Roll ↓", "min"),
    ("04_Max_Swing", "Max Swing ↓", "min"),
    ("04_S_Max", "s_max", "info"),
    ("04_Mask_NNZ_Mean", "Mask NNZ Mean", "info"),
    ("04_Final_Weight_NNZ_Mean", "Final Weight NNZ Mean", "info"),
    ("04_Final_Weight_NNZ_Max", "Final Weight NNZ Max", "info"),
]

BAR_METRIC_PRIORITY = [
    "04_L2_Err",
    "04_Hausdorff_Mean",
    "04_Hausdorff_Max",
    "04_Org_Loss",
    "04_ARAP_Ratio",
]


DEFAULT_SWEEP_VALUES = {
    "alpha": "0.5",
    "beta": "0.3",
    "gamma": "0.2",
    "s_max": "4",
    "tau": "adaptive",
    "omega_b": "1000.0",
}

REPORT_METRIC_CHART_SPECS = [
    ("04_L2_Err", "VD / L2 Error ↓", "min"),
    ("04_Org_Loss", "SO / Org Loss ↓", "min"),
    ("04_Hausdorff_Mean", "HD Mean ↓", "min"),
    ("04_Hausdorff_Max", "HD Max ↓", "min"),
    ("04_Max_Roll", "Max Roll ↓", "min"),
    ("04_Max_Swing", "Max Swing ↓", "min"),
]

PER_FRAME_STD_FIELD_BY_METRIC = {
    # Per-case, per-frame final standard deviations computed from
    # opt_out/metrics_history_per_frame.npz by aggregate_metrics.py.
    #
    # These std fields are meaningful for model-grouped raw charts:
    # each raw bar is one model/value case, and the error bar reflects
    # frame-to-frame variation inside that case.
    "04_L2_Err": "history_l2_err_final_std",
    "04_Org_Loss": "history_org_loss_final_std",
    "04_Max_Roll": "history_bone_roll_max_final_std",
    "04_Max_Swing": "history_bone_swing_max_final_std",

    # Current history npz does not contain per-frame Hausdorff values.
    "04_Hausdorff_Mean": None,
    "04_Hausdorff_Max": None,
}

CONFIG_COLUMNS_BY_MODE = {
    "sweep": [
        "model",
        "param",
        "value",
        "case_id",
        "stage",
        "eff_alpha",
        "eff_beta",
        "eff_gamma",
        "eff_s_max",
        "eff_tau",
        "eff_omega_b",
        "summary_config_s_max",
        "summary_config_mask_mode",
        "summary_config_mask_threshold",
    ],
    "cv": [
        "model",
        "fold_index",
        "split",
        "split_train_count",
        "split_test_count",
        "split_special_count",
        "summary_config_s_max",
        "summary_config_mask_mode",
        "summary_config_mask_threshold",
        "summary_config_freeze_weights",
    ],
    "generic": [
        "run_name",
        "relative_run_root",
        "summary_config_s_max",
        "summary_config_mask_mode",
        "summary_config_mask_threshold",
        "summary_config_freeze_weights",
    ],
}



PER_FRAME_HISTORY_FIELDS_BY_METRIC = {
    # Per-frame final fluctuation ranges and strict std values derived from
    # opt_out/metrics_history_per_frame.npz by aggregate_metrics.py.
    #
    # Chart semantics:
    #   bar      = case-level metric value, e.g. 04_L2_Err
    #   whisker  = per-frame final min-max fluctuation range
    #   tooltip/table = strict per-frame std
    "04_L2_Err": {
        "label": "VD",
        "mean": "history_l2_err_final_mean",
        "std": "history_l2_err_final_std",
        "min": "history_l2_err_final_min",
        "max": "history_l2_err_final_max",
    },
    "04_Org_Loss": {
        "label": "SO",
        "mean": "history_org_loss_final_mean",
        "std": "history_org_loss_final_std",
        "min": "history_org_loss_final_min",
        "max": "history_org_loss_final_max",
    },
    "04_Max_Roll": {
        "label": "Roll",
        "mean": "history_bone_roll_max_final_mean",
        "std": "history_bone_roll_max_final_std",
        "min": "history_bone_roll_max_final_min",
        "max": "history_bone_roll_max_final_max",
    },
    "04_Max_Swing": {
        "label": "Swing",
        "mean": "history_bone_swing_max_final_mean",
        "std": "history_bone_swing_max_final_std",
        "min": "history_bone_swing_max_final_min",
        "max": "history_bone_swing_max_final_max",
    },

    # Current history npz does not contain per-frame Hausdorff values.
    "04_Hausdorff_Mean": None,
    "04_Hausdorff_Max": None,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate minimal standalone OGD-LBS experiment HTML report."
    )
    parser.add_argument("--aggregate_dir", type=str, required=True)
    parser.add_argument("--mode", choices=["sweep", "cv", "generic"], default=None)
    parser.add_argument("--output_html", type=str, default=None)
    parser.add_argument("--title", type=str, default="OGD-LBS Experiment Report")
    parser.add_argument(
        "--default_sweep_values",
        type=str,
        default="",
        help=(
            "Comma-separated default sweep values for highlighting, e.g. "
            "alpha=0.5,beta=0.3,gamma=0.2,s_max=4,tau=adaptive,omega_b=1000.0"
        ),
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def format_cell(value: Any) -> str:
    x = finite_float(value)
    if x is None:
        return html_escape(value)

    if abs(x) >= 1_000_000 or (abs(x) > 0.0 and abs(x) < 0.001):
        return html_escape(f"{x:.4e}")
    return html_escape(f"{x:.6g}")


def slugify(value: Any) -> str:
    import re
    s = str(value)
    s = s.replace(".", "p").replace("-", "m").replace("+", "p")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def infer_mode(aggregate_dir: Path, manifest: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit

    mode = manifest.get("mode")
    if mode in {"sweep", "cv", "generic"}:
        return str(mode)

    rows = read_csv_rows(aggregate_dir / "aggregated_metrics.csv")
    if rows:
        first = rows[0]
        if first.get("mode") in {"sweep", "cv", "generic"}:
            return str(first["mode"])
        if "param" in first and "value" in first:
            return "sweep"
        if "fold_index" in first and "split" in first:
            return "cv"

    return "generic"


def decorate_rows(rows: list[dict[str, str]], mode: str) -> list[dict[str, Any]]:
    decorated = []

    for idx, row in enumerate(rows):
        out: dict[str, Any] = dict(row)

        if mode == "sweep":
            model = row.get("model", "")
            case_id = row.get("case_id", f"case_{idx}")
            param = row.get("param", "")
            value = row.get("value", "")
            display_id = f"{model}:{case_id}" if model else case_id
            label = f"{model} | {param}={value}" if model and param else display_id
            group_key = param or model or "sweep"

        elif mode == "cv":
            model = row.get("model", "")
            fold = row.get("fold_index", "")
            split = row.get("split", "")
            display_id = f"{model}:fold{fold}:{split}" if model else f"fold{fold}:{split}"
            label = f"{model} | fold {fold} | {split}" if model else display_id
            group_key = split or model or "cv"

        else:
            display_id = row.get("run_name") or row.get("relative_run_root") or f"run_{idx}"
            label = display_id
            group_key = row.get("run_name") or "generic"

        out["display_id"] = display_id
        out["label"] = label
        out["groupKey"] = group_key
        decorated.append(out)

    return decorated


def present_columns(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    return [c for c in columns if any(str(row.get(c, "")).strip() != "" for row in rows)]


def varying_columns(rows: list[dict[str, Any]], columns: list[str]) -> set[str]:
    out = set()
    for col in columns:
        vals = {str(row.get(col, "")) for row in rows}
        if len(vals) > 1:
            out.add(col)
    return out


def build_config_table(rows: list[dict[str, Any]], mode: str) -> str:
    cols = ["display_id"] + present_columns(rows, CONFIG_COLUMNS_BY_MODE.get(mode, []))

    if not rows:
        return "<div class='empty-section'>No rows found.</div>"

    varying = varying_columns(rows, cols)

    thead = "<tr>" + "".join(
        f"<th class='{'varying-col' if c in varying else ''}'>{html_escape(c)}</th>"
        for c in cols
    ) + "</tr>"

    tbody = ""
    for row in rows:
        tbody += "<tr>"
        for c in cols:
            cls = " class='varying-col'" if c in varying else ""
            tbody += f"<td{cls}>{format_cell(row.get(c, ''))}</td>"
        tbody += "</tr>"

    return f"<table class='config-table'><thead>{thead}</thead><tbody>{tbody}</tbody></table>"


def metric_specs_present(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    return [spec for spec in METRIC_SPECS if any(finite_float(row.get(spec[0])) is not None for row in rows)]


def compute_best_values(rows: list[dict[str, Any]], specs: list[tuple[str, str, str]]) -> dict[str, float]:
    best = {}
    for key, _title, direction in specs:
        if direction not in {"min", "max"}:
            continue
        values = [x for x in (finite_float(row.get(key)) for row in rows) if x is not None]
        if not values:
            continue
        best[key] = min(values) if direction == "min" else max(values)
    return best


def build_metrics_table(rows: list[dict[str, Any]]) -> str:
    specs = metric_specs_present(rows)
    cols = [("label", "Experiment", "info")] + specs

    if not rows:
        return "<div class='empty-section'>No metrics rows found.</div>"

    best = compute_best_values(rows, specs)

    thead = "<tr>" + "".join(f"<th>{html_escape(title)}</th>" for _key, title, _dir in cols) + "</tr>"

    tbody = ""
    for row in rows:
        tbody += "<tr>"
        for key, _title, direction in cols:
            val = row.get(key, "")
            cls = ""
            x = finite_float(val)
            if direction in {"min", "max"} and x is not None and key in best and abs(x - best[key]) <= 1e-12:
                cls = " class='best-val'"
            tbody += f"<td{cls}>{format_cell(val)}</td>"
        tbody += "</tr>"

    return f"<table class='metrics-table'><thead>{thead}</thead><tbody>{tbody}</tbody></table>"



def build_frame_fluctuation_table(rows: list[dict[str, Any]]) -> str:
    """
    Table view for strict per-frame std and min-max fluctuation ranges.

    This table intentionally separates strict std values from the chart visual
    whiskers, which represent min-max fluctuation ranges.
    """
    if not rows:
        return "<div class='empty-section'>No rows found.</div>"

    metric_specs = [
        ("VD", "04_L2_Err", PER_FRAME_HISTORY_FIELDS_BY_METRIC.get("04_L2_Err")),
        ("SO", "04_Org_Loss", PER_FRAME_HISTORY_FIELDS_BY_METRIC.get("04_Org_Loss")),
        ("Roll", "04_Max_Roll", PER_FRAME_HISTORY_FIELDS_BY_METRIC.get("04_Max_Roll")),
        ("Swing", "04_Max_Swing", PER_FRAME_HISTORY_FIELDS_BY_METRIC.get("04_Max_Swing")),
    ]

    cols: list[tuple[str, str]] = [
        ("model", "model"),
        ("param", "param"),
        ("value", "value"),
    ]

    for label, metric_key, hist in metric_specs:
        if not isinstance(hist, dict):
            continue
        cols.extend([
            (metric_key, label),
            (str(hist.get("min")), f"{label}_frame_min"),
            (str(hist.get("max")), f"{label}_frame_max"),
            (str(hist.get("std")), f"{label}_frame_std"),
        ])

    thead = "<tr>" + "".join(
        f"<th>{html_escape(title)}</th>"
        for _key, title in cols
    ) + "</tr>"

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            str(r.get("model", "")),
            str(r.get("param", "")),
            sweep_value_sort_key(r.get("value", "")),
        )
    )

    tbody = ""
    for row in sorted_rows:
        tbody += "<tr>"
        for key, _title in cols:
            tbody += f"<td>{format_cell(row.get(key, ''))}</td>"
        tbody += "</tr>"

    return f"<table class='frame-fluctuation-table'><thead>{thead}</thead><tbody>{tbody}</tbody></table>"


def build_plain_table(rows: list[dict[str, str]], *, max_cols: int = 24) -> str:
    if not rows:
        return "<div class='empty-section'>No rows found.</div>"

    cols = list(rows[0].keys())[:max_cols]
    thead = "<tr>" + "".join(f"<th>{html_escape(c)}</th>" for c in cols) + "</tr>"

    tbody = ""
    for row in rows:
        tbody += "<tr>"
        for c in cols:
            tbody += f"<td>{format_cell(row.get(c, ''))}</td>"
        tbody += "</tr>"

    return f"<table><thead>{thead}</thead><tbody>{tbody}</tbody></table>"


def chart_items_for_metric(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        x = finite_float(row.get(metric))
        if x is None:
            continue
        items.append(
            {
                "display_id": row.get("display_id", ""),
                "label": row.get("label", row.get("display_id", "")),
                "groupKey": row.get("groupKey", ""),
                "value": x,
            }
        )
    return items


def prepare_bar_charts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    title_by_metric = {key: title for key, title, _dir in METRIC_SPECS}

    bars = []
    for metric in BAR_METRIC_PRIORITY:
        items = chart_items_for_metric(rows, metric)
        if not items:
            continue
        bars.append(
            {
                "key": slugify(metric),
                "metric": metric,
                "title": title_by_metric.get(metric, metric),
                "yName": title_by_metric.get(metric, metric),
                "items": items,
            }
        )
    return bars


def choose_pareto_axes(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    def has_metric(m: str) -> bool:
        return any(finite_float(row.get(m)) is not None for row in rows)

    x = "04_L2_Err" if has_metric("04_L2_Err") else None

    y = None
    for cand in ["04_Hausdorff_Mean", "04_Org_Loss", "04_Hausdorff_Max"]:
        if has_metric(cand):
            y = cand
            break

    if x == y:
        y = None

    return x, y


def prepare_pareto(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    x_metric, y_metric = choose_pareto_axes(rows)

    if not x_metric or not y_metric:
        return {
            "xName": "",
            "yName": "",
            "filterKeys": [],
            "items": [],
        }

    filter_keys = [k for k in ["model", "param", "value", "split", "fold_index"] if any(k in row for row in rows)]

    items = []
    for row in rows:
        x = finite_float(row.get(x_metric))
        y = finite_float(row.get(y_metric))
        if x is None or y is None:
            continue

        params = {k: str(row.get(k, "N/A")) for k in filter_keys}

        items.append(
            {
                "display_id": row.get("display_id", ""),
                "label": row.get("label", row.get("display_id", "")),
                "groupKey": row.get("groupKey", ""),
                "x": x,
                "y": y,
                "params": params,
            }
        )

    return {
        "xName": x_metric,
        "yName": y_metric,
        "filterKeys": filter_keys,
        "items": items,
    }



def parse_default_sweep_values(text: str | None) -> dict[str, str]:
    out = dict(DEFAULT_SWEEP_VALUES)
    text = str(text or "").strip()
    if not text:
        return out

    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid default sweep value item {item!r}; expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            out[key] = value

    return out


def sweep_value_sort_key(value: Any) -> tuple[int, float | str]:
    text = str(value)
    try:
        return (0, float(text))
    except Exception:
        return (1, text)


def sweep_values_equal(a: Any, b: Any) -> bool:
    a_text = str(a)
    b_text = str(b)
    if a_text == b_text:
        return True

    try:
        return abs(float(a_text) - float(b_text)) <= 1e-12
    except Exception:
        return False


def is_default_sweep_value(param: Any, value: Any, default_sweep_values: dict[str, str]) -> bool:
    p = str(param)
    if p not in default_sweep_values:
        return False
    return sweep_values_equal(value, default_sweep_values[p])


def prepare_sweep_mean_std_charts(
    summary_rows: list[dict[str, str]],
    default_sweep_values: dict[str, str],
) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []

    if not summary_rows:
        return charts

    for metric, title, direction in REPORT_METRIC_CHART_SPECS:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        n_key = f"{metric}_n"

        items: list[dict[str, Any]] = []

        for row in summary_rows:
            if "param" not in row or "value" not in row:
                continue

            mean = finite_float(row.get(mean_key))
            if mean is None:
                continue

            std = finite_float(row.get(std_key))
            if std is None:
                std = 0.0

            n_val = finite_float(row.get(n_key))
            if n_val is None:
                n_val = finite_float(row.get("n_runs"))
            n = int(n_val) if n_val is not None else 0

            param = str(row.get("param", ""))
            value = str(row.get("value", ""))

            items.append(
                {
                    "param": param,
                    "value": value,
                    "mean": mean,
                    "std": std,
                    "n": n,
                    "low": mean - std,
                    "high": mean + std,
                    "direction": direction,
                    "isDefault": is_default_sweep_value(param, value, default_sweep_values),
                    "label": f"{param}={value}",
                }
            )

        items.sort(key=lambda x: (str(x.get("param", "")), sweep_value_sort_key(x.get("value", ""))))

        if items:
            charts.append(
                {
                    "key": slugify(f"sweep_mean_std_{metric}"),
                    "metric": metric,
                    "title": f"{title} — mean ± std across models",
                    "yName": title,
                    "items": items,
                }
            )

    return charts


def prepare_model_grouped_raw_charts(
    rows: list[dict[str, Any]],
    default_sweep_values: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Build raw metric chart data grouped visually by model.

    Each bar is one model/value result.

    For metrics with per-frame history:
      - bar height is the case-level metric value;
      - vertical whisker is the per-frame final min-max fluctuation range;
      - strict per-frame std is included for tooltip/table display.

    Hausdorff metrics currently have no per-frame history and therefore have
    no whiskers.
    """
    charts: list[dict[str, Any]] = []

    if not rows:
        return charts

    for metric, title, direction in REPORT_METRIC_CHART_SPECS:
        items: list[dict[str, Any]] = []
        chart_max = 0.0
        positive_metric = True

        history_spec = PER_FRAME_HISTORY_FIELDS_BY_METRIC.get(metric)

        for row in rows:
            x = finite_float(row.get(metric))
            if x is None:
                continue

            model = str(row.get("model", ""))
            param = str(row.get("param", ""))
            value = str(row.get("value", ""))

            if not model or not param:
                continue

            frame_mean = None
            frame_std = None
            frame_min = None
            frame_max = None
            has_frame_fluctuation = False

            if isinstance(history_spec, dict):
                frame_mean = finite_float(row.get(str(history_spec.get("mean"))))
                frame_std = finite_float(row.get(str(history_spec.get("std"))))
                frame_min = finite_float(row.get(str(history_spec.get("min"))))
                frame_max = finite_float(row.get(str(history_spec.get("max"))))
                has_frame_fluctuation = frame_min is not None and frame_max is not None

            if has_frame_fluctuation:
                assert frame_min is not None
                assert frame_max is not None
                raw_low = frame_min
                raw_high = frame_max
                draw_low = max(0.0, frame_min) if positive_metric else frame_min
                draw_high = max(frame_max, x)
                lower_clipped = draw_low != raw_low
                chart_max = max(chart_max, x, draw_high)
            else:
                raw_low = None
                raw_high = None
                draw_low = None
                draw_high = None
                lower_clipped = False
                chart_max = max(chart_max, x)

            items.append(
                {
                    "model": model,
                    "param": param,
                    "value": value,
                    "metricValue": x,
                    "direction": direction,
                    "isDefault": is_default_sweep_value(param, value, default_sweep_values),
                    "label": f"{model} | {param}={value}",

                    "historyLabel": history_spec.get("label") if isinstance(history_spec, dict) else "",
                    "frameMean": frame_mean,
                    "frameStd": frame_std,
                    "frameMin": frame_min,
                    "frameMax": frame_max,
                    "hasFrameFluctuation": bool(has_frame_fluctuation),
                    "rawLow": raw_low,
                    "rawHigh": raw_high,
                    "drawLow": draw_low,
                    "drawHigh": draw_high,
                    "lowerClippedAtZero": bool(lower_clipped),

                    # Backward-compatible R2f fields retained for tooltip/tests.
                    "perFrameStdField": str(history_spec.get("std")) if isinstance(history_spec, dict) else "",
                    "perFrameStd": frame_std,
                    "hasPerFrameStd": frame_std is not None,
                    "fluctuationLabel": "per-frame min-max range" if has_frame_fluctuation else "",
                }
            )

        items.sort(
            key=lambda x: (
                str(x.get("model", "")),
                str(x.get("param", "")),
                sweep_value_sort_key(x.get("value", "")),
            )
        )

        if items:
            padding = max(chart_max * 0.08, 1e-12)
            y_axis_max = chart_max + padding

            charts.append(
                {
                    "key": slugify(f"model_grouped_raw_{metric}"),
                    "metric": metric,
                    "title": f"{title} — raw values by model",
                    "yName": title,
                    "models": sorted({str(x.get("model", "")) for x in items}),
                    "items": items,
                    "positiveMetric": positive_metric,
                    "yAxisMin": 0.0 if positive_metric else None,
                    "yAxisMax": y_axis_max,
                    "fluctuationMode": "per-frame min-max range",
                    "stdSemantics": "Strict per-frame std is shown in tooltip/table, not as ±std error bar.",
                }
            )

    return charts


def json_dumps_strict(obj: Any) -> str:
    def sanitize(v: Any) -> Any:
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        if isinstance(v, dict):
            return {str(k): sanitize(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sanitize(x) for x in v]
        return v

    return json.dumps(sanitize(obj), ensure_ascii=False, allow_nan=False)


def read_template_file(name: str) -> str:
    path = Path(__file__).resolve(strict=False).parent / "report_templates" / name
    return path.read_text(encoding="utf-8")


def build_html(
    *,
    title: str,
    mode: str,
    aggregate_dir: Path,
    aggregate_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, str]],
    ttest_rows: list[dict[str, str]],
    default_sweep_values: dict[str, str] | None = None,
) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    default_sweep_values = default_sweep_values or dict(DEFAULT_SWEEP_VALUES)

    css = read_template_file("ogd_report_styles.css")
    js = read_template_file("ogd_report_dashboard.js")

    config_table = build_config_table(rows, mode)
    metrics_table = build_metrics_table(rows)
    summary_table = build_plain_table(summary_rows) if summary_rows else "<div class='empty-section'>No summary_by_group.csv found.</div>"
    ttest_table = build_plain_table(ttest_rows) if ttest_rows else "<div class='empty-section'>No t-test results found.</div>"
    frame_fluctuation_table = build_frame_fluctuation_table(rows)

    report_data = {
        "meta": {
            "title": title,
            "mode": mode,
            "aggregate_dir": str(aggregate_dir),
            "generated_at": now,
            "row_count": len(rows),
            "summary_row_count": len(summary_rows),
            "ttest_row_count": len(ttest_rows),
            "aggregate_manifest": aggregate_manifest,
        },
        "rows": rows,
        "summaryRows": summary_rows,
        "ttestRows": ttest_rows,
        "charts": {
            "bars": prepare_bar_charts(rows),
            "pareto": prepare_pareto(rows, mode),
            "modelGroupedRaw": prepare_model_grouped_raw_charts(rows, default_sweep_values),
            "defaultSweepValues": default_sweep_values,
        },
    }

    report_json = json_dumps_strict(report_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{html_escape(title)}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/jspdf@2/dist/jspdf.umd.min.js"></script>
    <style>
{css}
    </style>
    <script>
window.DASHBOARD_DATA = {report_json};
    </script>
</head>
<body>
    <div id="settings-btn" class="settings-btn">⚙️</div>
    <div id="settings-panel" class="settings-panel hidden">
        <h3>🎨 Appearance Settings</h3>
        <div class="setting-item">
            <label>Custom Font (.ttf/.otf):</label>
            <input type="file" id="font-upload" accept=".ttf,.otf">
        </div>
        <div class="setting-item">
            <label>Group Colors:</label>
            <div id="color-pickers-container" class="color-pickers-grid"></div>
        </div>
        <button id="btn-reset-appearance" class="btn-reset">↺ Reset to Default</button>
    </div>

    <h1>📊 {html_escape(title)}</h1>
    <div class="meta-line">
        Generated: {html_escape(now)} |
        Mode: {html_escape(mode)} |
        Rows: {len(rows)} |
        Aggregate: {html_escape(str(aggregate_dir))}
    </div>

    <h2>⚙️ Experiment Configurations</h2>
    <div class="info-note">Parameters with varying values across experiments are highlighted in yellow.</div>
    <div class="table-container">{config_table}</div>

    <h2>🏆 Highlighted Metrics Table</h2>
    <div class="table-container">{metrics_table}</div>

    <h2>📌 Grouped Summary Table</h2>
    <div class="table-container">{summary_table}</div>

    <h2>🧪 T-test Results</h2>
    <div class="table-container">{ttest_table}</div>

    <h2>📋 Per-frame Fluctuation / Std Table</h2>
    <div class="info-note">
        Strict per-frame standard deviation values are listed here.
        Chart whiskers show min-max fluctuation ranges, not ±std.
    </div>
    <div class="table-container">{frame_fluctuation_table}</div>

    <h2>🧩 Model-grouped Raw Metric Charts</h2>
    <div class="info-note">
        Bars are sorted by model and sweep value. Blank gaps separate models.
        When available, vertical whiskers show the per-frame final min-max fluctuation range within each case.
        Strict per-frame standard deviations are shown in the table and tooltip.
        HD currently has no per-frame history and is shown without whiskers.
    </div>
    <div id="model-grouped-raw-container"></div>

    <h2>📈 Interactive Charts</h2>
    <div id="dynamic-charts-container"></div>

    <h2>⚖️ Pareto Trade-off</h2>
    <div id="pareto-filters" class="filter-panel"></div>
    <div class="img-box">
        <h3>Pareto Scatter</h3>
        <div class="export-btn-container">
            <button class="btn-sm btn-svg" onclick="exportChart('chart-pareto', 'svg', 'chart-pareto')">📥 SVG</button>
            <button class="btn-sm btn-png" onclick="exportChart('chart-pareto', 'png', 'chart-pareto')">🖼️ PNG</button>
            <button class="btn-sm btn-pdf" onclick="exportChart('chart-pareto', 'pdf', 'chart-pareto')">📄 PDF</button>
        </div>
        <div id="chart-pareto" class="pareto-wrapper"></div>
    </div>

    <div class="footer">
        Generated by OGD-LBS Experiment Pipeline | Serverless Portable Dashboard | Frontend SVG/PNG/PDF Export
    </div>

    <script>
{js}
    </script>
</body>
</html>
"""


def generate_report(
    *,
    aggregate_dir: Path,
    output_html: Path,
    mode: str | None,
    title: str,
    default_sweep_values: dict[str, str] | None = None,
) -> Path:
    aggregate_dir = Path(aggregate_dir).expanduser().resolve(strict=False)
    output_html = Path(output_html).expanduser().resolve(strict=False)

    aggregate_manifest = load_json_if_exists(aggregate_dir / "aggregate_manifest.json")
    inferred_mode = mode or str(aggregate_manifest.get("mode", "generic"))

    raw_rows = read_csv_rows(aggregate_dir / "aggregated_metrics.csv")
    rows = decorate_rows(raw_rows, inferred_mode)
    summary_rows = read_csv_rows(aggregate_dir / "summary_by_group.csv")
    ttest_rows = read_csv_rows(aggregate_dir / "ttest_results.csv")

    html_text = build_html(
        title=title,
        mode=inferred_mode,
        aggregate_dir=aggregate_dir,
        aggregate_manifest=aggregate_manifest,
        rows=rows,
        summary_rows=summary_rows,
        ttest_rows=ttest_rows,
        default_sweep_values=default_sweep_values,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    return output_html


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def serve_dir(directory: Path, html_path: Path, port: int) -> None:
    import os
    os.chdir(directory)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    url = f"http://{get_local_ip()}:{port}/{html_path.name}"

    print("\n" + "=" * 80, flush=True)
    print("🌐 Starting local report server", flush=True)
    print(f"👉 Open: {url}", flush=True)
    print("=" * 80 + "\n", flush=True)

    with ReusableTCPServer(("", int(port)), http.server.SimpleHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.", flush=True)


def main() -> None:
    args = parse_args()

    aggregate_dir = Path(args.aggregate_dir).expanduser().resolve(strict=False)

    if args.output_html:
        output_html = Path(args.output_html).expanduser().resolve(strict=False)
    else:
        output_html = aggregate_dir / "OGD_LBS_Report.html"

    out = generate_report(
        aggregate_dir=aggregate_dir,
        output_html=output_html,
        mode=args.mode,
        title=str(args.title),
        default_sweep_values=parse_default_sweep_values(args.default_sweep_values),
    )

    print("=" * 96, flush=True)
    print("OGD-LBS report generated", flush=True)
    print(f"aggregate_dir : {aggregate_dir}", flush=True)
    print(f"output_html   : {out}", flush=True)
    print("=" * 96, flush=True)

    if args.serve:
        serve_dir(out.parent, out, int(args.port))


if __name__ == "__main__":
    main()
