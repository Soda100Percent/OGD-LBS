# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Shared SSDR staged-pipeline I/O helpers.

This module contains only script-level infrastructure:
    - safe output directory preparation;
    - JSON-safe summary writing;
    - lightweight required-object helpers;
    - generic stage file discovery.

It intentionally contains no SSDR algorithm logic, no dataset-specific paths,
and no Blender dependency.
"""

try:
    from _env import ensure_repo_paths_in_syspath
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from ._env import ensure_repo_paths_in_syspath

ensure_repo_paths_in_syspath(__file__)

import datetime
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ogd_lbs.utils.common import (  # noqa: E402
    fail,
    warn,
    load_data,
    require_file,
    require_dir,
)


def path_has_any_children(path: str | Path) -> bool:
    path = Path(path)

    if not path.exists() or not path.is_dir():
        return False

    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def prepare_output_dir(
    output_tree: str | Path,
    *,
    overwrite_output: bool,
    allow_merge: bool,
) -> Path:
    """
    Prepare an SSDR stage output directory.

    Semantics are intentionally shared by SSDR Step 00-04:
        - --overwrite_output and --allow_merge are mutually exclusive;
        - non-empty existing output fails unless one of those flags is set;
        - overwrite removes the whole output directory and recreates it;
        - allow_merge writes into an existing directory.
    """
    output_tree = Path(output_tree).expanduser().resolve(strict=False)

    if overwrite_output and allow_merge:
        fail("--overwrite_output and --allow_merge are mutually exclusive.")

    if output_tree.exists():
        if not output_tree.is_dir():
            fail(f"Output path exists but is not a directory: {output_tree}")

        if overwrite_output:
            warn(f"Removing existing output directory because --overwrite_output is set: {output_tree}")
            shutil.rmtree(output_tree)
            output_tree.mkdir(parents=True, exist_ok=True)
            return output_tree

        if path_has_any_children(output_tree) and not allow_merge:
            fail(
                f"Output directory already exists and is non-empty: {output_tree}\n"
                "Use --overwrite_output to clear it, or --allow_merge to write into it."
            )

        output_tree.mkdir(parents=True, exist_ok=True)
        return output_tree

    output_tree.mkdir(parents=True, exist_ok=True)
    return output_tree


def json_safe(obj: Any) -> Any:
    """
    Convert common NumPy / Path objects into JSON-serializable values.

    This is kept in scripts/ssdr for now because it is used for SSDR stage
    summaries. If later needed more broadly, it can be promoted to
    ogd_lbs.utils.common.
    """
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
        return sorted((json_safe(x) for x in obj), key=lambda x: str(x))

    return obj


def write_stage_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    """
    Write one SSDR step summary JSON.

    Uses indent=4 to preserve the formatting style used by the original SSDR
    step scripts.
    """
    out = Path(path).expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=4, ensure_ascii=False)

    return out



_PROGRESS_SEQUENCE = 0


def _progress_utc_now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _validate_progress_count(value: int | float, *, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a non-negative finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a non-negative finite number")
    return int(value) if isinstance(value, int) else float(value)


def report_stage_progress(
    *,
    stage: str | None = None,
    phase: str,
    completed_units: int | float,
    total_units: int | float | None,
    unit: str,
    message: str = "",
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Write an optional pd-rew stage progress snapshot.

    This helper is intentionally inert unless PD_REW_STAGE_PROGRESS_FILE is set.
    It lets OGD-LBS remain usable as a standalone CLI while allowing pd-rew to
    observe long-running stages through a bind-mounted JSON progress file.

    The helper writes by same-directory temporary file followed by os.replace(),
    so readers should never observe a partially-written JSON object.
    """
    raw_output = os.environ.get("PD_REW_STAGE_PROGRESS_FILE")
    if not raw_output:
        return None

    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be a non-empty string")
    if not isinstance(unit, str) or not unit:
        raise ValueError("unit must be a non-empty string")

    completed = _validate_progress_count(completed_units, field="completed_units")

    if total_units is None:
        total: int | float | None = None
    else:
        total = _validate_progress_count(total_units, field="total_units")
        if completed > total:
            raise ValueError("completed_units must not exceed total_units")

    if extra is not None and not isinstance(extra, dict):
        raise ValueError("extra must be a dict or None")

    global _PROGRESS_SEQUENCE
    _PROGRESS_SEQUENCE += 1

    stage_name = str(stage or os.environ.get("PD_REW_STAGE_NAME") or "unknown")

    obj: dict[str, Any] = {
        "schema": "pd_rew_stage_progress_v1",
        "run_id": os.environ.get("PD_REW_RUN_ID"),
        "worker_id": os.environ.get("PD_REW_WORKER_ID"),
        "case_id": os.environ.get("PD_REW_CASE_ID"),
        "attempt_id": os.environ.get("PD_REW_ATTEMPT_ID"),
        "stage": stage_name,
        "phase": str(phase),
        "sequence": int(_PROGRESS_SEQUENCE),
        "completed_units": completed,
        "total_units": total,
        "unit": str(unit),
        "message": str(message),
        "updated_at_utc": _progress_utc_now_iso(),
    }

    if extra:
        obj["extra"] = json_safe(extra)

    out = Path(raw_output).expanduser().resolve(strict=False)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.{_PROGRESS_SEQUENCE}.tmp")
    data = (
        json.dumps(json_safe(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, out)

        try:
            dir_fd = os.open(str(out.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass

        return out

    except Exception as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        warn(f"Failed to write pd-rew stage progress file {out}: {exc}")
        return None



def require_dict(
    obj: Any,
    *,
    role: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(obj, dict):
        if path is None:
            fail(f"{role} must be a dictionary object.")
        fail(f"{role} must be a dictionary object: {path}")

    return obj


def load_required_dict(path: str | Path, *, role: str) -> dict[str, Any]:
    resolved = require_file(path, role=role)
    obj = load_data(resolved)
    return require_dict(obj, role=role, path=resolved)


def _normalize_suffixes(suffixes: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(suffixes, str):
        raw_items = [suffixes]
    else:
        raw_items = list(suffixes)

    out: list[str] = []
    for item in raw_items:
        text = str(item).strip().lower()
        if not text:
            continue
        if not text.startswith("."):
            text = "." + text
        if text not in out:
            out.append(text)

    if not out:
        raise ValueError("At least one suffix is required.")

    return tuple(out)


def discover_stage_files(
    root: str | Path,
    *,
    suffixes: str | Sequence[str],
    exclude_stems: Iterable[str] | None = None,
    recursive: bool = False,
    resolve_by_stem: bool = False,
) -> list[Path]:
    """
    Discover stage files with optional extension-priority resolution.

    Args:
        root:
            Directory to scan.
        suffixes:
            Accepted suffixes in priority order, e.g. (".pkl", ".json").
        exclude_stems:
            File stems to skip.
        recursive:
            If True, scan recursively.
        resolve_by_stem:
            If True, keep only one file per stem, using suffix priority.

    Returns:
        Sorted list of Paths.
    """
    root_path = require_dir(root, role="stage root")
    suffix_priority = _normalize_suffixes(suffixes)
    exclude = {str(x) for x in (exclude_stems or set())}

    if recursive:
        iterator = root_path.rglob("*")
    else:
        iterator = root_path.iterdir()

    candidates: list[Path] = []
    for item in iterator:
        if not item.is_file():
            continue

        if item.stem in exclude:
            continue

        if item.suffix.lower() not in suffix_priority:
            continue

        candidates.append(item.resolve(strict=False))

    candidates = sorted(candidates, key=lambda p: str(p.relative_to(root_path)))

    if not resolve_by_stem:
        return candidates

    best_by_stem: dict[str, Path] = {}

    for path in candidates:
        stem = path.stem
        if stem not in best_by_stem:
            best_by_stem[stem] = path
            continue

        current = best_by_stem[stem]
        new_rank = suffix_priority.index(path.suffix.lower())
        old_rank = suffix_priority.index(current.suffix.lower())

        if new_rank < old_rank:
            best_by_stem[stem] = path

    return sorted(best_by_stem.values(), key=lambda p: p.name)
