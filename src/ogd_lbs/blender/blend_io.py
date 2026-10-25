# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic Blender .blend file IO helpers.

Asset-agnostic by design:
    - no dataset paths
    - no character names
    - no Data Prep stage naming
    - no SSDR naming

These helpers wrap Blender's open/save operators with consistent path
resolution, parent directory creation, and FINISHED-result checks.
"""

from pathlib import Path
from typing import Any

import bpy

from ogd_lbs.utils.common import require_file


def _result_is_finished(result: Any) -> bool:
    return result == {"FINISHED"}


def open_blend_file(
    path: str | Path,
    *,
    role: str = "blend file",
    load_ui: bool | None = None,
    verbose: bool = True,
) -> Path:
    """
    Open a Blender .blend file and validate the operator result.

    Args:
        path: Input .blend path.
        role: Human-readable role used in error messages.
        load_ui:
            None: call open_mainfile without load_ui.
            True/False: try passing load_ui, falling back if unsupported.
        verbose: Print a short loading line.

    Returns:
        Resolved input path.
    """
    resolved = require_file(path, role=role)

    if verbose:
        print(f"\n📂 Loading {role}: {resolved}", flush=True)

    try:
        if load_ui is None:
            result = bpy.ops.wm.open_mainfile(filepath=str(resolved))
        else:
            result = bpy.ops.wm.open_mainfile(
                filepath=str(resolved),
                load_ui=bool(load_ui),
            )
    except TypeError:
        result = bpy.ops.wm.open_mainfile(filepath=str(resolved))

    if not _result_is_finished(result):
        raise RuntimeError(f"Failed to open {role}: {resolved}, result={result}")

    return resolved


def save_blend_file(
    path: str | Path,
    *,
    role: str = "blend file",
    verbose: bool = True,
) -> Path:
    """
    Save the current Blender main file to a .blend path.

    Args:
        path: Output .blend path.
        role: Human-readable role used in logs / errors.
        verbose: Print a short saving line.

    Returns:
        Resolved output path.
    """
    resolved = Path(path).expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n💾 Saving {role} to: {resolved}", flush=True)

    result = bpy.ops.wm.save_as_mainfile(filepath=str(resolved))

    if not _result_is_finished(result):
        raise RuntimeError(f"Failed to save {role}: {resolved}, result={result}")

    return resolved
