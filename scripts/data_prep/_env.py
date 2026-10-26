# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Data Prep CLI bootstrap.

This module exists only for scripts under scripts/data_prep/.  It makes the
repository's src/ package and scripts/data_prep/ helpers importable when the
entrypoint is executed directly by Python, Blender Python, or bpy environments.

It intentionally contains no Data Prep business logic, no Blender dependency,
and no asset-specific logic.
"""

from pathlib import Path
import sys


def find_repo_root(current_file: str | Path | None = None) -> Path:
    cur = Path(current_file or __file__).expanduser().resolve(strict=False)

    for parent in [cur.parent, *cur.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent

    raise RuntimeError(
        "Could not locate repository root containing pyproject.toml and src/."
    )


def ensure_repo_paths_in_syspath(current_file: str | Path | None = None) -> Path:
    repo_root = find_repo_root(current_file)
    src_dir = repo_root / "src"
    data_prep_dir = repo_root / "scripts" / "data_prep"

    for item in [src_dir, data_prep_dir]:
        item_s = str(item)
        if item_s not in sys.path:
            sys.path.insert(0, item_s)

    return repo_root


REPO_ROOT = ensure_repo_paths_in_syspath()
SRC_DIR = REPO_ROOT / "src"
DATA_PREP_DIR = REPO_ROOT / "scripts" / "data_prep"
