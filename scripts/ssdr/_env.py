# -*- coding: utf-8 -*-
from __future__ import annotations

"""
SSDR CLI bootstrap.

This module exists only for scripts under scripts/ssdr/. It makes the
repository's src/ package and scripts/ssdr/ helpers importable when an SSDR
entrypoint is executed directly by Python.

It intentionally contains no SSDR business logic, no Blender dependency, and
no asset-specific logic.
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
    ssdr_dir = repo_root / "scripts" / "ssdr"

    for item in [src_dir, ssdr_dir]:
        item_s = str(item)
        if item_s not in sys.path:
            sys.path.insert(0, item_s)

    return repo_root


REPO_ROOT = ensure_repo_paths_in_syspath()
SRC_DIR = REPO_ROOT / "src"
SSDR_DIR = REPO_ROOT / "scripts" / "ssdr"
