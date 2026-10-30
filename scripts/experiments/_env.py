# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Experiment CLI bootstrap.

This module exists only for scripts under scripts/experiments/.
It makes repository src/, scripts/ssdr/, scripts/data_prep/, and
scripts/experiments/ importable when experiment entrypoints are executed
directly.
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

    for item in [
        repo_root / "src",
        repo_root / "scripts" / "experiments",
        repo_root / "scripts" / "ssdr",
        repo_root / "scripts" / "data_prep",
    ]:
        item_s = str(item)
        if item_s not in sys.path:
            sys.path.insert(0, item_s)

    return repo_root


REPO_ROOT = ensure_repo_paths_in_syspath()
SRC_DIR = REPO_ROOT / "src"
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
SSDR_DIR = REPO_ROOT / "scripts" / "ssdr"
DATA_PREP_DIR = REPO_ROOT / "scripts" / "data_prep"
