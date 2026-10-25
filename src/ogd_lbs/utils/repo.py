# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Repository discovery and sys.path bootstrap helpers.

Note:
    Standalone scripts that cannot import ogd_lbs yet may still need a tiny
    inline bootstrap before importing this module.  Once ogd_lbs is importable,
    these helpers provide the canonical implementation.
"""

import sys
from pathlib import Path
from collections.abc import Sequence


def _start_dir(current_file: str | Path | None) -> Path:
    if current_file is None:
        return Path.cwd().expanduser().resolve(strict=False)

    path = Path(current_file).expanduser().resolve(strict=False)

    # __file__ points to a file; many CLI callers pass directories.
    if path.suffix:
        return path.parent

    return path


def resolve_repo_root(
    current_file: str | Path | None = None,
    *,
    required_markers: Sequence[str] = ("pyproject.toml", "src"),
    require: bool = True,
) -> Path | None:
    """
    Find the repository root by walking upward from current_file or cwd.

    A directory is considered the repo root when all required_markers exist
    under it.  For OGD-LBS the canonical markers are:

        pyproject.toml
        src
    """
    start = _start_dir(current_file)

    for parent in [start, *start.parents]:
        if all((parent / marker).exists() for marker in required_markers):
            return parent.resolve(strict=False)

    if require:
        raise RuntimeError(
            "Could not locate repository root containing markers: "
            f"{list(required_markers)} from start={start}"
        )

    return None


def ensure_repo_paths_in_syspath(
    current_file: str | Path | None = None,
    *,
    extra_relative_dirs: Sequence[str | Path] = (),
    include_src: bool = True,
    prepend: bool = True,
) -> Path:
    """
    Add repository paths to sys.path and return repo root.

    Typical use:
        ensure_repo_paths_in_syspath(
            __file__,
            extra_relative_dirs=("scripts/data_prep",),
        )
    """
    repo_root = resolve_repo_root(current_file, require=True)
    assert repo_root is not None

    paths: list[Path] = []

    if include_src:
        paths.append(repo_root / "src")

    for rel in extra_relative_dirs:
        paths.append(repo_root / Path(rel))

    for path in paths:
        path_s = str(path.resolve(strict=False))
        if path_s in sys.path:
            continue

        if prepend:
            sys.path.insert(0, path_s)
        else:
            sys.path.append(path_s)

    return repo_root


__all__ = [
    "resolve_repo_root",
    "ensure_repo_paths_in_syspath",
]
