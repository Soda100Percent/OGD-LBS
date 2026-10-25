# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Filesystem and output-directory helpers.

This module is deliberately independent from ogd_lbs.utils.common to avoid
circular imports.  Callers may pass log/warn/fail functions when they want
project-style logging.
"""

import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NoReturn, Literal


ExpectedPathType = Literal["file", "dir", "any"]


def resolve_path_no_strict(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def path_has_any_children(path: str | Path) -> bool:
    path = Path(path)

    if not path.exists() or not path.is_dir():
        return False

    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def _raise_or_fail(message: str, fail_fn: Callable[[str], Any] | None = None) -> NoReturn:
    if fail_fn is not None:
        fail_fn(message)
    raise RuntimeError(message)


def require_path(
    path: str | Path,
    *,
    role: str = "path",
    expected_type: ExpectedPathType = "any",
    must_exist: bool = True,
) -> Path:
    path = Path(path).expanduser().resolve(strict=False)

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Missing required {role}: {path}")

    if path.exists():
        if expected_type == "file" and not path.is_file():
            raise FileNotFoundError(f"Required {role} is not a file: {path}")
        if expected_type == "dir" and not path.is_dir():
            raise FileNotFoundError(f"Required {role} is not a directory: {path}")

    return path


def require_file(path: str | Path, *, role: str = "file") -> Path:
    return require_path(path, role=role, expected_type="file", must_exist=True)


def require_dir(path: str | Path, *, role: str = "directory") -> Path:
    return require_path(path, role=role, expected_type="dir", must_exist=True)


def prepare_output_dir(
    output_tree: str | Path,
    *,
    overwrite_output: bool = False,
    allow_merge: bool = False,
    warn_fn: Callable[[str], Any] | None = None,
    fail_fn: Callable[[str], Any] | None = None,
) -> Path:
    """
    Prepare an output directory with consistent overwrite / merge semantics.

    Rules:
        - --overwrite_output and --allow_merge are mutually exclusive.
        - If output exists and is non-empty:
            - overwrite_output=True removes and recreates it.
            - allow_merge=True keeps it.
            - otherwise fail.
        - If output exists as a file, fail.
    """
    output_tree = Path(output_tree).expanduser().resolve(strict=False)

    if overwrite_output and allow_merge:
        _raise_or_fail("--overwrite_output and --allow_merge are mutually exclusive.", fail_fn)

    if output_tree.exists():
        if not output_tree.is_dir():
            _raise_or_fail(f"Output path exists but is not a directory: {output_tree}", fail_fn)

        if overwrite_output:
            if warn_fn is not None:
                warn_fn(f"Removing existing output directory because --overwrite_output is set: {output_tree}")
            shutil.rmtree(output_tree)
            output_tree.mkdir(parents=True, exist_ok=True)
            return output_tree

        if path_has_any_children(output_tree) and not allow_merge:
            _raise_or_fail(
                f"Output directory already exists and is non-empty: {output_tree}\n"
                "Use --overwrite_output to clear it, or --allow_merge to write into it.",
                fail_fn,
            )

        output_tree.mkdir(parents=True, exist_ok=True)
        return output_tree

    output_tree.mkdir(parents=True, exist_ok=True)
    return output_tree


def rm_dir_contents(
    dir_path: str | Path,
    *,
    log_fn: Callable[[str], Any] | None = None,
    warn_fn: Callable[[str], Any] | None = None,
) -> None:
    dir_path = Path(dir_path)

    if dir_path.exists():
        if dir_path.is_dir():
            shutil.rmtree(dir_path)
            if log_fn is not None:
                log_fn(f"Dir and contents removed: {dir_path}")
            return

        raise IOError(f"Not a valid dir path: {dir_path}")

    if warn_fn is not None:
        warn_fn(f"Dir path doesn't exist: {dir_path}")


def create_or_clear_dir(
    dir_path: str | Path,
    *,
    log_fn: Callable[[str], Any] | None = None,
    warn_fn: Callable[[str], Any] | None = None,
) -> Path:
    dir_path = Path(dir_path)

    if dir_path.exists():
        rm_dir_contents(dir_path, log_fn=log_fn, warn_fn=warn_fn)
        dir_path.mkdir(parents=True, exist_ok=True)
        if log_fn is not None:
            log_fn(f"Dir cleared: {dir_path}")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)
        if log_fn is not None:
            log_fn(f"Dir created: {dir_path}")

    return dir_path


def resolve_specific_file(
    dir_path: str | Path,
    stem: str,
    ext_priority: list[str],
) -> Path | None:
    dir_path = Path(dir_path)
    ext_priority_lower = [ext.lower() for ext in ext_priority]

    for ext in ext_priority_lower:
        candidate = dir_path / f"{stem}{ext}"
        if candidate.is_file():
            return candidate

    return None


def discover_and_resolve_files(
    dir_path: str | Path,
    ext_priority: list[str],
    exclude_stems: Iterable[str] | None = None,
) -> list[Path]:
    """
    Discover direct child files and resolve extension conflicts by priority.

    Example:
        ext_priority=[".pkl", ".json"] means a.pkl wins over a.json.
    """
    dir_path = Path(dir_path)
    exclude_set = set(exclude_stems) if exclude_stems else set()
    ext_priority_lower = [ext.lower() for ext in ext_priority]

    files_by_stem: dict[str, Path] = {}

    if not dir_path.exists():
        return []

    for item in dir_path.iterdir():
        if not item.is_file():
            continue

        stem = item.stem
        ext = item.suffix.lower()

        if stem in exclude_set:
            continue
        if ext not in ext_priority_lower:
            continue

        if stem not in files_by_stem:
            files_by_stem[stem] = item
            continue

        current_best = files_by_stem[stem]
        if ext_priority_lower.index(ext) < ext_priority_lower.index(current_best.suffix.lower()):
            files_by_stem[stem] = item

    return sorted(list(files_by_stem.values()), key=lambda p: p.name)


__all__ = [
    "ExpectedPathType",
    "resolve_path_no_strict",
    "path_has_any_children",
    "require_path",
    "require_file",
    "require_dir",
    "prepare_output_dir",
    "rm_dir_contents",
    "create_or_clear_dir",
    "resolve_specific_file",
    "discover_and_resolve_files",
]
