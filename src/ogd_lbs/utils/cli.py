# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Small CLI parsing helpers shared by scripts.

This module is pure standard library and has no project import dependencies.
"""

import argparse
import os
import sys
from typing import Any


def script_argv() -> list[str]:
    """
    Return script args after Blender's '--' separator when present.

    Works both for:
        python script.py --foo
    and:
        blender --python script.py -- --foo
    """
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def parse_csv_items(text: str | None) -> list[str]:
    if text is None or not str(text).strip():
        return []

    return [item.strip() for item in str(text).split(",") if item.strip()]


def flatten_cli_items(items: list[str] | tuple[str, ...] | None) -> list[str]:
    out: list[str] = []

    for item in items or []:
        out.extend(parse_csv_items(str(item)))

    return out


def parse_repeated_names(items: list[str] | tuple[str, ...] | None) -> list[str]:
    """
    Backward-compatible alias used by several debug scripts.
    """
    return flatten_cli_items(items)


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"yes", "true", "t", "y", "1", "on"}:
        return True

    if text in {"no", "false", "f", "n", "0", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}.")


def parse_idx_ranges(text: str | None) -> set[tuple[int, int]]:
    """
    Parse left-closed, right-open integer ranges.

    Example:
        "20:60,260:340" -> {(20, 60), (260, 340)}
    """
    if text is None or not str(text).strip():
        return set()

    ranges: set[tuple[int, int]] = set()

    for chunk in str(text).split(","):
        item = chunk.strip()
        if not item:
            continue

        if ":" not in item:
            raise ValueError(f"Invalid range item {item!r}. Expected format start:end.")

        start_s, end_s = item.split(":", 1)
        start = int(start_s.strip())
        end = int(end_s.strip())

        if start < 0 or end < 0:
            raise ValueError(f"Range indices must be non-negative: {item}")

        if end < start:
            raise ValueError(f"Range end must be >= start: {item}")

        ranges.add((start, end))

    return ranges


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


__all__ = [
    "script_argv",
    "parse_csv_items",
    "flatten_cli_items",
    "parse_repeated_names",
    "str2bool",
    "parse_idx_ranges",
    "env_bool",
]
