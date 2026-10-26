# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic file-state helpers.

Asset-agnostic by design:
    - no dataset paths
    - no character names
    - no pipeline stage naming

This module is intended for lightweight provenance / manifest logic, for
example recording file size and mtime_ns, then later checking whether the file
still matches the recorded state.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import time


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def unix_timestamp() -> float:
    return float(time.time())


def file_metadata(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=False)
    stat = resolved.stat()

    return {
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def metadata_matches(
    path: str | Path,
    metadata: dict[str, Any],
    *,
    fields: Iterable[str] = ("size", "mtime_ns"),
) -> bool:
    """
    Return True when selected metadata fields match current file state.

    Missing file, non-dict metadata, or missing fields all return False.
    """
    if not isinstance(metadata, dict):
        return False

    try:
        current = file_metadata(path)
    except OSError:
        return False

    for field in fields:
        name = str(field)

        if name not in current or name not in metadata:
            return False

        if name in {"size", "mtime_ns"}:
            try:
                if int(current[name]) != int(metadata[name]):
                    return False
            except Exception:
                return False
        else:
            if current[name] != metadata[name]:
                return False

    return True
