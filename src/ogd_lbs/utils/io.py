# -*- coding: utf-8 -*-
from __future__ import annotations

"""
General JSON / pickle IO helpers.

This module is deliberately bpy-free and mathutils-free.  It provides a safe
JSON serialization layer for common experiment-report objects such as:

    - pathlib.Path
    - numpy arrays / scalars
    - dataclasses
    - sets / tuples
    - best-effort mathutils-like objects via duck typing

Existing code may continue importing IO helpers from ogd_lbs.utils.common;
common.py re-exports these functions for compatibility.
"""

import dataclasses
import json
import math
import pickle
from pathlib import Path
from typing import Any, Mapping


try:
    import orjson  # type: ignore

    HAS_ORJSON = True
except ImportError:
    orjson = None  # type: ignore
    HAS_ORJSON = False


try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    HAS_NUMPY = False


def _is_numpy_array(obj: Any) -> bool:
    return HAS_NUMPY and isinstance(obj, np.ndarray)


def _is_numpy_scalar(obj: Any) -> bool:
    return HAS_NUMPY and isinstance(obj, np.generic)


def _json_safe_key(key: Any) -> str:
    key_safe = json_safe(key)

    if isinstance(key_safe, str):
        return key_safe

    if key_safe is None or isinstance(key_safe, (int, float, bool)):
        return str(key_safe)

    return str(key_safe)


def _looks_like_mathutils_object(obj: Any) -> bool:
    module = getattr(type(obj), "__module__", "")
    return module.startswith("mathutils")


def json_safe(obj: Any) -> Any:
    """
    Recursively convert common non-JSON objects into JSON-serializable values.

    This function intentionally avoids importing Blender or mathutils.  For
    mathutils.Matrix / Vector-like values, it uses best-effort duck typing.
    """
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, float):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if _is_numpy_array(obj):
        return obj.tolist()

    if _is_numpy_scalar(obj):
        return obj.item()

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(dataclasses.asdict(obj))

    if isinstance(obj, Mapping):
        return {_json_safe_key(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]

    if isinstance(obj, set):
        return sorted((json_safe(x) for x in obj), key=lambda x: str(x))

    # Common non-Blender duck-typing hooks.
    to_list = getattr(obj, "to_list", None)
    if callable(to_list):
        try:
            return json_safe(to_list())
        except Exception:
            pass

    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return json_safe(tolist())
        except Exception:
            pass

    # Best-effort support for mathutils.Matrix / Vector without importing it.
    if _looks_like_mathutils_object(obj):
        try:
            return json_safe([list(row) for row in obj])
        except Exception:
            try:
                return json_safe(list(obj))
            except Exception:
                pass

    return obj


def _assert_no_nonfinite_numbers(obj: Any, *, path: str = "$") -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float is not allowed in JSON at {path}: {obj!r}")
        return

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_nonfinite_numbers(item, path=f"{path}[{i}]")
        return

    if isinstance(obj, tuple):
        for i, item in enumerate(obj):
            _assert_no_nonfinite_numbers(item, path=f"{path}[{i}]")
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_nonfinite_numbers(v, path=f"{path}.{k}")
        return


def load_json(path: str | Path) -> Any:
    path = Path(path)

    if HAS_ORJSON and orjson is not None:
        with path.open("rb") as f:
            return orjson.loads(f.read())

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(
    path: str | Path,
    obj: Any,
    *,
    safe: bool = True,
    indent: int = 2,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
) -> None:
    """
    Dump JSON with safe conversion.

    Defaults are intentionally strict:
        allow_nan=False

    This catches accidental NaN / Inf in reports instead of silently writing
    non-standard JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json_safe(obj) if safe else obj

    if not allow_nan:
        _assert_no_nonfinite_numbers(payload)

    if HAS_ORJSON and orjson is not None:
        option = 0
        if indent and indent > 0:
            option |= orjson.OPT_INDENT_2
        with path.open("wb") as f:
            f.write(orjson.dumps(payload, option=option))
        return

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=indent,
            ensure_ascii=ensure_ascii,
            allow_nan=allow_nan,
        )


def load_pkl(path: str | Path) -> Any:
    path = Path(path)

    with path.open("rb") as f:
        return pickle.load(f)


def dump_pkl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(path: str | Path) -> Any:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in {".pkl", ".pickle"}:
        return load_pkl(path)

    if ext == ".json":
        return load_json(path)

    raise ValueError(f"Unsupported data format extension: {ext} for file {path}")


__all__ = [
    "HAS_ORJSON",
    "json_safe",
    "load_json",
    "dump_json",
    "load_pkl",
    "dump_pkl",
    "load_data",
]
