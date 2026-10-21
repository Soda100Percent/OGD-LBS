# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Data-prep manifest helpers.

This module intentionally lives under scripts/data_prep because the manifest
schema here is tied to the data-prep VDB lineage convention:
    vdb_mapping
    Source_<VDB>
    source_objects
    unmerged_meshes
"""

import json
from pathlib import Path
from typing import Any


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(f"Manifest must be a JSON object: {path}")

    return obj


def dump_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)


def require_vdb_mapping(manifest: dict[str, Any]) -> dict[str, list[str]]:
    mapping = manifest.get("vdb_mapping", None)

    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(
            "manifest.json must contain a non-empty object field: 'vdb_mapping'."
        )

    normalized: dict[str, list[str]] = {}

    for target_name, source_names in mapping.items():
        if not isinstance(target_name, str) or not target_name:
            raise RuntimeError(f"Invalid VDB target name in vdb_mapping: {target_name!r}")

        if not isinstance(source_names, list) or not source_names:
            raise RuntimeError(
                f"vdb_mapping[{target_name!r}] must be a non-empty list of source mesh names."
            )

        normalized[target_name] = [str(name) for name in source_names]

    return normalized


def require_exports_for_mapping(
    manifest: dict[str, Any],
    vdb_mapping: dict[str, list[str]],
) -> dict[str, str]:
    exports = manifest.get("exports", None)

    if not isinstance(exports, dict) or not exports:
        raise RuntimeError("manifest.json must contain a non-empty object field: 'exports'.")

    missing = sorted(set(vdb_mapping.keys()) - set(exports.keys()))
    if missing:
        raise RuntimeError(f"manifest exports missing VDB targets: {missing}")

    normalized: dict[str, str] = {}

    for vdb_name in vdb_mapping.keys():
        value = exports.get(vdb_name)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"manifest exports[{vdb_name!r}] must be a non-empty path string.")
        normalized[vdb_name] = value

    return normalized


def resolve_manifest_reference_path(
    manifest_path: str | Path,
    reference_path: str | Path,
) -> Path:
    """
    Resolve a path stored in manifest.

    Absolute references are used directly.
    Relative references are resolved relative to manifest.json's parent.
    """
    manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
    raw = Path(reference_path).expanduser()

    if raw.is_absolute():
        return raw.resolve(strict=False)

    return (manifest_path.parent / raw).resolve(strict=False)


def source_object_name_for_vdb(vdb_name: str) -> str:
    return f"Source_{vdb_name}"


def get_source_objects_mapping(
    manifest: dict[str, Any],
    vdb_mapping: dict[str, list[str]],
) -> dict[str, str]:
    obj = manifest.get("source_objects", {})

    if obj is None:
        obj = {}

    if not isinstance(obj, dict):
        raise RuntimeError("manifest field 'source_objects' must be an object if present.")

    out: dict[str, str] = {}

    for vdb_name in vdb_mapping.keys():
        value = obj.get(vdb_name, source_object_name_for_vdb(vdb_name))
        out[vdb_name] = str(value)

    return out
