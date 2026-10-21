# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Work directory and file-resolution utilities.

Design goals:
    - Pure Python standard library. No bpy dependency.
    - Explicitly distinguish:
        EXPLICIT_OVERRIDE: user provided a CLI path.
        WORK_DIR_DEFAULT: path derived from the primary input's work directory.
        MANIFEST_REFERENCE: path referenced by a manifest file.
    - No fallback search.
    - Fail fast for missing required inputs.
    - Print a clear file-resolution table before doing destructive/heavy work.

This module is intentionally generic. Data-prep-specific names and suffixes
belong in ogd_lbs.data_prep.paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence
import os
import sys


PathKind = Literal["INPUT", "OUTPUT", "INFO"]
PathSource = Literal["EXPLICIT_OVERRIDE", "WORK_DIR_DEFAULT", "MANIFEST_REFERENCE", "DERIVED"]
ExpectedPathType = Literal["file", "dir", "any"]


@dataclass
class ResolvedPathRecord:
    kind: PathKind
    name: str
    path: Path
    source: str
    must_exist: bool
    expected_type: ExpectedPathType
    note: str = ""

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def type_ok_if_exists(self) -> bool:
        if not self.path.exists():
            return True

        if self.expected_type == "file":
            return self.path.is_file()

        if self.expected_type == "dir":
            return self.path.is_dir()

        return True

    def expected_type_text(self) -> str:
        return self.expected_type


def resolve_path_no_strict(path: str | Path) -> Path:
    """
    Resolve a path without requiring it to exist.
    """
    return Path(path).expanduser().resolve(strict=False)


def path_is_probably_file(path: Path) -> bool:
    """
    Heuristic used only for choosing work_dir from primary_input.

    For actual validation, WorkDirContext uses explicit expected_type.
    """
    return bool(path.suffix)


def extract_base_stem(path_or_name: str | Path, strip_suffixes: Sequence[str]) -> str:
    """
    Extract an asset base stem by removing one known stage suffix from the end.

    Suffixes are matched only at the end of the stem. The longest suffix should
    appear first in strip_suffixes; this function also sorts defensively.
    """
    stem = Path(path_or_name).stem

    for suffix in sorted(strip_suffixes, key=len, reverse=True):
        if suffix and stem.endswith(suffix):
            return stem[: -len(suffix)]

    return stem


def ensure_repo_src_in_syspath(
    *,
    current_file: str | Path,
    repo_root_markers: Sequence[str] = ("pyproject.toml", ".git"),
    src_dir_name: str = "src",
) -> Path | None:
    """
    Add <repo_root>/src to sys.path before importing ogd_lbs.

    This helper is useful for standalone CLI scripts, especially in the
    Blender 5.x bpy environment where the project may not have been installed
    with `pip install -e .`.

    Because this function lives inside the package, it is mostly useful for
    scripts that copy the same logic inline before package imports. It remains
    here for completeness and tests.
    """
    cur = Path(current_file).expanduser().resolve(strict=False)

    for parent in [cur.parent, *cur.parents]:
        if any((parent / marker).exists() for marker in repo_root_markers):
            src = parent / src_dir_name
            if src.exists():
                src_str = str(src)
                if src_str not in sys.path:
                    sys.path.insert(0, src_str)
                return src

    return None


class WorkDirContext:
    """
    Resolve stage inputs and outputs relative to a primary input.

    Typical usage:
        ctx = WorkDirContext(args.input_blend, strip_suffixes=DATA_PREP_SUFFIXES)
        input_blend = ctx.resolve_input("input_blend", args.input_blend, None)
        manifest = ctx.resolve_input("manifest", args.manifest, ctx.rel(".../manifest.json"))
        output = ctx.resolve_output("output_blend", args.output_blend, ctx.rel("...blend"))
        ctx.print_and_validate("Prep XX")
    """

    def __init__(
        self,
        primary_input: str | Path,
        *,
        strip_suffixes: Sequence[str] = (),
        primary_expected_type: ExpectedPathType = "file",
    ) -> None:
        self.primary_path = resolve_path_no_strict(primary_input)
        self.primary_expected_type = primary_expected_type

        if primary_expected_type == "dir":
            self.work_dir = self.primary_path
        elif primary_expected_type == "file":
            self.work_dir = self.primary_path.parent
        else:
            self.work_dir = self.primary_path.parent if path_is_probably_file(self.primary_path) else self.primary_path

        self.base_stem = extract_base_stem(self.primary_path, strip_suffixes)
        self.records: list[ResolvedPathRecord] = []

    def rel(self, relative_path: str | Path) -> Path:
        return self.work_dir / Path(relative_path)

    def _resolve(
        self,
        *,
        kind: PathKind,
        name: str,
        explicit_path: str | Path | None,
        default_path: str | Path | None,
        must_exist: bool,
        expected_type: ExpectedPathType,
        note: str,
    ) -> Path:
        if explicit_path is not None and str(explicit_path) != "":
            path = resolve_path_no_strict(explicit_path)
            source: PathSource = "EXPLICIT_OVERRIDE"
        else:
            if default_path is None:
                raise ValueError(
                    f"No explicit path and no default path were provided for {kind.lower()} '{name}'."
                )
            path = resolve_path_no_strict(self.work_dir / Path(default_path))
            source = "WORK_DIR_DEFAULT"

        self.records.append(
            ResolvedPathRecord(
                kind=kind,
                name=name,
                path=path,
                source=source,
                must_exist=must_exist,
                expected_type=expected_type,
                note=note,
            )
        )
        return path

    def resolve_input(
        self,
        name: str,
        explicit_path: str | Path | None,
        default_path: str | Path | None = None,
        *,
        must_exist: bool = True,
        expected_type: ExpectedPathType = "file",
        note: str = "",
    ) -> Path:
        return self._resolve(
            kind="INPUT",
            name=name,
            explicit_path=explicit_path,
            default_path=default_path,
            must_exist=must_exist,
            expected_type=expected_type,
            note=note,
        )

    def resolve_output(
        self,
        name: str,
        explicit_path: str | Path | None,
        default_path: str | Path | None = None,
        *,
        expected_type: ExpectedPathType = "file",
        note: str = "",
    ) -> Path:
        return self._resolve(
            kind="OUTPUT",
            name=name,
            explicit_path=explicit_path,
            default_path=default_path,
            must_exist=False,
            expected_type=expected_type,
            note=note,
        )

    def add_manifest_reference_input(
        self,
        name: str,
        path: str | Path,
        *,
        manifest_path: str | Path,
        must_exist: bool = True,
        expected_type: ExpectedPathType = "file",
        note: str = "",
    ) -> Path:
        """
        Register an input referenced by a manifest.

        Absolute paths are used directly.
        Relative paths are resolved relative to the manifest file's directory.
        """
        raw = Path(path).expanduser()
        if raw.is_absolute():
            resolved = raw.resolve(strict=False)
        else:
            resolved = (Path(manifest_path).expanduser().resolve(strict=False).parent / raw).resolve(strict=False)

        self.records.append(
            ResolvedPathRecord(
                kind="INPUT",
                name=name,
                path=resolved,
                source="MANIFEST_REFERENCE",
                must_exist=must_exist,
                expected_type=expected_type,
                note=note,
            )
        )
        return resolved

    def add_info_path(
        self,
        name: str,
        path: str | Path,
        *,
        source: str = "DERIVED",
        expected_type: ExpectedPathType = "any",
        note: str = "",
    ) -> Path:
        resolved = resolve_path_no_strict(path)
        self.records.append(
            ResolvedPathRecord(
                kind="INFO",
                name=name,
                path=resolved,
                source=source,
                must_exist=False,
                expected_type=expected_type,
                note=note,
            )
        )
        return resolved

    def validate(
        self,
        *,
        forbid_output_overwrite_inputs: bool = True,
    ) -> None:
        errors: list[str] = []

        for rec in self.records:
            if rec.kind == "INPUT" and rec.must_exist and not rec.path.exists():
                errors.append(
                    f"Missing required input '{rec.name}': {rec.path} "
                    f"(source={rec.source}, expected_type={rec.expected_type})"
                )
                continue

            if rec.path.exists() and not rec.type_ok_if_exists():
                errors.append(
                    f"Path type mismatch for '{rec.name}': {rec.path} "
                    f"(source={rec.source}, expected_type={rec.expected_type})"
                )

        if forbid_output_overwrite_inputs:
            input_paths = {
                rec.path
                for rec in self.records
                if rec.kind == "INPUT" and rec.path.exists()
            }

            for rec in self.records:
                if rec.kind != "OUTPUT":
                    continue
                if rec.expected_type == "file" and rec.path in input_paths:
                    errors.append(
                        f"Output '{rec.name}' would overwrite an input file: {rec.path}"
                    )

        if errors:
            msg = "\n".join(f"  - {err}" for err in errors)
            raise FileNotFoundError("WorkDir validation failed:\n" + msg)

    def ensure_output_parents(self) -> None:
        """
        Create parent directories for output files and output directories.
        Call this after validate().
        """
        for rec in self.records:
            if rec.kind != "OUTPUT":
                continue

            if rec.expected_type == "dir":
                rec.path.mkdir(parents=True, exist_ok=True)
            else:
                rec.path.parent.mkdir(parents=True, exist_ok=True)

    def print_resolution_info(self, stage_title: str) -> None:
        line = "=" * 96
        subline = "-" * 96

        print(line, flush=True)
        print(f"📂 [{stage_title}] WorkDir Resolution Info", flush=True)
        print(f"   work_dir     : {self.work_dir}", flush=True)
        print(f"   base_stem    : {self.base_stem}", flush=True)
        print(subline, flush=True)

        if not self.records:
            print("   <no paths registered>", flush=True)
            print(line, flush=True)
            return

        max_name_len = max(len(rec.name) for rec in self.records)
        max_kind_len = max(len(rec.kind) for rec in self.records)

        for rec in self.records:
            exists_text = f"exists={rec.path.exists()}"
            type_text = f"type={rec.expected_type_text()}"
            note_text = f" | {rec.note}" if rec.note else ""
            print(
                f"   [{rec.kind:<{max_kind_len}}] "
                f"{rec.name:<{max_name_len}} : {rec.path} "
                f"({rec.source} | {exists_text} | {type_text}){note_text}",
                flush=True,
            )

        print(line, flush=True)

    def print_and_validate(
        self,
        stage_title: str,
        *,
        create_output_parents: bool = True,
        forbid_output_overwrite_inputs: bool = True,
    ) -> None:
        self.print_resolution_info(stage_title)
        self.validate(forbid_output_overwrite_inputs=forbid_output_overwrite_inputs)
        if create_output_parents:
            self.ensure_output_parents()


def parse_idx_ranges(text: str) -> set[tuple[int, int]]:
    """
    Generic parser for left-closed right-open index ranges.

    Example:
        "20:60,260:340" -> {(20, 60), (260, 340)}
    """
    if not text:
        return set()

    ranges: set[tuple[int, int]] = set()

    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue

        if ":" not in item:
            raise ValueError(f"Invalid range item '{item}'. Expected format start:end.")

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
