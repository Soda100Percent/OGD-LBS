"""Stage file integrity helpers; no pickle loading or import-time operations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from stage_contracts import StageContractError, require, absolute_path

INPUT_SCHEMA = "ogd_stage_input_files_v1"


def safe_path(value):
    path = Path(value).expanduser().absolute()
    absolute_path(str(path), "filesystem path")
    for part in [path, *path.parents]:
        require(not part.is_symlink(), f"Symlink path rejected: {part}")
    return path


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def bad_constant(value):
        raise StageContractError(f"Nonfinite JSON constant: {value}")

    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=bad_constant,
    )


def encoded(obj):
    return (
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def object_hash(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def bytes_hash(data):
    return hashlib.sha256(data).hexdigest()


def read_regular(path):
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), f"Not a regular file: {path}")
        data = stream.read()
        after = os.fstat(stream.fileno())
    current = path.lstat()

    def identity(value):
        return (
            value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
        )

    require(identity(before) == identity(after) == identity(current),
            f"File changed during read: {path}")
    return data


def fingerprint(path):
    """Streaming hashing avoids loading an entire payload merely to hash it."""
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), f"Nonregular file: {path}")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = path.lstat()

    def identity(value):
        return (
            value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
        )

    require(identity(before) == identity(after) == identity(current),
            f"File changed during hashing: {path}")
    return {"size": after.st_size, "sha256": digest.hexdigest()}


def tree_manifest(root):
    root = safe_path(root)
    require(root.is_dir(), f"Missing input/output directory: {root}")
    files, directories = {}, []
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        names.sort()
        for name in dirs:
            path = Path(directory) / name
            require(stat.S_ISDIR(path.lstat().st_mode),
                    f"Symlink/special directory rejected: {path}")
            directories.append(path.relative_to(root).as_posix())
        for name in names:
            path = Path(directory) / name
            files[path.relative_to(root).as_posix()] = fingerprint(path)
    return {
        "files": dict(sorted(files.items())),
        "directories": sorted(directories),
    }


def validate_input_manifest(manifest, binding):
    require(isinstance(manifest, dict)
            and set(manifest) == {
                "schema", "artifact_id", "artifact_type", "files", "directories"
            }, "Unsupported input manifest fields")
    require(manifest["schema"] == INPUT_SCHEMA, "Unsupported input manifest schema")
    require(manifest["artifact_id"] == binding["artifact_id"]
            and manifest["artifact_type"] == "ssdr_data1",
            "Input manifest artifact identity mismatch")
    require(object_hash(manifest) == binding["payload_manifest_sha256"],
            "Input manifest differs from frozen request")
    expected = {"files": manifest["files"], "directories": manifest["directories"]}
    require(tree_manifest(binding["payload_path"]) == expected,
            "Input payload inventory/hash differs from manifest")
    return expected


def verified_bytes(root, relative, manifest):
    """Read exactly the bytes checked against the manifest before pickle.loads."""
    path = safe_path(Path(root) / relative)
    require(path.is_relative_to(safe_path(root)), "File escaped input tree")
    require(relative in manifest["files"], f"Input file is not declared: {relative}")
    data = read_regular(path)
    expected = manifest["files"][relative]
    require(expected == {"size": len(data), "sha256": bytes_hash(data)},
            f"Input file changed before loading: {relative}")
    return data


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, data, *, replace=False):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".stage-write-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp, 0o600)
        if replace:
            os.replace(temp, path)
        else:
            os.link(temp, path)
            temp.unlink()
        sync_dir(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_json(path, obj, *, replace=False):
    atomic_bytes(path, encoded(obj), replace=replace)
