# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Shared helpers for experiment runners.

No algorithm logic lives here; this is runner infrastructure:
    - JSON loading/writing;
    - slug creation;
    - command rendering/execution;
    - basic config normalization.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve(strict=False)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj: Any) -> Path:
    p = Path(path).expanduser().resolve(strict=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=4, ensure_ascii=False)
        f.write("\n")
    return p


def json_safe(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]

    if isinstance(obj, set):
        return sorted((json_safe(x) for x in obj), key=lambda x: str(x))

    return obj


def slugify(value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "none"
    else:
        text = str(value)

    text = text.strip()
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    text = text.replace("+", "p")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


def command_to_string(cmd: list[str]) -> str:
    import shlex
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_logged_command(
    cmd: list[str],
    *,
    log_path: Path,
    cwd: Path,
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rendered = command_to_string(cmd)
    print("\n" + "=" * 96, flush=True)
    print(f"[cmd] {rendered}", flush=True)
    print(f"[log] {log_path}", flush=True)
    print("=" * 96, flush=True)

    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write("\n" + "=" * 96 + "\n")
        log_f.write(rendered + "\n")
        log_f.write("=" * 96 + "\n")

    if dry_run:
        print("[dry-run] command not executed.", flush=True)
        return

    with log_path.open("a", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)

        status = proc.wait()

    if status != 0:
        raise RuntimeError(f"Command failed with status {status}: {rendered}")


def python_cmd(
    *,
    conda_env: str | None,
    no_conda: bool,
) -> list[str]:
    if no_conda:
        return [sys.executable]

    if conda_env is None or not str(conda_env).strip():
        return [sys.executable]

    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(conda_env),
        "python",
    ]


def append_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return

    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return

    cmd.extend([flag, str(value)])


def append_store_true(cmd: list[str], flag: str, value: Any) -> None:
    if bool(value):
        cmd.append(flag)


def copy_tree_contents(src: Path, dst: Path, *, dry_run: bool) -> None:
    src = Path(src).expanduser().resolve(strict=False)
    dst = Path(dst).expanduser().resolve(strict=False)

    if dry_run:
        print(f"[dry-run] copy tree contents: {src} -> {dst}", flush=True)
        return

    if not src.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src}")

    if dst.exists():
        shutil_rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            import shutil
            shutil.copytree(item, target)
        else:
            import shutil
            shutil.copy2(item, target)


def shutil_rmtree(path: Path) -> None:
    import shutil
    if path.exists():
        shutil.rmtree(path)
