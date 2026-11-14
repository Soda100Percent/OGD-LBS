# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import sys


def ensure_repo_paths_in_syspath(current_file: str | Path | None = None) -> Path:
    cur = Path(current_file or __file__).expanduser().resolve(strict=False)
    for parent in [cur.parent, *cur.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            src_dir = str(parent / "src")
            script_dir = str(parent / "scripts" / "diff_render")
            for p in (src_dir, script_dir):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return parent
    raise RuntimeError("Could not locate OGD-LBS repository root containing pyproject.toml and src/.")


REPO_ROOT = ensure_repo_paths_in_syspath()
