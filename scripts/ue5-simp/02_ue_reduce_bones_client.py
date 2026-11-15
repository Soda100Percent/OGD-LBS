#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
UE5-Simp Step 02: Reduce bones via UE5 `/api/reduce_bones` (data1 -> data2).
Aligned with `scripts/ssdr/02_ue_ver_reduce_bones_client.py` and additionally
copies `textures/` from `data1` to `data2` while preserving `face_uvs`,
`face_materials`, and `material_names` in each mesh `.pkl`.
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _env import ensure_repo_paths_in_syspath  # noqa: E402
ensure_repo_paths_in_syspath(__file__)

from paths import resolve_ssdr_stage_dirs  # noqa: E402
from ogd_lbs.utils.common import script_argv  # noqa: E402


def _load_ssdr_ue_step02():
    p = Path(__file__).resolve().parents[1] / "ssdr" / "02_ue_ver_reduce_bones_client.py"
    spec = importlib.util.spec_from_file_location("_ssdr_ue_step02", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    mod02 = _load_ssdr_ue_step02()
    mod02.main()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    args, _ = parser.parse_known_args(script_argv())

    res = resolve_ssdr_stage_dirs(
        step_id="02",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    src_tex = res.input_tree / "textures"
    dst_tex = res.output_tree / "textures"
    if src_tex.is_dir():
        dst_tex.mkdir(parents=True, exist_ok=True)
        for png in src_tex.glob("*.png"):
            shutil.copyfile(png, dst_tex / png.name)


if __name__ == "__main__":
    main()
