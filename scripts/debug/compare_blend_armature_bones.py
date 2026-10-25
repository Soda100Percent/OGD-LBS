#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Compare armature bone sets across multiple .blend files.

This debug utility is useful for comparing a cleaned-full armature against
manual / configured bone-reduced variants.

Armature selection:
    - If --main_armature / --compare_armature is supplied, use that exact name.
    - Otherwise, if the blend has exactly one ARMATURE, use it.
    - Otherwise, if the blend has exactly one non-Armature_Original ARMATURE,
      use that one. This matches Data Prep 04a blends, which usually keep both:
          Armature_Original
          Armature_Cleaned_Full / Armature_Bone_Reduced_*
    - Otherwise, fail fast and ask for explicit armature names.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def ensure_repo_paths_in_syspath() -> Path:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            src_dir = parent / "src"
            src_s = str(src_dir)
            if src_s not in sys.path:
                sys.path.insert(0, src_s)
            return parent

    raise RuntimeError("Could not locate repository root containing pyproject.toml and src/.")


ensure_repo_paths_in_syspath()

import bpy  # noqa: E402

from ogd_lbs.utils.common import script_argv  # noqa: E402
from ogd_lbs.blender.blend_io import open_blend_file  # noqa: E402


def choose_armature(
    *,
    blend_path: Path,
    armatures: list[bpy.types.Object],
    preferred_name: str | None,
    role: str,
) -> bpy.types.Object:
    if preferred_name is not None and str(preferred_name).strip():
        name = str(preferred_name).strip()
        matches = [obj for obj in armatures if obj.name == name]
        if len(matches) != 1:
            raise RuntimeError(
                f"{role}: requested armature {name!r} not found uniquely in {blend_path}. "
                f"Available={[obj.name for obj in armatures]}"
            )
        return matches[0]

    if len(armatures) == 1:
        return armatures[0]

    non_original = [obj for obj in armatures if obj.name != "Armature_Original"]

    if len(non_original) == 1:
        print(
            f"NOTE: {role}: blend contains multiple armatures; "
            f"auto-selected non-Armature_Original: {non_original[0].name}",
            flush=True,
        )
        return non_original[0]

    raise RuntimeError(
        f"{role}: could not auto-select armature in {blend_path}. "
        f"Found {len(armatures)} ARMATURE objects: {[obj.name for obj in armatures]}. "
        "Please pass --main_armature or matching --compare_armature."
    )


def load_armature_bones(
    blend_path: str | Path,
    *,
    armature_name: str | None,
    role: str,
) -> dict[str, Any]:
    path = open_blend_file(blend_path, role=role, load_ui=False)

    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"{role}: no ARMATURE found in blend: {path}")

    armature = choose_armature(
        blend_path=path,
        armatures=armatures,
        preferred_name=armature_name,
        role=role,
    )

    bone_names = [bone.name for bone in armature.data.bones]

    return {
        "blend_path": str(path),
        "armature_name": armature.name,
        "armature_data_name": armature.data.name,
        "bone_count": int(len(bone_names)),
        "bone_names": bone_names,
        "available_armatures": [obj.name for obj in armatures],
    }


def normalize_variant_name(path: Path, explicit_name: str | None) -> str:
    if explicit_name is not None and str(explicit_name).strip():
        name = str(explicit_name).strip()
    else:
        name = path.stem

    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise RuntimeError(f"Invalid variant name: {name!r}")

    if name.endswith(".blend"):
        name = Path(name).stem

    return name


def compare_bones(
    *,
    main_info: dict[str, Any],
    variant_info: dict[str, Any],
    variant_name: str,
    fail_on_extra_bones: bool,
) -> dict[str, Any]:
    main_names = [str(x) for x in main_info["bone_names"]]
    variant_names = [str(x) for x in variant_info["bone_names"]]

    main_set = set(main_names)
    variant_set = set(variant_names)

    removed = [name for name in main_names if name not in variant_set]
    extra = [name for name in variant_names if name not in main_set]
    common = [name for name in main_names if name in variant_set]

    if fail_on_extra_bones and extra:
        raise RuntimeError(
            f"Variant {variant_name!r} contains bones not present in main blend: {extra}"
        )

    return {
        "variant_name": variant_name,
        "variant_blend": variant_info["blend_path"],
        "variant_armature_name": variant_info["armature_name"],
        "variant_bone_count": int(variant_info["bone_count"]),
        "main_bone_count": int(main_info["bone_count"]),
        "common_bone_count": int(len(common)),
        "removed_bone_count": int(len(removed)),
        "extra_bone_count": int(len(extra)),
        "removed_bones_in_main_order": removed,
        "extra_bones_in_variant_order": extra,
    }


def make_extra_reduction_config(
    *,
    main_info: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "ogd_lbs_extra_bone_reduction_config_v1",
        "base": {
            "kind": "cleaned_full_or_reference_blend",
            "blend_path": main_info["blend_path"],
            "armature_name": main_info["armature_name"],
            "bone_count": int(main_info["bone_count"]),
        },
        "variants": {
            item["variant_name"]: {
                "source_blend": item["variant_blend"],
                "source_armature_name": item["variant_armature_name"],
                "delete_bones": list(item["removed_bones_in_main_order"]),
            }
            for item in comparisons
        },
        "notes": [
            "delete_bones are computed as main_blend bones missing from each comparison blend.",
            "Review this config before using it in Prep 04a.",
            "This config does not include bones that exist only in comparison blends.",
        ],
    }


def dump_json(path: str | Path, obj: Any) -> None:
    out = Path(path).expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

    print(f"Saved JSON: {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare armature bones between a main .blend and reduced .blend files."
    )

    parser.add_argument("--main_blend", type=str, required=True)
    parser.add_argument(
        "--main_armature",
        type=str,
        default=None,
        help="Optional exact armature name in --main_blend.",
    )
    parser.add_argument(
        "--compare_blend",
        type=str,
        action="append",
        default=[],
        help="Comparison .blend. Can be repeated.",
    )
    parser.add_argument(
        "compare_blends",
        nargs="*",
        help="Additional comparison .blend files.",
    )
    parser.add_argument(
        "--compare_armature",
        type=str,
        action="append",
        default=[],
        help=(
            "Optional exact armature name for each comparison blend, in order. "
            "Can be repeated. If provided, count must match comparison blend count."
        ),
    )
    parser.add_argument(
        "--variant_name",
        type=str,
        action="append",
        default=[],
        help="Optional variant name for each comparison blend, in order.",
    )
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_config", type=str, default=None)
    parser.add_argument(
        "--fail_on_extra_bones",
        action="store_true",
        help="Fail if a comparison blend has bones not present in the main blend.",
    )

    args = parser.parse_args(script_argv())

    compare_paths = [Path(p).expanduser().resolve(strict=False) for p in args.compare_blend]
    compare_paths.extend(Path(p).expanduser().resolve(strict=False) for p in args.compare_blends)

    if not compare_paths:
        raise RuntimeError("At least one comparison blend is required.")

    if args.variant_name and len(args.variant_name) != len(compare_paths):
        raise RuntimeError(
            f"--variant_name count ({len(args.variant_name)}) must match comparison blend count ({len(compare_paths)})."
        )

    if args.compare_armature and len(args.compare_armature) != len(compare_paths):
        raise RuntimeError(
            f"--compare_armature count ({len(args.compare_armature)}) must match comparison blend count ({len(compare_paths)})."
        )

    main_info = load_armature_bones(
        args.main_blend,
        armature_name=args.main_armature,
        role="main armature comparison blend",
    )

    print("\n" + "=" * 96, flush=True)
    print("Armature bone comparison", flush=True)
    print("=" * 96, flush=True)
    print(f"Main blend    : {main_info['blend_path']}", flush=True)
    print(f"Main armature : {main_info['armature_name']}", flush=True)
    print(f"Main bones    : {main_info['bone_count']}", flush=True)

    comparisons: list[dict[str, Any]] = []

    for idx, compare_path in enumerate(compare_paths):
        explicit_variant_name = args.variant_name[idx] if args.variant_name else None
        variant_name = normalize_variant_name(compare_path, explicit_variant_name)
        compare_armature_name = args.compare_armature[idx] if args.compare_armature else None

        variant_info = load_armature_bones(
            compare_path,
            armature_name=compare_armature_name,
            role=f"comparison blend: {variant_name}",
        )

        item = compare_bones(
            main_info=main_info,
            variant_info=variant_info,
            variant_name=variant_name,
            fail_on_extra_bones=bool(args.fail_on_extra_bones),
        )
        comparisons.append(item)

        print("\n" + "-" * 96, flush=True)
        print(f"Variant       : {variant_name}", flush=True)
        print(f"Blend         : {item['variant_blend']}", flush=True)
        print(f"Armature      : {item['variant_armature_name']}", flush=True)
        print(f"Bones         : {item['variant_bone_count']}", flush=True)
        print(f"Removed count : {item['removed_bone_count']}", flush=True)
        print(f"Extra count   : {item['extra_bone_count']}", flush=True)

        if item["removed_bones_in_main_order"]:
            print("Removed bones:", flush=True)
            for name in item["removed_bones_in_main_order"]:
                print(f"  - {name}", flush=True)
        else:
            print("Removed bones: <none>", flush=True)

        if item["extra_bones_in_variant_order"]:
            print("Extra bones in variant:", flush=True)
            for name in item["extra_bones_in_variant_order"]:
                print(f"  - {name}", flush=True)

    result = {
        "schema": "ogd_lbs_armature_bone_diff_v1",
        "main": main_info,
        "comparisons": comparisons,
        "extra_reduction_config_preview": make_extra_reduction_config(
            main_info=main_info,
            comparisons=comparisons,
        ),
    }

    if args.output_json:
        dump_json(args.output_json, result)

    if args.output_config:
        dump_json(args.output_config, result["extra_reduction_config_preview"])

    print("\n" + "=" * 96, flush=True)
    print("Bone comparison complete.", flush=True)
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
