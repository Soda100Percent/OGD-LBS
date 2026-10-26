#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Import a Fitbox / SSDR stage directory into a Blender .blend.

This is the single unified import/check CLI.

It supports:
    --import_mode shape_key
    --import_mode LBS
    --open

Typical examples:

    python scripts/debug/import_fitbox_to_blend.py \
        --input_dir "$SSDR/data3" \
        --mesh main_mesh_VDB \
        --import_mode shape_key \
        --output_blend "$SSDR/data3/import_shape_key.blend" \
        --open

    python scripts/debug/import_fitbox_to_blend.py \
        --input_dir "$SSDR/data3" \
        --mesh main_mesh_VDB \
        --import_mode LBS \
        --output_blend "$SSDR/data3/import_lbs.blend" \
        --open
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def ensure_repo_paths_in_syspath() -> None:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src_dir = parent / "src"
            src_s = str(src_dir)
            if src_s not in sys.path:
                sys.path.insert(0, src_s)
            return

    raise RuntimeError("Could not locate repository root containing pyproject.toml and src/.")


ensure_repo_paths_in_syspath()

from ogd_lbs.utils.common import script_argv  # noqa: E402
from ogd_lbs.fitbox.blender_io import (  # noqa: E402
    import_fitbox_stage_to_blend,
    open_fitbox_import_blend_gui,
)


def resolve_default_output_blend(input_dir: Path, import_mode: str, mesh_name: str | None) -> Path:
    mesh_tag = mesh_name or "ALL"
    mode_tag = str(import_mode).lower()
    return (input_dir / f"import_fitbox_{mesh_tag}_{mode_tag}.blend").resolve(strict=False)



def default_bone_length_from_env() -> float:
    """
    Default LBS import edit-bone display length.

    CLI --bone_length has highest priority. This env var is only a convenience
    for asset-specific visual checks, e.g. Winter Samurai can use shorter bones:

        export OGD_FITBOX_IMPORT_BONE_LENGTH=0.08
    """
    raw = os.environ.get("OGD_FITBOX_IMPORT_BONE_LENGTH", "").strip()

    if not raw:
        return 0.5

    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid OGD_FITBOX_IMPORT_BONE_LENGTH={raw!r}; expected positive float."
        ) from exc

    if value <= 0.0:
        raise RuntimeError(
            f"Invalid OGD_FITBOX_IMPORT_BONE_LENGTH={raw!r}; expected positive float."
        )

    return value

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Fitbox / SSDR stage data into a Blender .blend."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("--mesh", type=str, default=None, help="Mesh name/stem. Use ALL, or omit, to import all mesh files.")
    parser.add_argument(
        "-o",
        "--output_blend",
        type=str,
        default=None,
        help="Output .blend path. Default: <input_dir>/import_fitbox_<mesh>_<mode>.blend",
    )

    parser.add_argument(
        "--import_mode",
        type=str,
        choices=["LBS", "shape_key"],
        default="shape_key",
        help="Import mode. shape_key uses baked vertices; LBS keyframes armature poses.",
    )

    parser.add_argument("--object_name", type=str, default=None)
    parser.add_argument("--armature_name", type=str, default="Armature")
    parser.add_argument("--weight_eps", type=float, default=1e-8)

    parser.add_argument(
        "--max_shape_keys",
        type=int,
        default=-1,
        help="For shape_key mode. <=0 imports all baked frames.",
    )
    parser.add_argument(
        "--no_shape_key_animation",
        action="store_true",
        help="Create shape keys but do not keyframe their values.",
    )

    parser.add_argument(
        "--max_lbs_frames",
        type=int,
        default=-1,
        help="For LBS mode. <=0 imports all animation frames.",
    )

    parser.add_argument("--no_armature_modifier", action="store_true")
    parser.add_argument("--no_debug_scene", action="store_true")
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument(
        "--use_scene_frames",
        action="store_true",
        help=(
            "Use stored scene_frames for imported keyframes. Disabled by default "
            "to match tested Blender plugin import behavior, which uses frame_idx + 1."
        ),
    )
    parser.add_argument(
        "--bone_length",
        type=float,
        default=default_bone_length_from_env(),
        help=(
            "LBS mode edit-bone display length. "
            "Default: $OGD_FITBOX_IMPORT_BONE_LENGTH if set, otherwise 0.5."
        ),
    )
    parser.add_argument(
        "--matrix_space",
        type=str,
        choices=["POSE", "WORLD"],
        default="POSE",
        help=(
            "LBS mode matrix interpretation. POSE matches the current plugin-aligned importer. "
            "WORLD converts animation matrices from WORLD to POSE space, useful for legacy data checks."
        ),
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help="Open saved .blend with Blender GUI after import.",
    )
    parser.add_argument(
        "--blender_bin",
        type=str,
        default=None,
        help="Blender executable for --open. Default uses $BLENDER_BIN or /workspace/blender-4.5.12-linux-x64/blender.",
    )

    args = parser.parse_args(script_argv())

    input_dir = Path(args.input_dir).expanduser().resolve(strict=False)

    if args.output_blend:
        output_blend = Path(args.output_blend).expanduser().resolve(strict=False)
    else:
        output_blend = resolve_default_output_blend(
            input_dir,
            import_mode=args.import_mode,
            mesh_name=args.mesh,
        )

    result = import_fitbox_stage_to_blend(
        input_dir=input_dir,
        output_blend=output_blend,
        mesh_name=args.mesh,
        import_mode=args.import_mode,
        object_name=args.object_name,
        armature_name=args.armature_name,
        weight_eps=float(args.weight_eps),
        max_shape_keys=int(args.max_shape_keys),
        animate_shape_keys=not bool(args.no_shape_key_animation),
        max_lbs_frames=int(args.max_lbs_frames),
        add_armature_modifier_enabled=not bool(args.no_armature_modifier),
        setup_debug_scene=not bool(args.no_debug_scene),
        write_metadata=not bool(args.no_metadata),
        clear_existing_scene=True,
        use_scene_frames=bool(args.use_scene_frames),
        bone_length=float(args.bone_length),
        matrix_space=str(args.matrix_space),
    )

    print("\nImport summary:")
    for key in [
        "import_mode",
        "use_scene_frames",
        "matrix_space",
        "bone_length",
        "mesh_name",
        "vertex_count",
        "face_count",
        "bone_count",
        "frame_count",
        "shape_key_count",
        "lbs_keyed_frame_count",
        "output_blend",
    ]:
        print(f"  {key:<23}: {result.get(key)}")

    if args.open:
        open_fitbox_import_blend_gui(result["output_blend"], args.blender_bin)


if __name__ == "__main__":
    main()
