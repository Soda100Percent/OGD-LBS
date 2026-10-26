#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Direct SSDR / Fitbox result viewer independent of Blender.

This tool loads a Fitbox / SSDR stage directory, reconstructs LBS animation
directly with NumPy using:

    rest vertices
    rest_pose
    animation_poses
    vertex_bone_weights

and visualizes:

    reconstructed SSDR mesh
    baked_animation_vertices target mesh
    skeleton joints from animation_poses

Typical use:

    python scripts/debug/ssdr_res_render.py \
        --input_dir "$SSDR/data4/opt_out" \
        --mesh main_mesh_VDB \
        --heatmap

For headless screenshot:

    xvfb-run -a python scripts/debug/ssdr_res_render.py \
        --input_dir "$SSDR/data4/opt_out" \
        --mesh main_mesh_VDB \
        --frame 0 \
        --off_screen \
        --screenshot /tmp/ssdr_frame0.png
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def ensure_repo_src_in_syspath() -> None:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src = parent / "src"
            src_s = str(src)
            if src_s not in sys.path:
                sys.path.insert(0, src_s)
            return

    raise RuntimeError("Could not locate repository root containing pyproject.toml and src/.")


ensure_repo_src_in_syspath()

from ogd_lbs.utils.common import (  # noqa: E402
    Timer,
    discover_and_resolve_files,
    fail,
    load_data,
    log,
    resolve_specific_file,
    script_argv,
    warn,
)
from ogd_lbs.fitbox.schema import (  # noqa: E402
    KEY_ANIMATION_POSES,
    KEY_BAKED_VERTICES,
    KEY_BONE_NAMES,
    KEY_FACES,
    KEY_REST_POSE,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTICES,
)
from ogd_lbs.ssdr.skinning import skinning_s, skinning_t, skinning_w  # noqa: E402


ANIMATION_STEM = "animation_data"
SKELETON_STEM = "skeleton_hierarchy_data"

EXCLUDED_STEMS = {
    ANIMATION_STEM,
    SKELETON_STEM,
    "step00_convert_summary",
    "step01_frame_sampling_summary",
    "step02_reduce_bone_weight_summary",
    "step03_qem_summary",
    "step04_optimize_summary",
    "metrics_history_per_frame",
}


try:
    import pyvista as pv  # type: ignore

    HAS_PYVISTA = True
except Exception:
    pv = None
    HAS_PYVISTA = False


def require_pyvista() -> Any:
    if not HAS_PYVISTA or pv is None:
        raise RuntimeError(
            "PyVista is not available in the current environment.\n"
            "Install it if you want interactive rendering, for example:\n"
            "    pip install pyvista\n"
            "Stats-only mode does not require PyVista:\n"
            "    python scripts/debug/ssdr_res_render.py --input_dir ... --stats_only"
        )
    return pv


def normalize_parent_indices(parent_indices: Any, bone_count: int) -> list[int | None]:
    raw = list(parent_indices)

    if len(raw) != int(bone_count):
        fail(f"parent_indices length mismatch: {len(raw)} vs bone_count={bone_count}")

    out: list[int | None] = []

    for value in raw:
        if value is None:
            out.append(None)
            continue

        idx = int(value)
        if idx < 0 or idx >= int(bone_count):
            out.append(None)
        else:
            out.append(idx)

    return out


def stats_line(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        print(f"  {name:<32} <empty>")
        return

    print(
        f"  {name:<32} "
        f"mean={float(np.mean(arr)):.8g} "
        f"p50={float(np.percentile(arr, 50)):.8g} "
        f"p95={float(np.percentile(arr, 95)):.8g} "
        f"p99={float(np.percentile(arr, 99)):.8g} "
        f"max={float(np.max(arr)):.8g}"
    )


def to_pv_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)

    if faces.ndim != 2 or faces.shape[1] < 3:
        fail(f"faces must have shape (F, K), K>=3, got {faces.shape}")

    prefix = np.full((faces.shape[0], 1), int(faces.shape[1]), dtype=np.int64)
    return np.hstack([prefix, faces]).reshape(-1)


def get_reconstructed_sequence(
    vertices: np.ndarray,
    bt: np.ndarray,
    inv_rbt: np.ndarray,
    vbw: np.ndarray,
) -> np.ndarray:
    n_p = int(bt.shape[0])
    n_v = int(vertices.shape[0])
    out = np.zeros((n_p, n_v, 3), dtype=np.float64)

    for p_id in range(n_p):
        constructed = skinning_t(vertices, bt, inv_rbt, p_id)
        constructed = skinning_w(constructed, vbw)
        out[p_id] = skinning_s(constructed)

    return out


def get_skeleton_lines(parent_indices: list[int | None]) -> np.ndarray:
    lines: list[int] = []

    for idx, parent in enumerate(parent_indices):
        if parent is None:
            continue
        lines.extend([2, int(parent), int(idx)])

    return np.asarray(lines, dtype=np.int64)


def mesh_name_means_all(mesh_name: str | None) -> bool:
    if mesh_name is None:
        return True

    text = str(mesh_name).strip().lower()
    return text in {"", "all", "*", "__all__", "auto", "auto_mesh"}


def discover_all_mesh_paths(input_dir: Path) -> list[Path]:
    ext_priority = [".pkl", ".json"]
    mesh_paths = discover_and_resolve_files(
        input_dir,
        ext_priority,
        exclude_stems=EXCLUDED_STEMS,
    )

    if not mesh_paths:
        fail(f"No mesh pkl/json files found under: {input_dir}")

    return mesh_paths


def discover_mesh_paths(input_dir: Path, mesh_name: str | None) -> list[Path]:
    ext_priority = [".pkl", ".json"]

    if mesh_name_means_all(mesh_name):
        return discover_all_mesh_paths(input_dir)

    assert mesh_name is not None
    path = resolve_specific_file(input_dir, mesh_name, ext_priority)
    if path is not None:
        return [path]

    # Backward-compatibility for old multi-component SSDR outputs:
    # New OGD-LBS examples often use main_mesh_VDB, while old datasets usually
    # contain many component meshes and no main_mesh_VDB.pkl.
    candidates = discover_all_mesh_paths(input_dir)
    candidate_names = [p.stem for p in candidates]

    if str(mesh_name) == "main_mesh_VDB" and len(candidates) > 1:
        warn(
            "Requested mesh 'main_mesh_VDB' was not found, but this directory "
            "contains multiple mesh files. Falling back to ALL meshes for "
            "legacy multi-component SSDR compatibility. "
            f"Candidates={candidate_names}"
        )
        return candidates

    fail(
        f"Mesh file not found for mesh={mesh_name!r} under {input_dir}. "
        f"Available mesh stems: {candidate_names}. "
        "Use --mesh ALL or omit --mesh to load all meshes."
    )
    raise RuntimeError("unreachable")


def load_stage(input_dir: Path, mesh_name: str | None) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve(strict=False)

    if not input_dir.is_dir():
        fail(f"Input directory does not exist or is not a directory: {input_dir}")

    ext_priority = [".pkl", ".json"]
    anim_path = resolve_specific_file(input_dir, ANIMATION_STEM, ext_priority)
    skel_path = resolve_specific_file(input_dir, SKELETON_STEM, ext_priority)

    if anim_path is None:
        fail(f"Missing {ANIMATION_STEM}.pkl/json under {input_dir}")
    if skel_path is None:
        fail(f"Missing {SKELETON_STEM}.pkl/json under {input_dir}")

    anim_obj = load_data(anim_path)
    skel_obj = load_data(skel_path)

    if not isinstance(anim_obj, dict):
        fail(f"animation_data must be dict: {anim_path}")
    if not isinstance(skel_obj, dict):
        fail(f"skeleton_hierarchy_data must be dict: {skel_path}")

    bone_names = [str(x) for x in skel_obj.get(KEY_BONE_NAMES, anim_obj.get(KEY_BONE_NAMES, []))]
    if not bone_names:
        fail("Missing non-empty bone_names in skeleton/animation data.")

    parent_indices = normalize_parent_indices(
        skel_obj.get("parent_indices", []),
        bone_count=len(bone_names),
    )

    rbt = np.asarray(anim_obj[KEY_REST_POSE], dtype=np.float64)
    bt = np.asarray(anim_obj[KEY_ANIMATION_POSES], dtype=np.float64)

    if rbt.shape != (len(bone_names), 4, 4):
        fail(f"rest_pose shape mismatch: expected ({len(bone_names)}, 4, 4), got {rbt.shape}")

    if bt.ndim != 4 or bt.shape[1:] != (len(bone_names), 4, 4):
        fail(f"animation_poses shape mismatch: expected (P, {len(bone_names)}, 4, 4), got {bt.shape}")

    inv_rbt = np.linalg.inv(rbt)

    mesh_paths = discover_mesh_paths(input_dir, mesh_name)

    all_vertices: list[np.ndarray] = []
    all_baked: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    mesh_summaries: list[dict[str, Any]] = []

    vertex_offset = 0

    for mesh_path in mesh_paths:
        mesh_obj = load_data(mesh_path)
        if not isinstance(mesh_obj, dict):
            warn(f"Skipping non-dict mesh object: {mesh_path}")
            continue

        name = str(mesh_obj.get("name", mesh_path.stem))

        v = np.asarray(mesh_obj[KEY_VERTICES], dtype=np.float64)
        f = np.asarray(mesh_obj[KEY_FACES], dtype=np.int64)
        baked = np.asarray(mesh_obj[KEY_BAKED_VERTICES], dtype=np.float64)
        vbw = np.asarray(mesh_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)

        if v.ndim != 2 or v.shape[1] != 3:
            fail(f"[{name}] vertices expected (V,3), got {v.shape}")
        if f.ndim != 2 or f.shape[1] < 3:
            fail(f"[{name}] faces expected (F,K), K>=3, got {f.shape}")
        if baked.ndim != 3 or baked.shape[1:] != v.shape:
            fail(f"[{name}] baked_animation_vertices expected (P,{v.shape[0]},3), got {baked.shape}")
        if baked.shape[0] != bt.shape[0]:
            fail(f"[{name}] baked frame count {baked.shape[0]} != animation frame count {bt.shape[0]}")
        if vbw.ndim != 2 or vbw.shape != (v.shape[0], len(bone_names)):
            fail(f"[{name}] vertex_bone_weights expected ({v.shape[0]},{len(bone_names)}), got {vbw.shape}")

        all_vertices.append(v)
        all_baked.append(baked)
        all_weights.append(vbw)
        all_faces.append(f + vertex_offset)

        mesh_summaries.append(
            {
                "name": name,
                "path": str(mesh_path),
                "vertices": int(v.shape[0]),
                "faces": int(f.shape[0]),
            }
        )

        vertex_offset += int(v.shape[0])

    if not all_vertices:
        fail(f"No valid mesh data found under: {input_dir}")

    vertices_cb = np.concatenate(all_vertices, axis=0)
    baked_cb = np.concatenate(all_baked, axis=1)
    weights_cb = np.concatenate(all_weights, axis=0)
    faces_cb = np.concatenate(all_faces, axis=0)

    with Timer("direct_renderer_lbs_reconstruction"):
        recon_cb = get_reconstructed_sequence(
            vertices=vertices_cb,
            bt=bt,
            inv_rbt=inv_rbt,
            vbw=weights_cb,
        )

    l2_error = np.linalg.norm(recon_cb - baked_cb, axis=2)

    scene_frames = np.asarray(anim_obj.get("scene_frames", []), dtype=np.int64)

    return {
        "input_dir": input_dir,
        "animation_path": anim_path,
        "skeleton_path": skel_path,
        "mesh_paths": mesh_paths,
        "mesh_summaries": mesh_summaries,
        "bone_names": bone_names,
        "parent_indices": parent_indices,
        "rbt": rbt,
        "bt": bt,
        "inv_rbt": inv_rbt,
        "vertices": vertices_cb,
        "faces": faces_cb,
        "baked": baked_cb,
        "weights": weights_cb,
        "recon": recon_cb,
        "l2_error": l2_error,
        "scene_frames": scene_frames,
    }


def print_stage_stats(stage: dict[str, Any]) -> None:
    vertices = stage["vertices"]
    baked = stage["baked"]
    recon = stage["recon"]
    weights = stage["weights"]
    faces = stage["faces"]
    l2_error = stage["l2_error"]
    bt = stage["bt"]

    print("\n" + "=" * 96)
    print("Direct SSDR / Fitbox stage stats")
    print("=" * 96)
    print(f"input_dir      : {stage['input_dir']}")
    print(f"animation      : {stage['animation_path']}")
    print(f"skeleton       : {stage['skeleton_path']}")
    print(f"meshes         : {len(stage['mesh_summaries'])}")
    for item in stage["mesh_summaries"]:
        print(f"  - {item['name']}: V={item['vertices']}, F={item['faces']}, path={item['path']}")

    print(f"frames         : {bt.shape[0]}")
    print(f"bones          : {bt.shape[1]}")
    print(f"vertices total : {vertices.shape[0]}")
    print(f"faces total    : {faces.shape[0]}")
    print(f"baked shape    : {baked.shape}")
    print(f"recon shape    : {recon.shape}")
    print(f"weights shape  : {weights.shape}")

    if stage["scene_frames"].size > 0:
        sf = stage["scene_frames"]
        print(f"scene_frames   : shape={sf.shape}, first={int(sf[0])}, last={int(sf[-1])}")
    else:
        print("scene_frames   : <missing>")

    print("\n[Weight checks]")
    row_sums = weights.sum(axis=1)
    print(f"  finite weights              : {bool(np.isfinite(weights).all())}")
    print(f"  weight min/max              : {float(weights.min()):.8g} / {float(weights.max()):.8g}")
    print(f"  row sum min/max             : {float(row_sums.min()):.8g} / {float(row_sums.max()):.8g}")
    print(f"  bad row sums > 1e-5         : {int(np.count_nonzero(np.abs(row_sums - 1.0) > 1e-5))}")

    print("\n[Target motion]")
    disp = np.linalg.norm(baked - vertices[None, :, :], axis=2)
    stats_line("||baked[p,i]-rest[i]||", disp)

    print("\n[Direct NumPy LBS recon vs baked target]")
    stats_line("per-vertex L2 error", l2_error)
    frame_mean = l2_error.mean(axis=1)
    frame_max = l2_error.max(axis=1)
    print(f"  worst frame by mean error    : {int(np.argmax(frame_mean))}, mean={float(np.max(frame_mean)):.8g}")
    print(f"  worst frame by max error     : {int(np.argmax(frame_max))}, max={float(np.max(frame_max)):.8g}")

    bbox_diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    print(f"  bbox diag                    : {bbox_diag:.8g}")
    print(f"  mean error / bbox diag       : {float(np.mean(l2_error) / max(bbox_diag, 1e-10)):.8g}")
    print(f"  max  error / bbox diag       : {float(np.max(l2_error) / max(bbox_diag, 1e-10)):.8g}")


def build_plotter(
    stage: dict[str, Any],
    *,
    frame: int,
    heatmap: bool,
    show_gt: bool,
    show_skeleton: bool,
    off_screen: bool,
    window_size: tuple[int, int],
) -> Any:
    pv_mod = require_pyvista()

    frame_count = int(stage["recon"].shape[0])
    frame = max(0, min(int(frame), frame_count - 1))

    vertices_recon = stage["recon"]
    vertices_target = stage["baked"]
    faces = stage["faces"]
    bt = stage["bt"]
    parent_indices = stage["parent_indices"]
    l2_error = stage["l2_error"]

    pv_faces = to_pv_faces(faces)

    pv_mod.set_plot_theme("document")
    plotter = pv_mod.Plotter(
        title="Direct SSDR / Fitbox LBS Viewer",
        off_screen=bool(off_screen),
        window_size=window_size,
    )

    recon_mesh = pv_mod.PolyData(vertices_recon[frame], pv_faces)
    target_mesh = pv_mod.PolyData(vertices_target[frame], pv_faces)

    heatmap_clim_max = float(np.percentile(l2_error, 99)) if l2_error.size else 1.0
    heatmap_clim_max = max(heatmap_clim_max, 1e-10)

    if heatmap:
        recon_mesh["L2_Error"] = l2_error[frame]
        recon_actor = plotter.add_mesh(
            recon_mesh,
            scalars="L2_Error",
            cmap="turbo",
            clim=(0.0, heatmap_clim_max),
            show_edges=False,
            smooth_shading=True,
            name="recon_mesh",
        )
    else:
        recon_actor = plotter.add_mesh(
            recon_mesh,
            color="lightblue",
            show_edges=False,
            smooth_shading=True,
            name="recon_mesh",
        )

    target_actor = plotter.add_mesh(
        target_mesh,
        style="wireframe",
        color="orange",
        opacity=0.35,
        name="target_baked_mesh",
    )
    target_actor.SetVisibility(bool(show_gt))

    joint_points = bt[frame, :, :3, 3]
    skeleton_mesh = pv_mod.PolyData(joint_points)
    skeleton_mesh.lines = get_skeleton_lines(parent_indices)

    skeleton_actor = plotter.add_mesh(
        skeleton_mesh,
        color="red",
        line_width=4,
        render_lines_as_tubes=True,
        name="skeleton",
    )
    skeleton_actor.SetVisibility(bool(show_skeleton))

    state = {
        "frame": frame,
    }

    text_actor = plotter.add_text(
        f"frame={frame}/{frame_count - 1} | mean L2={float(l2_error[frame].mean()):.6g}",
        position="upper_left",
        font_size=10,
    )

    def update_frame(value: float) -> None:
        f_id = int(np.round(value))
        f_id = max(0, min(f_id, frame_count - 1))
        state["frame"] = f_id

        recon_mesh.points = vertices_recon[f_id]
        target_mesh.points = vertices_target[f_id]
        skeleton_mesh.points = bt[f_id, :, :3, 3]

        if heatmap:
            recon_mesh["L2_Error"] = l2_error[f_id]
            try:
                plotter.update_scalars(l2_error[f_id], mesh=recon_mesh, render=False)
            except Exception:
                pass

        try:
            text_actor.SetInput(
                f"frame={f_id}/{frame_count - 1} | mean L2={float(l2_error[f_id].mean()):.6g}"
            )
        except Exception:
            pass

        plotter.render()

    def toggle_gt(flag: bool) -> None:
        target_actor.SetVisibility(bool(flag))
        plotter.render()

    def toggle_skeleton(flag: bool) -> None:
        skeleton_actor.SetVisibility(bool(flag))
        plotter.render()

    if frame_count > 1 and not off_screen:
        plotter.add_slider_widget(
            update_frame,
            rng=[0, frame_count - 1],
            value=frame,
            title="Frame",
            pointa=(0.10, 0.08),
            pointb=(0.90, 0.08),
            style="modern",
            fmt="%0.0f",
        )

        plotter.add_checkbox_button_widget(
            toggle_gt,
            value=bool(show_gt),
            position=(10, 90),
            color_on="orange",
            size=28,
        )
        plotter.add_text("Show baked target", position=(48, 94), font_size=9)

        plotter.add_checkbox_button_widget(
            toggle_skeleton,
            value=bool(show_skeleton),
            position=(10, 52),
            color_on="red",
            size=28,
        )
        plotter.add_text("Show skeleton", position=(48, 56), font_size=9)

    plotter.add_axes()
    plotter.reset_camera()

    return plotter


def parse_window_size(text: str) -> tuple[int, int]:
    if "x" not in text.lower():
        raise argparse.ArgumentTypeError("Expected window size format WIDTHxHEIGHT, e.g. 1600x1000")

    a, b = text.lower().split("x", 1)
    return int(a), int(b)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct SSDR / Fitbox LBS viewer independent of Blender."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("--mesh", type=str, default=None, help="Optional mesh name/stem. Use ALL, or omit, to load all meshes.")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--stats_only", action="store_true")

    parser.add_argument("--heatmap", action="store_true", help="Color recon mesh by per-vertex L2 error.")
    parser.add_argument("--no_gt", action="store_true", help="Hide baked target wireframe initially.")
    parser.add_argument("--no_skeleton", action="store_true", help="Hide skeleton initially.")

    parser.add_argument("--off_screen", action="store_true")
    parser.add_argument("--screenshot", type=str, default=None)
    parser.add_argument("--window_size", type=parse_window_size, default=(1600, 1000))

    args = parser.parse_args(script_argv())

    stage = load_stage(
        Path(args.input_dir),
        mesh_name=args.mesh,
    )

    print_stage_stats(stage)

    if args.stats_only:
        return

    plotter = build_plotter(
        stage,
        frame=int(args.frame),
        heatmap=bool(args.heatmap),
        show_gt=not bool(args.no_gt),
        show_skeleton=not bool(args.no_skeleton),
        off_screen=bool(args.off_screen),
        window_size=args.window_size,
    )

    if args.screenshot:
        out = Path(args.screenshot).expanduser().resolve(strict=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(out), auto_close=True)
        print(f"Saved screenshot: {out}")
        return

    plotter.show()


if __name__ == "__main__":
    main()
