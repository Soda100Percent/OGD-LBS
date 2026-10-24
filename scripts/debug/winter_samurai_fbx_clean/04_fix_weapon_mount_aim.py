#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Weapon mount extension:
    preserve the direct right-hand mount's Weapon origin;
    point Weapon local -Y toward the evaluated left-hand midpoint + offset.

Reuses the existing sibling 04_fix_weapon_mount.py.
No mirror, negative scale, stretch or persistent constraint is created.

Offset:
    target_offset_fraction_of_length is expressed in target-bone local axes,
    multiplied by target rest bone length, then transformed into world space.

The original implementation handles detach, action rebake, save/reload.
This extension adds independent direction/origin/non-descendant-motion audits.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np

TAG = "[ws-weapon-aim]"


def log(text):
    print(f"{TAG} {text}", flush=True)


def require(ok, text):
    if not ok:
        raise RuntimeError(text)


def argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


def dump(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_core(path):
    spec = importlib.util.spec_from_file_location("ws_weapon_mount_core", path)
    require(spec is not None and spec.loader is not None, "Cannot load core script.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in (
        "execute", "make_expected_world_for_spec",
        "set_armature_action", "frame_set", "error_arrays",
    ):
        require(callable(getattr(module, name, None)),
                f"Existing core is incompatible; missing {name}")
    return module


def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    length = float(np.linalg.norm(v))
    require(np.isfinite(v).all() and length > eps, "Degenerate direction.")
    return v / length


def aim_matrix(base, target):
    """Keep origin and positive axis scales; set local -Y toward target."""
    base = np.asarray(base, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    require(np.isfinite(base).all() and np.isfinite(target).all(),
            "Nonfinite aim input.")
    require(np.allclose(base[3], [0, 0, 0, 1], atol=1e-6, rtol=0),
            "Non-affine baseline.")

    linear = base[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    require(np.all(scales > 1e-12), "Singular baseline.")
    rotation = linear / scales[None, :]
    require(np.linalg.det(rotation) > 0, "Reflected baseline is unsupported.")
    require(np.max(np.abs(rotation.T @ rotation - np.eye(3))) < 1e-4,
            "Baseline contains shear; refusing silent approximation.")

    origin = base[:3, 3]
    direction = target - origin
    require(np.linalg.norm(direction) > 1e-9,
            "Aim target coincides with Weapon origin.")

    # Weapon -Y points toward the target.
    y = -unit(direction)

    # Preserve baseline Z as closely as possible to fix roll.
    z0 = rotation[:, 2]
    projected_z = z0 - np.dot(z0, y) * y

    if np.linalg.norm(projected_z) >= 1e-4:
        z = unit(projected_z)
        x = unit(np.cross(y, z))
        z = unit(np.cross(x, y))
        branch = "PROJECT_BASELINE_Z"
    else:
        # Explicitly reported fallback near the roll singularity.
        x0 = rotation[:, 0]
        x = unit(x0 - np.dot(x0, y) * y)
        z = unit(np.cross(x, y))
        x = unit(np.cross(y, z))
        branch = "FALLBACK_PROJECT_BASELINE_X"

    r = np.column_stack((x, y, z))
    require(abs(np.linalg.det(r) - 1.0) < 1e-8, "Improper aim rotation.")

    out = base.copy()
    out[:3, :3] = r * scales[None, :]
    return out, branch


def set_action(core, arm, action):
    core.set_armature_action(arm, action)
    slots = list(getattr(action, "slots", []))
    if slots:
        require(len(slots) == 1, "Aim test supports one-slot Actions.")
        arm.animation_data.action_slot = slots[0]
    core.update_view_layer()


def evaluated_state(arm, target_name, offset_fraction, watched_names):
    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = arm.evaluated_get(dg)
    a = np.asarray(evaluated.matrix_world, dtype=np.float64)
    pb = evaluated.pose.bones[target_name]

    # Midpoint of evaluated head/tail, in armature space.
    midpoint = (
        np.asarray(pb.head, dtype=np.float64)
        + np.asarray(pb.tail, dtype=np.float64)
    ) * 0.5

    length = float(arm.data.bones[target_name].length)
    local_offset = np.asarray(offset_fraction, dtype=np.float64) * length
    pose_linear = np.asarray(pb.matrix, dtype=np.float64)[:3, :3]

    target_armature = midpoint + pose_linear @ local_offset
    target_world = a[:3, :3] @ target_armature + a[:3, 3]

    watched = np.stack([
        a @ np.asarray(evaluated.pose.bones[n].matrix, dtype=np.float64)
        for n in watched_names
    ])
    return target_world, watched


def summarize_matrix_error(core, actual, expected):
    errors = core.error_arrays(actual, expected)
    return {key: float(np.max(value)) for key, value in errors.items()}


def within(errors, position, rotation, linear):
    return (
        errors["translation_l2"] <= position
        and errors["rotation_deg"] <= rotation
        and errors["linear_relative"] <= linear
    )


def independent_audit(core, out):
    manifest = json.loads((out / "aim_audit_manifest.json").read_text(encoding="utf-8"))
    core.open_blend(Path(manifest["output_blend"]))
    arm = bpy.data.objects[manifest["armature"]]
    weapon = manifest["weapon_bone"]
    require(arm.data.bones[weapon].parent is None, "Weapon was not detached.")

    result = {"status": "running", "actions": []}
    for entry in manifest["actions"]:
        with np.load(out / entry["cache"], allow_pickle=False) as z:
            times = z["times"]
            expected = z["expected_world"]
            baseline = z["baseline_world"]
            targets = z["target_world"]
            watched_expected = z["watched_world"]
            watched_names = z["watched_names"].tolist()

        set_action(core, arm, bpy.data.actions[entry["action"]])
        actual = []
        watched_actual = []
        current_targets = []

        for t in times:
            core.frame_set(float(t))
            target, watched = evaluated_state(
                arm, entry["target_bone"],
                entry["target_offset_fraction_of_length"], watched_names,
            )
            evaluated = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
            w = np.asarray(
                evaluated.matrix_world @ evaluated.pose.bones[weapon].matrix,
                dtype=np.float64,
            )
            actual.append(w)
            watched_actual.append(watched)
            current_targets.append(target)

        actual = np.stack(actual)
        watched_actual = np.stack(watched_actual)
        current_targets = np.stack(current_targets)

        pose_error = summarize_matrix_error(core, actual, expected)
        other_error = summarize_matrix_error(core, watched_actual, watched_expected)

        ray = current_targets - actual[:, :3, 3]
        ray = ray / np.linalg.norm(ray, axis=1)[:, None]
        minus_y = -actual[:, :3, 1]
        minus_y = minus_y / np.linalg.norm(minus_y, axis=1)[:, None]
        cross = np.linalg.norm(np.cross(minus_y, ray), axis=1)
        dot = np.clip(np.sum(minus_y * ray, axis=1), -1, 1)
        angles = np.degrees(np.arctan2(cross, dot))

        origin_error = float(np.max(np.linalg.norm(
            actual[:, :3, 3] - baseline[:, :3, 3], axis=1
        )))
        target_error = float(np.max(np.linalg.norm(current_targets - targets, axis=1)))
        det_min = float(np.min(np.linalg.det(actual[:, :3, :3])))

        tol = manifest["tolerances"]
        passed = bool(
            within(pose_error, tol["position"], tol["rotation_deg"], tol["linear"])
            and within(other_error, tol["position"], tol["rotation_deg"], tol["linear"])
            and origin_error <= tol["position"]
            and target_error <= tol["position"]
            and float(angles.max()) <= tol["aim_angle_deg"]
            and det_min > 0
        )

        item = {
            "action": entry["action"],
            "passed": passed,
            "sample_count": len(times),
            "aim_minus_y_angle_max_deg": float(angles.max()),
            "weapon_origin_vs_right_mount_max": origin_error,
            "left_target_change_max": target_error,
            "weapon_world_error": pose_error,
            "non_weapon_non_descendant_world_error": other_error,
            "excluded_weapon_descendants": entry["excluded_descendants"],
            "world_linear_det_min": det_min,
            "roll_fallback_count": entry["roll_fallback_count"],
        }
        result["actions"].append(item)
        log(str(item))

    require(result["actions"], "No aim actions were audited.")
    result["status"] = (
        "passed" if all(x["passed"] for x in result["actions"]) else "failed"
    )
    dump(out / "aim_audit_report.json", result)
    require(result["status"] == "passed", "Independent aim audit failed.")
    return result


def self_test():
    for target in ([1, 2, 3], [0, 0, 1], [0, 0, -1], [-2, 1, 0]):
        base = np.eye(4)
        base[:3, :3] *= 2
        result, _ = aim_matrix(base, target)
        require(np.allclose(-unit(result[:3, 1]), unit(target)),
                "-Y direction self-test failed.")
        require(np.array_equal(result[:3, 3], base[:3, 3]),
                "Origin preservation self-test failed.")
        require(np.allclose(np.linalg.norm(result[:3, :3], axis=0), [2, 2, 2]),
                "Scale preservation self-test failed.")
        require(np.linalg.det(result[:3, :3]) > 0, "Reflection self-test failed.")
    rejected = False
    try:
        aim_matrix(np.eye(4), [0, 0, 0])
    except RuntimeError:
        rejected = True
    require(rejected, "Coincident target must fail.")
    log(f"Math self-test passed; bpy={bpy.app.version_string}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--audit_saved", action="store_true")
    parser.add_argument("--input_blend")
    parser.add_argument("--output_dir")
    parser.add_argument("--output_blend")
    parser.add_argument("--config")
    parser.add_argument("--core_script")
    parser.add_argument("--armature_name")
    parser.add_argument("--position_tol", type=float, default=1e-4)
    parser.add_argument("--rotation_tol_deg", type=float, default=0.05)
    parser.add_argument("--linear_tol", type=float, default=1e-5)
    args = parser.parse_args(argv())

    if args.self_test:
        self_test()
        return

    require(args.output_dir, "--output_dir is required.")
    core_path = (
        Path(args.core_script).resolve() if args.core_script
        else Path(__file__).resolve().with_name("04_fix_weapon_mount.py")
    )
    require(core_path.is_file(), f"Missing existing core: {core_path}")
    core = load_core(core_path)
    out = Path(args.output_dir).expanduser().resolve()

    if args.audit_saved:
        independent_audit(core, out)
        return

    require(args.input_blend and args.config, "--input_blend and --config are required.")
    for value in (args.position_tol, args.rotation_tol_deg, args.linear_tol):
        require(math.isfinite(value) and value > 0, "Invalid tolerance.")

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    require(all(m.get("type", "direct") == "direct" for m in cfg["mounts"].values()),
            "This aim experiment accepts direct mounts only; mirror is disabled.")
    require(cfg.get("detach_weapon_bone", {}).get("enabled", False),
            "This experiment requires Weapon detach.")
    aims = {
        name: policy["aim"]
        for name, policy in cfg["action_policies"].items()
        if policy.get("enabled", True) and "aim" in policy
    }
    require(aims, "No enabled aim policy.")

    out.mkdir(parents=True, exist_ok=True)
    for name in ("aim_cache", "aim_audit_manifest.json", "aim_audit_report.json",
                 "aim_extension_report.json"):
        require(not (out / name).exists(), f"Output exists: {out/name}")
    (out / "aim_cache").mkdir()

    output_blend = (
        Path(args.output_blend).expanduser().resolve() if args.output_blend
        else out / "Winter_Samurai_weapon_aim.blend"
    )
    require(not output_blend.exists(), f"Output exists: {output_blend}")

    report = {
        "status": "running",
        "core_script": str(core_path),
        "core_sha256": sha(core_path),
        "output_blend": str(output_blend),
    }
    manifest = {
        "output_blend": str(output_blend),
        "weapon_bone": cfg["weapon_bone"],
        "actions": [],
        "tolerances": {
            "position": args.position_tol,
            "rotation_deg": args.rotation_tol_deg,
            "linear": args.linear_tol,
            "aim_angle_deg": 0.05,
        },
    }
    original_generator = core.make_expected_world_for_spec

    def extended_generator(*, arm, spec, mount_offsets, original_worlds):
        baseline = original_generator(
            arm=arm, spec=spec, mount_offsets=mount_offsets,
            original_worlds=original_worlds,
        )
        action_name = spec["action"]
        if action_name not in aims:
            return baseline

        settings = aims[action_name]
        require(settings.get("axis", "-Y") == "-Y", "This version supports only -Y aim.")
        target_name = settings["target_bone"]
        offset = np.asarray(
            settings.get("target_offset_fraction_of_length", [0, 0, 0]),
            dtype=float,
        )
        require(offset.shape == (3,) and np.isfinite(offset).all(), "Invalid target offset.")

        weapon = arm.data.bones[cfg["weapon_bone"]]
        excluded = sorted(b.name for b in weapon.children_recursive)
        require(target_name in arm.pose.bones, f"Missing target bone: {target_name}")
        require(target_name != weapon.name and target_name not in excluded,
                "Aim target must not depend on Weapon.")
        watched_names = sorted(
            b.name for b in arm.data.bones
            if b.name != weapon.name and b.name not in excluded
        )
        require(watched_names, "No independent bones to audit.")

        set_action(core, arm, bpy.data.actions[action_name])
        targets, watched, expected, branches = [], [], [], []

        for i, t in enumerate(spec["frames"]):
            core.frame_set(float(t))
            target, other = evaluated_state(arm, target_name, offset, watched_names)
            result, branch = aim_matrix(baseline[i], target)
            targets.append(target)
            watched.append(other)
            expected.append(result)
            branches.append(branch)

        expected = np.stack(expected)
        cache = f"aim_cache/action_{len(manifest['actions']):03d}.npz"
        np.savez_compressed(
            out / cache,
            times=np.asarray(spec["frames"], dtype=float),
            expected_world=expected,
            baseline_world=baseline,
            target_world=np.stack(targets),
            watched_world=np.stack(watched),
            watched_names=np.asarray(watched_names, dtype=str),
        )
        manifest["armature"] = arm.name
        manifest["actions"].append({
            "action": action_name,
            "cache": cache,
            "target_bone": target_name,
            "target_offset_fraction_of_length": offset.tolist(),
            "excluded_descendants": excluded,
            "roll_fallback_count": branches.count("FALLBACK_PROJECT_BASELINE_X"),
        })
        dump(out / "aim_audit_manifest.json", manifest)
        log(f"{action_name}: -Y aim; target={target_name}; offset_fraction={offset.tolist()}")
        return expected

    try:
        core.make_expected_world_for_spec = extended_generator
        core.execute(argparse.Namespace(
            input_blend=args.input_blend,
            output_dir=str(out),
            output_blend=str(output_blend),
            config=args.config,
            armature_name=args.armature_name,
            position_tol=args.position_tol,
            rotation_tol_deg=args.rotation_tol_deg,
            linear_tol=args.linear_tol,
            overwrite=False,
        ))
        require({x["action"] for x in manifest["actions"]} == set(aims),
                "Not all requested aim policies were executed.")
        independent_audit(core, out)
        report["status"] = "passed"
        report["scope"] = (
            "Sampled -Y aiming and Weapon-origin preservation; "
            "not a guarantee of mesh grip-point preservation or subframe contact."
        )
        dump(out / "aim_extension_report.json", report)
        log(f"COMPLETE: {output_blend}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        dump(out / "aim_extension_report.json", report)
        raise
    finally:
        core.make_expected_world_for_spec = original_generator


if __name__ == "__main__":
    main()
