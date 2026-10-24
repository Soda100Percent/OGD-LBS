#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Optional Winter Samurai Step 02a.

Input:
    Step 02 blend, containing one mesh-bound armature and baked Actions.
Reference:
    <dataset_dir>/model/rest_pose_base.blend

World-space LBS-preserving basis conversion:
    old_rest_world = old_arm_world @ old_rest
    new_rest_world = new_arm_world @ new_rest
    C = inverse(old_rest_world) @ new_rest_world
    new_pose_world[f] = old_pose_world[f] @ C

Keep source meshes, topology, weights, materials and armature object identity.
Append only reference armature data, not reference meshes or Actions.
Rebake all input Actions and preserve their public names.

Validation:
    - reference rest matrices, heads/tails and parent relationships
    - sampled bone world poses and world deformation operators
    - evaluated world vertices of every source mesh at every sample
    - mesh object world matrices
    - base geometry/weight/material/binding signatures
    - before save, after reload, and optional independent --audit_saved

Strict scope:
    - positive uniform-scale armature object transforms
    - standard vertex-group LBS, one enabled Armature modifier per mesh
    - no source constraints, drivers, NLA, Shape Keys or enabled extra modifiers
    - one-slot, complete location/quaternion/scale Actions from Step 02
    - identical source/reference bone-name sets
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import struct
import sys
import tempfile
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

TAG = "[ws-rest-rebase-02a]"
CHANNELS = (("location", 3), ("rotation_quaternion", 4), ("scale", 3))


def log(text):
    print(f"{TAG} {text}", flush=True)


def require(ok, text):
    if not ok:
        raise RuntimeError(text)


def argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


def nm(value):
    return np.asarray(value, dtype=np.float64).copy()


def bm(value):
    return Matrix(np.asarray(value, dtype=np.float64).tolist())


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def update():
    bpy.context.view_layer.update()
    bpy.context.evaluated_depsgraph_get().update()


def object_mode():
    obj = bpy.context.view_layer.objects.active
    if obj is not None and obj.mode != "OBJECT":
        require(bpy.ops.object.mode_set(mode="OBJECT") == {"FINISHED"},
                "Cannot enter OBJECT mode.")


def activate(obj):
    object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def frame(t):
    base = math.floor(float(t))
    bpy.context.scene.frame_set(base, subframe=float(t) - base)
    update()


def open_blend(path):
    log(f"Opening: {path}")
    require(bpy.ops.wm.open_mainfile(
        filepath=str(path), load_ui=False
    ) == {"FINISHED"}, f"Cannot open: {path}")
    object_mode()
    update()


def save_blend(path):
    require(bpy.ops.wm.save_as_mainfile(
        filepath=str(path), check_existing=False
    ) == {"FINISHED"}, f"Cannot save: {path}")


def has_animation(id_block):
    ad = getattr(id_block, "animation_data", None)
    return bool(ad and (
        ad.action is not None or len(ad.nla_tracks) or len(ad.drivers)
    ))


def affine(a, label):
    a = np.asarray(a, dtype=float)
    require(np.isfinite(a).all(), f"{label}: nonfinite matrix.")
    require(np.allclose(a[..., 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0),
            f"{label}: non-affine matrix.")
    singular = np.linalg.svd(a[..., :3, :3], compute_uv=False)
    require(np.all(singular[..., -1] > 1e-12),
            f"{label}: singular/near-singular matrix.")


def require_similarity(a, label):
    affine(a, label)
    linear = a[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    require(np.linalg.det(linear) > 0, f"{label}: reflected transform.")
    relative = float((singular.max() - singular.min()) / singular.mean())
    require(relative <= 1e-5,
            f"{label}: nonuniform scale/shear unsupported; "
            f"singular_values={singular.tolist()}")
    return float(singular.mean())


def errors(a, b):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape, f"Matrix shape mismatch: {a.shape} vs {b.shape}")
    affine(a, "actual")
    affine(b, "expected")

    def rotation(x):
        u, _, vt = np.linalg.svd(x[..., :3, :3])
        sign = np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)
        u = u.copy()
        u[..., :, -1] *= sign[..., None]
        return u @ vt

    d = np.swapaxes(rotation(a), -1, -2) @ rotation(b)
    cosine = np.clip((np.trace(d, axis1=-2, axis2=-1) - 1) * 0.5, -1, 1)
    skew = np.stack((
        d[..., 2, 1] - d[..., 1, 2],
        d[..., 0, 2] - d[..., 2, 0],
        d[..., 1, 0] - d[..., 0, 1],
    ), axis=-1)
    angle = np.degrees(np.arctan2(0.5 * np.linalg.norm(skew, axis=-1), cosine))

    return {
        "position": np.linalg.norm(a[..., :3, 3] - b[..., :3, 3], axis=-1),
        "rotation_deg": angle,
        "linear_relative": (
            np.linalg.norm(a[..., :3, :3] - b[..., :3, :3], axis=(-2, -1))
            / np.maximum(np.linalg.norm(b[..., :3, :3], axis=(-2, -1)), 1e-12)
        ),
        "matrix_abs": np.max(np.abs(a - b), axis=(-2, -1)),
    }


def maxima(a, b):
    return {k: float(np.max(v)) for k, v in errors(a, b).items()}


def acceptable(e, tol):
    return bool(
        e["position"] <= tol["position"]
        and e["rotation_deg"] <= tol["rotation_deg"]
        and e["linear_relative"] <= tol["linear_relative"]
    )


def choose_armature(name=None, source=False):
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if name:
        candidates = [o for o in arms if o.name == name]
        require(len(candidates) == 1, f"Armature {name!r} not found; "
                f"available={[o.name for o in arms]}")
        chosen = candidates[0]
    else:
        targets = {}
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                for m in obj.modifiers:
                    if m.type == "ARMATURE" and m.object is not None:
                        targets[m.object.name] = m.object
        if len(targets) == 1:
            chosen = next(iter(targets.values()))
        elif len(arms) == 1:
            chosen = arms[0]
        else:
            raise RuntimeError(
                f"Ambiguous reference/source armature: {[o.name for o in arms]}; "
                f"mesh_bound={sorted(targets)}. Specify its exact name."
            )
    if source:
        require(len(arms) == 1, "Step 02a requires a one-armature Step 02 output.")
        require(not chosen.get("ws_02a_rest_base_rebased", False),
                "Input is already marked as Step 02a output; use original Step 02.")
    return chosen


def skeleton(arm):
    names = sorted(b.name for b in arm.data.bones)
    require(names, "Empty skeleton.")
    return {
        "names": names,
        "data_name": arm.data.name,
        "rest": np.stack([nm(arm.data.bones[n].matrix_local) for n in names]),
        "heads": np.stack([nm(arm.data.bones[n].head_local) for n in names]),
        "tails": np.stack([nm(arm.data.bones[n].tail_local) for n in names]),
        "parents": [
            arm.data.bones[n].parent.name if arm.data.bones[n].parent else None
            for n in names
        ],
        "deform": [bool(arm.data.bones[n].use_deform) for n in names],
        "connected": [bool(arm.data.bones[n].use_connect) for n in names],
        "segments": [int(arm.data.bones[n].bbone_segments) for n in names],
    }


def read_reference(path, arm_name, ref_frame):
    open_blend(path)
    arm = choose_armature(arm_name)
    require(not has_animation(arm.data),
            "Reference Armature data has animation; this step reads a static rest template.")
    frame(ref_frame)
    data = skeleton(arm)
    data["object_name"] = arm.name
    data["world"] = nm(arm.evaluated_get(
        bpy.context.evaluated_depsgraph_get()).matrix_world)
    data["uniform_scale"] = require_similarity(data["world"], "reference object")
    require(all(x == 1 for x in data["segments"]),
            "Reference has segmented B-Bones; unsupported.")
    return data


def action_curves(action):
    slots = list(getattr(action, "slots", []))
    if slots:
        require(len(slots) == 1, f"{action.name}: multiple Action slots unsupported.")
        layers = list(action.layers)
        require(len(layers) == 1 and len(layers[0].strips) == 1,
                f"{action.name}: expected one layer and one keyframe strip.")
        bag = layers[0].strips[0].channelbag(slots[0])
        require(bag is not None, f"{action.name}: missing channelbag.")
        return list(bag.fcurves)
    return list(action.fcurves)


def reset_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.scale = (1, 1, 1)


def set_action(arm, action):
    ad = arm.animation_data_create()
    ad.action = None
    reset_pose(arm)
    ad.action = action
    slots = list(getattr(action, "slots", []))
    if slots:
        require(len(slots) == 1, f"{action.name}: multiple slots unsupported.")
        ad.action_slot = slots[0]
    arm.data.pose_position = "POSE"
    update()


def source_plans(arm, samples_per_frame):
    require(arm.parent is None, "Source armature must have no object parent.")
    require(not arm.constraints, "Source armature has object constraints.")
    require(not has_animation(arm.data), "Source Armature data is animated.")
    ad = arm.animation_data
    require(ad is not None, "Source armature has no animation data.")
    require(not ad.nla_tracks and not ad.drivers, "Source NLA/drivers unsupported.")
    require(not any(pb.constraints for pb in arm.pose.bones),
            "Source pose constraints unsupported.")

    expected = {
        (pb.path_from_id(channel), i)
        for pb in arm.pose.bones
        for channel, count in CHANNELS
        for i in range(count)
    }

    plans = []
    for index, action in enumerate(sorted(bpy.data.actions, key=lambda a: a.name)):
        curves = action_curves(action)
        actual = {(fc.data_path, fc.array_index) for fc in curves}
        require(actual == expected and len(curves) == len(expected),
                f"{action.name}: expected complete Step 02 bone TRS curves. "
                f"missing={sorted(expected-actual)[:10]}, "
                f"extra={sorted(actual-expected)[:10]}")
        require(not any(fc.modifiers for fc in curves),
                f"{action.name}: FCurve modifiers unsupported.")

        start, end = map(float, action.frame_range)
        require(math.isfinite(start) and math.isfinite(end) and end >= start,
                f"{action.name}: invalid range.")
        count = int(math.floor((end - start) * samples_per_frame + 1e-7)) + 1
        require(count <= 100000, f"{action.name}: too many samples.")
        times = list(start + np.arange(count) / samples_per_frame)
        times.append(end)
        for fc in curves:
            times.extend(float(k.co.x) for k in fc.keyframe_points)
        times = np.asarray(sorted(set(round(float(t), 7) for t in times)))
        require(len(times) <= 100000, "Too many combined sample times.")

        plans.append({
            "index": index,
            "name": action.name,
            "times": times,
            "cache": f"cache/action_{index:03d}.npz",
            "vertices": f"cache/vertices_{index:03d}.npy",
        })
    require(plans, "No Actions.")
    return plans


def topology_hash(mesh):
    h = hashlib.sha256()
    h.update(np.asarray([l.vertex_index for l in mesh.loops], dtype="<i8").tobytes())
    h.update(np.asarray([(p.loop_start, p.loop_total) for p in mesh.polygons],
                        dtype="<i8").tobytes())
    return h.hexdigest()


def mesh_signature(obj):
    h = hashlib.sha256()
    h.update(np.asarray([tuple(v.co) for v in obj.data.vertices],
                        dtype="<f4").tobytes())
    h.update(topology_hash(obj.data).encode())
    h.update(json.dumps([g.name for g in obj.vertex_groups]).encode())
    for v in obj.data.vertices:
        for g in v.groups:
            h.update(struct.pack("<IIf", v.index, g.group, float(g.weight)))
    h.update(json.dumps(
        [s.material.name if s.material else None for s in obj.material_slots]
    ).encode())
    h.update(np.asarray([p.material_index for p in obj.data.polygons],
                        dtype="<i4").tobytes())
    for uv in obj.data.uv_layers:
        h.update(uv.name.encode())
        h.update(np.asarray([tuple(x.uv) for x in uv.data],
                            dtype="<f4").tobytes())
    return {
        "hash": h.hexdigest(),
        "data_name": obj.data.name,
        "parent": obj.parent.name if obj.parent else None,
        "parent_type": obj.parent_type,
        "parent_bone": obj.parent_bone,
        "modifier_targets": [
            m.object.name if m.object else None
            for m in obj.modifiers if m.type == "ARMATURE"
        ],
    }


def mesh_layout(arm):
    layout = []
    offset = 0
    meshes = sorted(
        [o for o in bpy.context.scene.objects if o.type == "MESH"],
        key=lambda o: o.name,
    )
    require(meshes, "No source meshes.")
    for obj in meshes:
        require(not has_animation(obj) and not has_animation(obj.data),
                f"{obj.name}: mesh/object animation unsupported.")
        require(not obj.constraints, f"{obj.name}: mesh constraints unsupported.")
        require(obj.data.shape_keys is None, f"{obj.name}: Shape Keys unsupported.")
        mods = [m for m in obj.modifiers if m.type == "ARMATURE"]
        require(len(mods) == 1 and mods[0].object is arm,
                f"{obj.name}: expected exactly one modifier targeting {arm.name}.")
        m = mods[0]
        require(m.show_viewport and m.show_render,
                f"{obj.name}: Armature modifier must be enabled.")
        require(m.use_vertex_groups and not m.use_bone_envelopes,
                f"{obj.name}: expected vertex groups without envelopes.")
        require(not m.use_deform_preserve_volume and not m.use_multi_modifier,
                f"{obj.name}: Preserve Volume/Multi Modifier unsupported.")
        require(not m.vertex_group, f"{obj.name}: modifier mask unsupported.")
        require(not any(x.type != "ARMATURE" and (x.show_viewport or x.show_render)
                        for x in obj.modifiers),
                f"{obj.name}: enabled extra modifier unsupported.")

        count = len(obj.data.vertices)
        require(count > 0, f"{obj.name}: empty mesh.")
        layout.append({
            "name": obj.name,
            "start": offset,
            "end": offset + count,
            "topology": topology_hash(obj.data),
            "signature": mesh_signature(obj),
        })
        offset += count
    return layout, offset


def pose_world(arm, names):
    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = arm.evaluated_get(dg)
    a = nm(evaluated.matrix_world)
    p = np.stack([nm(evaluated.pose.bones[n].matrix) for n in names])
    return a @ p, a


def evaluated_meshes(layout):
    dg = bpy.context.evaluated_depsgraph_get()
    vertices = []
    worlds = []
    for item in layout:
        obj = bpy.data.objects[item["name"]].evaluated_get(dg)
        mesh = obj.to_mesh()
        try:
            require(mesh is not None, f"No evaluated mesh: {item['name']}")
            require(len(mesh.vertices) == item["end"] - item["start"],
                    f"Evaluated vertex count changed: {item['name']}")
            require(topology_hash(mesh) == item["topology"],
                    f"Evaluated topology changed: {item['name']}")
            co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            world = nm(obj.matrix_world)
            vertices.append(co @ world[:3, :3].T + world[:3, 3])
            worlds.append(world)
        finally:
            obj.to_mesh_clear()
    v = np.concatenate(vertices, axis=0)
    require(np.isfinite(v).all(), "Nonfinite evaluated vertex.")
    return v, np.stack(worlds)


def capture_sources(out, arm, names, plans, layout, total_vertices, old_world, tol):
    estimated = sum(len(p["times"]) for p in plans) * total_vertices * 3 * 8
    require(shutil.disk_usage(out).free > estimated * 1.15 + 64 * 1024 * 1024,
            f"Insufficient free disk for all-vertex cache; estimated {estimated/2**30:.2f} GiB.")
    log(f"Vertex cache estimate: {estimated/2**30:.3f} GiB")

    for plan in plans:
        action = bpy.data.actions[plan["name"]]
        set_action(arm, action)
        times = plan["times"]
        poses = np.empty((len(times), len(names), 4, 4))
        mesh_worlds = np.empty((len(times), len(layout), 4, 4))
        cache = np.lib.format.open_memmap(
            out / plan["vertices"], mode="w+", dtype=np.float64,
            shape=(len(times), total_vertices, 3),
        )
        for i, t in enumerate(times):
            frame(t)
            w, a = pose_world(arm, names)
            require(acceptable(maxima(a, old_world), tol),
                    f"{action.name}: armature object world transform varies.")
            poses[i] = w
            cache[i], mesh_worlds[i] = evaluated_meshes(layout)
            if i == 0 or i == len(times)-1 or (i+1) % 25 == 0:
                log(f"Capture {action.name}: {i+1}/{len(times)} frame={t:g}")
        cache.flush()
        del cache
        affine(poses, f"{action.name}/source poses")
        np.savez_compressed(out / plan["cache"],
                            times=times, old_world=poses, mesh_world=mesh_worlds)
    return estimated


def parent_order(arm):
    out = []
    pending = set(arm.pose.bones.keys())
    while pending:
        ready = sorted(n for n in pending
                       if arm.pose.bones[n].parent is None
                       or arm.pose.bones[n].parent.name in out)
        require(ready, "Invalid target hierarchy.")
        out.extend(ready)
        pending.difference_update(ready)
    return out


def replace_armature_data(arm, ref_path, ref, old_world):
    child_inverses = {}
    for obj in arm.children:
        require(obj.parent_type == "OBJECT",
                f"{obj.name}: bone/vertex parenting requires a separate migration.")
        require(not has_animation(obj) and not obj.constraints,
                f"{obj.name}: animated/constrained child unsupported.")
        child_inverses[obj.name] = nm(obj.matrix_parent_inverse)

    old_data = arm.data
    old_action_names = set(bpy.data.actions.keys())
    old_object_names = set(bpy.data.objects.keys())

    with bpy.data.libraries.load(str(ref_path), link=False) as (available, requested):
        require(ref["data_name"] in available.armatures,
                f"Reference armature data not found: {ref['data_name']}")
        requested.armatures = [ref["data_name"]]
    data = requested.armatures[0]
    require(data is not None, "Could not append reference Armature data.")
    require(set(bpy.data.objects.keys()) == old_object_names,
            "Reference data unexpectedly appended object dependencies.")
    require(set(bpy.data.actions.keys()) == old_action_names,
            "Reference data unexpectedly appended animation dependencies.")

    arm.animation_data_clear()
    arm.data = data
    data.name = arm.name + "_02a_RestBase"
    data.animation_data_clear()
    data.pose_position = "POSE"

    arm.delta_location = (0, 0, 0)
    arm.delta_rotation_euler = (0, 0, 0)
    arm.delta_rotation_quaternion = (1, 0, 0, 0)
    arm.delta_scale = (1, 1, 1)
    arm.rotation_mode = "QUATERNION"
    loc, quat, scale = bm(ref["world"]).decompose()
    arm.location, arm.rotation_quaternion, arm.scale = loc, quat, scale
    update()

    new_world = nm(arm.matrix_world)
    for name, old_inverse in child_inverses.items():
        # A_new @ MPI_new @ basis == A_old @ MPI_old @ basis
        bpy.data.objects[name].matrix_parent_inverse = bm(
            np.linalg.inv(new_world) @ old_world @ old_inverse
        )
    update()

    activate(arm)
    require(bpy.ops.object.mode_set(mode="EDIT") == {"FINISHED"},
            "Cannot edit target armature.")
    try:
        for eb in data.edit_bones:
            eb.use_connect = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    reset_pose(arm)
    for pb in arm.pose.bones:
        require(not pb.constraints, "Unexpected target pose constraints.")
    update()

    return old_data, new_world, sorted(child_inverses)


def write_action(arm, name, names, times, values):
    action = bpy.data.actions.new(name)
    require(action.name == name, f"Action name collision: {name}")
    action.use_fake_user = True

    if bpy.app.version >= (4, 4, 0):
        slot = action.slots.new(id_type="OBJECT", name=arm.name)
        layer = action.layers.new("Rebased")
        strip = layer.strips.new(type="KEYFRAME")
        bag = strip.channelbag(slot, ensure=True)
        curves, groups = bag.fcurves, bag.groups
    else:
        curves, groups = action.fcurves, action.groups

    for bi, bone_name in enumerate(names):
        pb = arm.pose.bones[bone_name]
        group = groups.new(bone_name)
        offset = 0
        for channel, count in CHANNELS:
            path = pb.path_from_id(channel)
            for index in range(count):
                fc = curves.new(data_path=path, index=index)
                fc.group = group
                fc.extrapolation = "CONSTANT"
                fc.keyframe_points.add(len(times))
                co = np.column_stack((times, values[:, bi, offset+index]))
                fc.keyframe_points.foreach_set("co", co.reshape(-1))
                for k in fc.keyframe_points:
                    k.interpolation = "LINEAR"
                fc.update()
            offset += count
    set_action(arm, action)
    require(len(action_curves(action)) == len(names) * 10,
            f"{name}: incomplete baked channels.")
    return action


def bake(out, arm, names, plans, correction, target_world, tol):
    indices = {name: i for i, name in enumerate(names)}
    order = parent_order(arm)
    target_inv = np.linalg.inv(target_world)

    for plan in plans:
        with np.load(out / plan["cache"], allow_pickle=False) as z:
            old = z["old_world"]
            times = z["times"]
        expected = old @ correction

        arm.animation_data_clear()
        reset_pose(arm)
        values = np.empty((len(times), len(names), 10))
        previous = {}
        for fi, t in enumerate(times):
            frame(t)
            reset_pose(arm)
            update()
            desired = target_inv @ expected[fi]

            for name in order:
                bi = indices[name]
                pb = arm.pose.bones[name]
                pb.matrix = bm(desired[bi])
                update()
                q = pb.rotation_quaternion.copy()
                if name in previous and q.dot(previous[name]) < 0:
                    q.negate()
                    pb.rotation_quaternion = q
                previous[name] = q.copy()
                values[fi, bi] = [
                    *pb.location, *q, *pb.scale,
                ]

            update()
            actual, _ = pose_world(arm, names)
            e = maxima(actual, expected[fi])
            require(acceptable(e, tol),
                    f"Unrepresentable/inaccurate target pose: "
                    f"{plan['name']} frame={t:g}, errors={e}")
            if fi == 0 or fi == len(times)-1 or (fi+1) % 25 == 0:
                log(f"Bake {plan['name']}: {fi+1}/{len(times)} frame={t:g}")

        write_action(arm, plan["name"], names, times, values)


def audit(out, manifest):
    arm = bpy.data.objects.get(manifest["armature"])
    require(arm is not None and arm.type == "ARMATURE", "Final armature missing.")
    require(set(bpy.data.objects.keys()) == set(manifest["object_names"]),
            "Final object inventory changed.")
    require(sorted(bpy.data.actions.keys()) == sorted(
        p["name"] for p in manifest["plans"]), "Final Action inventory mismatch.")

    with np.load(out / "basis_map.npz", allow_pickle=False) as z:
        names = z["bone_names"].tolist()
        new_rest = z["new_rest"]
        target_requested_rest = z["reference_rest"]
        target_heads, target_tails = z["reference_heads"], z["reference_tails"]
        old_rest_world = z["old_rest_world"]
        target_world = z["target_object_world"]
        correction = z["correction_world"]
        new_rest_world = z["new_rest_world"]

    now = skeleton(arm)
    require(now["names"] == names, "Final bone names changed.")
    require(now["parents"] == manifest["reference_parents"],
            "Final hierarchy differs from reference.")
    require(now["deform"] == manifest["reference_deform"],
            "Final deform flags differ from reference.")
    require(not any(now["connected"]), "Connected flags unexpectedly enabled.")
    rest_error = max(
        float(np.max(np.abs(now["rest"] - target_requested_rest))),
        float(np.max(np.abs(now["heads"] - target_heads))),
        float(np.max(np.abs(now["tails"] - target_tails))),
    )
    require(rest_error <= manifest["rest_tol"],
            f"Final rest geometry differs from reference: {rest_error}")

    for item in manifest["meshes"]:
        require(mesh_signature(bpy.data.objects[item["name"]]) == item["signature"],
                f"Base geometry/weights/materials/binding changed: {item['name']}")

    tol = manifest["tolerances"]
    records, mesh_rows = [], []
    for plan in manifest["plans"]:
        with np.load(out / plan["cache"], allow_pickle=False) as z:
            times = z["times"]
            old = z["old_world"]
            old_mesh_world = z["mesh_world"]
        expected = old @ correction
        vertex_cache = np.load(out / plan["vertices"], mmap_mode="r", allow_pickle=False)

        set_action(arm, bpy.data.actions[plan["name"]])
        pose_max = dict.fromkeys(("position", "rotation_deg", "linear_relative", "matrix_abs"), 0.0)
        op_max = dict(pose_max)
        mesh_world_max = dict(pose_max)
        per_mesh = {
            x["name"]: {"max": 0.0, "sum": 0.0, "count": 0,
                        "worst_frame": None, "worst_vertex": None}
            for x in manifest["meshes"]
        }
        for fi, t in enumerate(times):
            frame(t)
            actual, a = pose_world(arm, names)
            require(acceptable(maxima(a, target_world), tol),
                    f"{plan['name']}: final object transform drift.")
            pe = maxima(actual, expected[fi])
            oe = maxima(
                actual @ np.linalg.inv(new_rest_world),
                old[fi] @ np.linalg.inv(old_rest_world),
            )
            current_vertices, current_worlds = evaluated_meshes(manifest["meshes"])
            me = maxima(current_worlds, old_mesh_world[fi])

            for dst, src in ((pose_max, pe), (op_max, oe), (mesh_world_max, me)):
                for key in dst:
                    dst[key] = max(dst[key], src[key])

            delta = np.linalg.norm(current_vertices - vertex_cache[fi], axis=1)
            for item in manifest["meshes"]:
                d = delta[item["start"]:item["end"]]
                r = per_mesh[item["name"]]
                max_d = float(d.max())
                if r["worst_frame"] is None or max_d > r["max"]:
                    r["max"] = max_d
                    r["worst_frame"] = float(t)
                    r["worst_vertex"] = int(d.argmax())
                r["sum"] += float(d.sum())
                r["count"] += len(d)

        del vertex_cache
        vertex_max = max(r["max"] for r in per_mesh.values())
        passed = bool(
            acceptable(pose_max, tol)
            and acceptable(op_max, tol)
            and acceptable(mesh_world_max, tol)
            and vertex_max <= tol["mesh_position"]
        )
        record = {
            "action": plan["name"],
            "samples": len(times),
            "passed": passed,
            "bone_world": pose_max,
            "deformation_operator_world": op_max,
            "mesh_object_world": mesh_world_max,
            "mesh_vertex_world_l2_max": vertex_max,
        }
        records.append(record)
        for name, r in per_mesh.items():
            mesh_rows.append({
                "action": plan["name"], "mesh": name,
                "mean": r["sum"] / r["count"], "max": r["max"],
                "worst_frame": r["worst_frame"],
                "worst_vertex": r["worst_vertex"],
            })
        log(f"Audit {plan['name']}: passed={passed}, mesh_max={vertex_max:.9g}")

    return {
        "passed": all(x["passed"] for x in records),
        "rest_geometry_abs_max": rest_error,
        "bone_count": len(names),
        "mesh_count": len(manifest["meshes"]),
        "action_count": len(records),
        "actions": records,
        "mesh_rows": mesh_rows,
    }


def write_mesh_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        fields = ["action", "mesh", "mean", "max", "worst_frame", "worst_vertex"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    input_path = Path(args.input_blend).expanduser().resolve()
    reference_path = (
        Path(args.dataset_dir).expanduser().resolve() / "model" / "rest_pose_base.blend"
    )
    out = Path(args.output_dir).expanduser().resolve()
    require(input_path.is_file(), f"Missing Step 02 blend: {input_path}")
    require(reference_path.is_file(), f"Missing reference: {reference_path}")
    output_path = out / "Winter_Samurai_clean02a_rest_base_rebased.blend"
    require(output_path not in {input_path, reference_path}, "Refusing input overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        output_path.name, "rebase_report.json", "audit_manifest.json",
        "basis_map.npz", "cache", "mesh_error_after_reload.csv",
    ):
        require(not (out / name).exists(),
                f"Output exists: {out/name}. Archive the run before rerunning.")
    (out / "cache").mkdir()

    report_path = out / "rebase_report.json"
    report = {
        "schema": "ogd_lbs_ws_02a_rest_base_rebase_v1",
        "status": "running", "stage": "reference",
        "bpy_version": bpy.app.version_string,
        "input_blend": str(input_path),
        "reference_blend": str(reference_path),
        "output_blend": str(output_path),
    }

    try:
        hashes = {str(p): sha_file(p) for p in (input_path, reference_path)}
        ref = read_reference(reference_path, args.reference_armature, args.reference_frame)

        report["reference"] = {
            "object": ref["object_name"], "data": ref["data_name"],
            "bone_count": len(ref["names"]),
            "object_world": ref["world"].tolist(),
            "uniform_scale": ref["uniform_scale"],
        }

        report["stage"] = "source_preflight"
        open_blend(input_path)
        arm = choose_armature(args.source_armature, source=True)
        old = skeleton(arm)
        require(set(old["names"]) == set(ref["names"]),
                "Source/reference bone names differ. No automatic remapping.\n"
                f"source_only={sorted(set(old['names'])-set(ref['names']))}\n"
                f"reference_only={sorted(set(ref['names'])-set(old['names']))}")
        names = old["names"]
        require(old["deform"] == ref["deform"],
                "Source/reference use_deform flags differ; refusing silent binding changes.")
        require(all(x == 1 for x in old["segments"]), "Source segmented B-Bones unsupported.")

        plans = source_plans(arm, args.samples_per_frame)
        set_action(arm, bpy.data.actions[plans[0]["name"]])
        frame(plans[0]["times"][0])
        _, old_world = pose_world(arm, names)
        source_scale = require_similarity(old_world, "source object")
        layout, total_vertices = mesh_layout(arm)

        # No other object may own animation that would be broken by replacing Actions.
        for obj in bpy.data.objects:
            if obj is not arm:
                require(not has_animation(obj),
                        f"Other animated object unsupported: {obj.name}")

        tol = {
            "position": args.position_tol,
            "rotation_deg": args.rotation_tol_deg,
            "linear_relative": args.linear_tol,
            "mesh_position": args.mesh_position_tol,
        }
        report["counts"] = {
            "bones": len(names), "meshes": len(layout),
            "vertices": total_vertices, "actions": len(plans),
            "samples": sum(len(p["times"]) for p in plans),
        }
        report["source_armature"] = arm.name
        report["source_uniform_scale"] = source_scale
        report["tolerances"] = tol
        report["source_reference_parent_differences"] = [
            {"bone": n, "old": old["parents"][i], "new": ref["parents"][i]}
            for i, n in enumerate(names) if old["parents"][i] != ref["parents"][i]
        ]
        old_rest_world = old_world @ old["rest"]
        requested_rest_world = ref["world"] @ ref["rest"]
        report["rest_head_world_shift_max"] = float(np.max(np.linalg.norm(
            old_rest_world[:, :3, 3] - requested_rest_world[:, :3, 3], axis=1
        )))
        dump(report_path, report)
        log(f"Counts: {report['counts']}")

        report["stage"] = "capture_source_vertices_and_poses"
        report["vertex_cache_estimated_bytes"] = capture_sources(
            out, arm, names, plans, layout, total_vertices, old_world, tol
        )

        report["stage"] = "replace_reference_rest"
        old_data, target_world, adjusted_children = replace_armature_data(
            arm, reference_path, ref, old_world
        )
        new = skeleton(arm)
        require(new["names"] == names and new["parents"] == ref["parents"],
                "Appended target skeleton mismatch.")
        rest_error = max(
            float(np.max(np.abs(new["rest"] - ref["rest"]))),
            float(np.max(np.abs(new["heads"] - ref["heads"]))),
            float(np.max(np.abs(new["tails"] - ref["tails"]))),
        )
        require(rest_error <= args.rest_tol,
                f"Target rest geometry mismatch after append/disconnect: {rest_error}")
        require(acceptable(maxima(target_world, ref["world"]), tol),
                "Target object world transform mismatch.")

        new_rest_world = target_world @ new["rest"]
        correction = np.linalg.inv(old_rest_world) @ new_rest_world
        affine(correction, "basis correction")
        np.savez_compressed(
            out / "basis_map.npz",
            bone_names=np.asarray(names, dtype=str),
            old_rest=old["rest"], new_rest=new["rest"],
            reference_rest=ref["rest"],
            reference_heads=ref["heads"], reference_tails=ref["tails"],
            old_object_world=old_world, target_object_world=target_world,
            old_rest_world=old_rest_world, new_rest_world=new_rest_world,
            correction_world=correction,
        )
        report["adjusted_child_parent_inverses"] = adjusted_children
        report["disabled_connect_bones"] = [
            n for i, n in enumerate(names) if ref["connected"][i]
        ]
        report["formula"] = "W_new = W_old @ inverse(R_old_world) @ R_new_world"

        # Keep only the known input Actions as temporary old Actions.
        old_actions = [bpy.data.actions[p["name"]] for p in plans]
        for i, action in enumerate(old_actions):
            action.name = f"__WS02A_OLD_{i:04d}"

        report["stage"] = "bake"
        bake(out, arm, names, plans, correction, target_world, tol)

        for action in old_actions:
            bpy.data.actions.remove(action, do_unlink=True)
        if old_data.users == 0:
            bpy.data.armatures.remove(old_data)

        manifest = {
            "schema": "ogd_lbs_ws_02a_audit_manifest_v1",
            "output_blend": str(output_path),
            "armature": arm.name,
            "object_names": sorted(bpy.data.objects.keys()),
            "reference_parents": ref["parents"],
            "reference_deform": ref["deform"],
            "meshes": layout,
            "plans": [
                {k: p[k] for k in ("index", "name", "cache", "vertices")}
                for p in plans
            ],
            "tolerances": tol,
            "rest_tol": args.rest_tol,
            "input_hashes": hashes,
        }
        dump(out / "audit_manifest.json", manifest)

        report["stage"] = "audit_before_save"
        before = audit(out, manifest)
        report["validation_before_save"] = {
            k: v for k, v in before.items() if k != "mesh_rows"
        }
        dump(report_path, report)
        require(before["passed"], "Pre-save numerical audit failed.")

        # Restore a useful, deterministic viewing state before saving.
        arm["ws_02a_rest_base_rebased"] = True
        arm["ws_02a_reference_blend"] = str(reference_path)
        arm["ws_02a_policy"] = "preserve sampled world LBS deformation operators"
        first = plans[0]
        set_action(arm, bpy.data.actions[first["name"]])
        bpy.context.scene.frame_start = int(math.floor(first["times"][0]))
        bpy.context.scene.frame_end = int(math.ceil(first["times"][-1]))
        frame(first["times"][0])

        report["stage"] = "save"
        save_blend(output_path)

        report["stage"] = "audit_after_reload"
        open_blend(output_path)
        after = audit(out, manifest)
        report["validation_after_reload"] = {
            k: v for k, v in after.items() if k != "mesh_rows"
        }
        write_mesh_csv(out / "mesh_error_after_reload.csv", after["mesh_rows"])
        require(after["passed"], "After-reload numerical audit failed.")

        unchanged = {p: sha_file(p) == h for p, h in hashes.items()}
        report["input_files_unchanged"] = unchanged
        require(all(unchanged.values()), "An input/reference file changed on disk.")

        report["status"] = "passed"
        report["stage"] = "complete"
        report["scope"] = (
            "Sampled mesh deformation preservation while changing rest basis and hierarchy. "
            "No weapon-mount correction. No claim of continuous-time equivalence."
        )
        dump(report_path, report)
        log(f"PASSED: {output_path}")
        log(f"Report: {report_path}")

    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        dump(report_path, report)
        log(f"FAILED at {report['stage']}; report={report_path}")
        raise


def audit_saved(out):
    out = Path(out).expanduser().resolve()
    manifest = read(out / "audit_manifest.json")
    open_blend(manifest["output_blend"])
    result = audit(out, manifest)
    unchanged = {
        p: Path(p).is_file() and sha_file(p) == h
        for p, h in manifest["input_hashes"].items()
    }
    result["input_files_unchanged"] = unchanged
    result["passed"] = bool(result["passed"] and all(unchanged.values()))
    dump(out / "independent_audit.json", result)
    require(result["passed"], "Independent audit failed.")
    log(f"INDEPENDENT AUDIT PASSED: {out / 'independent_audit.json'}")


def make_test_arm(name, world, target=False):
    data = bpy.data.armatures.new(name + "_Data")
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = world
    activate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    root = data.edit_bones.new("root")
    root.head, root.tail = (0, 0, 0), (0, 1, 0)
    child = data.edit_bones.new("child")
    child.head, child.tail = (0, 1, 0), (0, 2, 0)
    if not target:
        child.parent = root
    else:
        root.matrix = Matrix.Rotation(0.6, 4, "Z")
        child.matrix = Matrix.Translation((0, 1, 0)) @ Matrix.Rotation(0.5, 4, "X")
    bpy.ops.object.mode_set(mode="OBJECT")
    for b in obj.data.bones:
        b.use_deform = True
    reset_pose(obj)
    return obj


def self_test():
    # Algebra test.
    ro = nm(Matrix.Translation((1, 2, 3)) @ Matrix.Rotation(0.4, 4, "X"))
    rn = nm(Matrix.Translation((-2, 1, 0)) @ Matrix.Rotation(0.7, 4, "Z"))
    wo = nm(Matrix.Translation((3, 1, 4)) @ Matrix.Rotation(0.9, 4, "Y"))
    wn = wo @ np.linalg.inv(ro) @ rn
    require(np.allclose(wn @ np.linalg.inv(rn), wo @ np.linalg.inv(ro)),
            "Basis algebra self-test failed.")

    # End-to-end real Blender test: two bones, changed hierarchy/orientations,
    # different armature object transforms, a parented weighted mesh, two frames.
    with tempfile.TemporaryDirectory(prefix="ogd_ws02a_selftest_") as temp:
        root = Path(temp)
        (root / "model").mkdir()
        source_path = root / "source.blend"
        ref_path = root / "model" / "rest_pose_base.blend"

        bpy.ops.wm.read_factory_settings(use_empty=True)
        old_a = Matrix.Translation((1, 2, 0)) @ Matrix.Rotation(0.2, 4, "Z") @ Matrix.Scale(2, 4)
        arm = make_test_arm("SourceRig", old_a)

        data = bpy.data.meshes.new("TestMeshData")
        data.from_pydata([(0, 0, 0), (1, 0, 0), (1, 2, 0), (0, 2, 0)],
                         [], [(0, 1, 2, 3)])
        mesh = bpy.data.objects.new("TestMesh", data)
        bpy.context.scene.collection.objects.link(mesh)
        mesh.parent = arm
        mesh.matrix_parent_inverse = Matrix.Identity(4)
        mesh.matrix_basis = Matrix.Identity(4)
        for name, weight in (("root", 0.4), ("child", 0.6)):
            mesh.vertex_groups.new(name=name).add(list(range(4)), weight, "REPLACE")
        mod = mesh.modifiers.new("Skin", "ARMATURE")
        mod.object = arm
        mod.use_vertex_groups = True
        mod.use_bone_envelopes = False
        mod.use_deform_preserve_volume = False

        for t in (1, 2):
            frame(t)
            for i, pb in enumerate(arm.pose.bones):
                pb.location = (0.07 * t * i, 0, 0)
                pb.rotation_quaternion = Matrix.Rotation(0.15 * t * (i+1), 4, "Z").to_quaternion()
                pb.scale = (1, 1, 1)
                for channel, _ in CHANNELS:
                    require(pb.keyframe_insert(data_path=channel, frame=t),
                            "Synthetic keyframe insertion failed.")
        arm.animation_data.action.name = "WS_Test"
        arm.animation_data.action.use_fake_user = True
        save_blend(source_path)

        bpy.ops.wm.read_factory_settings(use_empty=True)
        new_a = Matrix.Translation((-1, 0, 2)) @ Matrix.Rotation(-0.3, 4, "Y") @ Matrix.Scale(0.5, 4)
        make_test_arm("ReferenceRig", new_a, target=True)
        save_blend(ref_path)

        args = parser().parse_args([
            "--input_blend", str(source_path),
            "--dataset_dir", str(root),
            "--output_dir", str(root / "result"),
        ])
        run(args)
        audit_saved(root / "result")
    log(f"SELF TEST PASSED; bpy={bpy.app.version_string}")


def parser():
    p = argparse.ArgumentParser(description="Optional rest_pose_base LBS-preserving rebase.")
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--audit_saved", action="store_true")
    p.add_argument("--input_blend")
    p.add_argument("--output_dir")
    p.add_argument("--dataset_dir", default="/workspace/ogd_dataset/Winter_Samurai")
    p.add_argument("--source_armature")
    p.add_argument("--reference_armature")
    p.add_argument("--reference_frame", type=float, default=1.0)
    p.add_argument("--samples_per_frame", type=int, default=1)
    p.add_argument("--position_tol", type=float, default=1e-4)
    p.add_argument("--rotation_tol_deg", type=float, default=0.05)
    p.add_argument("--linear_tol", type=float, default=1e-5)
    p.add_argument("--mesh_position_tol", type=float, default=1e-4)
    p.add_argument("--rest_tol", type=float, default=1e-5)
    return p


def main():
    p = parser()
    args = p.parse_args(argv())
    if args.self_test:
        self_test()
        return
    if args.audit_saved:
        if not args.output_dir:
            p.error("--audit_saved requires --output_dir.")
        audit_saved(args.output_dir)
        return
    if not args.input_blend or not args.output_dir:
        p.error("--input_blend and --output_dir are required.")
    if args.samples_per_frame < 1:
        p.error("--samples_per_frame must be positive.")
    for key in ("position_tol", "rotation_tol_deg", "linear_tol",
                "mesh_position_tol", "rest_tol"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            p.error(f"--{key} must be finite and positive.")
    if not math.isfinite(args.reference_frame):
        p.error("--reference_frame must be finite.")
    run(args)


if __name__ == "__main__":
    main()
