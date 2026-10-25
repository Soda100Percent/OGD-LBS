# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic object hierarchy cleanup helpers.
"""

from typing import Any, Iterable

import bpy
from mathutils import Matrix

from ogd_lbs.blender.context import update_view_layer
from ogd_lbs.blender.object import ensure_object_mode


def has_animation_data(id_block: Any) -> bool:
    ad = getattr(id_block, "animation_data", None)
    if ad is None:
        return False

    try:
        if ad.action is not None:
            return True
    except Exception:
        pass

    try:
        if len(ad.nla_tracks) > 0:
            return True
    except Exception:
        pass

    try:
        if len(ad.drivers) > 0:
            return True
    except Exception:
        pass

    return False


def get_object_animation_info(id_block: Any) -> dict[str, Any]:
    ad = getattr(id_block, "animation_data", None)

    if ad is None:
        return {
            "has_animation_data": False,
            "active_action": None,
            "nla_track_count": 0,
            "nla_strip_count": 0,
            "driver_count": 0,
            "animation_score": 0,
        }

    active_action = None
    nla_track_count = 0
    nla_strip_count = 0
    driver_count = 0

    try:
        active_action = ad.action.name if ad.action is not None else None
    except Exception:
        active_action = None

    try:
        nla_track_count = int(len(ad.nla_tracks))
        nla_strip_count = int(sum(len(track.strips) for track in ad.nla_tracks))
    except Exception:
        pass

    try:
        driver_count = int(len(ad.drivers))
    except Exception:
        pass

    score = 0
    if active_action:
        score += 10
    score += nla_strip_count * 5
    score += driver_count

    return {
        "has_animation_data": True,
        "active_action": active_action,
        "nla_track_count": nla_track_count,
        "nla_strip_count": nla_strip_count,
        "driver_count": driver_count,
        "animation_score": int(score),
    }


def object_parent_depth(obj: bpy.types.Object) -> int:
    depth = 0
    cur = obj.parent
    seen: set[str] = set()

    while cur is not None and cur.name not in seen:
        seen.add(cur.name)
        depth += 1
        cur = cur.parent

    return depth


def nearest_surviving_parent(
    obj: bpy.types.Object,
    objects_to_remove: set[bpy.types.Object],
) -> bpy.types.Object | None:
    cur = obj.parent

    while cur is not None and cur in objects_to_remove:
        cur = cur.parent

    return cur


def remove_objects_preserve_surviving_children(
    objects_to_remove: Iterable[bpy.types.Object],
    *,
    fail_on_animated: bool = False,
) -> dict[str, Any]:
    ensure_object_mode()
    update_view_layer()

    remove_list = sorted(
        list(objects_to_remove),
        key=lambda obj: (object_parent_depth(obj), obj.name),
        reverse=True,
    )
    remove_set = set(remove_list)

    animated = []
    for obj in remove_list:
        info = get_object_animation_info(obj)
        if int(info.get("animation_score", 0)) > 0:
            animated.append(
                {
                    "name": obj.name,
                    "type": obj.type,
                    "animation": info,
                    "parent": obj.parent.name if obj.parent else None,
                    "children": [child.name for child in obj.children],
                }
            )

    if animated and fail_on_animated:
        raise RuntimeError(
            "Refusing to remove animated objects: "
            + ", ".join(str(item["name"]) for item in animated)
        )

    reparented = []
    survivors = [obj for obj in bpy.data.objects if obj not in remove_set]

    for obj in sorted(survivors, key=object_parent_depth, reverse=True):
        if obj.parent is None:
            continue

        cur = obj.parent
        contains_removed = False
        seen: set[str] = set()

        while cur is not None and cur.name not in seen:
            seen.add(cur.name)
            if cur in remove_set:
                contains_removed = True
                break
            cur = cur.parent

        if not contains_removed:
            continue

        old_parent = obj.parent.name if obj.parent else None
        old_world = obj.matrix_world.copy()
        new_parent = nearest_surviving_parent(obj, remove_set)

        obj.parent = new_parent

        if new_parent is not None:
            try:
                obj.matrix_parent_inverse = new_parent.matrix_world.inverted()
            except Exception:
                obj.matrix_parent_inverse = Matrix.Identity(4)
        else:
            obj.matrix_parent_inverse = Matrix.Identity(4)

        obj.matrix_world = old_world

        reparented.append(
            {
                "object": obj.name,
                "type": obj.type,
                "old_parent": old_parent,
                "new_parent": new_parent.name if new_parent else None,
            }
        )

    update_view_layer()

    removed = []
    failed = []

    for obj in remove_list:
        name = obj.name
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed.append(name)
        except Exception as exc:
            failed.append({"name": name, "error": str(exc)})

    update_view_layer()

    return {
        "requested_count": int(len(remove_list)),
        "removed_count": int(len(removed)),
        "removed_names": removed,
        "failed_remove": failed,
        "animated_count": int(len(animated)),
        "animated_summaries": animated,
        "reparented_surviving_object_count": int(len(reparented)),
        "reparented_surviving_objects": reparented,
    }


def remove_empty_objects_preserve_children(
    *,
    fail_on_animated_empty: bool = False,
) -> dict[str, Any]:
    empties = [obj for obj in bpy.data.objects if obj.type == "EMPTY"]

    result = remove_objects_preserve_surviving_children(
        empties,
        fail_on_animated=bool(fail_on_animated_empty),
    )

    remaining = sorted([obj.name for obj in bpy.data.objects if obj.type == "EMPTY"])

    result.update(
        {
            "enabled": True,
            "policy": "remove_all_EMPTY_objects_preserve_surviving_current_world_matrix",
            "removed_empty_count": int(result["removed_count"]),
            "removed_empty_names": list(result["removed_names"]),
            "remaining_empty_count": int(len(remaining)),
            "remaining_empty_names": remaining,
            "animated_empty_count": int(result["animated_count"]),
            "animated_empty_summaries": list(result["animated_summaries"]),
        }
    )

    return result
