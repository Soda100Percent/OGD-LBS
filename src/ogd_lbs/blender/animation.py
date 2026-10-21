# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Reusable Blender animation utilities.

This module is intentionally generic and not tied to OGD-LBS data-prep file
naming conventions. It contains reusable helpers for:
    - inspecting Actions / Action Slots
    - clearing NLA tracks
    - concatenating Actions into one NLA track
    - binding Blender 4.4+ Slotted Actions to NLA strips
"""

from dataclasses import dataclass
from typing import Iterable, Any

import bpy

from ogd_lbs.blender.object import ensure_object_mode, deselect_all_objects, set_active_object
from ogd_lbs.blender.context import update_view_layer


@dataclass
class NLAActionItem:
    action_name: str
    frame_start: int
    frame_end: int
    action_frame_start: int
    action_frame_end: int
    action_length: int
    slot_bound: bool
    slot_info: str


@dataclass
class NLAConcatResult:
    track_name: str
    added_count: int
    scene_frame_start: int
    scene_frame_end: int
    actions: list[NLAActionItem]


def get_action_slot_display_names(action: bpy.types.Action) -> list[str]:
    """
    Return Blender 4.4+ Action Slot display names if available.

    Older Blender versions or non-slotted Actions simply return an empty list.
    """
    if not hasattr(action, "slots"):
        return []

    try:
        return [
            str(getattr(slot, "name_display", getattr(slot, "identifier", "Unknown")))
            for slot in action.slots
        ]
    except Exception:
        return []


def bind_action_slot_to_nla_strip(
    strip: bpy.types.NlaStrip,
    action: bpy.types.Action,
) -> tuple[bool, str]:
    """
    Bind the first Action Slot to an NLA strip for Blender 4.4+.

    Returns:
        (slot_bound, slot_info)

    Semantics:
        - If the Action has no slots, returns (False, "no_slots").
        - If binding succeeds, returns (True, "...").
        - If binding fails, returns (False, error_message).
    """
    if not hasattr(action, "slots"):
        return False, "action_has_no_slots_attr"

    try:
        if len(action.slots) <= 0:
            return False, "no_slots"

        slot = action.slots[0]
    except Exception as exc:
        return False, f"failed_to_read_slots: {exc}"

    slot_name = str(getattr(slot, "name_display", getattr(slot, "identifier", "Unknown")))

    if hasattr(strip, "action_slot"):
        try:
            strip.action_slot = slot
            return True, f"action_slot={slot_name}"
        except Exception as exc:
            return False, f"failed_action_slot={slot_name}: {exc}"

    if hasattr(strip, "action_slot_handle"):
        try:
            strip.action_slot_handle = slot.handle
            return True, f"action_slot_handle={slot_name}"
        except Exception as exc:
            return False, f"failed_action_slot_handle={slot_name}: {exc}"

    return False, f"strip_has_no_slot_binding_api={slot_name}"


def clear_armature_nla_tracks(
    armature_obj: bpy.types.Object,
    *,
    clear_active_action: bool = True,
) -> int:
    """
    Clear NLA tracks from an armature object.

    Returns the number of removed tracks.
    """
    if armature_obj is None or armature_obj.type != "ARMATURE":
        raise RuntimeError("clear_armature_nla_tracks expects an ARMATURE object.")

    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()

    if clear_active_action:
        armature_obj.animation_data.action = None

    removed = 0
    for track in reversed(armature_obj.animation_data.nla_tracks):
        armature_obj.animation_data.nla_tracks.remove(track)
        removed += 1

    return removed


def configure_nla_strip_defaults(strip: bpy.types.NlaStrip) -> None:
    """
    Set explicit NLA strip defaults to avoid implicit non-playing behavior.
    """
    try:
        strip.blend_type = "REPLACE"
    except Exception:
        pass

    try:
        strip.extrapolation = "NOTHING"
    except Exception:
        pass

    try:
        strip.use_auto_blend = False
    except Exception:
        pass

    try:
        strip.mute = False
    except Exception:
        pass


def iter_actions_for_concat(
    actions: Iterable[bpy.types.Action],
    *,
    exclude_names: set[str],
) -> list[bpy.types.Action]:
    """
    Preserve bpy.data.actions order while filtering by exact action name.
    """
    return [action for action in actions if action.name not in exclude_names]


def concat_all_actions_to_nla(
    armature_obj: bpy.types.Object,
    *,
    actions: Iterable[bpy.types.Action] | None = None,
    gap: int = 10,
    exclude_names: Iterable[str] | None = None,
    track_name: str = "OGD_Concat_Track",
    start_frame: int = 1,
    fail_if_empty: bool = True,
    verbose: bool = True,
) -> NLAConcatResult:
    """
    Concatenate all Actions into one NLA track on the target armature.

    This preserves the behavior used by the original Prep 01 script:
        - clear existing NLA tracks
        - clear active action
        - create one OGD_Concat_Track
        - add actions in bpy.data.actions order
        - bind Blender 4.4+ Action Slot if available
        - set explicit strip playback options
        - set scene frame range to the concatenated range
    """
    if armature_obj is None or armature_obj.type != "ARMATURE":
        raise RuntimeError("concat_all_actions_to_nla expects an ARMATURE object.")

    ensure_object_mode()

    if actions is None:
        action_list = list(bpy.data.actions)
    else:
        action_list = list(actions)

    exclude_set = {str(name).strip() for name in (exclude_names or []) if str(name).strip()}
    action_list = iter_actions_for_concat(action_list, exclude_names=exclude_set)

    if fail_if_empty and not action_list:
        raise RuntimeError(
            "No actions available for NLA concatenation after applying exclusions. "
            f"exclude_names={sorted(exclude_set)}"
        )

    try:
        armature_obj.data.pose_position = "POSE"
    except Exception:
        pass

    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()

    if verbose:
        print("🧹 Clearing existing NLA tracks and active actions...", flush=True)

    removed = clear_armature_nla_tracks(armature_obj, clear_active_action=True)

    if verbose:
        print(f"   removed NLA tracks: {removed}", flush=True)

    track = armature_obj.animation_data.nla_tracks.new()
    track.name = track_name

    try:
        track.is_solo = False
    except Exception:
        pass

    try:
        track.mute = False
    except Exception:
        pass

    current_frame = int(start_frame)
    added_items: list[NLAActionItem] = []

    if verbose:
        print("🔗 Starting NLA concatenation...", flush=True)
        if exclude_set:
            print(f"   exclude_names: {sorted(exclude_set)}", flush=True)
        print(f"   gap: {int(gap)}", flush=True)

    for action in action_list:
        frame_start_raw, frame_end_raw = action.frame_range
        action_frame_start = int(frame_start_raw)
        action_frame_end = int(frame_end_raw)
        action_length = int(frame_end_raw - frame_start_raw)

        strip = track.strips.new(
            name=action.name,
            start=current_frame,
            action=action,
        )

        slot_bound, slot_info = bind_action_slot_to_nla_strip(strip, action)
        configure_nla_strip_defaults(strip)

        strip_frame_start = int(strip.frame_start)
        strip_frame_end = int(strip.frame_end)

        added_items.append(
            NLAActionItem(
                action_name=action.name,
                frame_start=strip_frame_start,
                frame_end=strip_frame_end,
                action_frame_start=action_frame_start,
                action_frame_end=action_frame_end,
                action_length=action_length,
                slot_bound=slot_bound,
                slot_info=slot_info,
            )
        )

        if verbose:
            slot_msg = f", slot={slot_info}" if slot_info else ""
            print(
                f"  ➕ Added: {action.name} "
                f"(Start: {strip_frame_start}, End: {strip_frame_end}, "
                f"ActionFrames: {action_frame_start}..{action_frame_end}, "
                f"Length: {action_length}{slot_msg})",
                flush=True,
            )

        current_frame = int(strip.frame_end) + int(gap)

    if added_items:
        scene_frame_start = int(start_frame)
        scene_frame_end = current_frame - int(gap)
    else:
        scene_frame_start = int(start_frame)
        scene_frame_end = int(start_frame)

    bpy.context.scene.frame_start = scene_frame_start
    bpy.context.scene.frame_end = scene_frame_end

    update_view_layer()

    if verbose:
        print(f"\n✅ Concatenation complete! Added {len(added_items)} actions.", flush=True)
        print(f"⏱️ New scene frame range: {scene_frame_start} to {scene_frame_end}", flush=True)

    return NLAConcatResult(
        track_name=track.name,
        added_count=len(added_items),
        scene_frame_start=scene_frame_start,
        scene_frame_end=scene_frame_end,
        actions=added_items,
    )


def print_action_inspect_summary() -> None:
    """
    Print action ranges and Action Slot metadata.
    """
    print(f"\n🎬 Actions ({len(bpy.data.actions)}):", flush=True)

    for action in bpy.data.actions:
        frame_start, frame_end = action.frame_range
        slots = get_action_slot_display_names(action)
        slots_info = f" | Slots: {slots}" if slots else ""
        print(
            f"  - {action.name} "
            f"(Frames: {int(frame_start)} to {int(frame_end)}){slots_info}",
            flush=True,
        )
