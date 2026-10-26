# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Shared SSDR stage constants.

This module centralizes file stems that are common across SSDR staged pipeline
scripts. It intentionally does not contain dataset names, asset-specific
configuration, or SSDR algorithm logic.
"""

ANIMATION_STEM = "animation_data"
SKELETON_STEM = "skeleton_hierarchy_data"

LEGACY_REDUCED_SKELETON_STEM = "skeleton_hierarchy_data_bone_reduced"
LEGACY_REDUCED_ANIMATION_STEM = "animation_data_bone_reduced"

CORE_STEMS = {
    ANIMATION_STEM,
    SKELETON_STEM,
}

LEGACY_REDUCED_STEMS = {
    LEGACY_REDUCED_SKELETON_STEM,
    LEGACY_REDUCED_ANIMATION_STEM,
}

STEP_SUMMARY_STEMS = {
    "step00_convert_summary",
    "step01_frame_sampling_summary",
    "step02_reduce_bone_weight_summary",
    "step03_qem_summary",
    "step04_optimize_summary",
}

METRIC_STEMS = {
    "metrics_history_per_frame",
}

KNOWN_NON_MESH_STEMS = (
    CORE_STEMS
    | LEGACY_REDUCED_STEMS
    | STEP_SUMMARY_STEMS
    | METRIC_STEMS
)
