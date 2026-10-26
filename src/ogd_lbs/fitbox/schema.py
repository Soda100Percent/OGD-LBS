# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Shared Fitbox key names.

This module is intentionally small.  It exists so SSDR-related modules can
depend on stable key names without importing Blender-specific Fitbox IO code.
"""

KEY_REST_POSE = "rest_pose"
KEY_ANIMATION_POSES = "animation_poses"
KEY_BONE_NAMES = "bone_names"
KEY_SCENE_FRAMES = "scene_frames"
KEY_FIXED_BONES = "fixed_bones"

KEY_VERTICES = "vertices"
KEY_FACES = "faces"
KEY_FACE_UVS = "face_uvs"
KEY_FACE_MATERIALS = "face_materials"
KEY_VERTEX_NORMALS = "vertex_normals"
KEY_MATERIAL_NAMES = "material_names"

KEY_BAKED_VERTICES = "baked_animation_vertices"
KEY_BAKED_NORMALS = "baked_animation_vertex_normals"
KEY_VERTEX_BONE_WEIGHTS = "vertex_bone_weights"
KEY_VERTEX_BONE_WEIGHTS_MASK = "vertex_bone_weights_mask"
