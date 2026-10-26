# -*- coding: utf-8 -*-
from __future__ import annotations

"""
SSDR core utilities for OGD-LBS.

This package contains reusable algorithmic code for the SSDR pipeline.
CLI entrypoints live under scripts/ssdr/, while reusable pure-Python / NumPy /
SciPy implementation lives here.

Step 04 optimization modules are imported explicitly by callers to avoid
eagerly importing SciPy-heavy modules at package import time.
"""

__all__ = [
    "frame_sampling",
    "weight_reduction",
    "utils",
    "sparse_laplacian",
    "skinning",
    "skeleton_organization",
    "schedulers",
    "rotation_utils",
    "reconstruction_err",
    "optimize_weight",
    "optimize_bone_trans",
    "mesh_health",
    "convex_ls",
    "bone_health",
    "weight_mask",
]
