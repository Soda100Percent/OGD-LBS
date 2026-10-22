# -*- coding: utf-8 -*-
from __future__ import annotations

"""
SSDR core utilities for OGD-LBS.

This package contains reusable algorithmic code for the SSDR pipeline.
CLI entrypoints should live under scripts/ssdr/, while reusable pure-Python /
NumPy implementation lives here.
"""

__all__ = [
    "frame_sampling",
    "weight_reduction",
]
