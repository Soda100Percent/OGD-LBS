# -*- coding: utf-8 -*-
"""
Winter Samurai FBX cleaning utilities.

These scripts are asset-specific preprocessing / diagnostics tools for
normalizing the commercial Winter Samurai FBX asset before it enters the formal
OGD-LBS data_prep staged pipeline.

Current planned stages:
    01_import_model_and_anims.py
        Import model FBX and one/multiple animation FBXs, then remove EMPTY
        objects while preserving surviving objects' world transforms as much as
        possible.

    02_...
        Rebase imported animation armatures to the skinning armature rest
        pose/hierarchy, bake clean Actions, and remove imported animation
        armatures plus pre-rebase Actions.
"""
