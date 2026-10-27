#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def ensure_repo_src_in_syspath() -> Path:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src_dir = parent / "src"
            ssdr_dir = parent / "scripts" / "ssdr"

            for item in [src_dir, ssdr_dir]:
                item_s = str(item)
                if item_s not in sys.path:
                    sys.path.insert(0, item_s)

            return parent

    raise RuntimeError("Could not locate repository root.")


REPO_ROOT = ensure_repo_src_in_syspath()

from ogd_lbs.utils.common import load_pkl, dump_pkl  # noqa: E402


def load_step02_module():
    path = REPO_ROOT / "scripts" / "ssdr" / "02_reduce_bone_weight.py"
    spec = importlib.util.spec_from_file_location("ogd_lbs_ssdr_step02", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_skeleton(bone_names: list[str], parent_indices: list[int | None]) -> dict:
    return {
        "bone_names": list(bone_names),
        "parent_indices": list(parent_indices),
    }


def make_animation(
    bone_names: list[str],
    *,
    frame_count: int = 4,
    scene_start: int = 100,
    fill_offset: float = 0.0,
) -> dict:
    bone_count = len(bone_names)

    rest_pose = np.zeros((bone_count, 4, 4), dtype=np.float64)
    animation_poses = np.zeros((frame_count, bone_count, 4, 4), dtype=np.float64)

    for f in range(frame_count):
        for b in range(bone_count):
            rest_pose[b] = np.eye(4, dtype=np.float64) * (b + 1 + fill_offset)
            animation_poses[f, b] = np.eye(4, dtype=np.float64) * (1000 + f * 10 + b + fill_offset)

    return {
        "bone_names": list(bone_names),
        "rest_pose": rest_pose,
        "animation_poses": animation_poses,
        "scene_frames": np.arange(scene_start, scene_start + frame_count, dtype=np.int64),
        "rest_pose_semantics": "test",
    }


def make_mesh(
    *,
    name: str = "main_mesh_VDB",
    bone_names: list[str] | None = None,
) -> dict:
    if bone_names is None:
        bone_names = ["root", "a", "b"]

    vertex_count = 4
    bone_count = len(bone_names)

    vbw_by_full = {
        "root": np.array([1.0, 0.0, 0.0, 0.2], dtype=np.float64),
        "a": np.array([0.0, 1.0, 0.0, 0.5], dtype=np.float64),
        "b": np.array([0.0, 0.0, 1.0, 0.3], dtype=np.float64),
    }

    vbw = np.zeros((vertex_count, bone_count), dtype=np.float64)
    for j, bname in enumerate(bone_names):
        vbw[:, j] = vbw_by_full[bname]

    return {
        "name": name,
        "vertices": np.zeros((vertex_count, 3), dtype=np.float64),
        "faces": [[0, 1, 2]],
        "bone_names": list(bone_names),
        "vertex_bone_weights": vbw,
        "baked_animation_vertices": np.zeros((4, vertex_count, 3), dtype=np.float64),
        "baked_animation_vertex_normals": np.ones((4, vertex_count, 3), dtype=np.float64),
    }


def write_legacy_flat_dataset(data1: Path) -> None:
    data1.mkdir(parents=True, exist_ok=True)

    full_bones = ["root", "a", "b"]
    reduced_bones = ["root", "b"]

    dump_pkl(data1 / "skeleton_hierarchy_data.pkl", make_skeleton(full_bones, [None, 0, 1]))
    dump_pkl(data1 / "animation_data.pkl", make_animation(full_bones))
    dump_pkl(data1 / "skeleton_hierarchy_data_bone_reduced.pkl", make_skeleton(reduced_bones, [None, 0]))
    dump_pkl(data1 / "animation_data_bone_reduced.pkl", make_animation(reduced_bones, fill_offset=500.0))
    dump_pkl(data1 / "body.pkl", make_mesh(name="body", bone_names=full_bones))


def write_new_vdb_root_dataset(data1: Path) -> None:
    data1.mkdir(parents=True, exist_ok=True)

    full_bones = ["root", "a", "b"]

    dump_pkl(data1 / "skeleton_hierarchy_data.pkl", make_skeleton(full_bones, [None, 0, 1]))
    dump_pkl(data1 / "animation_data.pkl", make_animation(full_bones))
    dump_pkl(data1 / "main_mesh_VDB.pkl", make_mesh(name="main_mesh_VDB", bone_names=full_bones))

    variants = {
        "cleaned_full": ["root", "a", "b"],
        "bone_reduced_002": ["root", "a"],
        "bone_reduced_001": ["root", "b"],
    }

    for vname, bones in variants.items():
        vdir = data1 / "bone_reduced_versions" / vname
        vdir.mkdir(parents=True, exist_ok=True)

        if bones == ["root", "a", "b"]:
            parents = [None, 0, 1]
        else:
            parents = [None, 0]

        dump_pkl(vdir / "skeleton_hierarchy_data.pkl", make_skeleton(bones, parents))
        dump_pkl(vdir / "animation_data.pkl", make_animation(bones, fill_offset=100.0 + len(bones)))


class TestSSDRStep02ReduceBoneWeight(unittest.TestCase):
    def test_legacy_flat_layout_reduces_to_data2(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_") as td:
            root = Path(td)
            data1 = root / "data1"
            data2 = root / "data2"

            write_legacy_flat_dataset(data1)

            summary = mod.reduce_bone_weight_tree(
                input_tree=data1,
                output_tree=data2,
                variant=None,
                par_weight=0.5,
                bone_weight=0.3,
                child_weight=0.2,
                affinity_eps=1e-5,
                max_workers=1,
                overwrite_output=False,
                allow_merge=False,
                write_summary_file=True,
            )

            self.assertEqual(summary["layout"]["layout_kind"], "legacy_flat")
            self.assertEqual(summary["layout"]["variant_name"], "legacy_bone_reduced")
            self.assertEqual(summary["skeleton"]["full_bone_count"], 3)
            self.assertEqual(summary["skeleton"]["reduced_bone_count"], 2)
            self.assertEqual(summary["reduced_mesh_count"], 1)

            out_skel = load_pkl(data2 / "skeleton_hierarchy_data.pkl")
            out_anim = load_pkl(data2 / "animation_data.pkl")
            out_mesh = load_pkl(data2 / "body.pkl")

            self.assertEqual(out_skel["bone_names"], ["root", "b"])
            self.assertEqual(out_anim["bone_names"], ["root", "b"])
            self.assertEqual(out_anim["animation_poses"].shape, (4, 2, 4, 4))
            self.assertTrue(np.array_equal(out_anim["scene_frames"], np.arange(100, 104)))

            self.assertEqual(out_mesh["bone_names"], ["root", "b"])
            self.assertEqual(out_mesh["vertex_bone_weights"].shape, (4, 2))
            self.assertTrue(np.allclose(out_mesh["vertex_bone_weights"].sum(axis=1), 1.0, atol=1e-8))
            self.assertTrue((data2 / "step02_reduce_bone_weight_summary.json").is_file())

    def test_new_vdb_root_auto_selects_smallest_non_cleaned_full(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_") as td:
            root = Path(td)
            data1 = root / "data1"
            data2 = root / "data2"

            write_new_vdb_root_dataset(data1)

            summary = mod.reduce_bone_weight_tree(
                input_tree=data1,
                output_tree=data2,
                variant=None,
                par_weight=0.5,
                bone_weight=0.3,
                child_weight=0.2,
                affinity_eps=1e-5,
                max_workers=1,
                overwrite_output=False,
                allow_merge=False,
                write_summary_file=False,
            )

            self.assertEqual(summary["layout"]["layout_kind"], "new_vdb_root_auto")
            self.assertEqual(summary["layout"]["variant_name"], "bone_reduced_001")

            out_skel = load_pkl(data2 / "skeleton_hierarchy_data.pkl")
            out_mesh = load_pkl(data2 / "main_mesh_VDB.pkl")

            self.assertEqual(out_skel["bone_names"], ["root", "b"])
            self.assertEqual(out_mesh["bone_names"], ["root", "b"])
            self.assertEqual(out_mesh["vertex_bone_weights"].shape, (4, 2))
            self.assertTrue(np.allclose(out_mesh["vertex_bone_weights"].sum(axis=1), 1.0, atol=1e-8))
            self.assertFalse((data2 / "bone_reduced_versions").exists())

    def test_new_vdb_root_explicit_variant(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_") as td:
            root = Path(td)
            data1 = root / "data1"
            data2 = root / "data2"

            write_new_vdb_root_dataset(data1)

            summary = mod.reduce_bone_weight_tree(
                input_tree=data1,
                output_tree=data2,
                variant="bone_reduced_002",
                par_weight=0.5,
                bone_weight=0.3,
                child_weight=0.2,
                affinity_eps=1e-5,
                max_workers=1,
                overwrite_output=False,
                allow_merge=False,
                write_summary_file=False,
            )

            self.assertEqual(summary["layout"]["layout_kind"], "new_vdb_root_explicit")
            self.assertEqual(summary["layout"]["variant_name"], "bone_reduced_002")

            out_skel = load_pkl(data2 / "skeleton_hierarchy_data.pkl")
            out_mesh = load_pkl(data2 / "main_mesh_VDB.pkl")

            self.assertEqual(out_skel["bone_names"], ["root", "a"])
            self.assertEqual(out_mesh["bone_names"], ["root", "a"])
            self.assertEqual(out_mesh["vertex_bone_weights"].shape, (4, 2))

    def test_cleaned_full_selected_only_when_it_is_the_only_variant(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_") as td:
            root = Path(td)
            data1 = root / "data1"
            data2 = root / "data2"
            data1.mkdir(parents=True)

            full_bones = ["root", "a", "b"]

            dump_pkl(data1 / "skeleton_hierarchy_data.pkl", make_skeleton(full_bones, [None, 0, 1]))
            dump_pkl(data1 / "animation_data.pkl", make_animation(full_bones))
            dump_pkl(data1 / "main_mesh_VDB.pkl", make_mesh(name="main_mesh_VDB", bone_names=full_bones))

            cleaned = data1 / "bone_reduced_versions" / "cleaned_full"
            cleaned.mkdir(parents=True)
            dump_pkl(cleaned / "skeleton_hierarchy_data.pkl", make_skeleton(full_bones, [None, 0, 1]))
            dump_pkl(cleaned / "animation_data.pkl", make_animation(full_bones))

            summary = mod.reduce_bone_weight_tree(
                input_tree=data1,
                output_tree=data2,
                variant=None,
                par_weight=0.5,
                bone_weight=0.3,
                child_weight=0.2,
                affinity_eps=1e-5,
                max_workers=1,
                overwrite_output=False,
                allow_merge=False,
                write_summary_file=False,
            )

            self.assertEqual(summary["layout"]["variant_name"], "cleaned_full")

            out_skel = load_pkl(data2 / "skeleton_hierarchy_data.pkl")
            self.assertEqual(out_skel["bone_names"], full_bones)


    def test_new_vdb_root_extra_remove_middle_bone_repairs_parent(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_extra_remove_") as td:
            root = Path(td)
            data1 = root / "data1"
            data2 = root / "data2"

            write_new_vdb_root_dataset(data1)

            summary = mod.reduce_bone_weight_tree(
                input_tree=data1,
                output_tree=data2,
                variant="cleaned_full",
                par_weight=0.5,
                bone_weight=0.3,
                child_weight=0.2,
                affinity_eps=1e-5,
                max_workers=1,
                overwrite_output=False,
                allow_merge=False,
                write_summary_file=True,
                extra_remove_bones=["a"],
                extra_variant_name="cleaned_full_minus_a",
                extra_remove_missing_ok=False,
            )

            self.assertEqual(summary["layout"]["variant_name"], "cleaned_full")
            self.assertEqual(summary["layout"]["effective_variant_name"], "cleaned_full_minus_a")
            self.assertTrue(summary["layout"]["extra_remove_bones_enabled"])
            self.assertEqual(summary["extra_remove_bones"]["removed_bones"], ["a"])
            self.assertEqual(summary["extra_remove_bones"]["base_bone_count"], 3)
            self.assertEqual(summary["extra_remove_bones"]["target_bone_count"], 2)

            out_skel = load_pkl(data2 / "skeleton_hierarchy_data.pkl")
            out_anim = load_pkl(data2 / "animation_data.pkl")
            out_mesh = load_pkl(data2 / "main_mesh_VDB.pkl")

            self.assertEqual(out_skel["bone_names"], ["root", "b"])
            # Original chain root -> a -> b. After removing a, b is reparented to root.
            self.assertEqual(out_skel["parent_indices"], [None, 0])

            self.assertEqual(out_anim["bone_names"], ["root", "b"])
            self.assertEqual(out_anim["animation_poses"].shape, (4, 2, 4, 4))
            self.assertEqual(out_anim["rest_pose"].shape, (2, 4, 4))
            self.assertEqual(out_anim["ssdr_step02_variant"]["variant_name"], "cleaned_full_minus_a")

            self.assertEqual(out_mesh["bone_names"], ["root", "b"])
            self.assertEqual(out_mesh["vertex_bone_weights"].shape, (4, 2))
            self.assertTrue(np.allclose(out_mesh["vertex_bone_weights"].sum(axis=1), 1.0, atol=1e-8))
            self.assertTrue((data2 / "step02_reduce_bone_weight_summary.json").is_file())

    def test_new_vdb_root_extra_remove_leaf_bone(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_extra_remove_") as td:
            root = Path(td)
            data1 = root / "data1"
            data2 = root / "data2"

            write_new_vdb_root_dataset(data1)

            summary = mod.reduce_bone_weight_tree(
                input_tree=data1,
                output_tree=data2,
                variant="cleaned_full",
                par_weight=0.5,
                bone_weight=0.3,
                child_weight=0.2,
                affinity_eps=1e-5,
                max_workers=1,
                overwrite_output=False,
                allow_merge=False,
                write_summary_file=False,
                extra_remove_bones=["b"],
                extra_variant_name="cleaned_full_minus_b",
                extra_remove_missing_ok=False,
            )

            self.assertEqual(summary["layout"]["effective_variant_name"], "cleaned_full_minus_b")
            self.assertEqual(summary["extra_remove_bones"]["removed_bones"], ["b"])

            out_skel = load_pkl(data2 / "skeleton_hierarchy_data.pkl")
            out_anim = load_pkl(data2 / "animation_data.pkl")
            out_mesh = load_pkl(data2 / "main_mesh_VDB.pkl")

            self.assertEqual(out_skel["bone_names"], ["root", "a"])
            self.assertEqual(out_skel["parent_indices"], [None, 0])
            self.assertEqual(out_anim["bone_names"], ["root", "a"])
            self.assertEqual(out_mesh["bone_names"], ["root", "a"])
            self.assertEqual(out_mesh["vertex_bone_weights"].shape, (4, 2))
            self.assertTrue(np.allclose(out_mesh["vertex_bone_weights"].sum(axis=1), 1.0, atol=1e-8))

    def test_new_vdb_root_extra_remove_missing_bone_fails_by_default(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_extra_remove_") as td:
            root = Path(td)
            data1 = root / "data1"
            data2 = root / "data2"

            write_new_vdb_root_dataset(data1)

            with self.assertRaises(RuntimeError):
                mod.reduce_bone_weight_tree(
                    input_tree=data1,
                    output_tree=data2,
                    variant="cleaned_full",
                    par_weight=0.5,
                    bone_weight=0.3,
                    child_weight=0.2,
                    affinity_eps=1e-5,
                    max_workers=1,
                    overwrite_output=False,
                    allow_merge=False,
                    write_summary_file=False,
                    extra_remove_bones=["missing_bone"],
                    extra_variant_name="bad_missing",
                    extra_remove_missing_ok=False,
                )

    def test_new_vdb_root_extra_remove_missing_bone_ok(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_extra_remove_") as td:
            root = Path(td)
            data1 = root / "data1"
            data2 = root / "data2"

            write_new_vdb_root_dataset(data1)

            summary = mod.reduce_bone_weight_tree(
                input_tree=data1,
                output_tree=data2,
                variant="cleaned_full",
                par_weight=0.5,
                bone_weight=0.3,
                child_weight=0.2,
                affinity_eps=1e-5,
                max_workers=1,
                overwrite_output=False,
                allow_merge=False,
                write_summary_file=False,
                extra_remove_bones=["missing_bone", "a"],
                extra_variant_name="missing_ok_minus_a",
                extra_remove_missing_ok=True,
            )

            self.assertEqual(summary["extra_remove_bones"]["missing_bones"], ["missing_bone"])
            self.assertEqual(summary["extra_remove_bones"]["removed_bones"], ["a"])
            self.assertEqual(summary["skeleton"]["reduced_bone_count"], 2)

    def test_wrapper_fitbox_vdb_layout_is_intentionally_not_supported(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_") as td:
            root = Path(td)
            data1 = root / "data1"
            fitbox_vdb = data1 / "fitbox_vdb"
            data2 = root / "data2"

            write_new_vdb_root_dataset(fitbox_vdb)

            with self.assertRaises(RuntimeError):
                mod.reduce_bone_weight_tree(
                    input_tree=data1,
                    output_tree=data2,
                    variant=None,
                    par_weight=0.5,
                    bone_weight=0.3,
                    child_weight=0.2,
                    affinity_eps=1e-5,
                    max_workers=1,
                    overwrite_output=False,
                    allow_merge=False,
                    write_summary_file=False,
                )

    def test_existing_non_empty_output_fails_without_overwrite_or_merge(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_") as td:
            root = Path(td)
            data1 = root / "data1"
            data2 = root / "data2"

            write_legacy_flat_dataset(data1)
            data2.mkdir()
            (data2 / "stale.pkl").write_bytes(b"stale")

            with self.assertRaises(RuntimeError):
                mod.reduce_bone_weight_tree(
                    input_tree=data1,
                    output_tree=data2,
                    variant=None,
                    par_weight=0.5,
                    bone_weight=0.3,
                    child_weight=0.2,
                    affinity_eps=1e-5,
                    max_workers=1,
                    overwrite_output=False,
                    allow_merge=False,
                    write_summary_file=False,
                )

    def test_overwrite_output_allows_rerun(self) -> None:
        mod = load_step02_module()

        with tempfile.TemporaryDirectory(prefix="ogd_lbs_ssdr_step02_") as td:
            root = Path(td)
            data1 = root / "data1"
            data2 = root / "data2"

            write_legacy_flat_dataset(data1)
            data2.mkdir()
            (data2 / "stale.pkl").write_bytes(b"stale")

            summary = mod.reduce_bone_weight_tree(
                input_tree=data1,
                output_tree=data2,
                variant=None,
                par_weight=0.5,
                bone_weight=0.3,
                child_weight=0.2,
                affinity_eps=1e-5,
                max_workers=1,
                overwrite_output=True,
                allow_merge=False,
                write_summary_file=False,
            )

            self.assertEqual(summary["reduced_mesh_count"], 1)
            self.assertFalse((data2 / "stale.pkl").exists())
            self.assertTrue((data2 / "body.pkl").is_file())


if __name__ == "__main__":
    unittest.main()
