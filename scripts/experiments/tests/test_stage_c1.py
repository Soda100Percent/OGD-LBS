from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for path in [ROOT / "src", ROOT / "scripts/experiments", ROOT / "scripts/ssdr"]:
    sys.path.insert(0, str(path))

import stage_contracts as contracts
import stage_validation as validation
from ogd_lbs.utils.common import dump_pkl, load_pkl

spec = importlib.util.spec_from_file_location(
    "ogd_c1_actual_step02", ROOT / "scripts/ssdr/02_reduce_bone_weight.py"
)
assert spec is not None and spec.loader is not None
step02 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = step02
spec.loader.exec_module(step02)


def request_fixture():
    prefix = {
        "artifact_type": "ssdr_data1", "model": "M",
        "input_source_kind": "synthetic_fixture",
        "input_dataset_version": "fixture_v1",
        "frame_sampling_policy": "fixture_all",
    }
    alpha, beta, gamma = 0.5625, 0.3375, 0.1
    return {
        "schema": contracts.REQUEST_SCHEMA,
        "identity": {
            "pipeline_id": contracts.PIPELINE_ID,
            "stage_id": "step02",
            "run_id": "gamma-fixture",
            "task_id": "task-0",
            "attempt_id": "attempt-0",
            "resource_group_id": "intel",
        },
        "experiment": {
            "case_parameters": {
                "model": "M", "param": "gamma", "value": gamma,
                "effective_params": {"alpha": alpha, "beta": beta, "gamma": gamma},
            },
            "stage_parameters": {
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "affinity_eps": 1e-5,
                "variant": {"mode": "legacy_flat", "name": None},
                "extra_remove_bones": [],
            },
            "cumulative_dimensions": {
                **prefix, "artifact_type": "ssdr_data2",
                "bone_variant": "default",
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "step02_affinity_eps": 1e-5,
                "step02_code_version": "step02_v1",
            },
        },
        "artifacts": {
            "input_bindings": [{
                "artifact_id": "a000001", "artifact_type": "ssdr_data1",
                "payload_path": "/tmp/input-pool/artifacts/a000001/payload/data1",
                "payload_manifest_sha256": "a" * 64,
                "dimensions": prefix,
            }],
            "attempt_root": "/tmp/gamma-fixture/attempt-0",
            "output_directory": "/tmp/gamma-fixture/attempt-0/data2",
            "metrics_directory": "/tmp/gamma-fixture/attempt-0",
        },
        "execution": {
            "cpu_quota": 8, "internal_workers": 8,
            "memory_request_bytes": 18 * 1024**3,
            "memory_limit_bytes": 18 * 1024**3,
            "log_directory": "/tmp/gamma-fixture/attempt-0/logs",
            "resource_policy_revision": "v1a_user_decisions_20261103_r1",
        },
    }


def object_fixture():
    full = {"bone_names": ["root", "a", "b"], "parent_indices": [None, 0, 1]}
    reduced = {"bone_names": ["root", "b"], "parent_indices": [None, 0]}
    rest = np.tile(np.eye(4), (3, 1, 1))
    rest[:, 1, 3] = [0, 1, 2]
    poses = np.tile(rest, (4, 1, 1, 1))
    for frame in range(4):
        poses[frame, :, 0, 3] += frame * 0.1
    full_anim = {
        "bone_names": full["bone_names"],
        "rest_pose": rest,
        "animation_poses": poses,
        "scene_frames": np.arange(100, 104, dtype=np.int64),
    }
    reduced_anim = {
        "bone_names": reduced["bone_names"],
        "rest_pose": rest[[0, 2]].copy(),
        "animation_poses": poses[:, [0, 2]].copy(),
        "scene_frames": full_anim["scene_frames"].copy(),
    }
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    normals = np.tile([0., 0., 1.], (3, 1))
    mesh = {
        "name": "body",
        "vertices": vertices,
        "faces": np.array([[0, 1, 2]], dtype=np.int64),
        "vertex_normals": normals,
        "bone_names": list(full["bone_names"]),
        "vertex_bone_weights": np.array([
            [1., 0., 0.], [0., 1., 0.], [0.2, 0.5, 0.3],
        ]),
        "baked_animation_vertices": np.stack([
            vertices + [frame * 0.1, 0, 0] for frame in range(4)
        ]),
        "baked_animation_vertex_normals": np.tile(normals, (4, 1, 1)),
        "scene_frames": full_anim["scene_frames"].copy(),
    }
    return full, full_anim, reduced, reduced_anim, {"body.pkl": mesh}


class TestStep02RequestContract(unittest.TestCase):
    def test_valid_request_is_copied_without_mutation(self):
        req = request_fixture()
        result = contracts.validate_step02_request(req)
        result["execution"]["internal_workers"] = 1
        self.assertEqual(req["execution"]["internal_workers"], 8)

    def test_gamma_five_points_keep_sum_one(self):
        for gamma in [0.1, 0.3, 0.5, 0.7, 0.9]:
            req = request_fixture()
            values = {"alpha": (1-gamma)*5/8, "beta": (1-gamma)*3/8, "gamma": gamma}
            exp = req["experiment"]
            exp["case_parameters"]["value"] = gamma
            exp["case_parameters"]["effective_params"].update(values)
            exp["stage_parameters"].update(values)
            exp["cumulative_dimensions"].update(values)
            contracts.validate_step02_request(req)

    def test_nonfinite_or_boolean_coefficient_rejected(self):
        for value in [float("nan"), float("inf"), True, -0.1]:
            req = request_fixture()
            req["experiment"]["stage_parameters"]["alpha"] = value
            with self.subTest(value=value), self.assertRaises(contracts.StageContractError):
                contracts.validate_step02_request(req)

    def test_parameter_dimension_mismatch_rejected(self):
        req = request_fixture()
        req["experiment"]["cumulative_dimensions"]["gamma"] = 0.9
        with self.assertRaises(contracts.StageContractError):
            contracts.validate_step02_request(req)

    def test_auto_variant_rejected(self):
        req = request_fixture()
        req["experiment"]["stage_parameters"]["variant"]["mode"] = "auto"
        with self.assertRaises(contracts.StageContractError):
            contracts.validate_step02_request(req)

    def test_named_variant_supported(self):
        req = request_fixture()
        req["experiment"]["stage_parameters"]["variant"] = {
            "mode": "named", "name": "bone_reduced_conservative",
        }
        req["experiment"]["cumulative_dimensions"]["bone_variant"] = "bone_reduced_conservative"
        contracts.validate_step02_request(req)

    def test_extra_removal_rejected(self):
        req = request_fixture()
        req["experiment"]["stage_parameters"]["extra_remove_bones"] = ["a"]
        with self.assertRaises(contracts.StageContractError):
            contracts.validate_step02_request(req)

    def test_ancestor_output_overlap_rejected(self):
        req = request_fixture()
        attempt = "/tmp/input-pool"
        req["artifacts"].update({
            "attempt_root": attempt, "output_directory": attempt + "/data2",
            "metrics_directory": attempt,
        })
        req["execution"]["log_directory"] = attempt + "/logs"
        with self.assertRaises(contracts.StageContractError):
            contracts.validate_step02_request(req)

    def test_missing_parameter_has_no_silent_default(self):
        req = request_fixture()
        del req["experiment"]["stage_parameters"]["affinity_eps"]
        with self.assertRaises(contracts.StageContractError):
            contracts.validate_step02_request(req)

    def test_invalid_execution_fields_rejected(self):
        for key, value in [("internal_workers", 0), ("internal_workers", True),
                           ("memory_limit_bytes", 1)]:
            req = request_fixture()
            req["execution"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(contracts.StageContractError):
                contracts.validate_step02_request(req)


class TestStep02StrictInputs(unittest.TestCase):
    def setUp(self):
        self.full, self.anim, self.reduced, self.ranim, self.meshes = object_fixture()

    def validate(self):
        return validation.validate_step02_inputs(
            full_skeleton=self.full, full_animation=self.anim,
            reduced_skeleton=self.reduced, reduced_animation=self.ranim,
            meshes=self.meshes,
        )

    def test_valid_input_and_reordered_skeleton(self):
        self.full = {"bone_names": ["a", "root", "b"], "parent_indices": [1, None, 0]}
        result = self.validate()
        self.assertEqual(result["full_bone_names"], ["root", "a", "b"])
        self.assertFalse(result["repair_performed"])

    def test_duplicate_bones_rejected(self):
        self.full["bone_names"] = ["root", "root", "b"]
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_cycles_and_invalid_parent_indices_rejected(self):
        for parents in [[2, 0, 1], [None, 7, 0], [None, 0.2, 0], [None, True, 0]]:
            self.full["parent_indices"] = parents
            with self.subTest(parents=parents), self.assertRaises(validation.StageValidationError):
                self.validate()

    def test_zero_frames_rejected(self):
        self.anim["animation_poses"] = self.anim["animation_poses"][:0]
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_singular_rest_pose_rejected(self):
        self.anim["rest_pose"][0] = 0
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_missing_required_mesh_fields_rejected(self):
        del self.meshes["body.pkl"]["baked_animation_vertex_normals"]
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_zero_weight_row_rejected(self):
        self.meshes["body.pkl"]["vertex_bone_weights"][0] = 0
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_negative_weights_rejected_even_when_sum_one(self):
        self.meshes["body.pkl"]["vertex_bone_weights"][0] = [-0.1, 0.1, 1.0]
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_fractional_faces_rejected_without_truncation(self):
        self.meshes["body.pkl"]["faces"] = np.array([[0., 1.5, 2.]])
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_scene_frame_mismatch_rejected(self):
        self.meshes["body.pkl"]["scene_frames"] += 1
        with self.assertRaises(validation.StageValidationError):
            self.validate()


class TestActualStep02Validation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.input, self.output = root / "data1", root / "data2"
        self.input.mkdir()
        self.full, self.anim, self.reduced, self.ranim, self.meshes = object_fixture()
        for name, obj in [
            ("skeleton_hierarchy_data.pkl", self.full),
            ("animation_data.pkl", self.anim),
            ("skeleton_hierarchy_data_bone_reduced.pkl", self.reduced),
            ("animation_data_bone_reduced.pkl", self.ranim),
            ("body.pkl", self.meshes["body.pkl"]),
        ]:
            dump_pkl(self.input / name, obj)
        self.parameters = {"alpha": 0.5625, "beta": 0.3375, "gamma": 0.1, "affinity_eps": 1e-5}
        self.summary = step02.reduce_bone_weight_tree(
            input_tree=self.input, output_tree=self.output,
            variant=None,
            par_weight=self.parameters["alpha"],
            bone_weight=self.parameters["beta"],
            child_weight=self.parameters["gamma"],
            affinity_eps=self.parameters["affinity_eps"],
            max_workers=1, overwrite_output=False, allow_merge=False,
            write_summary_file=True,
            extra_remove_bones=[],
        )
        self.oskel = load_pkl(self.output / "skeleton_hierarchy_data.pkl")
        self.oanim = load_pkl(self.output / "animation_data.pkl")
        self.omeshes = {"body.pkl": load_pkl(self.output / "body.pkl")}

    def validate(self):
        return validation.validate_step02_outputs(
            full_skeleton=self.full, full_animation=self.anim,
            reduced_skeleton=self.reduced, reduced_animation=self.ranim,
            input_meshes=self.meshes,
            output_skeleton=self.oskel, output_animation=self.oanim,
            output_meshes=self.omeshes,
            summary=self.summary, expected_parameters=self.parameters,
            expected_variant="legacy_bone_reduced",
        )

    def test_actual_algorithm_output_passes(self):
        result = self.validate()
        self.assertEqual(result["frames"], 4)
        self.assertEqual(result["bones"], 2)
        self.assertEqual(result["mesh_files"], ["body.pkl"])
        self.assertFalse(result["ingest_performed"])

    def test_changed_geometry_rejected(self):
        self.omeshes["body.pkl"]["vertices"][0, 0] += 1
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_wrong_animation_subset_rejected(self):
        self.oanim["animation_poses"][0, 0, 0, 3] += 1
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_missing_mesh_rejected(self):
        self.omeshes = {}
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_wrong_summary_parameters_rejected(self):
        self.summary["weights"]["child_weight"] = 0.9
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_stale_weight_mask_axis_rejected(self):
        self.omeshes["body.pkl"]["vertex_bone_weights_mask"] = np.ones((3, 3))
        with self.assertRaises(validation.StageValidationError):
            self.validate()

    def test_invalid_affinity_summary_rejected(self):
        self.summary["total_invalid_affinity_count"] = 1
        with self.assertRaises(validation.StageValidationError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
