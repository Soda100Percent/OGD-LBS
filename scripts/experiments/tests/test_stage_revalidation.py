from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stage_file_io as io
import stage_revalidation as revalidation


class TestStageRevalidation(unittest.TestCase):
    def setUp(self):
        f = json.loads(Path(os.environ["OGD_REVALIDATION_FIXTURE"]).read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.output = root / "output"
        shutil.copytree(f["input_root"], self.source)
        shutil.copytree(f["collected_root"], self.output)
        self.request = Path(f["request_path"]).read_bytes()
        self.manifest = Path(f["input_manifest_path"]).read_bytes()

    def validate(self, trusted=True):
        return revalidation.revalidate_step02(
            request_bytes=self.request,
            input_manifest_bytes=self.manifest,
            input_root=self.source,
            collected_root=self.output,
            trusted_input_pickle=trusted,
        )

    def rehash_output(self):
        p = self.output / "artifact_files.json"
        manifest = io.strict_json(p.read_bytes())
        for rel in manifest["files"]:
            manifest["files"][rel] = io.fingerprint(self.output / rel)
        p.write_bytes(io.encoded(manifest))
        digest = io.bytes_hash(p.read_bytes())
        for name in ["stage_result.json", "done"]:
            path = self.output / name
            obj = io.strict_json(path.read_bytes())
            obj["artifact_files_sha256"] = digest
            path.write_bytes(io.encoded(obj))

    def test_actual_collected_result_passes_independent_validation(self):
        before_source = io.tree_manifest(self.source)
        before_output = io.tree_manifest(self.output)
        result = self.validate()
        self.assertEqual(result["status"], "BUSINESS_REVALIDATED")
        self.assertTrue(result["independent_business_validation_performed"])
        self.assertFalse(result["ingest_permitted"])
        self.assertFalse(result["weight_solution_recomputed"])
        self.assertEqual(io.tree_manifest(self.source), before_source)
        self.assertEqual(io.tree_manifest(self.output), before_output)

    def test_trust_acknowledgement_required_before_unpickle(self):
        with patch.object(revalidation.pickle, "loads") as loads:
            with self.assertRaises(ValueError):
                self.validate(trusted=False)
        loads.assert_not_called()

    def test_input_tamper_rejected_before_unpickle(self):
        path = self.source / "animation_data.pkl"
        path.write_bytes(path.read_bytes() + b"tampered")
        with patch.object(revalidation.pickle, "loads") as loads:
            with self.assertRaises(ValueError):
                self.validate()
        loads.assert_not_called()

    def test_output_tamper_rejected_before_unpickle(self):
        path = self.output / "data2/animation_data.pkl"
        path.write_bytes(path.read_bytes() + b"tampered")
        with patch.object(revalidation.pickle, "loads") as loads:
            with self.assertRaises(ValueError):
                self.validate()
        loads.assert_not_called()

    def test_rehashed_geometry_change_is_still_rejected(self):
        validated = io.strict_json((self.output / "stage_validation.json").read_bytes())
        path = self.output / "data2" / validated["mesh_files"][0]
        obj = pickle.loads(path.read_bytes())
        obj["vertices"] = obj["vertices"].copy()
        obj["vertices"][0, 0] += 0.25
        path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        self.rehash_output()
        with self.assertRaisesRegex(ValueError, "preserved field changed"):
            self.validate()

    def test_rehashed_wrong_summary_parameter_rejected(self):
        path = self.output / "data2/step02_reduce_bone_weight_summary.json"
        obj = io.strict_json(path.read_bytes())
        obj["weights"]["child_weight"] = 0.99
        path.write_bytes(io.encoded(obj))
        self.rehash_output()
        with self.assertRaisesRegex(ValueError, "parameter mismatch"):
            self.validate()

    def test_symlink_output_rejected(self):
        path = self.output / "data2/animation_data.pkl"
        path.unlink()
        path.symlink_to(self.source / "animation_data.pkl")
        with self.assertRaises(ValueError):
            self.validate()

    def test_unexpected_collected_file_rejected(self):
        (self.output / "unexpected.txt").write_text("stale")
        with self.assertRaisesRegex(ValueError, "undeclared"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
