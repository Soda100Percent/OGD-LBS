import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import result_pool as rp
import result_pool_artifacts as rpa
import stage_file_io as io
import stage_ingest_sandbox as ingest


def make_pool(root):
    pool = root / "pool"
    rp.init_pool(pool)
    rpa.ensure_artifact_profiles(pool)
    io.atomic_json(pool / ingest.MARKER, {
        "schema": ingest.MARKER_SCHEMA,
        "pool": str(pool),
        "sandbox_root": str(root),
        "production_use_allowed": False,
    })
    return pool


class TestStageIngestSandbox(unittest.TestCase):
    def setUp(self):
        fixture = json.loads(Path(os.environ["OGD_REVALIDATION_FIXTURE"]).read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pool = make_pool(self.root)
        self.source = self.root / "source"
        self.output = self.root / "collected"
        shutil.copytree(fixture["input_root"], self.source)
        shutil.copytree(fixture["collected_root"], self.output)
        self.request = Path(fixture["request_path"]).read_bytes()
        self.manifest = Path(fixture["input_manifest_path"]).read_bytes()
        self.defaults = copy.deepcopy(rpa.SSDR_DATA2_DEFAULTS)

    def call(self, **kwargs):
        args = {
            "pool": self.pool, "sandbox_root": self.root,
            "request_bytes": self.request, "input_manifest_bytes": self.manifest,
            "input_root": self.source, "collected_root": self.output,
            "approved_defaults": self.defaults, "trusted_input_pickle": True,
        }
        args.update(kwargs)
        return ingest.ingest_validated_step02(**args)

    def index(self):
        return json.loads((self.pool / "index.json").read_text())

    def test_complete_payload_and_provenance_ingested(self):
        result = self.call()
        self.assertEqual(result["status"], "INGESTED")
        artifact = self.pool / "artifacts" / result["artifact_id"]
        self.assertEqual(io.tree_manifest(artifact / "payload/data2"),
                         io.tree_manifest(self.output / "data2"))
        self.assertTrue((artifact / "business_revalidation.json").is_file())
        self.assertTrue((artifact / "evidence/request.json").is_file())
        self.assertEqual(len(self.index()["artifacts"]), 1)

    def test_repeat_is_idempotent(self):
        first = self.call()
        before = self.index()
        second = self.call()
        self.assertEqual(second["status"], "REUSED")
        self.assertEqual(first["artifact_id"], second["artifact_id"])
        self.assertEqual(self.index(), before)

    def test_missing_sandbox_marker_rejected(self):
        (self.pool / ingest.MARKER).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.call()
        self.assertFalse(self.index()["artifacts"])

    def test_unapproved_missing_dimension_rejected(self):
        with self.assertRaisesRegex(ValueError, "No explicit"):
            self.call(approved_defaults={})
        self.assertFalse(self.index()["artifacts"])

    def test_output_tamper_does_not_register_artifact(self):
        p = self.output / "data2/animation_data.pkl"
        p.write_bytes(p.read_bytes() + b"bad")
        with self.assertRaises(ValueError):
            self.call()
        self.assertFalse(self.index()["artifacts"])

    def test_corrupt_existing_artifact_is_not_reused(self):
        result = self.call()
        p = self.pool / "artifacts" / result["artifact_id"] / "payload/data2/animation_data.pkl"
        p.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            self.call()

    def test_same_dimensions_different_content_is_conflict(self):
        self.call()
        # Diagnostics change, then hashes are updated: data2 remains valid,
        # but the immutable produced artifact is not identical.
        p = self.output / "metrics_02.json"
        metrics = io.strict_json(p.read_bytes())
        metrics["sandbox_extra_diagnostic"] = 1
        p.write_bytes(io.encoded(metrics))
        ap = self.output / "artifact_files.json"
        artifact = io.strict_json(ap.read_bytes())
        artifact["files"]["metrics_02.json"] = io.fingerprint(p)
        ap.write_bytes(io.encoded(artifact))
        digest = io.bytes_hash(ap.read_bytes())
        for name in ["stage_result.json", "done"]:
            p = self.output / name
            obj = io.strict_json(p.read_bytes())
            obj["artifact_files_sha256"] = digest
            p.write_bytes(io.encoded(obj))
        with self.assertRaises(ingest.IngestConflict):
            self.call()
        self.assertEqual(len(self.index()["artifacts"]), 1)

    def test_index_failure_then_explicit_retry_recovers(self):
        with patch.object(ingest, "_write_index", side_effect=OSError("injected index failure")):
            with self.assertRaises(OSError):
                self.call()
        self.assertEqual(len(self.index()["artifacts"]), 0)
        self.assertEqual(len(list((self.pool / "artifacts").iterdir())), 1)
        result = self.call()
        self.assertEqual(result["status"], "RECOVERED")
        self.assertEqual(len(self.index()["artifacts"]), 1)

    def test_write_lock_blocks_second_writer(self):
        with ingest.pool_lock(self.pool):
            with self.assertRaises(RuntimeError):
                self.call()
        self.assertFalse(self.index()["artifacts"])

    def test_existing_lookup_api_finds_ingested_dimensions(self):
        result = self.call()
        raw = json.loads((
            self.pool / "artifacts" / result["artifact_id"] / "raw_dimensions.json"
        ).read_text())["dimensions"]
        hits = rp.lookup_artifacts(pool=self.pool, raw_dims=raw, profile_name="ssdr_data2")
        self.assertEqual([hit["artifact_id"] for hit in hits], [result["artifact_id"]])


if __name__ == "__main__":
    unittest.main()
