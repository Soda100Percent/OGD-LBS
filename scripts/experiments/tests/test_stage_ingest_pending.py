from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import stage_file_io as io
import stage_ingest_sandbox as ingest
import test_stage_ingest_sandbox as fixtures


class TestPendingTransactionAdmission(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TestStageIngestSandbox(
            "test_complete_payload_and_provenance_ingested"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.pool = self.fixture.pool

    def interrupt_after_artifact_publication(self):
        with patch.object(
            ingest, "_write_index",
            side_effect=OSError("injected interruption before index publication"),
        ):
            with self.assertRaises(OSError):
                self.fixture.call()
        self.assertFalse(self.fixture.index()["artifacts"])
        self.assertEqual(len(list((self.pool / "artifacts").iterdir())), 1)
        transactions = list((self.pool / ".stage_ingest_transactions").glob("*.json"))
        self.assertEqual(len(transactions), 1)
        self.assertEqual(json.loads(transactions[0].read_text())["status"], "PENDING")
        return transactions[0]

    def change_diagnostic_and_rehash(self):
        # A second content-distinct request with the same scientific dimensions.
        # It remains business-valid, so transaction isolation must reject it.
        changed = [
            "metrics_02.json", "artifact_files.json", "stage_result.json", "done"
        ]
        originals = {
            name: (self.fixture.output / name).read_bytes() for name in changed
        }
        path = self.fixture.output / "metrics_02.json"
        metrics = io.strict_json(path.read_bytes())
        metrics["second_request_diagnostic"] = 1
        path.write_bytes(io.encoded(metrics))

        path = self.fixture.output / "artifact_files.json"
        manifest = io.strict_json(path.read_bytes())
        manifest["files"]["metrics_02.json"] = io.fingerprint(
            self.fixture.output / "metrics_02.json"
        )
        path.write_bytes(io.encoded(manifest))
        digest = io.bytes_hash(path.read_bytes())

        for name in ["stage_result.json", "done"]:
            path = self.fixture.output / name
            obj = io.strict_json(path.read_bytes())
            obj["artifact_files_sha256"] = digest
            path.write_bytes(io.encoded(obj))
        return originals

    def restore(self, originals):
        for name, data in originals.items():
            (self.fixture.output / name).write_bytes(data)

    def test_different_request_blocked_while_transaction_pending(self):
        tx = self.interrupt_after_artifact_publication()
        before = io.tree_manifest(self.pool)
        self.change_diagnostic_and_rehash()
        with self.assertRaises(ingest.PendingTransactionConflict):
            self.fixture.call()
        self.assertEqual(io.tree_manifest(self.pool), before)
        self.assertEqual(json.loads(tx.read_text())["status"], "PENDING")
        self.assertEqual(self.fixture.index()["next_artifact_id"], 1)

    def test_original_request_recovers_after_foreign_request_rejection(self):
        self.interrupt_after_artifact_publication()
        originals = self.change_diagnostic_and_rehash()
        with self.assertRaises(ingest.PendingTransactionConflict):
            self.fixture.call()
        self.restore(originals)
        result = self.fixture.call()
        self.assertEqual(result["status"], "RECOVERED")
        self.assertEqual(result["artifact_id"], "a000001")
        self.assertEqual(len(self.fixture.index()["artifacts"]), 1)
        self.assertEqual(self.fixture.index()["next_artifact_id"], 2)

    def test_after_recovery_different_content_is_conflict_not_new_allocation(self):
        self.interrupt_after_artifact_publication()
        result = self.fixture.call()
        self.assertEqual(result["status"], "RECOVERED")
        self.change_diagnostic_and_rehash()
        before = io.tree_manifest(self.pool)
        with self.assertRaises(ingest.IngestConflict) as caught:
            self.fixture.call()
        self.assertNotIsInstance(caught.exception, ingest.PendingTransactionConflict)
        self.assertEqual(io.tree_manifest(self.pool), before)

    def test_malformed_transaction_registry_fails_closed(self):
        directory = self.pool / ".stage_ingest_transactions"
        directory.mkdir()
        path = directory / ("0" * 64 + ".json")
        io.atomic_json(path, {
            "schema": "ogd_stage_ingest_transaction_v1",
            "transaction_id": "0" * 64,
            "status": "PENDING",
            "artifact_id": "../invalid",
        })
        # The persistent coordination lock is not an artifact publication.
        # Create it through the real lock API before taking the invariant snapshot.
        with ingest.pool_lock(self.pool):
            pass
        self.assertTrue((self.pool / ".ingest.lock").is_file())

        before = io.tree_manifest(self.pool)
        with self.assertRaises(ValueError):
            self.fixture.call()
        self.assertEqual(io.tree_manifest(self.pool), before)
        self.assertFalse(self.fixture.index()["artifacts"])


if __name__ == "__main__":
    unittest.main()
