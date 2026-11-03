from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
for p in [
    ROOT / "src", ROOT / "scripts/ssdr",
    ROOT / "scripts/experiments", ROOT / "scripts/experiments/tests",
]:
    sys.path.insert(0, str(p))

import test_stage_c1 as fixtures
import stage_adapter as adapter
import stage_contracts as contracts
import stage_file_io as file_io
from ogd_lbs.utils.common import dump_pkl


class TestStep02ShellAdapter(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ogd_c1b_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source/data1"
        self.source.mkdir(parents=True)
        self.attempt = self.root / "attempts/attempt-0"
        self.request_path = self.root / "request.json"
        self.manifest_path = self.root / "input_manifest.json"
        self.request = fixtures.request_fixture()
        self.request["artifacts"]["input_bindings"][0]["payload_path"] = str(self.source)
        self.request["artifacts"].update({
            "attempt_root": str(self.attempt),
            "output_directory": str(self.attempt / "data2"),
            "metrics_directory": str(self.attempt),
        })
        self.request["execution"]["log_directory"] = str(self.attempt / "logs")

        self.full, self.anim, self.reduced, self.ranim, self.meshes = fixtures.object_fixture()
        for name, obj in [
            ("skeleton_hierarchy_data.pkl", self.full),
            ("animation_data.pkl", self.anim),
            ("skeleton_hierarchy_data_bone_reduced.pkl", self.reduced),
            ("animation_data_bone_reduced.pkl", self.ranim),
            ("body.pkl", self.meshes["body.pkl"]),
        ]:
            dump_pkl(self.source / name, obj)
        self.refresh_manifest()

    def write_request(self):
        self.request_path.write_bytes(file_io.encoded(self.request))
        self.request_hash = file_io.bytes_hash(self.request_path.read_bytes())

    def refresh_manifest(self):
        tree = file_io.tree_manifest(self.source)
        self.manifest = {
            "schema": file_io.INPUT_SCHEMA,
            "artifact_id": "a000001",
            "artifact_type": "ssdr_data1",
            **tree,
        }
        self.manifest_path.write_bytes(file_io.encoded(self.manifest))
        self.request["artifacts"]["input_bindings"][0]["payload_manifest_sha256"] = (
            file_io.object_hash(self.manifest)
        )
        self.write_request()

    def shell(self, *, check=False, trusted=True, expected_hash=None):
        cmd = [
            "bash", str(ROOT / "scripts/experiments/stages/step02.sh"),
            "--request", str(self.request_path),
            "--request-sha256", expected_hash or self.request_hash,
            "--input-manifest", str(self.manifest_path),
            "--timeout-seconds", "60",
        ]
        if trusted:
            cmd.append("--trusted-input-pickle")
        if check:
            cmd.append("--check-only")
        env = dict(os.environ)
        env.update(adapter.ENV_OVERRIDES)
        env["OGD_STAGE_PYTHON"] = sys.executable
        return subprocess.run(
            cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            text=True, capture_output=True, timeout=90, check=False,
        )

    def test_check_only_validates_without_attempt_directory(self):
        proc = self.shell(check=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["status"], "INPUT_VALIDATED_NO_EXECUTION")
        self.assertFalse(self.attempt.exists())
        cmd = result["command"]
        self.assertEqual(cmd[cmd.index("--max_workers") + 1], "8")
        self.assertIn("--affinity_eps", cmd)
        self.assertNotIn("--overwrite_output", cmd)
        self.assertNotIn("--allow_merge", cmd)

    def test_real_shell_legacy_output_and_hashes(self):
        before = file_io.tree_manifest(self.source)
        proc = self.shell()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["status"], "VALIDATED")
        done = file_io.strict_json((self.attempt / "done").read_bytes())
        manifest = file_io.strict_json((self.attempt / "artifact_files.json").read_bytes())
        self.assertEqual(done["request_sha256"], self.request_hash)
        self.assertEqual(
            done["artifact_files_sha256"],
            file_io.fingerprint(self.attempt / "artifact_files.json")["sha256"],
        )
        for rel, expected in manifest["files"].items():
            self.assertEqual(file_io.fingerprint(self.attempt / rel), expected)
        self.assertEqual(file_io.tree_manifest(self.source), before)
        self.assertFalse(result["ingest_performed"])
        command = file_io.strict_json((self.attempt / "execution_command.json").read_bytes())["argv"]
        self.assertEqual(command[command.index("--max_workers") + 1], "8")

    def test_real_shell_explicit_named_variant(self):
        variant_dir = self.source / "bone_reduced_versions/bone_reduced_conservative"
        variant_dir.mkdir(parents=True)
        shutil.move(
            self.source / "skeleton_hierarchy_data_bone_reduced.pkl",
            variant_dir / "skeleton_hierarchy_data.pkl",
        )
        shutil.move(
            self.source / "animation_data_bone_reduced.pkl",
            variant_dir / "animation_data.pkl",
        )
        self.request["experiment"]["stage_parameters"]["variant"] = {
            "mode": "named", "name": "bone_reduced_conservative",
        }
        self.request["experiment"]["cumulative_dimensions"]["bone_variant"] = "bone_reduced_conservative"
        self.refresh_manifest()
        proc = self.shell()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = file_io.strict_json((self.attempt / "stage_validation.json").read_bytes())
        self.assertEqual(result["variant"], "bone_reduced_conservative")

    def test_untrusted_pickle_rejected_before_attempt(self):
        proc = self.shell(check=True, trusted=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("trusted-input-pickle", proc.stderr)
        self.assertFalse(self.attempt.exists())

    def test_wrong_request_hash_rejected(self):
        proc = self.shell(check=True, expected_hash="0" * 64)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(self.attempt.exists())

    def test_changed_input_bytes_rejected(self):
        with (self.source / "body.pkl").open("ab") as f:
            f.write(b"changed")
        proc = self.shell(check=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(self.attempt.exists())

    def test_undeclared_input_file_rejected(self):
        (self.source / "undeclared.txt").write_text("not in manifest")
        proc = self.shell(check=True)
        self.assertNotEqual(proc.returncode, 0)

    def test_symlink_input_rejected(self):
        (self.source / "link.pkl").symlink_to(self.source / "body.pkl")
        proc = self.shell(check=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(self.attempt.exists())

    def test_existing_attempt_not_overwritten(self):
        self.attempt.mkdir(parents=True)
        keep = self.attempt / "keep.txt"
        keep.write_text("preserve")
        proc = self.shell()
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(keep.read_text(), "preserve")
        self.assertFalse((self.attempt / "done").exists())

    def test_missing_affinity_is_not_defaulted(self):
        del self.request["experiment"]["stage_parameters"]["affinity_eps"]
        self.write_request()
        proc = self.shell(check=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(self.attempt.exists())

    def test_manifest_identity_mismatch_rejected(self):
        self.manifest["artifact_id"] = "a000999"
        self.manifest_path.write_bytes(file_io.encoded(self.manifest))
        self.request["artifacts"]["input_bindings"][0]["payload_manifest_sha256"] = (
            file_io.object_hash(self.manifest)
        )
        self.write_request()
        proc = self.shell(check=True)
        self.assertNotEqual(proc.returncode, 0)

    def test_child_failure_records_failed_without_done(self):
        with patch.object(adapter, "run_child", side_effect=adapter.ChildFailed(9)):
            with self.assertRaises(adapter.ChildFailed):
                adapter.execute_request(
                    request_path=self.request_path, request_sha256=self.request_hash,
                    input_manifest_path=self.manifest_path, trusted_input_pickle=True,
                )
        record = file_io.strict_json((self.attempt / "stage_result.json").read_bytes())
        self.assertEqual(record["status"], "FAILED")
        self.assertEqual(record["child_returncode"], 9)
        self.assertFalse((self.attempt / "done").exists())

    def test_zero_exit_without_outputs_is_failure(self):
        with patch.object(adapter, "run_child", return_value=None):
            with self.assertRaises(contracts.StageContractError):
                adapter.execute_request(
                    request_path=self.request_path, request_sha256=self.request_hash,
                    input_manifest_path=self.manifest_path, trusted_input_pickle=True,
                )
        record = file_io.strict_json((self.attempt / "stage_result.json").read_bytes())
        self.assertEqual(record["status"], "FAILED")
        self.assertFalse((self.attempt / "done").exists())

    def test_completion_marker_is_last_published_file(self):
        calls = []
        original = file_io.atomic_bytes
        def observed(path, data, **kwargs):
            calls.append(Path(path).name)
            return original(path, data, **kwargs)
        with patch.object(file_io, "atomic_bytes", side_effect=observed):
            result = adapter.execute_request(
                request_path=self.request_path, request_sha256=self.request_hash,
                input_manifest_path=self.manifest_path, trusted_input_pickle=True,
            )
        self.assertEqual(result["status"], "VALIDATED")
        self.assertEqual(calls[-1], "done")
        self.assertLess(calls.index("artifact_files.json"), len(calls) - 1)

    def test_manifest_digest_mismatch_rejected(self):
        self.request["artifacts"]["input_bindings"][0]["payload_manifest_sha256"] = "0" * 64
        self.write_request()
        proc = self.shell(check=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(self.attempt.exists())

    def test_shell_help_requires_no_pickle_loading(self):
        env = dict(os.environ)
        env["OGD_STAGE_PYTHON"] = sys.executable
        proc = subprocess.run(
            ["bash", str(ROOT / "scripts/experiments/stages/step02.sh"), "--help"],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--input-manifest", proc.stdout)
        self.assertFalse(self.attempt.exists())


if __name__ == "__main__":
    unittest.main()
