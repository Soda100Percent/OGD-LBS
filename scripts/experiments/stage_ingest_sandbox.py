"""Isolated Step02 ingest transaction candidate, NOT a production pool API.

Requires an explicitly marked sandbox pool. Revalidates trusted internal PKLs.
Uses existing dimension normalization/query semantics without migrating profiles.
No overwrite, remote operations, source deletion or automatic algorithm retry.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path

import result_pool as rp
import stage_file_io as io
import stage_revalidation as business

MARKER = ".stage_ingest_sandbox.json"
MARKER_SCHEMA = "ogd_stage_ingest_sandbox_v1"
PRODUCTION = Path("/workspace/ogd_experiment_pool")
PROFILE = "ssdr_data2"


class IngestConflict(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return io.strict_json(io.read_regular(path))


def guard(pool, sandbox_root):
    pool = io.safe_path(pool)
    sandbox = io.safe_path(sandbox_root)
    require(sandbox.is_dir() and pool != sandbox and pool.is_relative_to(sandbox),
            "Pool must be strictly inside the explicit sandbox")
    production = PRODUCTION.resolve()
    require(not pool.is_relative_to(production) and not production.is_relative_to(pool),
            "Production pool overlap is forbidden")
    marker = read(pool / MARKER)
    require(marker == {
        "schema": MARKER_SCHEMA,
        "pool": str(pool),
        "sandbox_root": str(sandbox),
        "production_use_allowed": False,
    }, "Missing or incompatible sandbox marker")
    return pool


@contextlib.contextmanager
def pool_lock(pool):
    path = io.safe_path(pool / ".ingest.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "r+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Sandbox pool writer is already active") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def pool_metadata(pool):
    dims = read(pool / "dimensions.json")
    rules = read(pool / "equivalence_profile.json")
    index = read(pool / "index.json")
    require(dims.get("schema") == "ogd_lbs_result_pool_dimensions_v2",
            "Existing v2 profiles required; migration is not performed")
    require(index.get("schema") == "ogd_lbs_result_pool_index_v2"
            and isinstance(index.get("artifacts"), dict), "Existing v2 index required")
    profile = dims.get("profiles", {}).get(PROFILE)
    require(isinstance(profile, dict) and isinstance(profile.get("dimensions"), list),
            "ssdr_data2 profile is missing")
    return dims, rules, index


def resolve_dimensions(request, dims, rules, approved_defaults):
    profile = dims["profiles"][PROFILE]
    names = profile["dimensions"]
    require(len(names) == len(set(names)), "Duplicate profile dimensions")
    require(isinstance(approved_defaults, dict), "Approved defaults snapshot required")
    supplied = request["experiment"]["cumulative_dimensions"]
    require(supplied.get("artifact_type") == PROFILE, "Wrong output artifact type")
    raw, defaulted = {}, {}
    for name in names:
        if name in supplied and supplied[name] is not None:
            raw[name] = supplied[name]
        else:
            require(name in approved_defaults,
                    f"No explicit historical/default interpretation approved: {name}")
            require(approved_defaults[name] == profile.get("defaults", {}).get(name),
                    f"Approved interpretation differs from pool profile: {name}")
            raw[name] = approved_defaults[name]
            defaulted[name] = approved_defaults[name]
        require(raw[name] is not None, f"Unresolved data2 dimension: {name}")

    unregistered = sorted(set(supplied) - set(names))
    require(not unregistered,
            f"Request contains unregistered result dimensions: {unregistered}")
    normalized, trace, fingerprint = rp.normalize_dimensions(
        raw, dimensions_obj=dims, equivalence_profile_obj=rules,
        profile_name=PROFILE, context={"source": "sandbox-validated-stage-ingest"},
    )
    return raw, normalized, trace, fingerprint, defaulted


def _write_index(pool, index):
    """Atomic index replacement. Kept separate for crash-injection tests."""
    io.atomic_json(pool / "index.json", index, replace=True)


def _verify_published(directory, transaction):
    tree = io.tree_manifest(directory)
    require(io.object_hash(tree) == transaction["published_tree_sha256"],
            "Published artifact differs from pending transaction")


def _register_transaction(pool, index, transaction_path, transaction):
    aid = transaction["artifact_id"]
    target = io.safe_path(pool / "artifacts" / aid)
    stage = io.safe_path(pool / transaction["staging_relative"])
    require(stage.parent == pool / ".stage_ingest_staging",
            "Transaction staging path escaped sandbox")

    existing = index["artifacts"].get(aid)
    require(existing is None or existing == transaction["index_record"],
            "Artifact ID is occupied by another record")

    if not target.exists():
        require(existing is None, "Index references missing artifact; manual repair required")
        require(stage.is_dir(), "Pending transaction payload is missing")
        _verify_published(stage, transaction)
        # All pool writers must honor .ingest.lock. This is a cooperative transaction.
        require(not os.path.lexists(target), "Artifact destination appeared concurrently")
        os.rename(stage, target)
        io.sync_dir(target.parent)
    _verify_published(target, transaction)

    updated = json.loads(json.dumps(index))
    updated["artifacts"][aid] = transaction["index_record"]
    updated["next_artifact_id"] = max(
        int(updated["next_artifact_id"]), transaction["next_artifact_id"]
    )
    updated["optional_fast_indexes"] = {}
    updated["updated_at_utc"] = rp.utc_now_iso()
    _write_index(pool, updated)

    committed = {**transaction, "status": "COMMITTED"}
    io.atomic_json(transaction_path, committed, replace=True)
    return aid


class PendingTransactionConflict(IngestConflict):
    """A different write must wait until the pending transaction is reconciled."""


def check_pending_transactions(pool, current_transaction_id):
    """Called under the pool write lock, before allocation or artifact publication."""
    import re

    directory = io.safe_path(pool / ".stage_ingest_transactions")
    if not directory.exists():
        return
    require(directory.is_dir(), "Transaction registry is not a directory")

    pending = []
    for path in sorted(directory.iterdir()):
        require(not path.is_symlink() and path.is_file()
                and path.suffix == ".json"
                and re.fullmatch(r"[0-9a-f]{64}", path.stem) is not None,
                f"Unexpected transaction registry entry: {path.name}")

        transaction = read(path)
        require(
            isinstance(transaction, dict)
            and transaction.get("schema") == "ogd_stage_ingest_transaction_v1"
            and transaction.get("transaction_id") == path.stem
            and transaction.get("status") in {"PENDING", "COMMITTED"},
            f"Invalid transaction registry record: {path.name}",
        )
        if transaction["status"] == "PENDING":
            aid = transaction.get("artifact_id")
            require(isinstance(aid, str)
                    and re.fullmatch(r"a[0-9]{6,}", aid) is not None,
                    f"Invalid pending artifact ID: {path.name}")
            require(type(transaction.get("next_artifact_id")) is int
                    and transaction["next_artifact_id"] > 0,
                    f"Invalid pending next_artifact_id: {path.name}")
            relative = transaction.get("staging_relative")
            require(isinstance(relative, str), "Missing pending staging path")
            relative_path = Path(relative)
            require(
                not relative_path.is_absolute()
                and ".." not in relative_path.parts
                and relative_path.parent == Path(".stage_ingest_staging")
                and str(relative_path) == relative,
                "Pending staging path escaped the private transaction directory",
            )
            record = transaction.get("index_record")
            require(
                isinstance(record, dict)
                and record.get("artifact_id") == aid
                and record.get("artifact_type") == PROFILE
                and record.get("artifact_dir") == f"artifacts/{aid}"
                and record.get("status") == "succeeded",
                "Pending transaction/index-record identity mismatch",
            )
            for field in [
                "request_sha256", "collected_tree_sha256", "published_tree_sha256"
            ]:
                value = transaction.get(field)
                require(isinstance(value, str)
                        and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                        f"Invalid pending transaction hash: {field}")
            pending.append(transaction)

    require(len(pending) <= 1,
            "Multiple pending transactions require explicit manual reconciliation")
    if pending and pending[0]["transaction_id"] != current_transaction_id:
        raise PendingTransactionConflict(
            "PENDING_TRANSACTION_BLOCKS_NEW_INGEST: "
            f"transaction={pending[0]['transaction_id']} "
            f"artifact={pending[0]['artifact_id']}; "
            "repeat/reconcile that transaction before submitting another request"
        )


def ingest_validated_step02(
    *, pool, sandbox_root, request_bytes, input_manifest_bytes,
    input_root, collected_root, approved_defaults,
    trusted_input_pickle=False,
):
    pool = guard(pool, sandbox_root)
    source = io.safe_path(input_root)
    collected = io.safe_path(collected_root)
    require(not source.is_relative_to(pool) and not pool.is_relative_to(source),
            "Input and destination pool overlap")
    require(not collected.is_relative_to(pool) and not pool.is_relative_to(collected),
            "Collected output and destination pool overlap")

    # Always recompute the business check. Do not trust an editable PASSED report.
    checked = business.revalidate_step02(
        request_bytes=request_bytes,
        input_manifest_bytes=input_manifest_bytes,
        input_root=source, collected_root=collected,
        trusted_input_pickle=trusted_input_pickle,
    )
    request = io.strict_json(request_bytes)
    before_tree = io.tree_manifest(collected)
    collected_digest = io.object_hash(before_tree)
    payload_tree = io.tree_manifest(collected / "data2")

    with pool_lock(pool):
        guard(pool, sandbox_root)
        dims, rules, index = pool_metadata(pool)
        raw, canonical, trace, fingerprint, defaulted = resolve_dimensions(
            request, dims, rules, approved_defaults
        )
        profile_digest = rp.profile_fingerprint(dims["profiles"][PROFILE])
        txid = io.object_hash({
            "request_sha256": checked["request_sha256"],
            "collected_tree_sha256": collected_digest,
            "raw_dimensions": raw,
            "profile_sha256": profile_digest,
            "equivalence_sha256": io.object_hash(rules),
        })
        # Do not allocate a new ID while another publication is pending.
        # This also validates the current pending record before its recovery path.
        check_pending_transactions(pool, txid)
        transactions = io.safe_path(pool / ".stage_ingest_transactions")
        transactions.mkdir(exist_ok=True)
        tx_path = transactions / (txid + ".json")

        # Same explicit request can reconcile a known pending publication.
        if tx_path.exists():
            transaction = read(tx_path)
            require(transaction.get("transaction_id") == txid
                    and transaction.get("request_sha256") == checked["request_sha256"]
                    and transaction.get("collected_tree_sha256") == collected_digest,
                    "Existing transaction does not match this request")
            if transaction["status"] == "PENDING":
                aid = _register_transaction(pool, index, tx_path, transaction)
                return {
                    "status": "RECOVERED", "artifact_id": aid,
                    "transaction_id": txid, "sandbox_only": True,
                }
            require(transaction["status"] == "COMMITTED", "Unknown transaction status")
            aid = transaction["artifact_id"]
            require(index["artifacts"].get(aid) == transaction["index_record"],
                    "Committed transaction/index mismatch")
            _verify_published(pool / "artifacts" / aid, transaction)
            return {
                "status": "REUSED", "artifact_id": aid,
                "transaction_id": txid, "sandbox_only": True,
            }

        matches = []
        for aid, rec in sorted(index["artifacts"].items()):
            if rec.get("artifact_type") != PROFILE or rec.get("status") != "succeeded":
                continue
            # Read only. Do not call lookup_artifacts() which can initialize/migrate.
            old_path = io.safe_path(pool / "artifacts" / aid / "raw_dimensions.json")
            require(old_path.is_file(), f"Missing dimensions for existing artifact: {aid}")
            old_dims = read(old_path).get("dimensions")
            require(isinstance(old_dims, dict), "Invalid existing raw dimensions")
            if rp.dimensions_equal(
                raw, old_dims, dimensions_obj=dims,
                equivalence_profile_obj=rules, profile_name=PROFILE,
            ):
                matches.append((aid, rec))
        if matches:
            # This candidate deliberately refuses legacy/ambiguous equivalence
            # instead of silently treating matching dimensions as valid payload.
            for aid, rec in matches:
                artifact = pool / "artifacts" / aid
                info = read(artifact / "artifact_manifest.json")
                if (
                    info.get("validated_collected_tree_sha256") == collected_digest
                    and info.get("request_sha256") == checked["request_sha256"]
                ):
                    inventory = read(artifact / "payload_files.json")
                    require(io.tree_manifest(artifact / "payload/data2") == inventory["tree"],
                            "Existing matching artifact payload is corrupt")
                    return {"status": "REUSED", "artifact_id": aid, "sandbox_only": True}
            raise IngestConflict(
                "Matching dimensions have different or unverified content; refusing overwrite"
            )

        next_id = index.get("next_artifact_id")
        require(type(next_id) is int and next_id > 0, "Invalid next_artifact_id")
        aid = f"a{next_id:06d}"
        require(aid not in index["artifacts"]
                and not os.path.lexists(pool / "artifacts" / aid),
                "Allocated artifact ID collides with existing data")

        staging_root = io.safe_path(pool / ".stage_ingest_staging")
        staging_root.mkdir(exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=txid[:16] + "-", dir=staging_root))
        # No deletion on failure: preserve staging for audit/recovery.
        (stage / "payload").mkdir()
        shutil.copytree(collected / "data2", stage / "payload/data2", symlinks=True)
        (stage / "evidence").mkdir()
        for name in before_tree["files"]:
            if not name.startswith("data2/"):
                shutil.copy2(collected / name, stage / "evidence" / name)
        require(io.tree_manifest(stage / "payload/data2") == payload_tree,
                "Copied data2 payload differs")
        require(io.tree_manifest(collected) == before_tree, "Collected source changed")

        io.atomic_json(stage / "raw_dimensions.json", {
            "schema": "ogd_lbs_result_pool_raw_dimensions_v1",
            "dimensions": raw,
        })
        io.atomic_json(stage / "canonical_dimensions.json", {
            "schema": "ogd_lbs_result_pool_canonical_dimensions_v1",
            "profile": PROFILE, "dimensions": canonical,
            "dimension_fingerprint": fingerprint,
        })
        io.atomic_json(stage / "override_trace.json", trace)
        io.atomic_json(stage / "payload_files.json", {"tree": payload_tree})
        io.atomic_json(stage / "business_revalidation.json", checked)
        io.atomic_json(stage / "source_info.json", {
            "request_sha256": checked["request_sha256"],
            "input_manifest_sha256": checked["input_manifest_sha256"],
            "input_bindings": request["artifacts"]["input_bindings"],
            "raw_request_dimensions": request["experiment"]["cumulative_dimensions"],
            "explicitly_approved_default_interpretations": defaulted,
            "profile_sha256": profile_digest,
            "equivalence_sha256": io.object_hash(rules),
            "sandbox_only": True,
        })

        artifact = {
            "schema": "ogd_validated_stage_artifact_v1",
            "artifact_id": aid, "artifact_type": PROFILE, "status": "succeeded",
            "payload_path": "payload/data2",
            "request_sha256": checked["request_sha256"],
            "validated_collected_tree_sha256": collected_digest,
            "dimension_fingerprint": fingerprint,
            "capabilities": ["step03_input"],
            "independent_business_validation_performed": True,
            "sandbox_only": True,
        }
        io.atomic_json(stage / "artifact_manifest.json", artifact)
        index_record = {
            "artifact_id": aid, "artifact_type": PROFILE, "status": "succeeded",
            "artifact_dir": f"artifacts/{aid}",
            "payload_path": "payload/data2",
            "model": canonical["model"],
            "param": request["experiment"]["case_parameters"]["param"],
            "value": request["experiment"]["case_parameters"]["value"],
            "dimension_fingerprint": fingerprint,
            "validation_schema": "ogd_step02_independent_revalidation_v1",
            "sandbox_only": True,
        }
        transaction = {
            "schema": "ogd_stage_ingest_transaction_v1",
            "transaction_id": txid, "status": "PENDING",
            "artifact_id": aid, "next_artifact_id": next_id + 1,
            "request_sha256": checked["request_sha256"],
            "collected_tree_sha256": collected_digest,
            "staging_relative": stage.relative_to(pool).as_posix(),
            "published_tree_sha256": io.object_hash(io.tree_manifest(stage)),
            "index_record": index_record,
        }
        # Flush payload/evidence before exposing its directory.
        for current, dirs, files in os.walk(stage):
            for name in files:
                with (Path(current) / name).open("rb") as stream:
                    os.fsync(stream.fileno())
        for current, dirs, files in os.walk(stage, topdown=False):
            io.sync_dir(Path(current))
        io.atomic_json(tx_path, transaction)
        _register_transaction(pool, index, tx_path, transaction)
        return {
            "status": "INGESTED", "artifact_id": aid,
            "transaction_id": txid, "sandbox_only": True,
        }
