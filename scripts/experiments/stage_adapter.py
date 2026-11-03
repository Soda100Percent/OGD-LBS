"""Step02 adapter: pinned request/input -> subprocess -> validation -> final marker.

Only trusted internal pickle inputs are supported.
Resource values are recorded and max_workers is passed explicitly.
CPU/RAM cgroup admission belongs to the wrapper, not this standalone adapter.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import gc
import importlib.util
import json
import os
import pickle
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in [ROOT / "src", ROOT / "scripts/experiments", ROOT / "scripts/ssdr"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stage_contracts as contracts
import stage_file_io as file_io
import stage_validation as validation

ALGORITHM = ROOT / "scripts/ssdr/02_reduce_bone_weight.py"
ENV_OVERRIDES = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "OGD_PROGRESS": "log",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class ChildFailed(RuntimeError):
    def __init__(self, returncode):
        self.returncode = returncode
        super().__init__(f"Step02 child exited with code {returncode}")


def now():
    return datetime.now(timezone.utc).isoformat()


def algorithm_module():
    name = "ogd_step02_stage_adapter_algorithm"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ALGORITHM)
        contracts.require(spec is not None and spec.loader is not None,
                          "Cannot import Step02 module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def pinned_dict(source, path, manifest):
    relative = path.relative_to(source).as_posix()
    data = file_io.verified_bytes(source, relative, manifest)
    obj = pickle.loads(data)
    contracts.require(isinstance(obj, dict), f"Expected dict in trusted PKL: {relative}")
    return obj


def prepare_inputs(request, manifest):
    """Load, validate and release input objects before starting the algorithm child."""
    source = file_io.safe_path(request["artifacts"]["input_bindings"][0]["payload_path"])
    params = request["experiment"]["stage_parameters"]
    mode = params["variant"]["mode"]
    variant = params["variant"]["name"]

    module = algorithm_module()
    layout = module.select_dataset_layout(input_tree=source, variant=variant)
    expected_kind = "legacy_flat" if mode == "legacy_flat" else "new_vdb_root_explicit"
    contracts.require(layout.layout_kind == expected_kind, "Selected input layout differs")
    expected_variant = "legacy_bone_reduced" if mode == "legacy_flat" else variant
    contracts.require(layout.variant_name == expected_variant, "Selected variant differs")
    contracts.require(layout.mesh_paths, "No mesh files selected")

    full_skeleton = pinned_dict(source, layout.full_skeleton_path, manifest)
    full_animation = pinned_dict(source, layout.full_animation_path, manifest)
    reduced_skeleton = pinned_dict(source, layout.reduced_skeleton_path, manifest)
    reduced_animation = pinned_dict(source, layout.reduced_animation_path, manifest)

    meshes = {}
    for path in layout.mesh_paths:
        contracts.require(path.name not in meshes, "Flat mesh filename collision")
        meshes[path.name] = pinned_dict(source, path, manifest)

    summary = validation.validate_step02_inputs(
        full_skeleton=full_skeleton, full_animation=full_animation,
        reduced_skeleton=reduced_skeleton, reduced_animation=reduced_animation,
        meshes=meshes,
    )
    return {
        "layout_kind": layout.layout_kind,
        "variant_name": expected_variant,
        "core_paths": {
            "full_skeleton": str(layout.full_skeleton_path),
            "full_animation": str(layout.full_animation_path),
            "reduced_skeleton": str(layout.reduced_skeleton_path),
            "reduced_animation": str(layout.reduced_animation_path),
        },
        "mesh_paths": {path.name: str(path) for path in layout.mesh_paths},
        "validation": summary,
    }


def command_for(request):
    p = request["experiment"]["stage_parameters"]
    a = request["artifacts"]
    command = [
        sys.executable, str(ALGORITHM),
        "--input_dir", a["input_bindings"][0]["payload_path"],
        "--output_dir", a["output_directory"],
        "--metrics_dir", a["metrics_directory"],
        "--alpha", str(p["alpha"]),
        "--beta", str(p["beta"]),
        "--gamma", str(p["gamma"]),
        "--affinity_eps", str(p["affinity_eps"]),
        "--max_workers", str(request["execution"]["internal_workers"]),
    ]
    if p["variant"]["mode"] == "named":
        command += ["--variant", p["variant"]["name"]]
    return command


def check_algorithm_paths(request):
    # The existing path resolver can infer a work_dir from directory contents.
    # Confirm it will consume exactly the bound input, not an inferred sibling.
    import paths
    source = request["artifacts"]["input_bindings"][0]["payload_path"]
    output = request["artifacts"]["output_directory"]
    resolved = paths.resolve_ssdr_stage_dirs(
        step_id="02", input_dir=source, output_dir=output,
    )
    contracts.require(str(resolved.input_tree) == source
                      and str(resolved.output_tree) == output,
                      "SSDR path inference differs from the explicit request")


def stop_child_group(proc):
    for sig in [signal.SIGTERM, signal.SIGKILL]:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
    proc.wait()


def run_child(command, log_path, timeout_seconds):
    env = dict(os.environ)
    env.update(ENV_OVERRIDES)
    with Path(log_path).open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command, cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout_seconds)
        except BaseException:
            # Best-effort cancellation of this locally started process group.
            # It does not replace wrapper container reconciliation.
            stop_child_group(proc)
            raise
    if rc != 0:
        raise ChildFailed(rc)


def validate_outputs(request, manifest, selected):
    source = Path(request["artifacts"]["input_bindings"][0]["payload_path"])
    output = file_io.safe_path(request["artifacts"]["output_directory"])
    contracts.require(output.is_dir(), "Step02 output directory missing")
    expected_names = {
        "skeleton_hierarchy_data.pkl",
        "animation_data.pkl",
        "step02_reduce_bone_weight_summary.json",
        *selected["mesh_paths"],
    }
    tree = file_io.tree_manifest(output)
    contracts.require(set(tree["files"]) == expected_names and not tree["directories"],
                      "Step02 output inventory differs from the expected flat layout")

    def source_dict(key):
        return pinned_dict(source, Path(selected["core_paths"][key]), manifest)

    def output_dict(name):
        obj = pickle.loads(file_io.read_regular(output / name))
        contracts.require(isinstance(obj, dict), f"Invalid output PKL object: {name}")
        return obj

    summary = file_io.strict_json(file_io.read_regular(
        output / "step02_reduce_bone_weight_summary.json"
    ))
    params = request["experiment"]["stage_parameters"]
    validated = validation.validate_step02_outputs(
        full_skeleton=source_dict("full_skeleton"),
        full_animation=source_dict("full_animation"),
        reduced_skeleton=source_dict("reduced_skeleton"),
        reduced_animation=source_dict("reduced_animation"),
        input_meshes={
            name: pinned_dict(source, Path(path), manifest)
            for name, path in selected["mesh_paths"].items()
        },
        output_skeleton=output_dict("skeleton_hierarchy_data.pkl"),
        output_animation=output_dict("animation_data.pkl"),
        output_meshes={name: output_dict(name) for name in selected["mesh_paths"]},
        summary=summary,
        expected_parameters=params,
        expected_variant=selected["variant_name"],
    )
    contracts.require(summary.get("input_tree") == str(source)
                      and summary.get("output_tree") == str(output),
                      "Algorithm summary input/output path mismatch")
    contracts.require(summary.get("mesh_candidate_count") == len(selected["mesh_paths"]),
                      "Algorithm candidate count mismatch")

    records = summary["files"]
    by_output = {}
    for record in records:
        path = Path(record["output"])
        contracts.require(path.parent == output and path.name not in by_output,
                          "Duplicate or outside output path in summary")
        contracts.require(record["input"] == selected["mesh_paths"].get(path.name),
                          "Summary mesh source differs")
        by_output[path.name] = record
    contracts.require(set(by_output) == set(selected["mesh_paths"]),
                      "Summary mesh output set mismatch")

    metrics = file_io.strict_json(file_io.read_regular(
        Path(request["artifacts"]["metrics_directory"]) / "metrics_02.json"
    ))
    contracts.require(isinstance(metrics, dict)
                      and metrics.get("02_reduced_mesh_count") == len(selected["mesh_paths"])
                      and metrics.get("02_reduced_bone_count") == validated["bones"],
                      "Step02 metrics do not match validated output")
    return validated


def execute_request(
    *, request_path, request_sha256, input_manifest_path,
    trusted_input_pickle=False, check_only=False, timeout_seconds=1800,
):
    contracts.require(trusted_input_pickle is True,
                      "Explicit --trusted-input-pickle is required before unpickling")
    contracts.require(type(timeout_seconds) is int and timeout_seconds > 0,
                      "Timeout must be a positive integer")
    request_path = file_io.safe_path(request_path)
    manifest_path = file_io.safe_path(input_manifest_path)
    request_bytes = file_io.read_regular(request_path)
    contracts.require(file_io.bytes_hash(request_bytes) == request_sha256,
                      "Request bytes differ from expected SHA256")
    request = contracts.validate_step02_request(file_io.strict_json(request_bytes))
    manifest_bytes = file_io.read_regular(manifest_path)
    manifest = file_io.strict_json(manifest_bytes)

    attempt = file_io.safe_path(request["artifacts"]["attempt_root"])
    source = file_io.safe_path(request["artifacts"]["input_bindings"][0]["payload_path"])
    for path in [request_path, manifest_path]:
        contracts.require(not path.is_relative_to(attempt),
                          "Frozen controls must be outside the writable attempt directory")
    contracts.require(not os.path.lexists(attempt),
                      "Attempt already exists; refusal to overwrite or resume implicitly")

    expected_input = file_io.validate_input_manifest(
        manifest, request["artifacts"]["input_bindings"][0]
    )
    check_algorithm_paths(request)

    code_paths = {
        "adapter": Path(__file__).resolve(strict=False),
        "algorithm": ALGORITHM,
        "weight_reduction": ROOT / "src/ogd_lbs/ssdr/weight_reduction.py",
        "stage_validator": ROOT / "scripts/experiments/stage_validation.py",
        "stage_io": ROOT / "scripts/ssdr/_stage_io.py",
        "paths": ROOT / "scripts/ssdr/paths.py",
    }
    code_hashes = {key: file_io.fingerprint(path)["sha256"] for key, path in code_paths.items()}
    selected = prepare_inputs(request, manifest)
    command = command_for(request)
    if check_only:
        return {
            "status": "INPUT_VALIDATED_NO_EXECUTION",
            "identity": request["identity"],
            "request_sha256": request_sha256,
            "layout": selected,
            "command": command,
            "resource_limits_enforced_by_adapter": False,
            "code_sha256": code_hashes,
        }

    # All preconditions above are read-only. Claim a fresh attempt only now.
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "logs").mkdir()
    result_path = attempt / "stage_result.json"
    base = {
        "schema": "ogd_stage_execution_result_v1",
        "identity": request["identity"],
        "request_sha256": request_sha256,
        "input_manifest_sha256": file_io.object_hash(manifest),
        "resources_requested": request["execution"],
        "resource_limits_enforced_by_adapter": False,
        "environment_overrides": ENV_OVERRIDES,
        "code_sha256": code_hashes,
        "ingest_performed": False,
    }
    try:
        file_io.atomic_bytes(attempt / "request.json", request_bytes)
        file_io.atomic_bytes(attempt / "input_manifest.json", manifest_bytes)
        file_io.atomic_json(attempt / "execution_command.json", {"argv": command})
        file_io.atomic_json(result_path, {**base, "status": "RUNNING", "at": now()})
        gc.collect()

        run_child(command, attempt / "logs/step02.log", timeout_seconds)
        contracts.require(file_io.tree_manifest(source) == expected_input,
                          "Input tree changed during algorithm execution")
        validated = validate_outputs(request, manifest, selected)
        contracts.require(file_io.tree_manifest(source) == expected_input,
                          "Input tree changed during output validation")
        for key, path in code_paths.items():
            contracts.require(file_io.fingerprint(path)["sha256"] == code_hashes[key],
                              f"Execution code changed: {key}")

        file_io.atomic_json(attempt / "stage_validation.json", validated)

        # Inventory only immutable final output/provenance; never hash this manifest,
        # the terminal result, done marker or mutable execution logs into itself.
        files = {}
        for rel in [
            "request.json", "input_manifest.json", "execution_command.json",
            "stage_validation.json", "metrics_02.json",
        ]:
            files[rel] = file_io.fingerprint(attempt / rel)
        payload_tree = file_io.tree_manifest(attempt / "data2")
        for rel, info in payload_tree["files"].items():
            files["data2/" + rel] = info
        artifact_manifest = {
            "schema": "ogd_stage_artifact_files_v1",
            "artifact_type": "ssdr_data2",
            "identity": request["identity"],
            "request_sha256": request_sha256,
            "input_manifest_sha256": file_io.object_hash(manifest),
            "files": dict(sorted(files.items())),
            "excluded_control_files": ["artifact_files.json", "stage_result.json", "done"],
            "logs_in_artifact_identity": False,
        }
        file_io.atomic_json(attempt / "artifact_files.json", artifact_manifest)
        artifact_hash = file_io.fingerprint(attempt / "artifact_files.json")["sha256"]
        file_io.atomic_json(result_path, {
            **base, "status": "VALIDATED",
            "artifact_files_sha256": artifact_hash,
            "validation": validated, "at": now(),
        }, replace=True)
        file_io.atomic_json(attempt / "done", {
            "schema": "ogd_stage_completion_v1",
            "status": "succeeded",
            "request_sha256": request_sha256,
            "artifact_files_sha256": artifact_hash,
        })
        return {
            "status": "VALIDATED",
            "attempt_root": str(attempt),
            "request_sha256": request_sha256,
            "artifact_files_sha256": artifact_hash,
            "ingest_performed": False,
        }
    except BaseException as exc:
        # Never issue a success marker on any failure.
        try:
            file_io.atomic_json(result_path, {
                **base, "status": "FAILED",
                "error_type": type(exc).__name__, "error": str(exc),
                "child_returncode": getattr(exc, "returncode", None), "at": now(),
            }, replace=True)
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description="Execute one explicit Step02 request.")
    parser.add_argument("--stage-id", choices=["step02"], required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--trusted-input-pickle", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    try:
        result = execute_request(
            request_path=args.request,
            request_sha256=args.request_sha256,
            input_manifest_path=args.input_manifest,
            trusted_input_pickle=args.trusted_input_pickle,
            check_only=args.check_only,
            timeout_seconds=args.timeout_seconds,
        )
    except ChildFailed as exc:
        print(str(exc), file=sys.stderr)
        code = exc.returncode if exc.returncode > 0 else 128 - exc.returncode
        raise SystemExit(code if 1 <= code <= 255 else 1)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
