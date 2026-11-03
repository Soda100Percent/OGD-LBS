"""Independent controller-side Step02 business revalidation.

Only explicitly trusted internal pickle data is supported.
Checks immutable bytes before deserialization, then reuses the strict
Step02 input/output validators. Does not execute the algorithm or ingest.

A valid report is evidence for a future ingest transaction, not authorization
to modify a pool.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import stage_contracts as contracts
import stage_file_io as io
import stage_validation as validation


def require(ok, message):
    if not ok:
        raise validation.StageValidationError(message)


def _relative(value):
    require(isinstance(value, str) and value not in {"", "."}
            and "\\" not in value, "Invalid relative artifact path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts
            and str(path) == value, "Artifact path escaped its root")
    return value


def _checked_bytes(root, relative, files):
    relative = _relative(relative)
    require(relative in files, f"Undeclared file: {relative}")
    path = io.safe_path(Path(root) / relative)
    data = io.read_regular(path)
    require(files[relative] == {
        "size": len(data), "sha256": io.bytes_hash(data)
    }, f"File differs from pinned manifest: {relative}")
    return data


def _dict_pickle(root, relative, files):
    obj = pickle.loads(_checked_bytes(root, relative, files))
    require(isinstance(obj, dict), f"Expected dictionary PKL: {relative}")
    return obj


def revalidate_step02(
    *, request_bytes, input_manifest_bytes,
    input_root, collected_root, trusted_input_pickle=False,
):
    require(trusted_input_pickle is True,
            "Explicit trust acknowledgement required before loading internal PKLs")
    request = contracts.validate_step02_request(io.strict_json(request_bytes))
    source_manifest = io.strict_json(input_manifest_bytes)
    require(isinstance(source_manifest, dict)
            and source_manifest.get("schema") == io.INPUT_SCHEMA,
            "Invalid source manifest")
    binding = request["artifacts"]["input_bindings"][0]
    require(source_manifest.get("artifact_id") == binding["artifact_id"]
            and source_manifest.get("artifact_type") == "ssdr_data1"
            and io.object_hash(source_manifest) == binding["payload_manifest_sha256"],
            "Input manifest differs from frozen request")

    source = io.safe_path(input_root)
    output = io.safe_path(collected_root)
    require(not source.is_relative_to(output) and not output.is_relative_to(source),
            "Input and collected output overlap")
    expected_input_tree = {
        "files": source_manifest["files"],
        "directories": source_manifest["directories"],
    }
    require(io.tree_manifest(source) == expected_input_tree,
            "Source input bytes differ from pinned manifest")

    raw = {
        name: io.read_regular(output / name)
        for name in ["artifact_files.json", "stage_result.json", "done"]
    }
    artifact = io.strict_json(raw["artifact_files.json"])
    result = io.strict_json(raw["stage_result.json"])
    done = io.strict_json(raw["done"])
    request_hash = io.bytes_hash(request_bytes)
    artifact_hash = io.bytes_hash(raw["artifact_files.json"])
    require(
        artifact.get("schema") == "ogd_stage_artifact_files_v1"
        and artifact.get("artifact_type") == "ssdr_data2"
        and artifact.get("identity") == request["identity"]
        and artifact.get("request_sha256") == request_hash
        and artifact.get("input_manifest_sha256") == io.object_hash(source_manifest),
        "Artifact manifest identity mismatch",
    )
    require(
        result.get("schema") == "ogd_stage_execution_result_v1"
        and result.get("status") == "VALIDATED"
        and result.get("identity") == request["identity"]
        and result.get("request_sha256") == request_hash
        and result.get("input_manifest_sha256") == io.object_hash(source_manifest)
        and result.get("resources_requested") == request["execution"]
        and result.get("artifact_files_sha256") == artifact_hash,
        "Producer result identity mismatch",
    )
    require(
        done.get("schema") == "ogd_stage_completion_v1"
        and done.get("status") == "succeeded"
        and done.get("request_sha256") == request_hash
        and done.get("artifact_files_sha256") == artifact_hash,
        "Completion marker mismatch",
    )
    files = artifact.get("files")
    require(isinstance(files, dict), "Artifact file inventory missing")
    # Hash all declared outputs before loading any of their PKLs.
    for relative, expected in files.items():
        _relative(relative)
        require(io.fingerprint(output / relative) == expected,
                f"Collected file changed: {relative}")
    require(_checked_bytes(output, "request.json", files) == request_bytes,
            "Collected request differs")
    require(_checked_bytes(output, "input_manifest.json", files) == input_manifest_bytes,
            "Collected input manifest differs")

    params = request["experiment"]["stage_parameters"]
    variant = params["variant"]
    source_files = source_manifest["files"]
    full_skel_rel = "skeleton_hierarchy_data.pkl"
    full_anim_rel = "animation_data.pkl"
    if variant["mode"] == "legacy_flat":
        reduced_skel_rel = "skeleton_hierarchy_data_bone_reduced.pkl"
        reduced_anim_rel = "animation_data_bone_reduced.pkl"
        expected_variant = "legacy_bone_reduced"
        expected_layout = "legacy_flat"
    else:
        prefix = "bone_reduced_versions/" + variant["name"] + "/"
        reduced_skel_rel = prefix + full_skel_rel
        reduced_anim_rel = prefix + full_anim_rel
        expected_variant = variant["name"]
        expected_layout = "new_vdb_root_explicit"

    core_stems = {
        "skeleton_hierarchy_data", "animation_data",
        "skeleton_hierarchy_data_bone_reduced", "animation_data_bone_reduced",
        "step00_convert_summary", "step01_frame_sampling_summary",
        "step02_reduce_bone_weight_summary",
    }
    mesh_paths = {}
    for relative in sorted(source_files):
        path = Path(_relative(relative))
        if path.suffix != ".pkl" or path.stem in core_stems:
            continue
        if variant["mode"] == "named" and path.parts[0] == "bone_reduced_versions":
            continue
        require(path.name not in mesh_paths, "Flat output filename collision")
        mesh_paths[path.name] = relative
    require(mesh_paths, "No input meshes")

    full_skel = _dict_pickle(source, full_skel_rel, source_files)
    full_anim = _dict_pickle(source, full_anim_rel, source_files)
    reduced_skel = _dict_pickle(source, reduced_skel_rel, source_files)
    reduced_anim = _dict_pickle(source, reduced_anim_rel, source_files)
    input_meshes = {
        name: _dict_pickle(source, rel, source_files)
        for name, rel in mesh_paths.items()
    }
    output_meshes = {
        name: _dict_pickle(output, "data2/" + name, files)
        for name in mesh_paths
    }
    summary = io.strict_json(_checked_bytes(
        output, "data2/step02_reduce_bone_weight_summary.json", files
    ))
    require(summary.get("input_tree") == binding["payload_path"]
            and summary.get("output_tree") == request["artifacts"]["output_directory"],
            "Algorithm summary paths differ from frozen request")
    require((summary.get("layout") or {}).get("layout_kind") == expected_layout,
            "Algorithm selected a different input layout")
    require(summary.get("mesh_candidate_count") == len(mesh_paths),
            "Algorithm mesh candidate count mismatch")

    processed = summary.get("files")
    require(isinstance(processed, list) and len(processed) == len(mesh_paths),
            "Incorrect processed mesh records")
    expected_pairs = {
        (
            str(Path(binding["payload_path"]) / relative),
            str(Path(request["artifacts"]["output_directory"]) / name),
        )
        for name, relative in mesh_paths.items()
    }
    require(
        all(isinstance(item, dict) for item in processed)
        and {(item.get("input"), item.get("output")) for item in processed} == expected_pairs,
        "Processed file mapping differs from frozen inputs",
    )

    validated = validation.validate_step02_outputs(
        full_skeleton=full_skel, full_animation=full_anim,
        reduced_skeleton=reduced_skel, reduced_animation=reduced_anim,
        input_meshes=input_meshes,
        output_skeleton=_dict_pickle(output, "data2/skeleton_hierarchy_data.pkl", files),
        output_animation=_dict_pickle(output, "data2/animation_data.pkl", files),
        output_meshes=output_meshes,
        summary=summary, expected_parameters=params,
        expected_variant=expected_variant,
    )
    producer_validation = io.strict_json(_checked_bytes(
        output, "stage_validation.json", files
    ))
    require(validated == producer_validation
            and result.get("validation") == producer_validation,
            "Independent validation differs from producer validation record")

    expected_files = {
        "request.json", "input_manifest.json", "execution_command.json",
        "stage_validation.json", "metrics_02.json",
        "data2/skeleton_hierarchy_data.pkl", "data2/animation_data.pkl",
        "data2/step02_reduce_bone_weight_summary.json",
        *("data2/" + name for name in mesh_paths),
    }
    require(set(files) == expected_files, "Unexpected declared artifact files")
    output_tree = io.tree_manifest(output)
    require(
        set(output_tree["files"]) == expected_files | set(raw)
        and output_tree["directories"] == ["data2"],
        "Collected output includes undeclared files or directories",
    )
    for relative, expected in files.items():
        require(output_tree["files"][relative] == expected,
                "Collected output changed during validation")
    for name, data in raw.items():
        require(io.read_regular(output / name) == data, "Terminal metadata changed")
    require(io.tree_manifest(source) == expected_input_tree,
            "Source changed during revalidation")

    metrics = io.strict_json(_checked_bytes(output, "metrics_02.json", files))
    require(metrics.get("02_reduced_mesh_count") == len(mesh_paths)
            and metrics.get("02_reduced_bone_count") == validated["bones"],
            "Metrics differ from independently validated output")

    return {
        "schema": "ogd_step02_independent_revalidation_v1",
        "status": "BUSINESS_REVALIDATED",
        "identity": request["identity"],
        "request_sha256": request_hash,
        "input_manifest_sha256": io.object_hash(source_manifest),
        "artifact_files_sha256": artifact_hash,
        "validated_output_tree_sha256": io.object_hash(output_tree),
        "validation": validated,
        "independent_business_validation_performed": True,
        "weight_solution_recomputed": False,
        "resource_limits_verified": False,
        "pool_compatibility_checked": False,
        "ingest_permitted": False,
        "ingest_performed": False,
        "scope": "Step02 business invariants; not solver optimality or pool transaction authorization",
    }
