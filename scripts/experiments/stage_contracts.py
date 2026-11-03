"""Step02 request contract. Pure validation; no filesystem or remote actions."""

from __future__ import annotations

import json
import math
import re
from pathlib import PurePosixPath

REQUEST_SCHEMA = "ogd_stage_task_request_v1"
PIPELINE_ID = "ssdr_cached_data1_to_data4"


class StageContractError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise StageContractError(message)


def number(value, name, *, positive=False):
    require(
        type(value) in (int, float) and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        f"Invalid numeric field: {name}",
    )
    return value


def identifier(value, name):
    require(
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is not None
        and ".." not in value,
        f"Invalid identifier: {name}",
    )
    return value


def absolute_path(value, name):
    require(
        isinstance(value, str) and value.startswith("/") and value != "/"
        and "\\" not in value and "${" not in value
        and not any(ord(c) < 32 or ord(c) == 127 for c in value),
        f"Invalid path: {name}",
    )
    p = PurePosixPath(value)
    require(str(p) == value and ".." not in p.parts, f"Noncanonical path: {name}")
    return p


def validate_step02_request(request):
    require(isinstance(request, dict), "Request must be an object")
    require(request.get("schema") == REQUEST_SCHEMA, "Unsupported request schema")
    try:
        frozen = json.loads(json.dumps(request, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise StageContractError("Request must contain finite JSON data") from exc

    identity = frozen.get("identity")
    experiment = frozen.get("experiment")
    artifacts = frozen.get("artifacts")
    execution = frozen.get("execution")
    require(all(isinstance(x, dict) for x in [
        identity, experiment, artifacts, execution
    ]), "Missing request sections")

    require(identity.get("pipeline_id") == PIPELINE_ID, "Wrong pipeline")
    require(identity.get("stage_id") == "step02", "Wrong stage")
    for name in ["run_id", "task_id", "attempt_id"]:
        identifier(identity.get(name), name)
    require(identity.get("resource_group_id") in {"intel", "amd"}, "Unknown resource group")

    case = experiment.get("case_parameters")
    params = experiment.get("stage_parameters")
    dimensions = experiment.get("cumulative_dimensions")
    require(all(isinstance(x, dict) for x in [case, params, dimensions]),
            "Invalid experiment sections")
    identifier(case.get("model"), "model")
    require(case.get("param") in {"alpha", "beta", "gamma"},
            "First Step02 adapter supports ABC OFAT only")
    value = number(case.get("value"), "sweep value")
    require(value <= 1, "Sweep value exceeds one")
    effective = case.get("effective_params")
    require(isinstance(effective, dict), "Missing frozen effective parameters")

    allowed = {
        "alpha", "beta", "gamma", "affinity_eps",
        "variant", "extra_remove_bones",
    }
    require(set(params) == allowed, "Step02 parameters must be explicit and supported")
    for name in ["alpha", "beta", "gamma"]:
        number(params[name], name)
        require(params[name] <= 1, f"{name} exceeds one")
        require(type(effective.get(name)) in (int, float)
                and effective[name] == params[name], f"Frozen effective {name} differs")
        require(type(dimensions.get(name)) in (int, float)
                and dimensions[name] == params[name], f"Dimension {name} differs")
    require(math.isclose(
        sum(params[k] for k in ["alpha", "beta", "gamma"]),
        1.0, rel_tol=0, abs_tol=1e-12,
    ), "keep_sum_one parameters do not sum to one")
    require(case["value"] == params[case["param"]], "Sweep value/parameter mismatch")
    number(params["affinity_eps"], "affinity_eps", positive=True)
    require(dimensions.get("step02_affinity_eps") == params["affinity_eps"],
            "Affinity parameter/dimension mismatch")
    require(dimensions.get("artifact_type") == "ssdr_data2"
            and dimensions.get("model") == case["model"], "Target dimensions mismatch")
    identifier(dimensions.get("step02_code_version"), "step02_code_version")
    require(params["extra_remove_bones"] == [],
            "Extra bone removals are not enabled for this adapter revision")

    variant = params["variant"]
    require(isinstance(variant, dict) and set(variant) == {"mode", "name"},
            "Variant must declare mode and name")
    if variant["mode"] == "legacy_flat":
        require(variant["name"] is None, "Legacy layout must not pass --variant")
        require(dimensions.get("bone_variant") == "default",
                "Legacy dimension compatibility requires bone_variant=default")
    else:
        require(variant["mode"] == "named", "Automatic variant selection is not allowed")
        identifier(variant["name"], "variant name")
        require(dimensions.get("bone_variant") == variant["name"],
                "Variant dimension mismatch")

    bindings = artifacts.get("input_bindings")
    require(isinstance(bindings, list) and len(bindings) == 1,
            "Step02 requires exactly one data1 binding")
    binding = bindings[0]
    require(isinstance(binding, dict)
            and binding.get("artifact_type") == "ssdr_data1", "Input is not data1")
    identifier(binding.get("artifact_id"), "artifact_id")
    digest = binding.get("payload_manifest_sha256")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            "Input manifest SHA256 is missing or invalid")
    prefix = binding.get("dimensions")
    require(isinstance(prefix, dict) and prefix.get("model") == case["model"],
            "Input dimensions/model mismatch")
    for key, value in prefix.items():
        if key != "artifact_type":
            require(key in dimensions and dimensions[key] == value,
                    f"Input dimension not preserved: {key}")

    source = absolute_path(binding.get("payload_path"), "input payload")
    attempt = absolute_path(artifacts.get("attempt_root"), "attempt_root")
    output = absolute_path(artifacts.get("output_directory"), "output_directory")
    metrics = absolute_path(artifacts.get("metrics_directory"), "metrics_directory")
    logs = absolute_path(execution.get("log_directory"), "log_directory")
    require(output == attempt / "data2", "Output must be attempt_root/data2")
    require(metrics == attempt, "Metrics must be written to attempt_root")
    require(logs == attempt / "logs", "Logs must be attempt_root/logs")
    require(not attempt.is_relative_to(source) and not source.is_relative_to(attempt),
            "Input and writable attempt trees overlap")

    number(execution.get("cpu_quota"), "cpu_quota", positive=True)
    for field in ["internal_workers", "memory_request_bytes", "memory_limit_bytes"]:
        require(type(execution.get(field)) is int and execution[field] > 0,
                f"Invalid execution field: {field}")
    require(execution["memory_request_bytes"] == execution["memory_limit_bytes"],
            "This contract does not allow memory overcommit")
    identifier(execution.get("resource_policy_revision"), "resource_policy_revision")

    return frozen
