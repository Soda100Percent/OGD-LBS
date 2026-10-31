#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Execute one explicit OFAT case with optional ssdr_data2/ssdr_data3 artifact cache hits.

Cache behavior:
  - if ssdr_data3 hit exists: copy cached data3 and run Step04 only
  - else if ssdr_data2 hit exists: copy cached data2, run Step03 + Step04
  - else:
      alpha/beta/gamma -> Step02 + Step03 + Step04
      omega_b          -> Step03 + Step04 from baseline data2
      s_max/tau        -> Step04 from baseline data3

This script is intended to be called by run_case_manifest.py.
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def ensure_repo_paths() -> Path:
    cur = Path(__file__).resolve(strict=False)
    for parent in [cur.parent, *cur.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            repo = parent
            break
    else:
        repo = Path.cwd()

    for item in [
        repo / "src",
        repo / "scripts" / "experiments",
        repo / "scripts" / "ssdr",
        repo / "scripts" / "data_prep",
    ]:
        s = str(item)
        if s not in sys.path:
            sys.path.insert(0, s)

    return repo


REPO_ROOT = ensure_repo_paths()

import run_param_sweep as rps  # noqa: E402


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def q(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_logged(cmd: list[str], *, log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 96, flush=True)
    print("[cmd]", q(cmd), flush=True)
    print("[log]", log_path, flush=True)
    print("=" * 96, flush=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 96 + "\n")
        f.write(q(cmd) + "\n")
        f.write("=" * 96 + "\n")

        if dry_run:
            print("[dry-run] command not executed.", flush=True)
            f.write("[dry-run] command not executed.\n")
            return

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)

        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"Command failed with returncode={rc}: {q(cmd)}")


def copy_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    print(f"[copy-cache] {src} -> {dst}", flush=True)
    if dry_run:
        print("[dry-run] cache payload not copied.", flush=True)
        return

    if not src.exists():
        raise FileNotFoundError(f"Cache payload not found: {src}")

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def resolve_payload_path(hit: dict[str, Any], expected_name: str) -> Path:
    artifact_dir = Path(str(hit["artifact_dir"])).expanduser().resolve(strict=False)
    rec = hit.get("record") or {}

    payload_path = rec.get("payload_path")
    if payload_path:
        p = artifact_dir / str(payload_path)
        if p.exists():
            return p

    p = artifact_dir / "payload" / expected_name
    return p


def python_executable_from_cfg(cfg: dict[str, Any], *, no_conda: bool) -> list[str]:
    if no_conda:
        return [sys.executable]

    conda_env = str(cfg.get("conda_env", "lao-lbs"))
    return ["conda", "run", "--no-capture-output", "-n", conda_env, "python"]


def bool_enabled(obj: dict[str, Any], key: str, default: bool = False) -> bool:
    value = obj.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def append_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def effective_params(case: dict[str, Any]) -> dict[str, Any]:
    return dict((case.get("case_config_payload") or {}).get("effective_params") or {})


def stage04_config(cfg: dict[str, Any], eff: dict[str, Any]) -> dict[str, Any]:
    s4 = dict(cfg.get("step04_defaults") or {})

    if "s_max" in eff:
        s4["s_max"] = eff["s_max"]

    tau = eff.get("tau", s4.get("tau", "adaptive"))
    s4["tau"] = tau

    return s4


def build_step02_cmd(
    *,
    py: list[str],
    input_data1: Path,
    output_data2: Path,
    metrics_dir: Path,
    cfg: dict[str, Any],
    case_payload: dict[str, Any],
) -> list[str]:
    eff = dict(case_payload.get("effective_params") or {})
    dataset = dict(case_payload.get("dataset") or {})
    ssdr_defaults = dict(cfg.get("ssdr_defaults") or {})

    cmd = py + [
        "scripts/ssdr/02_reduce_bone_weight.py",
        "--input_dir", str(input_data1),
        "--output_dir", str(output_data2),
        "--metrics_dir", str(metrics_dir),
        "--overwrite_output",
        "--alpha", str(eff.get("alpha", 0.5)),
        "--beta", str(eff.get("beta", 0.3)),
        "--gamma", str(eff.get("gamma", 0.2)),
    ]

    if ssdr_defaults.get("max_workers") is not None:
        cmd += ["--max_workers", str(ssdr_defaults.get("max_workers"))]

    variant = dataset.get("variant")
    if variant:
        cmd += ["--variant", str(variant)]

    return cmd


def build_step03_cmd(
    *,
    py: list[str],
    input_data2: Path,
    output_data3: Path,
    metrics_dir: Path,
    cfg: dict[str, Any],
    case_payload: dict[str, Any],
) -> list[str]:
    eff = dict(case_payload.get("effective_params") or {})
    dataset = dict(case_payload.get("dataset") or {})
    ssdr_defaults = dict(cfg.get("ssdr_defaults") or {})

    cmd = py + [
        "scripts/ssdr/03_create_low_poly_shapes.py",
        "--input_dir", str(input_data2),
        "--output_dir", str(output_data3),
        "--metrics_dir", str(metrics_dir),
        "--overwrite_output",
    ]

    qem_config = dataset.get("qem_config")
    if qem_config:
        cmd += ["--config", str(qem_config)]

    if eff.get("omega_b") is not None:
        cmd += ["--boundary_weight", str(eff.get("omega_b"))]

    qem_workers = ssdr_defaults.get("qem_max_workers", ssdr_defaults.get("max_workers"))
    if qem_workers is not None:
        cmd += ["--max_workers", str(qem_workers)]

    return cmd


def build_step04_cmd(
    *,
    py: list[str],
    input_data3: Path,
    output_data4: Path,
    metrics_dir: Path,
    cfg: dict[str, Any],
    eff: dict[str, Any],
) -> list[str]:
    s4 = stage04_config(cfg, eff)

    cmd = py + [
        "scripts/ssdr/04_optimize_weight_and_bone_trans.py",
        "--input_dir", str(input_data3),
        "--output_dir", str(output_data4),
        "--metrics_dir", str(metrics_dir),
        "--overwrite_output",
        "--iter", str(s4.get("iter", 70)),
        "--print_every_n", str(s4.get("print_every_n", 5)),
        "--org_type", str(s4.get("org_type", 3)),
        "--weight_solver", str(s4.get("weight_solver", "simplex")),
        "--bt_covariance_mode", str(s4.get("bt_covariance_mode", "corrected")),
        "--trans_affine", str(s4.get("trans_affine", 0.0)),
        "--trans_affine_norm", str(s4.get("trans_affine_norm", 4.0)),
        "--roll_affinity", str(s4.get("roll_affinity", 0.0)),
        "--roll_affinity_norm", str(s4.get("roll_affinity_norm", 2.0)),
        "--health_level", str(s4.get("health_level", 1)),
        "--health_every", str(s4.get("health_every", 5)),
        "--health_weight_scheme", str(s4.get("health_weight_scheme", "uniform")),
        "--bone_health_every", str(s4.get("bone_health_every", 5)),
        "--s_max", str(s4.get("s_max", 4)),
    ]

    tau = s4.get("tau", "adaptive")
    if str(tau).strip().lower() == "adaptive":
        cmd += ["--mask_mode", "adaptive"]
    else:
        cmd += ["--mask_mode", "fixed", "--tau", str(tau)]

    for flag, key in [
        ("--opt_weight", "opt_weight"),
        ("--input_poses", "input_poses"),
        ("--use_weight_mask", "use_weight_mask"),
        ("--use_weight_smoothness", "use_weight_smoothness"),
        ("--eval_mesh_health", "eval_mesh_health"),
        ("--eval_bone_health", "eval_bone_health"),
        ("--validate_output", "validate_output"),
        ("--weak_validate", "weak_validate"),
        ("--save_intermediate", "save_intermediate"),
    ]:
        append_flag(cmd, flag, bool_enabled(s4, key, False))

    for flag, key in [
        ("--weights_smooth_step", "weights_smooth_step"),
        ("--weight_soft_lock", "weight_soft_lock"),
        ("--lambda_reg", "lambda_reg"),
        ("--max_workers", "max_workers"),
    ]:
        if s4.get(key) is not None:
            cmd += [flag, str(s4.get(key))]

    return cmd



def normalize_target_stage(value: str) -> str:
    text = str(value or "ssdr_data4").strip()
    aliases = {
        "data2": "ssdr_data2",
        "ssdr_data2": "ssdr_data2",
        "stage02": "ssdr_data2",
        "stage02_data2": "ssdr_data2",

        "data3": "ssdr_data3",
        "ssdr_data3": "ssdr_data3",
        "stage03": "ssdr_data3",
        "stage03_data3": "ssdr_data3",

        "data4": "ssdr_data4",
        "ssdr_data4": "ssdr_data4",
        "stage04": "ssdr_data4",
        "stage04_data4": "ssdr_data4",
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported target_stage={value!r}; "
            "expected ssdr_data2, ssdr_data3, or ssdr_data4."
        )
    return aliases[text]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one OFAT case with optional artifact cache hits.")
    parser.add_argument("--case_json", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_conda", action="store_true", help="Use current Python instead of conda-run Python.")
    parser.add_argument("--target_stage", default="ssdr_data4", choices=["ssdr_data2", "ssdr_data3", "ssdr_data4"])
    parser.add_argument("--require_input_cache", action="store_true", help="Fail if the required upstream cache artifact is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    case_path = Path(args.case_json).expanduser().resolve(strict=False)
    case = load_json(case_path)

    single_cfg_path = Path(case["single_case_config"]).expanduser().resolve(strict=False)
    cfg_raw = rps.load_json(single_cfg_path)
    cfg = rps.normalize_config(cfg_raw, output_root_override=None)

    # The single-case config should expand to exactly one case.
    datasets = rps.selected_datasets(cfg, None)
    params = rps.selected_params(cfg, None)
    cases = rps.iter_cases(cfg, datasets=datasets, params=params)

    if len(cases) != 1:
        raise RuntimeError(f"single_case_config should expand to one case, got {len(cases)}: {single_cfg_path}")

    rps_case = cases[0]
    case_payload = case.get("case_config_payload") or {
        "model": rps_case.get("model"),
        "param": rps_case.get("param"),
        "value": rps_case.get("value"),
        "case_id": rps_case.get("case_id"),
        "stage": rps_case.get("stage"),
        "effective_params": rps_case.get("effective_params", {}),
        "dataset": rps_case.get("dataset", {}),
    }

    model = str(case_payload["model"])
    param = str(case_payload["param"])
    case_id = str(case_payload["case_id"])
    dataset = dict(case_payload.get("dataset") or {})
    eff = effective_params(case)

    output_root = Path(str(cfg["output_root"])).expanduser().resolve(strict=False)
    case_root = output_root / model / f"sweep_{param}" / case_id

    data2_out = case_root / "data2"
    data3_out = case_root / "data3"
    data4_out = case_root / "data4"
    logs_dir = case_root / "logs"

    ssdr_work_dir = Path(str(dataset["ssdr_work_dir"])).expanduser().resolve(strict=False)
    base_data1 = ssdr_work_dir / "data1"
    base_data2 = ssdr_work_dir / "data2"
    base_data3 = ssdr_work_dir / "data3"

    py = python_executable_from_cfg(cfg, no_conda=bool(args.no_conda))
    target_stage = normalize_target_stage(str(args.target_stage))

    print("=" * 96, flush=True)
    print("Run cached OFAT case", flush=True)
    print(f"case_json : {case_path}", flush=True)
    print(f"model     : {model}", flush=True)
    print(f"param     : {param}", flush=True)
    print(f"value     : {case_payload.get('value')}", flush=True)
    print(f"case_root : {case_root}", flush=True)
    print(f"dry_run   : {bool(args.dry_run)}", flush=True)
    print(f"target    : {target_stage}", flush=True)
    print("=" * 96, flush=True)

    if not args.dry_run:
        case_root.mkdir(parents=True, exist_ok=True)
        dump_json(case_root / "case_config.json", case_payload)

    cache_hits = case.get("cache_hits") or {}
    data1_hit = cache_hits.get("ssdr_data1")
    data2_hit = cache_hits.get("ssdr_data2")
    data3_hit = cache_hits.get("ssdr_data3")

    used_cache = {
        "ssdr_data1": False,
        "ssdr_data2": False,
        "ssdr_data3": False,
    }
    steps_run: list[str] = []
    outputs: dict[str, str] = {}

    data1_out = case_root / "data1"
    step03_input: Path | None = None
    step04_input: Path | None = None

    def materialize_data1_from_cache() -> Path | None:
        if not data1_hit:
            return None
        payload = resolve_payload_path(data1_hit, "data1")
        copy_tree(payload, data1_out, dry_run=bool(args.dry_run))
        used_cache["ssdr_data1"] = True
        outputs["data1"] = str(data1_out)
        return data1_out

    def run_step02_from(input_data1: Path) -> None:
        cmd02 = build_step02_cmd(
            py=py,
            input_data1=input_data1,
            output_data2=data2_out,
            metrics_dir=case_root,
            cfg=cfg,
            case_payload=case_payload,
        )
        run_logged(cmd02, log_path=logs_dir / "02_reduce_bone_weight.log", dry_run=bool(args.dry_run))
        steps_run.append("ssdr_step02")
        outputs["data2"] = str(data2_out)

    def run_step03_from(input_data2: Path) -> None:
        cmd03 = build_step03_cmd(
            py=py,
            input_data2=input_data2,
            output_data3=data3_out,
            metrics_dir=case_root,
            cfg=cfg,
            case_payload=case_payload,
        )
        run_logged(cmd03, log_path=logs_dir / "03_create_low_poly_shapes.log", dry_run=bool(args.dry_run))
        steps_run.append("ssdr_step03")
        outputs["data3"] = str(data3_out)

    def run_step04_from(input_data3: Path) -> None:
        cmd04 = build_step04_cmd(
            py=py,
            input_data3=input_data3,
            output_data4=data4_out,
            metrics_dir=case_root,
            cfg=cfg,
            eff=eff,
        )
        run_logged(cmd04, log_path=logs_dir / "04_optimize_weight_and_bone_trans.log", dry_run=bool(args.dry_run))
        steps_run.append("ssdr_step04")
        outputs["data4"] = str(data4_out)

    def finish_case() -> None:
        if args.dry_run:
            return

        done = case_root / "done"
        done.write_text("done\n", encoding="utf-8")

        dump_json(case_root / "cache_execution_summary.json", {
            "schema": "ogd_lbs_cached_ofat_case_execution_summary_v2",
            "case_json": str(case_path),
            "case_root": str(case_root),
            "target_stage": target_stage,
            "used_cache": used_cache,
            "steps_run": steps_run,
            "outputs": outputs,
        })

    if target_stage == "ssdr_data2":
        if data2_hit:
            payload = resolve_payload_path(data2_hit, "data2")
            copy_tree(payload, data2_out, dry_run=bool(args.dry_run))
            used_cache["ssdr_data2"] = True
            outputs["data2"] = str(data2_out)
        else:
            input_data1 = materialize_data1_from_cache()

            if input_data1 is None:
                if args.require_input_cache:
                    raise RuntimeError(
                        f"target_stage=ssdr_data2 requires ssdr_data1 cache for "
                        f"{model} {param}={case_payload.get('value')}"
                    )

                if param not in {"alpha", "beta", "gamma"}:
                    raise RuntimeError(f"Cannot build ssdr_data2 for param={param!r} without data1 cache.")

                input_data1 = base_data1

            if param not in {"alpha", "beta", "gamma"}:
                raise RuntimeError(
                    f"target_stage=ssdr_data2 is intended for Step02 params "
                    f"alpha/beta/gamma, got param={param!r}"
                )

            run_step02_from(input_data1)

        finish_case()

    elif target_stage == "ssdr_data3":
        if data3_hit:
            payload = resolve_payload_path(data3_hit, "data3")
            copy_tree(payload, data3_out, dry_run=bool(args.dry_run))
            used_cache["ssdr_data3"] = True
            outputs["data3"] = str(data3_out)
        else:
            if data2_hit:
                payload = resolve_payload_path(data2_hit, "data2")
                copy_tree(payload, data2_out, dry_run=bool(args.dry_run))
                used_cache["ssdr_data2"] = True
                outputs["data2"] = str(data2_out)
                step03_input = data2_out
            else:
                input_data1 = materialize_data1_from_cache()

                if input_data1 is not None:
                    if param not in {"alpha", "beta", "gamma"}:
                        raise RuntimeError(f"Cannot build Step02 for param={param!r} from data1 cache.")
                    run_step02_from(input_data1)
                    step03_input = data2_out
                elif args.require_input_cache:
                    raise RuntimeError(
                        f"target_stage=ssdr_data3 requires ssdr_data2 or ssdr_data1 cache for "
                        f"{model} {param}={case_payload.get('value')}"
                    )
                elif param in {"alpha", "beta", "gamma"}:
                    run_step02_from(base_data1)
                    step03_input = data2_out
                elif param == "omega_b":
                    step03_input = base_data2
                else:
                    raise RuntimeError(f"Unsupported target_stage=ssdr_data3 param: {param}")

            if step03_input is None:
                raise RuntimeError("Internal error: step03_input is not set.")

            run_step03_from(step03_input)

        finish_case()

    else:
        if data3_hit:
            payload = resolve_payload_path(data3_hit, "data3")
            copy_tree(payload, data3_out, dry_run=bool(args.dry_run))
            used_cache["ssdr_data3"] = True
            outputs["data3"] = str(data3_out)
            step04_input = data3_out

        elif data2_hit:
            payload = resolve_payload_path(data2_hit, "data2")
            copy_tree(payload, data2_out, dry_run=bool(args.dry_run))
            used_cache["ssdr_data2"] = True
            outputs["data2"] = str(data2_out)
            step03_input = data2_out

        else:
            if param in {"alpha", "beta", "gamma"}:
                input_data1 = materialize_data1_from_cache()
                if input_data1 is None:
                    if args.require_input_cache:
                        raise RuntimeError(
                            f"target_stage=ssdr_data4 requires upstream cache for "
                            f"{model} {param}={case_payload.get('value')}"
                        )
                    input_data1 = base_data1

                run_step02_from(input_data1)
                step03_input = data2_out

            elif param == "omega_b":
                step03_input = base_data2

            elif param in {"s_max", "tau"}:
                step04_input = base_data3

            else:
                raise RuntimeError(f"Unsupported cached OFAT param for this runner: {param}")

        if step04_input is None:
            if step03_input is None:
                raise RuntimeError("Internal error: neither step03_input nor step04_input is set")
            run_step03_from(step03_input)
            step04_input = data3_out

        run_step04_from(step04_input)
        finish_case()


    print("=" * 96, flush=True)
    print("Cached OFAT case complete", flush=True)
    print(f"case_root : {case_root}", flush=True)
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
