#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def find_repo_root() -> Path:
    cur = Path(__file__).expanduser().resolve(strict=False)
    for parent in [cur.parent, *cur.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root.")


REPO_ROOT = find_repo_root()

for item in [
    REPO_ROOT / "src",
    REPO_ROOT / "scripts" / "experiments",
    REPO_ROOT / "scripts" / "ssdr",
    REPO_ROOT / "scripts" / "data_prep",
]:
    item_s = str(item)
    if item_s not in sys.path:
        sys.path.insert(0, item_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and optionally run cloud worker shards for OGD-LBS experiments."
    )

    parser.add_argument("--kind", choices=["ofat", "cv"], required=True)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output_root", default="")

    parser.add_argument("--worker_id", "--machine_id", dest="worker_id", type=int, required=True)
    parser.add_argument("--num_workers", "--machine_count", dest="num_workers", type=int, required=True)

    parser.add_argument("--only_models", nargs="*", default=None)
    parser.add_argument("--only_params", nargs="*", default=None)
    parser.add_argument("--only_folds", nargs="*", type=int, default=None)

    parser.add_argument("--case_limit", type=int, default=-1)
    parser.add_argument("--plan_dir", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_conda", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()


def validate_worker(worker_id: int, num_workers: int) -> None:
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError(f"worker_id must be in [0, {num_workers}), got {worker_id}")


def default_plan_dir(kind: str, worker_id: int, num_workers: int) -> Path:
    import datetime as _dt
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/ogd_lbs_{kind}_worker_{worker_id}_of_{num_workers}_{stamp}")


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def script_python() -> str:
    return sys.executable


def create_run_script(plan_dir: Path, case_records: list[dict[str, Any]], *, label: str) -> Path:
    run_sh = plan_dir / "run_assigned_cases.sh"

    with run_sh.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write(f"cd {shlex.quote(str(REPO_ROOT))}\n")
        f.write(f"echo 'Running assigned {label} cases...'\n")

        for rec in case_records:
            f.write("\n")
            if label == "OFAT":
                f.write(
                    "echo "
                    + shlex.quote(
                        f"--- case global_index={rec['global_case_index']} "
                        f"model={rec['model']} param={rec['param']} value={rec['value']} ---"
                    )
                    + "\n"
                )
            else:
                f.write(
                    "echo "
                    + shlex.quote(
                        f"--- case global_index={rec['global_case_index']} "
                        f"model={rec['model']} fold={rec['fold']} ---"
                    )
                    + "\n"
                )

            f.write(shell_join(rec["command"]) + "\n")

    run_sh.chmod(0o755)
    return run_sh


def plan_ofat(args: argparse.Namespace, plan_dir: Path) -> dict[str, Any]:
    import run_param_sweep as rps  # type: ignore

    config_path = Path(args.config).expanduser().resolve(strict=False)

    raw = rps.load_json(config_path)
    cfg = rps.normalize_config(raw, output_root_override=args.output_root or None)

    datasets = rps.selected_datasets(cfg, args.only_models)
    params = rps.selected_params(cfg, args.only_params)

    all_cases = rps.iter_cases(cfg, datasets=datasets, params=params)

    assigned: list[dict[str, Any]] = []
    for idx, case in enumerate(all_cases):
        if idx % int(args.num_workers) == int(args.worker_id):
            c = dict(case)
            c["_global_case_index"] = idx
            assigned.append(c)

    if int(args.case_limit) > 0:
        assigned = assigned[: int(args.case_limit)]

    case_config_dir = plan_dir / "case_configs"
    case_config_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for local_idx, case in enumerate(assigned):
        single = copy.deepcopy(cfg)
        single["datasets"] = [copy.deepcopy(case["dataset"])]
        single["output_root"] = cfg["output_root"]

        param = str(case["param"])
        model = str(case["model"])
        value = case["value"]

        single["sweeps"] = {}
        single["delta_sweeps_by_model"] = {}

        if param == "delta":
            single["delta_sweeps_by_model"] = {model: [value]}
        else:
            single["sweeps"] = {param: [value]}

        safe_value = rps.slugify(value)
        case_cfg_path = case_config_dir / (
            f"case_{case['_global_case_index']:05d}_{model}_{param}_{safe_value}.json"
        )
        dump_json(case_cfg_path, single)

        cmd = [
            script_python(),
            "scripts/experiments/run_param_sweep.py",
            "--config",
            str(case_cfg_path),
            "--only_models",
            model,
            "--only_params",
            param,
        ]

        if args.force:
            cmd.append("--force")
        if args.no_conda:
            cmd.append("--no_conda")

        records.append(
            {
                "global_case_index": int(case["_global_case_index"]),
                "local_case_index": int(local_idx),
                "model": model,
                "param": param,
                "value": value,
                "case_id": str(case["case_id"]),
                "case_config": str(case_cfg_path),
                "command": cmd,
            }
        )

    run_sh = create_run_script(plan_dir, records, label="OFAT")

    plan = {
        "schema": "ogd_lbs_ofat_cloud_worker_plan_v2",
        "config": str(config_path),
        "output_root": str(cfg["output_root"]),
        "worker_id": int(args.worker_id),
        "num_workers": int(args.num_workers),
        "total_case_count": int(len(all_cases)),
        "assigned_case_count": int(len(records)),
        "selected_models": [str(d.get("model")) for d in datasets],
        "selected_params": params,
        "cases": records,
        "run_script": str(run_sh),
    }

    return plan


def plan_cv(args: argparse.Namespace, plan_dir: Path) -> dict[str, Any]:
    import run_4fold_cv as cv  # type: ignore

    config_path = Path(args.config).expanduser().resolve(strict=False)

    raw = cv.load_json(config_path)
    cfg = cv.normalize_config(raw, args.output_root or None)

    datasets = cv.selected_datasets(cfg, args.only_models)
    folds = cv.selected_folds(cfg, args.only_folds)

    all_cases: list[dict[str, Any]] = []
    for d in datasets:
        for fold in folds:
            all_cases.append(
                {
                    "model": str(d.get("model")),
                    "fold": int(fold),
                    "dataset": d,
                }
            )

    assigned: list[dict[str, Any]] = []
    for idx, case in enumerate(all_cases):
        if idx % int(args.num_workers) == int(args.worker_id):
            c = dict(case)
            c["_global_case_index"] = idx
            assigned.append(c)

    if int(args.case_limit) > 0:
        assigned = assigned[: int(args.case_limit)]

    records: list[dict[str, Any]] = []

    for local_idx, case in enumerate(assigned):
        model = str(case["model"])
        fold = int(case["fold"])

        cmd = [
            script_python(),
            "scripts/experiments/run_4fold_cv.py",
            "--config",
            str(config_path),
            "--only_models",
            model,
            "--only_folds",
            str(fold),
        ]

        if args.output_root:
            cmd += ["--output_root", str(args.output_root)]
        if args.force:
            cmd.append("--force")
        if args.no_conda:
            cmd.append("--no_conda")

        records.append(
            {
                "global_case_index": int(case["_global_case_index"]),
                "local_case_index": int(local_idx),
                "model": model,
                "fold": fold,
                "command": cmd,
            }
        )

    run_sh = create_run_script(plan_dir, records, label="CV4")

    plan = {
        "schema": "ogd_lbs_cv4_cloud_worker_plan_v2",
        "config": str(config_path),
        "output_root": str(cfg["output_root"]),
        "worker_id": int(args.worker_id),
        "num_workers": int(args.num_workers),
        "total_case_count": int(len(all_cases)),
        "assigned_case_count": int(len(records)),
        "selected_models": [str(d.get("model")) for d in datasets],
        "selected_folds": [int(x) for x in folds],
        "cases": records,
        "run_script": str(run_sh),
    }

    return plan


def main() -> None:
    args = parse_args()
    validate_worker(int(args.worker_id), int(args.num_workers))

    plan_dir = (
        Path(args.plan_dir).expanduser().resolve(strict=False)
        if args.plan_dir
        else default_plan_dir(args.kind, int(args.worker_id), int(args.num_workers))
    )
    plan_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "ofat":
        plan = plan_ofat(args, plan_dir)
    elif args.kind == "cv":
        plan = plan_cv(args, plan_dir)
    else:
        raise RuntimeError(f"Unsupported kind: {args.kind}")

    plan_path = plan_dir / "assigned_cases.json"
    dump_json(plan_path, plan)

    print("=" * 96, flush=True)
    print(f"OGD-LBS cloud shard plan: kind={args.kind}", flush=True)
    print(f"worker_id      : {args.worker_id}", flush=True)
    print(f"num_workers    : {args.num_workers}", flush=True)
    print(f"total cases    : {plan['total_case_count']}", flush=True)
    print(f"assigned cases : {plan['assigned_case_count']}", flush=True)
    print(f"plan_dir       : {plan_dir}", flush=True)
    print(f"plan_json      : {plan_path}", flush=True)
    print(f"run_script     : {plan['run_script']}", flush=True)
    print("=" * 96, flush=True)

    for rec in plan["cases"]:
        if args.kind == "ofat":
            print(
                f"[assigned] #{rec['global_case_index']} "
                f"{rec['model']} {rec['param']}={rec['value']} -> {rec['case_id']}",
                flush=True,
            )
        else:
            print(
                f"[assigned] #{rec['global_case_index']} "
                f"{rec['model']} fold={rec['fold']}",
                flush=True,
            )

    if args.dry_run:
        print("\nDry-run mode: generated run script content:", flush=True)
        print("-" * 96, flush=True)
        print(Path(plan["run_script"]).read_text(encoding="utf-8"), flush=True)
        print("-" * 96, flush=True)
        return

    subprocess.run(["bash", str(plan["run_script"])], cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
