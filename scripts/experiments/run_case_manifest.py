#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def command_to_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explicit OGD-LBS case manifest with worker sharding.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--worker_id", "--machine_id", dest="worker_id", type=int, default=0)
    parser.add_argument("--num_workers", "--machine_count", dest="num_workers", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Reserved for future use; currently just recorded.")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--log_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if int(args.num_workers) <= 0:
        raise SystemExit("--num_workers must be positive")
    if int(args.worker_id) < 0 or int(args.worker_id) >= int(args.num_workers):
        raise SystemExit(f"--worker_id must be in [0, {args.num_workers})")

    manifest_path = Path(args.manifest).expanduser().resolve(strict=False)
    manifest = load_json(manifest_path)

    cases = list(manifest.get("cases", []))
    assigned = [
        (idx, case)
        for idx, case in enumerate(cases)
        if idx % int(args.num_workers) == int(args.worker_id)
    ]

    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser().resolve(strict=False)
    else:
        log_dir = manifest_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96, flush=True)
    print("OGD-LBS case manifest runner", flush=True)
    print(f"manifest      : {manifest_path}", flush=True)
    print(f"total cases   : {len(cases)}", flush=True)
    print(f"worker_id     : {args.worker_id}", flush=True)
    print(f"num_workers   : {args.num_workers}", flush=True)
    print(f"assigned      : {len(assigned)}", flush=True)
    print(f"dry_run       : {bool(args.dry_run)}", flush=True)
    print(f"log_dir       : {log_dir}", flush=True)
    print("=" * 96, flush=True)

    failed: list[dict[str, Any]] = []

    for global_idx, case in assigned:
        cmd = [str(x) for x in case.get("command", [])]
        if not cmd:
            failed.append({"case_index": global_idx, "reason": "empty command"})
            if not args.continue_on_error:
                raise RuntimeError(f"Case {global_idx} has empty command")
            continue

        log_path = Path(str(case.get("log_path") or (log_dir / f"case_{global_idx:05d}.log")))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        print()
        print("-" * 96, flush=True)
        print(f"[case {global_idx}] {case.get('model')} {case.get('param')}={case.get('value')}", flush=True)
        print(f"[hash] {case.get('case_hash')}", flush=True)
        print(f"[cmd] {command_to_string(cmd)}", flush=True)
        print(f"[log] {log_path}", flush=True)
        print("-" * 96, flush=True)

        if args.dry_run:
            continue

        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write("\n" + "=" * 96 + "\n")
            log_f.write(command_to_string(cmd) + "\n")
            log_f.write("=" * 96 + "\n")

            proc = subprocess.Popen(
                cmd,
                cwd=str(Path.cwd()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_f.write(line)

            rc = proc.wait()

        if rc != 0:
            rec = {"case_index": global_idx, "returncode": rc, "command": cmd, "log_path": str(log_path)}
            failed.append(rec)
            if not args.continue_on_error:
                raise RuntimeError(f"Case {global_idx} failed with returncode {rc}")

    report = {
        "schema": "ogd_lbs_case_manifest_run_report_v1",
        "manifest": str(manifest_path),
        "worker_id": int(args.worker_id),
        "num_workers": int(args.num_workers),
        "total_cases": int(len(cases)),
        "assigned_cases": int(len(assigned)),
        "failed_count": int(len(failed)),
        "failed": failed,
    }

    report_path = log_dir / f"run_report_worker_{args.worker_id}.json"
    report_path.write_text(json.dumps(report, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print("=" * 96, flush=True)
    print("Case manifest runner finished", flush=True)
    print(f"failed_count : {len(failed)}", flush=True)
    print(f"report       : {report_path}", flush=True)
    print("=" * 96, flush=True)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
