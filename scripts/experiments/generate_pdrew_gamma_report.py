#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate OGD-LBS OFAT reports (delta, gamma, s_max, tau, omega_b) from pd-rew-v2 PipelinePool."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.request

HERE = Path(__file__).resolve().parent
OGD_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_PD_REW_ROOT = Path("/workspace/pd-rew-v2")
DEFAULT_POOL = Path("/workspace/pd_rew_exp_data/pools/ogd_lbs_ssdr_25_v1")
DEFAULT_BINDING = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_repo_binding.yaml"
DEFAULT_TABLE = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_experiment_table.yaml"
DEFAULT_ALIASES = DEFAULT_PD_REW_ROOT / "configs/ogd_lbs_parameter_aliases.yaml"
DEFAULT_REPORT_ROOT = Path("/workspace/ogd_reports")

PARAM_CHOICE_PREFIXES = {
    "delta": ("delta-",),
    "gamma": ("weight-",),
    "s_max": ("s_max-",),
    "tau": ("tau-",),
    "omega_b": ("boundary_weight-",),
}

PARAM_TITLES = {
    "delta": "OGD-LBS VDB Offset (δ) Sweep — pd-rew-v2 Result Pool",
    "gamma": "OGD-LBS Weight Allocation (α/β/γ) Sweep — pd-rew-v2 Result Pool",
    "s_max": "OGD-LBS Max Bones per Vertex (s_max) Sweep — pd-rew-v2 Result Pool",
    "tau": "OGD-LBS Weight Pruning Threshold (τ) Sweep — pd-rew-v2 Result Pool",
    "omega_b": "OGD-LBS QEM Boundary Penalty (ω_b) Sweep — pd-rew-v2 Result Pool",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def strict_json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json_object(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(obj, dict), f"JSON root must be object: {path}")
    return obj


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_pool_data_dir(pool: Path, value: Any) -> Path:
    require(isinstance(value, str) and value, f"Invalid pool data_dir: {value!r}")
    rel = PurePosixPath(value)
    require(
        not rel.is_absolute()
        and ".." not in rel.parts
        and str(rel) == value
        and len(rel.parts) == 2
        and rel.parts[0] == "results"
        and rel.parts[1].startswith("r"),
        f"Unsafe or unexpected pool data_dir: {value!r}",
    )
    return pool / rel


def materialize_pool_file(*, datalad_api: Any, pool: Path, path: Path, required: bool) -> bool:
    if not os.path.lexists(path):
        if required:
            raise RuntimeError(f"Required pool report file is not tracked: {path}")
        return False
    datalad_api.get(path=str(path), dataset=str(pool), result_renderer="disabled", on_failure="stop")
    if not path.is_file():
        if required:
            raise RuntimeError(f"Required pool report file is unavailable: {path}")
        return False
    return True


def _row_matches_param_family(row: dict[str, Any], param_family: str) -> bool:
    choices = row.get("sweep_choices") or {}
    w_choice = str(choices.get("weights") or "")
    prefixes = PARAM_CHOICE_PREFIXES[param_family]
    return any(w_choice.startswith(pfx) for pfx in prefixes)


def case_from_expanded_row(
    row: dict[str, Any],
    *,
    param_family: str,
    stage04_data_dir: str,
) -> dict[str, Any]:
    flat = row.get("flat_parameters") or {}
    choices = row.get("sweep_choices") or {}

    model = flat.get("data_prep_handoff.model_name") or flat.get("ssdr_handoff.model_name")
    alpha = flat.get("ssdr_stage02.par_weight", flat.get("stage02.par_weight", 0.5))
    beta = flat.get("ssdr_stage02.bone_weight", flat.get("stage02.bone_weight", 0.3))
    gamma = flat.get("ssdr_stage02.child_weight", flat.get("stage02.child_weight", 0.2))
    omega_b = flat.get("ssdr_stage03.boundary_weight", flat.get("stage03.boundary_weight", 1000.0))
    s_max = flat.get("ssdr_stage04.s_max", flat.get("stage04.s_max", 4))
    mask_mode = flat.get("ssdr_stage04.mask_mode", flat.get("stage04.mask_mode", "adaptive"))
    mask_threshold = flat.get("ssdr_stage04.mask_threshold", flat.get("stage04.mask_threshold", -1.0))
    tau = "adaptive" if str(mask_mode).lower() == "adaptive" else mask_threshold
    delta = flat.get("data_prep_stage02.offset")

    value_by_family = {
        "delta": delta,
        "gamma": gamma,
        "s_max": s_max,
        "tau": tau,
        "omega_b": omega_b,
    }
    swept_value = value_by_family[param_family]
    require(swept_value is not None, f"{row.get('row_id')}: missing value for {param_family}")

    return {
        "schema": "ogd_lbs_ofat_case_v1",
        "model": model,
        "param": param_family,
        "value": swept_value,
        "stage": "data_prep_prep02" if param_family == "delta" else f"ssdr_{param_family}",
        "case_id": row.get("row_id"),
        "effective_params": {
            "delta": delta,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "omega_b": omega_b,
            "s_max": s_max,
            "tau": tau,
        },
        "dataset": {"model": model},
        "pd_rew": {
            "row_id": row.get("row_id"),
            "row_index": row.get("row_index"),
            "sweep_choices": choices,
            "sweep_values": row.get("sweep_values") or {},
            "stage04_data_dir": stage04_data_dir,
        },
    }


def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.241.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def stop_existing_server_on_port(port: int) -> list[int]:
    """Find and terminate any process listening on TCP `port` via /proc/net/tcp{,6}."""
    port_hex = f"{int(port):04X}"
    target_inodes: set[str] = set()
    for net_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        p = Path(net_file)
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_addr, state, inode = parts[1], parts[3], parts[9]
            if state == "0A" and local_addr.upper().endswith(f":{port_hex}"):
                target_inodes.add(inode)

    killed_pids: list[int] = []
    if not target_inodes:
        return killed_pids

    my_pid = os.getpid()
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
        except ValueError:
            continue
        if pid == my_pid:
            continue
        fd_dir = proc_dir / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    link = os.readlink(fd)
                except OSError:
                    continue
                for ino in target_inodes:
                    if f"socket:[{ino}]" in link:
                        killed_pids.append(pid)
                        break
        except OSError:
            continue

    killed_pids = sorted(set(killed_pids))
    for pid in killed_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        alive = [pid for pid in killed_pids if Path(f"/proc/{pid}").exists()]
        if not alive:
            break
        time.sleep(0.1)

    for pid in killed_pids:
        if Path(f"/proc/{pid}").exists():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return killed_pids


def start_daemon_http_server(report_dir: Path, html_filename: str, port: int = 8080) -> dict[str, Any]:
    killed = stop_existing_server_on_port(port)
    if killed:
        print(f"[server] Replaced existing process(es) on port {port}: {killed}", flush=True)

    log_file = report_dir / "http_server.log"
    pid_file = report_dir / "http_server.pid"

    with log_file.open("a", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                "0.0.0.0",
                "--directory",
                str(report_dir),
            ],
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )

    pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")

    url_local = f"http://127.0.0.1:{port}/{html_filename}"
    last_err = None
    for _ in range(30):
        try:
            with urllib.request.urlopen(url_local, timeout=2.0) as resp:
                if resp.status == 200:
                    break
        except Exception as exc:
            last_err = exc
            time.sleep(0.15)
    else:
        raise RuntimeError(f"HTTP server on port {port} failed healthcheck: {last_err}")

    lan_ip = get_lan_ip()
    return {
        "pid": proc.pid,
        "port": port,
        "replaced_pids": killed,
        "lan_ip": lan_ip,
        "report_url": f"http://{lan_ip}:{port}/{html_filename}",
        "index_url": f"http://{lan_ip}:{port}/index.html",
        "log_file": str(log_file),
    }


def generate_param_report(
    *,
    param_family: str,
    pd_rew_root: Path = DEFAULT_PD_REW_ROOT,
    pool: Path = DEFAULT_POOL,
    binding_path: Path = DEFAULT_BINDING,
    table_path: Path = DEFAULT_TABLE,
    aliases_path: Path = DEFAULT_ALIASES,
    report_root: Path = DEFAULT_REPORT_ROOT,
    output_dir: Path | None = None,
    title: str | None = None,
    require_history: bool = True,
) -> dict[str, Any]:
    src = pd_rew_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import datalad.api as dl
    import aggregate_metrics
    import generate_report
    from add_pdrew_report_downloads import build_downloads
    from pd_rew_v2.experiment import expand_experiment_table, query_cache_status
    from pd_rew_v2.pool import PipelinePool

    pipe = PipelinePool(pool)
    pipe.verify(check_annex_hashes=False)

    expanded = expand_experiment_table(binding_path, table_path)
    final_stage = expanded["stages"][-1]["name"]
    cache_status = query_cache_status(
        binding_path,
        expanded,
        pipe,
        start_stage=final_stage,
        end_stage=final_stage,
    )
    st_sum = cache_status["stage_summaries_by_name"][final_stage]
    require(
        st_sum["missing_rows"] == 0 and st_sum["query_error_rows"] == 0,
        f"{final_stage} cache is incomplete: {st_sum}",
    )

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        output_dir = report_root / f"pdrew_{param_family}_report_{timestamp}"
    else:
        output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compat_root = output_dir / "compat_input"
    aggregate_dir = output_dir / "aggregate"
    html_filename = f"{param_family.capitalize()}_Report.html"
    report_html = output_dir / html_filename
    compat_root.mkdir()
    aggregate_dir.mkdir()

    cache_rows = {r["row_id"]: r for r in cache_status["rows"]}
    selected_rows = [r for r in expanded["rows"] if _row_matches_param_family(r, param_family)]
    require(bool(selected_rows), f"No rows matched parameter family {param_family!r}")

    source_records: list[dict[str, Any]] = []
    for row in selected_rows:
        row_id = row["row_id"]
        st_info = cache_rows[row_id]["stage_statuses"][final_stage]
        require(st_info["status"] == "HIT", f"{row_id}: {final_stage} not HIT")
        data_dir = st_info["data_dir"]
        data4_dir = safe_pool_data_dir(pool, data_dir) / "data4"

        summary_src = data4_dir / "step04_optimize_summary.json"
        history_src = data4_dir / "opt_out/metrics_history_per_frame.npz"
        materialize_pool_file(datalad_api=dl, pool=pool, path=summary_src, required=True)
        history_present = materialize_pool_file(
            datalad_api=dl, pool=pool, path=history_src, required=require_history
        )

        summary = load_json_object(summary_src)
        final_metrics = summary.get("final_metrics") or {}
        require(bool(final_metrics), f"{row_id}: empty final_metrics in {summary_src}")

        case_cfg = case_from_expanded_row(row, param_family=param_family, stage04_data_dir=data_dir)
        model = str(case_cfg["model"])
        val = case_cfg["value"]
        val_slug = str(val).replace(".", "p").replace("-", "m")
        case_root = compat_root / model / f"sweep_{param_family}" / f"{param_family}_{val_slug}"
        case_root.mkdir(parents=True, exist_ok=True)

        strict_json_write(case_root / "case_config.json", case_cfg)
        strict_json_write(case_root / "metrics_04.json", final_metrics)

        summary_dst = case_root / "data4/step04_optimize_summary.json"
        summary_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_src, summary_dst)
        if history_present:
            history_dst = case_root / "data4/opt_out/metrics_history_per_frame.npz"
            history_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(history_src, history_dst)

        source_records.append({
            "row_id": row_id,
            "model": model,
            "param": param_family,
            "value": val,
            "stage04_data_dir": data_dir,
            "summary_sha256": sha256_file(summary_src),
            "history_sha256": sha256_file(history_src) if history_present else None,
            "compatibility_case_root": str(case_root),
        })
        print(f"[materialized] {row_id} model={model} {param_family}={val} -> {data_dir}", flush=True)

    strict_json_write(
        output_dir / "pdrew_source_manifest.json",
        {
            "schema": "ogd_lbs_pdrew_report_source_v2",
            "param_family": param_family,
            "pool": str(pool),
            "binding": str(binding_path),
            "table": str(table_path),
            "row_count": len(source_records),
            "records": source_records,
        },
    )

    agg_manifest = aggregate_metrics.aggregate(
        input_root=compat_root,
        output_dir=aggregate_dir,
        mode="sweep",
        metrics_requested=None,
        group_by=["model", "param", "value"] if param_family == "delta" else ["param", "value"],
        ttest_kind="none",
        baseline_query_text="",
        pair_by=None,
        ttest_group_by=None,
        where_text=f"param={param_family}",
    )

    report_title = title or PARAM_TITLES.get(param_family, f"OGD-LBS {param_family} Sweep Report")
    generate_report.generate_report(
        aggregate_dir=aggregate_dir,
        output_html=report_html,
        mode="sweep",
        title=report_title,
        default_sweep_values=None,
        model_scale_values=None,
        scaled_metrics=None,
        axis_mode=generate_report.DEFAULT_AXIS_MODE,
        pseudo_log_ratio_threshold=generate_report.DEFAULT_PSEUDO_LOG_RATIO_THRESHOLD,
    )

    strict_json_write(
        output_dir / "report_manifest.json",
        {
            "schema": "ogd_lbs_pdrew_report_v2",
            "status": "GENERATED",
            "param_family": param_family,
            "pool": str(pool),
            "row_count": len(source_records),
            "aggregate_dir": str(aggregate_dir),
            "report_html": str(report_html),
        },
    )
    shutil.rmtree(compat_root, ignore_errors=True)

    dl_receipt = build_downloads(output_dir, html_filename=html_filename, title=report_title)
    return {
        "param_family": param_family,
        "row_count": len(source_records),
        "output_dir": str(output_dir),
        "html_filename": html_filename,
        "report_html": str(report_html),
        "index_html": dl_receipt["index_html"],
        "zip_path": dl_receipt["zip_path"],
        "aggregate_manifest": agg_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OGD-LBS OFAT report from pd-rew-v2 pool.")
    parser.add_argument(
        "-p", "--param",
        choices=["delta", "gamma", "s_max", "tau", "omega_b"],
        default="delta",
        help="Parameter family to generate report for (default: delta).",
    )
    parser.add_argument("--pd-rew-root", type=Path, default=DEFAULT_PD_REW_ROOT)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--require-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--serve", action="store_true", help="Serve report in foreground on --port (replaces existing listener).")
    parser.add_argument("--serve-daemon", action="store_true", help="Serve report in background on --port (replaces existing listener).")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port (default: 8080).")
    args = parser.parse_args()

    res = generate_param_report(
        param_family=args.param,
        pd_rew_root=args.pd_rew_root.expanduser().resolve(),
        pool=args.pool.expanduser().resolve(),
        binding_path=args.binding.expanduser().resolve(),
        table_path=args.table.expanduser().resolve(),
        aliases_path=args.aliases.expanduser().resolve(),
        report_root=args.report_root.expanduser().resolve(strict=False),
        output_dir=args.output_dir,
        title=args.title,
        require_history=bool(args.require_history),
    )
    print("=" * 88, flush=True)
    print(f"PD-REW-V2 REPORT GENERATED ({res['param_family'].upper()})", flush=True)
    print("ROWS        :", res["row_count"], flush=True)
    print("REPORT HTML :", res["report_html"], flush=True)
    print("INDEX HTML  :", res["index_html"], flush=True)
    print("DOWNLOAD ZIP:", res["zip_path"], flush=True)
    print("=" * 88, flush=True)

    if args.serve_daemon:
        srv = start_daemon_http_server(
            Path(res["output_dir"]),
            res["html_filename"],
            port=int(args.port),
        )
        print(f"🌐 LAN Server Active (PID={srv['pid']}, Port={srv['port']})", flush=True)
        print(f"👉 Interactive Report : {srv['report_url']}", flush=True)
        print(f"👉 Download Center    : {srv['index_url']}", flush=True)
    elif args.serve:
        stop_existing_server_on_port(int(args.port))
        import generate_report
        generate_report.serve_dir(Path(res["output_dir"]), Path(res["report_html"]), int(args.port))


if __name__ == "__main__":
    main()
