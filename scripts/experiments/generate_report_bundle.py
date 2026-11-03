#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generate a bundle of multiple OGD-LBS HTML reports from one spec.

The bundle generator intentionally keeps individual reports separate.  It does
not combine different sweep axes into one chart.  Instead, it produces an
index.html with links to each standalone report plus an optional ZIP download.

Typical use:
    python scripts/experiments/generate_report_bundle.py \
      --spec /tmp/alpha_beta_data4_bundle_spec.json \
      --output_dir /workspace/ogd_reports/alpha_beta_data4_bundle_<timestamp>
"""

import argparse
import datetime as _dt
import html
import json
import shutil
import socket
import http.server
import socketserver
import zipfile
from pathlib import Path
from typing import Any

try:
    from _env import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from ._env import REPO_ROOT  # noqa: F401

try:
    import aggregate_metrics
    import generate_report
except ModuleNotFoundError:  # pragma: no cover
    from . import aggregate_metrics  # type: ignore
    from . import generate_report  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate multi-report OGD-LBS bundle.")
    p.add_argument("--spec", required=True, help="Bundle spec JSON path.")
    p.add_argument("--output_dir", required=True, help="Output bundle directory.")
    p.add_argument("--overwrite", action="store_true", help="Delete output_dir if it already exists.")
    p.add_argument("--serve", action="store_true", help="Serve output_dir after generation.")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return obj


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def slugify_id(value: Any) -> str:
    import re

    s = str(value or "").strip()
    s = s.replace(".", "p").replace("-", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("empty report id")
    return s


def html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def relpath_from(path: Path, root: Path) -> str:
    return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()


def build_index_html(
    *,
    title: str,
    output_dir: Path,
    bundle_manifest: dict[str, Any],
    zip_relpath: str | None,
) -> str:
    generated_at = bundle_manifest.get("generated_at_utc", "")
    reports = bundle_manifest.get("reports", []) or []

    report_cards = []
    for rec in reports:
        rid = rec.get("id", "")
        rtitle = rec.get("title", rid)
        where = rec.get("where", "")
        rows = rec.get("row_count", "")
        html_rel = rec.get("html_relpath", "")

        report_cards.append(
            f"""
            <article class="card">
                <h2>{html_escape(rtitle)}</h2>
                <p><strong>ID:</strong> {html_escape(rid)}</p>
                <p><strong>Filter:</strong> <code>{html_escape(where or "<none>")}</code></p>
                <p><strong>Rows:</strong> {html_escape(rows)}</p>
                <p><a class="button" href="{html_escape(html_rel)}">Open report</a></p>
            </article>
            """
        )

    zip_block = ""
    if zip_relpath:
        zip_block = f"""
        <section class="section">
            <h2>Download</h2>
            <p>
                <a class="button download" href="{html_escape(zip_relpath)}" download>
                    Download offline report ZIP
                </a>
            </p>
            <p class="note">
                The HTML reports include data and report logic inline.
                ECharts/jsPDF are loaded from CDN when internet access is available.
            </p>
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{
    font-family: Segoe UI, Tahoma, sans-serif;
    margin: 40px auto;
    max-width: 1100px;
    line-height: 1.6;
    background: #f4f6f8;
    color: #2f3440;
}}
h1 {{
    color: #1f2937;
    border-bottom: 2px solid #d0d5dd;
    padding-bottom: 12px;
}}
.section {{
    background: #fff;
    border-radius: 10px;
    padding: 22px;
    margin: 20px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 18px;
}}
.card {{
    background: #fff;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    border-left: 5px solid #1C328A;
}}
.button {{
    display: inline-block;
    padding: 9px 14px;
    border-radius: 6px;
    background: #1C328A;
    color: #fff;
    text-decoration: none;
    font-weight: 600;
}}
.button.download {{
    background: #0f766e;
}}
code {{
    background: #f1f5f9;
    padding: 2px 5px;
    border-radius: 4px;
}}
.note {{
    color: #667085;
}}
.footer {{
    color: #667085;
    margin-top: 30px;
    font-size: 0.9em;
}}
</style>
</head>
<body>
<h1>{html_escape(title)}</h1>
<p class="note">Generated at UTC: {html_escape(generated_at)}</p>

<section class="section">
    <h2>Reports</h2>
    <div class="grid">
        {''.join(report_cards)}
    </div>
</section>

{zip_block}

<section class="section">
    <h2>Bundle manifest</h2>
    <p><a href="bundle_manifest.json">Open bundle_manifest.json</a></p>
</section>

<div class="footer">
Generated by OGD-LBS report bundle generator.
</div>
</body>
</html>
"""


def create_zip(*, output_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    files = [
        p
        for p in sorted(output_dir.rglob("*"))
        if p.is_file() and p.resolve(strict=False) != zip_path.resolve(strict=False)
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(output_dir).as_posix())


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def serve_dir(directory: Path, port: int) -> None:
    import os

    os.chdir(directory)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    url = f"http://{get_local_ip()}:{port}/index.html"

    print("\n" + "=" * 80, flush=True)
    print("🌐 Starting local report bundle server", flush=True)
    print(f"👉 Open: {url}", flush=True)
    print("=" * 80 + "\n", flush=True)

    with ReusableTCPServer(("", int(port)), http.server.SimpleHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.", flush=True)


def generate_report_bundle(*, spec_path: Path, output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    spec_path = Path(spec_path).expanduser().resolve(strict=False)
    output_dir = Path(output_dir).expanduser().resolve(strict=False)

    spec = load_json(spec_path)

    if spec.get("schema") not in {None, "ogd_lbs_report_bundle_spec_v1"}:
        raise RuntimeError(f"Unsupported bundle spec schema: {spec.get('schema')}")

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    input_root = Path(str(spec["input_root"])).expanduser().resolve(strict=False)
    mode = str(spec.get("mode", "sweep"))
    group_by = list(spec.get("group_by") or [])
    metrics = spec.get("metrics")
    reports = spec.get("reports") or []
    if not isinstance(reports, list) or not reports:
        raise RuntimeError("spec.reports must be a non-empty list")

    title = str(spec.get("title") or "OGD-LBS Report Bundle")
    generated_at = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    bundle_reports: list[dict[str, Any]] = []

    for report_spec in reports:
        if not isinstance(report_spec, dict):
            raise RuntimeError("Each report spec must be an object.")

        report_id = slugify_id(report_spec.get("id"))
        report_title = str(report_spec.get("title") or report_id)
        where_text = str(report_spec.get("where") or "")
        output_html_name = str(report_spec.get("output_html") or "OGD_LBS_Report.html")

        report_dir = output_dir / "reports" / report_id
        report_dir.mkdir(parents=True, exist_ok=True)

        manifest = aggregate_metrics.aggregate(
            input_root=input_root,
            output_dir=report_dir,
            mode=mode,
            metrics_requested=list(metrics) if isinstance(metrics, list) else None,
            group_by=group_by,
            ttest_kind=str(report_spec.get("ttest", spec.get("ttest", "none"))),
            baseline_query_text=str(report_spec.get("baseline_query", spec.get("baseline_query", ""))),
            pair_by=report_spec.get("pair_by", spec.get("pair_by")),
            ttest_group_by=report_spec.get("ttest_group_by", spec.get("ttest_group_by")),
            where_text=where_text,
        )

        output_html = report_dir / output_html_name
        generate_report.generate_report(
            aggregate_dir=report_dir,
            output_html=output_html,
            mode=mode,
            title=report_title,
            default_sweep_values=None,
            model_scale_values=None,
            scaled_metrics=None,
            axis_mode=str(report_spec.get("axis_mode", spec.get("axis_mode", generate_report.DEFAULT_AXIS_MODE))),
            pseudo_log_ratio_threshold=float(
                report_spec.get(
                    "pseudo_log_ratio_threshold",
                    spec.get("pseudo_log_ratio_threshold", generate_report.DEFAULT_PSEUDO_LOG_RATIO_THRESHOLD),
                )
            ),
        )

        report_manifest = {
            "schema": "ogd_lbs_report_manifest_v1",
            "id": report_id,
            "title": report_title,
            "where": where_text,
            "mode": mode,
            "input_root": str(input_root),
            "aggregate_dir": str(report_dir),
            "output_html": str(output_html),
            "html_relpath": relpath_from(output_html, output_dir),
            "row_count": int(manifest.get("row_count") or 0),
            "aggregate_manifest": manifest,
        }
        dump_json(report_dir / "report_manifest.json", report_manifest)

        bundle_reports.append(
            {
                "id": report_id,
                "title": report_title,
                "where": where_text,
                "mode": mode,
                "row_count": int(manifest.get("row_count") or 0),
                "report_dir": str(report_dir),
                "html_path": str(output_html),
                "html_relpath": relpath_from(output_html, output_dir),
                "aggregate_manifest": relpath_from(report_dir / "aggregate_manifest.json", output_dir),
                "report_manifest": relpath_from(report_dir / "report_manifest.json", output_dir),
            }
        )

    zip_spec = spec.get("zip") if isinstance(spec.get("zip"), dict) else {}
    zip_enabled = bool(zip_spec.get("enabled", True))
    zip_name = str(zip_spec.get("name") or "ogd_lbs_report_bundle.zip")
    zip_relpath = None
    zip_path = None

    bundle_manifest = {
        "schema": "ogd_lbs_report_bundle_manifest_v1",
        "spec_path": str(spec_path),
        "output_dir": str(output_dir),
        "title": title,
        "generated_at_utc": generated_at,
        "input_root": str(input_root),
        "mode": mode,
        "group_by": group_by,
        "report_count": len(bundle_reports),
        "reports": bundle_reports,
        "zip": {
            "enabled": zip_enabled,
            "path": "",
            "relpath": "",
        },
    }

    dump_json(output_dir / "bundle_manifest.json", bundle_manifest)

    index_html = build_index_html(
        title=title,
        output_dir=output_dir,
        bundle_manifest=bundle_manifest,
        zip_relpath=None,
    )
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    if zip_enabled:
        zip_path = output_dir / "downloads" / zip_name
        create_zip(output_dir=output_dir, zip_path=zip_path)
        zip_relpath = relpath_from(zip_path, output_dir)

        bundle_manifest["zip"]["path"] = str(zip_path)
        bundle_manifest["zip"]["relpath"] = zip_relpath
        dump_json(output_dir / "bundle_manifest.json", bundle_manifest)

        index_html = build_index_html(
            title=title,
            output_dir=output_dir,
            bundle_manifest=bundle_manifest,
            zip_relpath=zip_relpath,
        )
        (output_dir / "index.html").write_text(index_html, encoding="utf-8")

        # Rebuild zip so it contains the final index and manifest with zip metadata.
        create_zip(output_dir=output_dir, zip_path=zip_path)

    return bundle_manifest


def main() -> None:
    args = parse_args()

    manifest = generate_report_bundle(
        spec_path=Path(args.spec),
        output_dir=Path(args.output_dir),
        overwrite=bool(args.overwrite),
    )

    print("=" * 96, flush=True)
    print("OGD-LBS report bundle generated", flush=True)
    print(f"output_dir : {manifest['output_dir']}", flush=True)
    print(f"index_html : {Path(manifest['output_dir']) / 'index.html'}", flush=True)
    if manifest.get("zip", {}).get("path"):
        print(f"zip        : {manifest['zip']['path']}", flush=True)
    print("=" * 96, flush=True)

    if args.serve:
        serve_dir(Path(manifest["output_dir"]), int(args.port))


if __name__ == "__main__":
    main()
