#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add a download center and ZIP bundle to a generated pd-rew Gamma report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any


TOOLBAR_START = "<!-- OGD_PDREW_DOWNLOADS_START -->"
TOOLBAR_END = "<!-- OGD_PDREW_DOWNLOADS_END -->"

REQUIRED_FILES = [
    "Gamma_Report.html",
    "report_manifest.json",
    "pdrew_source_manifest.json",
    "aggregate/aggregate_manifest.json",
    "aggregate/aggregated_metrics.csv",
    "aggregate/aggregated_metrics.json",
    "aggregate/summary_by_group.csv",
    "aggregate/summary_table.md",
    "aggregate/summary_table.tex",
]

OPTIONAL_FILES = [
    "aggregate/ttest_results.csv",
    "aggregate/ttest_results.json",
    "aggregate/ttest_results.md",
]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
    )


def require_regular_file(path: Path, role: str) -> None:
    require(path.exists(), f"Missing {role}: {path}")
    require(path.is_file() and not path.is_symlink(), f"Unsafe {role}: {path}")


def zip_name_for(report_dir: Path) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", report_dir.name).strip("_.-")
    require(bool(safe), "Could not derive ZIP name")
    return f"ogd_lbs_{safe}.zip"


def download_toolbar(zip_name: str) -> str:
    return f"""
{TOOLBAR_START}
<style>
.ogd-download-bar {{
    position: sticky;
    top: 0;
    z-index: 900;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    margin: 0 auto 20px;
    padding: 10px 12px;
    background: rgba(255,255,255,0.97);
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}}
.ogd-download-bar a {{
    display: inline-block;
    padding: 7px 11px;
    border-radius: 5px;
    background: #1C328A;
    color: #fff;
    text-decoration: none;
    font-size: 0.9em;
    font-weight: 600;
}}
.ogd-download-bar a.ogd-download-primary {{
    background: #0f766e;
}}
</style>
<nav class="ogd-download-bar" aria-label="Report downloads">
    <a href="index.html">Download Center</a>
    <a class="ogd-download-primary"
       href="downloads/{html.escape(zip_name, quote=True)}"
       download>Download Full Report ZIP</a>
    <a href="Gamma_Report.html" download>Download HTML</a>
    <a href="aggregate/aggregated_metrics.csv" download>Download CSV</a>
    <a href="aggregate/aggregated_metrics.json" download>Download JSON</a>
</nav>
{TOOLBAR_END}
"""


def patch_report_html(report_html: Path, zip_name: str) -> None:
    require_regular_file(report_html, "Gamma report HTML")
    source = report_html.read_text(encoding="utf-8")
    toolbar = download_toolbar(zip_name)

    if TOOLBAR_START in source:
        require(TOOLBAR_END in source, "Incomplete existing download toolbar marker")
        before, remainder = source.split(TOOLBAR_START, 1)
        _old, after = remainder.split(TOOLBAR_END, 1)
        updated = before + toolbar + after
    else:
        require("<body>" in source, "Gamma report HTML has no <body> element")
        updated = source.replace("<body>", "<body>\n" + toolbar, 1)

    atomic_write_text(report_html, updated)


def build_index_html(report_dir: Path, zip_name: str) -> str:
    title = "OGD-LBS Gamma Sweep Report"
    links = [
        ("Open interactive Gamma report", "Gamma_Report.html", False),
        ("Download complete report ZIP", f"downloads/{zip_name}", True),
        ("Download aggregated metrics CSV", "aggregate/aggregated_metrics.csv", True),
        ("Download aggregated metrics JSON", "aggregate/aggregated_metrics.json", True),
        ("Download grouped summary CSV", "aggregate/summary_by_group.csv", True),
        ("Download Markdown summary", "aggregate/summary_table.md", True),
        ("Download LaTeX summary", "aggregate/summary_table.tex", True),
        ("Download report manifest", "report_manifest.json", True),
        ("Download pd-rew source manifest", "pdrew_source_manifest.json", True),
        ("Download bundle content manifest", "download_bundle_manifest.json", True),
    ]

    rows = []
    for label, href, download in links:
        attr = " download" if download else ""
        rows.append(
            f'<li><a href="{html.escape(href, quote=True)}"{attr}>'
            f"{html.escape(label)}</a></li>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{
    font-family: Segoe UI, Tahoma, sans-serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    background: #f4f6f8;
    color: #2f3440;
}}
main {{
    background: #fff;
    padding: 28px;
    border-radius: 10px;
    box-shadow: 0 4px 14px rgba(0,0,0,.08);
}}
h1 {{ color: #1f2937; }}
li {{ margin: 12px 0; }}
a {{
    color: #1C328A;
    font-weight: 600;
}}
.primary {{
    display: inline-block;
    padding: 10px 15px;
    color: white;
    background: #0f766e;
    border-radius: 6px;
    text-decoration: none;
}}
.note {{ color: #667085; }}
code {{ background: #f1f5f9; padding: 2px 5px; }}
</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>

<p>
<a class="primary" href="downloads/{html.escape(zip_name, quote=True)}" download>
Download complete report ZIP
</a>
</p>

<p><a href="Gamma_Report.html">Open interactive Gamma report</a></p>

<h2>Downloads</h2>
<ul>
{''.join(rows)}
</ul>

<p class="note">
The ZIP contains the report HTML, aggregate CSV/JSON data, summaries and
provenance manifests. It intentionally excludes <code>compat_input/</code>
and complete Stage04 model payloads.
</p>

<p class="note">
Charts use CDN-hosted ECharts/jsPDF. CSV, JSON and ZIP downloads are served
directly by this server and do not depend on those CDN resources.
</p>
</main>
</body>
</html>
"""


def build_downloads(report_dir: str | Path) -> dict[str, Any]:
    root = Path(report_dir).expanduser().resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), f"Unsafe report directory: {root}")

    selected: list[str] = []

    for relative in REQUIRED_FILES:
        path = root / relative
        require_regular_file(path, relative)
        selected.append(relative)

    for relative in OPTIONAL_FILES:
        path = root / relative
        if path.exists():
            require_regular_file(path, relative)
            selected.append(relative)

    zip_name = zip_name_for(root)
    zip_path = root / "downloads" / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    require(not zip_path.parent.is_symlink(), "downloads directory is a symlink")

    patch_report_html(root / "Gamma_Report.html", zip_name)

    file_records = []
    for relative in selected:
        path = root / relative
        file_records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    bundle_manifest = {
        "schema": "ogd_lbs_pdrew_report_download_bundle_v1",
        "report_kind": "pdrew_gamma",
        "report_directory": str(root),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file_count": len(file_records),
        "files": file_records,
        "excluded": [
            "compat_input/",
            "complete Stage04 model payloads",
            "source result-pool internals",
        ],
        "chart_runtime": {
            "echarts": "CDN",
            "jspdf": "CDN",
            "fully_offline_charts": False,
        },
    }
    manifest_path = root / "download_bundle_manifest.json"
    atomic_write_json(manifest_path, bundle_manifest)

    index_path = root / "index.html"
    atomic_write_text(index_path, build_index_html(root, zip_name))

    archive_files = [
        "index.html",
        "download_bundle_manifest.json",
        *selected,
    ]

    # Preserve order while removing duplicates.
    archive_files = list(dict.fromkeys(archive_files))

    temp_zip = zip_path.with_name(f".{zip_path.name}.tmp")
    temp_zip.unlink(missing_ok=True)

    try:
        with zipfile.ZipFile(
            temp_zip,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for relative in archive_files:
                source = root / relative
                require_regular_file(source, f"archive member {relative}")
                archive.write(source, arcname=relative)

        with zipfile.ZipFile(temp_zip, "r") as archive:
            bad = archive.testzip()
            require(bad is None, f"ZIP integrity failure at member: {bad}")

        os.replace(temp_zip, zip_path)
    finally:
        temp_zip.unlink(missing_ok=True)

    receipt = {
        "schema": "ogd_lbs_pdrew_report_download_receipt_v1",
        "status": "CREATED",
        "report_directory": str(root),
        "index_html": str(index_path),
        "zip_path": str(zip_path),
        "zip_relpath": f"downloads/{zip_name}",
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256_file(zip_path),
        "archive_file_count": len(archive_files),
        "archive_files": archive_files,
    }
    atomic_write_json(root / "download_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add download assets to a generated pd-rew Gamma report."
    )
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    receipt = build_downloads(args.report_dir)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
