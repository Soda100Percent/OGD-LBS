from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SSDR_DIR = REPO / "scripts" / "ssdr"
SRC_DIR = REPO / "src"

for item in [str(SSDR_DIR), str(SRC_DIR)]:
    if item not in sys.path:
        sys.path.insert(0, item)


def test_report_stage_progress_is_inert_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PD_REW_STAGE_PROGRESS_FILE", raising=False)

    from _stage_io import report_stage_progress

    result = report_stage_progress(
        stage="stage00",
        phase="scanning",
        completed_units=0,
        total_units=None,
        unit="file",
        message="scan",
    )

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_report_stage_progress_writes_atomic_json(monkeypatch, tmp_path):
    out = tmp_path / "telemetry" / "stage_progress.json"

    monkeypatch.setenv("PD_REW_STAGE_PROGRESS_FILE", str(out))
    monkeypatch.setenv("PD_REW_RUN_ID", "run-progress")
    monkeypatch.setenv("PD_REW_WORKER_ID", "w0")
    monkeypatch.setenv("PD_REW_CASE_ID", "case0")
    monkeypatch.setenv("PD_REW_ATTEMPT_ID", "attempt0")

    from _stage_io import report_stage_progress

    first = report_stage_progress(
        stage="stage00",
        phase="scanning",
        completed_units=0,
        total_units=None,
        unit="file",
        message="scanning",
    )
    second = report_stage_progress(
        stage="stage00",
        phase="completed",
        completed_units=2,
        total_units=2,
        unit="file",
        message="done",
        extra={"converted_json_count": 1},
    )

    assert first == out
    assert second == out

    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["schema"] == "pd_rew_stage_progress_v1"
    assert obj["run_id"] == "run-progress"
    assert obj["worker_id"] == "w0"
    assert obj["case_id"] == "case0"
    assert obj["attempt_id"] == "attempt0"
    assert obj["stage"] == "stage00"
    assert obj["phase"] == "completed"
    assert obj["sequence"] >= 2
    assert obj["completed_units"] == 2
    assert obj["total_units"] == 2
    assert obj["unit"] == "file"
    assert obj["extra"]["converted_json_count"] == 1
    assert obj["updated_at_utc"].endswith("Z")

    assert not list(out.parent.glob("*.tmp"))


def test_report_stage_progress_rejects_invalid_counts(monkeypatch, tmp_path):
    out = tmp_path / "stage_progress.json"
    monkeypatch.setenv("PD_REW_STAGE_PROGRESS_FILE", str(out))

    from _stage_io import report_stage_progress

    with pytest.raises(ValueError, match="completed_units must not exceed total_units"):
        report_stage_progress(
            stage="stage00",
            phase="bad",
            completed_units=3,
            total_units=2,
            unit="file",
        )

    assert not out.exists()


def test_stage00_progress_and_outputs(tmp_path):
    work_dir = tmp_path / "work"
    input_dir = work_dir / "data0_json"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)

    source_obj = {
        "name": "sample",
        "values": [1, 2, 3],
        "nested": {"ok": True},
    }
    (input_dir / "sample.json").write_text(
        json.dumps(source_obj, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    copied_obj = {"copied": True, "value": 42}
    copied_src = nested / "copied.pkl"
    with copied_src.open("wb") as f:
        pickle.dump(copied_obj, f)

    (input_dir / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    progress_file = tmp_path / "telemetry" / "stage_progress.json"

    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PD_REW_STAGE_PROGRESS_FILE": str(progress_file),
            "PD_REW_RUN_ID": "run-stage00",
            "PD_REW_WORKER_ID": "w0",
            "PD_REW_CASE_ID": "case0",
            "PD_REW_ATTEMPT_ID": "attempt0",
        }
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "ssdr" / "00_convert_json_to_pickle.py"),
            "--input_dir",
            str(work_dir),
            "--max_workers",
            "1",
            "--overwrite_output",
        ],
        cwd=str(REPO),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    output_dir = work_dir / "data0"
    summary_path = output_dir / "step00_convert_summary.json"
    converted_path = output_dir / "sample.pkl"
    copied_path = output_dir / "nested" / "copied.pkl"

    assert summary_path.is_file()
    assert converted_path.is_file()
    assert copied_path.is_file()
    assert not (output_dir / "ignored.pkl").exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema"] == "ogd_lbs_ssdr_step00_to_data0_v2"
    assert summary["source_file_count"] == 2
    assert summary["processed_count"] == 2
    assert summary["converted_json_count"] == 1
    assert summary["copied_pkl_count"] == 1

    with converted_path.open("rb") as f:
        converted = pickle.load(f)
    with copied_path.open("rb") as f:
        copied = pickle.load(f)

    assert converted == source_obj
    assert copied == copied_obj

    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    assert progress["schema"] == "pd_rew_stage_progress_v1"
    assert progress["run_id"] == "run-stage00"
    assert progress["worker_id"] == "w0"
    assert progress["case_id"] == "case0"
    assert progress["attempt_id"] == "attempt0"
    assert progress["stage"] == "stage00"
    assert progress["phase"] == "completed"
    assert progress["completed_units"] == 2
    assert progress["total_units"] == 2
    assert progress["unit"] == "file"
    assert progress["message"] == "SSDR Step 00 complete."
    assert progress["extra"]["source_file_count"] == 2
    assert progress["extra"]["processed_count"] == 2
    assert progress["extra"]["converted_json_count"] == 1
    assert progress["extra"]["copied_pkl_count"] == 1
