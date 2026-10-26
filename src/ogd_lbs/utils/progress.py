# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Small dependency-free progress reporting utility.

Environment:
    OGD_PROGRESS=auto   default; TTY bar if possible, /dev/tty bar under tee,
                        sparse log only when no terminal is available
    OGD_PROGRESS=tty    force writing a bar to /dev/tty if available
    OGD_PROGRESS=bar    force writing a bar to the provided stream
    OGD_PROGRESS=log    force sparse checkpoint logs
    OGD_PROGRESS=off    disable progress output

Design:
    - No tqdm dependency.
    - In normal terminal use, uses one carriage-return line.
    - Under `... 2>&1 | tee file.log`, writes the progress bar directly to
      /dev/tty when available, so run.log is not polluted by progress lines.
    - In non-interactive jobs without a controlling terminal, falls back to
      sparse checkpoint logs unless disabled.
"""

import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import TextIO


_VALID_MODES = {"auto", "tty", "bar", "log", "off"}
_STATES: dict[str, "_ProgressState"] = {}
_TTY_STREAM: TextIO | None = None


@dataclass
class _ProgressState:
    last_emit_time: float = 0.0
    last_current: int = 0
    last_line_len: int = 0


def _env_mode() -> str:
    value = os.environ.get("OGD_PROGRESS", "auto").strip().lower()
    return value if value in _VALID_MODES else "auto"


def _open_tty() -> TextIO | None:
    global _TTY_STREAM

    if _TTY_STREAM is not None:
        return _TTY_STREAM

    try:
        if os.path.exists("/dev/tty"):
            _TTY_STREAM = open("/dev/tty", "w", buffering=1, encoding="utf-8", errors="replace")
            return _TTY_STREAM
    except Exception:
        return None

    return None


def _stream_is_tty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def resolve_progress_output(stream: TextIO | None = None) -> tuple[str, TextIO | None]:
    """
    Resolve progress mode and output stream.

    Returns:
        (mode, output_stream)

    mode can be:
        off
        log
        bar
    """
    mode = _env_mode()
    stream = stream or sys.stderr

    if mode == "off":
        return "off", None

    if mode == "log":
        return "log", stream

    if mode == "bar":
        return "bar", stream

    if mode == "tty":
        tty = _open_tty()
        if tty is not None:
            return "bar", tty
        return "log", stream

    # auto
    if _stream_is_tty(stream):
        return "bar", stream

    # Important case: output is piped to tee, but the process still has a
    # controlling terminal. Use it so log files do not receive progress lines.
    tty = _open_tty()
    if tty is not None:
        return "bar", tty

    return "log", stream


def get_progress_mode(stream: TextIO | None = None) -> str:
    mode, _out = resolve_progress_output(stream)
    return mode


def _default_every(total: int) -> int:
    total = max(int(total), 1)
    return max(1, total // 20)


def _format_progress_line(
    *,
    label: str,
    current: int,
    total: int,
    extra: str = "",
    width: int = 32,
) -> str:
    total = max(int(total), 1)
    current = max(0, min(int(current), total))
    ratio = current / float(total)

    filled = int(round(width * ratio))
    filled = max(0, min(filled, width))

    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100.0

    if extra:
        return f"[{bar}] {percent:6.2f}% {current}/{total} | {label} | {extra}"

    return f"[{bar}] {percent:6.2f}% {current}/{total} | {label}"


def progress_step(
    label: str,
    current: int,
    total: int,
    *,
    extra: str = "",
    every: int | None = None,
    min_interval: float = 0.20,
    stream: TextIO | None = None,
    key: str | None = None,
) -> None:
    """
    Emit one progress update.

    In bar mode:
        update one terminal line and finish with newline at completion.

    In log mode:
        emit sparse checkpoint lines only.

    In off mode:
        do nothing.
    """
    mode, out_stream = resolve_progress_output(stream)

    if mode == "off" or out_stream is None:
        return

    total_i = max(int(total), 1)
    current_i = max(0, min(int(current), total_i))
    now = time.monotonic()

    state_key = key or str(label)
    state = _STATES.setdefault(state_key, _ProgressState())

    is_first = current_i <= 1 and state.last_current <= 0
    is_done = current_i >= total_i

    if mode == "log":
        every_i = int(every) if every is not None and int(every) > 0 else _default_every(total_i)
        should_emit = (
            is_first
            or is_done
            or current_i - state.last_current >= every_i
            or now - state.last_emit_time >= max(float(min_interval), 1.0)
        )

        if not should_emit:
            return

        line = _format_progress_line(
            label=label,
            current=current_i,
            total=total_i,
            extra=extra,
            width=20,
        )
        print(f"[progress] {line}", file=out_stream, flush=True)

        state.last_emit_time = now
        state.last_current = current_i

        if is_done:
            _STATES.pop(state_key, None)

        return

    if mode == "bar":
        if not (is_first or is_done or now - state.last_emit_time >= float(min_interval)):
            return

        try:
            term_width = shutil.get_terminal_size((120, 20)).columns
        except Exception:
            term_width = 120

        bar_width = max(10, min(40, term_width // 3))

        line = _format_progress_line(
            label=label,
            current=current_i,
            total=total_i,
            extra=extra,
            width=bar_width,
        )

        max_len = max(40, term_width - 1)
        if len(line) > max_len:
            line = line[: max_len - 3] + "..."

        padding = " " * max(0, state.last_line_len - len(line))

        try:
            out_stream.write("\r" + line + padding)
            out_stream.flush()
        except Exception:
            # Last-resort fallback: avoid failing the actual computation because
            # progress output failed.
            return

        state.last_line_len = len(line)
        state.last_emit_time = now
        state.last_current = current_i

        if is_done:
            try:
                out_stream.write("\n")
                out_stream.flush()
            except Exception:
                pass
            _STATES.pop(state_key, None)

        return


class ProgressReporter:
    """
    Context-manager style progress helper.

    Example:
        with ProgressReporter(total, "Bake Action") as p:
            for i in range(total):
                ...
                p.update(i + 1, extra="frame=10")
    """

    def __init__(
        self,
        total: int,
        label: str,
        *,
        every: int | None = None,
        min_interval: float = 0.20,
        stream: TextIO | None = None,
        key: str | None = None,
    ) -> None:
        self.total = int(total)
        self.label = str(label)
        self.every = every
        self.min_interval = float(min_interval)
        self.stream = stream
        self.key = key or f"{id(self)}:{self.label}"

    def update(self, current: int, *, extra: str = "") -> None:
        progress_step(
            self.label,
            current,
            self.total,
            extra=extra,
            every=self.every,
            min_interval=self.min_interval,
            stream=self.stream,
            key=self.key,
        )

    def close(self) -> None:
        progress_step(
            self.label,
            self.total,
            self.total,
            extra="done",
            every=1,
            min_interval=0.0,
            stream=self.stream,
            key=self.key,
        )

    def __enter__(self) -> "ProgressReporter":
        self.update(0, extra="start")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()


def self_test() -> None:
    old = os.environ.get("OGD_PROGRESS")
    try:
        os.environ["OGD_PROGRESS"] = "off"
        for i in range(3):
            progress_step("off-test", i + 1, 3)

        os.environ["OGD_PROGRESS"] = "log"
        for i in range(5):
            progress_step("log-test", i + 1, 5, every=2, extra=f"i={i+1}")

        os.environ["OGD_PROGRESS"] = "auto"
        mode, out = resolve_progress_output()
        print(
            "progress.self_test OK "
            f"(auto_resolved_mode={mode}, "
            f"uses_tty={getattr(out, 'name', '') == '/dev/tty'})",
            flush=True,
        )
    finally:
        if old is None:
            os.environ.pop("OGD_PROGRESS", None)
        else:
            os.environ["OGD_PROGRESS"] = old


if __name__ == "__main__":
    self_test()
