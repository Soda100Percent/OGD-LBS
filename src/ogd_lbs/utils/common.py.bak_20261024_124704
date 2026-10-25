# filepath: /workspace/lao-lbs/src/lao_lbs/utils/common.py
from __future__ import annotations

import os
import re
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Iterable

# -----------------------------------------------------------------------------
# 优雅降级：尝试导入 orjson，如果失败则使用标准库 json
# -----------------------------------------------------------------------------
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    import json
    HAS_ORJSON = False

# -----------------------------------------------------------------------------
# Global Metrics Registry (实验指标全局注册表)
# -----------------------------------------------------------------------------
_GLOBAL_METRICS: dict[str, Any] = {}

def record_metric(key: str, value: Any) -> None:
    """记录自定义业务指标到全局注册表"""
    _GLOBAL_METRICS[key] = value

def dump_metrics(path: str | Path) -> None:
    """将全局注册表中的所有指标一键导出为 JSON"""
    dump_json(path, _GLOBAL_METRICS)
    log(f"Metrics successfully dumped to {path}")

# -----------------------------------------------------------------------------
# Logging & CLI (已升级：增加时间戳)
# -----------------------------------------------------------------------------

def _get_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg: str) -> None:
    print(f"[{_get_timestamp()}][ogd-lbs] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f"[{_get_timestamp()}][ogd-lbs][WARN] {msg}", flush=True)

def fail(msg: str) -> None:
    error_msg = f"[{_get_timestamp()}][ogd-lbs][ERROR] {msg}"
    print(error_msg, flush=True) # 抛出异常前先打印，确保终端能直观看到时间
    raise RuntimeError(msg)

def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]

# -----------------------------------------------------------------------------
# Data IO (JSON & Pickle)
# -----------------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    path = Path(path)
    if HAS_ORJSON:
        with path.open("rb") as f:
            return orjson.loads(f.read())
    else:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

def dump_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_ORJSON:
        with path.open("wb") as f:
            f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

def load_pkl(path: str | Path) -> Any:
    import pickle
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)

def dump_pkl(path: str | Path, obj: Any) -> None:
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(path: str | Path) -> Any:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in [".pkl", ".pickle"]:
        return load_pkl(path)
    elif ext == ".json":
        return load_json(path)
    else:
        raise ValueError(f"Unsupported data format extension: {ext} for file {path}")

# -----------------------------------------------------------------------------
# File Discovery & Conflict Resolution
# -----------------------------------------------------------------------------

def resolve_specific_file(dir_path: str | Path, stem: str, ext_priority: list[str]) -> Optional[Path]:
    dir_path = Path(dir_path)
    ext_priority_lower = [ext.lower() for ext in ext_priority]
    
    for ext in ext_priority_lower:
        candidate = dir_path / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None

def discover_and_resolve_files(
    dir_path: str | Path, 
    ext_priority: list[str], 
    exclude_stems: Optional[Iterable[str]] = None
) -> list[Path]:
    dir_path = Path(dir_path)
    exclude_set = set(exclude_stems) if exclude_stems else set()
    ext_priority_lower = [ext.lower() for ext in ext_priority]
    
    files_by_stem: dict[str, Path] = {}
    
    if not dir_path.exists():
        return []
        
    for item in dir_path.iterdir():
        if not item.is_file():
            continue
            
        stem = item.stem
        ext = item.suffix.lower()
        
        if stem in exclude_set:
            continue
        if ext not in ext_priority_lower:
            continue
            
        if stem not in files_by_stem:
            files_by_stem[stem] = item
        else:
            current_best = files_by_stem[stem]
            if ext_priority_lower.index(ext) < ext_priority_lower.index(current_best.suffix.lower()):
                files_by_stem[stem] = item
                
    return sorted(list(files_by_stem.values()), key=lambda p: p.name)

# -----------------------------------------------------------------------------
# File & Directory Operations
# -----------------------------------------------------------------------------

def rm_dir_contents(dir_path: str | Path) -> None:
    dir_path = Path(dir_path)
    if dir_path.exists():
        if dir_path.is_dir():
            shutil.rmtree(dir_path)
            log(f"Dir and contents removed: {dir_path}")
        else:
            raise IOError(f"Not a valid dir path: {dir_path}")
    else:
        warn(f"Dir path doesn't exist: {dir_path}")

def create_or_clear_dir(dir_path: str | Path) -> None:
    dir_path = Path(dir_path)
    if dir_path.exists():
        rm_dir_contents(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        log(f"Dir cleared: {dir_path}")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)
        log(f"Dir created: {dir_path}")

# -----------------------------------------------------------------------------
# Path & Naming Utilities
# -----------------------------------------------------------------------------

def extract_id(folder_name: str | Path, fallback: int = 999999999) -> int:
    name = Path(folder_name).name
    match = re.search(r"\d+$", name)
    return int(match.group()) if match else fallback

def resolve_under_root(root: Path, maybe_rel_path: str | Path) -> Path:
    p = Path(maybe_rel_path)
    if p.is_absolute():
        return p
    return root / p

def safe_relpath(path: str | Path | None, root: str | Path) -> Optional[str]:
    if path is None:
        return None
    return os.path.relpath(str(path), str(root))

def _remove_parts(s: str, parts_to_remove: tuple[str, ...]) -> str:
    parts = s.split('_')
    parts = [p for p in parts if p not in parts_to_remove]
    return '_'.join(parts)

def _contains_parts(s: str, parts_to_check: tuple[str, ...]) -> bool:
    return any(part in parts_to_check for part in s.split('_'))

def remove_mark_parts(dir_path: str | Path, parts_to_remove: tuple[str, ...] = ('pass', 'done')) -> str:
    dir_path = Path(dir_path)
    dir_name = dir_path.name
    parent_path = dir_path.parent
    new_dir_name = _remove_parts(dir_name, parts_to_remove)
    return str(parent_path / new_dir_name)

def is_obsolete(dir_path: str | Path, skip_marks: tuple[str, ...] = ('obsolete',)) -> bool:
    return _contains_parts(Path(dir_path).name, skip_marks)

def is_done(dir_path: str | Path, skip_marks: tuple[str, ...] = ('done',)) -> bool:
    return _contains_parts(Path(dir_path).name, skip_marks)

def should_be_skipped(dir_path: str | Path, skip_marks: tuple[str, ...] = ('pass', 'done')) -> bool:
    return _contains_parts(Path(dir_path).name, skip_marks)

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    if not hex_str:
        return (0.0, 1.0, 0.0)
    value = hex_str.strip().lstrip("#")
    if len(value) != 6:
        warn(f"Invalid hex color '{hex_str}', fallback to green #00FF00")
        return (0.0, 1.0, 0.0)
    try:
        return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore
    except Exception:
        warn(f"Invalid hex color '{hex_str}', fallback to green #00FF00")
        return (0.0, 1.0, 0.0)

class Timer:
    """
    智能性能计时器上下文管理器。
    升级版：无论是否开启 profiling，都会将耗时记录到 _GLOBAL_METRICS 中。
    """
    _enabled: bool | None = None

    def __init__(self, name: str):
        self.name = name
        if Timer._enabled is None:
            Timer._enabled = "--enable_profiling" in sys.argv
        self.enabled = Timer._enabled
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        record_metric(self.name, elapsed)
        if self.enabled:
            log(f"[PROFILER] {self.name} took {elapsed:.4f} seconds")