# filepath: /workspace/lao-lbs/src/lao_lbs/utils/pipeline.py
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from collections import deque

from ogd_lbs.utils.common import extract_id

_GLOBAL_TIMESTAMP = None

def get_global_timestamp() -> str:
    """
    获取全局唯一的时间戳。在同一次流水线执行周期内，该时间戳保持固定。
    """
    global _GLOBAL_TIMESTAMP
    if _GLOBAL_TIMESTAMP is None:
        _GLOBAL_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M_%S')
    return _GLOBAL_TIMESTAMP

def detect_config_depth(source_dir_abs: Path | str, max_search_depth: int = 6) -> int:
    """
    使用广度优先搜索 (BFS) 自动探测 config_local.json 所在的层级深度。
    """
    source_path = Path(source_dir_abs)
    queue = deque([(source_path, 0)])
    
    while queue:
        curr_path, curr_depth = queue.popleft()
        
        if curr_depth > max_search_depth:
            continue
            
        if (curr_path / 'config_local.json').exists():
            return curr_depth
            
        try:
            for item in curr_path.iterdir():
                if item.is_dir():
                    queue.append((item, curr_depth + 1))
        except PermissionError:
            pass
            
    return -1

def resolve_execution_dir(target_cmp_dir: Path | str, new_out_dir: bool, prefix: str = "exec_") -> Path:
    """
    决定目标端的时间戳文件夹名称 (断点续传 vs 全新执行)。
    返回最终的绝对路径。
    """
    target_cmp_dir = Path(target_cmp_dir)
    target_out_name = None
    
    if not new_out_dir and target_cmp_dir.exists():
        # 如果不强制新建，尝试寻找已有的 prefix 文件夹
        target_sub_dirs = [d for d in target_cmp_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
        if target_sub_dirs:
            # 取最新的一个进行覆盖/断点续传
            target_out_name = sorted(target_sub_dirs, key=lambda p: p.name)[-1].name
            
    # 如果强制新建，或者目标目录里根本没有符合条件的文件夹，则生成新时间戳
    if target_out_name is None:
        target_out_name = f"{prefix}{get_global_timestamp()}"

    return target_cmp_dir / target_out_name

def extract_dataset_id(path_str: Path | str, fallback: int = 999) -> int:
    """
    从路径字符串中提取 datasetXX 的数字 ID。
    """
    match = re.search(r'dataset(\d+)', str(path_str))
    return int(match.group(1)) if match else fallback

def parse_iter_name(folder_name: Path | str, iter_max: int) -> tuple[int, str]:
    """
    解析 iter 文件夹名称，返回 (iter_id, 标准化文件夹名)。
    """
    name = Path(folder_name).name
    if name == "opt_out":
        iter_id = iter_max
    else:
        iter_id = extract_id(name, fallback=-1)
        
    if iter_id == iter_max and iter_max >= 0:
        standard_name = "opt_out"
    else:
        standard_name = f"iter_{iter_id}"
        
    return iter_id, standard_name