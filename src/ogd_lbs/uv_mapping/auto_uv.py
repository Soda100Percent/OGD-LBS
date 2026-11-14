# filepath: /workspace/lao-lbs/src/ogd_lbs/uv_mapping/auto_uv.py
from __future__ import annotations

import numpy as np
import xatlas

from ogd_lbs.fitbox.schema import KEY_VERTICES, KEY_FACES, KEY_FACE_UVS
from ogd_lbs.utils.common import log, warn

def generate_fitbox_uvs(mesh_obj: dict, method: str = "xatlas", overwrite: bool = False) -> dict:
    """
    为 fitbox 网格数据自动生成 UV。
    该方法不会改变原始顶点数量，完美兼容现有的蒙皮权重和动画数据。

    参数:
        mesh_obj: fitbox 格式的网格字典
        method: 使用的 UV 展开算法，目前支持 "xatlas"
        overwrite: 如果已经存在 UV，是否强制覆盖重新生成
    """
    # 1. 检查是否需要生成
    if not overwrite and KEY_FACE_UVS in mesh_obj:
        existing_uvs = np.array(mesh_obj[KEY_FACE_UVS])
        if len(existing_uvs) > 0:
            log("UVs already exist and overwrite is False. Skipping UV generation.")
            return mesh_obj

    # 2. 提取顶点和面数据 (确保类型正确，防止 C++ 绑定报错)
    vertices = np.array(mesh_obj.get(KEY_VERTICES, []), dtype=np.float32)
    faces = np.array(mesh_obj.get(KEY_FACES, []), dtype=np.int32)

    if len(vertices) == 0 or len(faces) == 0:
        warn("Mesh has no vertices or faces. Skipping UV generation.")
        return mesh_obj

    # 3. 算法路由
    if method == "xatlas":
        # 调用 xatlas 进行自动展 UV
        # vmapping: 新顶点到旧顶点的映射
        # new_indices: 展 UV 后的新面索引 (F, 3)
        # new_uvs: 生成的 UV 坐标 (V_new, 2)
        vmapping, new_indices, new_uvs = xatlas.parametrize(vertices, faces)

        # 转换为 FitBox 要求的 (面数, 3, 2) 面角格式
        # 利用 NumPy 的高级索引，直接用 new_indices 去索引 new_uvs
        face_uvs = new_uvs[new_indices]

    # elif method == "other_algo":
    #     预留给未来的其他算法
    else:
        raise ValueError(f"Unsupported UV mapping method: {method}")

    # 4. 写入回 mesh_obj
    mesh_obj[KEY_FACE_UVS] = face_uvs.tolist()

    return mesh_obj
