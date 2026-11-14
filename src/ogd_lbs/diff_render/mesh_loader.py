# filepath: src/ogd_lbs/diff_render/mesh_loader.py
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict

from ogd_lbs.utils.common import log, warn, load_data, dump_pkl
from ogd_lbs.diff_render.nvdiffmodeling import mesh, material

# 🌟 混合材质映射字典
#
# ID 约定：
#   0: diffuse                         legacy diffuse，语义接近 diffuse_2sided
#   1: ambient_1sided                  无光照，背面 alpha=0
#   2: ambient_2sided                  无光照，双面可见
#   3: diffuse_1sided                  diffuse 光照，背面 alpha=0
#   4: diffuse_2sided                  diffuse 光照，双面可见
#   5: diffuse_2sided_back_deepen      diffuse 光照，双面可见，但背面颜色加深
#
# 重要：
#   two-stage Stage01 几何阶段会将：
#       ambient_1sided -> diffuse_2sided_back_deepen
#       ambient_2sided -> diffuse_2sided
#
#   因此 back_deepen 语义由 t_bsdf_id 逐面携带，而不是由 material["back_deepen"]
#   全局控制。这样 ambient_2sided 转换来的面不会被误加深。
BSDF_TO_ID = {
    "diffuse": 0,

    "ambient_1sided": 1,
    "ambient_2sided": 2,

    "diffuse_1sided": 3,
    "diffuse_2sided": 4,

    "diffuse_2sided_back_deepen": 5,

    # 可选别名，方便命令行或旧实验配置手写。
    "diffuse_2sided_deep": 5,
    "diffuse_back_deepen": 5,
}

BSDF_ID_TO_NAME = {
    0: "diffuse",
    1: "ambient_1sided",
    2: "ambient_2sided",
    3: "diffuse_1sided",
    4: "diffuse_2sided",
    5: "diffuse_2sided_back_deepen",
}

def _bsdf_name_to_id(bsdf_name: str) -> int:
    key = str(bsdf_name).strip().lower()

    if key not in BSDF_TO_ID:
        warn(
            f"未知 BSDF 名称 '{bsdf_name}'，已回退为 diffuse。"
            f"可用值: {list(BSDF_TO_ID.keys())}"
        )
        return BSDF_TO_ID["diffuse"]

    return BSDF_TO_ID[key]


class MeshLoader:
    def __init__(self, device: torch.device):
        self.device = device

    def _get_mesh_paths(self, folder: Path) -> List[Path]:
        exclude_stems = {
            "animation_data",
            "skeleton_hierarchy_data",
            "animation_data_bone_reduced",
            "skeleton_hierarchy_data_bone_reduced",
            "step00_convert_summary",
            "step01_frame_sampling_summary",
            "step02_reduce_bone_weight_summary",
            "step03_qem_summary",
            "step04_optimize_summary",
            "step04_test_summary",
            "step05_diff_render_summary",
            "target_export_summary",
            "fold_plan",
            "case_parameters",
            "stage_result",
            "stage_progress",
            "normalized_metrics_v1",
            "body_bone_alpha",
            "handoff_manifest",
            "export_summary",
            "manifest",
            "config_run",
            "config_local",
            "config_local_iter",
            "render_meta",
            "metrics_01",
            "metrics_02",
            "metrics_04",
        }
        exclude_dirs = {
            "renders",
            "bone_reduced_versions",
            "opt_out_test",
            "masks",
            "textures",
            "previews_transparent",
            "previews_checkerboard",
            "eval_images",
        }
        ext_priority = [".pkl", ".pickle", ".json"]

        files_by_stem_dir = {}
        for ext in ext_priority:
            for p in folder.rglob(f"*{ext}"):
                rel_parts = set(p.relative_to(folder).parts[:-1])
                if rel_parts & exclude_dirs:
                    continue
                if (
                    p.stem in exclude_stems
                    or p.stem.endswith("_summary")
                    or p.stem.startswith("metrics_")
                    or p.stem.startswith("normalized_metrics")
                ):
                    continue
                key = (p.parent, p.stem)
                if key not in files_by_stem_dir:
                    files_by_stem_dir[key] = p

        paths = list(files_by_stem_dir.values())
        paths.sort(key=lambda x: x.name)
        return paths

    def _get_mesh_info(self, sm_path: Path, ignore_suffixes: List[str], sm_obj: dict = None) -> dict:
        if sm_obj is None:
            sm_obj = load_data(sm_path)

        raw_stem = sm_path.stem
        raw_name = sm_obj.get('name', raw_stem)

        if raw_stem.lower() != raw_name.lower():
            warn(f"网格文件名 '{raw_stem}' 与内部 name '{raw_name}' 不一致！")

        clean_stem = raw_stem
        clean_name = raw_name

        for s in ignore_suffixes:
            if clean_stem.lower().endswith(s.lower()):
                clean_stem = clean_stem[:-len(s)]
            if clean_name.lower().endswith(s.lower()):
                clean_name = clean_name[:-len(s)]

        return {
            "path": sm_path,  # 🌟 保留绝对路径，供 Trainer 读取 JSON 使用
            "raw_stem": raw_stem,
            "raw_name": raw_name,
            "clean_stem": clean_stem,
            "clean_name": clean_name
        }

    def scan_mesh_names(self, folder: str | Path, ignore_suffixes: List[str]) -> List[dict]:
        folder = Path(folder)
        sm_paths = self._get_mesh_paths(folder)

        infos = []
        for sm_path in sm_paths:
            infos.append(self._get_mesh_info(sm_path, ignore_suffixes))
        return infos

    def load_base_mesh(self, folder: str | Path, mats: list, ignore_suffixes: List[str], bsdf_rules: Optional[List[List[str]]] = None):
        folder = Path(folder)
        log(f"开始加载基础网格 (Base Mesh) 从: {folder}")

        from ogd_lbs.utils.common import resolve_specific_file
        a_path = resolve_specific_file(folder, "animation_data", [".pkl", ".pickle", ".json"])
        if not a_path:
            raise FileNotFoundError(f"Missing animation_data in {folder}")

        a_obj = load_data(a_path)
        poses = np.array(a_obj["animation_poses"])
        bones_count = poses.shape[1]

        sm_paths = self._get_mesh_paths(folder)

        t, t_dict = [], {}
        full_face_uvs = np.zeros((0, 3, 2))
        f = np.zeros((0, 3), dtype=int)
        tf = np.zeros((0, 3), dtype=int)
        mf = np.zeros((0,), dtype=int)
        v = np.zeros((0, 3))
        n = np.zeros((0, 3))
        vbw = np.zeros((0, bones_count))

        t_bsdf_id = np.zeros((0, 1), dtype=np.int64)

        v_offset, f_offset = [0], [0]
        mesh_names = []

        for i, sm_path in enumerate(sm_paths):
            sm_obj = load_data(sm_path)
            info = self._get_mesh_info(sm_path, ignore_suffixes, sm_obj)
            clean_stem = info["clean_stem"]
            mesh_names.append(clean_stem)

            sub_v = np.array(sm_obj['vertices'])
            sub_n = np.array(sm_obj['vertex_normals'])
            sub_vbw = np.array(sm_obj['vertex_bone_weights'])
            face_uvs = np.array(sm_obj['face_uvs'])
            sub_f = np.array(sm_obj['faces'])

            bsdf_id = BSDF_TO_ID["diffuse"]
            if bsdf_rules:
                for rule in bsdf_rules:
                    if len(rule) == 2 and str(rule[0]).lower() in clean_stem.lower():
                        bsdf_id = _bsdf_name_to_id(rule[1])
                        break
            sub_t_bsdf_id = np.full((len(sub_f), 1), bsdf_id, dtype=np.int64)

            v_offset.append(sub_v.shape[0] + v_offset[-1])
            f_offset.append(sub_f.shape[0] + f_offset[-1])

            v = np.concatenate((v, sub_v), axis=0)
            n = np.concatenate((n, sub_n), axis=0)
            vbw = np.concatenate((vbw, sub_vbw), axis=0)
            full_face_uvs = np.concatenate((full_face_uvs, face_uvs), axis=0)
            t_bsdf_id = np.concatenate((t_bsdf_id, sub_t_bsdf_id), axis=0)

            sub_f += v_offset[-2]
            f = np.concatenate((f, sub_f), axis=0)
            mat_idx = i if i < len(mats) else 0
            mf = np.concatenate((mf, np.full(len(sub_f), mat_idx)), axis=0)

            sub_tf = np.zeros((face_uvs.shape[0], face_uvs.shape[1]), dtype=int)
            for f_id, f_uv in enumerate(face_uvs):
                for f_uv_id, uv in enumerate(f_uv):
                    epsilon = 1e-5
                    uv_tuple = (max(epsilon, min(uv[0], 1 - epsilon)), max(epsilon, min(uv[1], 1 - epsilon)))
                    if uv_tuple in t_dict:
                        sub_tf[f_id, f_uv_id] = t_dict[uv_tuple]
                    else:
                        idx = len(t)
                        t_dict[uv_tuple] = idx
                        t.append(uv_tuple)
                        sub_tf[f_id, f_uv_id] = idx
            tf = np.concatenate((tf, sub_tf), axis=0)

        uber_material, t, tf = material.merge_materials(mats, t, tf, mf)

        vertices = torch.tensor(v, dtype=torch.float32, device=self.device)
        texcoords = torch.tensor(t, dtype=torch.float32, device=self.device) if len(t) > 0 else None
        normals = torch.tensor(n, dtype=torch.float32, device=self.device) if len(n) > 0 else None
        faces = torch.tensor(f, dtype=torch.int64, device=self.device)
        tfaces = torch.tensor(tf, dtype=torch.int64, device=self.device) if texcoords is not None else None
        nfaces = torch.tensor(f, dtype=torch.int64, device=self.device) if normals is not None else None
        v_weights = torch.tensor(vbw, dtype=torch.float32, device=self.device) if len(vbw) > 0 else None
        t_bsdf_id_tensor = torch.tensor(t_bsdf_id, dtype=torch.int64, device=self.device)

        base_mesh = mesh.Mesh(vertices, faces, normals, nfaces, texcoords, tfaces,
                              v_weights=v_weights, bone_mtx=None, material=uber_material, t_bsdf_id=t_bsdf_id_tensor)

        log(f"基础网格加载完成！总顶点数: {vertices.shape[0]}, 总面数: {faces.shape[0]}")
        return base_mesh, mesh_names, v_offset, f_offset, full_face_uvs

    def load_ref_mesh_sequence(self, folder: str | Path, mats: list, ignore_suffixes: List[str],
                               use_face_materials: bool = False,
                               material_indices: Optional[List[int]] = None,
                               bsdf_rules: Optional[List[List[str]]] = None):
        folder = Path(folder)
        log(f"开始加载目标网格序列 (Target Sequence) 从: {folder} | 启用多材质贴图: {use_face_materials}")

        from ogd_lbs.utils.common import resolve_specific_file
        a_path = resolve_specific_file(folder, "animation_data", [".pkl", ".pickle", ".json"])
        if not a_path:
            raise FileNotFoundError(f"Missing animation_data in {folder}")

        a_obj = load_data(a_path)
        poses = np.array(a_obj["animation_poses"])
        poses_count = poses.shape[0]

        sm_paths = self._get_mesh_paths(folder)

        t, t_dict = [], {}
        f = np.zeros((0, 3), dtype=int)
        tf = np.zeros((0, 3), dtype=int)
        mf = np.zeros((0,), dtype=int)

        ba_v = np.zeros((poses_count, 0, 3))
        ba_n = np.zeros((poses_count, 0, 3))

        t_bsdf_id = np.zeros((0, 1), dtype=np.int64)

        v_offset, f_offset = [0], [0]
        m_offset = 0
        mesh_names = []

        for i, sm_path in enumerate(sm_paths):
            sm_obj = load_data(sm_path)
            info = self._get_mesh_info(sm_path, ignore_suffixes, sm_obj)
            clean_stem = info["clean_stem"]
            mesh_names.append(clean_stem)

            sub_ba_v = np.array(sm_obj['baked_animation_vertices'])
            sub_ba_n = np.array(sm_obj['baked_animation_vertex_normals'])
            face_uvs = np.array(sm_obj['face_uvs'])
            sub_f = np.array(sm_obj['faces'])

            bsdf_id = BSDF_TO_ID["diffuse"]
            if bsdf_rules:
                for rule in bsdf_rules:
                    if len(rule) == 2 and str(rule[0]).lower() in clean_stem.lower():
                        bsdf_id = _bsdf_name_to_id(rule[1])
                        break
            sub_t_bsdf_id = np.full((len(sub_f), 1), bsdf_id, dtype=np.int64)

            if use_face_materials:
                mtl_names = sm_obj.get('material_names', ['default'])
                face_materials = np.array(sm_obj.get('face_materials', np.zeros(len(sub_f))))
                face_materials += m_offset
                m_offset += len(mtl_names)
                mf = np.concatenate((mf, face_materials), axis=0)
            else:
                if material_indices is not None:
                    mat_idx = material_indices[i]
                else:
                    mat_idx = i if i < len(mats) else 0
                mf = np.concatenate((mf, np.full(len(sub_f), mat_idx)), axis=0)

            v_offset.append(sub_ba_v.shape[1] + v_offset[-1])
            f_offset.append(sub_f.shape[0] + f_offset[-1])

            ba_v = np.concatenate((ba_v, sub_ba_v), axis=1)
            ba_n = np.concatenate((ba_n, sub_ba_n), axis=1)
            t_bsdf_id = np.concatenate((t_bsdf_id, sub_t_bsdf_id), axis=0)

            sub_f += v_offset[-2]
            f = np.concatenate((f, sub_f), axis=0)

            sub_tf = np.zeros((face_uvs.shape[0], face_uvs.shape[1]), dtype=int)
            for f_id, f_uv in enumerate(face_uvs):
                for f_uv_id, uv in enumerate(f_uv):
                    epsilon = 1e-5
                    uv_tuple = (max(epsilon, min(uv[0], 1 - epsilon)), max(epsilon, min(uv[1], 1 - epsilon)))
                    if uv_tuple in t_dict:
                        sub_tf[f_id, f_uv_id] = t_dict[uv_tuple]
                    else:
                        idx = len(t)
                        t_dict[uv_tuple] = idx
                        t.append(uv_tuple)
                        sub_tf[f_id, f_uv_id] = idx
            tf = np.concatenate((tf, sub_tf), axis=0)

        uber_material, t, tf = material.merge_materials(mats, t, tf, mf)
        t_bsdf_id_tensor = torch.tensor(t_bsdf_id, dtype=torch.int64, device=self.device)

        ref_mesh_sequence = []
        for p_id in range(poses_count):
            vertices = torch.tensor(ba_v[p_id], dtype=torch.float32, device=self.device)
            texcoords = torch.tensor(t, dtype=torch.float32, device=self.device) if len(t) > 0 else None
            n_arr = ba_n[p_id]
            normals = torch.tensor(n_arr, dtype=torch.float32, device=self.device) if len(n_arr) > 0 else None
            faces = torch.tensor(f, dtype=torch.int64, device=self.device)
            tfaces = torch.tensor(tf, dtype=torch.int64, device=self.device) if texcoords is not None else None
            nfaces = torch.tensor(f, dtype=torch.int64, device=self.device) if normals is not None else None

            ref_mesh = mesh.Mesh(vertices, faces, normals, nfaces, texcoords, tfaces, material=uber_material, t_bsdf_id=t_bsdf_id_tensor)
            ref_mesh_sequence.append(ref_mesh)

        log(f"参考网格序列加载完毕！序列长度(帧数): {len(ref_mesh_sequence)}")
        return ref_mesh_sequence, mesh_names, v_offset, f_offset

    def save_base_mesh(self, folder: str | Path, opt_mesh: mesh.Mesh, mesh_names: list,
                       v_offset: list, f_offset: list, face_uvs: np.ndarray):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        v = opt_mesh.v_pos.clone().detach().cpu().numpy()
        n = opt_mesh.v_nrm.clone().detach().cpu().numpy()
        f = opt_mesh.t_pos_idx.clone().detach().cpu().numpy()
        vbw = opt_mesh.v_weights.clone().detach().cpu().numpy()

        for i in range(len(v_offset) - 1):
            v_idx_from, v_idx_to = v_offset[i], v_offset[i + 1]
            f_idx_from, f_idx_to = f_offset[i], f_offset[i + 1]

            name = mesh_names[i]
            sm_obj = {
                'name': name,
                'vertices': v[v_idx_from:v_idx_to, :].tolist(),
                'vertex_normals': n[v_idx_from:v_idx_to, :].tolist(),
                'vertex_bone_weights': vbw[v_idx_from:v_idx_to, :].tolist(),
                'faces': (f[f_idx_from:f_idx_to, :] - v_idx_from).tolist(),
                'face_uvs': face_uvs[f_idx_from:f_idx_to, :].tolist()
            }

            # 🌟 扁平化：直接存入 geometry 文件夹，不再创建子文件夹
            sm_path = folder / f"{name}.pkl"
            dump_pkl(sm_path, sm_obj)

        log(f"已成功将优化后的网格拆分并导出至: {folder} (PKL 格式)")
