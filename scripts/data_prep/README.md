

# OGD-LBS Data Prep Pipeline

本目录包含 OGD-LBS 的通用 Data Prep 入口脚本，用于将已经清洗好的商业角色 `.blend` 文件转换为后续 SSDR 阶段可使用的 Fitbox 数据。

当前 Data Prep 主要支持：

```text
Griffin_A2
Winter_Samurai
```

其中：

- `Griffin_A2` 使用标准 `.blend` 输入；
- `Winter_Samurai` 需要先经过 `scripts/custom_model_clean/winter_samurai/` 清洗，生成 `Winter_Samurai_Normalized.blend`；
- `Vroid_18` / `Vroid_19` / `Vroid_20` 仍是旧布局，当前 runner 会识别但不支持。

---

## 1. 目标

Data Prep 的目标是将商业角色资产从 Blender `.blend` 形态转换为 Fitbox / SSDR 输入数据。

完整流程包括：

1. 将所有动作拼接为单条 NLA 时间轴；
2. 对多组件 mesh 进行 SDF / VDB-like 融合；
3. 保留融合前 source mesh lineage；
4. 建立 VDB mesh 与原始 source mesh 之间的 Surface Deform 关系；
5. 生成 Surface drive mode 与 LBS drive mode 两套 `.blend`；
6. 创建 `Armature_Original` 与 `Armature_Cleaned_Full`；
7. 根据配置生成 conservative / moderate / aggressive 等额外减骨版本；
8. 导出 Fitbox 数据；
9. 为 SSDR handoff 准备 `fitbox_vdb/`。

最终主要输出为：

```text
data_prep_work_dir/data_prep_05/fitbox_vdb/
```

该目录可同步到：

```text
ssdr_work_dir/data0_json/
```

作为 SSDR Step 00 的输入。

---

## 2. 数据准备

默认正式数据集根目录为：

```text
/workspace/ogd_dataset
```

推荐布局如下：

```text
/workspace/ogd_dataset/
├── Griffin_A2/
│   ├── Griffin_A2.blend
│   └── data_prep_work_dir/
│
└── Winter_Samurai/
    ├── Winter_Samurai_Normalized.blend
    └── data_prep_work_dir/
```

### 2.1 Griffin_A2 输入

Griffin_A2 直接使用：

```text
/workspace/ogd_dataset/Griffin_A2/Griffin_A2.blend
```

### 2.2 Winter_Samurai 输入

Winter Samurai 需要先运行 custom cleaning 管线，得到：

```text
/workspace/ogd_dataset/Winter_Samurai/Winter_Samurai_Normalized.blend
```

该文件由：

```text
scripts/custom_model_clean/winter_samurai/
```

生成。

Data Prep 不直接处理 Winter Samurai 原始 FBX。

---

## 3. 目录结构

当前 Data Prep 目录为：

```text
scripts/data_prep/
├── _env.py
├── manifest.py
├── paths.py
├── prep_01_concat_nla.py
├── prep_02_sdf_pipeline.sh
├── prep_02a_sdf_fuse_5x.py
├── prep_02b_import_fused_4x.py
├── prep_03_surface_deform.py
├── prep_04a_bone_pruning.py
├── prep_05_export_fitbox.py
├── run_dataset_data_prep.sh
├── configs/
│   ├── prep_02_sdf_merge_config_griffin_a2.json
│   ├── prep_02_sdf_merge_config_winter_samurai.json
│   ├── extra_bone_reduce_griffin_a2.json
│   └── extra_bone_reduce_winter_samurai.json
└── tests/
    ├── __init__.py
    ├── test_data_prep_paths.py
    └── test_fitbox_sampling.py
```

底层通用工具位于：

```text
src/ogd_lbs/
├── blender/
├── fitbox/
└── utils/
```

---

## 4. Data Prep 工作目录结构

标准工作目录为：

```text
data_prep_work_dir/
├── prep_02_sdf_merge_config.json
├── extra_bone_reduce_config.json
├── data_prep_00/
├── data_prep_01/
├── data_prep_02/
├── data_prep_03/
├── data_prep_04/
├── data_prep_05/
└── logs/
```

各阶段含义：

|阶段|输入|输出|说明|
|---|---|---|---|
|`data_prep_00`|原始 / 清洗后 `.blend`|—|Data Prep 输入|
|`data_prep_01`|`data_prep_00`|`*_prep_01.blend`|NLA 拼接|
|`data_prep_02`|`data_prep_01`|`*_prep_02.blend` + `sdf_exports/`|SDF fusion + fused OBJ 导回|
|`data_prep_03`|`data_prep_02`|Surface / LBS drive mode `.blend`|Surface Deform + LBS transfer|
|`data_prep_04`|`data_prep_03`|cleaned-full + bone reduced variants|骨架清理与配置式减骨|
|`data_prep_05`|`data_prep_04`|Fitbox 数据|导出 SSDR 可用输入|

---

## 5. 完整流程概览

完整 Data Prep 流程为：

```text
00 raw / normalized .blend
  → 01 NLA concat
  → 02a SDF fusion in Blender 5.x
  → 02b import fused OBJ in Blender 4.x
  → 03 Surface Deform + LBS drive modes
  → 04a cleaned-full armature + configured bone-reduced variants
  → 05 Fitbox export
```

最终关键输出：

```text
data_prep_work_dir/data_prep_05/fitbox_vdb/
```

其中包含：

```text
skeleton_hierarchy_data.pkl
animation_data.pkl
<VDB mesh>.pkl
<unmerged mesh>.pkl
bone_reduced_versions/
```

---

## 6. Step 01：NLA 拼接

脚本：

```text
prep_01_concat_nla.py
```

职责：

- 从 `data_prep_00/` 读取唯一 raw `.blend`；
- 找到目标 `Armature`；
- 将所有 Actions 拼接到一个 NLA track；
- 输出 `*_prep_01.blend`。

典型命令：

```bash
cd /workspace/git_projects/OGD-LBS

conda run --no-capture-output -n lao-lbs \
  python scripts/data_prep/prep_01_concat_nla.py \
  --input_dir /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir \
  --gap 0
```

输出示例：

```text
data_prep_work_dir/data_prep_01/Winter_Samurai_prep_01.blend
```

---

## 7. Step 02：SDF fusion + fused OBJ 导回

Step 02 分为两个子步骤：

```text
02a: Blender 5.x SDF Grid fusion
02b: Blender 4.x import fused OBJ
```

推荐使用 wrapper：

```text
prep_02_sdf_pipeline.sh
```

### 7.1 Prep 02a：SDF fusion

脚本：

```text
prep_02a_sdf_fuse_5x.py
```

职责：

- 读取 Prep 01 `.blend`；
- 读取 SDF merge config；
- 校验 mesh coverage；
- 对指定 mesh group 做 SDF / VDB-like fusion；
- 导出 fused OBJ；
- 生成 manifest。

输出：

```text
data_prep_work_dir/data_prep_02/sdf_exports/
├── manifest.json
├── <VDB target>.obj
└── ...
```

### 7.2 Prep 02b：导入 fused OBJ

脚本：

```text
prep_02b_import_fused_4x.py
```

职责：

- 读取 Prep 01 `.blend`；
- 读取 Prep 02a manifest；
- 创建 `Source_<VDB>` lineage mesh；
- 导入 fused OBJ；
- 删除被 VDB 消耗的 raw source meshes；
- 保留 `unmerged_meshes`；
- 输出 Prep 02 `.blend`。

输出：

```text
data_prep_work_dir/data_prep_02/<asset>_prep_02.blend
```

### 7.3 Prep 02 config schema

当前支持新 schema：

```json
{
    "schema": "ogd_lbs_prep02_sdf_merge_config_v2",
    "asset": "Winter_Samurai",
    "unmerged_meshes": [
        "Body2",
        "Eye",
        "Lashes",
        "Mask",
        "Boots",
        "Weapon"
    ],
    "merge_groups": {
        "top": {
            "source_meshes": [
                "Armor_Cloth",
                "Armor_Hand2"
            ],
            "voxel_size": 1.0,
            "offset": 1.5,
            "auto_offset": false
        }
    }
}
```

同时兼容旧 flat schema，例如 Griffin_A2：

```json
{
    "main_mesh_VDB": {
        "source_meshes": [
            "Griffin_A"
        ],
        "voxel_size": 10.0,
        "offset": 15.0
    }
}
```

### 7.4 典型命令

```bash
cd /workspace/git_projects/OGD-LBS

bash scripts/data_prep/prep_02_sdf_pipeline.sh \
  --input_dir /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir \
  --config /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/prep_02_sdf_merge_config.json \
  --bpy5_env ogd-bpy52 \
  --bpy4_env lao-lbs
```

---

## 8. Step 03：Surface Deform + LBS drive modes

脚本：

```text
prep_03_surface_deform.py
```

职责：

- 读取 Prep 02 `.blend` 与 manifest；
- 对每个 VDB target 建立：

```text
Source_<VDB> -> <VDB>
```

的 Surface Deform 关系；

- 将 source mesh 权重转移到 VDB mesh；
- 生成两种 drive mode 输出：

```text
<asset>_prep_03_surface_drive_mode.blend
<asset>_prep_03_lbs_drive_mode.blend
```

### 8.1 Surface Deform bind proxy

当前 `--surface_bind_proxy_mode` 默认值为：

```text
auto
```

含义：

- 如果 Surface Deform bind source mesh 存在 edge shared by more than 2 polygons；
- 自动创建 hidden repaired bind proxy；
- 对 proxy 做 triangulate / edge split repair；
- Surface Deform 绑定到 repaired proxy；
- 不 fallback 到 LBS；
- Surface Deform bind 仍必须成功。

相关参数：

```bash
--surface_bind_proxy_mode off|auto|always
--surface_bind_proxy_repair_max_iters 8
```

Winter Samurai 的 `Source_top` 曾存在 edge>2 非流形边，因此需要该机制。

### 8.2 典型命令

```bash
cd /workspace/git_projects/OGD-LBS

conda run --no-capture-output -n lao-lbs \
  python scripts/data_prep/prep_03_surface_deform.py \
  --input_dir /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir \
  --drive_mode BOTH
```

---

## 9. Step 04a：cleaned-full armature 与减骨版本

脚本：

```text
prep_04a_bone_pruning.py
```

职责：

- 从 Prep 03 LBS drive mode `.blend` 读取主 armature；
- 复制出：

```text
Armature_Original
Armature_Cleaned_Full
```

- 自动删除 invalid / tiny / non-deform bones；
- 可根据额外配置生成多个 configured bone-reduced variants；
- 输出 cleaned-full `.blend` 与 variants。

输出：

```text
data_prep_04/
├── <asset>_prep_04a_cleaned_full.blend
└── bone_reduced_versions/
    ├── manifest.json
    ├── bone_reduced_conservative.blend
    ├── bone_reduced_moderate.blend
    └── bone_reduced_aggressive.blend
```

### 9.1 Extra bone reduction config

配置 schema：

```json
{
    "schema": "ogd_lbs_extra_bone_reduction_config_v1",
    "asset": "Winter_Samurai",
    "base": {
        "preferred_base_armature": "Armature_Cleaned_Full",
        "reference_original_armature": "Armature_Original"
    },
    "variants": {
        "bone_reduced_aggressive": {
            "armature_name": "Armature_Bone_Reduced_aggressive",
            "description": "Aggressive requested removal list.",
            "delete_bones": []
        }
    }
}
```

语义：

```text
delete_bones 是 requested removal list。
Prep 04a 中：
- 若 bone 仍在 Armature_Cleaned_Full 中，则实际删除；
- 若 bone 已不在 Armature_Cleaned_Full 但存在于 Armature_Original，则记录为 already_absent_in_cleaned_full；
- 若 bone 同时不在二者中，则视为命名错误并 fail fast。
```

### 9.2 当前配置文件

Griffin_A2：

```text
scripts/data_prep/configs/extra_bone_reduce_griffin_a2.json
```

Winter_Samurai：

```text
scripts/data_prep/configs/extra_bone_reduce_winter_samurai.json
```


### 9.3 典型命令

```bash
cd /workspace/git_projects/OGD-LBS

conda run --no-capture-output -n lao-lbs \
  python scripts/data_prep/prep_04a_bone_pruning.py \
  --input_dir /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir \
  --extra_bone_reduce_config /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/extra_bone_reduce_config.json \
  --overwrite_config_variants \
  --no_manual_template
```


---

## 10. Step 05：Fitbox 导出

脚本：

```text
prep_05_export_fitbox.py
```

职责：

- 读取 Prep 04 cleaned-full `.blend`；
- 读取 Prep 02 manifest；
- 导出：

```text
fitbox_original/
fitbox_vdb/
```

- 导出 cleaned-full 与 bone-reduced variants 的 armature-only Fitbox 数据；
- 支持按目标帧数均匀采样；
- 导出 mesh faces 时强制使用三角面。

### 10.1 输出结构

```text
data_prep_05/
├── export_summary.json
├── fitbox_original/
│   ├── skeleton_hierarchy_data.pkl
│   ├── animation_data.pkl
│   └── Source_<VDB>.pkl
└── fitbox_vdb/
    ├── skeleton_hierarchy_data.pkl
    ├── animation_data.pkl
    ├── <VDB>.pkl
    ├── <unmerged>.pkl
    └── bone_reduced_versions/
        ├── cleaned_full/
        │   ├── skeleton_hierarchy_data.pkl
        │   └── animation_data.pkl
        ├── bone_reduced_conservative/
        │   ├── skeleton_hierarchy_data.pkl
        │   └── animation_data.pkl
        ├── bone_reduced_moderate/
        │   ├── skeleton_hierarchy_data.pkl
        │   └── animation_data.pkl
        └── bone_reduced_aggressive/
            ├── skeleton_hierarchy_data.pkl
            └── animation_data.pkl
```

### 10.2 Frame sampling

支持：

```bash
--target_frame_count N
```

当 `target_frame_count > 0` 时，优先使用目标帧数均匀采样，优先级高于 `removal_rate`。

例如导出 60 帧：

```bash
cd /workspace/git_projects/OGD-LBS

conda run --no-capture-output -n lao-lbs \
  python scripts/data_prep/prep_05_export_fitbox.py \
  --input_dir /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir \
  --target_frame_count 60
```

### 10.3 Mesh faces 三角化

Prep 05 导出 mesh 时使用三角面 faces：

```text
faces shape = (F, 3)
```

同时对齐：

```text
face_uvs
face_materials
```

这样可以避免 mixed polygon / ragged faces 在 Fitbox 导入检查时失败。

---

## 11. Dataset-side runner

正式 dataset-side runner：

```text
run_dataset_data_prep.sh
```

支持模型：

```text
Griffin_A2
Winter_Samurai
All
```

不支持但会识别：

```text
Vroid_18
Vroid_19
Vroid_20
```

### 11.1 全量运行 Griffin_A2 + Winter_Samurai

```bash
cd /workspace/git_projects/OGD-LBS

bash scripts/data_prep/run_dataset_data_prep.sh All \
  --dataset_root /workspace/ogd_dataset \
  --conda_env lao-lbs \
  --bpy5_env ogd-bpy52 \
  --bpy4_env lao-lbs \
  --archive_existing \
  --target_frame_count 60 \
  --nla_gap 0
```

含义：

- `All`：依次跑 Griffin_A2 与 Winter_Samurai；
- `--archive_existing`：如果已有非空 `data_prep_work_dir`，自动归档；
- `--target_frame_count 60`：Prep 05 导出 60 帧；
- `--nla_gap 0`：动作拼接不插入空帧；
- 自动复制模型对应的 Prep 02 config；
- 如果存在模型对应的 extra bone reduction config，则自动复制并用于 Prep 04a。

归档目录：

```text
/workspace/ogd_backups/data_prep_work_dir_archives/
```

### 11.2 单模型运行

Griffin_A2：

```bash
cd /workspace/git_projects/OGD-LBS

bash scripts/data_prep/run_dataset_data_prep.sh Griffin_A2 \
  --archive_existing \
  --target_frame_count 60 \
  --nla_gap 0
```

Winter_Samurai：

```bash
cd /workspace/git_projects/OGD-LBS

bash scripts/data_prep/run_dataset_data_prep.sh Winter_Samurai \
  --archive_existing \
  --target_frame_count 60 \
  --nla_gap 0
```

### 11.3 成功输出

每个模型成功后应有：

```text
data_prep_work_dir/
├── final_dataset_data_prep_summary.txt
├── data_prep_01/<asset>_prep_01.blend
├── data_prep_02/<asset>_prep_02.blend
├── data_prep_03/<asset>_prep_03_surface_drive_mode.blend
├── data_prep_03/<asset>_prep_03_lbs_drive_mode.blend
├── data_prep_04/<asset>_prep_04a_cleaned_full.blend
└── data_prep_05/export_summary.json
```

---

## 12. Fitbox 视觉检查

统一导入检查工具：

```text
scripts/debug/import_fitbox_to_blend.py
```

### 12.1 Griffin_A2 ShapeKey 检查

```bash
cd /workspace/git_projects/OGD-LBS

mkdir -p /workspace/ogd_dataset/Griffin_A2/data_prep_work_dir/visual_checks

conda run --no-capture-output -n lao-lbs \
  python scripts/debug/import_fitbox_to_blend.py \
  --input_dir /workspace/ogd_dataset/Griffin_A2/data_prep_work_dir/data_prep_05/fitbox_vdb \
  --import_mode shape_key \
  --output_blend /workspace/ogd_dataset/Griffin_A2/data_prep_work_dir/visual_checks/import_fitbox_vdb_shape_key_60f.blend \
  --open
```

### 12.2 Griffin_A2 LBS 检查

```bash
cd /workspace/git_projects/OGD-LBS

mkdir -p /workspace/ogd_dataset/Griffin_A2/data_prep_work_dir/visual_checks

conda run --no-capture-output -n lao-lbs \
  python scripts/debug/import_fitbox_to_blend.py \
  --input_dir /workspace/ogd_dataset/Griffin_A2/data_prep_work_dir/data_prep_05/fitbox_vdb \
  --import_mode LBS \
  --output_blend /workspace/ogd_dataset/Griffin_A2/data_prep_work_dir/visual_checks/import_fitbox_vdb_lbs_60f.blend \
  --open
```

### 12.3 Winter_Samurai ShapeKey 检查

```bash
cd /workspace/git_projects/OGD-LBS

mkdir -p /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/visual_checks

conda run --no-capture-output -n lao-lbs \
  python scripts/debug/import_fitbox_to_blend.py \
  --input_dir /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/data_prep_05/fitbox_vdb \
  --import_mode shape_key \
  --output_blend /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/visual_checks/import_fitbox_vdb_shape_key_60f.blend \
  --open
```

### 12.4 Winter_Samurai LBS 检查

Winter Samurai 模型相对 Griffin 更小，建议使用较短骨骼显示长度：

```bash
cd /workspace/git_projects/OGD-LBS

mkdir -p /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/visual_checks

conda run --no-capture-output -n lao-lbs \
  python scripts/debug/import_fitbox_to_blend.py \
  --input_dir /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/data_prep_05/fitbox_vdb \
  --import_mode LBS \
  --bone_length 0.08 \
  --output_blend /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/visual_checks/import_fitbox_vdb_lbs_60f_bonelen0p08.blend \
  --open
```

也可以通过环境变量设置默认值：

```bash
export OGD_FITBOX_IMPORT_BONE_LENGTH=0.08
```

---

## 13. SSDR handoff

Data Prep 完成后，将：

```text
data_prep_work_dir/data_prep_05/fitbox_vdb/
```

同步到：

```text
ssdr_work_dir/data0_json/
```

推荐使用 `rsync -a --delete`。

### 13.1 Griffin_A2

```bash
cd /workspace/git_projects/OGD-LBS

mkdir -p /workspace/ogd_dataset/Griffin_A2/ssdr_work_dir/data0_json

conda run --no-capture-output -n lao-lbs \
  rsync -a --delete \
  /workspace/ogd_dataset/Griffin_A2/data_prep_work_dir/data_prep_05/fitbox_vdb/ \
  /workspace/ogd_dataset/Griffin_A2/ssdr_work_dir/data0_json/
```

### 13.2 Winter_Samurai

```bash
cd /workspace/git_projects/OGD-LBS

mkdir -p /workspace/ogd_dataset/Winter_Samurai/ssdr_work_dir/data0_json

conda run --no-capture-output -n lao-lbs \
  rsync -a --delete \
  /workspace/ogd_dataset/Winter_Samurai/data_prep_work_dir/data_prep_05/fitbox_vdb/ \
  /workspace/ogd_dataset/Winter_Samurai/ssdr_work_dir/data0_json/
```

注意：

```text
SSDR 是否完成不属于 Data Prep 成功范围。
Data Prep 只负责产出 fitbox_vdb 并完成 handoff。
```


