

# OGD-LBS

> Code of Paper: **“OGD-LBS: Animation-Guided Collaborative Lightweighting for Pre-Bound Multi-Component LBS Characters”**

本项目包含 OGD-LBS 论文的官方实现代码。针对商业级、多组件、预绑定（Pre-Bound）的复杂 LBS 角色资产，本项目提供了一套自动化数据预处理、Fitbox 导出、SSDR 前置处理与后续轻量化实验管线。

---

## 1. 项目结构

```text
OGD-LBS/
├── configs/
│   ├── prep_02_sdf_merge_config_example.json
│   ├── prep_02_merge_config_example.json
│   ├── prep_02_sdf_merge_config_griffin.json
│   └── ssdr_03_qem_config_example.json
│
├── scripts/
│   ├── data_prep/
│   │   ├── prep_01_concat_nla.py
│   │   ├── prep_02_sdf_pipeline.sh
│   │   ├── prep_02a_sdf_fuse_5x.py
│   │   ├── prep_02b_import_fused_4x.py
│   │   ├── prep_03_surface_deform.py
│   │   ├── prep_04a_bone_pruning.py
│   │   ├── prep_05_export_fitbox.py
│   │   ├── paths.py
│   │   ├── manifest.py
│   │   └── tests/
│   │
│   ├── ssdr/
│   │   ├── 00_convert_json_to_pickle.py
│   │   ├── 01_reduce_animation_frame.py
│   │   ├── 02_reduce_bone_weight.py
│   │   ├── 03_create_low_poly_shapes.py
│   │   ├── paths.py
│   │   └── tests/
│   │
│   └── debug/
│       └── import_fitbox_to_blend.py
│
├── src/
│   └── ogd_lbs/
│       ├── blender/
│       ├── fitbox/
│       │   ├── blender_io.py
│       │   ├── sampling.py
│       │   ├── schema.py
│       │   └── validation.py
│       ├── qem/
│       │   ├── __init__.py
│       │   └── qem.py
│       ├── ssdr/
│       └── utils/
│
├── pyproject.toml
├── LICENSE
└── README.md
```

说明：

- `scripts/data_prep/`：商业 `.blend` 到 Fitbox 数据的预处理流水线。
- `scripts/ssdr/`：SSDR 前置阶段，目前包含 Step 00 / 01 / 02 / 03。
- `scripts/debug/`：调试和检查工具，不属于某个 SSDR 阶段。
- `src/ogd_lbs/fitbox/blender_io.py`：Fitbox 导出与导入 Blender 的核心逻辑。
- `src/ogd_lbs/qem/qem.py`：QEM 网格减面核心实现。

---

## 2. 环境依赖

项目目前使用两个 Conda 环境。

### 2.1 主环境：`lao-lbs`

用于运行：

- Prep 01
- Prep 02b
- Prep 03
- Prep 04a
- Prep 05
- SSDR Step 00 / 01 / 02 / 03
- Fitbox 导入检查工具
- 常规 Python 测试

示例：

```bash
conda activate lao-lbs
pip install -e .
```

已验证环境包括：

```text
Python 3.11
bpy 4.x
numpy
pandas
```

如果需要运行旧的后续优化实验模块，可能还需要：

```text
scipy
```

### 2.2 SDF 环境：`ogd-bpy52`

用于运行：

- Prep 02a
- Blender 5.x SDF Grid / VDB-like fusion

示例：

```bash
conda create -n ogd-bpy52 python=3.13 -y
conda activate ogd-bpy52
pip install bpy==5.2.2 trimesh networkx numpy
```

已验证环境包括：

```text
Python 3.13
bpy 5.2.2 LTS
numpy
trimesh
networkx
```

---

## 3. 推荐实验目录布局

推荐将一次完整实验放在一个实验大目录中，并在下面拆分不同管线的工作目录。

以 Griffin 为例：

```text
/workspace/ogd_experiments/Griffin_A2_exp01/
├── data_prep_work_dir/
├── ssdr_work_dir/
└── logs/
```

其中：

```text
data_prep_work_dir/
```

用于商业 `.blend` 到 Fitbox 的预处理。

```text
ssdr_work_dir/
```

用于 SSDR 前置阶段：

```text
data0_json -> data0 -> data1 -> data2 -> data3
```

---

## 4. Data Prep 工作目录约定

`data_prep_work_dir` 内部使用 staged directory 约定：

```text
data_prep_work_dir/
├── prep_02_sdf_merge_config.json
│
├── data_prep_00/
│   └── Griffin_A2.blend
│
├── data_prep_01/
│   └── Griffin_A2_prep_01.blend
│
├── data_prep_02/
│   ├── Griffin_A2_prep_02.blend
│   └── sdf_exports/
│       ├── manifest.json
│       ├── main_mesh_VDB.obj
│       └── main_mesh_VDB.mtl
│
├── data_prep_03/
│   ├── Griffin_A2_prep_03_surface_drive_mode.blend
│   └── Griffin_A2_prep_03_lbs_drive_mode.blend
│
├── data_prep_04/
│   ├── Griffin_A2_prep_04a_cleaned_full.blend
│   └── bone_reduced_versions/
│       ├── manifest.json
│       └── bone_reduced_001.blend
│
└── data_prep_05/
    ├── export_summary.json
    ├── fitbox_original/
    │   ├── skeleton_hierarchy_data.pkl
    │   ├── animation_data.pkl
    │   └── Source_main_mesh_VDB.pkl
    └── fitbox_vdb/
        ├── skeleton_hierarchy_data.pkl
        ├── animation_data.pkl
        ├── main_mesh_VDB.pkl
        └── bone_reduced_versions/
            ├── cleaned_full/
            │   ├── skeleton_hierarchy_data.pkl
            │   └── animation_data.pkl
            └── bone_reduced_001/
                ├── skeleton_hierarchy_data.pkl
                └── animation_data.pkl
```

阶段语义：

|阶段|输入目录|输出目录|说明|
|---|---|---|---|
|Prep 01|`data_prep_00`|`data_prep_01`|raw `.blend` → NLA concatenated `.blend`|
|Prep 02a|`data_prep_01`|`data_prep_02/sdf_exports`|Blender 5.x SDF fusion|
|Prep 02b|`data_prep_01` + `data_prep_02/sdf_exports`|`data_prep_02`|import fused OBJ into Blender 4.x|
|Prep 03|`data_prep_02`|`data_prep_03`|Surface Deform + LBS drive mode|
|Prep 04a|`data_prep_03`|`data_prep_04`|cleaned-full skeleton + manual template|
|Prep 05|`data_prep_04`|`data_prep_05`|Fitbox export|

`data_prep_05/fitbox_vdb/` 是 SSDR 的 `data0_json` 数据源。

---

## 5. Data Prep：人工运行流程

下面流程是当前推荐的人工测试流程，已经按 Griffin 资产验证到 Data Prep → SSDR handoff。

### 5.1 初始化实验目录

```bash
cd /workspace/git_projects/OGD-LBS

EXP_ROOT=/workspace/ogd_experiments
EXP="$EXP_ROOT/Griffin_A2_exp01_$(date +%Y%m%d_%H%M%S)"
DP="$EXP/data_prep_work_dir"
SSDR="$EXP/ssdr_work_dir"
LOG="$EXP/logs"

INPUT_BLEND=/workspace/data/dataset/Griffin_A2/Blender/Griffin_A2.blend
CONFIG_SRC=/workspace/git_projects/OGD-LBS/configs/prep_02_sdf_merge_config_griffin.json
ASSET=Griffin_A2

mkdir -p "$DP/data_prep_00" "$SSDR" "$LOG"

cp "$INPUT_BLEND" "$DP/data_prep_00/${ASSET}.blend"
cp "$CONFIG_SRC" "$DP/prep_02_sdf_merge_config.json"

echo "EXP  = $EXP"
echo "DP   = $DP"
echo "SSDR = $SSDR"
echo "LOG  = $LOG"
```

---

### 5.2 Prep 01：NLA 拼接

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/data_prep/prep_01_concat_nla.py \
    --input_dir "$DP" \
    2>&1 | tee "$LOG/01_prep_01_concat_nla.log"
```

也可以直接指定当前阶段输入目录：

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/data_prep/prep_01_concat_nla.py \
    --input_dir "$DP/data_prep_00"
```

输出：

```text
$DP/data_prep_01/Griffin_A2_prep_01.blend
```

Prep 01 会在 `data_prep_00` 中查找唯一 raw `.blend`。如果有多个 raw `.blend`，会 fail-fast，避免误选资产。

---

### 5.3 Prep 02：SDF 融合与导回

Prep 02 是二阶段步骤，也可以使用 wrapper 一次运行。

#### 5.3.1 推荐：使用 wrapper

```bash
bash scripts/data_prep/prep_02_sdf_pipeline.sh \
    --input_dir "$DP" \
    --offset 15 \
    --no_auto_offset \
    2>&1 | tee "$LOG/02_prep_02_sdf_pipeline.log"
```

输出：

```text
$DP/data_prep_02/sdf_exports/manifest.json
$DP/data_prep_02/sdf_exports/main_mesh_VDB.obj
$DP/data_prep_02/Griffin_A2_prep_02.blend
```

#### 5.3.2 手动分两步运行

Prep 02a：

```bash
conda run --no-capture-output -n ogd-bpy52 \
    python scripts/data_prep/prep_02a_sdf_fuse_5x.py \
    --input_dir "$DP" \
    --offset 15 \
    --no_auto_offset \
    2>&1 | tee "$LOG/02a_prep_02a_sdf_fuse_5x.log"
```

Prep 02b：

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/data_prep/prep_02b_import_fused_4x.py \
    --input_dir "$DP" \
    2>&1 | tee "$LOG/02b_prep_02b_import_fused_4x.log"
```

---

### 5.4 Prep 03：Surface Deform + LBS drive mode

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/data_prep/prep_03_surface_deform.py \
    --input_dir "$DP" \
    --drive_mode BOTH \
    2>&1 | tee "$LOG/03_prep_03_surface_deform.log"
```

输出：

```text
$DP/data_prep_03/Griffin_A2_prep_03_surface_drive_mode.blend
$DP/data_prep_03/Griffin_A2_prep_03_lbs_drive_mode.blend
```

说明：

- `surface_drive_mode`：VDB mesh 由 Surface Deform 驱动。
- `lbs_drive_mode`：VDB mesh 由 LBS Armature modifier 驱动。

Prep 04a 默认使用 LBS drive mode 版本继续处理。

---

### 5.5 Prep 04a：清理 full skeleton + 生成手工减骨模板

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/data_prep/prep_04a_bone_pruning.py \
    --input_dir "$DP" \
    2>&1 | tee "$LOG/04_prep_04a_bone_pruning.log"
```

输出：

```text
$DP/data_prep_04/Griffin_A2_prep_04a_cleaned_full.blend
$DP/data_prep_04/bone_reduced_versions/bone_reduced_001.blend
$DP/data_prep_04/bone_reduced_versions/manifest.json
```

`bone_reduced_001.blend` 是手动减骨模板。

如果需要人工编辑减骨版本，可打开：

```text
$DP/data_prep_04/bone_reduced_versions/bone_reduced_001.blend
```

编辑并保存后，再运行 Prep 05。

如果重跑 Prep 04a 并需要覆盖模板，可加：

```bash
--overwrite_manual_template
```

---

### 5.6 Prep 05：导出 Fitbox

Debug 60 帧示例：

```bash
DEBUG_FRAME_MAX=60

conda run --no-capture-output -n lao-lbs \
    python scripts/data_prep/prep_05_export_fitbox.py \
    --input_dir "$DP" \
    --debug_frame_max "$DEBUG_FRAME_MAX" \
    --include_unmodified_templates \
    2>&1 | tee "$LOG/05_prep_05_export_fitbox.log"
```

输出：

```text
$DP/data_prep_05/export_summary.json
$DP/data_prep_05/fitbox_original/
$DP/data_prep_05/fitbox_vdb/
```

其中：

```text
$DP/data_prep_05/fitbox_vdb/
```

是 SSDR 的输入源。

如果保留：

```bash
--include_unmodified_templates
```

则即使 `bone_reduced_001.blend` 未人工修改，也会导出：

```text
$DP/data_prep_05/fitbox_vdb/bone_reduced_versions/bone_reduced_001/
```

这对于测试 SSDR Step 02 的显式 variant 很方便：

```bash
--variant bone_reduced_001
```

生产运行时，如果不希望导出未修改模板，可以去掉：

```bash
--include_unmodified_templates
```

---

## 6. 从 Data Prep 同步到 SSDR

SSDR 的输入目录是：

```text
ssdr_work_dir/data0_json/
```

需要把：

```text
data_prep_work_dir/data_prep_05/fitbox_vdb/
```

复制到：

```text
ssdr_work_dir/data0_json/
```

如果系统有 `rsync`，推荐：

```bash
mkdir -p "$SSDR/data0_json"

rsync -a --delete \
    "$DP/data_prep_05/fitbox_vdb/" \
    "$SSDR/data0_json/"
```

注意：

- 源路径末尾的 `/` 很重要。
- `--delete` 只会删除目标目录中多余文件，不会删除源目录。

如果系统没有 `rsync`，使用：

```bash
SRC="$DP/data_prep_05/fitbox_vdb"
DST="$SSDR/data0_json"

test -d "$SRC" || { echo "ERROR: source fitbox_vdb not found: $SRC"; exit 1; }

rm -rf "$DST"
mkdir -p "$DST"

cp -a "$SRC/." "$DST/"
```

检查：

```bash
find "$SSDR/data0_json" -maxdepth 4 -type f | sort

ls -lh \
    "$SSDR/data0_json/skeleton_hierarchy_data.pkl" \
    "$SSDR/data0_json/animation_data.pkl" \
    "$SSDR/data0_json/main_mesh_VDB.pkl" \
    "$SSDR/data0_json/bone_reduced_versions/cleaned_full/skeleton_hierarchy_data.pkl" \
    "$SSDR/data0_json/bone_reduced_versions/cleaned_full/animation_data.pkl"
```

如果导出了 `bone_reduced_001`：

```bash
ls -lh \
    "$SSDR/data0_json/bone_reduced_versions/bone_reduced_001/skeleton_hierarchy_data.pkl" \
    "$SSDR/data0_json/bone_reduced_versions/bone_reduced_001/animation_data.pkl"
```

---

## 7. SSDR 前置管线

SSDR 工作目录使用 staged directory 约定：

```text
ssdr_work_dir/
├── data0_json/
├── data0/
├── data1/
├── data2/
└── data3/
```

阶段语义：

|阶段|输入目录|输出目录|说明|
|---|---|---|---|
|Step 00|`data0_json`|`data0`|JSON/PKL 标准化到 PKL|
|Step 01|`data0`|`data1`|动态帧采样|
|Step 02|`data1`|`data2`|骨骼权重映射到 reduced skeleton|
|Step 03|`data2`|`data3`|QEM low-poly shape generation|

---

### 7.1 Step 00：`data0_json -> data0`

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/ssdr/00_convert_json_to_pickle.py \
    --input_dir "$SSDR" \
    --overwrite_output \
    --max_workers 1 \
    2>&1 | tee "$LOG/06_ssdr_00_convert_json_to_pickle.log"
```

默认：

```text
input : $SSDR/data0_json
output: $SSDR/data0
```

Step 00 会：

- `.json -> .pkl`
- `.pkl -> copy`
- 保持相对目录结构

---

### 7.2 Step 01：`data0 -> data1` 动态帧采样

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/ssdr/01_reduce_animation_frame.py \
    --input_dir "$SSDR" \
    --overwrite_output \
    --max_workers 1 \
    2>&1 | tee "$LOG/07_ssdr_01_reduce_animation_frame.log"
```

默认：

```text
input : $SSDR/data0
output: $SSDR/data1
```

动态帧采样规则：

```text
若 frame_count <= 210，则保留全部；
若 frame_count > 210，则均匀采样到 200 帧。
```

---

### 7.3 Step 02：`data1 -> data2` 骨骼权重映射

如果 Prep 05 使用了：

```bash
--include_unmodified_templates
```

则可以显式选择：

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/ssdr/02_reduce_bone_weight.py \
    --input_dir "$SSDR" \
    --variant bone_reduced_001 \
    --overwrite_output \
    --max_workers 1 \
    2>&1 | tee "$LOG/08_ssdr_02_reduce_bone_weight.log"
```

如果没有导出 `bone_reduced_001`，可以不传 `--variant`，让 Step 02 自动选择可用 reduced variant：

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/ssdr/02_reduce_bone_weight.py \
    --input_dir "$SSDR" \
    --overwrite_output \
    --max_workers 1 \
    2>&1 | tee "$LOG/08_ssdr_02_reduce_bone_weight.log"
```

默认：

```text
input : $SSDR/data1
output: $SSDR/data2
```

输出：

```text
$SSDR/data2/skeleton_hierarchy_data.pkl
$SSDR/data2/animation_data.pkl
$SSDR/data2/main_mesh_VDB.pkl
$SSDR/data2/step02_reduce_bone_weight_summary.json
```

---

### 7.4 Step 03：`data2 -> data3` QEM 减面

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/ssdr/03_create_low_poly_shapes.py \
    --input_dir "$SSDR" \
    --simp_rate_base 0.1 \
    --overwrite_output \
    --max_workers 8 \
    2>&1 | tee "$LOG/09_ssdr_03_create_low_poly_shapes.log"
```

默认：

```text
input : $SSDR/data2
output: $SSDR/data3
```

输出：

```text
$SSDR/data3/skeleton_hierarchy_data.pkl
$SSDR/data3/animation_data.pkl
$SSDR/data3/main_mesh_VDB.pkl
$SSDR/data3/step03_qem_summary.json
```

#### Step 03 配置文件

Step 03 支持可选配置文件：

```bash
--config /path/to/ssdr_03_qem_config.json
```

如果未传 `--config`，脚本会尝试读取：

```text
$SSDR/ssdr_03_qem_config.json
```

如果默认配置文件不存在，则使用内置默认值。

配置示例：

```text
configs/ssdr_03_qem_config_example.json
```

配置覆盖优先级：

```text
显式 CLI 参数
  > config mesh_overrides[mesh_name]
  > config global
  > built-in defaults
```

每个 mesh 的目标面数计算规则：

```text
如果 mesh_overrides 中直接指定 target_faces：
    target_faces 优先

否则：
    target_faces = max(
        triangulated_face_count * simp_rate_base * simp_rate_multiplier,
        min_target_faces
    )
```

#### Polygon faces / quad faces

Step 03 的 QEM 核心要求三角面。若输入 mesh 是 quad 或其他固定宽度 polygon faces，Step 03 会在 QEM 前自动进行 triangle fan triangulation。

例如：

```text
113840 quads -> 227680 triangles -> QEM target 22768 faces
```

如果 VDB / QEM 数据没有有效 UV，`face_uvs` 可能会被跳过或丢弃。这对当前 SSDR/QEM/优化流程通常不是问题。

---

## 8. Fitbox 导入 Blender 检查工具

Fitbox / SSDR stage 数据可以导入回 Blender 进行检查。

统一脚本位置：

```text
scripts/debug/import_fitbox_to_blend.py
```

核心导入逻辑位于：

```text
src/ogd_lbs/fitbox/blender_io.py
```

该工具支持两种导入模式：

```bash
--import_mode shape_key
--import_mode LBS
```

### 8.1 Shape key 模式

Shape key 模式直接使用 `baked_animation_vertices` 创建 shape keys。该逻辑对齐已测试有效的 Blender 插件版 `ImportShapeKeyAnimation`：

- 默认使用连续帧 `frame_idx + 1`；
- shape key 名称为 `frame_1`, `frame_2`, ...；
- 每个 shape key 按插件逻辑插入 `0 -> 1 -> 0` 的硬切换关键帧；
- 默认不使用 `scene_frames`。

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/debug/import_fitbox_to_blend.py \
    --input_dir "$SSDR/data3" \
    --mesh main_mesh_VDB \
    --import_mode shape_key \
    --output_blend "$SSDR/data3/import_main_mesh_VDB_shape_key.blend" \
    --open \
    2>&1 | tee "$LOG/10_import_data3_shape_key.log"
```

如果需要按原始采样 scene frames 导入关键帧，可加：

```bash
--use_scene_frames
```

### 8.2 LBS 模式

LBS 模式会创建：

- Armature；
- mesh；
- vertex groups；
- Armature modifier；
- pose bone animation keyframes。

该逻辑对齐已测试有效的 Blender 插件版 `SkdMeshArmatureImporter`：

- armature 默认名称为 `Armature`；
- edit bone `head = rest_pose translation`；
- edit bone `tail = location + rotation @ Vector((0, 0.5, 0))`；
- animation keyframes 默认使用连续帧 `frame_idx + 1`；
- vertex group 写入使用 `REPLACE`；
- Armature modifier 名称为 `ArmatureMod`；
- mesh object parent 设置为 armature object。

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/debug/import_fitbox_to_blend.py \
    --input_dir "$SSDR/data3" \
    --mesh main_mesh_VDB \
    --import_mode LBS \
    --output_blend "$SSDR/data3/import_main_mesh_VDB_lbs.blend" \
    --open \
    2>&1 | tee "$LOG/11_import_data3_lbs.log"
```

如果你的 Blender 可执行文件路径不是默认：

```text
/workspace/blender-4.5.12-linux-x64/blender
```

可以设置：

```bash
export BLENDER_BIN=/path/to/blender
```

或者显式传入：

```bash
--blender_bin /path/to/blender
```

---

## 9. 当前 Griffin 测试参考命令

下面是一组完整命令，可用于从头跑 Griffin 的：

```text
data_prep -> ssdr 00/01/02/03
```

```bash
cd /workspace/git_projects/OGD-LBS

EXP_ROOT=/workspace/ogd_experiments
EXP="$EXP_ROOT/Griffin_A2_exp01_$(date +%Y%m%d_%H%M%S)"
DP="$EXP/data_prep_work_dir"
SSDR="$EXP/ssdr_work_dir"
LOG="$EXP/logs"

INPUT_BLEND=/workspace/data/dataset/Griffin_A2/Blender/Griffin_A2.blend
CONFIG_SRC=/workspace/git_projects/OGD-LBS/configs/prep_02_sdf_merge_config_griffin.json
ASSET=Griffin_A2
DEBUG_FRAME_MAX=60

mkdir -p "$DP/data_prep_00" "$SSDR" "$LOG"

cp "$INPUT_BLEND" "$DP/data_prep_00/${ASSET}.blend"
cp "$CONFIG_SRC" "$DP/prep_02_sdf_merge_config.json"

conda run --no-capture-output -n lao-lbs \
    python scripts/data_prep/prep_01_concat_nla.py \
    --input_dir "$DP" \
    2>&1 | tee "$LOG/01_prep_01_concat_nla.log"

bash scripts/data_prep/prep_02_sdf_pipeline.sh \
    --input_dir "$DP" \
    --offset 15 \
    --no_auto_offset \
    2>&1 | tee "$LOG/02_prep_02_sdf_pipeline.log"

conda run --no-capture-output -n lao-lbs \
    python scripts/data_prep/prep_03_surface_deform.py \
    --input_dir "$DP" \
    --drive_mode BOTH \
    2>&1 | tee "$LOG/03_prep_03_surface_deform.log"

conda run --no-capture-output -n lao-lbs \
    python scripts/data_prep/prep_04a_bone_pruning.py \
    --input_dir "$DP" \
    2>&1 | tee "$LOG/04_prep_04a_bone_pruning.log"

conda run --no-capture-output -n lao-lbs \
    python scripts/data_prep/prep_05_export_fitbox.py \
    --input_dir "$DP" \
    --debug_frame_max "$DEBUG_FRAME_MAX" \
    --include_unmodified_templates \
    2>&1 | tee "$LOG/05_prep_05_export_fitbox.log"

SRC="$DP/data_prep_05/fitbox_vdb"
DST="$SSDR/data0_json"

rm -rf "$DST"
mkdir -p "$DST"
cp -a "$SRC/." "$DST/"

conda run --no-capture-output -n lao-lbs \
    python scripts/ssdr/00_convert_json_to_pickle.py \
    --input_dir "$SSDR" \
    --overwrite_output \
    --max_workers 1 \
    2>&1 | tee "$LOG/06_ssdr_00_convert_json_to_pickle.log"

conda run --no-capture-output -n lao-lbs \
    python scripts/ssdr/01_reduce_animation_frame.py \
    --input_dir "$SSDR" \
    --overwrite_output \
    --max_workers 1 \
    2>&1 | tee "$LOG/07_ssdr_01_reduce_animation_frame.log"

conda run --no-capture-output -n lao-lbs \
    python scripts/ssdr/02_reduce_bone_weight.py \
    --input_dir "$SSDR" \
    --variant bone_reduced_001 \
    --overwrite_output \
    --max_workers 1 \
    2>&1 | tee "$LOG/08_ssdr_02_reduce_bone_weight.log"

conda run --no-capture-output -n lao-lbs \
    python scripts/ssdr/03_create_low_poly_shapes.py \
    --input_dir "$SSDR" \
    --simp_rate_base 0.1 \
    --overwrite_output \
    --max_workers 8 \
    2>&1 | tee "$LOG/09_ssdr_03_create_low_poly_shapes.log"
```

可选：导入 `data3` 结果到 Blender 检查：

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/debug/import_fitbox_to_blend.py \
    --input_dir "$SSDR/data3" \
    --mesh main_mesh_VDB \
    --import_mode shape_key \
    --output_blend "$SSDR/data3/import_main_mesh_VDB_shape_key.blend" \
    --open \
    2>&1 | tee "$LOG/10_import_data3_shape_key.log"
```

```bash
conda run --no-capture-output -n lao-lbs \
    python scripts/debug/import_fitbox_to_blend.py \
    --input_dir "$SSDR/data3" \
    --mesh main_mesh_VDB \
    --import_mode LBS \
    --output_blend "$SSDR/data3/import_main_mesh_VDB_lbs.blend" \
    --open \
    2>&1 | tee "$LOG/11_import_data3_lbs.log"
```

---

## 10. 单测与编译检查

### 10.1 data_prep 路径测试

```bash
python -m unittest discover -s scripts/data_prep/tests -p 'test_*.py' -v
```

### 10.2 SSDR 测试

```bash
python -m unittest discover -s scripts/ssdr/tests -p 'test_*.py' -v
```

### 10.3 常用编译检查

```bash
python -m py_compile \
    scripts/ssdr/03_create_low_poly_shapes.py \
    scripts/debug/import_fitbox_to_blend.py \
    src/ogd_lbs/fitbox/blender_io.py \
    src/ogd_lbs/fitbox/validation.py \
    src/ogd_lbs/qem/qem.py
```

---

## 11. 当前状态说明

目前已迁移并验证的 SSDR 前置阶段包括：

```text
Step 00: data0_json -> data0
Step 01: data0 -> data1
Step 02: data1 -> data2
Step 03: data2 -> data3
```

其中 Step 03 已在 Griffin 测试数据上验证：

```text
quad/polygon faces -> triangulated faces -> QEM reduced mesh
```

后续联合优化相关阶段仍在迁移规划中。