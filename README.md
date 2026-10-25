



# OGD-LBS

> Code of Paper: **“OGD-LBS: Animation-Guided Collaborative Lightweighting for Pre-Bound Multi-Component LBS Characters”**

OGD-LBS 是面向商业级、多组件、预绑定（Pre-Bound）LBS 角色资产的轻量化实验管线。项目包含：

- Blender 侧数据预处理；
- Fitbox 数据导出 / 导入；
- SSDR 前置处理；
- QEM 低模生成；
- 权重与骨骼变换优化相关实验代码；
- 针对部分商业资产的清洗入口脚本。

本仓库只提供代码与配置示例，**不包含商业资产文件**。

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
│   ├── custom_model_clean/
│   │   └── winter_samurai/
│   │       ├── 01_import_model_and_anims.py
│   │       ├── 02_rebase_anims_to_skinning_armature.py
│   │       ├── 03_rebase_to_rest_pose_base.py
│   │       ├── 04_fix_weapon_mount_aim.py
│   │       ├── run_dataset_cleaning.sh
│   │       ├── configs/
│   │       └── tests/
│   │
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
│   │   ├── 04_optimize_weight_and_bone_trans.py
│   │   ├── paths.py
│   │   └── tests/
│   │
│   └── debug/
│       ├── import_fitbox_to_blend.py
│       └── ssdr_res_render.py
│
├── src/
│   └── ogd_lbs/
│       ├── blender/
│       ├── fitbox/
│       ├── qem/
│       ├── ssdr/
│       └── utils/
│
├── pyproject.toml
├── LICENSE
└── README.md
```

主要目录说明：

|路径|说明|
|---|---|
|`scripts/custom_model_clean/`|资产专用清洗入口。目前包含 Winter Samurai 清洗流程。|
|`scripts/data_prep/`|商业 `.blend` 到 Fitbox 数据的预处理管线。|
|`scripts/ssdr/`|SSDR 前置与优化阶段。|
|`scripts/debug/`|Fitbox / SSDR 结果检查、渲染与调试工具。|
|`src/ogd_lbs/blender/`|可复用 Blender 工具。|
|`src/ogd_lbs/fitbox/`|Fitbox 导入、导出、采样与校验。|
|`src/ogd_lbs/qem/`|QEM 网格减面核心实现。|
|`src/ogd_lbs/ssdr/`|SSDR 相关算法实现。|
|`src/ogd_lbs/utils/`|通用 IO、work dir、progress 等工具。|

---

## 2. 环境依赖

项目当前主要使用两个 Conda 环境。

### 2.1 主环境：`lao-lbs`

用于运行：

- Winter Samurai 清洗；
- Prep 01 / 02b / 03 / 04a / 05；
- SSDR Step 00 / 01 / 02 / 03 / 04；
- Fitbox 导入检查工具；
- 常规 Python 测试。

示例：

```bash
conda activate lao-lbs
pip install -e .
```

已验证环境包括：

```text
Python 3.11
bpy 4.x
Blender 4.5.x LTS binary
numpy
scipy
pandas
PyYAML
```

如果直接使用 Blender binary 运行脚本，当前项目中常用路径为：

```text
/workspace/blender-4.5.12-linux-x64/blender
```

### 2.2 SDF 环境：`ogd-bpy52`

用于运行：

- Prep 02a；
- Blender 5.x SDF Grid / VDB-like fusion。

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

## 3. 数据集与资产约定

推荐正式数据集根目录为：

```text
/workspace/ogd_dataset/
```

当前项目涉及的资产类型包括：

|资产|状态|来源|
|---|---|---|
|`Griffin_A2`|标准 `.blend` 输入，无需 custom clean|[https://superhivemarket.com/products/griffin_a2](https://superhivemarket.com/products/griffin_a2)|
|`Winter_Samurai`|需要先运行 custom clean|[https://superhivemarket.com/products/winter-samurai](https://superhivemarket.com/products/winter-samurai)|
|`Vroid_*`|预制角色资产，旧数据已预处理|[https://vroid.com/en/studio](https://vroid.com/en/studio)|

本仓库不提供商业资产本体。请将资产放置到本地数据集目录后再运行管线。

### 3.1 推荐 `ogd_dataset` 布局

```text
/workspace/ogd_dataset/
├── Griffin_A2/
│   └── Griffin_A2.blend
│
├── Vroid_18/
│   ├── data0/
│   └── target/
│
├── Vroid_19/
│   ├── data0/
│   └── target/
│
├── Vroid_20/
│   ├── data0/
│   └── target/
│
└── Winter_Samurai/
    ├── model/
    │   ├── Base_Mesh_Rig.fbx
    │   └── rest_pose_base.blend
    ├── anim/
    │   ├── Anim_Winter_Samurai_Attack1.fbx
    │   ├── ...
    │   └── Anim_Winter_Samurai_Walk3.fbx
    └── Winter_Samurai_Normalized.blend
```

说明：

- `Griffin_A2/Griffin_A2.blend` 是标准 Data Prep 输入。
- `Vroid_*` 当前按 `data0/` 和 `target/` 组织。项目涉及的 VRoid 预制资产 Fitbox `data0` 文件后续考虑发布，但目前暂未上传；派生的 `data1` / `data2` / `data3` / `data4` 请用本仓库脚本自行计算生成。
- `Winter_Samurai/Winter_Samurai_Normalized.blend` 由 custom clean 管线生成，是后续 Data Prep 输入。

---

## 4. Winter Samurai 清洗

Winter Samurai 原始资产需要先经过专用清洗流程，生成规范化 `.blend`。

清洗入口位于：

```text
scripts/custom_model_clean/winter_samurai/
```

正式 runner：

```text
scripts/custom_model_clean/winter_samurai/run_dataset_cleaning.sh
```

### 4.1 输入目录要求

默认输入目录：

```text
/workspace/ogd_dataset/Winter_Samurai/
├── model/
│   ├── Base_Mesh_Rig.fbx
│   └── rest_pose_base.blend
└── anim/
    ├── Anim_Winter_Samurai_Attack1.fbx
    ├── Anim_Winter_Samurai_Attack2.fbx
    ├── Anim_Winter_Samurai_Attack3.fbx
    ├── Anim_Winter_Samurai_Attack4.fbx
    ├── Anim_Winter_Samurai_Death.fbx
    ├── Anim_Winter_Samurai_Death2.fbx
    ├── Anim_Winter_Samurai_Idle1.fbx
    ├── Anim_Winter_Samurai_Idle2.fbx
    ├── Anim_Winter_Samurai_Idle3.fbx
    ├── Anim_Winter_Samurai_Idle4.fbx
    ├── Anim_Winter_Samurai_Idle5.fbx
    ├── Anim_Winter_Samurai_Run1.fbx
    ├── Anim_Winter_Samurai_Run2.fbx
    ├── Anim_Winter_Samurai_Run3.fbx
    ├── Anim_Winter_Samurai_Stance.fbx
    ├── Anim_Winter_Samurai_Walk.fbx
    ├── Anim_Winter_Samurai_Walk2.fbx
    └── Anim_Winter_Samurai_Walk3.fbx
```

`rest_pose_base.blend` 使用 **Better FBX 6.3.5** 导入模型后保存得到，用于统一 Winter Samurai 清洗后的 rest pose / 骨架结构。

### 4.2 正式清洗命令

```bash
cd /workspace/git_projects/OGD-LBS

export OGD_PROGRESS=auto

bash scripts/custom_model_clean/winter_samurai/run_dataset_cleaning.sh \
  --dataset_dir /workspace/ogd_dataset/Winter_Samurai \
  --publish copy \
  --overwrite_publication
```

默认输出：

```text
/workspace/ogd_dataset/Winter_Samurai/model_clean_work_dir/
├── 01/
├── 02/
├── 03/
├── 04_weapon/
├── config_used.json
├── code_config_sha256.txt
├── git_state.txt
├── final_dataset_cleaning_summary.json
├── final_invariants.json
└── publication_report.json
```

发布结果：

```text
/workspace/ogd_dataset/Winter_Samurai/Winter_Samurai_Normalized.blend
```

`model_clean_work_dir` 是清洗工作目录。正式 runner 不复制源码脚本到工作目录，也不会生成 `model_clean_work_dir/code/`。

### 4.3 清洗结果检查

结构结果应满足：

```text
armatures = 1
bones     = 110
meshes    = 40
actions   = 18
empties   = 0
root bones = Character1_Hips, Weapon
Weapon parent = None
```

打开 Blender 视觉检查：

```bash
/workspace/blender-4.5.12-linux-x64/blender \
  /workspace/ogd_dataset/Winter_Samurai/Winter_Samurai_Normalized.blend
```

建议检查动作：

```text
WS_Stance
WS_Attack1–WS_Attack4
WS_Run2 / WS_Run3
WS_Walk3
WS_Idle*
WS_Walk / WS_Walk2
WS_Death / WS_Death2
```

---

## 5. Data Prep 工作目录

推荐将一次完整 Data Prep / SSDR 实验放在实验目录中：

```text
/workspace/ogd_experiments/<asset>_exp01/
├── data_prep_work_dir/
├── ssdr_work_dir/
└── logs/
```

Data Prep staged layout：

```text
data_prep_work_dir/
├── prep_02_sdf_merge_config.json
├── data_prep_00/
├── data_prep_01/
├── data_prep_02/
├── data_prep_03/
├── data_prep_04/
└── data_prep_05/
```

阶段语义：

|阶段|输入|输出|说明|
|---|---|---|---|
|Prep 01|`data_prep_00`|`data_prep_01`|raw `.blend` → NLA concatenated `.blend`|
|Prep 02a|`data_prep_01`|`data_prep_02/sdf_exports`|Blender 5.x SDF fusion|
|Prep 02b|`data_prep_01` + SDF exports|`data_prep_02`|导入 fused OBJ|
|Prep 03|`data_prep_02`|`data_prep_03`|Surface Deform + LBS drive mode|
|Prep 04a|`data_prep_03`|`data_prep_04`|cleaned-full skeleton + manual template|
|Prep 05|`data_prep_04`|`data_prep_05`|Fitbox export|

`data_prep_05/fitbox_vdb/` 是 SSDR 的 `data0_json` 数据源。

---

## 6. Data Prep 最小运行流程

下面以一个已经准备好的 `.blend` 为例。对 Winter Samurai，应使用：

```text
/workspace/ogd_dataset/Winter_Samurai/Winter_Samurai_Normalized.blend
```

对 Griffin，应使用：

```text
/workspace/ogd_dataset/Griffin_A2/Griffin_A2.blend
```

### 6.1 初始化实验目录

```bash
cd /workspace/git_projects/OGD-LBS

ASSET=Winter_Samurai
INPUT_BLEND=/workspace/ogd_dataset/Winter_Samurai/Winter_Samurai_Normalized.blend
CONFIG_SRC=/workspace/git_projects/OGD-LBS/configs/prep_02_sdf_merge_config_example.json

EXP_ROOT=/workspace/ogd_experiments
EXP="$EXP_ROOT/${ASSET}_exp01_$(date +%Y%m%d_%H%M%S)"
DP="$EXP/data_prep_work_dir"
SSDR="$EXP/ssdr_work_dir"
LOG="$EXP/logs"

mkdir -p "$DP/data_prep_00" "$SSDR" "$LOG"

cp "$INPUT_BLEND" "$DP/data_prep_00/${ASSET}.blend"
cp "$CONFIG_SRC" "$DP/prep_02_sdf_merge_config.json"

echo "EXP  = $EXP"
echo "DP   = $DP"
echo "SSDR = $SSDR"
echo "LOG  = $LOG"
```

### 6.2 Prep 01：NLA 拼接

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/data_prep/prep_01_concat_nla.py \
  --input_dir "$DP" \
  2>&1 | tee "$LOG/01_prep_01_concat_nla.log"
```

### 6.3 Prep 02：SDF fusion + 导回

推荐使用 wrapper：

```bash
bash scripts/data_prep/prep_02_sdf_pipeline.sh \
  --input_dir "$DP" \
  --offset 15 \
  --no_auto_offset \
  2>&1 | tee "$LOG/02_prep_02_sdf_pipeline.log"
```

`--offset` 是当前核心参数之一，需要按资产规模调整。

### 6.4 Prep 03：Surface Deform + LBS drive mode

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/data_prep/prep_03_surface_deform.py \
  --input_dir "$DP" \
  --drive_mode BOTH \
  2>&1 | tee "$LOG/03_prep_03_surface_deform.log"
```

### 6.5 Prep 04a：cleaned-full skeleton

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/data_prep/prep_04a_bone_pruning.py \
  --input_dir "$DP" \
  2>&1 | tee "$LOG/04_prep_04a_bone_pruning.log"
```

### 6.6 Prep 05：导出 Fitbox

Debug 运行可限制帧数：

```bash
DEBUG_FRAME_MAX=60

conda run --no-capture-output -n lao-lbs \
  python scripts/data_prep/prep_05_export_fitbox.py \
  --input_dir "$DP" \
  --debug_frame_max "$DEBUG_FRAME_MAX" \
  --include_unmodified_templates \
  2>&1 | tee "$LOG/05_prep_05_export_fitbox.log"
```

正式实验可去掉 `--debug_frame_max`，并根据是否需要导出未修改人工模板决定是否保留 `--include_unmodified_templates`。

---

## 7. SSDR 工作目录与最小运行流程

SSDR staged layout：

```text
ssdr_work_dir/
├── data0_json/
├── data0/
├── data1/
├── data2/
├── data3/
└── data4/
```

阶段语义：

|阶段|输入|输出|说明|
|---|---|---|---|
|Step 00|`data0_json`|`data0`|JSON / PKL 标准化|
|Step 01|`data0`|`data1`|动画帧采样|
|Step 02|`data1`|`data2`|骨骼权重映射到 reduced skeleton|
|Step 03|`data2`|`data3`|QEM low-poly shape generation|
|Step 04|`data3`|`data4` / `opt_out`|权重与骨骼变换优化|

### 7.1 同步 Fitbox 到 SSDR

```bash
SRC="$DP/data_prep_05/fitbox_vdb"
DST="$SSDR/data0_json"

test -d "$SRC" || { echo "ERROR: source fitbox_vdb not found: $SRC"; exit 1; }

rm -rf "$DST"
mkdir -p "$DST"
cp -a "$SRC/." "$DST/"
```

### 7.2 Step 00：`data0_json -> data0`

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/ssdr/00_convert_json_to_pickle.py \
  --input_dir "$SSDR" \
  --overwrite_output \
  --max_workers 1 \
  2>&1 | tee "$LOG/06_ssdr_00_convert_json_to_pickle.log"
```

### 7.3 Step 01：`data0 -> data1`

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/ssdr/01_reduce_animation_frame.py \
  --input_dir "$SSDR" \
  --overwrite_output \
  --max_workers 1 \
  2>&1 | tee "$LOG/07_ssdr_01_reduce_animation_frame.log"
```

默认帧采样规则：

```text
frame_count <= 210: 保留全部
frame_count > 210 : 均匀采样到 200 帧
```

### 7.4 Step 02：`data1 -> data2`

如果导出了 `bone_reduced_001`：

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/ssdr/02_reduce_bone_weight.py \
  --input_dir "$SSDR" \
  --variant bone_reduced_001 \
  --overwrite_output \
  --max_workers 1 \
  2>&1 | tee "$LOG/08_ssdr_02_reduce_bone_weight.log"
```

如果没有显式 reduced variant，可以不传 `--variant`：

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/ssdr/02_reduce_bone_weight.py \
  --input_dir "$SSDR" \
  --overwrite_output \
  --max_workers 1 \
  2>&1 | tee "$LOG/08_ssdr_02_reduce_bone_weight.log"
```

### 7.5 Step 03：`data2 -> data3`

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/ssdr/03_create_low_poly_shapes.py \
  --input_dir "$SSDR" \
  --simp_rate_base 0.1 \
  --overwrite_output \
  --max_workers 8 \
  2>&1 | tee "$LOG/09_ssdr_03_create_low_poly_shapes.log"
```

核心参数：

```text
--simp_rate_base
```

可选配置文件：

```text
configs/ssdr_03_qem_config_example.json
```

如果未显式传入 `--config`，脚本会尝试读取：

```text
$SSDR/ssdr_03_qem_config.json
```

### 7.6 Step 04：`data3 -> data4 / opt_out`

Step 04 用于后续权重与骨骼变换优化。具体参数与实验设置相关，建议先查看：

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/ssdr/04_optimize_weight_and_bone_trans.py \
  --help
```

最小约定是从 `ssdr_work_dir/data3` 读取，并将结果写入 `data4` 或 `opt_out` 相关目录。

---

## 8. Fitbox / SSDR 结果导入 Blender 检查

统一检查工具：

```text
scripts/debug/import_fitbox_to_blend.py
```

### 8.1 Shape key 模式

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

### 8.2 LBS 模式

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

如果 Blender binary 不是默认路径，可设置：

```bash
export BLENDER_BIN=/path/to/blender
```

或显式传入：

```bash
--blender_bin /path/to/blender
```

