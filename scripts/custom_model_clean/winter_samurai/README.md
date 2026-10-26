

# Winter Samurai Custom Cleaning Pipeline

本目录包含 **Winter Samurai** 资产的专用清洗入口脚本，用于将原始模型 FBX、动画 FBX 与 BetterFBX rest pose 参考文件规范化为 OGD-LBS 后续 Data Prep 可使用的 Blender 文件。

---

## 1. 目标

Winter Samurai 原始 FBX 资产在进入 OGD-LBS Data Prep 前需要完成以下清洗：

1. 导入模型 FBX 与全部动画 FBX；
2. 移除 FBX 导入产生的 `EMPTY` / 控制器节点；
3. 将动画 FBX 导入得到的动画骨架动作 rebase 到模型 skinning armature；
4. 将模型 FBX 骨架进一步 rebase 到 BetterFBX （如6.3.5版本）生成的 `rest_pose_base.blend` 骨架；
5. 保持采样帧下的 LBS mesh 世界空间变形一致；
6. 修复武器骨骼 `Weapon`：
    - detach 为 root bone；
    - 按动作重新烘焙 hand / back 挂载；
    - 对 `WS_Walk3` 使用右手原点加左手 aim 策略；
7. 输出最终 normalized Blender 文件：

```text
Winter_Samurai_Normalized.blend
```

最终清洗结果预期满足：

```text
armatures = 1
bones     = 110
meshes    = 40
actions   = 18
empties   = 0
Weapon parent = None
root bones = Character1_Hips, Weapon
```

---

## 2. 数据准备

默认数据目录为：

```text
/workspace/ogd_dataset/Winter_Samurai
```

目录结构要求：

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

### 2.1 `Base_Mesh_Rig.fbx`

原始模型 FBX。Step 01 会导入该文件作为 mesh 与 skinning armature 的来源。

### 2.2 `rest_pose_base.blend`

使用 BetterFBX 导入模型后保存得到的 Blender 文件。该文件提供最终使用的规范 rest pose armature data。

本流程假设：

```text
model/rest_pose_base.blend
```

中存在与模型匹配的 armature，并且骨骼名集合与保留 leaf bones 的 Blender FBX 导入结果一致。

### 2.3 为什么必须保留 leaf bones

Step 01 必须使用：

```text
--no_ignore_leaf_bones
```

原因是 `rest_pose_base.blend` 保留了手指末端、脚趾、布料、装备等 leaf bones。若 Blender FBX 导入阶段忽略 leaf bones，则模型骨架与 BetterFBX rest pose 骨架的骨名集合不一致，Step 03 会停止执行。

---

## 3. 目录结构

当前 custom cleaning 目录为：

```text
scripts/custom_model_clean/winter_samurai/
├── __init__.py
├── 01_import_model_and_anims.py
├── 02_rebase_anims_to_skinning_armature.py
├── 03_rebase_to_rest_pose_base.py
├── 04_fix_weapon_mount_aim.py
├── configs/
│   └── winter_samurai_final.json
└── tests/
    └── run_final_cleaning.sh
```

底层通用工具位于：

```text
src/ogd_lbs/blender/
├── transforms.py
├── fbx_io.py
├── object_cleanup.py
├── action_utils.py
├── pose_bake.py
├── armature_ops.py
└── rebase.py

src/ogd_lbs/utils/
└── progress.py
```

---

## 4. 通用工具分层

### 4.1 `src/ogd_lbs/blender/transforms.py`

通用矩阵工具：

- Blender Matrix / NumPy matrix 转换；
- affine matrix 检查；
- similarity transform 检查；
- closest rotation；
- 矩阵误差统计；
- reflection matrix；
- local axis aim 到目标点。

### 4.2 `src/ogd_lbs/blender/fbx_io.py`

通用 FBX 导入封装：

- 启用 / 检查 Blender FBX importer；
- full kwargs / reduced kwargs / filepath-only fallback；
- 导入前后 object / action 差集统计。

### 4.3 `src/ogd_lbs/blender/object_cleanup.py`

通用对象清理：

- 删除指定对象；
- 保持 surviving child object 当前 world matrix；
- EMPTY 清理；
- animated object / empty 统计。

### 4.4 `src/ogd_lbs/blender/action_utils.py`

通用 Action / FCurve 工具：

- legacy Action 与 Blender 4.4+ slotted Action 兼容遍历；  
    -设置 armature active action；
- reset pose；
- 删除指定 bone fcurves；
- 设置 interpolation / extrapolation。

### 4.5 `src/ogd_lbs/blender/pose_bake.py`

通用 pose capture / bake：

- 采样 evaluated bone world matrices；
- parent-first pose bone order；
- 使用 `PoseBone.matrix` setter 反解局部 TRS；
- 烘焙 location / rotation_quaternion / scale；
- quaternion sign continuity；
- action world matrix validation。

### 4.6 `src/ogd_lbs/blender/armature_ops.py`

通用 armature 操作：

- mesh-bound armature 检测；
- skeleton snapshot；
- detach 任意 bone 并保持 rest matrix。

### 4.7 `src/ogd_lbs/blender/rebase.py`

通用 rebase / virtual mount 数学：

- rest basis correction；
- LBS deformation operator；
- direct mount offset；
- mirror mount offset；
- aimed world sequence。

### 4.8 `src/ogd_lbs/utils/progress.py`

无依赖进度工具。支持：

```text
OGD_PROGRESS=auto
OGD_PROGRESS=tty
OGD_PROGRESS=bar
OGD_PROGRESS=log
OGD_PROGRESS=off
```

默认 `auto` 模式下：

- TTY 中使用单行刷新进度条；
- `tee` 管道下优先写 `/dev/tty`，避免污染 log；
- 无可用 TTY 时退化为稀疏 checkpoint log。

---

## 5. 完整清洗流程

完整流程为：

```text
01 import model + animations
  → 02 rebase animation armatures to skinning armature
  → 03 rebase to BetterFBX rest pose base
  → 04 detach / rebake Weapon mount and aim
  → Winter_Samurai_Normalized.blend
```

---

## 6. Step 01：导入模型与动画

脚本：

```text
01_import_model_and_anims.py
```

职责：

- 导入 `model/Base_Mesh_Rig.fbx`；
- 导入一个、多个或全部动画 FBX；
- 清理 FBX 导入产生的 `EMPTY` 对象；
- 保持 surviving objects 的当前 world matrix；
- 记录 `model/rest_pose_base.blend` 是否存在；
- 保存 clean01 blend；
- 输出导入报告和 Action inventory。

典型命令：

```bash
/workspace/blender-4.5.12-linux-x64/blender -b --factory-startup \
  --python scripts/custom_model_clean/winter_samurai/01_import_model_and_anims.py -- \
  --dataset_dir /workspace/ogd_dataset/Winter_Samurai \
  --no_ignore_leaf_bones \
  --output_dir /path/to/run/01 \
  --output_blend /path/to/run/01/Winter_Samurai_clean01.blend
```

输出：

```text
/path/to/run/01/
├── Winter_Samurai_clean01.blend
├── import_report.json
├── action_inventory.json
└── run.log
```

验收要点：

```text
armatures = 19
meshes    = 40
empties   = 0
actions   = 18 * importer action count
bones per armature = 110
ignore_leaf_bones = false
```

---

## 7. Step 02：动画 rebase 到 skinning armature

脚本：

```text
02_rebase_anims_to_skinning_armature.py
```

职责：

- 识别唯一 mesh-bound skinning armature；
- 将动画 FBX 导入得到的 animation armatures 逐个 rebase 到 skinning armature；
- 生成 clean `WS_*` Actions；
- 删除 animation armatures；
- 删除 pre-rebase FBX importer Actions；
- 保存 clean02 blend；
- 保存采样 world matrix cache；
- 保存重读后验证报告。

典型命令：

```bash
/workspace/blender-4.5.12-linux-x64/blender -b --factory-startup \
  --python scripts/custom_model_clean/winter_samurai/02_rebase_anims_to_skinning_armature.py -- \
  --input_blend /path/to/run/01/Winter_Samurai_clean01.blend \
  --import_report /path/to/run/01/import_report.json \
  --output_dir /path/to/run/02 \
  --output_blend /path/to/run/02/Winter_Samurai_clean02.blend \
  --position_tol 1e-4
```

输出：

```text
/path/to/run/02/
├── Winter_Samurai_clean02.blend
├── rebase_report.json
├── source_world_cache.npz
├── world_pose_error_per_action_bone.csv
└── run.log
```

验收要点：

```text
armatures = 1
meshes    = 40
empties   = 0
actions   = 18
generated actions = WS_Attack1 ... WS_Walk3
```

---

## 8. Step 03：rebase 到 BetterFBX rest pose base

脚本：

```text
03_rebase_to_rest_pose_base.py
```

这是旧 debug 流程中的 `02a_rebase_to_rest_pose_base.py` 的 custom pipeline 版本。

职责：

- 读取 Step 02 blend；
- 读取 `model/rest_pose_base.blend` 中的 BetterFBX armature data；
- 替换当前 armature data；
- 将动作从模型 FBX rest pose 二次 rebase 到 BetterFBX rest pose；
- 保持采样帧下的 LBS mesh 世界空间变形一致；
- 重新烘焙所有 `WS_*` Actions；
- 输出 clean03 blend；
- 生成完整审计 cache；
- 支持 `--audit_saved` 独立重读审计。

核心关系：

```text
old_rest_world = old_armature_world @ old_rest
new_rest_world = new_armature_world @ new_rest

C = inverse(old_rest_world) @ new_rest_world
new_pose_world = old_pose_world @ C
```

该关系保证采样帧下：

```text
new_pose_world @ inverse(new_rest_world)
=
old_pose_world @ inverse(old_rest_world)
```

即 LBS deformation operator 在世界空间保持一致。

典型命令：

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/custom_model_clean/winter_samurai/03_rebase_to_rest_pose_base.py \
  --input_blend /path/to/run/02/Winter_Samurai_clean02.blend \
  --dataset_dir /workspace/ogd_dataset/Winter_Samurai \
  --output_dir /path/to/run/03 \
  --samples_per_frame 1 \
  --position_tol 1e-4 \
  --mesh_position_tol 1e-4
```

独立审计：

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/custom_model_clean/winter_samurai/03_rebase_to_rest_pose_base.py \
  --audit_saved \
  --output_dir /path/to/run/03
```

输出：

```text
/path/to/run/03/
├── Winter_Samurai_clean03_rest_base_rebased.blend
├── rebase_report.json
├── independent_audit.json
├── audit_manifest.json
├── basis_map.npz
├── mesh_error_after_reload.csv
└── cache/
    ├── action_000.npz
    ├── ...
    ├── vertices_000.npy
    └── ...
```

验收要点：

```text
bones   = 110
meshes  = 40
actions = 18
validation_after_reload.passed = true
independent_audit.passed = true
mesh vertex world error <= mesh_position_tol
```

### 8.1 Capture 与 Bake 的含义

Step 03 中会看到类似：

```text
Capture WS_Attack1
Bake WS_Attack1
```

含义如下：

- **Capture**：在旧 rest / 旧骨架下记录每个 Action 的骨骼 world matrix、mesh world vertices、mesh object world matrix，作为 rebase 后需要复现的目标；
- **Bake**：替换到 BetterFBX rest pose 后，根据 Capture 阶段缓存的目标 world pose 重新生成适配新 rest pose 的 `WS_*` Actions；
- **Audit**：重新读取保存后的 blend，验证新 Action 是否在采样帧上复现 Capture 阶段的结果。

流程为：

```text
Capture old evaluated result
  → replace armature data with reference rest pose
  → Bake new Actions
  → Audit new evaluated result
```

### 8.2 归档目录下的独立审计

`audit_manifest.json` 内部保存的是首次运行时的绝对路径。如果整个 run 目录从：

```text
current/
```

归档为：

```text
previous_###/
```

Step 03 的 `--audit_saved` 会自动 fallback 到当前 `--output_dir` 下的同名 blend 与输入文件，并在 `independent_audit.json` 中记录：

```text
manifest_output_blend
resolved_output_blend
resolved_input_hashes
missing_input_hash_candidates
```

---

## 9. Step 04：武器 detach、挂载与 aim 修复

脚本：

```text
04_fix_weapon_mount_aim.py
```

custom 目录中只保留这个单入口 Step 04。它内嵌旧 weapon mount core，并叠加 aim extension。

职责：

- 将 `Weapon` bone 从原父骨骼 detach；
- 保持 `Weapon` rest matrix；
- 让 `Weapon` 成为 root bone；
- 根据配置对所有动作重新烘焙 `Weapon` bone；
- `WS_Run2` / `WS_Run3` 使用后背挂载；
- `WS_Walk3` 使用右手原点 + 左手 aim；
- 其余动作使用右手挂载；
- 保持 mesh 基础数据、权重、材质、UV、绑定不变；
- 保持非 Weapon 动画曲线不变；
- 输出最终 normalized blend。

典型命令：

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/custom_model_clean/winter_samurai/04_fix_weapon_mount_aim.py \
  --input_blend /path/to/run/03/Winter_Samurai_clean03_rest_base_rebased.blend \
  --output_dir /path/to/run/04_weapon \
  --output_blend /path/to/run/04_weapon/Winter_Samurai_Normalized.blend \
  --config scripts/custom_model_clean/winter_samurai/configs/winter_samurai_final.json \
  --position_tol 1e-4
```

独立审计：

```bash
conda run --no-capture-output -n lao-lbs \
  python scripts/custom_model_clean/winter_samurai/04_fix_weapon_mount_aim.py \
  --audit_saved \
  --output_dir /path/to/run/04_weapon
```

输出：

```text
/path/to/run/04_weapon/
├── Winter_Samurai_Normalized.blend
├── weapon_mount_fix_report.json
├── weapon_mount_fix_frame_errors.csv
├── aim_extension_report.json
├── aim_audit_report.json
├── aim_audit_manifest.json
├── independent_aim_audit.log
└── aim_cache/
```

验收要点：

```text
armatures = 1
meshes    = 40
empties   = 0
actions   = 18
Weapon parent = None
Weapon root = true
aim audit passed = true
```

---

## 10. 最终武器配置

配置文件：

```text
configs/winter_samurai_final.json
```

主要策略：

|Action|策略|
|---|---|
|`WS_Attack1`–`WS_Attack4`|右手挂载|
|`WS_Death`, `WS_Death2`|右手挂载|
|`WS_Idle1`–`WS_Idle5`|右手挂载|
|`WS_Run1`|右手挂载|
|`WS_Run2`|后背挂载|
|`WS_Run3`|后背挂载|
|`WS_Stance`|右手挂载|
|`WS_Walk`, `WS_Walk2`|右手挂载|
|`WS_Walk3`|右手原点 + 左手 aim|

右手挂载：

```json
"hand_r": {
  "type": "direct",
  "mount_bone": "Character1_RightHand",
  "reference_action": "WS_Stance",
  "reference_frame": 1
}
```

后背挂载：

```json
"back": {
  "type": "direct",
  "mount_bone": "Character1_Spine1",
  "reference_action": "WS_Run2",
  "reference_frame": 1
}
```

Walk3 aim：

```json
"aim": {
  "axis": "-Y",
  "target_bone": "Character1_LeftHand",
  "target_offset_fraction_of_length": [-0.5, 0.5, 0.0]
}
```

含义：

- 使用右手挂载计算 `Weapon` 原点；
- 让 `Weapon` 局部 `-Y` 轴指向左手目标点；
- 左手目标点为 `Character1_LeftHand` 中点加局部偏移：
    - `-0.5 * LeftHand_length` 沿左手局部 X；
    - `+0.5 * LeftHand_length` 沿左手局部 Y；
    - `0.0 * LeftHand_length` 沿左手局部 Z。

---

## 11. 一键完整回归

脚本：

```text
tests/run_final_cleaning.sh
```

运行：

```bash
cd /workspace/git_projects/OGD-LBS

export OGD_PROGRESS=auto

bash scripts/custom_model_clean/winter_samurai/tests/run_final_cleaning.sh
```

默认输入：

```text
/workspace/ogd_dataset/Winter_Samurai
```

默认输出：

```text
/workspace/ogd_tmp_blends/custom_model_clean/winter_samurai/final_regression_v1/current
```

可覆盖环境变量：

```bash
OGD_WS_DATASET_DIR=/path/to/Winter_Samurai \
OGD_WS_FINAL_TEST_ROOT=/path/to/output_root \
OGD_WS_CONDA_ENV=lao-lbs \
OGD_BLENDER_BIN=/workspace/blender-4.5.12-linux-x64/blender \
OGD_PROGRESS=auto \
bash scripts/custom_model_clean/winter_samurai/tests/run_final_cleaning.sh
```

完整回归会依次执行：

```text
01 import model and animations
02 rebase animations to skinning armature
03 rebase to BetterFBX rest pose base
03 independent audit
04 weapon mount + aim
04 independent aim audit
final invariants check
final summary
```

成功后输出：

```text
FINAL CLEANING REGRESSION PASSED
Final blend: .../04_weapon/Winter_Samurai_Normalized.blend
Summary:     .../final_test_summary.json
```

---

## 12. 完整回归输出结构

```text
$RUN/
├── config_used.json
├── final_test_summary.json
├── final_invariants.json
├── source_assets_sha256.json
├── code_config_sha256.txt
├── git_state.txt
├── check_stage.py
├── code/
│   ├── 01_import_model_and_anims.py
│   ├── 02_rebase_anims_to_skinning_armature.py
│   ├── 03_rebase_to_rest_pose_base.py
│   └── 04_fix_weapon_mount_aim.py
├── 01/
│   ├── Winter_Samurai_clean01.blend
│   ├── import_report.json
│   ├── action_inventory.json
│   └── run.log
├── 02/
│   ├── Winter_Samurai_clean02.blend
│   ├── rebase_report.json
│   ├── source_world_cache.npz
│   ├── world_pose_error_per_action_bone.csv
│   └── run.log
├── 03/
│   ├── Winter_Samurai_clean03_rest_base_rebased.blend
│   ├── rebase_report.json
│   ├── independent_audit.json
│   ├── audit_manifest.json
│   ├── basis_map.npz
│   ├── mesh_error_after_reload.csv
│   ├── run.log
│   ├── independent_audit.log
│   └── cache/
└── 04_weapon/
    ├── Winter_Samurai_Normalized.blend
    ├── weapon_mount_fix_report.json
    ├── weapon_mount_fix_frame_errors.csv
    ├── aim_extension_report.json
    ├── aim_audit_report.json
    ├── aim_audit_manifest.json
    ├── run.log
    ├── independent_aim_audit.log
    └── aim_cache/
```

---

## 13. 成功标志

`final_test_summary.json` 中应包含：

```json
{
  "status": "passed",
  "final_blend": ".../04_weapon/Winter_Samurai_Normalized.blend",
  "source_assets_unchanged": true,
  "walk3_target_offset_fraction_of_length": [-0.5, 0.5, 0.0],
  "back_reference": "WS_Run2 / frame 1 / Character1_Spine1"
}
```

`final_invariants.json` 中应包含：

```json
{
  "status": "passed",
  "armatures": 1,
  "bones": 110,
  "meshes": 40,
  "vertices": 24953,
  "empty_count": 0,
  "root_bones": ["Character1_Hips", "Weapon"],
  "weapon_descendants": ["Weapon2"],
  "base_meshes_weights_material_assignments_uvs_and_binding_unchanged": true,
  "non_weapon_animation_curves_unchanged": true
}
```

注意：

```json
"visual_review": "pending_for_this_end_to_end_run"
```

表示脚本已完成数值审计，但最新输出仍需人工视觉检查。

---

## 14. 人工视觉检查

打开最终文件：

```bash
/workspace/blender-4.5.12-linux-x64/blender \
  /workspace/ogd_tmp_blends/custom_model_clean/winter_samurai/final_regression_v1/current/04_weapon/Winter_Samurai_Normalized.blend
```

建议至少检查：

1. `WS_Stance`
    - 右手握持是否正确；
2. `WS_Attack1`–`WS_Attack4`
    - 武器是否稳定跟随右手；
3. `WS_Run2`, `WS_Run3`
    - 武器是否在背部；
4. `WS_Walk3`
    - 武器是否以右手为原点；
    - `Weapon` 局部 `-Y` 是否朝向左手目标；
    - 左手握持视觉是否合理；
5. `WS_Idle*`, `WS_Walk`, `WS_Walk2`, `WS_Death*`
    - 是否无明显武器漂移；
    - mesh 是否无飞散；
    - 场景中是否没有多余 animation armature 或 EMPTY 残留。
