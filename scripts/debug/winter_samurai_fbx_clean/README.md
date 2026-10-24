

# Winter Samurai FBX Cleaning Pipeline

本目录包含用于清洗商业角色资产 **Winter Samurai** 的资产专用脚本。该流程用于将原始模型 FBX 与动画 FBX 规范化为 OGD-LBS 可继续处理的 Blender 文件，并复现论文实验所需的清洗版本。

> **重要说明**
> 
> - 本仓库 **不包含** Winter Samurai 商业资产本体。
> - 用户需要自行购买并准备对应 FBX 资源。
---

## 1. 目标

Winter Samurai 原始 FBX 资源存在以下问题：

1. Maya / FBX 导入后会带入大量 `EMPTY` / 控制器节点；
2. Blender 默认 FBX 导入得到的模型骨架 rest pose 坐标系质量不够稳定；
3. 动画 FBX 与模型 FBX 的骨架 rest pose 不一致，需要 rebase；
4. BetterFBX 导入的模型骨架 rest pose 更适合作为最终规范骨架；
5. 武器 `Weapon` 原本在部分动画中位置或挂载关系错误；
6. `Weapon` 原本是手部层级下的骨骼，最终需要 detach 为 root bone，并用虚拟挂载点方式重新烘焙。

最终目标是生成：

```text
Winter_Samurai_Normalized.blend
```

---

## 2. 数据准备

假设数据目录为：

```text
/workspace/ogd_dataset/Winter_Samurai
```

目录结构必须为：

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

这是原始 Winter Samurai 模型 FBX。

### 2.2 `rest_pose_base.blend`

这是使用 BetterFBX 导入器导入模型后保存得到的 Blender 文件。  
它提供最终使用的规范 rest pose 骨架。

本流程假设：

```text
model/rest_pose_base.blend
```

中存在与模型匹配的 BetterFBX  6.3.5 版 armature，骨骼名集合与保留 leaf bones 的 Blender FBX 导入结果一致。

### 2.3 为什么必须保留 leaf bones

本流程要求 Step 01 导入时使用：

```text
--no_ignore_leaf_bones
```

原因是 `rest_pose_base.blend` 中默认保留了若干末端骨骼，包括手指末端、脚趾、布料、装备等骨骼。若 Blender 默认导入时忽略 leaf bones，则模型骨架与 BetterFBX rest pose 骨架骨名集合不一致，Step 02a 会停止执行。

---

## 3. 脚本与配置文件

最终清洗流程依赖以下文件：

```text
scripts/debug/winter_samurai_fbx_clean/
├── 01_import_model_and_anims.py
├── 02_rebase_anims_to_skinning_armature.py
├── 02a_rebase_to_rest_pose_base.py
├── 04_fix_weapon_mount.py
├── 04_fix_weapon_mount_aim.py
├── configs/
│   └── winter_samurai_final.json
└── tests/
    └── run_final_cleaning.sh
```

### 3.1 `01_import_model_and_anims.py`

负责：

- 导入模型 FBX；
- 导入全部动画 FBX；
- 清理 FBX 导入产生的 `EMPTY` 对象；
- 记录 `model/rest_pose_base.blend` 是否存在；
- 保存 clean01 中间 `.blend`；
- 输出导入报告。

不负责：

- rest pose 清洗；
- 动画 rebase；
- 骨架合并；
- 武器修复。

### 3.2 `02_rebase_anims_to_skinning_armature.py`

负责：

- 识别模型 mesh 绑定的 skinning armature；
- 将动画 FBX 导入的动画骨架逐个 rebase 到模型骨架；
- 生成 `WS_*` Actions；
- 删除动画 FBX 导入骨架；
- 删除 rebase 前的旧 Action；
- 保存 clean02 `.blend`。


### 3.3 `02a_rebase_to_rest_pose_base.py`

负责：

- 读取 Step 02 输出；
- 读取 `model/rest_pose_base.blend` 中的 BetterFBX rest pose 骨架；
- 将 Step 02 动画从模型 FBX 骨架二次 rebase 到 BetterFBX rest pose 骨架；
- 保持采样帧下的 LBS mesh 世界空间变形一致；
- 重新烘焙所有 Actions；
- 输出 clean02a `.blend`；
- 生成完整数值审计报告；
- 可通过 `--audit_saved` 进行独立重读审计。

核心数学关系为：

```text
old_rest_world = old_armature_world @ old_rest
new_rest_world = new_armature_world @ new_rest

C = inverse(old_rest_world) @ new_rest_world
new_pose_world = old_pose_world @ C
```

该关系保证：

```text
new_pose_world @ inverse(new_rest_world)
=
old_pose_world @ inverse(old_rest_world)
```

即采样帧下的 LBS 变形算子保持一致。

### 3.4 `04_fix_weapon_mount.py`

负责通用 Weapon bone detach 与基于虚拟挂载点的 world-space rebake。

虽然文件名为 `04`，在最终资产清洗逻辑中它是武器清洗阶段的核心底层脚本。

### 3.5 `04_fix_weapon_mount_aim.py`

负责扩展武器清洗逻辑，支持 Walk3 这类特殊动作：

- 保留右手挂载位置；
- 将 `Weapon` 的局部 `-Y` 轴指向左手目标点；
- 目标点为 `Character1_LeftHand` 中点加局部偏移；
- 不使用 mirror；
- 不使用负 scale；
- 不创建持久 constraint；
- 不新增挂点骨骼。

### 3.6 `configs/winter_samurai_final.json`

最终资产清洗配置。

策略如下：

|Action|武器策略|
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

挂载定义：

```json
"hand_r": {
  "type": "direct",
  "mount_bone": "Character1_RightHand",
  "reference_action": "WS_Stance",
  "reference_frame": 1
}
```

```json
"back": {
  "type": "direct",
  "mount_bone": "Character1_Spine1",
  "reference_action": "WS_Run2",
  "reference_frame": 1
}
```

Walk3 特殊 aim 参数：

```json
"aim": {
  "axis": "-Y",
  "target_bone": "Character1_LeftHand",
  "target_offset_fraction_of_length": [-0.5, 0.5, 0.0]
}
```

含义：

- 使用右手挂载计算 `Weapon` 原点；
- 让 `Weapon` 的局部 `-Y` 轴指向左手目标点；
- 目标点为左手骨骼中点，外加：
    - `-0.5 * LeftHand_length` 沿左手局部 X；
    - `+0.5 * LeftHand_length` 沿左手局部 Y；
    - `0.0 * LeftHand_length` 沿左手局部 Z。

---

## 4. 一键完整清洗与回归测试

运行：

```bash
cd /workspace/git_projects/OGD-LBS

bash scripts/debug/winter_samurai_fbx_clean/tests/run_final_cleaning.sh
```

默认数据路径：

```text
/workspace/ogd_dataset/Winter_Samurai
```

默认输出路径：

```text
/workspace/ogd_tmp_blends/winter_samurai_fbx_clean/final_regression_v1/current
```

可通过环境变量覆盖：

```bash
OGD_WS_DATASET_DIR=/path/to/Winter_Samurai \
OGD_WS_FINAL_TEST_ROOT=/path/to/output_root \
OGD_WS_CONDA_ENV=lao-lbs \
bash scripts/debug/winter_samurai_fbx_clean/tests/run_final_cleaning.sh
```

---

## 5. 完整流程阶段

`run_final_cleaning.sh` 会依次执行：

### 5.1 Step 01: 导入全动画

命令内部等价于：

```bash
python 01_import_model_and_anims.py \
  --dataset_dir "$DATASET" \
  --no_ignore_leaf_bones \
  --output_dir "$RUN/01" \
  --output_blend "$RUN/01/Winter_Samurai_clean01.blend"
```

验收：

- 18 个动画 FBX 全部导入成功；
- 模型骨架 + 18 个动画骨架，共 19 个 armature；
- 每个 armature 有 110 根骨骼；
- 40 个 mesh；
- 0 个 EMPTY；
- `rest_pose_base.blend` 存在；
- 所有导入均使用完整 FBX importer 参数。

### 5.2 Step 02: 动画 rebase 到模型骨架

输出：

```text
$RUN/02/Winter_Samurai_clean02.blend
```

验收：

- 只有一个 armature；
- 40 个 mesh；
- 18 个 Actions；
- `WS_*` Action 全部存在；
- 所有动作重读后骨骼世界矩阵验证通过。

### 5.3 Step 02a: rebase 到 BetterFBX rest pose 骨架

输出：

```text
$RUN/02a/Winter_Samurai_clean02a_rest_base_rebased.blend
```

验收：

- 骨架 rest pose 来自 `rest_pose_base.blend`；
- 仍保留 110 根骨骼；
- 40 个 mesh；
- 18 个 Actions；
- mesh 顶点采样误差通过；
- 独立重读审计通过；
- 生成完整 cache，可复查。

### 5.4 武器清洗

输出：

```text
$RUN/03_weapon/Winter_Samurai_Normalized.blend
```

处理：

- `Weapon` 从原父骨骼 detach；
- `Weapon` 成为 root bone；
- 不修改 Weapon mesh 权重；
- 按最终配置重烘焙所有动作的 Weapon bone；
- `Run2`、`Run3` 使用后背挂载；
- `Walk3` 使用左手 aim；
- 其他动作使用右手挂载。

验收：

- `Weapon` parent 为 `None`；
- `Hips` 与 `Weapon` 均为 root bone；
- 40 个 mesh；
- 18 个 Actions；
- 0 个 EMPTY；
- 非 Weapon 动画曲线不被修改；
- mesh 基础数据、权重、UV、材质分配和绑定关系不变；
- weapon aim 独立审计通过；
- 最终结构审计通过。

---

## 6. 输出文件

完整测试通过后，关键输出为：

```text
$RUN/03_weapon/Winter_Samurai_Normalized.blend
```

以及报告：

```text
$RUN/final_test_summary.json
$RUN/final_invariants.json
$RUN/source_assets_sha256.json
$RUN/code_config_sha256.txt
$RUN/git_state.txt
$RUN/01/import_report.json
$RUN/02/rebase_report.json
$RUN/02a/rebase_report.json
$RUN/02a/independent_audit.json
$RUN/03_weapon/weapon_mount_fix_report.json
$RUN/03_weapon/aim_extension_report.json
$RUN/03_weapon/aim_audit_report.json
```

完整审计 cache：

```text
$RUN/02a/cache/
$RUN/03_weapon/aim_cache/
```

这些 cache 用于独立复查，不建议在最终确认前删除。

---

## 7. 成功标志

如果完整测试通过，会输出类似：

```text
FINAL CLEANING REGRESSION PASSED
Final blend: .../Winter_Samurai_Normalized.blend
Summary: .../final_test_summary.json
```

`final_test_summary.json` 中应包含：

```json
{
  "status": "passed",
  "final_blend": ".../Winter_Samurai_Normalized.blend",
  "walk3_target_offset_fraction_of_length": [-0.5, 0.5, 0.0],
  "back_reference": "WS_Run2 / frame 1 / Character1_Spine1"
}
```

注意：

```json
"visual_review": "pending_for_this_end_to_end_run"
```

表示脚本只记录数值回归通过。  
人工视觉检查需由用户手动完成。

---

## 8. 人工视觉检查

打开最终文件：

```bash
/workspace/blender-4.5.12-linux-x64/blender \
  /workspace/ogd_tmp_blends/winter_samurai_fbx_clean/final_regression_v1/current/03_weapon/Winter_Samurai_Normalized.blend
```

至少检查以下动作：

1. `WS_Stance`
    - 右手握持正确；
2. `WS_Attack1`–`WS_Attack4`
    - 攻击动作武器跟随右手；
3. `WS_Run2`, `WS_Run3`
    - 武器挂在背后；
4. `WS_Walk3`
    - 武器在右手原点基础上朝向左手；
    - 左手握持姿态正确；
5. 其他 Idle / Walk / Death 动作
    - 无明显武器漂移；
    - mesh 无飞散；
    - 无额外 EMPTY 或动画骨架残留。
