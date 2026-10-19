

# OGD-LBS

> Code of Paper: **"OGD-LBS: Animation-Guided Collaborative Lightweighting for Pre-Bound Multi-Component LBS Characters"**

本项目包含了 OGD-LBS 论文的官方实现代码。针对商业级、多组件、预绑定（Pre-Bound）的复杂 LBS 角色资产，本项目提供了一套完整的自动化数据预处理与轻量化管线。

## 📂 项目结构

```text
OGD-LBS/
├── configs/                # 预处理管线配置文件 (如 SDF 融合参数)
├── src/
│   └── ogd_lbs/
│       ├── blender/        # Blender (bpy) 底层操控与渲染工具库
│       └── utils/          # 通用工具（日志、IO、文件发现等）
├── scripts/                # 自动化执行脚本 (CLI)
│   ├── prep_01_concat_nla.py       # 动作序列自动拼接
│   ├── prep_02a_sdf_fuse_5x.py     # 基于拓扑感知的 SDF 网格融合 (Blender 5.x)
│   └── prep_02b_import_fused_4x.py # 融合网格导回与场景清理 (Blender 4.x)
├── pyproject.toml          # Python 项目配置
└── README.md
```

## 🛠️ 环境依赖

为了兼顾主线管线的稳定性与最新 SDF 几何节点的特性，本项目采用**双 Conda 环境**隔离策略：

1. **主环境 (`lao-lbs`)**:  
    用于运行 Prep 01、Prep 02b 以及后续的大部分处理。
    
    ```bash
    conda activate lao-lbs
    pip install -e .
    ```
    
    _(依赖: `bpy==4.4.0` 及其他常规数据处理库)_
    
2. **SDF 融合环境 (`ogd-bpy52`)**:  
    专用于运行 Prep 02a，利用 Blender 5.2 LTS 的原生 SDF Grid 节点进行精确的体素融合，并使用 `trimesh` 进行拓扑分析。
    
    ```bash
    conda create -n ogd-bpy52 python=3.13 -y
    conda activate ogd-bpy52
    pip install bpy==5.2.2 trimesh networkx numpy
    ```
    

## 🚀 数据预处理管线 (Data Pipeline)

针对购买的商业 `.blend` 资产（包含复杂的骨架、零散的动作、多组件交叉网格），我们设计了以下自动化预处理管线：

### Step 1: 动作序列拼接 (NLA Concatenation)

将商业模型中零散的动作（Actions）提取并依次拼接为一条完整的非线性动画（NLA）轨道。

- **环境**: `lao-lbs` (Blender 4.x)

```bash
python scripts/prep_01_concat_nla.py \
    --input_blend /path/to/your/model.blend \
    --gap 10
```

- **特性**: 自动适配 Blender 4.4+ 的 Slotted Actions (动作插槽)，强制解除 Rest 锁定，输出 `*_prep_01.blend`。

### Step 2a: 拓扑感知的 SDF 网格融合 (Topology-Aware SDF Fusion)

将多组件网格转化为水密（Watertight）、流形（Manifold）且无内部相交的单一网格。脚本内置了**自动调参（Auto-Tuning）**逻辑，通过内存直读评估网格的欧拉示性数（Euler Characteristic），自动寻找能消除孔洞（Genus g=0g=0g=0）的最佳膨胀偏移量。

- **环境**: `ogd-bpy52` (Blender 5.x)

```bash
python scripts/prep_02a_sdf_fuse_5x.py \
    --input_blend /path/to/your/model_prep_01.blend \
    --config configs/prep_02_sdf_merge_config_griffin.json \
    --auto_offset
```

- **特性**:
    - 使用 Blender 5.x 原生 `Mesh to SDF Grid` 节点。
    - 自动步进搜索 Offset，直至满足连通分量 C=1C=1C=1 且亏格 g=0g=0g=0。
    - 导出融合后的静态 OBJ 及 `manifest.json`。

### Step 2b: 融合网格导回与场景清理 (Import & Cleanup)

读取 2a 阶段生成的 `manifest.json`，将完美的静态融合网格导回主干工程，并隐藏原始多组件网格。

- **环境**: `lao-lbs` (Blender 4.x)

```bash
python scripts/prep_02b_import_fused_4x.py \
    --input_blend /path/to/your/model_prep_01.blend \
    --manifest /path/to/your/manifest.json
```

- **输出**: 最终生成干净、规范的 `*_prep_02.blend`，为后续的重拓扑（Remesh）与权重传递（Weight Transfer）做好准备。

## ⚠️ 核心兼容性说明

1. **Blender 4.4+ 动画系统**: 本项目已在底层适配 Blender 4.4+ 的新版动作插槽（Slotted Actions），解决了 NLA Strip 默认不评估的问题，确保多动作拼接后的依赖图（Depsgraph）能被正确驱动。
2. **拓扑评估优化**: Prep 02a 摒弃了传统的 OBJ 临时导出方案，直接从 Blender 内存中读取 Evaluated Mesh 并构建拓扑图。这彻底解决了因 OBJ 导出导致的“法线/UV撕裂”引发的伪拓扑（Triangle Soup）问题，使自动调参既精准又极速。