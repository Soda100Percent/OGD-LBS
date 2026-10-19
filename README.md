

# OGD-LBS

> Code of Paper: **"OGD-LBS: Animation-Guided Collaborative Lightweighting for Pre-Bound Multi-Component LBS Characters"**

本项目包含了 OGD-LBS 论文的官方实现代码。

## 📂 项目结构

```text
OGD-LBS/
├── src/
│   └── ogd_lbs/
│       ├── blender/        # Blender (bpy) 底层操控与渲染工具库
│       └── utils/          # 通用工具（日志、IO、文件发现等）
├── scripts/                # 自动化执行脚本 (CLI)
│   └── prep_01_concat_nla.py  # 动作序列自动拼接脚本
├── pyproject.toml          # Python 项目配置
└── README.md
```

## 🛠️ 环境依赖

本项目依赖 Python 环境。

1. **Python 环境**:  
    推荐使用 Conda 管理环境。项目已配置为标准的 Python Package。
    
    ```bash
    conda activate lao-lbs
    pip install -e .
    ```
    

## 🚀 数据预处理管线 (Data Pipeline)

针对购买的商业 `.blend` 资产（包含复杂的骨架和零散的动作），我们设计了一套自动化的预处理管线，以生成符合 OGD-LBS 训练要求的输入格式。

### Step 1: 动作序列拼接 (NLA Concatenation)

将商业模型中零散的动作（Actions）提取并依次拼接为一条完整的非线性动画（NLA）轨道，用于后续的连续训练。

```bash
python scripts/prep_01_concat_nla.py \
    --input_blend /path/to/your/model.blend \
    --gap 10 \
    --exclude "T-pose,Idle"
```

- **自动输出**: 脚本会自动在同目录下生成 `*_prep_01.blend` 文件，绝不覆盖原始资产。
- **特性**: 自动清理冗余轨道、自动计算帧长度、强制进入 Pose 模式以防静止锁定。


## ⚠️ Blender 4.4+ 兼容性说明

本项目脚本（如 `prep_01_concat_nla.py`）已针对 Blender 4.4+ 的 **Slotted Actions (动作插槽)** 进行了底层适配：

- 解决了新版动画系统中 NLA Strip 默认不评估（不播放）的问题。
- 脚本会自动读取 Action 的 `slots` 属性，并将 `Legacy Slot` 或对应的兼容插槽显式绑定至 `strip.action_slot`。
- 保证了在 Blender 4.5+ 环境下，多动作拼接后的依赖图（Depsgraph）能够被正确驱动。

