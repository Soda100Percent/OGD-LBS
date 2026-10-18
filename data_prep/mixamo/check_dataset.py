#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

def check_dataset_integrity(base_dir="./mixamo_dataset"):
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"❌ 错误: 找不到目录 '{base_dir}'。请确认下载脚本是否执行成功，或者路径是否正确。")
        return

    print(f"🔍 开始检查本地数据集: {base_path.resolve()}\n")

    # 1. 检查角色文件
    char_dir = base_path / "character_refined"
    char_files = list(char_dir.glob("*.fbx")) if char_dir.exists() else []
    print(f"👤 找到精修角色 (character_refined) FBX 文件数量: {len(char_files)}")

    # 2. 检查动画文件 (兼容 animation 和 animation_motion 两个可能的文件夹)
    anim_dir_1 = base_path / "animation"
    anim_dir_2 = base_path / "animation_motion"
    
    anim_files_1 = list(anim_dir_1.glob("*.fbx")) if anim_dir_1.exists() else []
    anim_files_2 = list(anim_dir_2.glob("*.fbx")) if anim_dir_2.exists() else []
    total_anims = len(anim_files_1) + len(anim_files_2)
    
    print(f"🏃 找到动画序列 FBX 文件数量: {total_anims}")
    if anim_dir_1.exists():
        print(f"   - animation 目录下: {len(anim_files_1)} 个")
    if anim_dir_2.exists():
        print(f"   - animation_motion 目录下: {len(anim_files_2)} 个")

    # 3. 检查元数据文件
    meta_files = list(base_path.glob("*.csv")) + list(base_path.glob("*.json"))
    print(f"\n📄 找到元数据文件 (CSV/JSON) 数量: {len(meta_files)}")
    for m in meta_files:
        print(f"   - {m.name}")

    # 4. 执行完整性/损坏检查 (检查是否有 0 字节的空文件)
    print("\n🛠️ 开始执行文件损坏检查 (扫描 0 字节文件)...")
    all_fbx = char_files + anim_files_1 + anim_files_2
    empty_files = [f for f in all_fbx if f.stat().st_size == 0]

    if empty_files:
        print(f"⚠️ 警告: 发现 {len(empty_files)} 个损坏(0字节)的 FBX 文件:")
        for f in empty_files[:5]:
            print(f"   - {f.relative_to(base_path)}")
        if len(empty_files) > 5:
            print("   ... (仅显示前5个)")
        print("💡 建议: 请重新运行下载脚本，huggingface_hub 会自动覆盖修复这些损坏的文件。")
    else:
        print("✅ 完美！所有 FBX 文件大小均正常，没有发现 0 字节的损坏文件。")

    # 5. 总结评估
    print("\n📊 总结报告:")
    if len(char_files) >= 100 and total_anims >= 2000:
        print("🎉 数据集看起来非常完整！角色数和动画数均符合预期，可以放心用于后续的 OGD-LBS 管线测试了。")
    else:
        print("⚠️ 数据量似乎偏少 (预期角色应有 ~114 个, 动画应有 ~2400+ 个)。")
        print("💡 如果你之前手动停止了下载，请再次运行 `python download_mixamo_full.py` 以继续断点续传。")

if __name__ == "__main__":
    check_dataset_integrity()