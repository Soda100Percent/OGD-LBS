#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from huggingface_hub import snapshot_download

def main():
    REPO = "Linzhan/Mixamo-Animations-Characters"
    OUT_DIR = "./mixamo_dataset"  # 下载到当前目录下的 mixamo_dataset 文件夹

    print(f"🚀 开始从 Hugging Face 下载 {REPO} 的数据集...")
    print(f"📁 保存路径: {os.path.abspath(OUT_DIR)}")
    print("⏳ 数据集包含 114 个角色和 2400+ 个动画，总大小可能在几GB到十几GB不等。")
    print("⚡ 正在多线程下载中（支持断点续传，如果中断再次运行即可）...\n")

    # 使用 snapshot_download 下载整个仓库的特定部分
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=OUT_DIR,
        # 只允许下载我们需要的文件类型（FBX和元数据）
        allow_patterns=[
            "character_refined/*.fbx",  # 精修过的带蒙皮角色
            "animation/*.fbx",          # 动画序列（根据你的终端输出路径）
            "animation_motion/*.fbx",   # 兼容另一种动作路径命名
            "*.csv",                    # 索引文件
            "*.json"                    # 描述文件
        ],
        # 忽略掉不需要的庞大文件，节省时间和空间
        ignore_patterns=[
            "animation_motion_render/**",  # 忽略庞大的预渲染视频文件
            "character_refined_glb/**",    # 忽略网页端用的 GLB 格式
            "character_tpose/**"           # 忽略未精修的原始 T-pose 文件
        ],
        max_workers=8  # 开启 8 个并发线程加速下载
    )

    print(f"\n✅ 下载完成！所有文件已保存至: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()