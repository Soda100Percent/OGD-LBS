#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("错误: 未安装 'datasets' 库。请先运行: pip install datasets")
    sys.exit(1)

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="查看 Hugging Face 上 Mixamo 数据集的索引信息")
    parser.add_argument("-t", "--type", choices=["all", "chars", "anims"], default="all", 
                        help="选择要查看的数据类型：all(全部), chars(仅角色), anims(仅动画)")
    parser.add_argument("-n", "--num", type=int, default=5, 
                        help="要显示的记录条数 (默认: 5)")
    parser.add_argument("-s", "--search", type=str, default="", 
                        help="关键词搜索 (例如: zombie, walk, mutant)")
    
    args = parser.parse_args()
    REPO = "Linzhan/Mixamo-Animations-Characters"
    
    print(f"正在从 Hugging Face 加载 {REPO} 的元数据 (仅加载文本索引，不下载大文件)...\n")

    # ---------------- 查看角色 (Characters) ----------------
    if args.type in ["all", "chars"]:
        chars = load_dataset(REPO, "characters", split="train")
        print(f"=== 角色 (Characters) 总数: {len(chars)} ===")
        
        if args.search:
            # 简单的关键词过滤 (忽略大小写)
            filtered_chars = [c for c in chars if args.search.lower() in str(c).lower()]
            print(f"找到 {len(filtered_chars)} 个包含关键词 '{args.search}' 的角色:")
            for i, c in enumerate(filtered_chars[:args.num]):
                print(f"  [{i+1}] {c}")
        else:
            for i in range(min(args.num, len(chars))):
                print(f"  [{i+1}] {chars[i]}")
        print()

    # ---------------- 查看动画 (Animations) ----------------
    if args.type in ["all", "anims"]:
        anims = load_dataset(REPO, "animations", split="train")
        print(f"=== 动画 (Animations) 总数: {len(anims)} ===")
        
        if args.search:
            filtered_anims = [a for a in anims if args.search.lower() in str(a).lower()]
            print(f"找到 {len(filtered_anims)} 个包含关键词 '{args.search}' 的动画:")
            for i, a in enumerate(filtered_anims[:args.num]):
                print(f"  [{i+1}] {a}")
        else:
            for i in range(min(args.num, len(anims))):
                print(f"  [{i+1}] {anims[i]}")
        print()

if __name__ == "__main__":
    main()