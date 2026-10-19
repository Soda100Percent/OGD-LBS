#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch render previews for FBX/GLB assets")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing 3D assets")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory for the data source")
    parser.add_argument("--whitelist", type=str, default="all", help="Comma-separated list of asset names, or 'all'")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    whitelist_raw = args.whitelist.strip().lower()

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    # 同时扫描 FBX 和 GLB/GLTF
    asset_files = list(input_dir.rglob("*.fbx")) + list(input_dir.rglob("*.glb")) + list(input_dir.rglob("*.gltf"))
    
    target_files = []
    if whitelist_raw == "all":
        target_files = asset_files
    else:
        whitelist = [name.strip() for name in whitelist_raw.split(",")]
        for asset in asset_files:
            if asset.stem.lower() in whitelist:
                target_files.append(asset)

    if not target_files:
        print(f"[WARN] No assets matched the whitelist '{args.whitelist}' in {input_dir}")
        sys.exit(0)

    target_files.sort(key=lambda x: x.stem)
    print(f"[INFO] Found {len(target_files)} assets to process.")

    success_count = 0
    for i, asset in enumerate(target_files, 1):
        char_dir = output_dir / asset.stem
        char_dir.mkdir(parents=True, exist_ok=True)
        out_png = char_dir / "preview.png"
        
        if out_png.exists():
            print(f"\n[{i}/{len(target_files)}] Skipping {asset.name}, preview already exists.")
            success_count += 1
            continue
            
        print(f"\n[{i}/{len(target_files)}] Processing {asset.name} ...")
        
        cmd = [
            sys.executable, 
            "-m", "ogd_lbs.dataset.mixamo.mixamo_renderer",
            "--asset", str(asset),
            "--out", str(out_png)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[SUCCESS] Saved preview to {out_png}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to process {asset.name}")
            print(f"--- STDOUT ---\n{e.stdout}")
            print(f"--- STDERR ---\n{e.stderr}")

    print(f"\n[INFO] Batch processing complete! {success_count}/{len(target_files)} successful.")
    print(f"[INFO] Assets are organized in: {output_dir}")

if __name__ == "__main__":
    main()
