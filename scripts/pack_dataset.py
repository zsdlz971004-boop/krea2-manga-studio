#!/usr/bin/env python3
"""
将训练图片与打标 .txt 打包为可直接上传至 Google Drive (AI_Manga_Studio/datasets/) 的 .zip 压缩包。
用法示例:
  python3 scripts/pack_dataset.py --name "inoue_style"
"""

import argparse
import zipfile
from pathlib import Path


def pack_dataset(dataset_name: str):
    src_dir = Path("datasets") / dataset_name
    if not src_dir.exists():
        raise FileNotFoundError(f"未找到数据集目录: {src_dir}")

    zip_path = Path("datasets") / f"{dataset_name}.zip"
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(src_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".txt"):
                zf.write(file_path, arcname=file_path.name)
                count += 1

    print(f"✅ 打包完成！共写入 {count} 个文件（图片 + .txt 打标）")
    print(f"📦 压缩包路径: {zip_path.resolve()}")
    print(f"🚀 现在只需将 [{zip_path.name}] 拖入 Google 云端硬盘的 `AI_Manga_Studio/datasets/` 文件夹即可在 Colab 一键开炼！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="打包 Krea 2 LoRA 训练数据集")
    parser.add_argument("--name", required=True, help="数据集文件夹名称")
    args = parser.parse_args()
    pack_dataset(args.name)
