#!/usr/bin/env python3
"""
从 Danbooru 自动抓取指定画师/角色的高赞高清代表作，并过滤不适合训练 Krea 2 LoRA 的废图（长条漫、低清图、动图）。
用法示例:
  python3 scripts/scrape_danbooru.py --tag "takehiko_inoue" --name "inoue_style" --limit 40
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


def fetch_danbooru_posts(tag: str, limit: int = 40, rating: str = "g,s"):
    """按分数排序获取 Danbooru 高赞原画帖子"""
    posts = []
    page = 1
    # 使用 order:score 优先抓取最能代表该画师/角色高水准的图，排除视频与动图
    search_tags = f"{tag} order:score -animated -video -comic"
    headers = {
        "User-Agent": "Krea2DatasetBuilder/1.0 (Antigravity Local Pipeline)"
    }

    while len(posts) < limit and page <= 5:
        params = urllib.parse.urlencode({
            "tags": search_tags,
            "limit": 100,
            "page": page,
        })
        url = f"https://danbooru.donmai.us/posts.json?{params}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"⚠️ 请求第 {page} 页出错: {e}")
            break

        if not data:
            break

        for post in data:
            file_url = post.get("large_file_url") or post.get("file_url")
            ext = post.get("file_ext", "")
            width = post.get("image_width", 0)
            height = post.get("image_height", 0)

            if not file_url or ext not in ("jpg", "jpeg", "png", "webp"):
                continue
            # 过滤分辨率过低的图（短边至少 768px）
            if min(width, height) < 768:
                continue
            # 过滤长条漫或极端长宽比（宽高比在 0.45 ~ 2.2 之间最适合 1024 分辨率分桶训练）
            aspect = width / max(height, 1)
            if aspect < 0.45 or aspect > 2.2:
                continue

            posts.append({
                "id": post["id"],
                "file_url": file_url,
                "ext": ext,
                "width": width,
                "height": height,
                "score": post.get("score", 0),
                "tag_string_general": post.get("tag_string_general", ""),
                "tag_string_character": post.get("tag_string_character", ""),
                "tag_string_copyright": post.get("tag_string_copyright", ""),
                "tag_string_artist": post.get("tag_string_artist", ""),
            })
            if len(posts) >= limit:
                break

        page += 1
        time.sleep(0.5)

    return posts


def download_dataset(tag: str, dataset_name: str, limit: int, trigger_word: str):
    out_dir = Path("datasets") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "_meta"
    meta_dir.mkdir(exist_ok=True)

    print(f"🔍 正在从 Danbooru 检索标签: [{tag}] (目标精选数量: {limit} 张)...")
    posts = fetch_danbooru_posts(tag, limit=limit)
    print(f"✅ 筛选出 {len(posts)} 张符合 Krea 2 1024px 训练标准的高赞原画，开始下载...")

    headers = {"User-Agent": "Krea2DatasetBuilder/1.0"}
    downloaded = 0

    for idx, p in enumerate(posts, 1):
        img_name = f"{idx:03d}_{p['id']}.{p['ext']}"
        txt_name = f"{idx:03d}_{p['id']}.txt"
        img_path = out_dir / img_name
        txt_path = out_dir / txt_name
        meta_path = meta_dir / f"{idx:03d}_{p['id']}.json"

        if not img_path.exists():
            try:
                req = urllib.request.Request(p["file_url"], headers=headers)
                with urllib.request.urlopen(req, timeout=30) as r:
                    img_path.write_bytes(r.read())
            except Exception as e:
                print(f"  ❌ 下载 {img_name} 失败: {e}")
                continue

        # 保存原始元数据供 Agent 多模态视觉打标参考
        meta_path.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")

        # 生成初始基础打标文件（后续由 Antigravity Agent 多模态看图升级为 Qwen3-VL 自然语言描述）
        if not txt_path.exists():
            char_tags = p["tag_string_character"].replace(" ", ", ").replace("_", " ")
            gen_tags = p["tag_string_general"].replace(" ", ", ").replace("_", " ")
            base_caption = f"{trigger_word}, {char_tags}, {gen_tags}".strip(", ")
            txt_path.write_text(base_caption, encoding="utf-8")

        downloaded += 1
        print(f"  [{downloaded}/{len(posts)}] 已保存: {img_name} ({p['width']}x{p['height']}, score:{p['score']})")

    print(f"\n🎉 数据集 [{dataset_name}] 下载完成！目录: {out_dir.resolve()}")
    print("💡 下一步：让 Antigravity Agent 读取这些图片，将其升级为 Krea 2 (Qwen3-VL) 专属自然语言打标并打包成 ZIP！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danbooru 高清画师/角色训练集抓取工具")
    parser.add_argument("--tag", required=True, help="Danbooru 标签，例如 takehiko_inoue 或 frieren")
    parser.add_argument("--name", required=True, help="保存的数据集名称，例如 inoue_manga")
    parser.add_argument("--trigger", default="by_manga_master", help="LoRA 触发词")
    parser.add_argument("--limit", type=int, default=35, help="精选下载图片数量（推荐 30~50 张）")
    args = parser.parse_args()

    download_dataset(args.tag, args.name, args.limit, args.trigger)
