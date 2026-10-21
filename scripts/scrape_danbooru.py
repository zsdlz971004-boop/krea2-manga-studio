#!/usr/bin/env python3
"""
从 Danbooru 自动抓取指定画师/角色的高赞高清代表作，并过滤不适合训练 Krea 2 LoRA 的废图（长条漫、低清图、动图）。
注意：Danbooru 匿名 API 限制每次查询最多 2 个 tag，因此我们在拉取 JSON 后在本地 Python 中完成精准过滤与按分数排序。
用法示例:
  python3 scripts/scrape_danbooru.py --tag "imigimuru make_heroine_ga_oo_sugiru!" --name "imigimuru_makeine" --trigger "by_imigimuru" --limit 36
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def fetch_danbooru_posts(tag: str, limit: int = 36):
    """获取 Danbooru 原画帖子并在本地按高分排序与清洗"""
    raw_posts = []
    page = 1
    # 如果只传了 1 个 tag，可以加上 order:score；如果已有 2 个 tag，则直接查并在本地按 score 排序
    tag_parts = tag.strip().split()
    search_tags = f"{tag_parts[0]} order:score" if len(tag_parts) == 1 else " ".join(tag_parts[:2])

    headers = {
        "User-Agent": "Krea2DatasetBuilder/1.0 (Antigravity Local Pipeline)"
    }

    while page <= 5:
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
            # Danbooru API 中 file_url 才是 4K 原图，large_file_url 是 850px 预览图
            file_url = post.get("file_url") or post.get("large_file_url")
            ext = post.get("file_ext", "")
            width = post.get("image_width", 0)
            height = post.get("image_height", 0)
            gen_tags = set(post.get("tag_string_general", "").split())

            if not file_url or ext not in ("jpg", "jpeg", "png", "webp"):
                continue
            # 排除条漫、四格漫画、动图与视频
            if gen_tags & {"comic", "4koma", "animated", "video", "multiple_views"}:
                continue
            # 过滤分辨率过低的图（短边至少 800px）
            if min(width, height) < 800:
                continue
            # 过滤极端长宽比（宽高比在 0.5 ~ 2.0 之间最适合 1024 分辨率训练）
            aspect = width / max(height, 1)
            if aspect < 0.5 or aspect > 2.0:
                continue

            raw_posts.append({
                "id": post["id"],
                "file_url": file_url,
                "ext": ext,
                "width": width,
                "height": height,
                "score": post.get("score", 0),
                "rating": post.get("rating", "g"),
                "tag_string_general": post.get("tag_string_general", ""),
                "tag_string_character": post.get("tag_string_character", ""),
                "tag_string_copyright": post.get("tag_string_copyright", ""),
                "tag_string_artist": post.get("tag_string_artist", ""),
                "tag_string_meta": post.get("tag_string_meta", ""),
            })

        if len(data) < 100:
            break
        page += 1
        time.sleep(0.4)

    # 按分数从高到低排序，取前 limit 张最高质量代表作
    raw_posts.sort(key=lambda x: x["score"], reverse=True)
    return raw_posts[:limit]


def download_dataset(tag: str, dataset_name: str, limit: int, trigger_word: str):
    out_dir = Path("datasets") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "_meta"
    meta_dir.mkdir(exist_ok=True)

    print(f"🔍 正在从 Danbooru 检索标签: [{tag}] (目标精选数量: {limit} 张)...")
    posts = fetch_danbooru_posts(tag, limit=limit)
    print(f"✅ 筛选出 {len(posts)} 张符合 Krea 2 1024px 训练标准的高分原画，开始下载...")

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

        meta_path.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")

        if not txt_path.exists():
            char_tags = p["tag_string_character"].replace(" ", ", ").replace("_", " ")
            gen_tags = p["tag_string_general"].replace(" ", ", ").replace("_", " ")
            base_caption = f"{trigger_word}, {char_tags}, {gen_tags}".strip(", ")
            txt_path.write_text(base_caption, encoding="utf-8")

        downloaded += 1
        print(f"  [{downloaded}/{len(posts)}] 已保存: {img_name} ({p['width']}x{p['height']}, score:{p['score']})")

    print(f"\n🎉 数据集 [{dataset_name}] 下载完成！目录: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danbooru 高清画师/角色训练集抓取工具")
    parser.add_argument("--tag", required=True, help="Danbooru 标签")
    parser.add_argument("--name", required=True, help="保存的数据集名称")
    parser.add_argument("--trigger", default="by_imigimuru", help="LoRA 触发词")
    parser.add_argument("--limit", type=int, default=36, help="精选下载图片数量")
    args = parser.parse_args()

    download_dataset(args.tag, args.name, args.limit, args.trigger)
