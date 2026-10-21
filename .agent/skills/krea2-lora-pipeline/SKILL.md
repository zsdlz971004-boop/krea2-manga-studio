---
name: krea2-lora-pipeline
description: 全自动从 Danbooru 扒取日本漫画家/原画师/角色高清图片、使用多模态能力进行 Krea 2 (Qwen3-VL) 自然语言精准打标、并打包生成 Google Colab A100/L4 一键训练包的完整工作流。
---

# Krea 2 画风与角色 LoRA 全自动一条龙 Skill

当用户希望训练某个日本漫画家手绘画风、动画工作室风格或特定动漫角色的 **Krea 2 (`krea/Krea-2-Raw`) LoRA** 时，按以下 4 步自动执行：

## Step 1: 确定 Danbooru 标签并自动扒取高清原画
1. 根据用户提供的画师名或角色名，确认其在 Danbooru 上的标准 Tag（例如井上雄彦为 `takehiko_inoue`，藤本树为 `fujimoto_tatsuki`，米山舞为 `yoneyama_mai`，大友克洋为 `otomo_katsuhiro`）。
2. 运行抓取脚本（默认精选 30~40 张最高分、长宽比适中、无动图的代表作）：
   ```bash
   python3 scripts/scrape_danbooru.py --tag "<danbooru_tag>" --name "<dataset_name>" --trigger "<trigger_word>" --limit 35
   ```

## Step 2: 针对 Krea 2 (`Qwen3-VL`) 的自然语言多模态打标（核心机密）
为什么贴吧老哥说“写好 Skill，用自带额度自动打标，还原度逆天”？
因为 **Krea 2 使用的是 `Qwen3-VL-4B` 自然语言视觉编码器**，并且遵循 **“打标写出来的特征会被剥离，没写出来的共性特征会全部凝聚进 `trigger_word`（触发词）”** 的铁律！
- **训练【画风/手绘风格 LoRA】时的打标铁律**：
  1. 使用 `read_file` 查看 `datasets/<dataset_name>/` 下的图片（或结合 `_meta/*.json` 中的内容标签）。
  2. **绝对不要**在 `.txt` 打标文件里写画风词（如不要写 `manga style, ink lines, cross-hatching, monochrome`），这样模型就会把“黑白网点、蘸水笔线条、独特的光影笔触”100% 绑定到开头的 `trigger_word` 上！
  3. 只用自然语言客观描述画面里的**主体人物、外貌服饰、动作神态、镜头视角与背景物体**。格式示例：
     > `by_inoue_style, A close-up portrait of a young samurai with messy tied-back black hair and a scar on his cheek, looking sideways with fierce determination. He wears a worn traditional kimono, standing against a bamboo forest background.`
- **训练【特定角色 LoRA】时的打标铁律**：
  1. **绝对不要**描述该角色的固定标志性外貌（如发色、瞳色、标志性发型），让这些特征自动绑定到角色 `trigger_word`。
  2. 重点描述每一张图里**变化的服装、动作姿势、镜头远近、光影与画风媒介**。

## Step 3: 打包成 Colab 专用 ZIP
完成所有 `.txt` 打标更新后，运行打包脚本：
```bash
python3 scripts/pack_dataset.py --name "<dataset_name>"
```

## Step 4: 引导用户在 Google Colab 一键开炼
提示用户将生成的 `datasets/<dataset_name>.zip` 上传至 Google Drive 的 `AI_Manga_Studio/datasets/` 目录，并在 Colab 中打开项目根目录下的 [Krea2_Manga_Studio_Colab.ipynb](file:///Users/zhangshuai/Documents/antigravity/charming-fermi/Krea2_Manga_Studio_Colab.ipynb) 点击 `▶` 即可。
