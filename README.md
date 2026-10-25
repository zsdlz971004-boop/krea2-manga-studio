# 🎨 生图（Krea 2 纯黑白手绘线稿与多角色六维解耦 LoRA 工作站 · 文件夹 `2/生图` 总册）

> **⚠️ 致接手本目录的 AI Agent（最高优先级准则）**：
> 1. **全自动代操作与实证汇报铁律**：用户习惯通过手机端对话，**严禁让用户自己去点 Colab 网页按钮或手动上传下载**！一切生图、LoRA 训练、显存常驻与断电操作，均由你直接在本机通过 [colab_instant_gen.py](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/scripts/colab_instant_gen.py) 与 [train_su_qingge_lora.py](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/scripts/train_su_qingge_lora.py)（基于 Chrome CDP 端口 `9223`）全自动完成；**向用户汇报当前 GPU 型号前，必须以 `nvidia-smi` 或连接按钮实际回显为准（如 `H100` 池满回退 `G4` 96GB 时必须如实汇报 `G4`）**；在启动任何 LoRA 重训前，必须先将新增/修改的训练集图片展示给用户并获得批准。
>    - **后台任务动态定时跟进铁律**：每次启动生图或训练任务进入后台时，**必须立即配合 `schedule` 工具（设 `TimerCondition="any"`）主动循环跟进并向用户汇报进度，严禁启动后静默干等**！根据当前任务类型动态调整单次跟进间隔：**① 显存常驻热启动出图（Hot-Hit）：20～30 秒**；**② 首次冷启动挂载授权与拉取底模出图（Cold-Start）：40～45 秒**（跟进时同步检查并自动点击可能延迟弹出的 Google Drive OAuth 授权窗）；**③ 角色 LoRA 1000 步训练（Training）：60～90 秒**。
> 2. **现阶段唯一最终画风方案（V4 纯黑白 · 纤巧手绘丸笔线稿与细微笔触方案 `V4-HandDrawn-Lineart` · 暂不考虑上色图）**：
>    - **底层架构**：**推理采用 `Krea-2-Turbo`（`krea2_fp8.safetensors`），角色 LoRA 训练采用非蒸馏完整底模 `Krea-2-Raw`（`krea2_raw_bf16.safetensors`，1000 steps）**；
>    - **画风核心突破（找回真正的手绘线稿感 + 必留 `Imigimuru` 前缀）**：
>      - 经同种子 A/B 实证：60 词丸笔技法描述决定 85% 的手绘黑白钢笔质感，而开头的 **`in the distinctive art style of Imigimuru (illustrator of Makeine: Too Many Losing Heroines and Lycoris Recoil)`** 决定剩余 15% 的**浅发色 100% 纯白留白不涂灰、败犬标志性圆润脸型与发饰锚定稳定性**，二者缺一不可；
>      - 彻底弃用 `cel-shaded`，改用**「纤巧手绘丸笔与铅笔速写墨线（`slender organic fine-nib Maru-pen and pencil-sketch ink lines on pure white paper`）+ 极细手绘折痕短排线（`ultra-fine light sketchy pen strokes along fabric creases and edges`）+ 75% 纯白不涂灰纸面留白（`mostly unshaded pure white surfaces`）」**。
> 3. **算力保护与目录句柄保护铁律**：连续生图期间保持模型常驻显存（`--gpu ""` 复用当前已连接 GPU，第 2 批起 0 秒加载）；一旦阶段测试完成或用户表示收尾，立即备份新训权重至 Google Drive、将 `outputs/` 下已完成角色的过程散图归档移入 `archive/outputs_history/<角色名>/` 子目录（保持 `outputs/` 第一层平铺文件数不超过 50 个，防止 macOS `kqueue` 文件夹监控耗尽 `10240` 句柄上限），并运行 `python3 /Users/zhangshuai/2/生图/scripts/colab_instant_gen.py --shutdown-only` 退租显卡（0 额度消耗）。

---

## 🚀 一、 一键全自动生图与断电命令

### 1. 生成【黑白手绘线稿轻小说插画】（支持纯净底模或挂载角色 LoRA）
```bash
# 调用苏清歌 v2 角色 LoRA（推荐权重 0.80，15步，CFG 1.65）
python3 /Users/zhangshuai/2/生图/scripts/colab_instant_gen.py \
  --mode bw \
  --lora "su_qingge_lora_v2:0.80" \
  --steps 15 --cfg 1.65 \
  --prefix sqg_scene \
  --artifact-dir "<当前会话Artifact目录>" \
  --prompts-json '<JSON数组或JSON文件路径>'
```

### 2. 阶段完成或用户说“收尾 / 关机”时立即释放 GPU（0 额度消耗）
```bash
python3 /Users/zhangshuai/2/生图/scripts/colab_instant_gen.py --shutdown-only
```

---

## ✨ 二、 V4 纯黑白 · 纤巧手绘丸笔线稿公式

### 📌 1. 必加固定画风前缀（V4 Hand-Drawn Lineart Master Prefix）
```text
in the distinctive art style of Imigimuru (illustrator of Makeine: Too Many Losing Heroines and Lycoris Recoil), delicate hand-drawn light novel lineart illustration, slender organic fine-nib Maru-pen and pencil-sketch ink lines on pure white paper, grayscale, no color, subtle natural hand-drawn line weight with loose wispy hair strands, ultra-fine light sketchy pen strokes along fabric creases and edges, delicate hand-drawn diagonal blush lines on softly rounded lower cheeks, expressive low-set almond eyes with thick upper lashes, mostly unshaded pure white surfaces with soft flat light-grey manga screentone accents only on secondary cloth, no digital gradient, no 3d shading.
```

### 📌 2. 标准负向提示词（已内置于 `colab_instant_gen.py` 的 `default_neg_bw`）
```text
color, polychrome, 3d render, CGI, photorealistic metallic reflection, glossy 3d shading, smooth airbrush gradient, soft grey gradient on armor, heavy dark chiaroscuro, dense western cross-hatching, thick heavy marker outlines, blotchy ink, bad hands, mutated hands, extra fingers, missing fingers, fused fingers, six fingers, deformed hands, speech bubble, text, watermark, signature
```

---

## 🧬 三、 多角色通用「36 张六维解耦 LoRA 训练集与推理调用黄金规范」

为彻底杜绝角色 LoRA 常见的「神态锁死闭嘴、视线死盯镜头、机位平视站桩、纯白背景粘连、多姿拼图重影、浅色头发泛灰」六大顽疾，后续为《末世：避难所太挤，女神请自重》全部角色（**二老婆·贺令仪、三老婆·姜楠、四老婆·柳依依、五老婆·马莉莉、神秘商人·小八、男主·张少岚**）制作专属 LoRA 时，必须 100% 遵循以下**六维解耦标准**：

1. **维度一：神态解耦与张嘴比例硬指标（35%~40% 张嘴率）**：
   - **严禁在角色核心外貌词块（`CHAR_CORE`）中写入任何固定性格/表情词**（如严禁在 `CHAR_CORE` 中写 `"slightly proud tsundere"`，外貌词块只锁定脸型、发型、发饰等纯物理特征，表情词一律后置到单图描述中）；
   - **36 张初始训练集中，「张嘴」图片（张嘴大笑、张嘴训斥、张嘴说话、张嘴呼喊、张嘴惊呼）必须占到 12~14 张（35%~40%）**，闭嘴图片控制在 60% 以内，从源头打破闭嘴惯性；
   - 在 `Krea-2` 中生成张嘴图片时，优先使用具象动作词 **`open talking mouth` / `open scolding mouth` / `wide-open calling mouth` / `wide open mouth showing upper teeth and tongue`** + **`deeply furrowed downward V-shaped angry eyebrows`**。
2. **维度二：浅发色纯白留白锁死（防穿白衣时头发变灰）**：
   - 凡是浅发色/白发/银发/金发角色（在黑白线稿中表现为白底留白发丝），**严禁使用模糊的 `"light-colored hair"`**（当角色身穿纯白大衣/白毛衣时，底模会为了区分“纯白衣服”与“浅色头发”而给头发铺一层浅灰网点）；
   - **必须统一写为明确的 `"unshaded pure white hair"`**，确保无论穿何种纯白服饰，头发永远 100% 纯白留白。
3. **维度三：视线方向与身体朝向解耦**：
   - 36 张图中必须包含至少 **8~10 张非盯镜头图片**（看向画面左侧 `looking toward the left side of the frame`、看向右侧 `looking toward the right side of the frame`、别过头不看人 `turning her head to look away`、90 度纯正侧面 `full-body 90-degree side profile shot`、背影回眸 `from-behind back view turning head over shoulder`）。
4. **维度四：摄影机位与大肢体动态解耦**：
   - 必须包含高角度俯拍（`high-angle shot looking down at her from above`）、低角度仰拍（`low-angle shot looking up at her`）、双手撑桌前倾（`leaning forward with both hands pressed on a wooden table`）、桌边坐姿（`sitting on a wooden chair at a dining table`）、蹲姿与行走回头动态。
5. **维度五：纯白背景与实景建筑背景打标解耦**：
   - 36 张图中包含 **30~32 张纯白底单图/设定图**（打标时**必须显式写明 `on a stark pure white background`**，把“纯白背景”绑定在该短语上而非绑定在角色触发词上）+ **4~6 张黑白手绘透视实景背景图**（如避难所物资仓库、风雪走廊、厨房餐厅、作战指挥室，打标写明具体背景）；
   - **实景背景防 3D 景深涂抹铁律**：室外空旷废墟街景词（如 `snowy ruined street`）极易诱发 `Krea-2` 底模产生 3D 景深虚化灰块与粗重动漫轮廓线，实景背景应优先采用带明确几何透视的室内/半室内建筑词（如 `snowy apocalypse concrete corridor with frosted window frames drawn in clean black-and-white manga lineart`），并在调用 `colab_instant_gen.py --prompts-json` 时确保 JSON 数组内每一条提示词都显式以完整 `V4_PREFIX` 开头（脚本不会对 `--prompts-json` 自动补前缀）。
6. **维度六：三视图占比控制、入库前地毯式复检与推理「表情前置 + 0.80 权重」黄金法则**：
   - 每套服装最多只保留 **1 张最清晰的 `turnaround` 三视图**（4 套服装共 4 张），**严禁塞入冗余的 `multiview` 多姿拼图或 2x2 表情包网格**，防止单人推理出现分身重影与手部像素退化；
   - **入库前 36 图地毯式视觉复检**：展示给用户前必须逐张放大复检：① 是否 100% 带有 `Imigimuru` V4 手绘丸笔质感与杏眼排线（剔除 `O_O` 简笔画圆圈眼）；② 手部五指数量与拇指/小指朝向是否正常（必要时优先采用双手插兜、抱臂、背手等零畸变姿态）；③ 标志性发饰是否为单枚且位置形态准确（剔除双发夹或戳出头顶轮廓的伪影）；④ 非三视图的 Q 版/单图严禁出现双人同框；
   - **推理调用黄金法则**：调用角色 LoRA 时，推荐权重设为 **`0.78 ~ 0.85`**（如 `--lora "su_qingge_lora_v2:0.80"` 或 `--lora "he_lingyi_lora_v1:0.85"`），并将**当前镜头的神态、张嘴状态与动作紧挨着放在 `<char_trigger>, 1girl,` 之后的第一顺位**（置于外貌与服装长词块之前）。

---

## 👑 四、 已完训角色 LoRA 档案库

### 1. 正宫“大老婆”· 苏清歌（`su_qingge_lora_v2` · 36 张六维解耦版）
- **云端权重路径**：`/content/drive/MyDrive/AI_Manga_Studio/loras/su_qingge_lora_v2/su_qingge_lora_v2.safetensors`（同时已映射别名 `su_qingge_lora`）
- **本地数据集与压缩包**：[datasets/su_qingge_lora_v2/](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/su_qingge_lora_v2) / [datasets/su_qingge_lora_v2.zip](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/su_qingge_lora_v2.zip)（已同步推送至 GitHub 仓库 `datasets/su_qingge_lora_v2.zip`）
- **服装与外貌预设配置**：[datasets/su_qingge_lora_v2/outfit_presets.json](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/su_qingge_lora_v2/outfit_presets.json)
  - **角色触发词与外貌锁（`CHAR_CORE`）**：
    `su_qingge, 1girl, college campus idol girl, medium-long softly wavy unshaded pure white hair reaching her chest with fluffy curled ends, thick full front bangs covering her forehead down to her eyebrows, small dark ribbon bow hairpin on upper left with two sprouted ahoge wisps, single rectangular barrette hair clip on right side above ear`
  - **四套专属服装触发词**：
    1. **`sqg_parka`（白色毛领派克大衣 + 高领麻花毛衣 + 灰百褶裙 + 黑连裤袜 + 雪地靴）**
    2. **`sqg_puffer`（白色蓬松羽绒服 + 流苏粗针织围巾 + 灰百褶裙 + 黑连裤袜 + 系带雪地靴）**
    3. **`sqg_jacket`（中灰羊羔绒翻领飞行员夹克 + 纯白竖纹高领毛衣 + 灰百褶裙 + 黑连裤袜）**
    4. **`sqg_apron`（中灰圆领套头毛衣 + 纯白荷叶边蝴蝶结围裙 + 深灰百褶裙 + 黑连裤袜）**
  - **Q版触发词**：**`sqg_chibi`**（2头身超萌Q版）

### 2. 二老婆（学生会长/第一执政官）· 贺令仪（`he_lingyi_lora_v1` · 36 张六维解耦版）
- **云端权重路径**：`/content/drive/MyDrive/AI_Manga_Studio/loras/he_lingyi_lora_v1/he_lingyi_lora_v1.safetensors`（同时已映射别名 `he_lingyi_lora`）
- **本地数据集与压缩包**：[datasets/he_lingyi_lora_v1/](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/he_lingyi_lora_v1) / [datasets/he_lingyi_lora_v1.zip](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/he_lingyi_lora_v1.zip)（已同步推送至 GitHub 仓库 `datasets/he_lingyi_lora_v1.zip`）
- **服装与外貌预设配置**：[datasets/he_lingyi_lora_v1/outfit_presets.json](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/he_lingyi_lora_v1/outfit_presets.json)
  - **角色触发词与外貌锁（`CHAR_CORE`）**：
    `he_lingyi, 1girl, tall regal student council president girl, long sleek straight jet-black hair reaching her waist with glossy white ink highlights and sleek cheek-framing sidelocks, neat straight bangs covering her forehead down to her eyebrows, single clean white rectangular barrette hair clip on left side above ear, sharp confident eyebrows, expressive almond eyes`
  - **四套专属服装触发词**：
    1. **`hly_coat`（纯白敞开式中长款大衣外套 + 中灰修身竖纹高领毛衣 + 浅灰高腰宽皮带铅笔中长裙 + 黑丝连裤袜 + 黑色短靴）**
    2. **`hly_maid`（反差萌经典黑白荷叶边女仆装 + 中灰领结 + 纯白围裙 + 蕾丝女仆发带 + 黑色过膝长筒袜）**
    3. **`hly_parka`（纯白毛领极寒战术派克防寒服 + 左前臂护腕 + 深灰高领毛衣 + 黑色修身战术长裤 + 浅灰系带雪地靴）**
    4. **`hly_trench`（纯白毛领双排扣束腰长款风衣 + 纯白流苏粗针织围巾 + 深灰高领毛衣 + 深色百褶中长裙 + 黑丝连裤袜）**
  - **Q版触发词**：**`hly_chibi`**（2头身超萌Q版）

### 3. 三老婆（刑警副队长/防务指挥官）· 姜楠（`jiang_nan_lora_v1` · 36 张六维解耦版）
- **云端权重路径**：`/content/drive/MyDrive/AI_Manga_Studio/loras/jiang_nan_lora_v1/jiang_nan_lora_v1.safetensors`（同时已映射别名 `jiang_nan_lora`）
- **本地数据集与压缩包**：[datasets/jiang_nan_lora_v1/](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/jiang_nan_lora_v1) / [datasets/jiang_nan_lora_v1.zip](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/jiang_nan_lora_v1.zip)（已同步推送至 GitHub 仓库 `datasets/jiang_nan_lora_v1.zip`）
- **服装与外貌预设配置**：[datasets/jiang_nan_lora_v1/outfit_presets.json](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/jiang_nan_lora_v1/outfit_presets.json)
  - **角色触发词与外貌锁（`CHAR_CORE`）**：
    `jiang_nan, 1girl, tall athletic police detective girl, sleek chin-length layered jet-black short bob hair with glossy white ink highlights and cheek-framing short sidelocks, lightly parted bangs covering her forehead down to her eyebrows, single clean white crossed X-shaped hairpin clip on left side above ear, sharp heroic eyebrows, expressive almond eyes`
  - **四套专属服装触发词**：
    1. **`jn_vest`（「铁面花木兰」纯白多口袋战术背心 + 深灰作训长袖衬衫 + 胸前金属联络哨子 + 黑色绑腿枪套战术长裤 + 作战靴）**
    2. **`jn_sweater`（「反差萌纯情姐姐」纯白一字肩竖罗纹针织毛衣 + 中灰高腰宽皮带裹身短裙 + 纯黑不透肉连裤袜 + 黑色短靴）**
    3. **`jn_sport`（「训练场汗水荷尔蒙」纯白露脐运动短背心 + 挂颈复古耳机 + 中灰慢跑运动长裤 + 运动鞋）**
    4. **`jn_police`（「干练女刑警」纯白卷袖警用衬衫 + 中灰领带 + 黑色腋下双肩战术枪套背带 + 黑色高腰制服长裤 + 作战靴）**
  - **Q版触发词**：**`jn_chibi`**（2头身超萌Q版）

### 4. 四老婆（高挑反差萌宅女电竞少女）· 柳依依（`liu_yiyi_lora_v1` · 36 张六维解耦版）
- **云端权重路径**：`/content/drive/MyDrive/AI_Manga_Studio/loras/liu_yiyi_lora_v1/liu_yiyi_lora_v1.safetensors`（同时已映射别名 `liu_yiyi_lora`）
- **本地数据集与压缩包**：[datasets/liu_yiyi_lora_v1/](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/liu_yiyi_lora_v1) / [datasets/liu_yiyi_lora_v1.zip](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/liu_yiyi_lora_v1.zip)（扁平根目录结构，已同步推送至 GitHub 仓库 `datasets/liu_yiyi_lora_v1.zip`）
- **服装与外貌预设配置**：[datasets/liu_yiyi_lora_v1/outfit_presets.json](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/liu_yiyi_lora_v1/outfit_presets.json)
  - **角色触发词与外貌锁（`CHAR_CORE`）**：
    `liu_yiyi, 1girl, an energetic and cute 19-year-old Chinese girl with a bright expressive face, glossy ink-black hair with soft wispy full bangs and face-framing side locks tied in a high bouncy ponytail`（注：调用柳依依时建议去掉画风前缀中的 `expressive low-set almond eyes`，以充分释放八奈见杏菜式弯月笑眼 `^▽^`、四方露齿笑 `^皿^`、鼓腮帮 `)-(`、十字星星眼 `✦_✦` 与空心圆圈眼 `〇_〇` 等漫画颜艺表现力）
  - **四套专属极寒外勤 & 避难所室内服装触发词（严格契合零下六七十度极寒末世设定，且头戴/挂颈耳机位置完全解耦）**：
    1. **`lyy_parka`（【极寒外勤主锚点】纯白重型大毛领防寒远征派克大衣 + 灰色抓绒拉链高领内胆 + 纯黑加厚防寒长裤 + 重型雪地靴 + 纯白猫耳耳机戴在头上）**
    2. **`lyy_mount`（【极寒外勤登山装】深灰羊羔绒翻领/袖口重型登山大衣 + 厚白针织围巾 + 白色粗棒针麻花毛衣 + 纯黑加厚防寒长裤 + 系带雪地靴 + 纯白猫耳耳机戴在头上）**
    3. **`lyy_hoodie`（【避难所居家电竞装】深灰羊羔绒拉链连帽卫衣外套 + 纯白竖罗纹高领毛衣 + 深色百褶短裙 + 纯黑厚保暖连裤袜 + 纯白猫耳耳机挂在脖子上 `resting around her neck`）**
    4. **`lyy_suspender`（【避难所室内修身装】修身纯白竖罗纹高领针织毛衣 + 中灰高腰百褶背带裙 + 纯黑厚保暖连裤袜 + 室内毛绒短靴 + 纯白猫耳耳机戴在头上）**
  - **三视图与八奈见漫画分镜触发词**：
    - **`lyy_turnaround`**：正+侧+背全身三视图设定集（4 张）
    - **`lyy_comic`**：八奈见杏菜风「主立绘 + 旁侧 Q 版颜艺反应头 / 漫画分镜」（4 张；常规单人立绘不加 `lyy_comic` 时 100% 保持纯净 `1girl, solo` 零串扰）

---

## 📂 五、 本文件夹（`/Users/zhangshuai/2/生图`）目录结构导航

- [README.md](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/README.md) / [AGENTS.md](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/AGENTS.md)：本无缝接管总册。
- [scripts/colab_instant_gen.py](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/scripts/colab_instant_gen.py)：**核心一键生图与断电控制脚本**（默认锁定 `H100`，已升级原生 `setRuntime` API 与 `start_new_session=True` 后台守护）。
- [scripts/resident_engine.py](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/scripts/resident_engine.py)：**云端显存常驻秒出 + 闲置自动断电引擎**（支持 `lora_name:weight` 动态权重解析、多角色触发词自动映射与 `sys.last_traceback = None` 显存彻底释放）。
- [scripts/train_su_qingge_lora.py](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/scripts/train_su_qingge_lora.py) / [scripts/train_he_lingyi_lora.py](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/scripts/train_he_lingyi_lora.py) / [scripts/train_jiang_nan_lora.py](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/scripts/train_jiang_nan_lora.py) / [scripts/train_liu_yiyi_lora.py](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/scripts/train_liu_yiyi_lora.py)：**角色 LoRA（`Krea-2-Raw` 1000 步）NVMe 直训与 8 图多维解耦自动验收脚本**（内置 `gc.get_objects()` 父进程 CUDA 显存彻底清零机制，防止 40GB A100 子进程 OOM）。
- [datasets/su_qingge_lora_v2/](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/su_qingge_lora_v2)：**苏清歌 `su_qingge_lora_v2` 36 张六维解耦训练集与 `outfit_presets.json`**。
- [datasets/he_lingyi_lora_v1/](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/he_lingyi_lora_v1)：**贺令仪 `he_lingyi_lora_v1` 36 张六维解耦训练集与 `outfit_presets.json`**。
- [datasets/jiang_nan_lora_v1/](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/jiang_nan_lora_v1)：**姜楠 `jiang_nan_lora_v1` 36 张六维解耦训练集与 `outfit_presets.json`**。
- [datasets/liu_yiyi_lora_v1/](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/datasets/liu_yiyi_lora_v1)：**柳依依 `liu_yiyi_lora_v1` 36 张六维解耦训练集（32 张全角度单人立绘/三视图 + 4 张精选八奈见风颜艺/分镜）与 `outfit_presets.json`**。
- [archive/krea2_handdrawn_lineart_report.md](file:///Users/zhangshuai/2/%E7%94%9F%E5%9B%BE/archive/krea2_handdrawn_lineart_report.md)：V4 纯黑白「真正的手绘线稿感与纤巧笔触」迭代总报告。
