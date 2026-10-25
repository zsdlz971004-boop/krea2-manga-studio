#!/usr/bin/env python3
import base64, json, os, sys, time, urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from cdp_helper import cdp_eval
from colab_instant_gen import ensure_chrome_cdp, auto_approve_drive_oauth

ARTIFACT_DIR = "/Users/zhangshuai/.gemini/antigravity/brain/6a4b563e-e92c-48ab-b3f9-494d9458eba3"
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs")

V4_PREFIX = "in the distinctive art style of Imigimuru (illustrator of Makeine: Too Many Losing Heroines and Lycoris Recoil), delicate hand-drawn light novel lineart illustration, slender organic fine-nib Maru-pen and pencil-sketch ink lines on pure white paper, grayscale, no color, subtle natural hand-drawn line weight with loose wispy hair strands, ultra-fine light sketchy pen strokes along fabric creases and edges, delicate hand-drawn diagonal blush lines on softly rounded lower cheeks, mostly unshaded pure white surfaces with soft flat light-grey manga screentone accents only on secondary cloth, no digital gradient, no 3d shading"
CHAR_REST = "an energetic and cute 19-year-old Chinese girl with a bright expressive face, glossy ink-black hair with soft wispy full bangs and face-framing side locks tied in a high bouncy ponytail"

PARKA_BLOCK = "lyy_parka, wearing cute unshaded pure white cat-ear headphones over her head, wearing a heavy-duty thick white fur-hooded winter expedition parka over a grey fleece zip-up turtleneck, thick black thermal winter pants, and heavy snow boots"
MOUNT_BLOCK = "lyy_mount, wearing cute unshaded pure white cat-ear headphones over her head, wearing a heavy grey shearling-lined winter mountaineering coat with thick white sherpa fleece collar and cuffs and a thick white knit scarf over a chunky white cable-knit sweater, thick black thermal winter pants, and heavy lace-up snow boots"
HOODIE_BLOCK = "lyy_hoodie, wearing cute unshaded pure white cat-ear headphones resting around her neck, wearing an oversized charcoal-grey plush fleece zip-up hoodie jacket over a thick white ribbed turtleneck sweater, dark pleated mini skirt, and thick black thermal leggings"
SUSPENDER_BLOCK = "lyy_suspender, wearing cute unshaded pure white cat-ear headphones over her head, wearing a fitted white ribbed turtleneck sweater under a grey high-waisted pleated suspender skirt, thick black thermal tights, and indoor plush boots"

TEST_PROMPTS = [
    # 0: lyy_parka - Full-body standing in snowy apocalypse street, cheerful open-mouth smile
    f"{V4_PREFIX}, liu_yiyi, 1girl, solo, bright cheerful open-mouth smile, {CHAR_REST}, {PARKA_BLOCK}, full-body standing front portrait with hands tucked casually into coat pockets in a snowy apocalypse street background drawn in clean black-and-white manga lineart.",
    # 1: lyy_parka - Yanami-style happy closed crescent laughing eyes (^▽^) holding a steaming hot meat bun
    f"{V4_PREFIX}, liu_yiyi, 1girl, solo, beaming with happy closed crescent laughing eyes and a wide cheerful open-mouth laugh, {CHAR_REST}, {PARKA_BLOCK}, upper-body close-up portrait holding a steaming hot meat bun with both hands, on a stark pure white background.",
    # 2: lyy_mount - Full-body standing in heavy grey shearling mountaineering coat, energetic wave
    f"{V4_PREFIX}, liu_yiyi, 1girl, solo, beaming with happy closed crescent laughing eyes and a wide open-mouth laugh, {CHAR_REST}, {MOUNT_BLOCK}, full-body standing portrait raising one hand in an energetic wave, on a stark pure white background.",
    # 3: lyy_mount - High-angle sulky puffed-cheeks pout )-(
    f"{V4_PREFIX}, liu_yiyi, 1girl, solo, cute sulky puffed-cheeks pout and furrowed brows looking up at the camera, {CHAR_REST}, {MOUNT_BLOCK}, upper-body high-angle close-up portrait, on a stark pure white background.",
    # 4: lyy_hoodie - Holding wireless gamepad controller with happy closed crescent laughing eyes
    f"{V4_PREFIX}, liu_yiyi, 1girl, solo, beaming with happy closed crescent laughing eyes and a wide open-mouth laugh, {CHAR_REST}, {HOODIE_BLOCK}, upper-body close-up portrait holding a wireless gamepad controller with both hands in a cozy shelter living room background drawn in clean black-and-white manga lineart.",
    # 5: lyy_hoodie - Pointing accusingly with cool/sulky expression
    f"{V4_PREFIX}, liu_yiyi, 1girl, solo, cool accusing expression with closed mouth, {CHAR_REST}, {HOODIE_BLOCK}, upper-body portrait pointing one index finger forward while clenching her other fist, on a stark pure white background.",
    # 6: lyy_suspender - Full-body standing in white turtleneck & grey pleated suspender skirt, cheerful laugh
    f"{V4_PREFIX}, liu_yiyi, 1girl, solo, beaming with happy closed crescent laughing eyes and a wide cheerful open-mouth smile, {CHAR_REST}, {SUSPENDER_BLOCK}, full-body standing portrait from head to indoor plush boots with hands in skirt pockets, on a stark pure white background.",
    # 7: lyy_comic - Yanami Anna-style multi-expression character reaction sheet
    f"{V4_PREFIX}, liu_yiyi, 1girl, {CHAR_REST}, {PARKA_BLOCK}, lyy_comic, manga character expression sheet, main upper-body portrait on the left pointing at herself with a smug triumphant clenched-teeth grin, accompanied by three small chibi head reaction icons on the right showing blank white circle eyes shock, puffed-cheeks sulky pout, and wide-eyed surprise, on a stark pure white background."
]

def main():
    ws = ensure_chrome_cdp()
    marker = f"TRAIN_LYY_V1_{int(time.time())}"
    resident_src_path = os.path.join(SCRIPT_DIR, "resident_engine.py")
    with open(resident_src_path, "r", encoding="utf-8") as f:
        resident_src = f.read()

    cell_code = f"""import os, sys, time, shutil, zipfile, subprocess, yaml, gc, importlib, builtins
print("MARKER:" + "{marker}", flush=True)
if not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
print("DRIVE_MOUNTED:" + "{marker}", flush=True)

# Free any resident VRAM model and kill any leftover run.py subprocess before starting!
os.system("pkill -9 -f '/content/ai-toolkit/run.py' >/dev/null 2>&1 || true")
time.sleep(2)
import torch
for attr in ["last_traceback", "last_value", "last_type"]:
    if hasattr(sys, attr):
        setattr(sys, attr, None)
builtins.KREA_PIPE = None
builtins.KREA_LOADED_LORAS = None
builtins.GLOBAL_PROCESS = None
builtins.GLOBAL_JOB_NAME = None
gc.collect()
for _obj in gc.get_objects():
    try:
        if torch.is_tensor(_obj) and _obj.is_cuda:
            _obj.data = torch.empty(0, device="cpu")
    except Exception:
        pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

WORK_ROOT = "/content/AI_Manga_Studio_Local"
for sub in ["datasets", "loras", "checkpoints", "output", "configs"]:
    os.makedirs(f"{{WORK_ROOT}}/{{sub}}", exist_ok=True)

_re_path = f"{{WORK_ROOT}}/resident_engine.py"
with open(_re_path, "w", encoding="utf-8") as _f:
    _f.write({json.dumps(resident_src)})
if WORK_ROOT not in sys.path:
    sys.path.insert(0, WORK_ROOT)

import resident_engine
importlib.reload(resident_engine)
resident_engine.ensure_env()

gpu_info = subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")
print(f"🔍 GPU: {{gpu_info}}", flush=True)

LORA_NAME = "liu_yiyi_lora_v1"
local_loras_root = f"{{WORK_ROOT}}/loras"
local_lora_dir = f"{{local_loras_root}}/{{LORA_NAME}}"
os.makedirs(local_lora_dir, exist_ok=True)
final_safetensors = f"{{local_lora_dir}}/{{LORA_NAME}}.safetensors"

if os.path.exists(final_safetensors):
    print(f"✅ Found completed 1000-step Krea-2-Raw LoRA weights: {{final_safetensors}} ({{os.path.getsize(final_safetensors)}} bytes)! Skipping re-training.", flush=True)
else:
    # 1. Pull latest liu_yiyi_lora_v1 dataset from GitHub
    extract_dir = "/content/dataset/liu_yiyi_lora_v1"
    os.makedirs(extract_dir, exist_ok=True)
    if not os.path.exists("/content/krea2-manga-studio"):
        os.system("git clone -q https://github.com/zsdlz971004-boop/krea2-manga-studio.git /content/krea2-manga-studio")
    else:
        os.system("cd /content/krea2-manga-studio && git fetch origin main && git reset --hard origin/main")
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)
    repo_zip = "/content/krea2-manga-studio/datasets/liu_yiyi_lora_v1.zip"
    with zipfile.ZipFile(repo_zip, "r") as zf:
        zf.extractall(extract_dir)
    for fn in list(os.listdir(extract_dir)):
        if not (fn.endswith(".jpg") or fn.endswith(".txt")):
            os.remove(os.path.join(extract_dir, fn))
    print(f"✅ Dataset ready ({{len([f for f in os.listdir(extract_dir) if f.endswith('.jpg')])}} images in {{extract_dir}})!", flush=True)

    # 2. Locate Krea-2-Raw (non-distilled base model specifically for LoRA training) on local NVMe
    ckpt_path = "/content/krea2_raw_bf16.safetensors"
    if not os.path.exists(ckpt_path):
        print("⬇️ Downloading Krea-2-Raw full base model (krea2_raw_bf16.safetensors) via 10Gbps aria2c...", flush=True)
        os.system("apt-get -y install -qq aria2 >/dev/null 2>&1")
        os.system('aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_bf16.safetensors" -d "/content" -o "krea2_raw_bf16.safetensors"')

    # 3. Train liu_yiyi_lora_v1 directly on local NVMe
    shutil.rmtree(local_lora_dir, ignore_errors=True)
    os.makedirs(local_lora_dir, exist_ok=True)

    is_high_vram = any(k in gpu_info for k in ["H100", "A100", "80", "40960", "97887", "Blackwell"])
    batch_size = 2 if is_high_vram else 1
    steps = 1000

    train_cfg = {{
        "job": "extension",
        "config": {{
            "name": LORA_NAME,
            "process": [{{
                "type": "sd_trainer",
                "training_folder": local_loras_root,
                "device": "cuda:0",
                "trigger_word": "liu_yiyi",
                "network": {{"type": "lora", "linear": 32, "linear_alpha": 32}},
                "save": {{"dtype": "bf16", "save_every": 250, "max_step_saves_to_keep": 2}},
                "datasets": [{{
                    "folder_path": extract_dir,
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "cache_text_embeddings": True,
                    "resolution": [1024]
                }}],
                "train": {{
                    "batch_size": batch_size,
                    "steps": steps,
                    "gradient_accumulation_steps": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit",
                    "lr": 1.2e-4,
                    "lr_scheduler": "cosine",
                    "dtype": "bf16"
                }},
                "model": {{
                    "name_or_path": ckpt_path,
                    "arch": "krea2",
                    "quantize": not is_high_vram,
                    "low_vram": not is_high_vram
                }}
            }}]
        }}
    }}

    cfg_path = f"/content/{{LORA_NAME}}.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(train_cfg, f, sort_keys=False, allow_unicode=True)

    print(f"🔥 Starting training {{LORA_NAME}} ({{steps}} steps, batch_size={{batch_size}})...", flush=True)
    builtins.IS_BUSY = True
    proc_train = subprocess.Popen(
        ["python", "-u", "/content/ai-toolkit/run.py", cfg_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    last_print = 0
    buf = ""
    while True:
        chunk = os.read(proc_train.stdout.fileno(), 4096)
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        parts = buf.replace("\\r", "\\n").split("\\n")
        buf = parts[-1]
        for p in parts[:-1]:
            s = p.strip()
            if not s:
                continue
            now = time.time()
            builtins.LAST_ACTIVE_TIME = now
            if any(k in s for k in ["Saving", "Error", "Traceback", "Caching", "Loading", "1000/1000"]) or (now - last_print >= 10.0):
                print(s[-140:], flush=True)
                last_print = now
    if buf.strip():
        print(buf.strip()[-140:], flush=True)
    ret = proc_train.wait()
    builtins.IS_BUSY = False
    if ret != 0:
        raise RuntimeError(f"Training failed with exit code {{ret}}")

print(f"✅ Trained files in {{local_lora_dir}}: {{os.listdir(local_lora_dir)}}", flush=True)
alias_dir = f"{{local_loras_root}}/liu_yiyi_lora"
os.makedirs(alias_dir, exist_ok=True)
if os.path.exists(final_safetensors):
    shutil.copy2(final_safetensors, f"{{alias_dir}}/liu_yiyi_lora.safetensors")

if os.path.exists("/content/drive/MyDrive"):
    drive_lora_dir = f"/content/drive/MyDrive/AI_Manga_Studio/loras/{{LORA_NAME}}"
    os.makedirs(drive_lora_dir, exist_ok=True)
    os.system(f"cp -r {{local_lora_dir}}/*.safetensors {{drive_lora_dir}}/")
    print(f"☁️ Backed up {{LORA_NAME}} to Google Drive: {{os.listdir(drive_lora_dir)}}", flush=True)

print("🎉 Training complete! Running 8-image Liu Yiyi LoRA v1 Verification Batch...", flush=True)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

default_neg_bw = "color, polychrome, 3d render, CGI, photorealistic metallic reflection, glossy 3d shading, smooth airbrush gradient, soft grey gradient on armor, heavy dark chiaroscuro, dense western cross-hatching, thick heavy marker outlines, blotchy ink, bad hands, mutated hands, extra fingers, missing fingers, fused fingers, six fingers, deformed hands, speech bubble, text, watermark, signature"

resident_engine.generate_instant(
    prompts={json.dumps(TEST_PROMPTS, ensure_ascii=False)},
    steps=15,
    cfg=1.65,
    width=1024,
    height=1376,
    seed=889966,
    prefix="liu_yiyi_lora_v1_test",
    lora_name=f"{{LORA_NAME}}:0.85",
    neg=default_neg_bw
)
"""

    was_running = cdp_eval(ws, """
    (() => {
      const cell = document.getElementById("cell-one_click_train_krea2");
      const isRunning = cell && (cell.classList.contains("running") || cell.classList.contains("pending"));
      const rb = cell && cell.querySelector("colab-run-button");
      const btn = rb && rb.shadowRoot ? rb.shadowRoot.querySelector("button") : rb;
      if (isRunning && btn) {
        btn.click();
        return true;
      }
      return false;
    })()
    """)
    if was_running:
        print("⏸️ Interrupted previous cell execution, waiting 3s...", flush=True)
        time.sleep(3)

    cdp_eval(ws, f"""
    (() => {{
      window.monaco.editor.getModels()[4].setValue({json.dumps(cell_code)});
      const rb = document.getElementById("cell-one_click_train_krea2").querySelector("colab-run-button");
      const btn = rb && rb.shadowRoot ? rb.shadowRoot.querySelector("button") : rb;
      btn.click();
    }})()
    """)
    print(f"🚀 Launched NVMe-direct train_liu_yiyi_lora_v1 on Colab (marker={marker})!", flush=True)
    auto_approve_drive_oauth(marker)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    while True:
        try:
            tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9223/json", timeout=5).read())
            ws = [t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page" and "colab.research.google.com" in t.get("url", "")][0]

            out_tail = ""
            main_out = cdp_eval(ws, """
            (() => {
              const cell = document.getElementById("cell-one_click_train_krea2");
              if (!cell) return "";
              const out = cell.querySelector(".output, .output-content, colab-static-output-renderer");
              return out ? out.innerText : "";
            })()
            """) or ""
            candidates = [main_out]
            for t in tabs:
                if "outputframe" in t.get("url", ""):
                    txt = cdp_eval(t["webSocketDebuggerUrl"], "document.body ? document.body.innerText : \"\"") or ""
                    if txt:
                        candidates.append(txt)

            for txt in candidates:
                if f"MARKER:{marker}" in txt:
                    sub = txt[txt.rfind(f"MARKER:{marker}"):]
                    out_tail = sub
                    if "INSTANT_B64_START:" in sub and ":INSTANT_B64_END" in sub:
                        i1 = sub.find("INSTANT_B64_START:")
                        i2 = sub.find(":INSTANT_B64_END")
                        if i1 != -1 and i2 != -1 and i2 > i1:
                            data = json.loads(sub[i1 + len("INSTANT_B64_START:"):i2])
                            for k, b64 in data.items():
                                raw = base64.b64decode(b64)
                                p_out = os.path.join(OUT_DIR, k)
                                with open(p_out, "wb") as f:
                                    f.write(raw)
                                with open(os.path.join(ARTIFACT_DIR, k), "wb") as f:
                                    f.write(raw)
                                print(f"✅ SAVED: {p_out} ({len(raw)} bytes)", flush=True)
                            return
                    if "Traceback (most recent call last)" in sub or "RuntimeError: Training failed" in sub:
                        print(f"❌ COLAB ERROR:\n{sub[-2500:]}", flush=True)
                        sys.exit(1)

            lines = [l.strip() for l in out_tail.splitlines() if l.strip()]
            last = lines[-1] if lines else "Waiting for cell output..."
            print(f"  [{time.strftime('%H:%M:%S')}] {last[:120]}", flush=True)
        except Exception as e:
            print(f"  [{time.strftime('%H:%M:%S')}] Poll warning: {e}", flush=True)
        time.sleep(10)

if __name__ == "__main__":
    main()
