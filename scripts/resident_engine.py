import os, sys, time, glob, yaml, json, base64, builtins, threading
from PIL import Image

STUDIO = "/content/drive/MyDrive/AI_Manga_Studio"
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["HF_HUB_CACHE"] = "/content/hf_cache/hub"
os.makedirs("/content/hf_cache/hub", exist_ok=True)
try:
    import huggingface_hub.constants as _hfc
    _hfc.HF_HOME = "/content/hf_cache"
    _hfc.HF_HUB_CACHE = "/content/hf_cache/hub"
    _hfc.HUGGINGFACE_HUB_CACHE = "/content/hf_cache/hub"
except Exception:
    pass
try:
    import transformers.utils.hub as _thub
    _thub.TRANSFORMERS_CACHE = "/content/hf_cache/hub"
except Exception:
    pass

def ensure_env():
    if "/content/ai-toolkit" not in sys.path or not os.path.exists("/content/ai-toolkit/requirements.txt"):
        if not os.path.exists("/content/ai-toolkit/requirements.txt") or not os.path.exists("/content/ai-toolkit/run.py"):
            print("[Cold-Start 1/4] Cloning/syncing ai-toolkit (excluding .git FUSE bottleneck)...", flush=True)
            ret = os.system("rm -rf /content/ai-toolkit && git clone --depth 1 https://github.com/ostris/ai-toolkit.git /content/ai-toolkit")
            if ret != 0 or not os.path.exists("/content/ai-toolkit/requirements.txt"):
                os.system(f"mkdir -p /content/ai-toolkit && rsync -a --exclude='.git' --exclude='output' --exclude='__pycache__' {STUDIO}/ai-toolkit/ /content/ai-toolkit/")
        if not os.path.exists("/content/ai-toolkit/reqs_py313.txt"):
            print("[Cold-Start 2/4] Installing Python dependencies...", flush=True)
            with open("/content/ai-toolkit/requirements.txt") as f:
                reqs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            fixed = [r.split("==")[0] if any(r.startswith(k) for k in ["scipy","numpy","pillow","opencv","albumentations","torch","torchaudio","torchvision"]) else r for r in reqs]
            with open("/content/ai-toolkit/reqs_py313.txt", "w") as f:
                f.write("\n".join(fixed))
            os.system("pip install -q -r /content/ai-toolkit/reqs_py313.txt")
            os.system("pip install -q --upgrade transformers accelerate safetensors huggingface_hub optimum-quanto lycoris-lora flatten_json pyyaml oyaml tensorboard kornia invisible-watermark einops prodigyopt")
            import torchao.quantization as tq
            if not hasattr(tq, "FqnToConfig"):
                with open(tq.__file__, "a") as f: f.write("\nFqnToConfig = dict\n")
            import diffusers.pipelines.ltx2.export_utils as ltx_exp
            if not hasattr(ltx_exp, "encode_video"):
                with open(ltx_exp.__file__, "a") as f: f.write("\nencode_video = None\n")
        sys.path.insert(0, "/content/ai-toolkit")

def _start_watchdog():
    builtins.IDLE_TIMEOUT_SEC = 3600
    if getattr(builtins, "WATCHDOG_STARTED", False):
        return
    builtins.WATCHDOG_STARTED = True
    def _loop():
        while True:
            time.sleep(15)
            if getattr(builtins, "IS_BUSY", False):
                builtins.LAST_ACTIVE_TIME = time.time()
                continue
            idle = time.time() - getattr(builtins, "LAST_ACTIVE_TIME", time.time())
            if idle > getattr(builtins, "IDLE_TIMEOUT_SEC", 300):
                print("[Watchdog] Idle 5 mins, unassigning GPU...", flush=True)
                from google.colab import runtime
                runtime.unassign()
                break
    threading.Thread(target=_loop, daemon=True).start()

class _FirstBatchDone(Exception):
    pass

def generate_instant(prompts, steps=8, cfg=1.5, width=896, height=1200, seed=2026, prefix="gen", lora_name=None, neg=""):
    builtins.IS_BUSY = True
    builtins.LAST_ACTIVE_TIME = time.time()
    _start_watchdog()
    base_lora = lora_name
    lora_weight = 1.0
    if lora_name and ":" in lora_name:
        base_lora, w_str = lora_name.split(":", 1)
        try:
            lora_weight = float(w_str)
        except Exception:
            lora_weight = 1.0
    job_name = base_lora if base_lora else "krea2_base_direct"
    work_root = "/content/AI_Manga_Studio_Local"
    infer_root = f"{work_root}/infer_runtime"
    os.makedirs(f"{infer_root}/krea2_resident/samples", exist_ok=True)
    sample_dir = f"{infer_root}/krea2_resident/samples"
    t0 = time.time()

    try:
        current_mode = getattr(builtins, "GLOBAL_JOB_NAME", None)
        if not hasattr(builtins, "GLOBAL_PROCESS") or builtins.GLOBAL_PROCESS is None:
            print(f"[Cold-Start] Initializing environment & GPU VRAM cache (mode={job_name})...", flush=True)
            ensure_env()
            infer_job_dir = f"{infer_root}/krea2_resident"
            if os.path.exists(infer_job_dir):
                for sf in glob.glob(f"{infer_job_dir}/*.safetensors"):
                    os.remove(sf)
            os.makedirs(f"{infer_job_dir}/samples", exist_ok=True)
            dummy_dir = f"{work_root}/dummy_dataset"
            if not os.path.exists(f"{dummy_dir}/dummy.jpg"):
                os.makedirs(dummy_dir, exist_ok=True)
                Image.new("RGB", (1024, 1024), (255, 255, 255)).save(f"{dummy_dir}/dummy.jpg")
                with open(f"{dummy_dir}/dummy.txt", "w") as df:
                    df.write("anime illustration\n")
            ckpt_path = f"{STUDIO}/checkpoints/krea2_turbo_bf16.safetensors" if os.path.exists(f"{STUDIO}/checkpoints/krea2_turbo_bf16.safetensors") else "/content/krea2_turbo_bf16.safetensors"
            if not os.path.exists(ckpt_path):
                print("[Cold-Start 2.5/4] Fetching krea2_turbo_bf16.safetensors via 10Gbps aria2c to local NVMe...", flush=True)
                os.system("apt-get -y install -qq aria2 >/dev/null 2>&1")
                os.system('aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_bf16.safetensors" -d "/content" -o "krea2_turbo_bf16.safetensors"')
            print("[Cold-Start 3/4] Loading 35.5GB Krea-2-Turbo & Qwen3-VL into GPU VRAM...", flush=True)
            from toolkit.job import get_job
            proc_cfg = {
                "type": "sd_trainer",
                "training_folder": infer_root,
                "device": "cuda:0",
                "network": {"type": "lora", "linear": 32, "linear_alpha": 32},
                "save": {"dtype": "bf16", "save_every": 500, "max_step_saves_to_keep": 4},
                "datasets": [{
                    "folder_path": dummy_dir,
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "resolution": [1024],
                    "cache_latents_to_disk": True
                }],
                "train": {
                    "batch_size": 1, "steps": 1500, "gradient_accumulation_steps": 2,
                    "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch", "optimizer": "adamw8bit", "lr": 1e-4, "dtype": "bf16"
                },
                "model": {
                    "name_or_path": ckpt_path,
                    "arch": "krea2", "quantize": False
                },
                "sample": {
                    "sampler": "flowmatch", "sample_every": 500,
                    "width": width, "height": height, "prompts": prompts, "neg": neg,
                    "seed": seed, "walk_seed": True, "guidance_scale": cfg, "sample_steps": steps
                }
            }
            if lora_name:
                if "jiang_nan" in lora_name:
                    proc_cfg["trigger_word"] = "jiang_nan"
                elif "he_lingyi" in lora_name:
                    proc_cfg["trigger_word"] = "he_lingyi"
                elif "su_qingge" in lora_name:
                    proc_cfg["trigger_word"] = "su_qingge"
                else:
                    proc_cfg["trigger_word"] = "by_imigimuru"
            config = {
                "job": "extension",
                "config": {
                    "name": "krea2_resident",
                    "process": [proc_cfg]
                }
            }
            cfg_path = "/content/instant_job.yaml"
            with open(cfg_path, "w") as f:
                yaml.dump(config, f, sort_keys=False, allow_unicode=True)
            import gc, torch
            builtins.GLOBAL_PROCESS = None
            sys.last_traceback = None
            sys.last_value = None
            sys.last_type = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            job = get_job(cfg_path)
            proc = job.process[0]
            orig_sample = proc.sample
            def _hook(step, is_first=False):
                if not base_lora and getattr(proc, "network", None) is not None:
                    proc.network.multiplier = 0.0
                    proc.network.is_active = False
                elif base_lora and getattr(proc, "network", None) is not None:
                    cands = sorted(glob.glob(f"{work_root}/loras/{base_lora}/*.safetensors") + glob.glob(f"{STUDIO}/loras/{base_lora}/*.safetensors") + glob.glob(f"/content/loras_train/{base_lora}/*.safetensors"))
                    if not cands:
                        raise RuntimeError(f"LoRA weights not found for {base_lora} in {work_root}/loras or {STUDIO}/loras!")
                    if hasattr(proc.network, "load_weights"):
                        print(f"[LoRA-Load] Explicitly loading trained weights from {cands[-1]} (multiplier={lora_weight})...", flush=True)
                        proc.network.load_weights(cands[-1])
                    proc.network.multiplier = lora_weight
                    proc.network.is_active = True
                print(f"[Cold-Start 4/4] Model in VRAM (lora={base_lora}, weight={lora_weight})! Sampling images now...", flush=True)
                orig_sample(step, is_first=is_first)
                raise _FirstBatchDone()
            proc.sample = _hook
            try:
                proc.run()
            except _FirstBatchDone:
                proc.sample = orig_sample
            builtins.GLOBAL_PROCESS = proc
            builtins.GLOBAL_JOB_NAME = job_name
            print(f"[VRAM-Resident] Model ({job_name}) pinned in GPU VRAM! Next batches will take 0s load time!", flush=True)
        else:
            print(f"[Hot-Hit] Model already in VRAM (0s load, mode={job_name}, weight={lora_weight})! Generating immediately...", flush=True)
            from toolkit.config_modules import SampleConfig
            proc = builtins.GLOBAL_PROCESS
            try:
                dev = getattr(proc, "device_torch", "cuda:0")
                if hasattr(proc, "sd") and proc.sd is not None:
                    if getattr(proc.sd, "unet", None) is not None:
                        proc.sd.unet.to(dev)
                    if getattr(proc.sd, "vae", None) is not None:
                        proc.sd.vae.to(dev)
                    te = getattr(proc.sd, "text_encoder", None)
                    if isinstance(te, list):
                        for t in te:
                            if t is not None: t.to(dev)
                    elif te is not None:
                        te.to(dev)
            except Exception as _e:
                print(f"[Hot-Hit Device Check] {_e}", flush=True)
            if not base_lora and getattr(proc, "network", None) is not None:
                proc.network.multiplier = 0.0
                proc.network.is_active = False
            elif base_lora and getattr(proc, "network", None) is not None:
                if current_mode != job_name:
                    cands = sorted(glob.glob(f"{work_root}/loras/{base_lora}/*.safetensors") + glob.glob(f"{STUDIO}/loras/{base_lora}/*.safetensors") + glob.glob(f"/content/loras_train/{base_lora}/*.safetensors"))
                    if not cands:
                        raise RuntimeError(f"LoRA weights not found for {base_lora} in {work_root}/loras or {STUDIO}/loras!")
                    if hasattr(proc.network, "load_weights"):
                        print(f"[LoRA-Load] Hot-swapping trained weights from {cands[-1]} (multiplier={lora_weight})...", flush=True)
                        proc.network.load_weights(cands[-1])
                proc.network.multiplier = lora_weight
                proc.network.is_active = True
            builtins.GLOBAL_JOB_NAME = job_name
            proc.sample_config = SampleConfig(
                sampler="flowmatch", sample_every=500,
                width=width, height=height, prompts=prompts, neg=neg,
                seed=seed, walk_seed=True, guidance_scale=cfg, sample_steps=steps
            )
            proc.sample(0 if not lora_name else 1500)
    finally:
        builtins.IS_BUSY = False
        builtins.LAST_ACTIVE_TIME = time.time()

    builtins.LAST_ACTIVE_TIME = time.time()
    new_files = sorted([f for f in glob.glob(f"{sample_dir}/*.jpg") if os.path.getmtime(f) > t0])[:len(prompts)]
    out = {}
    for i, f in enumerate(new_files):
        with open(f, "rb") as fp:
            out[f"{prefix}_{i}.jpg"] = base64.b64encode(fp.read()).decode("ascii")
    print("INSTANT_B64_START:" + json.dumps(out) + ":INSTANT_B64_END", flush=True)
