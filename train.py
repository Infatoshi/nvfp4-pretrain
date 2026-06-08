#!/usr/bin/env python3
"""Config-driven launcher for nvfp4-pretrain.

Reads config.yaml and maps it to the trainer's env-vars + CLI args, then runs the
(validated, unchanged) trainer in scaling/train_text.py. This is the one entry point
so you never touch NVFP4_CUDA / FP8 / MUON / ... by hand.

    uv run train.py [config.yaml]
"""
import os, sys, subprocess, pathlib
try:
    import yaml
except ImportError:
    sys.exit("pyyaml missing — `uv sync` or `uv add pyyaml`")

ROOT = pathlib.Path(__file__).resolve().parent
cfg_path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
cfg = yaml.safe_load(open(cfg_path))

env = dict(os.environ)
env.setdefault("CUDA_HOME", "/usr/local/cuda-13")
env.pop("LD_PRELOAD", None)
env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- precision -> env flags ---
prec = cfg["precision"].lower()
if prec == "nvfp4":
    n = cfg.get("nvfp4", {})
    env["NVFP4_CUDA"] = "1"
    env["NVFP4_AMORTIZE"] = "1" if n.get("amortize", True) else "0"
    env["NVFP4_CUTLASS"] = "1" if n.get("cutlass", True) else "0"
    env["NVFP4_FUSED_CT"] = "1" if n.get("fused_ct", True) else "0"
    env["NVFP4_NAIVE"] = "1" if n.get("naive", False) else "0"
elif prec == "fp8":
    env["FP8"] = "1"
    env["FP8_RECIPE"] = cfg.get("fp8", {}).get("recipe", "tensorwise")
elif prec != "bf16":
    sys.exit(f"unknown precision: {prec}")

# --- optimizer -> env flags ---
opt = cfg["optimizer"]
name = opt.get("name", "adamw").lower()
if name == "muon":
    env["MUON"] = "1"
    env["MUON_LR"] = str(opt.get("muon_lr", 0.01))
elif name == "adamw8bit":
    env["BNB8"] = "1"
elif name != "adamw":
    sys.exit(f"unknown optimizer: {name}")

# --- model/train/data/log -> CLI args ---
m, t, lg = cfg["model"], cfg["train"], cfg["log"]
for p in (lg["traj"], lg["out"]):
    pathlib.Path(ROOT / p).parent.mkdir(parents=True, exist_ok=True)
args = [
    "--dim", m["dim"], "--nl", m["layers"], "--nh", m["heads"], "--nkv", m["kv_heads"],
    "--T", m["seq_len"], "--bs", m["batch_size"],
    "--steps", t["steps"], "--warmup", t["warmup"], "--grad_accum", t.get("grad_accum", 1),
    "--lr", opt.get("lr", 6e-4), "--compile", 1 if t.get("compile", True) else 0,
    "--data", cfg["data"]["path"], "--logevery", lg["every"], "--tag", lg["tag"],
    "--traj", str(ROOT / lg["traj"]), "--out", str(ROOT / lg["out"]),
]
cmd = [sys.executable, "-u", str(ROOT / "scaling" / "train_text.py")] + [str(a) for a in args]
print(f"[launch] precision={prec} optimizer={name} dim={m['dim']} nl={m['layers']} "
      f"bs={m['batch_size']} steps={t['steps']}", flush=True)
sys.exit(subprocess.run(cmd, env=env, cwd=str(ROOT / "scaling")).returncode)
