"""P2: max model that fits one fwd+bwd+optimizer step in 96GB, per precision.
Sweeps dim (layers fixed) upward until OOM at a fixed batch/seq; reports the
largest that fits + peak VRAM. Precision via env: (none)=bf16, FP8=1, NVFP4_CUDA=1...,
BNB8=1 for 8-bit optimizer. Run one precision per process (env set externally)."""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_text import GPT, get_batch
from nvfp4_train import mark_step, _AMORTIZE
import torch.nn as nn
DEV = "cuda"
USE_FP4 = os.environ.get("NVFP4_CUDA", "0") == "1"
FP8 = os.environ.get("FP8", "0") == "1"
BNB8 = os.environ.get("BNB8", "0") == "1"
tag = "nvfp4" if USE_FP4 else ("fp8" if FP8 else "bf16")
B, T, NL = 8, 1024, 12
train = np.memmap("/home/infatoshi/data/owt/train.bin", dtype=np.uint16, mode="r")
g = torch.Generator().manual_seed(1)

def try_dim(dim):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    nh = max(1, dim // 128); nkv = max(1, nh // 4); hd = dim // nh; ffn = 4 * dim
    try:
        model = GPT(50257, dim, NL, nh, nkv, hd, ffn, USE_FP4).to(DEV)
        if FP8:
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig
            convert_to_float8_training(model, config=Float8LinearConfig(),
                                       module_filter_fn=lambda m, fqn: "blocks." in fqn and isinstance(m, nn.Linear))
        N = model.num_params()
        if BNB8:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            x, y = get_batch(train, B, T, g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward(); opt.step()
            if _AMORTIZE: mark_step()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del model, opt; torch.cuda.empty_cache()
        return N, peak, None
    except RuntimeError as e:
        del_e = "OOM" if "out of memory" in str(e).lower() else str(e)[:60]
        try:
            del model, opt
        except Exception:
            pass
        torch.cuda.empty_cache()
        return None, None, del_e

print(f"[{tag}] B={B} T={T} NL={NL} BNB8={BNB8}  (max model in 96GB)")
best = None
for dim in [1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192]:
    N, peak, err = try_dim(dim)
    if N is None:
        print(f"  dim {dim:>5}: OOM/err ({err})"); break
    print(f"  dim {dim:>5}: N={N/1e9:.3f}B  peak {peak:.1f} GB  OK", flush=True)
    best = (dim, N, peak)
if best:
    print(f"RESULT {tag}: max dim {best[0]}, N={best[1]/1e9:.3f}B non-emb, peak {best[2]:.1f}GB")
