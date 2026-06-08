"""Profile the FP8 training step (dim4096 bs32, the ~42% MFU config) and bucket CUDA
kernels to show exactly what caps MFU: GEMM vs attention vs cast vs CE vs norm/elementwise
vs optimizer."""
import os, sys, torch, numpy as np, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_text import GPT, get_batch
from torch.profiler import profile, ProfilerActivity
DEV = "cuda"
DIM, NL, B, T = 4096, 12, 32, 512
nh, nkv, hd = DIM // 128, 8, 128; ffn = 4 * DIM
torch.manual_seed(0)
model = GPT(50304, DIM, NL, nh, nkv, hd, ffn, False).to(DEV)
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
model.head.weight = nn.Parameter(model.emb.weight.detach().clone())
convert_to_float8_training(model, config=Float8LinearConfig(),
                           module_filter_fn=lambda m, fqn: ("blocks." in fqn or fqn == "head") and isinstance(m, nn.Linear))
import torch._dynamo as d; d.config.suppress_errors = True
cm = torch.compile(model)
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
train = np.memmap("/home/infatoshi/data/owt/train.bin", dtype=np.uint16, mode="r")
g = torch.Generator().manual_seed(1)
def step():
    opt.zero_grad(set_to_none=True)
    x, y = get_batch(train, B, T, g)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = cm(x, y)
    loss.backward(); opt.step()
for _ in range(6): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5): step()
    torch.cuda.synchronize()
buckets = {}
def bk(n):
    s = n.lower()
    if "gemm" in s or "cutlass" in s or "scaled_mm" in s or "s16816" in s: return "FP8/bf16 GEMM"
    if "flash" in s or "attention" in s or "fmha" in s: return "attention"
    if "softmax" in s or "nll" in s or "cross_entropy" in s: return "CE softmax"
    if "amax" in s or "to_fp8" in s or "float8" in s or "scaled" in s or "quant" in s: return "fp8 cast/scale"
    if "adam" in s or "multi_tensor" in s or "optimizer" in s: return "optimizer"
    if "norm" in s or "rms" in s or "rope" in s or "cat" in s or "silu" in s or "relu" in s or "triton" in s: return "norm/rope/act"
    if "elementwise" in s or "copy" in s or "vectorized" in s or "reduce" in s: return "elementwise/copy"
    if "memcpy" in s: return "memcpy"
    return "other"
tot = 0.0
for e in prof.key_averages():
    t = e.self_device_time_total; tot += t
    buckets[bk(e.key)] = buckets.get(bk(e.key), 0) + t
print("=== FP8 step kernel buckets (dim4096 bs32) ===")
for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
    if v > 0: print(f"  {k:18s} {100*v/tot:5.1f}%  ({v/5/1e3:.1f} ms/step)")
print(f"  total {tot/5/1e3:.0f} ms/step")
