"""Profile one GPT training step (dim4096/nl6, matching the e2e sweep) and bucket the
CUDA kernels, so we can diff BF16 vs optimized-NVFP4 and see exactly what NVFP4 adds."""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_text import GPT, get_batch
from nvfp4_train import mark_step, _AMORTIZE
from torch.profiler import profile, ProfilerActivity
DEV = "cuda"
USE_FP4 = os.environ.get("NVFP4_CUDA", "0") == "1"
dim, nl, nh, nkv, hd = 4096, 6, 32, 8, 128
ffn = 4 * dim
torch.manual_seed(0)
model = GPT(50257, dim, nl, nh, nkv, hd, ffn, USE_FP4).to(DEV)
cmodel = model
if os.environ.get("COMPILE", "0") == "1":
    import torch._dynamo as _d; _d.config.suppress_errors = True
    cmodel = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
train = np.memmap("/home/infatoshi/data/owt/train.bin", dtype=np.uint16, mode="r")
g = torch.Generator().manual_seed(1)
def step():
    opt.zero_grad(set_to_none=True)
    x, y = get_batch(train, 12, 512, g)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = cmodel(x, y)
    loss.backward(); opt.step()
    if _AMORTIZE: mark_step()
for _ in range(6): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5): step()
    torch.cuda.synchronize()

# bucket by kernel-name substring
ka = prof.key_averages()
buckets = {"cuBLAS GEMM": 0.0, "CUTLASS FP4 GEMM": 0.0, "quant cast (quant_kernel)": 0.0,
           "scale reduce (amax)": 0.0, "swizzle/copy/cast (elementwise)": 0.0,
           "flash attention": 0.0, "norm/rope/act (triton)": 0.0, "softmax/CE": 0.0,
           "optimizer (adamw)": 0.0, "memcpy": 0.0, "other": 0.0}
def bucket(n):
    s = n.lower()
    if "cutlass" in s or "kernel2" in s: return "CUTLASS FP4 GEMM"
    if "gemm" in s or "cublas" in s or "s16816" in s or "ampere" in s: return "cuBLAS GEMM"
    if "quant_kernel" in s: return "quant cast (quant_kernel)"
    if "reduce" in s: return "scale reduce (amax)"
    if "flash" in s: return "flash attention"
    if "softmax" in s or "nll" in s or "cross_entropy" in s: return "softmax/CE"
    if "adam" in s or "optimizer" in s or "multi_tensor" in s: return "optimizer (adamw)"
    if "memcpy" in s: return "memcpy"
    if "triton" in s or "rms" in s or "rope" in s or "silu" in s or "relu" in s: return "norm/rope/act (triton)"
    if "elementwise" in s or "copy" in s or "cat" in s or "index" in s: return "swizzle/copy/cast (elementwise)"
    return "other"
tot = 0.0
for e in ka:
    t = e.self_device_time_total
    tot += t
    buckets[bucket(e.key)] += t
if os.environ.get("RAW", "0") == "1":
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=24))
tag = "OPT-NVFP4" if USE_FP4 else "BF16"
print(f"\n=== [{tag}] per-step CUDA buckets (us, %) total={tot/5/1e3:.1f} ms/step ===")
for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
    if v > 0:
        print(f"  {k:34s} {v/5/1e3:8.2f} ms  {100*v/tot:5.1f}%")
