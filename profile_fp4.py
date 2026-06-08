"""Profile the NVFP4 FP4Linear path (dim 8192 block, fwd+bwd) to see where time goes:
quant kernels (quant_launch) vs CUTLASS GEMM (Kernel2) vs attention vs the rest.
Decides whether prologue-fusing the quant is worth the effort."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scaling"))
from block_bench import QwenBlock
from torch.profiler import profile, ProfilerActivity
DEV = "cuda"
dim = int(os.environ.get("DIM", "8192"))
hd = 128; nh = dim // hd; nkv = max(1, nh // 4); ffn = int(round(dim * 8 / 3 / 128)) * 128
blk = QwenBlock(dim, nh, nkv, hd, ffn, True).to(DEV)
x = torch.randn(8, 1024, dim, device=DEV, dtype=torch.bfloat16, requires_grad=True)
def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = blk(x); loss = y.float().pow(2).mean()
    loss.backward(); x.grad = None
    for p in blk.parameters(): p.grad = None
for _ in range(5): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5): step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=22))
