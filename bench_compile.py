"""#1: torch.compile the FP4 block to fuse the ~40% eager elementwise (RoPE/SiLU/
RMSNorm/residuals). Dynamo graph-breaks around the custom FP4Linear op but compiles
the regions between. Compares eager vs compiled for NVFP4, plus BF16 compiled."""
import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scaling"))
from block_bench import QwenBlock
import torch._dynamo as dyn
dyn.config.suppress_errors = True   # fall back to eager on any un-traceable region
DEV = "cuda"
B, T = 8, 1024
dim = int(os.environ.get("DIM", "8192"))
hd = 128; nh = dim // hd; nkv = max(1, nh // 4); ffn = int(round(dim * 8 / 3 / 128)) * 128
klist = [(dim, nh*hd), (dim, nkv*hd), (dim, nkv*hd), (nh*hd, dim), (dim, ffn), (dim, ffn), (ffn, dim)]
fl = 6 * B * T * sum(ki*ko for ki, ko in klist)

def bench(fp4, compiled, iters=20):
    blk = QwenBlock(dim, nh, nkv, hd, ffn, fp4).to(DEV)
    if not fp4:
        blk = blk.to(torch.bfloat16)
    fwd = torch.compile(blk) if compiled else blk
    x = torch.randn(B, T, dim, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = fwd(x); loss = y.float().pow(2).mean()
        loss.backward(); x.grad = None
        for p in blk.parameters(): p.grad = None
    for _ in range(8): step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters): step()
    torch.cuda.synchronize(); dt = (time.time() - t) / iters
    return dt * 1e3, fl / dt / 1e12

print("%6s %7s %9s %9s %11s" % ("dim", "backend", "compile", "ms/step", "lin TFLOPS"))
for fp4, nm in [(False, "bf16"), (True, "nvfp4")]:
    for comp in ([False, True] if fp4 else [True]):
        try:
            ms, tf = bench(fp4, comp)
            print("%6d %7s %9s %9.1f %11.0f" % (dim, nm, str(comp), ms, tf))
        except Exception as e:
            print("%6d %7s %9s  FAIL %s: %s" % (dim, nm, str(comp), type(e).__name__, str(e)[:50]))
