"""Tuned FP8 end-to-end MFU: proper FLOP accounting (trunk linears + tied head GEMM +
attention), batch from env. Measures steady-state vs FP8 dense peak (~1000 TFLOPS on
RTX PRO 6000). One batch per process."""
import os, sys, time, torch, numpy as np, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_text import GPT, get_batch
DEV = "cuda"
DIM = int(os.environ.get("DIM", "4096")); NL = int(os.environ.get("NL", "12"))
B = int(os.environ.get("BS", "16")); T = int(os.environ.get("T", "512"))
FP8_PEAK = 1000e12
FP8_HEAD = os.environ.get("FP8_HEAD", "0") == "1"
VOCAB = 50304 if FP8_HEAD else 50257   # pad vocab to mult-of-128 so the head GEMM is FP8-aligned
nh, nkv, hd = DIM // 128, 8, 128; ffn = 4 * DIM
torch.manual_seed(0)
model = GPT(VOCAB, DIM, NL, nh, nkv, hd, ffn, False).to(DEV)
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
if FP8_HEAD:
    model.head.weight = nn.Parameter(model.emb.weight.detach().clone())  # untie so head can be FP8 (FLOPs identical)
    filt = lambda m, fqn: ("blocks." in fqn or fqn == "head") and isinstance(m, nn.Linear)
else:
    filt = lambda m, fqn: "blocks." in fqn and isinstance(m, nn.Linear)
convert_to_float8_training(model, config=Float8LinearConfig(), module_filter_fn=filt)
import torch._dynamo as d; d.config.suppress_errors = True
cm = torch.compile(model)
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
train = np.memmap("/home/infatoshi/data/owt/train.bin", dtype=np.uint16, mode="r")
g = torch.Generator().manual_seed(1)
nonemb = (sum(p.numel() for p in model.parameters()) - model.emb.weight.numel())
# FLOPs/step (fwd+bwd, factor 6 = 2*3): trunk linears + head GEMM + attention
Sblk = (DIM*nh*hd + 2*DIM*nkv*hd + nh*hd*DIM + 2*DIM*ffn)   # q,k,v,o,up,down
trunk = 6 * B*T * NL * Sblk
head = 6 * B*T * DIM * VOCAB
attn = 12 * NL * B * nh * T * T * hd
FL = trunk + head + attn
def step():
    opt.zero_grad(set_to_none=True)
    x, y = get_batch(train, B, T, g)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = cm(x, y)
    loss.backward(); opt.step()
try:
    for _ in range(6): step()
    torch.cuda.synchronize(); t = time.time()
    N = 15
    for _ in range(N): step()
    torch.cuda.synchronize(); dt = (time.time() - t) / N
    tps = B * T / dt; eff = FL / dt; mfu = eff / FP8_PEAK
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"RESULT dim{DIM} nl{NL} bs{B} T{T} ({nonemb/1e9:.2f}B): {dt*1000:.0f} ms | {tps:,.0f} tok/s | "
          f"{eff/1e12:.0f} TFLOPS | MFU {mfu*100:.1f}% (head={head/FL*100:.0f}% attn={attn/FL*100:.0f}%) | peak {peak:.1f}GB")
except RuntimeError as e:
    print(f"FAIL bs{B}: {'OOM' if 'out of memory' in str(e).lower() else str(e)[:50]}")
