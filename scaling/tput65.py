"""Sustained training throughput of a ~6.5B model: FP8 block linears (torchao tensorwise
e4m3) + bf16 flash attention + 8-bit AdamW. Warmup (compile) then time steady-state."""
import os, sys, time, torch, numpy as np, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_text import GPT, get_batch
DEV = "cuda"
dim, nl = 7168, 12
nh, nkv, hd = dim // 128, 8, 128
ffn = 4 * dim
B, T = 8, 1024
torch.manual_seed(0)
model = GPT(50257, dim, nl, nh, nkv, hd, ffn, False).to(DEV)
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
convert_to_float8_training(model, config=Float8LinearConfig(),
                           module_filter_fn=lambda m, fqn: "blocks." in fqn and isinstance(m, nn.Linear))
import torch._dynamo as d; d.config.suppress_errors = True
cm = torch.compile(model)
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
train = np.memmap("/home/infatoshi/data/owt/train.bin", dtype=np.uint16, mode="r")
g = torch.Generator().manual_seed(1)
nonemb = (sum(p.numel() for p in model.parameters()) - model.emb.weight.numel()) / 1e9
def step():
    opt.zero_grad(set_to_none=True)
    x, y = get_batch(train, B, T, g)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = cm(x, y)
    loss.backward(); opt.step()
print(f"building/compiling {nonemb:.2f}B model (dim{dim} nl{nl}, fp8 linears + bf16 attn)...", flush=True)
for _ in range(6): step()
torch.cuda.synchronize(); t = time.time()
N = 20
for _ in range(N): step()
torch.cuda.synchronize(); dt = (time.time() - t) / N
tps = B * T / dt
peak = torch.cuda.max_memory_allocated() / 1e9
mfu_flops = 6 * nonemb * 1e9 * tps / 1e12   # 6N*tok/s, TFLOPS (model-FLOPs, no attn)
print(f"RESULT {nonemb:.2f}B: {dt*1000:.0f} ms/step | {tps:,.0f} tok/s sustained | "
      f"{6*nonemb*1e9*tps/1e12:.0f} eff TFLOPS (6ND) | peak {peak:.1f} GB | tok/param/day {tps*86400/(nonemb*1e9):.2f}")
