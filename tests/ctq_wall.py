import os, sys, torch, time, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scaling"))
from train_text import GPT, get_batch
from nvfp4_train import mark_step, _AMORTIZE
DEV = "cuda"; dim, nl, nh, nkv, hd = 4096, 6, 32, 8, 128; ffn = 4 * dim
torch.manual_seed(0)
m = GPT(50257, dim, nl, nh, nkv, hd, ffn, True).to(DEV)
import torch._dynamo as _d; _d.config.suppress_errors = True
cm = torch.compile(m)
opt = torch.optim.AdamW(m.parameters(), lr=6e-4, fused=True)
tr = np.memmap("/home/infatoshi/data/owt/train.bin", dtype=np.uint16, mode="r")
g = torch.Generator().manual_seed(1)


def step():
    opt.zero_grad(set_to_none=True)
    x, y = get_batch(tr, 12, 512, g)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = cm(x, y)
    loss.backward(); opt.step()
    if _AMORTIZE: mark_step()


for _ in range(12): step()
torch.cuda.synchronize(); t0 = time.time()
N = 40
for _ in range(N): step()
torch.cuda.synchronize()
print(f"wall ms/step = {(time.time()-t0)/N*1e3:.2f}  FUSED_CT={os.environ.get('NVFP4_FUSED_CT')}")
