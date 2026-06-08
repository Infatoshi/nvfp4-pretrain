"""Empirical wall-clock crossover: fwd+bwd of a Qwen3-style block, BF16 vs NVFP4,
across model dim. No LM head (isolates the transformer-block GEMMs). Reports step
time, tokens/sec, and effective TFLOPS on the block linears (6*M*Kin*Kout / time)."""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from nvfp4_train import FP4Linear
DEV = "cuda"


def lin(i, o, fp4):
    return FP4Linear(i, o) if fp4 else nn.Linear(i, o, bias=False)


def rope(x, T):
    B, H, _, Dh = x.shape
    half = Dh // 2
    inv = 1.0 / (10000 ** (torch.arange(0, half, device=x.device).float() / half))
    ang = torch.outer(torch.arange(T, device=x.device).float(), inv)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class QwenBlock(nn.Module):
    # RMSNorm pre-norm, GQA + QK-norm, SwiGLU, RoPE, no bias (Qwen3-dense)
    def __init__(s, dim, nh, nkv, hd, ffn, fp4):
        super().__init__()
        s.nh, s.nkv, s.hd = nh, nkv, hd
        s.n1, s.n2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        s.qn, s.kn = nn.RMSNorm(hd), nn.RMSNorm(hd)
        s.q, s.k, s.v = lin(dim, nh * hd, fp4), lin(dim, nkv * hd, fp4), lin(dim, nkv * hd, fp4)
        s.o = lin(nh * hd, dim, fp4)
        s.gate, s.up = lin(dim, ffn, fp4), lin(dim, ffn, fp4)
        s.down = lin(ffn, dim, fp4)

    def forward(s, x):
        B, T, D = x.shape
        h = s.n1(x)
        q = s.qn(s.q(h).view(B, T, s.nh, s.hd)).transpose(1, 2)
        k = s.kn(s.k(h).view(B, T, s.nkv, s.hd)).transpose(1, 2)
        v = s.v(h).view(B, T, s.nkv, s.hd).transpose(1, 2)
        q, k = rope(q, T), rope(k, T)
        rep = s.nh // s.nkv
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, s.nh * s.hd)
        x = x + s.o(a)
        h = s.n2(x)
        x = x + s.down(F.silu(s.gate(h)) * s.up(h))
        return x


def run(dim, fp4, B=16, T=1024, iters=30):
    hd = 128
    nh = dim // hd
    nkv = max(1, nh // 4)
    ffn = int(round(dim * 8 / 3 / 128)) * 128
    blk = QwenBlock(dim, nh, nkv, hd, ffn, fp4).to(DEV)
    if not fp4:
        blk = blk.to(torch.bfloat16)
    x = torch.randn(B, T, dim, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    klist = [(dim, nh * hd), (dim, nkv * hd), (dim, nkv * hd), (nh * hd, dim), (dim, ffn), (dim, ffn), (ffn, dim)]
    fl = 6 * B * T * sum(ki * ko for ki, ko in klist)  # fwd+bwd linear FLOPs
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = blk(x)
            loss = y.float().pow(2).mean()
        loss.backward()
        x.grad = None
        for p in blk.parameters():
            p.grad = None
    for _ in range(8):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    tok = B * T
    return dt * 1e3, tok / dt, fl / dt / 1e12


print("%6s %7s %9s %10s %11s" % ("dim", "backend", "ms/step", "tok/s", "lin TFLOPS"))
for dim in [1024, 2048, 4096]:
    for fp4, nm in [(False, "bf16"), (True, "nvfp4")]:
        try:
            ms, tps, tf = run(dim, fp4)
            print("%6d %7s %9.1f %10.0f %11.0f" % (dim, nm, ms, tps, tf))
        except Exception as e:
            print("%6d %7s  FAILED %s: %s" % (dim, nm, type(e).__name__, str(e)[:60]))
