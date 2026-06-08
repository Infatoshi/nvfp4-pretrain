"""Isolated timing: fused cast-transpose-quant vs .t().contiguous()+quant, and
FP4Linear fwd+bwd parity (fused wgrad vs original) within bf16 noise."""
import sys, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nvfp4_cuda import quant_nvfp4_cuda, hadamard16, BLK
from cast_transpose_quant import cast_transpose_quant_nvfp4_cuda

DEV = "cuda"


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms


def isolated():
    print("=== isolated: fused CTQ vs .t().contiguous()+quant ===")
    H = hadamard16(DEV)
    g = torch.Generator().manual_seed(1)
    signs = (torch.randint(0, 2, (BLK,), generator=g).float() * 2 - 1).to(DEV)
    for (M, K) in [(6144, 16384), (6144, 4096)]:
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        for sr, rht in [(True, True), (False, False)]:
            ref = bench(lambda: quant_nvfp4_cuda(x.t().contiguous(), stochastic=sr,
                                                 rht=rht, H=H, signs=signs, seed=7))
            fus = bench(lambda: cast_transpose_quant_nvfp4_cuda(x, stochastic=sr, rht=rht,
                                                               H=H, signs=signs, seed=7))
            print(f"  [{M}x{K}] SR={int(sr)} RHT={int(rht)} | "
                  f"transpose+quant={ref:.3f}ms  fused={fus:.3f}ms  speedup={ref/fus:.2f}x")


def wgrad_parity():
    """Deterministic: original wgrad vs fused wgrad with the SAME explicit seed.
    Same-seed -> must be bitwise-identical output (rel 0). Different seed -> SR
    noise ~1e-2 (negative control proves SR is actually firing)."""
    print("\n=== wgrad parity: fused vs original (.t().contiguous()) ===")
    from nvfp4_cuda import fp4_matmul_cuda
    from cast_transpose_quant import wgrad_fp4_matmul_ct, hadamard16

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()

    H = hadamard16(DEV)
    g = torch.Generator().manual_seed(1)
    signs = (torch.randint(0, 2, (BLK,), generator=g).float() * 2 - 1).to(DEV)
    torch.manual_seed(0)
    M, N, K = 4096, 768, 1024            # gy:[M,N], x:[M,K] -> dW:[N,K]
    gy = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)

    for tag, sr in [("RNE", False), ("SR", True)]:
        S = 4242
        dw_o = fp4_matmul_cuda(gy.t().contiguous(), x, sr_a=sr, sr_b=False,
                               rht=True, H=H, signs=signs, seed=S)
        dw_f = wgrad_fp4_matmul_ct(gy, x, sr_gy=sr, rht=True, H=H, signs=signs, seed=S)
        print(f"  [{tag}] same-seed rel-L2 = {rel(dw_f, dw_o):.2e}  (expect 0.00 -> bitwise)")
    # negative control: SR with different seeds must DIFFER (proves SR fires)
    dw_a = wgrad_fp4_matmul_ct(gy, x, sr_gy=True, rht=True, H=H, signs=signs, seed=1)
    dw_b = wgrad_fp4_matmul_ct(gy, x, sr_gy=True, rht=True, H=H, signs=signs, seed=2)
    print(f"  [SR] neg-control diff-seed rel-L2 = {rel(dw_a, dw_b):.2e}  (expect ~1e-2, nonzero)")


if __name__ == "__main__":
    isolated()
    wgrad_parity()
