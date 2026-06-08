"""Bitwise validation of the fused cast-transpose-quant kernel vs reference
quant_nvfp4_cuda(x.t().contiguous(), ...)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nvfp4_cuda import quant_nvfp4_cuda, hadamard16, BLK
from cast_transpose_quant import cast_transpose_quant_nvfp4_cuda

DEV = "cuda"
SHAPES = [(6144, 4096), (6144, 16384), (2048, 2048)]
SEED = 1234


def check(M, K, sr, rht, use_smem):
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    H = hadamard16(DEV)
    g = torch.Generator().manual_seed(1)
    signs = (torch.randint(0, 2, (BLK,), generator=g).float() * 2 - 1).to(DEV)
    ref = quant_nvfp4_cuda(x.t().contiguous(), stochastic=sr, rht=rht, H=H, signs=signs, seed=SEED)
    got = cast_transpose_quant_nvfp4_cuda(x, stochastic=sr, rht=rht, H=H, signs=signs,
                                          seed=SEED, use_smem=use_smem)
    qd_eq = torch.equal(ref.qdata, got.qdata)
    sc_eq = torch.equal(ref.scale.view(torch.uint8), got.scale.view(torch.uint8))
    qd_rate = (ref.qdata == got.qdata).float().mean().item()
    sc_rate = (ref.scale.view(torch.uint8) == got.scale.view(torch.uint8)).float().mean().item()
    return qd_eq, sc_eq, qd_rate, sc_rate


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "smem"
    use_smem = backend == "smem"
    print(f"=== backend={backend} (use_smem={use_smem}) ===")
    allpass = True
    for (M, K) in SHAPES:
        for sr in (False, True):
            for rht in (False, True):
                qd_eq, sc_eq, qdr, scr = check(M, K, sr, rht, use_smem)
                ok = qd_eq and sc_eq
                allpass &= ok
                print(f"[{M:>5}x{K:<6}] SR={int(sr)} RHT={int(rht)} | "
                      f"qdata={'OK ' if qd_eq else 'FAIL'}({qdr*100:6.2f}%) "
                      f"scale={'OK ' if sc_eq else 'FAIL'}({scr*100:6.2f}%)")
    print("ALL PASS" if allpass else "SOME FAILED")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
