"""Parity: CUTLASS SM120 NVFP4 GEMM vs the existing cuBLAS (torch._scaled_mm) path.

For each shape (M,N,K) all multiples of 128:
  A = randn[M,K] bf16, B = randn[N,K] bf16  (B stored as [N,K] = B_logical.t())
  a  = quant_nvfp4_cuda(A)          -> NVFP4Tensor, logical [M,K]
  bt = quant_nvfp4_cuda(B)          -> NVFP4Tensor, logical [N,K]
  ref  = _addmm_nvfp4_dispatch(a, bt.t(), mm)   # cuBLAS, computes A @ B_logical.t()... = [M,N]
  mine = nvfp4_bf16_gemm(a.qdata, a.scale, bt.qdata, bt.scale, M, N, K,
                         alpha = a.per_tensor_scale * b.per_tensor_scale)
  rel_L2 = ||mine - ref|| / ||ref||   must be < 1e-2
"""
import sys
import torch

sys.path.insert(0, "/home/infatoshi/experiments/_scratch/nvfp4-validate")
sys.path.insert(0, "/home/infatoshi/experiments/_scratch/nvfp4-validate/cutlass_gemm")

from nvfp4_cuda import quant_nvfp4_cuda, _addmm_nvfp4_dispatch, _MM
from cutlass_ext import nvfp4_bf16_gemm

SHAPES = [(4096, 4096, 4096), (8192, 8192, 8192), (16384, 2048, 2048)]


def run_shape(M, N, K):
    torch.manual_seed(0)
    dev = "cuda"
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    # B stored as [N, K] so quant blocks along K (contraction dim).
    Bt = torch.randn(N, K, device=dev, dtype=torch.bfloat16)

    a = quant_nvfp4_cuda(A)          # logical [M, K]
    bt = quant_nvfp4_cuda(Bt)        # logical [N, K]

    # Reference (cuBLAS path). bt.t() gives a [K, N] NVFP4Tensor whose qdata.t()
    # is contiguous -> _addmm computes a @ bt.t() = [M, N].
    ref = _addmm_nvfp4_dispatch(a, bt.t(), _MM)
    assert ref.shape == (M, N) and ref.dtype == torch.bfloat16, (ref.shape, ref.dtype)

    alpha = float((a.per_tensor_scale * bt.per_tensor_scale).item())

    mine = nvfp4_bf16_gemm(a.qdata, a.scale, bt.qdata, bt.scale, M, N, K, alpha)
    torch.cuda.synchronize()

    rel_l2 = (mine.float() - ref.float()).norm().item() / ref.float().norm().item()
    return rel_l2, ref, mine


def main():
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name()}")
    all_ok = True
    for (M, N, K) in SHAPES:
        rel_l2, ref, mine = run_shape(M, N, K)
        ok = rel_l2 < 1e-2
        all_ok &= ok
        print(f"({M},{N},{K})  rel_L2={rel_l2:.3e}  "
              f"ref[0,:3]={ref.float()[0,:3].tolist()}  "
              f"mine[0,:3]={mine.float()[0,:3].tolist()}  "
              f"-> {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
