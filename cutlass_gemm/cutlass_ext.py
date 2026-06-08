"""JIT loader for the CUTLASS SM120 NVFP4 -> bf16 GEMM torch extension.

Exposes nvfp4_bf16_gemm(A_data, A_sf, B_data, B_sf, M, N, K, alpha) -> bf16 [M,N].
Build flags mirror cutlass_gemm/build_bf16out.sh exactly (gencode 120a, the three
CUTLASS include dirs, --expt-relaxed-constexpr / --expt-extended-lambda, --threads 0,
and the CFG_* defines pinning the single verified config 128x128x128 / Pingpong /
StageCountAutoCarveout / cluster 1x1x1).
"""
import os

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13")
os.environ.pop("LD_PRELOAD", None)

from torch.utils.cpp_extension import load

_CUT = "/home/infatoshi/cuda/engines/cutlass"
_DIR = os.path.dirname(os.path.abspath(__file__))

_mod = None


def _ext():
    global _mod
    if _mod is None:
        _mod = load(
            name="cutlass_nvfp4_ext",
            sources=[os.path.join(_DIR, "cutlass_nvfp4_ext.cu")],
            extra_cuda_cflags=[
                "-std=c++17", "-O3",
                "-gencode", "arch=compute_120a,code=sm_120a",
                "--expt-relaxed-constexpr", "--expt-extended-lambda",
                "--threads", "0",
                "-DCFG_TILE_M=256", "-DCFG_TILE_N=128", "-DCFG_TILE_K=128",
                "-DCFG_SCHED=1", "-DCFG_STAGES=0",
            ],
            extra_include_paths=[
                f"{_CUT}/include",
                f"{_CUT}/tools/util/include",
                f"{_CUT}/examples/common",
            ],
            verbose=True,
        )
    return _mod


def nvfp4_bf16_gemm(A_data, A_sf, B_data, B_sf, M, N, K, alpha):
    return _ext().nvfp4_bf16_gemm(A_data, A_sf, B_data, B_sf,
                                  int(M), int(N), int(K), float(alpha))
