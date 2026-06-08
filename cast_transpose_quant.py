"""Fused cast-transpose-quant NVFP4 kernel for sm_120 (wgrad path).

Eliminates the separate `x.t().contiguous()` transpose copy before NVFP4
quantization. Reads bf16 X [M,K] (row-major, K contiguous) and produces the
NVFP4 quantized representation of X^T [K,M], block-scaled along M (16-blocks of
the token dim), in a single coalesced HBM pass via SMEM tiling.

Per-16-block math is IDENTICAL to nvfp4_cuda.quant_kernel (RHT/amax/scale/pack),
and the SR RNG is indexed by the row-major linear index of the [K,M] output so
results are BITWISE-equal to quant_nvfp4_cuda(X.t().contiguous(), ...).

Build: nvcc load_inline, arch sm_120a.
"""
import os, math, torch
from torch.utils.cpp_extension import load_inline
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor, hp_data_dims_to_swizzled_scale_dims_nvfp4)
import nvfp4_cuda as _ncu

F4_MAX, F8E4M3_MAX, E4M3_EPS, BLK = 6.0, 448.0, 0.015625, 16
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13")

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ float u01(unsigned int a, unsigned int b){
    unsigned int x = a * 0x9e3779b9u + b + 0x85ebca6bu;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return (x >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ int e2m1_sr_code(float m, float u){
    float lo, hi; int loi;
    if(m < 0.5f){lo=0.f;hi=0.5f;loi=0;}
    else if(m<1.0f){lo=0.5f;hi=1.0f;loi=1;}
    else if(m<1.5f){lo=1.0f;hi=1.5f;loi=2;}
    else if(m<2.0f){lo=1.5f;hi=2.0f;loi=3;}
    else if(m<3.0f){lo=2.0f;hi=3.0f;loi=4;}
    else if(m<4.0f){lo=3.0f;hi=4.0f;loi=5;}
    else if(m<6.0f){lo=4.0f;hi=6.0f;loi=6;}
    else return 7;
    float p = (m - lo) / (hi - lo);
    return (u < p) ? (loi + 1) : loi;
}

// Per-16-block math, identical to quant_kernel. v[16] are the 16 elements along
// the (contiguous in output) M dim for one output row k, starting at m0=bm*16.
// linidx = k*M + m0 is the row-major linear index of element 0 in [K,M] -> drives RNG.
template<bool DO_SR, bool DO_RHT>
__device__ __forceinline__ void quant_block16(
        float* v, uint8_t* data_out, uint8_t* scale_out,
        float s_enc, const float* __restrict__ h_ptr, const float* __restrict__ sg_ptr,
        long linidx, unsigned int seed){
    if(DO_RHT){
        float t[16];
        #pragma unroll
        for(int i=0;i<16;i++) t[i] = v[i] * sg_ptr[i];
        #pragma unroll
        for(int j=0;j<16;j++){
            float acc = 0.f;
            #pragma unroll
            for(int i=0;i<16;i++) acc += t[i] * h_ptr[i*16 + j];
            v[j] = acc;
        }
    }
    float amax = 0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ float a = fabsf(v[i]); amax = a>amax?a:amax; }
    float sbs = fminf(fmaxf((amax / 6.0f) * s_enc, 0.015625f), 448.0f);
    __nv_fp8_e4m3 sbs8(sbs);
    *scale_out = *reinterpret_cast<uint8_t*>(&sbs8);
    float recip = s_enc / float(sbs8);
    #pragma unroll
    for(int j=0;j<8;j++){
        float d0 = fminf(fmaxf(v[2*j]   * recip, -6.f), 6.f);
        float d1 = fminf(fmaxf(v[2*j+1] * recip, -6.f), 6.f);
        if(DO_SR){
            float u0 = u01(seed, (unsigned)(linidx + 2*j));
            float u1 = u01(seed, (unsigned)(linidx + 2*j + 1));
            int c0 = e2m1_sr_code(fabsf(d0), u0) | (d0 < 0.f ? 8 : 0);
            int c1 = e2m1_sr_code(fabsf(d1), u1) | (d1 < 0.f ? 8 : 0);
            data_out[j] = (uint8_t)(c0 | (c1 << 4));
        } else {
            __nv_fp4x2_storage_t p = __nv_cvt_float2_to_fp4x2(make_float2(d0, d1),
                                                              __NV_E2M1, cudaRoundNearest);
            data_out[j] = (uint8_t)p;
        }
    }
}

// ---- Scalar reference kernel (no SMEM): strided column reads. One thread per
// output 16-block. Used to validate the per-block math/RNG in isolation. ----
template<bool DO_SR, bool DO_RHT>
__global__ void ctq_scalar_kernel(const __nv_bfloat16* __restrict__ x,
        uint8_t* __restrict__ data, uint8_t* __restrict__ scale,
        const float* __restrict__ pts_ptr, const float* __restrict__ h_ptr,
        const float* __restrict__ sg_ptr, long M, long K, unsigned int seed){
    long nbm = M / 16;                 // blocks along M per output row
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;  // global block id over [K, M/16]
    if(idx >= K * nbm) return;
    long k  = idx / nbm;
    long bm = idx % nbm;
    long m0 = bm * 16;
    float v[16];
    #pragma unroll
    for(int i=0;i<16;i++) v[i] = __bfloat162float(x[(m0 + i) * K + k]);  // X[m0+i, k]
    float s_enc = 1.0f / pts_ptr[0];
    quant_block16<DO_SR,DO_RHT>(v, data + k * (M/2) + bm * 8, scale + k * (M/16) + bm,
                                s_enc, h_ptr, sg_ptr, k * M + m0, seed);
}

// ---- SMEM-tiled kernel. Block processes a TILE_M x TILE_K tile of X.
// Load coalesced along K into SMEM (padded), then each output 16-block (k, bm)
// reads its 16 M-values from SMEM. Threads map (tk = K within tile, tb = which
// 16-block within tile's M) so adjacent threads -> adjacent M-blocks of same k
// -> coalesced 8-byte qdata writes. ----
#define TILE_M 64
#define TILE_K 64
#define TM_BLK (TILE_M/16)   // = 4 blocks of 16 along M per tile

template<bool DO_SR, bool DO_RHT>
__global__ void ctq_smem_kernel(const __nv_bfloat16* __restrict__ x,
        uint8_t* __restrict__ data, uint8_t* __restrict__ scale,
        const float* __restrict__ pts_ptr, const float* __restrict__ h_ptr,
        const float* __restrict__ sg_ptr, long M, long K, unsigned int seed){
    // smem[m_local][k_local], padded +8 along k: avoids bank conflicts on column
    // reads AND keeps the row stride (72 bf16 = 144B) 16-byte aligned for float4 stores.
    __shared__ __nv_bfloat16 sm[TILE_M][TILE_K + 8];

    long tile_m0 = (long)blockIdx.y * TILE_M;   // M offset of this tile
    long tile_k0 = (long)blockIdx.x * TILE_K;   // K offset of this tile

    // ---- coalesced load: X[tile_m0 + r, tile_k0 + c], c fast (contiguous in K). ----
    // 256 threads load TILE_M*TILE_K = 4096 elems -> 16 each. Vectorize as bf16x8 (float4).
    int tid = threadIdx.x;                       // 0..255
    const int NTHREAD = TILE_M * (TILE_K / 8);   // 64*8 = 512 vec-slots; loop if fewer threads
    #pragma unroll
    for(int s = tid; s < TILE_M * (TILE_K/8); s += blockDim.x){
        int r = s / (TILE_K/8);
        int cv = (s % (TILE_K/8)) * 8;
        long gm = tile_m0 + r, gk = tile_k0 + cv;
        const float4* src = reinterpret_cast<const float4*>(&x[gm * K + gk]);
        *reinterpret_cast<float4*>(&sm[r][cv]) = *src;
    }
    (void)NTHREAD;
    __syncthreads();

    float s_enc = 1.0f / pts_ptr[0];
    // ---- each thread handles output blocks: k_local in [0,TILE_K), bm_local in [0,TM_BLK).
    // Map tid so bm_local is fast-varying -> coalesced writes along M. ----
    for(int t = tid; t < TILE_K * TM_BLK; t += blockDim.x){
        int k_local  = t / TM_BLK;
        int bm_local = t % TM_BLK;
        long k = tile_k0 + k_local;
        long bm = (tile_m0 / 16) + bm_local;     // global block index along M
        long m0 = bm * 16;
        float v[16];
        int m_loc = bm_local * 16;
        #pragma unroll
        for(int i=0;i<16;i++) v[i] = __bfloat162float(sm[m_loc + i][k_local]);
        quant_block16<DO_SR,DO_RHT>(v, data + k * (M/2) + bm * 8, scale + k * (M/16) + bm,
                                    s_enc, h_ptr, sg_ptr, k * M + m0, seed);
    }
}

void ctq_launch(torch::Tensor x, torch::Tensor data, torch::Tensor scale,
                torch::Tensor pts, torch::Tensor H, torch::Tensor signs,
                int64_t M, int64_t K, int64_t seed, bool do_sr, bool do_rht, bool use_smem){
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto dp = data.data_ptr<uint8_t>();
    auto sp = scale.data_ptr<uint8_t>();
    auto pp = pts.data_ptr<float>();
    auto hp = H.data_ptr<float>();
    auto gp = signs.data_ptr<float>();
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    unsigned int sd = (unsigned)seed;
#define LAUNCH_SMEM(SR,RHT) do { \
        dim3 grid(K / TILE_K, M / TILE_M); \
        ctq_smem_kernel<SR,RHT><<<grid, 256, 0, s>>>(xp,dp,sp,pp,hp,gp,M,K,sd); \
    } while(0)
#define LAUNCH_SCALAR(SR,RHT) do { \
        long nb = K * (M/16); int th = 256; long bl = (nb + th - 1)/th; \
        ctq_scalar_kernel<SR,RHT><<<bl, th, 0, s>>>(xp,dp,sp,pp,hp,gp,M,K,sd); \
    } while(0)
    if(use_smem){
        if(do_sr && do_rht) LAUNCH_SMEM(true,true);
        else if(do_sr) LAUNCH_SMEM(true,false);
        else if(do_rht) LAUNCH_SMEM(false,true);
        else LAUNCH_SMEM(false,false);
    } else {
        if(do_sr && do_rht) LAUNCH_SCALAR(true,true);
        else if(do_sr) LAUNCH_SCALAR(true,false);
        else if(do_rht) LAUNCH_SCALAR(false,true);
        else LAUNCH_SCALAR(false,false);
    }
}
'''

_CPP = ("void ctq_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, bool, bool, bool);")

_H16 = {}
_ZERO16 = {}
_mod = None


def _ext():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="ctq_nvfp4_ext", cpp_sources=_CPP, cuda_sources=_CUDA_SRC,
            functions=["ctq_launch"],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_120a,code=sm_120a"],
            verbose=False)
    return _mod


def hadamard16(device):
    Hm = torch.ones(1, 1, device=device, dtype=torch.float32)
    while Hm.shape[0] < BLK:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return Hm / math.sqrt(BLK)


def cast_transpose_quant_nvfp4_cuda(x, stochastic=False, rht=False, H=None,
                                    signs=None, seed=0, use_smem=True):
    """x: [M,K] bf16/fp32 (M,K % 16 == 0). Returns NVFP4Tensor of x^T [K,M],
    block-scaled along M. Bitwise-matches quant_nvfp4_cuda(x.t().contiguous(),...)."""
    M, K = x.shape
    # SMEM path tiles in TILE_M x TILE_K (=64) blocks and vec-loads float4; needs %64.
    # Scalar path only needs %16. Block-scaling along M always needs M%16==0.
    if use_smem:
        assert M % 64 == 0 and K % 64 == 0, f"SMEM path needs M,K %% 64 (got M={M},K={K})"
    else:
        assert M % BLK == 0 and K % BLK == 0, f"M={M},K={K} must be divisible by {BLK}"
    tf = x.to(torch.bfloat16).contiguous()
    dev = x.device
    # pts is identical for x and x^T (same elements). Match reference exactly.
    mn, mx = torch.aminmax(tf)
    amax = torch.maximum(mx.abs(), mn.abs()).float().clamp_min(1e-12)
    pts = (amax * ((4.0 if rht else 1.0) / (F4_MAX * F8E4M3_MAX))).reshape(1)
    if dev not in _ZERO16:
        _ZERO16[dev] = (torch.zeros(BLK, BLK, device=dev), torch.zeros(BLK, device=dev))
    if rht:
        Hd = (H if H is not None else _H16.setdefault(dev, hadamard16(dev))).contiguous().float()
        Sd = (signs if signs is not None else torch.ones(BLK, device=dev)).contiguous().float()
    else:
        Hd, Sd = _ZERO16[dev]
    # output is x^T: [K, M]
    data = torch.empty((K, M // 2), dtype=torch.uint8, device=dev)
    scale = torch.empty((K, M // BLK), dtype=torch.float8_e4m3fn, device=dev)
    _ext().ctq_launch(tf, data, scale.view(torch.uint8), pts, Hd, Sd,
                      M, K, int(seed), bool(stochastic), bool(rht), bool(use_smem))
    sM, sK = hp_data_dims_to_swizzled_scale_dims_nvfp4(K, M)
    sw = to_blocked(scale).flatten().view(sM, sK)
    return NVFP4Tensor(data, sw, BLK, x.dtype, pts.reshape(()),
                       None, True, False, None)


def wgrad_fp4_matmul_ct(gy, x, sr_gy=True, rht=True, H=None, signs=None, seed=None,
                        use_smem=True):
    """Fused wgrad: dW = gy^T @ x, with gy:[M,N], x:[M,K] (M = token/contract dim).
    Produces both transposed+quantized operands via the fused cast-transpose-quant
    kernel (no .t().contiguous() copy), then runs the existing NVFP4 GEMM.

    Mirrors the original `fp4_matmul(gy.t(), x, sr_a=sr_gy, sr_b=False, rht=rht)`:
      A = quant(gy.t())  [N,M]  (contract M)  -> SR on gradient
      B = quant(x.t())   [K,M]  (contract M)  -> RNE on activation
    Returns bf16 [N,K]."""
    if seed is None:
        seed = _ncu._SEED[0]
        if sr_gy:
            _ncu._SEED[0] += 1
    a = cast_transpose_quant_nvfp4_cuda(gy, stochastic=sr_gy, rht=rht, H=H, signs=signs,
                                        seed=seed, use_smem=use_smem)   # quant(gy.t())
    bt = cast_transpose_quant_nvfp4_cuda(x, stochastic=False, rht=rht, H=H, signs=signs,
                                         seed=seed, use_smem=use_smem)  # quant(x.t())
    return _ncu._gemm_AB(a, bt)
