# DEVLOG

The journey of getting NVFP4 (4-bit float) training to work, and to run fast, on
an RTX PRO 6000 Blackwell Workstation (sm_120, compute cap 12.0) where NVIDIA's
Transformer Engine does not. Newest entries at the bottom. Numbers here were read
from commands that exited 0; anything not yet measured is marked PENDING.

Hardware: RTX PRO 6000 Blackwell Workstation 96GB, sm_120, 600W. Host "anvil-lan"
(Ryzen 9 9950X3D, 96GB DDR5, Ubuntu 24.04). torch 2.11.0+cu130, torchao 0.17.0,
triton 3.6.0, CUDA 13.2 toolkit (nvcc V13.2.51), driver 610.43.02. CUTLASS v4.5.0
at /home/infatoshi/cuda/engines/cutlass (external dependency, not vendored here).

---

## 1. The paper and the four stabilizers

Started from NVIDIA's NVFP4 pre-training recipe (arXiv 2509.25149). NVFP4 is E2M1
(1 sign / 2 exp / 1 mantissa, values +/- {0, .5, 1, 1.5, 2, 3, 4, 6}), block size 16
along K, a per-block E4M3 scale, and a per-tensor FP32 scale (two-level scaling).
Naive 4-bit training diverges; the paper's four load-bearing techniques are:

1. Selective high-precision layers (~15%, weighted to the end of the network).
2. Random Hadamard Transform (16x16) on the Wgrad cast, to spread outliers.
3. 2D weight scaling so forward and backward see consistent quantization.
4. Stochastic rounding on gradients (unbiased), instead of round-to-nearest.

## 2. Reference sim and the addition probe

Built a fake-quant NVFP4 simulator implementing all four techniques
(`nvfp4_validate.py`) as a correctness oracle, then a downscaled Nemotron-style
decoder trained on a 3-digit-addition char-LM with held-out pairs as a
generalization probe. Verified result:

| config | held-out acc | val loss |
|---|---|---|
| bf16 reference | 100% | 0.965 |
| NVFP4 full recipe (SR+RHT), all blocks | 100% | 0.965 |
| NVFP4 without SR/RHT (the only path TE runs on sm_120) | 1.9% (stalls) | 1.34 |

The stabilizers are not optional: drop SR+RHT and the model collapses. This is the
ablation that answers "is the match real or just an under-trained tie" - at this
scale the recipe is clearly the difference between converging and stalling.

## 3. Transformer Engine does not run on sm_120

TE 2.15's fused NVFP4 path crashes on sm_120. Two root causes, both found by
bisection:
- Its RHT/SR mega-kernel requests dynamic shared memory sized for the sm_100
  datacenter budget (~232 KB), exceeding sm_120's 101376-byte opt-in cap
  (`cudaFuncSetAttribute` returns "invalid argument").
- sm_120 has no hardware stochastic-rounding cast: `cvt.rs.*.e2m1` is rejected by
  ptxas ("Feature '.rs' not supported"). sm_120 DOES have the round-to-nearest
  cast `__nv_cvt_float2_to_fp4x2` and the native FP4 tensor-core GEMM.

Filed upstream as NVIDIA/TransformerEngine#3062. `te/nvfp4_sm120_degrade.patch`
makes TE auto-disable RHT/SR with a warning instead of crashing (degraded path only).

## 4. Decompose: own the quant, borrow the GEMM

Since TE will not run and sm_120 lacks the hardware SR cast, the approach is to
decompose the problem: write our own quantization (software SR + in-kernel RHT +
two-level scaling + nibble packing) and hand the quantized operands to a native
FP4 GEMM via `torch._scaled_mm` (cuBLASLt, BlockWise1x16 + SWIZZLE_32_4_4, through
torchao's `_addmm_nvfp4_dispatch`).

What is from scratch: the quantizers. Triton (`nvfp4_triton_quant.py`,
`_quant_pack_v2`) and a faster CUDA kernel (`nvfp4_cuda.py`,
`quant_kernel<DO_SR, DO_RHT>`) using the hardware E2M1 cast intrinsic plus a
wang-hash software SR (since the hardware SR cast is unavailable). Plus the
autograd `FP4Linear` (Fprop RNE / Dgrad SR / Wgrad RHT+SR) and weight-quant
amortization.

What is borrowed: the FP4 matmul itself (cuBLASLt via `torch._scaled_mm`) and the
hardware float->FP4 cast intrinsic. Writing a competitive FP4 GEMM was deferred
(see section 7).

Quantizer speed @ 8192x4096 (verified): the bottleneck was the per-tensor amax in
fp32 plus a `.item()` host sync (86% of quant time), not the rounding. Fixed with
bf16 amax and on-device scale.

| quantizer | time | bandwidth |
|---|---|---|
| pure torch | 3.7 ms | 23 GB/s |
| fused Triton | 0.45 ms | ~195 GB/s |
| hardware-cast CUDA | 0.12 ms | ~730 GB/s |

Gotcha worth remembering: torchao's `F.linear` / `torch.mm` on two NVFP4Tensors
silently takes a dequant path (correct numerics, NOT the FP4 tensor cores). You
must call `_addmm_nvfp4_dispatch` for the real GEMM. Separately, do NOT LD_PRELOAD
the system cuBLASLt (that is the TE workaround); it breaks `torch._scaled_mm` with
CUBLAS_STATUS_NOT_INITIALIZED.

## 5. Scaling-law study: NVFP4 generalizes like BF16 on real text

To answer "does this generalize beyond a toy task," ran a Chinchilla-style
iso-token sweep on OpenWebText (GPT-2 BPE, nanoGPT-style decoder: RMSNorm, GQA,
RoPE, squared-ReLU FFN, weight-tied head). Fitted L(N) = E + A * N^(-alpha).
Verified fits:
- BF16:  L = 5.53 + 745 * N^(-0.500)
- NVFP4: L = 5.52 + 823 * N^(-0.500)

Same exponent, indistinguishable curves across the sweep (figure in
`results/scaling_law.png`). Honest caveat recorded here too: this is small and
under-trained (~25M params, ~20-25M tokens/model). The decisive test is
convergence-scale, which has NOT been run. At this scale part of the "match" is
simply that neither model is trained far enough for FP4 error to dominate; what we
can claim is that the full recipe holds parity and the stabilizer ablation is
decisive.

## 6. The 80%-SOL goal: build a real CUTLASS FP4 GEMM for sm_120

Goal: push every NVFP4 GEMM call we make to 80%+ of the dense FP4 speed-of-light.

SOL definition: NVIDIA's datasheet lists 4000 AI TOPS for this card "using
sparsity" (2:4). Dense FP4 (no sparsity), which is what we run, is half of that:
2000 TFLOPS. That is the 100%-SOL figure. (A compute-clock derivation gives
188 SM x 3090 MHz x 4096 FP4 FLOP/SM/clk = 2379 TFLOPS as an upper bound assuming
sustained max boost; the datasheet-derived 2000 is the realistic dense ceiling.)

Baseline (cuBLAS FP4 via `torch._scaled_mm`, gemm-only, GPU idle, verified):
- 16384^3: 1136.6 TFLOPS = 56.8% SOL  (best case)
- 8192^3:  1110.9 = 55.5%
- 4096^3:  1010.1 = 50.5%
- The six real training shapes are far worse: 6.3% to 33.0% SOL (skinny/K-heavy).

So cuBLAS leaves a large gap, especially on the training shapes. Built a standalone
CUTLASS NVFP4xNVFP4 GEMM from example 79b_blackwell_geforce_nvfp4_nvfp4_gemm,
ArchTag cutlass::arch::Sm120, OpClassBlockScaledTensorOp, e2m1 data + e4m3 per-16
block scale. Source: `cutlass_gemm/nvfp4_gemm.cu`, build via
`cutlass_gemm/build.sh TM TN TK SCHED`. Every benchmarked config is bitwise-verified
against the CUTLASS host reference (Disposition: Passed) before its number counts.

Hard sm_120 constraints discovered the hard way:
- ClusterShape MUST be 1x1x1 (GeForce SM120 has no TMA multicast).
- ONLY the 128x128x128 MMA tile + Pingpong schedule COMPILES. Every 256-dim tile
  and the Cooperative schedule fail with a hard static_assert in
  MainloopSm120TmaWarpSpecializedBlockScaled. So the winning config is forced, not
  chosen.

Verified square results (GPU idle, 100 iters, median via CUDA events):

| shape | CUTLASS FP4 TFLOPS | %SOL | cuBLAS baseline | delta |
|---|---|---|---|---|
| 4096^3  | 1419.97 | 71.0% | 1010.1 | +40.6% |
| 8192^3  | 1594.81 | 79.7% | 1110.9 | +43.6% |
| 16384^3 | 1577.34 | 78.9% | 1136.6 | +38.8% |

All three beat cuBLAS by ~39-44%. 8192 and 16384 are essentially at the 80% target;
4096 trails at 71%. IMPORTANT measurement gotcha: numbers collapse under GPU
contention (a contended 16384 read 1148 TFLOPS), so benchmark only with nvidia-smi
showing 0% util and no compute apps.

## 7. In progress / PENDING (where to resume)

The goal is NOT yet met. Two threads were running and were stopped when the GPU was
needed for other work; their kernels are built and on disk but their numbers were
never captured to a file, so they are PENDING, not verified.

PENDING-A, push squares over 80% (cheaper epilogue): built a BF16-output variant
from example 79a_blackwell_geforce_nvfp4_bf16_gemm (NVFP4 inputs, BF16 output, no
SFD scale-factor generation) - cheaper epilogue, and more realistic for training
since the GEMM result feeds the next op in higher precision. Source:
`cutlass_gemm/nvfp4_gemm_bf16out.cu`, build via `build_bf16out.sh TM TN TK SCHED
[STAGES]`. Binaries built for stage counts 0/4/6/8 and Pingpong/Cooperative. NOT
yet benchmarked. Hypothesis: should lift 4096 toward 8192's efficiency and may push
8192/16384 over 80%.

PENDING-B, the six real training shapes (the actual point of the goal). These are
skinny/K-heavy and were NEVER tuned past the cuBLAS baseline. They are:

| M | N | K | produced by | cuBLAS %SOL |
|---|---|---|---|---|
| 16384 | 512  | 512   | q/k/v/o fprop+dgrad | 11.5% |
| 16384 | 512  | 2048  | down.fprop, up.dgrad | 33.0% |
| 16384 | 2048 | 512   | up.fprop, down.dgrad | 17.6% |
| 512   | 512  | 16384 | q/k/v/o wgrad | 6.3% |
| 512   | 2048 | 16384 | down.wgrad | 24.4% |
| 2048  | 512  | 16384 | up.wgrad | 24.7% |

ROOFLINE DONE (cutlass_gemm/roofline.py, pure arithmetic, no GPU). This settles what
the target even is. Card: dense FP4 peak 2000 TFLOPS, GDDR7 BW 1792 GB/s, ridge point
1116 FLOP/byte. NVFP4 inputs 0.5 byte/elem + e4m3 block scales (1 byte/16) + bf16
output. Result:

| M | N | K | bound | real ceiling TFLOPS | =% of 2000 | cuBLAS now | cuBLAS % of REAL ceiling |
|---|---|---|---|---|---|---|---|
| 16384 | 512  | 512   | BW   | 711  | 35.5% | 11.5% | 32.3% |
| 16384 | 512  | 2048  | BW   | 1699 | 85.0% | 33.0% | 38.8% |
| 16384 | 2048 | 512   | BW   | 850  | 42.5% | 17.6% | 41.4% |
| 512   | 512  | 16384 | BW   | 1545 | 77.3% |  6.3% |  8.2% |
| 512   | 2048 | 16384 | comp | 2000 | 100%  | 24.4% | 24.4% |
| 2048  | 512  | 16384 | comp | 2000 | 100%  | 24.7% | 24.7% |

KEY CONCLUSION: "80% of 2000 TFLOPS" (1600) is PHYSICALLY IMPOSSIBLE on 4 of the 6
shapes - they are memory-bandwidth-bound, and even a kernel that perfectly saturates
1.79 TB/s cannot exceed AI*BW (e.g. 16384x512x512 is hard-capped at 711 TFLOPS = 35.5%
of the compute peak). No CUTLASS tuning moves a memory wall. The only honest target is
80% of each shape's APPLICABLE roofline (compute peak for the 2 compute-bound shapes,
AI*BW for the 4 BW-bound ones). Measured against that real ceiling, cuBLAS sits at
8-41%, so there is still real headroom everywhere - just not to 1600 TFLOPS on the
skinny shapes. The two K=16384 compute-bound shapes (512x2048x16384, 2048x512x16384)
are the ones where the full 80%-of-2000 push is meaningful.

Skinny tiles (128x32, 128x64, 256x128) were built to improve occupancy on the
small-M/N shapes but are NOT yet benchmarked. Split-K / stream-K along the huge
K=16384 dimension is the key untried lever for the wgrad shapes (4,5,6): K=16384 with
tiny M,N means one CTA does a very long serial K-reduction with little parallelism, so
splitting K across CTAs should help occupancy and approach the BW/compute ceiling.

Resume checklist:
1. `ssh anvil-lan`, confirm GPU idle (nvidia-smi 0% util). Note: GPU clock may be
   locked to 3090 MHz from a prior session; reset with `sudo nvidia-smi -rgc` if not
   benchmarking, or leave locked for stable numbers.
2. Benchmark PENDING-A (bf16out variants) at 4096/8192/16384, append a verified
   "## Square-shape push" section to `cutlass_gemm/RESULTS.md`.
3. Roofline DONE (roofline.py). Now benchmark/tune the six shapes against their REAL
   ceiling (table above): the 2 compute-bound K=16384 shapes target 80% of 2000; the 4
   BW-bound shapes target 80% of their AI*BW ceiling. Key lever for wgrad shapes is
   split-K/stream-K. Append "## Training shapes" to RESULTS.md with %-of-applicable-
   roofline clearly labeled (NOT %-of-2000 for the BW-bound ones - that would be
   dishonest since 1600 is unreachable there).
4. Only count a number if its command exited 0 and Disposition: Passed.

NOTE ON THE STATED GOAL: "80%+ SOL on all NVFP4 GEMM calls" where SOL=2000 TFLOPS is
not achievable on the 4 bandwidth-bound training shapes by physics, regardless of
kernel quality. This is documented above with the arithmetic. The achievable
reinterpretation is 80% of each shape's applicable roofline. This needs a human call
on whether to (a) accept the roofline-relative target, (b) change the GEMM shapes
(bigger batch/tokens to make them compute-bound), or (c) accept that cuBLAS is already
near-ceiling on some shapes and only chase the compute-bound ones.

## 8. 100M-token scaling rerun + throughput/crossover analysis (2026-06-06)

Reran the real-text L(N) study at 5x the tokens (~100M tok/model vs the original
~20M) to tighten the equivalence claim. Same 4 sizes, bs128 x T512 x 1525 steps.
Driver `scaling/run_sweep_100M.sh`, results `data/scaling_results_100M.jsonl`, figure
`data/scaling_law_100M.png` (plot via `scaling/plot_scaling_100M.py`). All read from
runs that exited 0.

Verified val loss (OpenWebText, GPT-2 BPE):

| N (non-emb) | BF16 | NVFP4 | gap (nvfp4-bf16) |
|---|---|---|---|
| 4.7M  | 4.777 | 4.812 | +0.035 |
| 10.6M | 4.476 | 4.510 | +0.034 |
| 25.2M | 4.331 | 4.340 | +0.008 |
| 49.2M | 4.189 | (skipped) | - |

The 49M NVFP4 point was killed mid-run (the slowest point, ~75 min) once the trend was
clear. Cleaner than the 20M run: all gaps small and positive (+0.008..+0.035), shrinking
with scale, no sign-flipping noise. The power-law fit is NOT the trustworthy evidence here
(NVFP4 at 3 points is underdetermined: alpha swings 0.46->0.76 depending on bounds). The
per-point gaps are. BF16 4-point fit: L = 3.96 + 2226*N^-0.515. Conclusion stands: NVFP4
tracks BF16 point-for-point at convergence-ish scale on real text.

THROUGHPUT (verified, single GPU, tok/s = 99.94M tok / wall-time):

| size | BF16 tok/s | NVFP4 tok/s | NVFP4 slowdown |
|---|---|---|---|
| dim256 | 374,900 | 118,500 | 3.2x |
| dim384 | 287,200 |  73,570 | 3.9x |
| dim512 | 209,100 |  38,060 | 5.5x |
| dim640 | 148,000 | ~22,300 (proj) | ~6.6x |

Aggregate over the sweep: BF16 ~226k tok/s, NVFP4 ~43k tok/s. NVFP4 is SLOWER here and gets
relatively worse with size. VRAM peak 71.6/95.6 GB (73%), power 444-470/600W (76%) - neither
maxed; 100% SM util is quant kernels + skinny GEMMs, not peak FLOPS. The fp32 logits head
[65536 x 50257] (~13-26 GB) dominates VRAM, not params.

WHY (and why batch/seq cannot fix it): the linears are [M=bs*T=65536] x [K=N=dim]. M is huge
and already SM-saturated (~11 waves over 188 SMs); the small dims are K,N=dim=256-640. For the
M-dominant regime, arithmetic intensity AI ~= 0.8*dim FLOP/byte. Ridge point is 1116 FLOP/byte
(2000 TFLOPS / 1792 GB/s), so the FP4 GEMM is compute-bound (where FP4 beats BF16) only at
dim >~ 1400. At dim 512: AI~410 = 37% of ridge -> BW-bound -> FP4 pays the SR/RHT/pack quant
tax for no GEMM payoff -> net loss vs BF16. Scaling batch or sequence only grows M, which
cancels out of AI (no change to the wall) and is already saturated; longer T also adds O(T^2)
BF16 attention that DILUTES the FP4 fraction. The only lever that moves FP4 efficiency is model
dim. This is consistent with the section-6/7 roofline: skinny training shapes are BW-bound by
physics. Takeaway for experiment design: the learning-equivalence sweep (this section) is the
right use of dim 256-640; a throughput-WIN demo needs a separate dim 512/1024/2048/4096 sweep
that crosses the ~dim-1400 compute-bound boundary.

## 9. CUTLASS GEMM wired into FP4Linear (fwd+bwd) + the wgrad-quant fix (2026-06-07)

Goal: route all three training GEMMs (fprop/dgrad/wgrad) through the standalone CUTLASS
sm_120 NVFP4 kernel (section 6) instead of cuBLAS torch._scaled_mm, to close the FP4
wall-clock gap.

NEW FILES (none of the verified .cu/.sh touched):
- cutlass_gemm/cutlass_nvfp4_ext.cu - torch extension; keeps the verified bf16out Gemm
  type byte-for-byte, replaces the host driver with nvfp4_bf16_gemm(A_data,A_sf,B_data,
  B_sf,M,N,K,alpha)->bf16[M,N].
- cutlass_gemm/cutlass_ext.py - torch.utils.cpp_extension.load JIT loader (build_bf16out.sh flags).
- cutlass_gemm/test_parity.py - parity vs cuBLAS _addmm_nvfp4_dispatch.

KEY WIN (the direct-swizzle hypothesis): torchao to_blocked scale swizzle == CUTLASS
Sm1xxBlkScaledConfig::tile_atom_to_shape_SF layout. So the SAME packed qdata + swizzled
scales our quantizer already emits feed CUTLASS directly, no re-layout. alpha =
a.per_tensor_scale * bt.per_tensor_scale, beta=0. Parity rel-L2 (verified): 4096^3 2.85e-3,
8192^3 2.86e-3, 16384x2048x2048 3.53e-3.

Wiring: nvfp4_cuda.py gained _gemm_AB() (NVFP4_CUTLASS=1 -> CUTLASS when M,N,K %128==0,
else cuBLAS fallback); fp4_matmul_cuda/fp4_mm_preqB route through it. nvfp4_train.py: under
NVFP4_CUDA=1, rebind fp4_matmul -> fp4_matmul_cuda so the WGRAD path uses the CUDA quant +
CUTLASS (previously wgrad used nvfp4_gemm.fp4_matmul = pure-torch quant + cuBLAS).
FP4Linear fwd+bwd parity cuBLAS vs CUTLASS backend: y[:3] identical, norms within ~0.2%.

THE DOMINANT WIN WAS THE WGRAD QUANT, NOT THE GEMM. Rebinding fp4_matmul to the CUDA
quant (wgrad was silently on the 23 GB/s pure-torch quant) collapsed NVFP4 from 290ms ->
75ms at dim4096 (~4x). CUTLASS GEMM then adds +4-6% more at compute-bound dims (isolated:
4096 +3.7%, 6144 +4.8%, 8192 +5.9%).

RESULT - NVFP4 now FASTER than BF16 at large dim (block fwd+bwd, B8 T1024, CUTLASS on,
lin TFLOPS): dim4096 nvfp4 251 vs bf16 274 (1.09x slower); dim6144 328 vs 291 (1.13x
FASTER); dim8192 397 vs 305 (1.30x FASTER). Crossover ~dim 5-6k. NVFP4 went from 4.3x
slower (session start, wgrad on pure-torch quant) to 1.3x faster at dim8192. Not the
theoretical 4x: 3 GEMMs each carry quant casts, the FP4 GEMM runs ~79% SOL not 100%, and
the block has bf16 attention/RMSNorm (Amdahl). Toggle: NVFP4_CUDA=1 NVFP4_AMORTIZE=1
NVFP4_CUTLASS=1.

## 10. torch.compile the FP4 block + the quant-overhead profile (2026-06-07)

#1 (DONE, big win): torch.compile the FP4 block (Dynamo graph-breaks around the custom
FP4Linear op, suppress_errors=True, fuses the eager RoPE/SiLU/RMSNorm/residuals between
the linears). Block fwd+bwd, B8 T1024, NVFP4_CUTLASS=1, lin TFLOPS:
  dim4096: nvfp4 eager 251 -> compiled 357 (+42%);  bf16 compiled 295  -> nvfp4 1.21x faster
  dim8192: nvfp4 eager 397 -> compiled 541 (+36%);  bf16 compiled 320  -> nvfp4 1.69x FASTER
Caveat: the _SEED global increments per stochastic quant -> Dynamo guard churn; benign here
(quant graph-breaks anyway) but pass seed via the op, not a guarded global, for real training.

Compiled NVFP4 profile (dim8192, 64ms, self-CUDA %): CUTLASS GEMM 40%, quant SUPPORT ~30%
(to_blocked scale swizzle ~13% = the single biggest non-GEMM kernel; amax reduces 3.6%;
.t().contiguous() transposes; abs/casts), quant_kernel cast ~8.5%, fused model elementwise
~10%, attention ~3%. So post-compile, GEMM (40%) and quant-related (~38%) are ~equal cost.

#2 (target, bounded kernel surgery, NOT yet done): make quant_kernel write scales directly
in the blocked/swizzled layout (kill the ~13% to_blocked gather) + fold per-tensor amax into
the quant kernel. ~+10-15% expected. Needs parity re-validation.
#3 (the architectural fix, multi-day, NOT done): CUTLASS prologue fusion - fold amax+cast+
block-scale+SR+RHT into the GEMM mainloop, eliminating the separate quant kernel + the FP4
HBM round-trip + all support. Subsumes #2; targets the full ~38%. sm_120 has no hw SR cast so
SR stays software. This is the real remaining lever; scoped, not faked.

## 11. Fused SMEM cast-transpose-quant kernel for wgrad (2026-06-07)

The 52ms "quant-support" bucket was NOT the to_blocked swizzle (micro-bench: 0.01ms) - it was
the wgrad operand transposes. NVFP4 block-scales along the contract dim; for wgrad the contract
dim is tokens (non-contiguous), so the old path did x.t().contiguous() (3 HBM passes) then quant.

Two cleanups:
1. aminmax: replaced tf.abs().amax() (full abs temp + 2 launches) with torch.aminmax (1 pass,
   no temp). Numerically identical. -4% (220->211 ms/step), parity still PASS.
2. Fused cast-transpose-quant kernel (cast_transpose_quant.py, NVFP4_FUSED_CT=1 default with
   CUDA backend): SMEM-tiled (TILE 64x64, float4 vec loads, SMEM pad +8 for bank-conflict-free
   transposed reads AND 16B-aligned float4 stores), coalesced load -> SMEM transpose -> per-16-
   block RHT+amax+scale+pack along the now-contiguous token dim -> coalesced FP4 writes. One HBM
   pass instead of three. Reuses the exact quant_kernel per-block math (RHT/SR/e4m3/pack).
   BITWISE-validated: 12/12 cases (3 shapes x 4 SR/RHT combos) match quant(x.t().contiguous())
   100% on qdata AND scale, SR included (RNG index = row-major k*M+m of the [K,M] output, matches
   the reference draws). FP4Linear fwd+bwd identical CT-on vs CT-off (gw.norm 408235.9688 both) -
   a perfect drop-in. Isolated kernel vs transpose+quant: RNE 2.5x, SR+RHT 1.15-1.29x (SR+RHT is
   compute-bound on the 16x16 RHT matmul, not transpose-bound). End-to-end 210->203.6 ms/step
   (-3%; swizzle/copy bucket 52.5->39.3, quant cast 9.4->4.3). Modest e2e because CUTLASS GEMMs
   (~92ms) dominate and some saved transpose time was already overlapped.

Running total (dim4096 block/model): naive NVFP4 4.3x slower -> after wgrad-quant fix + CUTLASS
GEMM (all 3 GEMMs) + torch.compile + aminmax + fused cast-transpose: optimized NVFP4 is faster
than BF16 at the block level and at dim>=6k end-to-end; at dim4096 full-model still ~1.1x slower
than bf16 (Amdahl: bf16 head/attention/2-dense-blocks + the CUTLASS GEMMs dominate). Remaining
lever to flip dim4096 e2e: prologue fusion (#3, fold quant into the GEMM mainloop).

## 12. Stream overlap is empirically dead; warp specialization is the unique remaining lever (2026-06-07)

Tested whether quant can be hidden by running it on a side CUDA stream concurrent with
other work (no custom kernel). Measured on sm_120:
- quant (bandwidth-bound) vs compute-bound FP4 GEMM 8192^3: GEMM 0.791ms + quant 0.202ms,
  concurrent 0.983ms = 1% overlap. Blocked by the OCCUPANCY WALL (the 90%-SM GEMM leaves
  no CTA slot for quant, even though tensor cores leave CUDA-cores + bandwidth idle).
- quant vs memory-bound op (adamw/norm proxy): 0.201+0.732=0.933ms, concurrent 0.933ms =
  0% overlap. Blocked by BANDWIDTH CONTENTION (quant is bandwidth-bound; so are the SM-idle
  phases; they fight over the same HBM bus).

Conclusion (closes the optimization frontier): the quant needs memory bandwidth; the only
place bandwidth is free is DURING the compute-bound GEMM; but that GEMM is occupancy-locked.
So the only way to spend that free bandwidth on quant is to place quant WARPS inside the
GEMMs CTAs = warp specialization within a fused mainloop. Streams operate at CTA granularity and cannot reach it. This is the unique remaining lever, and it is a custom sm120
CollectiveMainloop fork: multi-week, and it risks exceeding sm120's ~101KB smem cap, the
same constraint that broke TE here.

SESSION RESULT BANKED: NVFP4 on sm_120 went from 4.3x slower to FASTER than BF16 (block-level,
and end-to-end at dim>=6k), all bitwise/parity-validated. Path: wgrad-quant fix [4x] + CUTLASS
sm_120 GEMM wired into all 3 GEMMs [+5%] + torch.compile [+36%] + aminmax [-4%] + fused SMEM
cast-transpose-quant [-3%, bitwise drop-in]. At dim4096 full-model still ~1.1x slower than bf16
(Amdahl: bf16 head/attention/dense-blocks + the CUTLASS GEMM are ~45% of the step). The one
remaining lever to flip dim4096 end-to-end is the warp-specialized fused mainloop above -
documented as the frontier, not faked.

## 13. Autonomous low-precision sweep: FP8 is the sweet spot (2026-06-07)
Full results in REPORT.md + SWEEP_LOG.md + sweep_results/. FP8 path built via torchao
convert_to_float8_training (tensorwise e4m3, NO Hadamard/SR - FP8 range absorbs gradient
outliers that force RHT+SR at 4-bit). Findings: (1) precision equivalence ironclad - bf16/
fp8/nvfp4 within ~0.015 val at every size; (2) FP8 ~10% FASTER than BF16 end-to-end at
dim>=1024 (NVFP4 still 1.25-2x slower); (3) low-precision + 8-bit AdamW fits ~2x bigger
model in 96GB (bf16+fp32adam 3.3B -> fp8+8bit 6.47B), FP8 more memory-efficient than NVFP4
(89 vs 99GB peak). Verdict: FP8 = the robust default (faster + 2x memory + identical
convergence + no Hadamard/SR); NVFP4 = aggressive frontier, only wins on speed at very large
dim. New trainer flags: FP8=1 (torchao float8), BNB8=1 (bitsandbytes AdamW8bit).

## 14. Sweep closed: L(N)-decreasing demonstrated, thesis airtight (2026-06-07)
slack4: small-model scaling law at 98M tokens (converged) shows bigger N -> lower L (4.50->
4.21 over N 4.3M->39M), FP8 within ~0.008 of BF16 at every size; fits BF16 L=4.11+4784N^-0.62,
FP8 L=4.13+9492N^-0.66. Combined with P2 (FP8 fits 2x bigger model), the capability-per-budget
thesis is now empirically complete. 7 validated experiments total in REPORT.md + sweep_results/.
VERDICT: FP8 (tensorwise e4m3) = the default for training a bigger/better model on one 96GB
sm_120 card - identical convergence, ~10% faster, ~2x memory, no Hadamard/SR. NVFP4 = aggressive
frontier (more memory ceiling, but slower until very large dim + fragile + custom kernels).

## 15. slack5: NVFP4 does not cross BF16 end-to-end even at 2.3B (Amdahl) (2026-06-07)
Large-dim throughput: fp8/bf16 1.01 (dim5120) -> 0.94 (dim6144); nvfp4/bf16 1.26 -> 1.13 (still
slower). The block-level NVFP4>BF16 crossover (~dim6k) does NOT translate end-to-end at 2.3B -
bf16 head/attention/embedding dilute it. FP8 at/below parity, equivalence holds. Sweep complete:
8 validated experiments, FP8 tensorwise e4m3 = the verdict.

## 16. CUTLASS per-shape tuning + overnight NVFP4+Muon pretrain (2026-06-08)
Tuned bf16-out NVFP4 GEMM across 9 training shapes (dim2048, M=16384). KEY: the "only
128x128x128" constraint is for the FP4-OUT kernel; the BF16-OUT kernel compiles 256x128x128
Cooperative (+128x64/32, 128x256) -> 256x128 Coop wins 5/9 shapes. StreamK (added CFG_STREAMK,
GemmUniversal StreamKScheduler) lifts the 2 occupancy-starved wgrad shapes (qo_wgrad 56->75%,
kv_wgrad 58->67%). Results: 2 shapes hit 86%, most 70-80%; M-heavy small-K shapes cap ~71%
(sm_120 1x1x1-cluster wall - 85% physically unreachable there). Full table: cutlass_gemm/
SHAPE_TUNING.md. Wgrad/big-shape wins +14-33% over baseline. Training extension switched to
256x128x128 Coop (parity rel_L2 3e-3 PASS). NOTE: single-config (256x128 Coop) in the extension
captures the big wins; per-shape dispatch + StreamK for shapes 6,7 left as future work.
Overnight: NVFP4 + tuned CUTLASS + Muon(lr0.01) + compile, dim2048/nl12 (554M), bs32 T512,
5.5h bound, incremental traj. Launching now.
