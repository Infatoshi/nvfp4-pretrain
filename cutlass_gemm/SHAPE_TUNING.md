# NVFP4 bf16-output GEMM: per-shape tuning on sm_120 (RTX PRO 6000 Blackwell)

Kernel: `nvfp4_gemm_bf16out.cu` (NVFP4 e2m1 + ue4m3 per-16 block scale inputs -> BF16 output).
Split-K variant: `nvfp4_gemm_bf16out_splitk.cu` (adds StreamK tile scheduler, `--splits`/`--decomp`).
Card: dense FP4 peak 2000 TFLOPS, GDDR7 BW 1792 GB/s, ridge 1116 FLOP/byte.
All numbers are GPU-idle (nvidia-smi 0% util, no other compute apps), 100 iters,
CUDA-event median, and ONLY counted when `Disposition: Passed` (bitwise/rel-L2 verified
vs CUTLASS host reference).

## Roofline (recomputed for THESE 9 shapes; byte model = 0.5625*(MK+KN) + 2*MN)

8 of 9 shapes are COMPUTE-bound (ceiling 2000 TFLOPS, 85% target = 1700).
Only shape 3 (16384x2048x1024) is BW-bound (ceiling 1584, 85% target = 1347).
This is the OPPOSITE of the old DEVLOG training shapes (which were mostly BW-bound);
do not reuse that table.

| # | M | N | K | role | AI | bound | ceiling TFLOPS | 85% target | tiles@128 | waves |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16384 | 2048 | 2048  | q/o fprop+dgrad | 1556 | comp | 2000 | 1700 | 2048 | 10.9 |
| 2 | 16384 | 1024 | 2048  | k/v fprop       | 1282 | comp | 2000 | 1700 | 1024 | 5.4 |
| 3 | 16384 | 2048 | 1024  | k/v dgrad       |  884 | BW   | 1584 | 1347 | 2048 | 10.9 |
| 4 | 16384 | 8192 | 2048  | up fprop        | 1853 | comp | 2000 | 1700 | 8192 | 43.6 |
| 5 | 16384 | 2048 | 8192  | down fprop      | 3616 | comp | 2000 | 1700 | 2048 | 10.9 |
| 6 | 2048  | 2048 | 16384 | q/o wgrad       | 2979 | comp | 2000 | 1700 |  256 | 1.36 |
| 7 | 1024  | 2048 | 16384 | k/v wgrad       | 2114 | comp | 2000 | 1700 |  128 | 0.68 |
| 8 | 8192  | 2048 | 16384 | up wgrad        | 4297 | comp | 2000 | 1700 | 1024 | 5.4 |
| 9 | 2048  | 8192 | 16384 | down wgrad      | 4297 | comp | 2000 | 1700 | 1024 | 5.4 |

Key occupancy insight: only shapes 6 (1.36 waves) and 7 (0.68 waves, <1 full wave!)
have a tile-count/wave-quantization problem. Shapes 8,9 already fill the 188 SMs (5.4
waves); split-K is NOT expected to help them. So split-K is targeted at shapes 6 and 7.

## Compile space on sm_120 bf16-out (the REAL constraint)

Unlike the FP4-OUTPUT kernel (which static_asserts on everything but 128x128x128+Pingpong),
the bf16-output kernel is much LESS constrained. Verified compiles:

| tile | Pingpong(s0) | Cooperative(s1) | Auto(s2) |
|---|---|---|---|
| 128x128x128 | YES | YES | YES |
| 128x64x128  | YES | YES | - |
| 128x32x128  | YES | - | - |
| 256x128x128 | YES | YES | - |
| 128x256x128 | YES | - | - |

Stages: st0(auto), st4 compile+run OK; st6, st8 build but ERROR at runtime on the
128x128x128 tile (SMEM carveout overflow) -> auto/4 are the usable stage counts.

StreamK (split-K) scheduler: compiles ONLY with Cooperative (s1). Pingpong hard-asserts
"Ping-pong kernel does not currently support stream-K scheduler." So split-K binaries
are Cooperative-only.

## Best PLAIN (data-parallel) config per shape (verify=0 timing; configs pre-verified Passed)

Methodology note: each tile/sched config was bitwise/rel-L2 verified (Disposition: Passed)
on the M=16384 small-K shapes. The MMA + fp32-accumulate path is shape-independent, so
timing for these same configs at larger shapes was taken with --verify=0 (the host
O(n^3) reference is impractical at K=16384). Winners re-verified at real shape (below).

| # | shape (MxNxK) | best plain cfg | TFLOPS | % applicable ceiling | % of 2000 |
|---|---|---|---|---|---|
| 1 | 16384x2048x2048  | 128x128x128 s0 (Pingpong) st0 | 1419 | 71.0% (comp) | 71.0% |
| 2 | 16384x1024x2048  | 256x128x128 s1 (Coop) st0     | 1414 | 70.7% (comp) | 70.7% |
| 3 | 16384x2048x1024  | 128x128x128 s0 st4            | 1253 | 79.1% (BW)   | 62.6% |
| 4 | 16384x8192x2048  | 256x128x128 s1 (Coop) st0     | 1493 | 74.7% (comp) | 74.7% |
| 5 | 16384x2048x8192  | 256x128x128 s1 (Coop) st0     | 1608 | 80.4% (comp) | 80.4% |
| 6 | 2048x2048x16384  | 256x128x128 s1 (Coop) st0     | 1333 | 66.7% (comp) | 66.7% | (beaten by StreamK 1499, below) |
| 7 | 1024x2048x16384  | 128x128x128 s1 (Coop) st0     | 1275 | 63.7% (comp) | 63.7% | (beaten by StreamK 1348, below) |
| 8 | 8192x2048x16384  | 256x128x128 s1 (Coop) st0     | 1720 | 86.0% (comp) | 86.0% |
| 9 | 2048x8192x16384  | 256x128x128 s1 (Coop) st0     | 1720 | 86.0% (comp) | 86.0% |

(This table is PLAIN data-parallel configs only. Shapes 6 and 7 are improved further by
StreamK -- see the split-K section. The FINAL table below reflects the true per-shape best.)

Headline config finding: the **256x128x128 Cooperative** tile (which the FP4-OUT kernel
cannot even compile) is the single best config on 5 of 9 shapes and is the key win over
the forced 128x128 Pingpong baseline. It needs Cooperative; with Pingpong the 256-dim
tiles run at ~150-170 TFLOPS (broken/no warp-specialization overlap).

## Build commands for the winning binaries

```
# --- plain (data-parallel) binaries, win shapes 1-5, 8, 9 ---
bash build_bf16out.sh 128 128 128 0 0     # ..._128x128x128_s0_st0   Pingpong  shape 1
bash build_bf16out.sh 128 128 128 0 4     # ..._128x128x128_s0_st4   Pingpong  shape 3
bash build_bf16out.sh 256 128 128 1 0     # ..._256x128x128_s1_st0   Coop      shapes 2,4,5,8,9
# --- StreamK binaries (Cooperative only; Pingpong asserts), win shapes 6, 7 ---
bash build_splitk.sh 256 128 128 1 0 1    # ..._splitk_256x128x128_s1_st0_sk1   shape 6
bash build_splitk.sh 128 128 128 1 0 1    # ..._splitk_128x128x128_s1_st0_sk1   shape 7

# run commands for each shape's winner:
./nvfp4_gemm_bf16out_128x128x128_s0_st0 --m=16384 --n=2048 --k=2048  --iterations=100   # 1
./nvfp4_gemm_bf16out_256x128x128_s1_st0 --m=16384 --n=1024 --k=2048  --iterations=100   # 2
./nvfp4_gemm_bf16out_128x128x128_s0_st4 --m=16384 --n=2048 --k=1024  --iterations=100   # 3
./nvfp4_gemm_bf16out_256x128x128_s1_st0 --m=16384 --n=8192 --k=2048  --iterations=100   # 4
./nvfp4_gemm_bf16out_256x128x128_s1_st0 --m=16384 --n=2048 --k=8192  --iterations=100   # 5
./nvfp4_gemm_bf16out_splitk_256x128x128_s1_st0_sk1 --m=2048 --n=2048 --k=16384 --iterations=100 --splits=1 --decomp=2  # 6 StreamK
./nvfp4_gemm_bf16out_splitk_128x128x128_s1_st0_sk1 --m=1024 --n=2048 --k=16384 --iterations=100 --splits=1 --decomp=2  # 7 StreamK
./nvfp4_gemm_bf16out_256x128x128_s1_st0 --m=8192 --n=2048 --k=16384 --iterations=100   # 8
./nvfp4_gemm_bf16out_256x128x128_s1_st0 --m=2048 --n=8192 --k=16384 --iterations=100   # 9
# StreamK NOTE: decomp=2 (pure StreamK) MUST use --splits=1; an explicit split count hangs it.
```

## FINAL per-shape results (clean, GPU idle, all winners verified Passed at real shape)

| # | shape (MxNxK) | best config | TFLOPS | applicable ceiling | % of ceiling | % of 2000 | bound | verified |
|---|---|---|---|---|---|---|---|---|
| 1 | 16384x2048x2048  | 128x128x128 Pingpong(s0) st0       | 1421 | 2000 | 71.0% | 71.0% | comp | Passed |
| 2 | 16384x1024x2048  | 256x128x128 Coop(s1) st0           | 1414 | 2000 | 70.7% | 70.7% | comp | Passed |
| 3 | 16384x2048x1024  | 128x128x128 Pingpong(s0) st4       | 1254 | 1584 | 79.2% | 62.7% | BW   | Passed |
| 4 | 16384x8192x2048  | 256x128x128 Coop(s1) st0           | 1496 | 2000 | 74.8% | 74.8% | comp | Passed |
| 5 | 16384x2048x8192  | 256x128x128 Coop(s1) st0           | 1608 | 2000 | 80.4% | 80.4% | comp | Passed |
| 6 | 2048x2048x16384  | 256x128x128 Coop StreamK(decomp=2) | 1499 | 2000 | 75.0% | 75.0% | comp | Passed |
| 7 | 1024x2048x16384  | 128x128x128 Coop StreamK(decomp=2) | 1348 | 2000 | 67.4% | 67.4% | comp | Passed |
| 8 | 8192x2048x16384  | 256x128x128 Coop(s1) st0           | 1720 | 2000 | 86.0% | 86.0% | comp | Passed |
| 9 | 2048x8192x16384  | 256x128x128 Coop(s1) st0           | 1723 | 2000 | 86.1% | 86.1% | comp | Passed |

## Did split-K / stream-K help the wgrad shapes? How it was wired. (YES, on shapes 6 and 7)

WIRING (contained, in a SEPARATE copy `nvfp4_gemm_bf16out_splitk.cu`; original .cu untouched):
- New macro `CFG_STREAMK`: when 1, the 4th template arg of `GemmUniversal` is set to
  `cutlass::gemm::StreamKScheduler` instead of `void` (data-parallel). On sm_120 this maps
  to `PersistentTileSchedulerSm100StreamK`.
- New runtime args `--splits=<N>` and `--decomp=<0..3>` (0=Heuristic,1=SplitK,2=StreamK,
  3=DataParallel) wired into `arguments.scheduler.splits` / `.decomposition_mode`, plus
  `arguments.hw_info.sm_count` queried from the device.
- CRITICAL fix #1: StreamK keeps barrier/reduction flags in its workspace (8.39 MB here).
  The 2nd+ `gemm.run()` spins on stale flags unless reset; added a per-launch
  `cudaMemset(workspace,0,size)` inside the warmup+timing loops (guarded by CFG_STREAMK; the
  data-parallel path is unchanged). This per-call reset cost is included in timing (honest).
- CRITICAL fix #2: pure StreamK (decomp=2) must be run with `--splits=1` (its default). It
  computes its OWN work distribution; passing an explicit split count alongside it conflicts
  and hangs. With `--splits=1 --decomp=2` it runs and is the best split mode.
- Build: `build_splitk.sh TM TN TK SCHED STAGES STREAMK`. StreamK compiles ONLY with
  Cooperative (SCHED=1); Pingpong hard-asserts "Ping-pong does not support stream-K".

CORRECTNESS: bitwise-exact. Verified at small shape AND at the real shapes 6 and 7 with the
winning StreamK config: rel_L2=0, Disposition: Passed.

RESULT (measured, verify=0, 100 iters, GPU idle):
| shape | plain best | SplitK best (decomp=1) | StreamK (decomp=2) | winner | gain over plain |
|---|---|---|---|---|---|
| 6 (2048x2048x16384) | 1334 (256x128 Coop) | 1361 (128x128 sp2) | **1499 (256x128)** | StreamK | **+12.4%** |
| 7 (1024x2048x16384) | 1275 (128x128 Coop) | 1156 (128x128 sp4) | **1348 (128x128)** | StreamK | **+5.7%** |

Pure StreamK (decomp=2) is the only split mode that wins: it does cluster-launch-control
dynamic work-stealing across the 188 SMs, fixing the wave-quantization of these low-tile
shapes (6 = 1.36 waves, 7 = 0.68 waves) without the explicit-split reduction tax that made
SplitK (decomp=1) marginal-or-negative. Shapes 8,9 (5.4 waves) already saturate the SMs and
were not split. So split-K/stream-K helped exactly the two shapes the roofline predicted it
would (the occupancy-starved ones), and StreamK lifted shape 6 from 66.7% to 75.0% SoL.

## Which configs compile on sm_120 bf16-out (the REAL constraint, corrected)

The DEVLOG warning ("only 128x128x128 + Pingpong compiles") is for the FP4-OUTPUT kernel and
is FALSE for this bf16-output kernel. Confirmed-compiling here:
- Tiles: 128x128x128, 128x64x128, 128x32x128, 256x128x128, 128x256x128.
- Schedules: Pingpong(0), Cooperative(1), Auto(2) all compile.
- 256-dim-N tiles REQUIRE Cooperative to run well (with Pingpong they run ~150 TFLOPS).
- Stages: auto(0) and 4 work; 6 and 8 build but error at runtime (SMEM carveout overflow).
- StreamK scheduler: compiles only with Cooperative.
The single most valuable config is 256x128x128 Cooperative (unavailable on the FP4-out kernel).

## Honest assessment: which shapes hit ~85%, which cap and why

- HIT ~85%: shapes 8 (86.0%) and 9 (86.1%) -- large K-heavy wgrad shapes with enough tiles
  to fill the SMs; 256x128 Cooperative reaches the target.
- NEAR (79-80%): shape 5 (80.4%, comp) and shape 3 (79.2% of its BW ceiling). Shape 3 is the
  ONLY bandwidth-bound shape, so 85%-of-2000 is physically impossible; 79% of its 1584 ceiling
  is the honest story.
- LIFTED BY STREAMK: shape 6 (66.7% -> 75.0%) and shape 7 (63.8% -> 67.4%). These small-M/N,
  K=16384 shapes are occupancy-starved (128-256 tiles vs 188 SMs); StreamK recovers part of
  the gap but cannot reach 85% -- there is not enough work to both fill the SMs and amortize
  the cross-CTA reduction on a compute-bound problem.
- CAPPED ~70-75% (compute-bound but kernel-limited on sm_120): shapes 1 (71%), 2 (71%),
  4 (75%). No available config (tile, schedule, stage, split-K) exceeds ~1420-1496 on them.
  This is the same ~70-80% wall DEVLOG documented for square shapes. The GeForce sm_120
  blockscaled mainloop with the forced 1x1x1 cluster (no TMA multicast) caps here; 85% (1700)
  is not reachable on these shapes with the knobs this kernel exposes.

Summary: 2 of 9 shapes hit the 85% target (8,9); shape 5 and shape 3 (at its BW ceiling) sit
at ~79-80%; shapes 6,7 are lifted to 67-75% by StreamK; shapes 1,2,4 cap at 71-75% on the
sm_120 GeForce blockscaled mainloop ceiling. Two realized levers over the
forced-128x128-Pingpong baseline: (a) 256x128 Cooperative (e.g. shape 8: 1505 -> 1720, +14%),
and (b) StreamK on the occupancy-starved wgrad shapes 6,7 (+12% / +6%).
