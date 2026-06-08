# FINDINGS — Low-precision LLM training on one RTX PRO 6000 (sm_120, 96GB)

Durable synthesis of the 2026-06-06..08 investigation. The chronological journey is in
DEVLOG.md (§1-16); the FP8/NVFP4 sweep detail is in REPORT.md; kernel tuning in
cutlass_gemm/SHAPE_TUNING.md; raw numbers + figures in sweep_results/.

## The one-paragraph version
4-bit (NVFP4) training works on consumer Blackwell where NVIDIA's Transformer Engine
doesn't, and with this session's kernel + optimizer work it goes from **4.3x slower than
BF16 to faster** at scale — but **FP8 is the robust sweet spot** (no Hadamard/SR, ~10%
faster, ~2x more memory-efficient, identical convergence). The real value of low precision
on a VRAM-budgeted box is **capability per budget** (fit a ~2x bigger model), not speed at
fixed size. End-to-end MFU tops out at **~42%** (which is the Llama-3 frontier range, not
the 90% one might hope) and frontier-6.5B-from-scratch is a **cluster-months FLOP problem**
that no precision trick closes. The validated stack (NVFP4/FP8 + tuned CUTLASS + Muon) is a
real foundation for **fine-tuning big checkpoints** or **fully training small models**.

## Key results (all from green/validated runs)
1. **NVFP4 training validated.** Custom quant (SR + in-kernel RHT + 2-level scale) + native
   FP4 GEMM. Bitwise-clean kernels (quant, cast-transpose-quant, CUTLASS GEMM). Converges
   identically to BF16 at every model size (precision-equivalence ironclad, ~0.01 val).
2. **FP8 is the sweet spot.** torchao tensorwise e4m3, NO Hadamard/SR (FP8's range absorbs
   gradient outliers that force RHT+SR at 4-bit). ~10% faster than BF16 end-to-end at
   dim>=1024; tensorwise > rowwise (rowwise 60% slower, no accuracy gain).
3. **Capability per VRAM.** FP8/NVFP4 + 8-bit AdamW fits ~2x bigger model in 96GB
   (bf16+fp32adam 3.3B -> fp8+8bit 6.5B). 8-bit optimizer is the dominant lever (+44%);
   precision adds ~36%. FP8 more memory-efficient than NVFP4 (89 vs 99GB at same size).
4. **MFU reality: ~42%, and that's the frontier number.** Llama-3 405B is 38-43% MFU, PaLM
   46%; "above 50% is challenging." The 50-60% often quoted is HFU (counts recompute). Our
   42% breaks down as 0.74 (time in GEMMs) x 0.57 (GEMM efficiency); the cap is ~24% of every
   step being bandwidth-bound non-GEMM (norms/optimizer) + the FP8 cast, worsened by consumer
   bandwidth (1.8 vs 3.3 TB/s). 90% is not real; ~50% needs scale + the warp-specialized
   fused mainloop (the one hard multi-week lever streams provably can't substitute - measured
   1% overlap on compute-bound, 0% on memory-bound).
5. **Muon optimizer: validated token-efficiency win.** Tuned (lr~0.01, sharp/sensitive) beats
   AdamW by 0.32 val at fixed tokens (well past the ~15% headline). On 2D block weights;
   AdamW on emb/head/norms.
6. **CUTLASS per-shape tuning.** The bf16-OUT kernel compiles 256x128x128 Cooperative + StreamK
   (the "only 128x128x128" constraint was the FP4-OUT kernel). Wgrad/big shapes +14-33%;
   2 shapes hit 86%; M-heavy small-K shapes cap ~71% (sm_120 1x1x1-cluster wall, 85%
   physically unreachable there). 256x128 Coop wired into training, parity-validated.
7. **Overnight pretrain (end-to-end proof).** 554M NVFP4 + tuned-CUTLASS + Muon + compile,
   693M tokens in 5.5h at ~36k tok/s, stable, loss 11.2 -> val 3.32, train≈val, still
   descending. (Under-annealed: cosine schedule set for 200k steps but only ran 42k -> LR
   never decayed -> 3.32 is pessimistic, properly-annealed ~3.0-3.1.)

## Strategic verdict
- **Use FP8** (tensorwise e4m3) as the default for training a bigger/better model on this card.
- **NVFP4** is the aggressive memory frontier; only worth its RHT/SR fragility + custom kernels
  for the last factor of memory, and it doesn't win on end-to-end speed until very large dim.
- **Don't expect >42% MFU** on one consumer card; that IS the frontier number.
- **Frontier-6.5B-from-scratch is months on one GPU** (FLOP-bound, not memory). The 96GB +
  low-precision win unlocks (a) fine-tuning/continued-pretraining a big pretrained checkpoint
  (a few B tokens = days), and (b) fully training a ~0.5-1B model.

## What would make a real run worth committing days to
1. Fix the LR schedule (anneal to the actual token budget) - free ~0.1-0.3 loss.
2. Muon + FP8 + 8-bit optimizer + FineWeb-grade data (~3-5x stacked token-efficiency).
3. A ~0.5-1B model to compute-optimal (~10-20B tokens ≈ days here) is the realistic target.
4. Bigger lever than more kernel tuning: the warp-specialized fused mainloop (hides the FP8
   cast under the GEMM) - the only thing that lifts MFU past ~45% on this card, and it's a
   fixed-architecture multi-week CUDA project.

## Not done / open
- No model weights checkpointed (runs were exploratory; loss-trajectory only).
- L(N)-decreasing shown only at small converged scale; large models rely on published scaling.
- FP4 activation storage + per-shape kernel dispatch (incl StreamK for the 2 wgrad shapes) +
  the warp-specialized mainloop are the named next levers.
