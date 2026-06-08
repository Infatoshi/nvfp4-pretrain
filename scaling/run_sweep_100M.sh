#!/bin/bash
# 100M-token-per-model iso-token scaling sweep: BF16 vs NVFP4 on OpenWebText.
# 5x the original 20M-token run (305 steps). Writes to a NEW results file; does
# not touch the original scaling_results.jsonl.
cd /home/infatoshi/experiments/_scratch/nvfp4-validate/scaling
unset LD_PRELOAD
export CUDA_HOME=/usr/local/cuda-13
OUT=/home/infatoshi/data/scaling_results_100M.jsonl
rm -f "$OUT"
# bs128 x T512 = 65536 tok/step ; 1525 steps = 99.9M tokens/model
STEPS=1525
WARMUP=150
LR=6e-4
for cfg in "256 6 4" "384 6 6" "512 8 8" "640 10 10"; do
  read d l h <<< "$cfg"
  echo "=== BF16 dim=$d $(date) ==="
  python3 -u train_text.py --dim $d --nl $l --nh $h --nkv $h --T 512 --bs 128 \
    --steps $STEPS --warmup $WARMUP --lr $LR --tag bf16_$d --out "$OUT"
  echo "=== NVFP4 dim=$d $(date) ==="
  NVFP4_CUDA=1 python3 -u train_text.py --dim $d --nl $l --nh $h --nkv $h --T 512 --bs 128 \
    --steps $STEPS --warmup $WARMUP --lr $LR --tag nvfp4_$d --out "$OUT"
done
echo "=== SWEEP DONE $(date) ==="
cat "$OUT"
