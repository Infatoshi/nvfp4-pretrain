#!/usr/bin/env bash
set -euo pipefail
unset LD_PRELOAD || true
export CUDA_HOME=/usr/local/cuda-13
NVCC=/usr/local/cuda-13/bin/nvcc
CUT=/home/infatoshi/cuda/engines/cutlass
DIR=/home/infatoshi/experiments/_scratch/nvfp4-validate/cutlass_gemm
TM=${1:-128}; TN=${2:-128}; TK=${3:-128}; SC=${4:-0}; ST=${5:-0}; SK=${6:-1}
OUT=$DIR/nvfp4_gemm_bf16out_splitk_${TM}x${TN}x${TK}_s${SC}_st${ST}_sk${SK}
cd "$DIR"
$NVCC nvfp4_gemm_bf16out_splitk.cu -o "$OUT" \
  -std=c++17 -O3 \
  -gencode arch=compute_120a,code=sm_120a \
  --expt-relaxed-constexpr --expt-extended-lambda \
  --threads 0 \
  -I$CUT/include -I$CUT/tools/util/include -I$CUT/examples/common \
  -DCFG_TILE_M=$TM -DCFG_TILE_N=$TN -DCFG_TILE_K=$TK -DCFG_SCHED=$SC -DCFG_STAGES=$ST -DCFG_STREAMK=$SK
echo "BUILT $OUT"
