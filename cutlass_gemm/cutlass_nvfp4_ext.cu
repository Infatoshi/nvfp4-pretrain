// Python-callable torch extension wrapping the verified SM120 NVFP4xNVFP4 -> bf16
// CUTLASS GEMM. The type-definition block below is copied UNCHANGED from the
// verified standalone benchmark nvfp4_gemm_bf16out.cu (Gemm / CollectiveMainloop /
// CollectiveEpilogue / strides / layouts). Only the host-side driver (Options,
// random initialize, verify, main) is removed and replaced by a torch entry point.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <cmath>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/packed_stride.hpp"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations  (UNCHANGED from nvfp4_gemm_bf16out.cu)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using         LayoutATag  = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 32;

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using         LayoutBTag  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 32;

// C/D matrix configuration -- BF16 output, NO scale-factor generation.
using         ElementD    = cutlass::bfloat16_t;
using         ElementC    = cutlass::bfloat16_t;
using         LayoutCTag  = cutlass::layout::RowMajor;
using         LayoutDTag  = cutlass::layout::RowMajor;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;

// Kernel functional config
using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm120;
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;

#ifndef CFG_TILE_M
#define CFG_TILE_M 128
#endif
#ifndef CFG_TILE_N
#define CFG_TILE_N 128
#endif
#ifndef CFG_TILE_K
#define CFG_TILE_K 128
#endif
#ifndef CFG_SCHED
#define CFG_SCHED 0   // 0 = Pingpong, 1 = Cooperative, 2 = KernelScheduleAuto
#endif
#ifndef CFG_STAGES
#define CFG_STAGES 0  // 0 = StageCountAutoCarveout; >0 = fixed StageCount<N>
#endif

using ThreadBlockShape    = Shape<cute::Int<CFG_TILE_M>, cute::Int<CFG_TILE_N>, cute::Int<CFG_TILE_K>>;
using ClusterShape        = Shape<_1,_1,_1>;
#if CFG_SCHED == 0
using KernelMainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
#elif CFG_SCHED == 1
using KernelMainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
#else
using KernelMainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
#endif

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

#if CFG_STAGES > 0
using StageCountType = cutlass::gemm::collective::StageCount<CFG_STAGES>;
#else
using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
#endif

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    StageCountType,
    KernelMainloopSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;

// Scale-factor element type carried by nv_float4_t (ue4m3). Use it directly so
// we never have to guess between float_ue4m3_t / float_e4m3_t.
using ElementSF = typename ElementA::ScaleFactorType;
using ElementAData = typename ElementA::DataType;  // float_e2m1_t
using ElementBData = typename ElementB::DataType;  // float_e2m1_t

/////////////////////////////////////////////////////////////////////////////////////////////////

static void check_cutlass(cutlass::Status status, const char* what) {
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS ", what, " failed: ", cutlassGetStatusString(status));
}

// nvfp4_bf16_gemm: D[M,N] (bf16) = alpha * (A_nvfp4 @ B_nvfp4)
//   A_data : packed e2m1 [M, K/2] uint8 (RowMajor logical [M,K])
//   A_sf   : swizzled e4m3 scale factors for A (torchao "blocked" layout)
//   B_data : packed e2m1 [N, K/2] uint8 (ColumnMajor logical [K,N] == storage [N,K])
//   B_sf   : swizzled e4m3 scale factors for B
at::Tensor nvfp4_bf16_gemm(at::Tensor A_data, at::Tensor A_sf,
                           at::Tensor B_data, at::Tensor B_sf,
                           int64_t M, int64_t N, int64_t K, double alpha) {
  TORCH_CHECK(A_data.is_cuda() && B_data.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 128 == 0,
              "M,N,K must be multiples of 128");

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  // Precondition: torchao swizzled scale element count must match the CUTLASS
  // scale-factor layout footprint (early swizzle-divergence signal).
  auto sfa_need = size(filter_zeros(layout_SFA));
  auto sfb_need = size(filter_zeros(layout_SFB));
  TORCH_CHECK(A_sf.numel() == sfa_need,
              "A_sf numel ", A_sf.numel(), " != expected ", sfa_need);
  TORCH_CHECK(B_sf.numel() == sfb_need,
              "B_sf numel ", B_sf.numel(), " != expected ", sfb_need);

  auto D = at::empty({M, N}, A_data.options().dtype(at::kBFloat16));

  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    { // Mainloop arguments
      reinterpret_cast<ElementAData const*>(A_data.data_ptr()), stride_A,
      reinterpret_cast<ElementBData const*>(B_data.data_ptr()), stride_B,
      reinterpret_cast<ElementSF const*>(A_sf.data_ptr()), layout_SFA,
      reinterpret_cast<ElementSF const*>(B_sf.data_ptr()), layout_SFB
    },
    { // Epilogue arguments: alpha * acc + beta * C, C unused (beta=0)
      {static_cast<ElementAccumulator>(alpha), ElementAccumulator(0)},
      nullptr, stride_C,
      reinterpret_cast<ElementD*>(D.data_ptr()), stride_D
    }
  };

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                             A_data.options().dtype(at::kByte));

  check_cutlass(gemm.can_implement(arguments), "can_implement");
  check_cutlass(gemm.initialize(arguments, workspace.data_ptr()), "initialize");
  check_cutlass(gemm.run(at::cuda::getCurrentCUDAStream()), "run");

  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_bf16_gemm", &nvfp4_bf16_gemm,
        "SM120 NVFP4xNVFP4 -> bf16 GEMM (CUTLASS)");
}
