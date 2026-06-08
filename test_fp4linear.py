"""FP4Linear fwd+bwd under the active GEMM backend (cuBLAS vs CUTLASS via NVFP4_CUTLASS).
Run twice (NVFP4_CUTLASS=0 then =1) with identical seeds; outputs/grads should match
within bf16+SR noise. Prints norms + leading values for comparison."""
import torch, os
torch.manual_seed(0)
from nvfp4_train import FP4Linear, mark_step

D = 2048
lin = FP4Linear(D, D).cuda()
torch.manual_seed(1)
x = torch.randn(4096, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
mark_step()
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = lin(x)
loss = (y.float() ** 2).sum()
loss.backward()
tag = "CUTLASS" if os.environ.get("NVFP4_CUTLASS") == "1" else "cuBLAS"
print(f"[{tag}] y.norm={y.float().norm():.4f} y[:3]={[round(v,3) for v in y.flatten()[:3].float().tolist()]}")
print(f"[{tag}] gx.norm={x.grad.float().norm():.4f}  gw.norm={lin.weight.grad.float().norm():.4f}")
