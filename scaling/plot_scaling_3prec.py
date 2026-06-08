"""P3: cross-precision scaling law. Reads a jsonl with {backend,N,val_loss} rows for
bf16/fp8/nvfp4 across model sizes; fits L(N)=E+A*N^-alpha per precision; plots all three.
Shows whether FP8 and NVFP4 preserve the BF16 L(N) curve."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RES = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p3_results.jsonl"
rows = [json.loads(l) for l in open(RES) if l.strip()]
by = {}
for r in rows:
    by.setdefault(r["backend"], []).append((r["N"], r["val_loss"]))
for k in by:
    by[k].sort()

def powlaw(N, E, A, al):
    return E + A * np.power(N, -al)

C = {"bf16": "#1f77b4", "fp8": "#2ca02c", "nvfp4": "#d62728"}
fig, ax = plt.subplots(figsize=(8, 6))
allN = np.array([p[0] for v in by.values() for p in v], float)
xs = np.logspace(np.log10(allN.min() * 0.85), np.log10(allN.max() * 1.25), 120)
for be, pts in by.items():
    N = np.array([p[0] for p in pts], float)
    L = np.array([p[1] for p in pts], float)
    c = C.get(be, "#555")
    ax.scatter(N, L, s=70, color=c, zorder=3, label=f"{be}")
    try:
        p, _ = curve_fit(powlaw, N, L, p0=[min(L) * 0.8, 1e3, 0.4], maxfev=100000,
                         bounds=([0, 0, 0.01], [10, 1e7, 1.0]))
        ax.plot(xs, powlaw(xs, *p), "--", color=c, alpha=0.8,
                label=f"  {be} fit: E={p[0]:.2f} a={p[2]:.3f}")
        print(f"{be}: E={p[0]:.3f} A={p[1]:.0f} alpha={p[2]:.4f}  pts={[round(l,3) for _,l in pts]}")
    except Exception as e:
        print(f"{be} fit failed: {e}")
ax.set_xscale("log")
ax.set_xlabel("non-embedding parameters N")
ax.set_ylabel("OpenWebText val loss (GPT-2 BPE)")
ax.set_title("Cross-precision scaling law: BF16 vs FP8 vs NVFP4 (RTX PRO 6000, sm_120)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
out = "/tmp/scaling_law_3precision.png"
fig.savefig(out, dpi=140)
print("SAVED", out)
