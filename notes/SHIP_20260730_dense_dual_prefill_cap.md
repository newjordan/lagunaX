# Research note — Dense dual prefill ncols_dst cap 32→2048 (2026-07-30)

## Status: **REVERTED** (golden OK; large pp regress)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip prefill moe-down weighted** | **3148.4** | **128.9** | **+47.88%** | OK |
| dense dual cap 2048 (pp shexp dual) | 2627.5 | 128.4 | +40.92% | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal loser: `results/20260730T091435Z/`

## What

Raise dense dual SwiGLU `ncols_dst` cap **32 → 2048** (fuse + launch) so shared-expert
gate+up+swiglu dual hits full serial pp512 (was decode + tiny prefill only).

Kernel already multi-col (`group(0)=col`); cap was the only gate.

Hit would show: `[lx-control-dense-dual] ... ncols_dst=512`

## Findings

- **Golden OK** (short prompt already under old cap 32; same dual path).
- **Prefill −521 t/s** vs tip: multi-col reorder-MMVQ dual for T=512 is much slower
  than stock oneDNN/MMQ GEMM + separate GLU for shexp.
- Decode flat (ncols_dst=1 path unchanged).
- **Keep cap 32.** Do not default dual multi-col beyond stock MMVQ/MMQ sweet spot
  without a GEMM dual path.

## Tip unchanged

Reverted. Scored tip remains prefill moe-down weighted (`20260730T085918Z`, +47.88%).

Cosmetic kept: dense dual hit log prints `ncols_dst`.

## Next

1. Multi-token MoE dual via **expert-batched GEMM** (match stock regroup), not per-token MMVQ.
2. lm_head (beyond multi-sg pack which already regressed).
3. Avoid re-raising dense dual prefill cap without GEMM dual.
