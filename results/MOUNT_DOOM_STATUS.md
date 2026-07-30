# Mount Doom status — 2026-07-30 (packed reduce tip)

## LIVE NOW

**Scored tip:** dual+down + topk+gather+sum + down sgs=8 + **expert-loop packed reduce**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip packed reduce** | **3540.3** | **130.2** | **+53.41%** |
| prior down sgs=8 | 3402.1 | 130.5 | +52.22% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T115251Z/` · `notes/SHIP_20260730_moe_packed_reduce.md` · `patches/0039-*.patch`  
Kill packed reduce: `GGML_SYCL_DISABLE_MOE_PACKED_REDUCE=1`

## NEXT

1. **lm_head prune deprioritized** — ROI ceiling ~+4–5 tg / ~+3–4 pp score max
   (`notes/SHIP_20260730_lm_head_roi_ceiling.md`). Not highest leverage.
2. Re-trace under tip for non-lm_head hotspots (attn / residual / remaining MoE meta).
3. Prefill counts D2H experiments **closed** (copy-q + shared-USM both regress).
4. Packing / dual sgs closed under tip.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
