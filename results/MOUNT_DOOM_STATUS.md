# Mount Doom status — 2026-07-30 (softplus×mul tip)

## LIVE NOW

**Scored tip:** dual + hybrid7 + dense dual + moe-down/integrated + mmid + RMS+MUL + ADD+ADD + **softplus×mul attn gate**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip softplus×mul** | **1158.8** | **127.2** | **+14.07%** |
| prior ADD+ADD | 1145.0 | 125.2 | +12.35% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T074337Z/` · `notes/SHIP_20260730_softplus_mul_attn.md` · `patches/0020-*.patch`  
Kill: `GGML_SYCL_DISABLE_SOFTPLUS_MUL_FUSE=1`

## NEXT

1. Multi-token dual/MMVQ (host-sort parity).  
2. o_proj + residual ADD epilogue.  
3. lm_head.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
