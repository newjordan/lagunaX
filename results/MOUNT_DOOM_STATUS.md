# Mount Doom status — 2026-07-29 evening (dual win)

## LIVE NOW — tip + research track

**Scored tip:** MoE dual + hybrid m1 + dense dual shexp + **moe-down weighted** (all default ON)  
**Research:** hybrid mode2 gather-norm; multi-token mul_mat_id prefill

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip + moe-down-weighted** | **1131.7** | **119.2** | **+7.99%** |
| prior dual+hybrid+dense | 1143.8 | 113.3 | +4.26% |
| baseline pin | 1139 | 107.35 | 1.0 |

- **Golden:** OK
- **Floors:** OK
- **Hit log:** `[lx-control-moe-dual] fuse hit (gate+up+swiglu)`
- **Formal:** `results/20260729T232021Z/score.json` · `LATEST_SCORE.json`
- **A/B:** `results/ctrl-q4k-dual-20260729T231807Z/`
- **Code:** `treebeard-base-control-latest` (`mmvq.cpp`, `ggml-sycl.cpp`, `topk-moe` fuse)
- Disable A/B: `GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1`

### Why this worked
Package dual was **type-rejecting Q4_K** (Q5/Q6 only). Enabling Q4_K on package helped decode (~105→107) but package still loses ~28% prefill. Porting dual onto **control** (champion base) stacks the fuse without the package solo tax.

## POST-REBOOT A/B (pre-dual, serial ship flags)

| tag | pp512 | tg128 | note |
|-----|------:|------:|------|
| ctrl | 1136.18 | 107.58 | pre-dual champion |
| pkg | 818.25 | 104.95 | package solo tax |
| serial_pkg | 816.62 | 104.85 | dual-MoE package tree ≅ pkg |
| wf | FAIL | — | `failed to load model` |

## DONE (pre-reboot)

### Kernel trace (smoking gun)
- Decode oneDNN: **100% matmul time**
- Dominant: MoE expert GEMMs `512×2048 @ N≈9–29` (gate/up) + down `2048×512`
- Hot single shape: `1x512x2048:1x2048x9` — 9004 calls, 5.6% of GPU time
- Artifacts: `results/kernel-trace-20260729T210618Z/HOT_DECODE.md`, `HOT_PREFILL.md`, `TRACE_BOARD.md`

### Power
- Decode: **86 W / 37% of 230 W** → memory-bandwidth bound MoE

### Flag plateau
- Wave1 serial env/flag ladder: ±0.4% noise only

### Infrastructure
- `lx/` harness + baseline pin (pp1139 / tg107.35)
- Quest daemon scripts ready
- Worktrees: `lx-serial-kernel-pkg` (package dual-MoE), `lx-serial-kernel` (control hybrid abort)

## NEXT KERNEL LEVERS (in order)

1. **Stay on control binary as champion** for scored claims (dual ON).
2. **Bitexact hybrid gather-norm** (router already patterns + stock-oracle golden) — or park router.
3. **MoE down fuse / weighted reduce on control only** (not package tree).
4. **Dense dual SwiGLU for shared expert** on control.
5. Device multi-token `mul_mat_id` if decode tip stuck (prefill lever).
6. Tiny-N MMVQ launch geometry for N≈9–29 expert groups (trace smoking gun)

## Target

Beat **tg128 107.9 / pp512 1147** on control+Q4 — first real win must be kernel, not knobs.

## Commands

```bash
# status
tail -f /home/frosty40/turbo/lx/results/quest-mount-doom/quest.log
cat /home/frosty40/turbo/lx/results/post-reboot-ab-20260729T224751Z/SUMMARY.json

# score a candidate
export LX_BIN=/path/to/candidate/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
